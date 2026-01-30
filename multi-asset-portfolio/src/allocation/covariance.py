"""
Covariance Estimation Module - 共分散推定

リターンの共分散行列を推定する複数の手法を提供する。

実装手法:
1. サンプル共分散（Sample Covariance）
2. Ledoit-Wolf縮小推定（Shrinkage Estimation）
3. 指数加重移動共分散（EWMA Covariance）

設計根拠:
- 要求.md §8.1: 共分散推定 Σ（リターンの相関）
- 要求.md §8.3: Ledoit-Wolf等の縮小推定（簡易でも）
- 推定誤差に強い手法を推奨

使用方法:
    estimator = CovarianceEstimator(method="ledoit_wolf")
    result = estimator.estimate(returns_df)
    cov_matrix = result.covariance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.backtest.covariance_cache import CovarianceState

logger = logging.getLogger(__name__)


class CovarianceMethod(str, Enum):
    """共分散推定手法"""

    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    EWMA = "ewma"
    REGIME_CONDITIONAL = "regime_conditional"


class RegimeType(str, Enum):
    """マーケットレジームタイプ"""

    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"


@dataclass(frozen=True)
class CovarianceConfig:
    """共分散推定設定

    Attributes:
        method: 推定手法
        ewma_halflife: EWMAの半減期（日数）
        min_periods: 最小必要サンプル数
        shrinkage_target: 縮小推定のターゲット（"identity" or "constant_correlation"）
        annualization_factor: 年率化係数（252=日次データ）
        crisis_corr_adjustment: クライシス時の相関上方調整率（例: 0.30 = +30%）
        low_vol_corr_adjustment: 低ボラ時の相関下方調整率（例: -0.15 = -15%）
        vol_lookback: ボラティリティレジーム判定の参照期間（日数）
        crisis_vol_threshold: クライシスレジームのボラティリティ閾値（パーセンタイル）
        low_vol_threshold: 低ボラレジームのボラティリティ閾値（パーセンタイル）
    """

    method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF
    ewma_halflife: int = 60
    min_periods: int = 60
    shrinkage_target: str = "constant_correlation"
    annualization_factor: int = 252
    # Dynamic correlation adjustment parameters
    crisis_corr_adjustment: float = 0.30  # +30% correlation during crisis
    low_vol_corr_adjustment: float = -0.15  # -15% correlation during low vol
    vol_lookback: int = 252  # 1 year for regime detection
    crisis_vol_threshold: float = 0.80  # Top 20% volatility = crisis
    low_vol_threshold: float = 0.25  # Bottom 25% volatility = low vol

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.ewma_halflife <= 0:
            raise ValueError("ewma_halflife must be > 0")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be > 0")
        if self.shrinkage_target not in ("identity", "constant_correlation"):
            raise ValueError(
                f"Invalid shrinkage_target: {self.shrinkage_target}. "
                "Must be 'identity' or 'constant_correlation'"
            )
        if not -1.0 <= self.crisis_corr_adjustment <= 1.0:
            raise ValueError("crisis_corr_adjustment must be in [-1, 1]")
        if not -1.0 <= self.low_vol_corr_adjustment <= 1.0:
            raise ValueError("low_vol_corr_adjustment must be in [-1, 1]")


@dataclass
class CovarianceResult:
    """共分散推定結果

    Attributes:
        covariance: 共分散行列（DataFrame）
        correlation: 相関行列（DataFrame）
        volatilities: 各アセットのボラティリティ（Series）
        shrinkage_intensity: 縮小強度（Ledoit-Wolfの場合）
        effective_samples: 有効サンプル数
        method_used: 使用した手法
        metadata: 追加メタデータ
    """

    covariance: pd.DataFrame
    correlation: pd.DataFrame
    volatilities: pd.Series
    shrinkage_intensity: float | None = None
    effective_samples: int = 0
    method_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な推定結果かどうか"""
        return (
            not self.covariance.empty
            and not self.covariance.isna().any().any()
            and self.is_positive_definite
        )

    @property
    def is_positive_definite(self) -> bool:
        """正定値かどうか"""
        try:
            eigenvalues = np.linalg.eigvalsh(self.covariance.values)
            return bool(np.all(eigenvalues > 0))
        except np.linalg.LinAlgError:
            return False

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "covariance": self.covariance.to_dict(),
            "correlation": self.correlation.to_dict(),
            "volatilities": self.volatilities.to_dict(),
            "shrinkage_intensity": self.shrinkage_intensity,
            "effective_samples": self.effective_samples,
            "method_used": self.method_used,
            "is_valid": self.is_valid,
            "is_positive_definite": self.is_positive_definite,
            "metadata": self.metadata,
        }


class CovarianceEstimator:
    """共分散推定クラス

    複数の手法で共分散行列を推定する。

    Usage:
        config = CovarianceConfig(method=CovarianceMethod.LEDOIT_WOLF)
        estimator = CovarianceEstimator(config)

        # returns_df: (T, N) の日次リターンDataFrame
        result = estimator.estimate(returns_df)

        print(result.covariance)
        print(result.shrinkage_intensity)

    Dynamic Parameters:
        use_dynamic=True でデータに基づく動的パラメータを使用可能。
        動的パラメータが計算できない場合はデフォルト値にフォールバック。
    """

    def __init__(
        self,
        config: CovarianceConfig | None = None,
        use_dynamic: bool = False,
        returns: pd.DataFrame | None = None,
        lookback_days: int = 252,
    ) -> None:
        """初期化

        Args:
            config: 共分散推定設定。Noneの場合はデフォルト値を使用
            use_dynamic: 動的パラメータを使用するかどうか
            returns: リターンデータ（動的パラメータ計算用）
            lookback_days: ルックバック日数（動的パラメータ計算用）
        """
        self._use_dynamic = use_dynamic
        self._returns = returns
        self._lookback_days = lookback_days

        if use_dynamic and config is None:
            self.config = self._compute_dynamic_config()
        else:
            self.config = config or CovarianceConfig()

    def _compute_dynamic_config(self) -> CovarianceConfig:
        """動的パラメータからConfigを計算

        Returns:
            動的に計算されたCovarianceConfig
        """
        try:
            from .dynamic_covariance_params import (
                calculate_covariance_params,
                detect_market_regime,
            )

            if self._returns is None or self._returns.empty:
                logger.warning(
                    "Dynamic covariance params requested but no returns provided. "
                    "Using default config."
                )
                return CovarianceConfig()

            params = calculate_covariance_params(
                returns=self._returns,
                lookback_days=self._lookback_days,
            )
            return CovarianceConfig(
                ewma_halflife=params.ewma_halflife,
                crisis_corr_adjustment=params.crisis_corr_adjustment,
                low_vol_corr_adjustment=params.low_vol_corr_adjustment,
                crisis_vol_threshold=params.crisis_vol_threshold,
                low_vol_threshold=params.low_vol_threshold,
            )

        except Exception as e:
            logger.warning(
                "Failed to compute dynamic covariance config: %s. Using defaults.", e
            )
            return CovarianceConfig()

    def estimate(
        self,
        returns: pd.DataFrame,
        incremental_state: "CovarianceState | None" = None,
    ) -> CovarianceResult:
        """共分散行列を推定

        Args:
            returns: 日次リターンのDataFrame (T x N)
                     - index: DatetimeIndex
                     - columns: アセット名
            incremental_state: インクリメンタル更新用の状態（task_013_3）
                     - Noneの場合はフル計算
                     - 状態がある場合は増分更新（3-5倍高速化）

        Returns:
            CovarianceResult: 推定結果
        """
        if returns.empty:
            logger.warning("Empty returns DataFrame provided")
            return self._create_empty_result(returns.columns)

        # インクリメンタル更新モード
        if incremental_state is not None:
            return self._update_incremental(returns, incremental_state)

        # フル計算モード
        return self._compute_full(returns)

    def _compute_full(self, returns: pd.DataFrame) -> CovarianceResult:
        """フル計算で共分散行列を推定（従来の処理）

        Args:
            returns: 日次リターン

        Returns:
            CovarianceResult: 推定結果
        """
        # 欠損値処理
        returns_clean = returns.dropna()
        effective_samples = len(returns_clean)

        if effective_samples < self.config.min_periods:
            logger.warning(
                "Insufficient samples: %d < %d. Using fallback.",
                effective_samples,
                self.config.min_periods,
            )
            return self._create_empty_result(returns.columns)

        # 手法に応じて推定
        method = self.config.method
        if method == CovarianceMethod.SAMPLE:
            result = self._estimate_sample(returns_clean)
        elif method == CovarianceMethod.LEDOIT_WOLF:
            result = self._estimate_ledoit_wolf(returns_clean)
        elif method == CovarianceMethod.EWMA:
            result = self._estimate_ewma(returns_clean)
        elif method == CovarianceMethod.REGIME_CONDITIONAL:
            result = self._estimate_regime_conditional(returns_clean)
        else:
            raise ValueError(f"Unknown method: {method}")

        result.effective_samples = effective_samples
        result.method_used = method.value

        # 年率化
        result.covariance = result.covariance * self.config.annualization_factor
        result.volatilities = result.volatilities * np.sqrt(
            self.config.annualization_factor
        )

        logger.info(
            "Covariance estimated using %s: %d assets, %d samples",
            method.value,
            len(returns.columns),
            effective_samples,
        )

        return result

    def _update_incremental(
        self,
        returns: pd.DataFrame,
        state: "CovarianceState",
    ) -> CovarianceResult:
        """インクリメンタル更新で共分散行列を推定（task_013_3）

        既存の状態から増分更新することで、計算コストを3-5倍削減。

        Args:
            returns: 新しいリターンデータ（差分のみでOK）
            state: 前回の状態（CovarianceState）

        Returns:
            CovarianceResult: 推定結果
        """
        from src.backtest.covariance_cache import IncrementalCovarianceEstimator

        # 状態から推定器を復元
        n_assets = state.n_assets
        halflife = state.halflife
        asset_names = state.asset_names or list(returns.columns)

        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets,
            halflife=halflife,
            asset_names=asset_names,
        )
        estimator.set_state(state)

        # 新しいリターンで更新
        returns_clean = returns.dropna()
        if not returns_clean.empty:
            # 列の順序を合わせる
            if asset_names:
                common_assets = [a for a in asset_names if a in returns_clean.columns]
                if common_assets:
                    returns_aligned = returns_clean[common_assets].values
                    for row in returns_aligned:
                        estimator.update(row)

        # 共分散行列を取得
        cov_matrix = estimator.get_covariance()
        corr_matrix = estimator.get_correlation()
        volatilities = estimator.get_volatility()

        # 年率化
        annualization = self.config.annualization_factor
        cov_annualized = cov_matrix * annualization
        vol_annualized = volatilities * np.sqrt(annualization)

        # DataFrameに変換
        assets = pd.Index(asset_names[:n_assets])
        result = CovarianceResult(
            covariance=pd.DataFrame(cov_annualized, index=assets, columns=assets),
            correlation=pd.DataFrame(corr_matrix, index=assets, columns=assets),
            volatilities=pd.Series(vol_annualized, index=assets),
            effective_samples=estimator.n_updates,
            method_used="incremental_ewma",
            metadata={
                "incremental": True,
                "halflife": halflife,
                "n_updates": estimator.n_updates,
                "new_samples": len(returns_clean),
            },
        )

        logger.info(
            "Covariance updated incrementally: %d assets, %d total updates (+%d new)",
            n_assets,
            estimator.n_updates,
            len(returns_clean),
        )

        return result

    def create_incremental_state(
        self,
        returns: pd.DataFrame,
        halflife: int | None = None,
    ) -> "CovarianceState":
        """インクリメンタル更新用の初期状態を作成

        Args:
            returns: 初期リターンデータ
            halflife: 半減期（Noneの場合はconfig値を使用）

        Returns:
            CovarianceState: 初期状態
        """
        from src.backtest.covariance_cache import (
            CovarianceState,
            IncrementalCovarianceEstimator,
        )

        hl = halflife or self.config.ewma_halflife
        returns_clean = returns.dropna()
        asset_names = list(returns_clean.columns)
        n_assets = len(asset_names)

        estimator = IncrementalCovarianceEstimator(
            n_assets=n_assets,
            halflife=hl,
            asset_names=asset_names,
        )

        # 初期データで更新
        estimator.update_batch(returns_clean.values)

        return estimator.get_state()

    def _estimate_sample(self, returns: pd.DataFrame) -> CovarianceResult:
        """サンプル共分散を計算

        Args:
            returns: 日次リターン

        Returns:
            CovarianceResult
        """
        cov_matrix = returns.cov()
        corr_matrix = returns.corr()
        volatilities = returns.std()

        return CovarianceResult(
            covariance=cov_matrix,
            correlation=corr_matrix,
            volatilities=volatilities,
            metadata={"note": "Sample covariance (unbiased)"},
        )

    def _estimate_ledoit_wolf(self, returns: pd.DataFrame) -> CovarianceResult:
        """Ledoit-Wolf縮小推定

        縮小推定により、推定誤差を軽減する。
        ターゲット: 定数相関行列 または 単位行列

        Args:
            returns: 日次リターン

        Returns:
            CovarianceResult
        """
        X = returns.values
        n, p = X.shape

        # サンプル共分散
        sample_cov = np.cov(X, rowvar=False, bias=False)
        sample_mean = np.mean(X, axis=0)

        # ボラティリティ
        volatilities = np.sqrt(np.diag(sample_cov))

        # ターゲット行列の計算
        if self.config.shrinkage_target == "identity":
            # 単位行列（スケーリング済み）
            mu = np.trace(sample_cov) / p
            target = mu * np.eye(p)
        else:
            # 定数相関行列
            # 平均相関を使用
            corr = np.corrcoef(X, rowvar=False)
            np.fill_diagonal(corr, 0)
            avg_corr = np.sum(corr) / (p * (p - 1))

            target = np.outer(volatilities, volatilities) * avg_corr
            np.fill_diagonal(target, np.diag(sample_cov))

        # 縮小強度の推定（Ledoit-Wolf formula）
        shrinkage = self._compute_shrinkage_intensity(X, sample_cov, target)

        # 縮小推定
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

        # 正定値性の保証
        shrunk_cov = self._ensure_positive_definite(shrunk_cov)

        # 相関行列の計算
        std_diag = np.sqrt(np.diag(shrunk_cov))
        corr_matrix = shrunk_cov / np.outer(std_diag, std_diag)

        assets = returns.columns
        return CovarianceResult(
            covariance=pd.DataFrame(shrunk_cov, index=assets, columns=assets),
            correlation=pd.DataFrame(corr_matrix, index=assets, columns=assets),
            volatilities=pd.Series(volatilities, index=assets),
            shrinkage_intensity=float(shrinkage),
            metadata={
                "shrinkage_target": self.config.shrinkage_target,
                "n_samples": n,
                "n_assets": p,
            },
        )

    def _compute_shrinkage_intensity(
        self,
        X: NDArray[np.float64],
        sample_cov: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        """Ledoit-Wolf縮小強度を計算

        Args:
            X: 中心化されたデータ (n x p)
            sample_cov: サンプル共分散行列
            target: ターゲット行列

        Returns:
            縮小強度 (0-1)
        """
        n, p = X.shape
        X_centered = X - np.mean(X, axis=0)

        # delta: ターゲットとサンプル共分散の差のノルム
        delta = target - sample_cov
        delta_norm_sq = np.sum(delta**2)

        # pi: 推定誤差の分散
        # sum of asymptotic variances of entries of sample covariance
        X2 = X_centered**2
        pi_mat = (X2.T @ X2) / n - sample_cov**2
        pi = np.sum(pi_mat)

        # kappa: 縮小による二乗誤差の減少
        kappa = (pi - delta_norm_sq) / n

        # 縮小強度
        if delta_norm_sq == 0:
            shrinkage = 1.0
        else:
            shrinkage = max(0.0, min(1.0, kappa / delta_norm_sq))

        return shrinkage

    def _estimate_ewma(self, returns: pd.DataFrame) -> CovarianceResult:
        """指数加重移動共分散を計算

        直近のデータにより大きな重みを付ける。

        Args:
            returns: 日次リターン

        Returns:
            CovarianceResult
        """
        halflife = self.config.ewma_halflife
        alpha = 1 - np.exp(-np.log(2) / halflife)

        # EWMA共分散の計算
        ewma_cov = returns.ewm(halflife=halflife, min_periods=self.config.min_periods).cov()

        # 最新の共分散行列を取得
        last_date = returns.index[-1]
        cov_matrix = ewma_cov.loc[last_date]

        # 相関行列
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        corr_values = cov_matrix.values / np.outer(volatilities, volatilities)
        corr_matrix = pd.DataFrame(
            corr_values, index=cov_matrix.index, columns=cov_matrix.columns
        )

        return CovarianceResult(
            covariance=cov_matrix,
            correlation=corr_matrix,
            volatilities=pd.Series(volatilities, index=returns.columns),
            metadata={
                "halflife": halflife,
                "alpha": alpha,
                "decay_factor": 1 - alpha,
            },
        )

    def _estimate_regime_conditional(self, returns: pd.DataFrame) -> CovarianceResult:
        """レジーム条件付き共分散推定

        現在のボラティリティレジームに応じて相関行列を動的に調整する。
        - クライシス時: 相関を上方調整（+30%等）- リスクを過小評価しない
        - 低ボラ時: 相関を下方調整（-15%等）- 分散効果を活用

        Args:
            returns: 日次リターン

        Returns:
            CovarianceResult
        """
        # 1. ベースとなるLedoit-Wolf推定を実行
        base_result = self._estimate_ledoit_wolf(returns)

        # 2. 現在のレジームを判定
        regime, regime_percentile = self._detect_regime(returns)

        # 3. レジームに応じた相関調整
        adjusted_corr, adjustment_factor = self._adjust_correlation_for_regime(
            base_result.correlation, regime
        )

        # 4. 調整後の共分散行列を再構成
        vols = base_result.volatilities.values
        adjusted_cov = adjusted_corr.values * np.outer(vols, vols)

        # 正定値性を保証
        adjusted_cov = self._ensure_positive_definite(adjusted_cov)

        assets = returns.columns
        adjusted_cov_df = pd.DataFrame(adjusted_cov, index=assets, columns=assets)
        adjusted_corr_df = pd.DataFrame(adjusted_corr, index=assets, columns=assets)

        return CovarianceResult(
            covariance=adjusted_cov_df,
            correlation=adjusted_corr_df,
            volatilities=base_result.volatilities,
            shrinkage_intensity=base_result.shrinkage_intensity,
            metadata={
                "base_method": "ledoit_wolf",
                "regime": regime.value,
                "regime_percentile": regime_percentile,
                "correlation_adjustment": adjustment_factor,
                "crisis_threshold": self.config.crisis_vol_threshold,
                "low_vol_threshold": self.config.low_vol_threshold,
            },
        )

    def _detect_regime(self, returns: pd.DataFrame) -> tuple[RegimeType, float]:
        """現在のボラティリティレジームを検出

        Args:
            returns: 日次リターン

        Returns:
            (RegimeType, volatility_percentile)
        """
        # ポートフォリオ全体のボラティリティ（等重み仮定）
        portfolio_returns = returns.mean(axis=1)
        lookback = min(self.config.vol_lookback, len(returns))

        # 現在のボラティリティ（直近20日）
        current_vol = portfolio_returns.iloc[-20:].std() * np.sqrt(252)

        # 過去のボラティリティ分布
        historical_vols = []
        for i in range(20, lookback):
            vol = portfolio_returns.iloc[i - 20 : i].std() * np.sqrt(252)
            historical_vols.append(vol)

        if len(historical_vols) < 10:
            return RegimeType.NORMAL, 0.5

        historical_vols = np.array(historical_vols)
        percentile = (historical_vols < current_vol).mean()

        # レジーム判定
        if percentile >= self.config.crisis_vol_threshold:
            regime = RegimeType.CRISIS
        elif percentile >= 0.6:  # 60-80 percentile
            regime = RegimeType.HIGH_VOL
        elif percentile <= self.config.low_vol_threshold:
            regime = RegimeType.LOW_VOL
        else:
            regime = RegimeType.NORMAL

        return regime, float(percentile)

    def _adjust_correlation_for_regime(
        self, correlation: pd.DataFrame, regime: RegimeType
    ) -> tuple[pd.DataFrame, float]:
        """レジームに応じた相関調整

        Args:
            correlation: 元の相関行列
            regime: 現在のレジーム

        Returns:
            (adjusted_correlation, adjustment_factor)
        """
        corr_values = correlation.values.copy()

        if regime == RegimeType.CRISIS:
            # クライシス時: 相関を上方調整
            adjustment = self.config.crisis_corr_adjustment
        elif regime == RegimeType.HIGH_VOL:
            # 高ボラ時: 軽度の上方調整
            adjustment = self.config.crisis_corr_adjustment * 0.5
        elif regime == RegimeType.LOW_VOL:
            # 低ボラ時: 相関を下方調整
            adjustment = self.config.low_vol_corr_adjustment
        else:
            # 通常時: 調整なし
            adjustment = 0.0

        if adjustment != 0.0:
            # 対角成分以外を調整
            n = corr_values.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # 調整後も[-1, 1]の範囲に収める
                        adjusted = corr_values[i, j] * (1 + adjustment)
                        corr_values[i, j] = np.clip(adjusted, -0.99, 0.99)

        adjusted_df = pd.DataFrame(
            corr_values, index=correlation.index, columns=correlation.columns
        )

        return adjusted_df, adjustment

    def estimate_with_regime(
        self, returns: pd.DataFrame, force_regime: RegimeType | None = None
    ) -> CovarianceResult:
        """レジーム指定付きで共分散を推定

        Args:
            returns: 日次リターン
            force_regime: 強制的に適用するレジーム（None=自動検出）

        Returns:
            CovarianceResult
        """
        if returns.empty:
            return self._create_empty_result(returns.columns)

        returns_clean = returns.dropna()
        effective_samples = len(returns_clean)

        if effective_samples < self.config.min_periods:
            return self._create_empty_result(returns.columns)

        # ベース推定
        base_result = self._estimate_ledoit_wolf(returns_clean)

        # レジーム検出または強制指定
        if force_regime is not None:
            regime = force_regime
            regime_percentile = -1.0  # 強制指定を示す
        else:
            regime, regime_percentile = self._detect_regime(returns_clean)

        # 相関調整
        adjusted_corr, adjustment = self._adjust_correlation_for_regime(
            base_result.correlation, regime
        )

        # 共分散再構成
        vols = base_result.volatilities.values
        adjusted_cov = adjusted_corr.values * np.outer(vols, vols)
        adjusted_cov = self._ensure_positive_definite(adjusted_cov)

        # 年率化
        adjusted_cov = adjusted_cov * self.config.annualization_factor
        vols_annualized = vols * np.sqrt(self.config.annualization_factor)

        assets = returns.columns
        result = CovarianceResult(
            covariance=pd.DataFrame(adjusted_cov, index=assets, columns=assets),
            correlation=adjusted_corr,
            volatilities=pd.Series(vols_annualized, index=assets),
            shrinkage_intensity=base_result.shrinkage_intensity,
            effective_samples=effective_samples,
            method_used="regime_conditional",
            metadata={
                "regime": regime.value,
                "regime_percentile": regime_percentile,
                "correlation_adjustment": adjustment,
                "forced_regime": force_regime is not None,
            },
        )

        logger.info(
            "Regime-conditional covariance: regime=%s, adjustment=%.2f%%",
            regime.value,
            adjustment * 100,
        )

        return result

    def _ensure_positive_definite(
        self, matrix: NDArray[np.float64], epsilon: float = 1e-8
    ) -> NDArray[np.float64]:
        """正定値性を保証

        固有値が負の場合、小さな正の値に置き換える。

        Args:
            matrix: 対称行列
            epsilon: 最小固有値

        Returns:
            正定値行列
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # 負の固有値を補正
        eigenvalues = np.maximum(eigenvalues, epsilon)

        # 再構成
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _create_empty_result(self, columns: pd.Index) -> CovarianceResult:
        """空の結果を作成

        Args:
            columns: アセット名

        Returns:
            空のCovarianceResult
        """
        n = len(columns)
        empty_cov = pd.DataFrame(
            np.eye(n), index=columns, columns=columns
        )
        empty_corr = pd.DataFrame(
            np.eye(n), index=columns, columns=columns
        )
        empty_vol = pd.Series(np.ones(n), index=columns)

        return CovarianceResult(
            covariance=empty_cov,
            correlation=empty_corr,
            volatilities=empty_vol,
            metadata={"error": "Insufficient data"},
        )


def create_estimator_from_settings() -> CovarianceEstimator:
    """グローバル設定からEstimatorを生成

    Returns:
        設定済みのCovarianceEstimator
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        # デフォルトでLedoit-Wolf使用
        config = CovarianceConfig(
            method=CovarianceMethod.LEDOIT_WOLF,
        )
        return CovarianceEstimator(config)
    except ImportError:
        logger.warning("Settings not available, using default CovarianceConfig")
        return CovarianceEstimator()
