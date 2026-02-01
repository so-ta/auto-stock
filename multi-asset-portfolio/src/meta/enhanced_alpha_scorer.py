"""
Enhanced Alpha Scorer - リスク調整済みアルファスコア計算

シャープレシオを最大化するため、期待リターン（アルファスコア）を
リスク調整した形式で計算するモジュール。

主要コンポーネント:
- EnhancedAlphaScorer: リスク調整済みアルファスコア計算
- EnhancedAlphaResult: 計算結果のデータクラス
- OptimizableParams: 動的パラメータのデータクラス

設計目的:
- 期待リターン予測: DynamicReturnEstimatorを活用
- リスク調整アルファ: α_adj = (μ - rf) / σ でシャープレシオ形式
- 動的パラメータ: BayesianOptimizerでWalk-Forward CVにより自動調整

使用方法:
    from src.meta.enhanced_alpha_scorer import EnhancedAlphaScorer

    scorer = EnhancedAlphaScorer()
    result = scorer.compute(returns_df, signal_scores, covariance_matrix)
    print(result.alpha_scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.allocation.return_estimator import (
    DynamicReturnEstimator,
    DynamicReturnEstimatorConfig,
    MarketRegime,
)

if TYPE_CHECKING:
    from src.meta.alpha_param_optimizer import AlphaParamOptimizer

logger = logging.getLogger(__name__)


# =============================================================================
# 固定定数（コード内で定義、config不使用）
# =============================================================================

RISK_FREE_RATE = 0.02  # 日銀政策金利ベース（年率）
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 60  # 最小観測数


# =============================================================================
# データクラス
# =============================================================================


@dataclass
class OptimizableParams:
    """BayesianOptimizerで動的に最適化されるパラメータ

    Attributes:
        momentum_weight: モメンタムの重み
        quality_weight: シグナル品質の重み
        expected_return_weight: 期待リターンの重み
        risk_aversion: リスク回避係数
        lookback_days: ルックバック日数
    """

    momentum_weight: float = 0.33
    quality_weight: float = 0.33
    expected_return_weight: float = 0.34
    risk_aversion: float = 2.5
    lookback_days: int = 60

    def __post_init__(self) -> None:
        """バリデーション"""
        total_weight = self.momentum_weight + self.quality_weight + self.expected_return_weight
        if not np.isclose(total_weight, 1.0, atol=0.01):
            # 自動正規化
            total = self.momentum_weight + self.quality_weight + self.expected_return_weight
            if total > 0:
                self.momentum_weight /= total
                self.quality_weight /= total
                self.expected_return_weight /= total
            else:
                self.momentum_weight = 0.33
                self.quality_weight = 0.33
                self.expected_return_weight = 0.34

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "momentum_weight": self.momentum_weight,
            "quality_weight": self.quality_weight,
            "expected_return_weight": self.expected_return_weight,
            "risk_aversion": self.risk_aversion,
            "lookback_days": self.lookback_days,
        }


@dataclass
class EnhancedAlphaResult:
    """リスク調整済みアルファスコア計算結果

    Attributes:
        alpha_scores: シンボル -> リスク調整済みアルファスコア
        expected_returns: シンボル -> 期待リターン（ローテーション期間調整済み）
        volatilities: シンボル -> ボラティリティ（ローテーション期間調整済み）
        rankings: アルファスコアのランキング（高い順）
        optimized_params: 使用されたパラメータ
        regime: 検出されたレジーム
        rotation_days: ローテーション期間
        timestamp: 計算日時
        metadata: 追加メタデータ
    """

    alpha_scores: dict[str, float] = field(default_factory=dict)
    expected_returns: dict[str, float] = field(default_factory=dict)
    volatilities: dict[str, float] = field(default_factory=dict)
    rankings: list[str] = field(default_factory=list)
    optimized_params: OptimizableParams | None = None
    regime: MarketRegime | None = None
    rotation_days: int = 21
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_n(self, n: int) -> list[str]:
        """上位N銘柄を取得"""
        return self.rankings[:n]

    def get_alpha_series(self) -> pd.Series:
        """アルファスコアをSeriesで取得"""
        return pd.Series(self.alpha_scores)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "alpha_scores": self.alpha_scores,
            "expected_returns": self.expected_returns,
            "volatilities": self.volatilities,
            "rankings": self.rankings,
            "optimized_params": self.optimized_params.to_dict() if self.optimized_params else None,
            "regime": self.regime.value if self.regime else None,
            "rotation_days": self.rotation_days,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# メインクラス
# =============================================================================


class EnhancedAlphaScorer:
    """リスク調整済みアルファスコアを計算（config不使用、全て動的）

    シャープレシオ形式のリスク調整アルファを計算:
        α_adj = (μ_est - rf_period) / σ

    where:
        μ_est = DynamicReturnEstimator による期待リターン（期間調整済み）
        rf_period = RISK_FREE_RATE * (rotation_days / 252)
        σ = sqrt(diag(Σ)) * sqrt(rotation_days / 252)

    Usage:
        scorer = EnhancedAlphaScorer()
        result = scorer.compute(returns_df, signal_scores)
        print(result.alpha_scores)
    """

    def __init__(
        self,
        param_optimizer: "AlphaParamOptimizer | None" = None,
    ) -> None:
        """初期化

        Args:
            param_optimizer: パラメータ最適化器（オプション、遅延初期化可能）
        """
        self._return_estimator: DynamicReturnEstimator | None = None
        self._param_optimizer = param_optimizer
        self._current_params: OptimizableParams | None = None

    @property
    def return_estimator(self) -> DynamicReturnEstimator:
        """DynamicReturnEstimator（遅延初期化）"""
        if self._return_estimator is None:
            self._return_estimator = DynamicReturnEstimator()
        return self._return_estimator

    def compute(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float] | None = None,
        covariance_matrix: np.ndarray | pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        use_optimization: bool = True,
    ) -> EnhancedAlphaResult:
        """リスク調整済みアルファを計算

        α_adj = (μ_est - rf_period) / σ

        Args:
            returns_df: リターンデータ（列=アセット、行=日付）
            signal_scores: シグナルスコア辞書（オプション）
            covariance_matrix: 共分散行列（オプション）
            prices: 価格データ（期待リターン推定用、オプション）
            use_optimization: パラメータ最適化を使用するか

        Returns:
            EnhancedAlphaResult: リスク調整済みアルファ計算結果
        """
        if returns_df.empty:
            logger.warning("Empty returns data for enhanced alpha scoring")
            return EnhancedAlphaResult(metadata={"error": "empty_returns"})

        symbols = list(returns_df.columns)
        n_assets = len(symbols)

        # 0. ローテーション期間を自動推定
        rotation_days = self._infer_rotation_days(returns_df)

        # 1. パラメータを取得（最適化 or デフォルト）
        params = self._get_params(
            returns_df, signal_scores, use_optimization
        )
        self._current_params = params

        # 2. レジーム検出
        regime = self._detect_regime(returns_df, prices)

        # 3. 期待リターン推定（年率）
        expected_returns_annual = self._estimate_expected_returns(
            returns_df, prices, signal_scores, params, regime
        )

        # 4. ローテーション期間に調整
        period_factor = rotation_days / TRADING_DAYS_PER_YEAR
        expected_returns_period = {
            symbol: mu * period_factor
            for symbol, mu in expected_returns_annual.items()
        }
        rf_period = RISK_FREE_RATE * period_factor

        # 5. リスク（標準偏差）を計算
        volatilities = self._compute_volatilities(
            returns_df, covariance_matrix, period_factor, symbols
        )

        # 6. リスク調整アルファ = (μ - rf) / σ
        alpha_scores = {}
        min_sigma = 0.001  # 0除算回避用の最小値

        for symbol in symbols:
            mu = expected_returns_period.get(symbol, 0.0)
            sigma = volatilities.get(symbol, min_sigma)
            if sigma < min_sigma:
                sigma = min_sigma

            alpha_scores[symbol] = (mu - rf_period) / sigma

        # 7. ランキング
        rankings = self._compute_rankings(alpha_scores)

        logger.info(
            f"Enhanced alpha scoring completed: {n_assets} assets, "
            f"rotation_days={rotation_days}, regime={regime.value if regime else 'unknown'}"
        )

        return EnhancedAlphaResult(
            alpha_scores=alpha_scores,
            expected_returns=expected_returns_period,
            volatilities=volatilities,
            rankings=rankings,
            optimized_params=params,
            regime=regime,
            rotation_days=rotation_days,
            metadata={
                "n_assets": n_assets,
                "n_observations": len(returns_df),
                "risk_free_rate": RISK_FREE_RATE,
                "rf_period": rf_period,
                "period_factor": period_factor,
            },
        )

    def _infer_rotation_days(self, returns_df: pd.DataFrame) -> int:
        """リターンデータから実際のローテーション期間を自動推定

        Args:
            returns_df: リターンデータ

        Returns:
            推定されたローテーション期間（日数）
        """
        if len(returns_df) < 2:
            return 21  # デフォルト（月次）

        # インデックスの日付差分から推定
        if isinstance(returns_df.index, pd.DatetimeIndex):
            date_diffs = returns_df.index.to_series().diff().dropna()
            if len(date_diffs) == 0:
                return 21

            median_diff = date_diffs.median()
            if hasattr(median_diff, 'days'):
                median_diff_days = median_diff.days
            else:
                # timedeltaでない場合
                median_diff_days = 1

            # 一般的なローテーション期間にマッピング
            if median_diff_days <= 7:
                return 5  # 週次
            elif median_diff_days <= 25:
                return 21  # 月次
            elif median_diff_days <= 70:
                return 63  # 四半期
            else:
                return 252  # 年次

        return 21  # デフォルト

    def _get_params(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float] | None,
        use_optimization: bool,
    ) -> OptimizableParams:
        """最適パラメータを取得

        Args:
            returns_df: リターンデータ
            signal_scores: シグナルスコア
            use_optimization: 最適化を使用するか

        Returns:
            パラメータ
        """
        # 最適化が無効、または最適化器がない場合はデフォルト
        if not use_optimization or self._param_optimizer is None:
            return OptimizableParams()

        # データが不足している場合はデフォルト
        min_required = MIN_OBSERVATIONS * 2
        if len(returns_df) < min_required:
            logger.warning(
                f"Insufficient data for optimization: {len(returns_df)} < {min_required}, "
                f"using default params"
            )
            return OptimizableParams()

        try:
            return self._param_optimizer.get_params(returns_df, signal_scores)
        except Exception as e:
            logger.warning(f"Parameter optimization failed: {e}, using default params")
            return OptimizableParams()

    def _detect_regime(
        self,
        returns_df: pd.DataFrame,
        prices: pd.DataFrame | None,
    ) -> MarketRegime:
        """市場レジームを検出

        Args:
            returns_df: リターンデータ
            prices: 価格データ

        Returns:
            検出されたレジーム
        """
        if prices is not None and len(prices) >= 60:
            return self.return_estimator.detect_regime(prices)

        # リターンデータからの簡易検出
        if len(returns_df) < 20:
            return MarketRegime.NEUTRAL

        recent_returns = returns_df.tail(60) if len(returns_df) >= 60 else returns_df
        mean_return = recent_returns.mean().mean()
        volatility = recent_returns.std().mean()

        # 簡易レジーム判定
        if volatility > 0.03:  # 日次3%以上
            return MarketRegime.HIGH_VOL
        elif volatility < 0.01:  # 日次1%以下
            return MarketRegime.LOW_VOL
        elif mean_return > 0.001:  # 日次0.1%以上
            return MarketRegime.BULL_TREND
        elif mean_return < -0.001:  # 日次-0.1%以下
            return MarketRegime.BEAR_TREND
        else:
            return MarketRegime.NEUTRAL

    def _estimate_expected_returns(
        self,
        returns_df: pd.DataFrame,
        prices: pd.DataFrame | None,
        signal_scores: dict[str, float] | None,
        params: OptimizableParams,
        regime: MarketRegime,
    ) -> dict[str, float]:
        """期待リターンを推定（年率）

        DynamicReturnEstimatorを使用して4手法のアンサンブルで推定

        Args:
            returns_df: リターンデータ
            prices: 価格データ
            signal_scores: シグナルスコア
            params: パラメータ
            regime: 市場レジーム

        Returns:
            シンボル -> 期待リターン（年率）
        """
        symbols = list(returns_df.columns)

        # 価格データがある場合はDynamicReturnEstimatorを使用
        if prices is not None and len(prices) >= MIN_OBSERVATIONS:
            try:
                # DynamicReturnEstimatorの設定を更新
                config = DynamicReturnEstimatorConfig(
                    momentum_lookback=params.lookback_days,
                    risk_aversion=params.risk_aversion,
                )
                estimator = DynamicReturnEstimator(config=config)

                # 期待リターン推定
                result = estimator.estimate(prices, regime=regime)
                return result.combined.to_dict()

            except Exception as e:
                logger.warning(f"DynamicReturnEstimator failed: {e}, using fallback")

        # フォールバック: リターンデータからの簡易推定
        return self._simple_expected_returns(returns_df, signal_scores, params)

    def _simple_expected_returns(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float] | None,
        params: OptimizableParams,
    ) -> dict[str, float]:
        """簡易的な期待リターン推定

        Args:
            returns_df: リターンデータ
            signal_scores: シグナルスコア
            params: パラメータ

        Returns:
            シンボル -> 期待リターン（年率）
        """
        symbols = list(returns_df.columns)
        lookback = min(params.lookback_days, len(returns_df))

        if lookback < 5:
            return {s: 0.0 for s in symbols}

        recent_returns = returns_df.tail(lookback)

        # モメンタム（累積リターン）
        cumulative_returns = (1 + recent_returns).prod() - 1
        momentum_ranks = cumulative_returns.rank(pct=True)
        momentum_score = (momentum_ranks - 0.5) * 2  # -1 to +1

        # 品質スコア（シグナルスコアがある場合）
        if signal_scores:
            quality_values = np.array([signal_scores.get(s, 0.0) for s in symbols])
            if quality_values.std() > 1e-10:
                quality_score = (quality_values - quality_values.mean()) / quality_values.std()
                quality_score = np.clip(quality_score / 3, -1, 1)
            else:
                quality_score = np.zeros(len(symbols))
        else:
            # シグナルがない場合は低ボラティリティを品質とする
            volatilities = recent_returns.std()
            if volatilities.std() > 1e-10:
                inv_vol = 1 / (volatilities + 1e-10)
                quality_score = (inv_vol - inv_vol.mean()) / inv_vol.std()
                quality_score = np.clip(quality_score / 3, -1, 1).values
            else:
                quality_score = np.zeros(len(symbols))

        # 期待リターン（平均リターンの年率化）
        mean_returns = recent_returns.mean() * TRADING_DAYS_PER_YEAR
        if mean_returns.std() > 1e-10:
            exp_ret_score = (mean_returns - mean_returns.mean()) / mean_returns.std()
            exp_ret_score = np.clip(exp_ret_score / 3, -1, 1)
        else:
            exp_ret_score = pd.Series(0.0, index=symbols)

        # 加重平均
        base_premium = 0.10  # 年率10%のベースプレミアム
        expected_returns = {}

        for i, symbol in enumerate(symbols):
            mom = momentum_score[symbol] if isinstance(momentum_score, pd.Series) else momentum_score.iloc[i]
            qual = quality_score[i] if isinstance(quality_score, np.ndarray) else quality_score.iloc[i]
            exp_ret = exp_ret_score[symbol] if isinstance(exp_ret_score, pd.Series) else exp_ret_score.iloc[i]

            combined_score = (
                params.momentum_weight * mom +
                params.quality_weight * qual +
                params.expected_return_weight * exp_ret
            )

            # スコアを期待リターンに変換
            expected_returns[symbol] = combined_score * base_premium

        return expected_returns

    def _compute_volatilities(
        self,
        returns_df: pd.DataFrame,
        covariance_matrix: np.ndarray | pd.DataFrame | None,
        period_factor: float,
        symbols: list[str],
    ) -> dict[str, float]:
        """ボラティリティを計算（ローテーション期間調整済み）

        Args:
            returns_df: リターンデータ
            covariance_matrix: 共分散行列（オプション）
            period_factor: 期間調整係数
            symbols: シンボルリスト

        Returns:
            シンボル -> ボラティリティ
        """
        volatilities = {}

        # 共分散行列が提供された場合
        if covariance_matrix is not None:
            if isinstance(covariance_matrix, pd.DataFrame):
                diag = np.diag(covariance_matrix.values)
                cov_symbols = list(covariance_matrix.columns)
            else:
                diag = np.diag(covariance_matrix)
                cov_symbols = symbols

            for i, symbol in enumerate(cov_symbols):
                if symbol in symbols:
                    # 年率→期間調整
                    annual_vol = np.sqrt(diag[i])
                    volatilities[symbol] = annual_vol * np.sqrt(period_factor)

        # 共分散行列がない場合はリターンから計算
        if not volatilities:
            # 日次ボラティリティを計算
            daily_vol = returns_df.std()

            for symbol in symbols:
                if symbol in daily_vol.index:
                    # 日次→期間調整
                    vol = daily_vol[symbol] * np.sqrt(period_factor * TRADING_DAYS_PER_YEAR)
                    volatilities[symbol] = vol

        return volatilities

    def _compute_rankings(self, alpha_scores: dict[str, float]) -> list[str]:
        """アルファスコアのランキングを計算

        Args:
            alpha_scores: シンボル -> アルファスコア

        Returns:
            ランキング（高い順）
        """
        sorted_items = sorted(
            alpha_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [symbol for symbol, _ in sorted_items]


# =============================================================================
# 便利関数
# =============================================================================


def create_enhanced_alpha_scorer(
    param_optimizer: "AlphaParamOptimizer | None" = None,
) -> EnhancedAlphaScorer:
    """EnhancedAlphaScorerのファクトリ関数

    Args:
        param_optimizer: パラメータ最適化器（オプション）

    Returns:
        初期化されたEnhancedAlphaScorer
    """
    return EnhancedAlphaScorer(param_optimizer=param_optimizer)


def quick_enhanced_alpha_ranking(
    returns_df: pd.DataFrame,
    top_n: int = 10,
    signal_scores: dict[str, float] | None = None,
) -> list[str]:
    """簡易リスク調整アルファランキング（便利関数）

    Args:
        returns_df: リターンデータ
        top_n: 上位N銘柄を返す
        signal_scores: シグナルスコア（オプション）

    Returns:
        上位N銘柄のリスト
    """
    scorer = EnhancedAlphaScorer()
    result = scorer.compute(returns_df, signal_scores=signal_scores, use_optimization=False)
    return result.get_top_n(top_n)
