"""
Dynamic Return Estimator - 期待リターン推定の動的化

このモジュールは、複数の期待リターン推定手法を統合し、
市場レジームに応じて最適な手法を選択・統合する機能を提供する。

主要コンポーネント:
- DynamicReturnEstimator: 統合クラス（レジーム別統合）
- CrossSectionalMomentum: 相対モメンタム推定
- ImpliedReturns: Black-Litterman的な均衡リターン
- MeanReversionForecast: 平均回帰予測
- FactorPremiumTiming: ファクタープレミアム推定

設計根拠:
- 単一の推定手法は市場環境により精度が変動する
- レジームに応じた手法選択で安定性を向上
- 複数手法の統合で推定誤差を分散
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """市場レジーム。"""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE_BOUND = "range_bound"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


@dataclass
class ReturnEstimate:
    """期待リターン推定結果。

    Attributes:
        cross_sectional: クロスセクショナルモメンタムによる推定
        implied: インプライドリターン（均衡リターン）
        mean_reversion: 平均回帰予測による推定
        factor: ファクタープレミアムによる推定
        combined: 統合された期待リターン
        regime: 使用されたレジーム
        weights_used: 各手法の統合重み
        timestamp: 推定日時
        metadata: 追加情報
    """

    cross_sectional: pd.Series | None
    implied: pd.Series | None
    mean_reversion: pd.Series | None
    factor: pd.Series | None
    combined: pd.Series
    regime: MarketRegime
    weights_used: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "cross_sectional": self.cross_sectional.to_dict() if self.cross_sectional is not None else None,
            "implied": self.implied.to_dict() if self.implied is not None else None,
            "mean_reversion": self.mean_reversion.to_dict() if self.mean_reversion is not None else None,
            "factor": self.factor.to_dict() if self.factor is not None else None,
            "combined": self.combined.to_dict(),
            "regime": self.regime.value,
            "weights_used": self.weights_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ReturnEstimatorBase(ABC):
    """期待リターン推定の基底クラス。"""

    @abstractmethod
    def estimate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """期待リターンを推定する。

        Args:
            prices: 価格データ（columns=銘柄、index=日付）
            **kwargs: 追加パラメータ

        Returns:
            各銘柄の期待リターン（年率）
        """
        pass


class CrossSectionalMomentum(ReturnEstimatorBase):
    """クロスセクショナルモメンタム（相対モメンタム）推定。

    銘柄間の相対パフォーマンスをランキングし、
    上位銘柄に高い期待リターンを付与する。

    Attributes:
        lookback: ルックバック期間（日数）
        base_premium: ベースとなるプレミアム（年率）
    """

    def __init__(
        self,
        lookback: int = 60,
        base_premium: float = 0.10,
    ) -> None:
        """初期化。

        Args:
            lookback: ルックバック期間（日数）
            base_premium: ベースプレミアム（年率、例: 0.10 = 10%）
        """
        self.lookback = lookback
        self.base_premium = base_premium

    def estimate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """クロスセクショナルモメンタムによる期待リターンを推定する。

        Args:
            prices: 価格データ

        Returns:
            各銘柄の期待リターン（年率）
        """
        lookback = kwargs.get("lookback", self.lookback)

        if len(prices) < lookback:
            logger.warning(
                f"Insufficient data for lookback={lookback}, using all available data"
            )
            lookback = max(1, len(prices) - 1)

        # ルックバック期間のリターンを計算
        returns = prices.iloc[-1] / prices.iloc[-lookback] - 1

        # パーセンタイルランクを計算（0-1）
        ranks = returns.rank(pct=True)

        # ランクを期待リターンに変換
        # rank=0.5 -> 0, rank=1 -> +base_premium, rank=0 -> -base_premium
        expected_return = (ranks - 0.5) * 2 * self.base_premium

        logger.debug(
            f"Cross-sectional momentum: lookback={lookback}, "
            f"returns range=[{expected_return.min():.4f}, {expected_return.max():.4f}]"
        )

        return expected_return


class ImpliedReturns(ReturnEstimatorBase):
    """インプライドリターン（Black-Litterman的な均衡リターン）推定。

    市場均衡ウェイトと共分散から逆算して均衡期待リターンを計算。

    π = δ × Σ × w_mkt

    Attributes:
        risk_aversion: リスク回避係数（δ）
        annualization_factor: 年率化係数
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        annualization_factor: float = 252,
    ) -> None:
        """初期化。

        Args:
            risk_aversion: リスク回避係数
            annualization_factor: 年率化係数
        """
        self.risk_aversion = risk_aversion
        self.annualization_factor = annualization_factor

    def estimate(
        self,
        prices: pd.DataFrame,
        weights_market: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """インプライドリターンを推定する。

        Args:
            prices: 価格データ
            weights_market: 市場ウェイト（Noneの場合は等ウェイト）
            cov: 共分散行列（Noneの場合は計算）

        Returns:
            各銘柄の期待リターン（年率）
        """
        assets = prices.columns.tolist()

        # 市場ウェイト（デフォルトは等ウェイト）
        if weights_market is None:
            weights_market = pd.Series(1.0 / len(assets), index=assets)

        # 共分散行列
        if cov is None:
            returns = prices.pct_change().dropna()
            cov = returns.cov() * self.annualization_factor

        # インプライドリターン: π = δΣw
        risk_aversion = kwargs.get("risk_aversion", self.risk_aversion)
        pi = risk_aversion * cov @ weights_market

        logger.debug(
            f"Implied returns: risk_aversion={risk_aversion}, "
            f"returns range=[{pi.min():.4f}, {pi.max():.4f}]"
        )

        return pi


class MeanReversionForecast(ReturnEstimatorBase):
    """平均回帰予測による期待リターン推定。

    長期平均からの乖離を期待リターンに変換。
    乖離の一定割合が戻ると仮定。

    Attributes:
        lookback: 長期平均の計算期間（日数）
        reversion_rate: 乖離が戻る割合（0-1）
    """

    def __init__(
        self,
        lookback: int = 252,
        reversion_rate: float = 0.5,
    ) -> None:
        """初期化。

        Args:
            lookback: 長期平均の計算期間（日数）
            reversion_rate: 乖離が戻る割合
        """
        self.lookback = lookback
        self.reversion_rate = reversion_rate

    def estimate(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """平均回帰予測による期待リターンを推定する。

        Args:
            prices: 価格データ

        Returns:
            各銘柄の期待リターン（年率）
        """
        lookback = kwargs.get("lookback", self.lookback)
        reversion_rate = kwargs.get("reversion_rate", self.reversion_rate)

        if len(prices) < lookback:
            lookback = len(prices)

        # 長期平均
        long_term_mean = prices.iloc[-lookback:].mean()

        # 現在価格
        current_price = prices.iloc[-1]

        # 乖離率
        deviation = (long_term_mean - current_price) / current_price

        # 期待リターン（年率換算）
        # 乖離のreversion_rate分がlookback期間で戻ると仮定
        expected_return = deviation * reversion_rate / (lookback / 252)

        # 極端な値をクリップ
        expected_return = expected_return.clip(-0.5, 0.5)

        logger.debug(
            f"Mean reversion forecast: lookback={lookback}, "
            f"returns range=[{expected_return.min():.4f}, {expected_return.max():.4f}]"
        )

        return expected_return


class FactorPremiumTiming(ReturnEstimatorBase):
    """ファクタープレミアム推定。

    各銘柄のファクターエクスポージャーと
    過去のファクターリターンから期待リターンを推定。

    Attributes:
        ewm_halflife: 指数加重移動平均の半減期（日数）
        factor_names: 使用するファクター名のリスト
    """

    DEFAULT_FACTOR_NAMES = ["momentum", "value", "quality", "size", "volatility"]

    def __init__(
        self,
        ewm_halflife: int = 60,
        factor_names: list[str] | None = None,
    ) -> None:
        """初期化。

        Args:
            ewm_halflife: EWMAの半減期
            factor_names: ファクター名リスト
        """
        self.ewm_halflife = ewm_halflife
        self.factor_names = factor_names or self.DEFAULT_FACTOR_NAMES

    def estimate(
        self,
        prices: pd.DataFrame,
        factor_exposures: pd.DataFrame | None = None,
        factor_returns_history: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """ファクタープレミアムによる期待リターンを推定する。

        Args:
            prices: 価格データ
            factor_exposures: ファクターエクスポージャー（assets x factors）
            factor_returns_history: ファクターリターン履歴（dates x factors）

        Returns:
            各銘柄の期待リターン（年率）
        """
        assets = prices.columns.tolist()

        # ファクターエクスポージャーがない場合は簡易計算
        if factor_exposures is None or factor_returns_history is None:
            return self._estimate_from_prices(prices)

        # EWMAで期待ファクターリターンを計算
        halflife = kwargs.get("ewm_halflife", self.ewm_halflife)
        expected_factor_returns = (
            factor_returns_history.ewm(halflife=halflife).mean().iloc[-1]
        )

        # 期待リターン = エクスポージャー × 期待ファクターリターン
        expected_return = factor_exposures @ expected_factor_returns

        # 結果を元の資産順序に合わせる
        expected_return = expected_return.reindex(assets).fillna(0.0)

        logger.debug(
            f"Factor premium: halflife={halflife}, "
            f"returns range=[{expected_return.min():.4f}, {expected_return.max():.4f}]"
        )

        return expected_return

    def _estimate_from_prices(self, prices: pd.DataFrame) -> pd.Series:
        """価格データのみからファクター類似の推定を行う。

        モメンタム、ボラティリティ、サイズをプロキシとして使用。
        """
        assets = prices.columns.tolist()
        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return pd.Series(0.0, index=assets)

        # モメンタムファクター（12ヶ月リターン）
        if len(prices) >= 252:
            momentum = (prices.iloc[-1] / prices.iloc[-252] - 1).rank(pct=True) - 0.5
        else:
            momentum = pd.Series(0.0, index=assets)

        # ボラティリティファクター（低ボラ優位）
        volatility = returns.std()
        vol_score = (1 - volatility.rank(pct=True)) - 0.5  # 低ボラが高スコア

        # 簡易的なファクタープレミアム（モメンタム0.5、低ボラ0.5の加重）
        # 年率5%のプレミアムを仮定
        expected_return = (momentum * 0.5 + vol_score * 0.5) * 0.05

        return expected_return


@dataclass
class DynamicReturnEstimatorConfig:
    """DynamicReturnEstimator の設定。

    Attributes:
        momentum_lookback: モメンタムのルックバック期間
        momentum_premium: モメンタムのベースプレミアム
        risk_aversion: リスク回避係数
        reversion_lookback: 平均回帰のルックバック期間
        reversion_rate: 平均回帰率
        factor_halflife: ファクターEWMAの半減期
    """

    momentum_lookback: int = 60
    momentum_premium: float = 0.10
    risk_aversion: float = 2.5
    reversion_lookback: int = 252
    reversion_rate: float = 0.5
    factor_halflife: int = 60


class DynamicReturnEstimator:
    """動的期待リターン推定クラス。

    複数の期待リターン推定手法を統合し、
    市場レジームに応じて最適な手法を選択・統合する。

    Usage:
        estimator = DynamicReturnEstimator()
        result = estimator.estimate(prices, regime=MarketRegime.BULL_TREND)
        print(result.combined)
    """

    # レジーム別の手法重み
    REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
        MarketRegime.BULL_TREND: {
            "cross_sectional": 0.5,
            "implied": 0.3,
            "factor": 0.2,
            "mean_reversion": 0.0,
        },
        MarketRegime.BEAR_TREND: {
            "cross_sectional": 0.2,
            "implied": 0.4,
            "factor": 0.2,
            "mean_reversion": 0.2,
        },
        MarketRegime.RANGE_BOUND: {
            "cross_sectional": 0.0,
            "implied": 0.3,
            "factor": 0.2,
            "mean_reversion": 0.5,
        },
        MarketRegime.HIGH_VOL: {
            "cross_sectional": 0.2,
            "implied": 0.6,
            "factor": 0.0,
            "mean_reversion": 0.2,
        },
        MarketRegime.LOW_VOL: {
            "cross_sectional": 0.4,
            "implied": 0.3,
            "factor": 0.2,
            "mean_reversion": 0.1,
        },
        MarketRegime.NEUTRAL: {
            "cross_sectional": 0.3,
            "implied": 0.4,
            "factor": 0.0,
            "mean_reversion": 0.3,
        },
    }

    def __init__(
        self,
        config: DynamicReturnEstimatorConfig | None = None,
        regime_weights: dict[MarketRegime, dict[str, float]] | None = None,
    ) -> None:
        """初期化。

        Args:
            config: 設定
            regime_weights: レジーム別の手法重み（カスタム）
        """
        self.config = config or DynamicReturnEstimatorConfig()
        self.regime_weights = regime_weights or self.REGIME_WEIGHTS

        # 各推定器を初期化
        self.cross_sectional = CrossSectionalMomentum(
            lookback=self.config.momentum_lookback,
            base_premium=self.config.momentum_premium,
        )
        self.implied = ImpliedReturns(
            risk_aversion=self.config.risk_aversion,
        )
        self.mean_reversion = MeanReversionForecast(
            lookback=self.config.reversion_lookback,
            reversion_rate=self.config.reversion_rate,
        )
        self.factor = FactorPremiumTiming(
            ewm_halflife=self.config.factor_halflife,
        )

    def estimate(
        self,
        prices: pd.DataFrame,
        regime: MarketRegime | str = MarketRegime.NEUTRAL,
        weights_market: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
        factor_exposures: pd.DataFrame | None = None,
        factor_returns_history: pd.DataFrame | None = None,
    ) -> ReturnEstimate:
        """期待リターンを推定する。

        Args:
            prices: 価格データ
            regime: 市場レジーム
            weights_market: 市場ウェイト（インプライドリターン用）
            cov: 共分散行列（インプライドリターン用）
            factor_exposures: ファクターエクスポージャー
            factor_returns_history: ファクターリターン履歴

        Returns:
            推定結果
        """
        if isinstance(regime, str):
            regime = MarketRegime(regime)

        # 各手法で推定
        estimates = {}

        try:
            estimates["cross_sectional"] = self.cross_sectional.estimate(prices)
        except Exception as e:
            logger.warning(f"Cross-sectional momentum failed: {e}")
            estimates["cross_sectional"] = None

        try:
            estimates["implied"] = self.implied.estimate(
                prices, weights_market=weights_market, cov=cov
            )
        except Exception as e:
            logger.warning(f"Implied returns failed: {e}")
            estimates["implied"] = None

        try:
            estimates["mean_reversion"] = self.mean_reversion.estimate(prices)
        except Exception as e:
            logger.warning(f"Mean reversion forecast failed: {e}")
            estimates["mean_reversion"] = None

        try:
            estimates["factor"] = self.factor.estimate(
                prices,
                factor_exposures=factor_exposures,
                factor_returns_history=factor_returns_history,
            )
        except Exception as e:
            logger.warning(f"Factor premium failed: {e}")
            estimates["factor"] = None

        # レジームに応じて統合
        combined, weights_used = self._weighted_combine(estimates, regime)

        return ReturnEstimate(
            cross_sectional=estimates.get("cross_sectional"),
            implied=estimates.get("implied"),
            mean_reversion=estimates.get("mean_reversion"),
            factor=estimates.get("factor"),
            combined=combined,
            regime=regime,
            weights_used=weights_used,
            metadata={
                "n_assets": len(prices.columns),
                "n_observations": len(prices),
            },
        )

    def _weighted_combine(
        self,
        estimates: dict[str, pd.Series | None],
        regime: MarketRegime,
    ) -> tuple[pd.Series, dict[str, float]]:
        """推定値を重み付けして統合する。

        Args:
            estimates: 各手法の推定値
            regime: 市場レジーム

        Returns:
            (統合された推定値, 使用された重み)
        """
        weights = self.regime_weights.get(regime, self.regime_weights[MarketRegime.NEUTRAL])

        # 有効な推定値のみを使用
        valid_estimates = {k: v for k, v in estimates.items() if v is not None}

        if not valid_estimates:
            # フォールバック: 全てのアセットに0を返す
            first_estimate = next(iter(estimates.values()))
            if first_estimate is not None:
                return pd.Series(0.0, index=first_estimate.index), {}
            return pd.Series(dtype=float), {}

        # 有効な推定値に基づいて重みを再正規化
        valid_weights = {k: weights.get(k, 0.0) for k in valid_estimates.keys()}
        total_weight = sum(valid_weights.values())

        if total_weight == 0:
            # 全て等ウェイト
            n = len(valid_estimates)
            valid_weights = {k: 1.0 / n for k in valid_estimates.keys()}
        else:
            valid_weights = {k: v / total_weight for k, v in valid_weights.items()}

        # 加重平均
        combined = sum(
            valid_estimates[k] * w for k, w in valid_weights.items()
        )

        logger.debug(
            f"Combined returns for regime={regime.value}: "
            f"weights={valid_weights}, range=[{combined.min():.4f}, {combined.max():.4f}]"
        )

        return combined, valid_weights

    def detect_regime(
        self,
        prices: pd.DataFrame,
        lookback: int = 60,
    ) -> MarketRegime:
        """価格データから市場レジームを検出する。

        Args:
            prices: 価格データ
            lookback: ルックバック期間

        Returns:
            検出されたレジーム
        """
        if len(prices) < lookback:
            return MarketRegime.NEUTRAL

        # 市場全体の代表値（等ウェイトポートフォリオ）
        portfolio = prices.mean(axis=1)
        returns = portfolio.pct_change().dropna()

        if len(returns) < 20:
            return MarketRegime.NEUTRAL

        recent_returns = returns.iloc[-lookback:]

        # ボラティリティ
        volatility = recent_returns.std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)

        # トレンド
        cumulative_return = (1 + recent_returns).prod() - 1
        annualized_return = (1 + cumulative_return) ** (252 / lookback) - 1

        # レジーム判定
        if volatility > historical_vol * 1.5:
            return MarketRegime.HIGH_VOL
        elif volatility < historical_vol * 0.7:
            return MarketRegime.LOW_VOL
        elif annualized_return > 0.15:
            return MarketRegime.BULL_TREND
        elif annualized_return < -0.10:
            return MarketRegime.BEAR_TREND
        elif abs(annualized_return) < 0.05:
            return MarketRegime.RANGE_BOUND
        else:
            return MarketRegime.NEUTRAL


def create_return_estimator(
    config: DynamicReturnEstimatorConfig | None = None,
) -> DynamicReturnEstimator:
    """DynamicReturnEstimator のファクトリ関数。

    Args:
        config: 設定（オプション）

    Returns:
        初期化された DynamicReturnEstimator
    """
    return DynamicReturnEstimator(config=config)


def quick_estimate_returns(
    prices: pd.DataFrame,
    regime: MarketRegime | str | None = None,
) -> pd.Series:
    """便利関数: 簡易的に期待リターンを推定する。

    Args:
        prices: 価格データ
        regime: 市場レジーム（Noneの場合は自動検出）

    Returns:
        期待リターン
    """
    estimator = DynamicReturnEstimator()

    if regime is None:
        regime = estimator.detect_regime(prices)
    elif isinstance(regime, str):
        regime = MarketRegime(regime)

    result = estimator.estimate(prices, regime=regime)
    return result.combined
