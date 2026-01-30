"""
Dynamic Parameters Manager - 動的パラメータ一元管理モジュール

動的閾値・パラメータ計算機能を統合した単一モジュール。
市場環境に応じて動的にパラメータを計算する機能を提供。

統合対象:
- src/analysis/dynamic_threshold.py (VIX, 相関, ケリー)
- src/analysis/dynamic_thresholds.py (リバランス, スムージング, ポジション)

主要コンポーネント:
- DynamicParamsManager: 一元管理クラス
- 各種Calculator（リバランス、スムージング、VIX、相関、ケリー）
- レジーム適応パラメータ

使用例:
    from src.meta.dynamic_params import DynamicParamsManager

    manager = DynamicParamsManager()

    # リバランス閾値
    threshold = manager.calculate_threshold(
        returns=portfolio_returns,
        threshold_type="rebalance",
        transaction_cost_bps=10,
    )

    # 動的パラメータ一括計算
    params = manager.calculate_params(
        returns=portfolio_returns,
        asset_returns=asset_returns_df,
    )

    # レジーム適応パラメータ
    regime_params = manager.get_regime_adaptive_params(
        returns=portfolio_returns,
        vix_value=25.0,
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ThresholdType(str, Enum):
    """閾値タイプ"""
    REBALANCE = "rebalance"
    SMOOTHING = "smoothing"
    POSITION = "position"
    VIX = "vix"
    CORRELATION = "correlation"
    KELLY = "kelly"


class VolatilityRegime(str, Enum):
    """ボラティリティレジーム"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class MarketRegime(str, Enum):
    """市場レジーム"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class ThresholdResult:
    """閾値計算結果の基底クラス"""
    value: float
    threshold_type: ThresholdType
    timestamp: datetime = field(default_factory=datetime.now)
    lookback_days: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "threshold_type": self.threshold_type.value,
            "timestamp": self.timestamp.isoformat(),
            "lookback_days": self.lookback_days,
            "metadata": self.metadata,
        }


@dataclass
class RebalanceThresholdResult(ThresholdResult):
    """リバランス閾値結果"""
    cost_based_threshold: float = 0.0
    vol_based_threshold: float = 0.0
    dominant_factor: str = ""
    daily_volatility: float = 0.0
    avg_holding_period: float = 21.0
    transaction_cost: float = 0.0
    threshold_type: ThresholdType = field(default=ThresholdType.REBALANCE)


@dataclass
class SmoothingAlphaResult(ThresholdResult):
    """スムージングアルファ結果"""
    base_alpha: float = 0.3
    vol_ratio: float = 1.0
    current_volatility: float = 0.0
    long_term_volatility: float = 0.0
    threshold_type: ThresholdType = field(default=ThresholdType.SMOOTHING)


@dataclass
class PositionLimitResult(ThresholdResult):
    """ポジション上限結果"""
    base_limit: float = 0.25
    asset_volatility: float = 0.0
    target_volatility: float = 0.02
    vol_adjustment_factor: float = 1.0
    threshold_type: ThresholdType = field(default=ThresholdType.POSITION)


@dataclass
class VixThresholdResult(ThresholdResult):
    """VIX閾値結果"""
    low: float = 15.0
    high: float = 25.0
    extreme: float = 35.0
    current: Optional[float] = None
    median: float = 18.0
    mean: float = 19.0
    threshold_type: ThresholdType = field(default=ThresholdType.VIX)

    def get_regime(self, vix_value: float) -> str:
        if vix_value <= self.low:
            return "low_vol"
        elif vix_value <= self.high:
            return "normal"
        elif vix_value <= self.extreme:
            return "high_vol"
        else:
            return "extreme"


@dataclass
class CorrelationThresholdResult(ThresholdResult):
    """相関閾値結果"""
    baseline: float = 0.3
    warning: float = 0.6
    critical: float = 0.8
    current: Optional[float] = None
    std: float = 0.15
    threshold_type: ThresholdType = field(default=ThresholdType.CORRELATION)

    def get_level(self, correlation: float) -> str:
        if correlation <= self.baseline:
            return "normal"
        elif correlation <= self.warning:
            return "elevated"
        elif correlation <= self.critical:
            return "warning"
        else:
            return "critical"


@dataclass
class KellyResult(ThresholdResult):
    """ケリー基準結果"""
    full_kelly: float = 0.0
    half_kelly: float = 0.0
    quarter_kelly: float = 0.0
    win_rate: float = 0.5
    payoff_ratio: float = 1.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    edge: float = 0.0
    total_trades: int = 0
    threshold_type: ThresholdType = field(default=ThresholdType.KELLY)

    def get_recommended_fraction(self, risk_tolerance: str = "moderate") -> float:
        if risk_tolerance == "conservative":
            return self.quarter_kelly
        elif risk_tolerance == "aggressive":
            return self.half_kelly
        else:  # moderate
            return self.half_kelly * 0.75


@dataclass
class DynamicParamsBundle:
    """動的パラメータバンドル"""
    rebalance_threshold: Optional[RebalanceThresholdResult] = None
    smoothing_alpha: Optional[SmoothingAlphaResult] = None
    position_limits: Optional[Dict[str, PositionLimitResult]] = None
    vix_thresholds: Optional[VixThresholdResult] = None
    correlation_threshold: Optional[CorrelationThresholdResult] = None
    kelly_params: Optional[KellyResult] = None
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rebalance_threshold": self.rebalance_threshold.to_dict() if self.rebalance_threshold else None,
            "smoothing_alpha": self.smoothing_alpha.to_dict() if self.smoothing_alpha else None,
            "position_limits": {k: v.to_dict() for k, v in self.position_limits.items()} if self.position_limits else None,
            "vix_thresholds": self.vix_thresholds.to_dict() if self.vix_thresholds else None,
            "correlation_threshold": self.correlation_threshold.to_dict() if self.correlation_threshold else None,
            "kelly_params": self.kelly_params.to_dict() if self.kelly_params else None,
            "volatility_regime": self.volatility_regime.value,
            "market_regime": self.market_regime.value,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class RegimeAdaptiveParams:
    """レジーム適応パラメータ"""
    volatility_regime: VolatilityRegime
    market_regime: MarketRegime
    risk_multiplier: float = 1.0
    position_scale: float = 1.0
    rebalance_threshold: float = 0.05
    smoothing_alpha: float = 0.3
    cash_allocation: float = 0.0
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility_regime": self.volatility_regime.value,
            "market_regime": self.market_regime.value,
            "risk_multiplier": self.risk_multiplier,
            "position_scale": self.position_scale,
            "rebalance_threshold": self.rebalance_threshold,
            "smoothing_alpha": self.smoothing_alpha,
            "cash_allocation": self.cash_allocation,
            "computed_at": self.computed_at.isoformat(),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def detect_volatility_regime(
    returns: pd.Series,
    short_lookback: int = 20,
    long_lookback: int = 60,
) -> VolatilityRegime:
    """ボラティリティレジームを検出"""
    if len(returns) < long_lookback:
        return VolatilityRegime.NORMAL

    recent_vol = returns.tail(short_lookback).std() * np.sqrt(252)
    long_vol = returns.tail(long_lookback).std() * np.sqrt(252)

    if long_vol == 0:
        return VolatilityRegime.NORMAL

    vol_ratio = recent_vol / long_vol

    if vol_ratio < 0.7:
        return VolatilityRegime.LOW
    elif vol_ratio < 1.3:
        return VolatilityRegime.NORMAL
    elif vol_ratio < 2.0:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


def detect_market_regime(
    returns: pd.Series,
    lookback: int = 60,
) -> MarketRegime:
    """市場レジームを検出"""
    if len(returns) < lookback:
        return MarketRegime.SIDEWAYS

    cumret = (1 + returns.tail(lookback)).cumprod() - 1
    total_return = cumret.iloc[-1]
    max_dd = (cumret.cummax() - cumret).max()

    if max_dd > 0.2:
        return MarketRegime.CRISIS
    elif total_return > 0.10:
        return MarketRegime.BULL
    elif total_return < -0.10:
        return MarketRegime.BEAR
    else:
        return MarketRegime.SIDEWAYS


# =============================================================================
# Main Manager Class
# =============================================================================

class DynamicParamsManager:
    """動的パラメータの一元管理クラス

    Usage:
        manager = DynamicParamsManager()

        # 単一閾値計算
        threshold = manager.calculate_threshold(
            returns=portfolio_returns,
            threshold_type="rebalance",
        )

        # 一括計算
        params = manager.calculate_params(
            returns=portfolio_returns,
            asset_returns=asset_returns_df,
        )

        # レジーム適応
        regime_params = manager.get_regime_adaptive_params(
            returns=portfolio_returns,
            vix_value=25.0,
        )
    """

    DEFAULT_LOOKBACK = 60
    DEFAULT_TRANSACTION_COST_BPS = 10
    DEFAULT_HOLDING_PERIOD = 21

    def __init__(
        self,
        default_lookback: int = 60,
        min_observations: int = 30,
    ) -> None:
        self.default_lookback = default_lookback
        self.min_observations = min_observations

    def calculate_threshold(
        self,
        returns: pd.Series,
        threshold_type: Union[str, ThresholdType] = "rebalance",
        **kwargs,
    ) -> ThresholdResult:
        """動的閾値を計算

        Args:
            returns: リターン系列
            threshold_type: 閾値タイプ
            **kwargs: 閾値タイプ別パラメータ

        Returns:
            ThresholdResult
        """
        if isinstance(threshold_type, str):
            threshold_type = ThresholdType(threshold_type)

        if threshold_type == ThresholdType.REBALANCE:
            return self._calculate_rebalance_threshold(returns, **kwargs)
        elif threshold_type == ThresholdType.SMOOTHING:
            return self._calculate_smoothing_alpha(returns, **kwargs)
        elif threshold_type == ThresholdType.POSITION:
            return self._calculate_position_limit(returns, **kwargs)
        elif threshold_type == ThresholdType.VIX:
            return self._calculate_vix_thresholds(returns, **kwargs)
        elif threshold_type == ThresholdType.CORRELATION:
            asset_returns = kwargs.get("asset_returns")
            if asset_returns is None:
                raise ValueError("asset_returns required for correlation threshold")
            return self._calculate_correlation_threshold(asset_returns, **kwargs)
        elif threshold_type == ThresholdType.KELLY:
            return self._calculate_kelly_params(returns, **kwargs)
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

    def calculate_params(
        self,
        returns: pd.Series,
        asset_returns: Optional[pd.DataFrame] = None,
        vix_history: Optional[pd.Series] = None,
        transaction_cost_bps: float = 10,
        base_alpha: float = 0.3,
        base_limit: float = 0.25,
        target_vol: float = 0.02,
        lookback_days: int = 60,
    ) -> DynamicParamsBundle:
        """動的パラメータを一括計算

        Args:
            returns: ポートフォリオリターン
            asset_returns: 資産別リターン
            vix_history: VIX履歴
            transaction_cost_bps: 取引コスト(bps)
            base_alpha: ベーススムージングアルファ
            base_limit: ベースポジション上限
            target_vol: 目標ボラティリティ
            lookback_days: ルックバック日数

        Returns:
            DynamicParamsBundle
        """
        bundle = DynamicParamsBundle()

        # ボラティリティレジーム
        bundle.volatility_regime = detect_volatility_regime(returns, long_lookback=lookback_days)

        # 市場レジーム
        bundle.market_regime = detect_market_regime(returns, lookback=lookback_days)

        # リバランス閾値
        bundle.rebalance_threshold = self._calculate_rebalance_threshold(
            returns,
            transaction_cost_bps=transaction_cost_bps,
            lookback_days=lookback_days,
        )

        # スムージングアルファ
        bundle.smoothing_alpha = self._calculate_smoothing_alpha(
            returns,
            base_alpha=base_alpha,
            long_lookback=lookback_days,
        )

        # ポジション上限
        if asset_returns is not None:
            bundle.position_limits = {}
            for asset in asset_returns.columns:
                bundle.position_limits[asset] = self._calculate_position_limit(
                    asset_returns[asset],
                    base_limit=base_limit,
                    target_vol=target_vol,
                    lookback_days=lookback_days,
                )

        # VIX閾値
        if vix_history is not None and len(vix_history) > 0:
            bundle.vix_thresholds = self._calculate_vix_thresholds(
                vix_history,
                lookback_days=lookback_days,
            )

        # 相関閾値
        if asset_returns is not None and len(asset_returns.columns) > 1:
            bundle.correlation_threshold = self._calculate_correlation_threshold(
                asset_returns,
                lookback_days=lookback_days,
            )

        # ケリー基準
        bundle.kelly_params = self._calculate_kelly_params(
            returns,
            lookback_days=lookback_days,
        )

        return bundle

    def get_regime_adaptive_params(
        self,
        returns: pd.Series,
        vix_value: Optional[float] = None,
        vix_history: Optional[pd.Series] = None,
        base_risk_multiplier: float = 1.0,
    ) -> RegimeAdaptiveParams:
        """レジーム適応パラメータを取得

        Args:
            returns: リターン系列
            vix_value: 現在のVIX値
            vix_history: VIX履歴
            base_risk_multiplier: ベースリスク乗数

        Returns:
            RegimeAdaptiveParams
        """
        vol_regime = detect_volatility_regime(returns)
        market_regime = detect_market_regime(returns)

        # ボラティリティレジームに応じた調整
        vol_adjustments = {
            VolatilityRegime.LOW: {"risk": 1.1, "position": 1.1, "threshold": 0.03, "alpha": 0.2, "cash": 0.0},
            VolatilityRegime.NORMAL: {"risk": 1.0, "position": 1.0, "threshold": 0.05, "alpha": 0.3, "cash": 0.0},
            VolatilityRegime.HIGH: {"risk": 0.7, "position": 0.7, "threshold": 0.08, "alpha": 0.5, "cash": 0.15},
            VolatilityRegime.EXTREME: {"risk": 0.4, "position": 0.4, "threshold": 0.12, "alpha": 0.7, "cash": 0.30},
        }

        # 市場レジームに応じた追加調整
        market_adjustments = {
            MarketRegime.BULL: {"risk_mult": 1.0, "cash_add": 0.0},
            MarketRegime.BEAR: {"risk_mult": 0.8, "cash_add": 0.10},
            MarketRegime.SIDEWAYS: {"risk_mult": 0.9, "cash_add": 0.0},
            MarketRegime.CRISIS: {"risk_mult": 0.5, "cash_add": 0.25},
        }

        vol_adj = vol_adjustments[vol_regime]
        market_adj = market_adjustments[market_regime]

        risk_multiplier = base_risk_multiplier * vol_adj["risk"] * market_adj["risk_mult"]
        position_scale = vol_adj["position"]
        rebalance_threshold = vol_adj["threshold"]
        smoothing_alpha = vol_adj["alpha"]
        cash_allocation = min(vol_adj["cash"] + market_adj["cash_add"], 0.8)

        # VIXによる追加調整
        if vix_value is not None:
            if vix_value > 35:
                risk_multiplier *= 0.6
                cash_allocation = min(cash_allocation + 0.20, 0.8)
            elif vix_value > 25:
                risk_multiplier *= 0.8
                cash_allocation = min(cash_allocation + 0.10, 0.8)

        return RegimeAdaptiveParams(
            volatility_regime=vol_regime,
            market_regime=market_regime,
            risk_multiplier=risk_multiplier,
            position_scale=position_scale,
            rebalance_threshold=rebalance_threshold,
            smoothing_alpha=smoothing_alpha,
            cash_allocation=cash_allocation,
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _calculate_rebalance_threshold(
        self,
        returns: pd.Series,
        transaction_cost_bps: float = 10,
        lookback_days: int = 60,
        holding_period: int = 21,
    ) -> RebalanceThresholdResult:
        """リバランス閾値を計算"""
        if len(returns) < self.min_observations:
            return RebalanceThresholdResult(
                value=0.05,
                cost_based_threshold=transaction_cost_bps / 10000 * 2,
                vol_based_threshold=0.05,
                dominant_factor="default",
                metadata={"warning": "insufficient_data"},
            )

        recent = returns.tail(lookback_days)
        daily_vol = float(recent.std())

        # コストベース閾値
        cost_threshold = transaction_cost_bps / 10000 * 2

        # ボラベース閾値
        vol_threshold = daily_vol * np.sqrt(holding_period)

        # 最終閾値
        threshold = max(cost_threshold, vol_threshold)
        threshold = np.clip(threshold, 0.01, 0.20)

        dominant = "cost" if cost_threshold >= vol_threshold else "volatility"

        return RebalanceThresholdResult(
            value=float(threshold),
            cost_based_threshold=float(cost_threshold),
            vol_based_threshold=float(vol_threshold),
            dominant_factor=dominant,
            daily_volatility=daily_vol,
            avg_holding_period=float(holding_period),
            transaction_cost=transaction_cost_bps / 10000,
            lookback_days=lookback_days,
        )

    def _calculate_smoothing_alpha(
        self,
        returns: pd.Series,
        base_alpha: float = 0.3,
        short_lookback: int = 20,
        long_lookback: int = 60,
    ) -> SmoothingAlphaResult:
        """スムージングアルファを計算"""
        if len(returns) < self.min_observations:
            return SmoothingAlphaResult(
                value=base_alpha,
                base_alpha=base_alpha,
                metadata={"warning": "insufficient_data"},
            )

        current_vol = returns.tail(short_lookback).std()
        long_vol = returns.tail(long_lookback).std()

        vol_ratio = current_vol / long_vol if long_vol > 0 else 1.0
        alpha = base_alpha * (0.5 + 0.5 * vol_ratio)
        alpha = np.clip(alpha, 0.1, 0.8)

        return SmoothingAlphaResult(
            value=float(alpha),
            base_alpha=base_alpha,
            vol_ratio=float(vol_ratio),
            current_volatility=float(current_vol),
            long_term_volatility=float(long_vol),
            lookback_days=long_lookback,
        )

    def _calculate_position_limit(
        self,
        returns: pd.Series,
        base_limit: float = 0.25,
        target_vol: float = 0.02,
        lookback_days: int = 60,
    ) -> PositionLimitResult:
        """ポジション上限を計算"""
        if len(returns) < self.min_observations:
            return PositionLimitResult(
                value=base_limit,
                base_limit=base_limit,
                target_volatility=target_vol,
                metadata={"warning": "insufficient_data"},
            )

        asset_vol = returns.tail(lookback_days).std()
        vol_adjustment = target_vol / asset_vol if asset_vol > 0 else 1.0
        limit = base_limit * vol_adjustment
        limit = np.clip(limit, 0.05, 0.40)

        return PositionLimitResult(
            value=float(limit),
            base_limit=base_limit,
            asset_volatility=float(asset_vol),
            target_volatility=target_vol,
            vol_adjustment_factor=float(vol_adjustment),
            lookback_days=lookback_days,
        )

    def _calculate_vix_thresholds(
        self,
        vix_history: pd.Series,
        lookback_days: int = 252,
        percentiles: tuple = (20, 80, 95),
    ) -> VixThresholdResult:
        """VIX閾値を計算"""
        if len(vix_history) < self.min_observations:
            return VixThresholdResult(
                value=20.0,
                low=15.0,
                high=25.0,
                extreme=35.0,
                metadata={"warning": "insufficient_data"},
            )

        vix = vix_history.tail(lookback_days).dropna()
        low_pct, high_pct, extreme_pct = percentiles

        low = float(np.percentile(vix, low_pct))
        high = float(np.percentile(vix, high_pct))
        extreme = float(np.percentile(vix, extreme_pct))
        current = float(vix.iloc[-1]) if len(vix) > 0 else None

        return VixThresholdResult(
            value=float(np.median(vix)),
            low=low,
            high=high,
            extreme=extreme,
            current=current,
            median=float(np.median(vix)),
            mean=float(np.mean(vix)),
            lookback_days=lookback_days,
        )

    def _calculate_correlation_threshold(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 252,
        warning_sigma: float = 1.5,
        critical_sigma: float = 2.5,
    ) -> CorrelationThresholdResult:
        """相関閾値を計算"""
        if len(returns) < self.min_observations or len(returns.columns) < 2:
            return CorrelationThresholdResult(
                value=0.3,
                baseline=0.3,
                warning=0.6,
                critical=0.8,
                metadata={"warning": "insufficient_data"},
            )

        recent = returns.tail(lookback_days)
        corr_matrix = recent.corr()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack()

        baseline = float(correlations.mean())
        std = float(correlations.std())
        warning = min(baseline + warning_sigma * std, 0.95)
        critical = min(baseline + critical_sigma * std, 0.99)

        return CorrelationThresholdResult(
            value=baseline,
            baseline=baseline,
            warning=warning,
            critical=critical,
            std=std,
            lookback_days=lookback_days,
        )

    def _calculate_kelly_params(
        self,
        returns: pd.Series,
        lookback_days: int = 252,
        risk_free_rate: float = 0.0,
    ) -> KellyResult:
        """ケリー基準を計算"""
        if len(returns) < self.min_observations:
            return KellyResult(
                value=0.0,
                full_kelly=0.0,
                half_kelly=0.0,
                quarter_kelly=0.0,
                metadata={"warning": "insufficient_data"},
            )

        recent = returns.tail(lookback_days)
        excess = recent - risk_free_rate

        wins = excess[excess > 0]
        losses = excess[excess < 0]

        total_trades = len(excess)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.5
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0001

        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        q = 1 - win_rate
        full_kelly = (win_rate * payoff_ratio - q) / payoff_ratio if payoff_ratio > 0 else 0.0
        full_kelly = np.clip(full_kelly, 0.0, 1.0)

        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25
        edge = win_rate * avg_win - q * avg_loss

        return KellyResult(
            value=half_kelly,
            full_kelly=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            edge=edge,
            total_trades=total_trades,
            lookback_days=lookback_days,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_rebalance_threshold(
    returns: pd.Series,
    transaction_cost_bps: float = 10,
    lookback_days: int = 60,
) -> float:
    """リバランス閾値を計算する便利関数"""
    manager = DynamicParamsManager()
    result = manager.calculate_threshold(
        returns,
        threshold_type="rebalance",
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=lookback_days,
    )
    return result.value


def calculate_smoothing_alpha(
    returns: pd.Series,
    base_alpha: float = 0.3,
    lookback: int = 60,
) -> float:
    """スムージングアルファを計算する便利関数"""
    manager = DynamicParamsManager()
    result = manager.calculate_threshold(
        returns,
        threshold_type="smoothing",
        base_alpha=base_alpha,
        long_lookback=lookback,
    )
    return result.value


def calculate_position_limit(
    returns: pd.Series,
    base_limit: float = 0.25,
    target_vol: float = 0.02,
) -> float:
    """ポジション上限を計算する便利関数"""
    manager = DynamicParamsManager()
    result = manager.calculate_threshold(
        returns,
        threshold_type="position",
        base_limit=base_limit,
        target_vol=target_vol,
    )
    return result.value


def get_regime_params(
    returns: pd.Series,
    vix_value: Optional[float] = None,
) -> RegimeAdaptiveParams:
    """レジーム適応パラメータを取得する便利関数"""
    manager = DynamicParamsManager()
    return manager.get_regime_adaptive_params(returns, vix_value=vix_value)


def create_params_manager() -> DynamicParamsManager:
    """DynamicParamsManager のファクトリ関数"""
    return DynamicParamsManager()
