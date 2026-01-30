"""
Adaptive Lookback Module.

Provides dynamic lookback period selection based on market regime.
High volatility environments use shorter lookbacks for faster adaptation,
while low volatility environments use longer lookbacks for stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""

    BULL_TREND = "bull_trend"
    BEAR_MARKET = "bear_market"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


@dataclass
class LookbackConfig:
    """Configuration for lookback periods in a specific regime."""

    short: List[int] = field(default_factory=lambda: [5, 10, 20])
    medium: List[int] = field(default_factory=lambda: [20, 40, 60])
    long: List[int] = field(default_factory=lambda: [60, 120, 252])
    decay_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    def get_all_periods(self) -> List[int]:
        """Get all lookback periods combined."""
        return self.short + self.medium + self.long

    def get_category_weights(self) -> Dict[str, float]:
        """Get weights for each category (short, medium, long)."""
        return {
            "short": self.decay_weights[0],
            "medium": self.decay_weights[1],
            "long": self.decay_weights[2],
        }


@dataclass
class AdaptiveLookbackResult:
    """Result of adaptive lookback computation."""

    regime: str
    lookback_periods: Dict[str, List[int]]
    decay_weights: List[float]
    multi_period_score: Optional[pd.Series] = None
    component_scores: Optional[Dict[int, pd.Series]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Default regime configurations
DEFAULT_REGIME_CONFIGS: Dict[str, LookbackConfig] = {
    MarketRegime.BULL_TREND.value: LookbackConfig(
        short=[5, 10, 20],
        medium=[20, 40, 60],
        long=[60, 120, 252],
        decay_weights=[0.5, 0.3, 0.2],
    ),
    MarketRegime.BEAR_MARKET.value: LookbackConfig(
        short=[3, 5, 10],
        medium=[10, 20, 30],
        long=[30, 60, 90],
        decay_weights=[0.6, 0.3, 0.1],
    ),
    MarketRegime.HIGH_VOL.value: LookbackConfig(
        short=[3, 5, 10],
        medium=[10, 20, 40],
        long=[40, 60, 90],
        decay_weights=[0.55, 0.30, 0.15],
    ),
    MarketRegime.LOW_VOL.value: LookbackConfig(
        short=[10, 20, 40],
        medium=[40, 60, 90],
        long=[90, 120, 252],
        decay_weights=[0.25, 0.35, 0.40],
    ),
    MarketRegime.NEUTRAL.value: LookbackConfig(
        short=[5, 10, 20],
        medium=[20, 40, 60],
        long=[60, 90, 120],
        decay_weights=[0.40, 0.35, 0.25],
    ),
}


class AdaptiveLookback:
    """
    Adaptive lookback period selector based on market regime.

    Dynamically selects appropriate lookback periods based on current market
    conditions. High volatility environments benefit from shorter lookbacks
    for faster adaptation, while stable markets can use longer lookbacks.

    Example:
        >>> adaptive = AdaptiveLookback()
        >>> periods = adaptive.get_lookback_periods("high_vol")
        >>> print(periods["short"])
        [3, 5, 10]

        >>> # Compute multi-period score
        >>> def momentum_signal(prices, lookback):
        ...     return prices.pct_change(lookback).iloc[-1]
        >>> score = adaptive.compute_multi_period_score(
        ...     prices, momentum_signal, "bull_trend"
        ... )
    """

    def __init__(
        self,
        regime_configs: Optional[Dict[str, LookbackConfig]] = None,
        regime_detector: Optional[Any] = None,
    ):
        """
        Initialize AdaptiveLookback.

        Args:
            regime_configs: Custom regime configurations. If None, uses defaults.
            regime_detector: Optional RegimeDetector instance for auto-detection.
        """
        self._configs = regime_configs or DEFAULT_REGIME_CONFIGS.copy()
        self._regime_detector = regime_detector

        logger.info(
            "AdaptiveLookback initialized",
            regimes=list(self._configs.keys()),
            has_detector=regime_detector is not None,
        )

    def get_lookback_periods(self, regime: str) -> Dict[str, List[int]]:
        """
        Get lookback periods for a given market regime.

        Args:
            regime: Market regime name (bull_trend, bear_market, high_vol, low_vol, neutral)

        Returns:
            Dictionary with 'short', 'medium', 'long' period lists
        """
        config = self._get_config(regime)
        return {
            "short": config.short.copy(),
            "medium": config.medium.copy(),
            "long": config.long.copy(),
        }

    def get_decay_weights(self, regime: str) -> List[float]:
        """
        Get decay weights for short/medium/long categories.

        Args:
            regime: Market regime name

        Returns:
            List of weights [short_weight, medium_weight, long_weight]
        """
        config = self._get_config(regime)
        return config.decay_weights.copy()

    def get_all_periods(self, regime: str) -> List[int]:
        """
        Get all lookback periods for a regime (combined).

        Args:
            regime: Market regime name

        Returns:
            List of all unique periods, sorted
        """
        config = self._get_config(regime)
        all_periods = set(config.short + config.medium + config.long)
        return sorted(all_periods)

    def detect_regime(self, prices: pd.DataFrame) -> str:
        """
        Detect current market regime from price data.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            Detected regime name
        """
        if self._regime_detector is not None:
            # Use external regime detector
            try:
                result = self._regime_detector.compute(prices)
                return self._map_regime_score_to_name(result.scores.iloc[-1])
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}, using neutral")
                return MarketRegime.NEUTRAL.value

        # Simple built-in detection
        return self._simple_regime_detection(prices)

    def _simple_regime_detection(self, prices: pd.DataFrame) -> str:
        """
        Simple built-in regime detection.

        Args:
            prices: DataFrame with 'close' column

        Returns:
            Detected regime name
        """
        if "close" not in prices.columns:
            return MarketRegime.NEUTRAL.value

        close = prices["close"]
        if len(close) < 60:
            return MarketRegime.NEUTRAL.value

        # Calculate metrics
        returns = close.pct_change().dropna()
        recent_returns = returns.tail(20)
        long_returns = returns.tail(60)

        # Volatility (annualized)
        recent_vol = recent_returns.std() * np.sqrt(252)
        long_vol = long_returns.std() * np.sqrt(252)

        # Trend (20-day vs 60-day SMA)
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_60 = close.rolling(60).mean().iloc[-1]
        current_price = close.iloc[-1]

        # Regime classification
        vol_percentile = recent_vol / long_vol if long_vol > 0 else 1.0

        if vol_percentile > 1.3:
            return MarketRegime.HIGH_VOL.value
        elif vol_percentile < 0.7:
            return MarketRegime.LOW_VOL.value
        elif current_price > sma_20 > sma_60:
            return MarketRegime.BULL_TREND.value
        elif current_price < sma_20 < sma_60:
            return MarketRegime.BEAR_MARKET.value
        else:
            return MarketRegime.NEUTRAL.value

    def _map_regime_score_to_name(self, score: float) -> str:
        """
        Map regime detector score to regime name.

        Args:
            score: Regime score in [-1, 1]

        Returns:
            Regime name
        """
        if score > 0.5:
            return MarketRegime.BULL_TREND.value
        elif score < -0.5:
            return MarketRegime.BEAR_MARKET.value
        elif abs(score) < 0.2:
            return MarketRegime.LOW_VOL.value
        else:
            return MarketRegime.NEUTRAL.value

    def compute_multi_period_score(
        self,
        prices: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, int], pd.Series],
        regime: str,
        normalize: bool = True,
    ) -> AdaptiveLookbackResult:
        """
        Compute weighted average score across multiple lookback periods.

        Args:
            prices: DataFrame with price data
            signal_func: Function that takes (prices, lookback) and returns pd.Series
            regime: Market regime name
            normalize: Whether to normalize final score to [-1, 1]

        Returns:
            AdaptiveLookbackResult with multi-period score
        """
        config = self._get_config(regime)
        periods = self.get_lookback_periods(regime)
        decay_weights = config.decay_weights

        # Calculate scores for each period category
        category_scores: Dict[str, pd.Series] = {}
        component_scores: Dict[int, pd.Series] = {}

        for category, period_list in periods.items():
            category_weight = decay_weights[["short", "medium", "long"].index(category)]

            # Calculate scores for each period in category
            period_scores = []
            for period in period_list:
                try:
                    score = signal_func(prices, period)
                    if isinstance(score, pd.Series):
                        period_scores.append(score)
                        component_scores[period] = score
                except Exception as e:
                    logger.warning(f"Signal computation failed for period {period}: {e}")
                    continue

            if period_scores:
                # Average within category
                category_avg = pd.concat(period_scores, axis=1).mean(axis=1)
                category_scores[category] = category_avg * category_weight

        if not category_scores:
            logger.warning("No valid scores computed")
            return AdaptiveLookbackResult(
                regime=regime,
                lookback_periods=periods,
                decay_weights=decay_weights,
                multi_period_score=None,
                component_scores=None,
                metadata={"error": "No valid scores computed"},
            )

        # Combine category scores
        combined = pd.concat(list(category_scores.values()), axis=1).sum(axis=1)

        # Normalize if requested
        if normalize:
            combined = np.tanh(combined)

        return AdaptiveLookbackResult(
            regime=regime,
            lookback_periods=periods,
            decay_weights=decay_weights,
            multi_period_score=combined,
            component_scores=component_scores,
            metadata={
                "categories_computed": list(category_scores.keys()),
                "total_periods": len(component_scores),
                "normalized": normalize,
            },
        )

    def compute_adaptive_sharpe(
        self,
        returns: pd.Series,
        regime: str,
        annualization: int = 252,
    ) -> Dict[str, float]:
        """
        Compute Sharpe ratios across multiple lookback periods.

        Args:
            returns: Daily returns series
            regime: Market regime name
            annualization: Annualization factor (252 for daily)

        Returns:
            Dictionary with period -> Sharpe ratio
        """
        periods = self.get_all_periods(regime)
        sharpe_ratios: Dict[str, float] = {}

        for period in periods:
            if len(returns) < period:
                continue

            rolling_returns = returns.tail(period)
            mean_return = rolling_returns.mean()
            std_return = rolling_returns.std()

            if std_return > 0:
                sharpe = (mean_return * annualization) / (std_return * np.sqrt(annualization))
            else:
                sharpe = 0.0

            sharpe_ratios[f"sharpe_{period}d"] = sharpe

        return sharpe_ratios

    def compute_weighted_sharpe(
        self,
        returns: pd.Series,
        regime: str,
        annualization: int = 252,
    ) -> float:
        """
        Compute regime-weighted average Sharpe ratio.

        Args:
            returns: Daily returns series
            regime: Market regime name
            annualization: Annualization factor

        Returns:
            Weighted average Sharpe ratio
        """
        config = self._get_config(regime)
        periods = self.get_lookback_periods(regime)
        decay_weights = config.decay_weights

        weighted_sharpe = 0.0
        total_weight = 0.0

        for i, (category, period_list) in enumerate(periods.items()):
            category_weight = decay_weights[i]

            category_sharpes = []
            for period in period_list:
                if len(returns) < period:
                    continue

                rolling_returns = returns.tail(period)
                mean_ret = rolling_returns.mean()
                std_ret = rolling_returns.std()

                if std_ret > 0:
                    sharpe = (mean_ret * annualization) / (std_ret * np.sqrt(annualization))
                    category_sharpes.append(sharpe)

            if category_sharpes:
                avg_category_sharpe = np.mean(category_sharpes)
                weighted_sharpe += avg_category_sharpe * category_weight
                total_weight += category_weight

        if total_weight > 0:
            return weighted_sharpe / total_weight
        return 0.0

    def _get_config(self, regime: str) -> LookbackConfig:
        """
        Get configuration for a regime.

        Args:
            regime: Regime name

        Returns:
            LookbackConfig for the regime
        """
        if regime in self._configs:
            return self._configs[regime]

        logger.warning(f"Unknown regime '{regime}', using neutral")
        return self._configs.get(MarketRegime.NEUTRAL.value, LookbackConfig())

    def set_regime_config(self, regime: str, config: LookbackConfig) -> None:
        """
        Set or update configuration for a regime.

        Args:
            regime: Regime name
            config: LookbackConfig to set
        """
        self._configs[regime] = config
        logger.info(f"Updated config for regime '{regime}'")

    def get_regime_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all regime configurations.

        Returns:
            Dictionary with regime -> config summary
        """
        summary = {}
        for regime, config in self._configs.items():
            summary[regime] = {
                "short_periods": config.short,
                "medium_periods": config.medium,
                "long_periods": config.long,
                "decay_weights": config.decay_weights,
                "total_periods": len(config.get_all_periods()),
                "shortest": min(config.get_all_periods()),
                "longest": max(config.get_all_periods()),
            }
        return summary
