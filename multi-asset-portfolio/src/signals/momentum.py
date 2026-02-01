"""
Momentum Signals - Trend-following indicators.

Implements momentum-based signals including:
- N-day Return: Simple price momentum over N days
- ROC (Rate of Change): Percentage change over N periods
- Momentum Score: Weighted combination of multiple timeframes

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult, TimeframeAffinity, TimeframeConfig
from .registry import SignalRegistry


@SignalRegistry.register(
    "momentum_return",
    category="momentum",
    description="N-day price return momentum signal",
    tags=["trend", "simple"],
)
class MomentumReturnSignal(Signal):
    """
    N-day Return Momentum Signal.

    Computes the price return over N days and normalizes to [-1, +1].
    Positive returns generate positive scores (bullish).
    Negative returns generate negative scores (bearish).

    Parameters:
        lookback: Number of days for return calculation (searchable, 5-252)
        scale: Tanh scaling factor (fixed, default 5.0)

    Formula:
        raw_return = (close[t] - close[t-lookback]) / close[t-lookback]
        score = tanh(raw_return * scale)

    Academic reference:
        Jegadeesh & Titman (1993): Momentum effective from 1 to 12 months.
        12-month momentum (~252 days) shows strongest persistence.

    Example:
        signal = MomentumReturnSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Momentum: multi-timeframe (1-12 months per Jegadeesh & Titman 1993)."""
        return TimeframeConfig(
            affinity=TimeframeAffinity.MULTI_TIMEFRAME,
            min_period=5,
            max_period=252,  # Extended to support yearly variant
            supported_variants=["short", "medium", "long", "half_year", "yearly"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=252,  # Extended from 120 to 252 for yearly momentum
                step=5,
                description="Lookback period for return calculation (days)",
            ),
            ParameterSpec(
                name="scale",
                default=5.0,
                searchable=False,
                min_value=0.1,
                max_value=20.0,
                description="Tanh scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get expanded parameter grid for momentum return signal.

        Returns 9 lookback periods covering short to long-term momentum:
        - Short-term: 5, 10, 15, 20 days
        - Medium-term: 30, 40, 60 days
        - Long-term: 90, 120 days
        """
        return {
            "lookback": [5, 10, 15, 20, 30, 40, 60, 90, 120],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate N-day return
        returns = close.pct_change(periods=lookback)

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(returns, scale=scale)

        metadata = {
            "lookback": lookback,
            "scale": scale,
            "raw_return_mean": returns.mean(),
            "raw_return_std": returns.std(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "roc",
    category="momentum",
    description="Rate of Change momentum signal",
    tags=["trend", "oscillator"],
)
class ROCSignal(Signal):
    """
    Rate of Change (ROC) Signal.

    Measures the percentage change in price over N periods.
    Similar to momentum return but expressed as a percentage.

    Parameters:
        period: Number of periods for ROC calculation (searchable, 5-60)
        smooth_period: Optional smoothing MA period (fixed, default 1 = no smoothing)
        scale: Tanh scaling factor (fixed, default 0.5)

    Formula:
        ROC = ((close[t] - close[t-period]) / close[t-period]) * 100
        smoothed_roc = SMA(ROC, smooth_period)
        score = tanh(smoothed_roc * scale)

    Example:
        signal = ROCSignal(period=12)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """ROC: short-to-medium term momentum (5-60 days).

        ROC is typically used as a short-term momentum oscillator.
        Unlike longer-term momentum, very long periods reduce sensitivity.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=5,
            max_period=60,
            # half_year(126) and yearly(252) too long for ROC
            supported_variants=["short", "medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=12,
                searchable=True,
                min_value=5,
                max_value=60,  # Extended from 50 to 60 for long variant
                step=1,
                description="ROC calculation period",
            ),
            ParameterSpec(
                name="smooth_period",
                default=1,
                searchable=False,
                min_value=1,
                max_value=10,
                description="Smoothing MA period (1 = no smoothing)",
            ),
            ParameterSpec(
                name="scale",
                default=0.05,
                searchable=False,
                min_value=0.01,
                max_value=1.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        period = self._params["period"]
        smooth_period = self._params["smooth_period"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate Rate of Change (percentage)
        roc = ((close - close.shift(period)) / close.shift(period)) * 100

        # Apply smoothing if specified
        if smooth_period > 1:
            roc = roc.rolling(window=smooth_period, min_periods=1).mean()

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(roc, scale=scale)

        metadata = {
            "period": period,
            "smooth_period": smooth_period,
            "scale": scale,
            "roc_mean": roc.mean(),
            "roc_std": roc.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "momentum_composite",
    category="momentum",
    description="Composite momentum signal combining multiple timeframes",
    tags=["trend", "multi-timeframe"],
)
class MomentumCompositeSignal(Signal):
    """
    Composite Momentum Signal.

    Combines momentum signals across multiple timeframes for a more
    robust trend indication. Uses weighted average of short, medium,
    and long-term momentum.

    Parameters:
        short_period: Short-term lookback (searchable, 5-20)
        medium_period: Medium-term lookback (searchable, 20-60)
        long_period: Long-term lookback (searchable, 60-120)
        short_weight: Weight for short-term momentum (fixed, default 0.25)
        medium_weight: Weight for medium-term momentum (fixed, default 0.35)
        long_weight: Weight for long-term momentum (fixed, default 0.40)

    Formula:
        score = w_short * mom_short + w_medium * mom_medium + w_long * mom_long

    Note:
        This signal has multiple period parameters (short_period, medium_period,
        long_period) and is NOT subject to period variant generation. It is
        computed once with its default/configured parameters.

    Example:
        signal = MomentumCompositeSignal(
            short_period=10, medium_period=30, long_period=60
        )
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Composite: internally handles multiple timeframes.

        This signal has its own short/medium/long period parameters and
        doesn't use standard period variants. The long_period spec is [60, 120].
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MULTI_TIMEFRAME,
            min_period=5,
            max_period=120,
            # short(5), medium(20), long(60) are within respective spec ranges
            supported_variants=["short", "medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="short_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=20,
                step=5,
                description="Short-term momentum period",
            ),
            ParameterSpec(
                name="medium_period",
                default=30,
                searchable=True,
                min_value=20,
                max_value=60,
                step=10,
                description="Medium-term momentum period",
            ),
            ParameterSpec(
                name="long_period",
                default=60,
                searchable=True,
                min_value=60,
                max_value=120,
                step=10,
                description="Long-term momentum period",
            ),
            ParameterSpec(
                name="short_weight",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for short-term momentum",
            ),
            ParameterSpec(
                name="medium_weight",
                default=0.35,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for medium-term momentum",
            ),
            ParameterSpec(
                name="long_weight",
                default=0.40,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for long-term momentum",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        short_period = self._params["short_period"]
        medium_period = self._params["medium_period"]
        long_period = self._params["long_period"]
        short_weight = self._params["short_weight"]
        medium_weight = self._params["medium_weight"]
        long_weight = self._params["long_weight"]

        close = data["close"]

        # Calculate momentum for each timeframe
        mom_short = close.pct_change(periods=short_period)
        mom_medium = close.pct_change(periods=medium_period)
        mom_long = close.pct_change(periods=long_period)

        # Normalize each component
        mom_short_norm = self.normalize_tanh(mom_short, scale=10.0)
        mom_medium_norm = self.normalize_tanh(mom_medium, scale=5.0)
        mom_long_norm = self.normalize_tanh(mom_long, scale=3.0)

        # Weighted combination
        total_weight = short_weight + medium_weight + long_weight
        scores = (
            short_weight * mom_short_norm
            + medium_weight * mom_medium_norm
            + long_weight * mom_long_norm
        ) / total_weight

        # Ensure output is in [-1, +1]
        scores = scores.clip(-1, 1)

        metadata = {
            "short_period": short_period,
            "medium_period": medium_period,
            "long_period": long_period,
            "weights": {
                "short": short_weight,
                "medium": medium_weight,
                "long": long_weight,
            },
            "component_means": {
                "short": mom_short_norm.mean(),
                "medium": mom_medium_norm.mean(),
                "long": mom_long_norm.mean(),
            },
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "momentum_acceleration",
    category="momentum",
    description="Momentum acceleration (second derivative of price)",
    tags=["trend", "acceleration"],
)
class MomentumAccelerationSignal(Signal):
    """
    Momentum Acceleration Signal.

    Measures the rate of change of momentum (second derivative).
    Useful for detecting momentum shifts before price reversals.

    Parameters:
        momentum_period: Period for first momentum calculation (searchable, 5-30)
        acceleration_period: Period for acceleration calculation (searchable, 3-15)
        scale: Tanh scaling factor (fixed, default 50.0)

    Formula:
        momentum = close[t] - close[t-momentum_period]
        acceleration = momentum[t] - momentum[t-acceleration_period]
        score = tanh(acceleration * scale)

    Note:
        This signal has momentum_period and acceleration_period parameters.
        Since neither matches standard period param names (period, lookback),
        it is computed once without period variant generation.

    Example:
        signal = MomentumAccelerationSignal(
            momentum_period=10, acceleration_period=5
        )
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Acceleration: short-term signal (5-30 day momentum base)."""
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=5,
            max_period=30,
            # Acceleration needs fast response, long periods lose sensitivity
            supported_variants=["short", "medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="momentum_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=30,
                step=5,
                description="Period for momentum calculation",
            ),
            ParameterSpec(
                name="acceleration_period",
                default=5,
                searchable=True,
                min_value=3,
                max_value=15,
                step=1,
                description="Period for acceleration calculation",
            ),
            ParameterSpec(
                name="scale",
                default=50.0,
                searchable=False,
                min_value=1.0,
                max_value=200.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        momentum_period = self._params["momentum_period"]
        acceleration_period = self._params["acceleration_period"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate first momentum (price change)
        momentum = close.diff(periods=momentum_period)

        # Calculate acceleration (momentum change)
        acceleration = momentum.diff(periods=acceleration_period)

        # Normalize by price level to make it comparable across assets
        normalized_acceleration = acceleration / close

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(normalized_acceleration, scale=scale)

        metadata = {
            "momentum_period": momentum_period,
            "acceleration_period": acceleration_period,
            "scale": scale,
            "acceleration_mean": normalized_acceleration.mean(),
            "acceleration_std": normalized_acceleration.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)
