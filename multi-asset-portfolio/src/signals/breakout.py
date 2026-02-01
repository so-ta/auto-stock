"""
Breakout Signals Module.

Provides breakout-based signals including:
- Donchian Channel
- High/Low Breakout
- Range Breakout

All signals output normalized values in [-1, +1] range.
"""

from typing import List

import numpy as np
import pandas as pd

from src.signals.base import (
    ParameterSpec,
    Signal,
    SignalResult,
    TimeframeAffinity,
    TimeframeConfig,
)
from src.signals.registry import SignalRegistry


@SignalRegistry.register(
    "donchian_channel",
    category="breakout",
    description="Donchian Channel breakout signal",
    tags=["breakout", "channel", "trend"],
)
class DonchianChannelSignal(Signal):
    """
    Donchian Channel Signal.

    The Donchian Channel plots the highest high and lowest low over
    a specified period. Breakouts beyond these levels indicate
    potential trend continuation.

    Output interpretation:
    - +1: Price breaking above upper channel (bullish breakout)
    - -1: Price breaking below lower channel (bearish breakout)
    - 0: Price within channel (no breakout)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Donchian Channel: medium-term breakout (10-60 days).

        Turtle trading used 20-day and 55-day channels. Very long
        periods (126+) lose breakout sensitivity.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=10,
            max_period=60,
            # medium(20) and long(60) are within spec range [10, 60]
            supported_variants=["medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=60,
                step=5,
                description="Channel lookback period",
            ),
            ParameterSpec(
                name="entry_threshold",
                default=0.0,
                searchable=False,
                min_value=-0.1,
                max_value=0.1,
                description="Threshold for breakout confirmation (fraction of channel width)",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute Donchian Channel signal.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            SignalResult with channel breakout scores
        """
        self.validate_input(data)
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        period = self._params["period"]
        entry_threshold = self._params["entry_threshold"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate Donchian Channel
        upper_channel = high.rolling(window=period, min_periods=1).max()
        lower_channel = low.rolling(window=period, min_periods=1).min()

        # Channel width for normalization
        channel_width = upper_channel - lower_channel
        channel_width = channel_width.replace(0, np.nan).ffill().fillna(1)

        # Position within channel: 0 = lower, 1 = upper
        position = (close - lower_channel) / channel_width

        # Adjust for entry threshold
        adjusted_threshold = entry_threshold

        # Calculate signal scores
        scores = pd.Series(index=data.index, dtype=float)

        # Breakout above upper channel
        upper_breakout = position > (1 + adjusted_threshold)
        # Breakout below lower channel
        lower_breakout = position < -adjusted_threshold

        # Map to [-1, +1] with gradual scaling
        # Position 0.5 = center = 0 signal
        # Position 1.0 = upper channel = +1
        # Position 0.0 = lower channel = -1
        scores = 2 * position - 1

        # Amplify breakouts
        scores = scores.where(~upper_breakout, 1.0)
        scores = scores.where(~lower_breakout, -1.0)

        # Apply tanh smoothing for values within channel
        scores = self.normalize_tanh(scores, scale=1.0)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "donchian_channel",
                "period": period,
                "entry_threshold": entry_threshold,
            },
        )


@SignalRegistry.register(
    "high_low_breakout",
    category="breakout",
    description="N-day high/low breakout signal",
    tags=["breakout", "momentum", "trend"],
)
class HighLowBreakoutSignal(Signal):
    """
    High/Low Breakout Signal.

    Detects when price makes new N-day highs or lows.
    New highs indicate bullish momentum, new lows indicate bearish.

    Output interpretation:
    - +1: New N-day high (bullish)
    - -1: New N-day low (bearish)
    - Gradual values: Distance from highs/lows
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """High/Low Breakout: medium-term (5-60 days).

        Similar to Donchian, effective for detecting breakouts
        in medium-term timeframes.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=5,
            max_period=60,
            supported_variants=["short", "medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="high_period",
                default=20,
                searchable=True,
                min_value=5,
                max_value=60,
                step=5,
                description="Lookback period for high detection",
            ),
            ParameterSpec(
                name="low_period",
                default=20,
                searchable=True,
                min_value=5,
                max_value=60,
                step=5,
                description="Lookback period for low detection",
            ),
            ParameterSpec(
                name="confirmation_bars",
                default=1,
                searchable=False,
                min_value=1,
                max_value=5,
                description="Bars above/below level to confirm breakout",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute high/low breakout signal.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            SignalResult with breakout scores
        """
        self.validate_input(data)
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        high_period = self._params["high_period"]
        low_period = self._params["low_period"]
        confirmation_bars = self._params["confirmation_bars"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate rolling high and low (excluding current bar)
        rolling_high = high.shift(1).rolling(window=high_period, min_periods=1).max()
        rolling_low = low.shift(1).rolling(window=low_period, min_periods=1).min()

        # Detect breakouts
        high_breakout = high > rolling_high
        low_breakout = low < rolling_low

        # Confirm breakouts with multiple bars
        if confirmation_bars > 1:
            high_confirmed = high_breakout.rolling(
                window=confirmation_bars, min_periods=confirmation_bars
            ).sum() >= confirmation_bars
            low_confirmed = low_breakout.rolling(
                window=confirmation_bars, min_periods=confirmation_bars
            ).sum() >= confirmation_bars
        else:
            high_confirmed = high_breakout
            low_confirmed = low_breakout

        # Calculate position relative to recent range
        range_size = rolling_high - rolling_low
        range_size = range_size.replace(0, np.nan).ffill().fillna(1)

        # Position: -1 (at low) to +1 (at high)
        position = 2 * (close - rolling_low) / range_size - 1

        # Create base scores from position
        scores = position.copy()

        # Override with confirmed breakouts
        scores = scores.where(~high_confirmed, 1.0)
        scores = scores.where(~low_confirmed, -1.0)

        # Clip to ensure bounds
        scores = scores.clip(-1, 1)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "high_low_breakout",
                "high_period": high_period,
                "low_period": low_period,
                "confirmation_bars": confirmation_bars,
            },
        )


@SignalRegistry.register(
    "range_breakout",
    category="breakout",
    description="Range expansion/contraction breakout signal",
    tags=["breakout", "volatility", "range"],
)
class RangeBreakoutSignal(Signal):
    """
    Range Breakout Signal.

    Detects breakouts from price consolidation ranges.
    Uses Average True Range (ATR) to identify consolidation
    and expansion phases.

    Output interpretation:
    - +1: Strong upward range expansion
    - -1: Strong downward range expansion
    - Near 0: Consolidation phase
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Range Breakout: short-to-medium term (5-30 days).

        Range breakouts are most effective for short-term trading.
        Note: consolidation_period spec is [5, 30], atr_period is [7, 28]
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=7,
            max_period=28,
            # medium(20) is within atr_period spec range [7, 28]
            supported_variants=["medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="consolidation_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=30,
                step=5,
                description="Period to measure consolidation range",
            ),
            ParameterSpec(
                name="atr_period",
                default=14,
                searchable=True,
                min_value=7,
                max_value=28,
                step=7,
                description="ATR period for volatility measurement",
            ),
            ParameterSpec(
                name="expansion_multiplier",
                default=1.5,
                searchable=True,
                min_value=1.0,
                max_value=3.0,
                step=0.25,
                description="ATR multiplier to confirm expansion",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute range breakout signal.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            SignalResult with range breakout scores
        """
        self.validate_input(data)
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        consolidation_period = self._params["consolidation_period"]
        atr_period = self._params["atr_period"]
        expansion_multiplier = self._params["expansion_multiplier"]

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # Calculate ATR
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=atr_period, adjust=False).mean()

        # Calculate consolidation range
        range_high = high.rolling(window=consolidation_period, min_periods=1).max()
        range_low = low.rolling(window=consolidation_period, min_periods=1).min()
        range_midpoint = (range_high + range_low) / 2

        # Daily range relative to ATR
        daily_range = high - low
        range_ratio = daily_range / atr.replace(0, np.nan).ffill().fillna(1)

        # Identify expansion (range > multiplier * ATR)
        is_expansion = range_ratio > expansion_multiplier

        # Direction of expansion
        close_position = close - range_midpoint
        direction = np.sign(close_position)

        # Calculate base score from position relative to range
        range_size = range_high - range_low
        range_size = range_size.replace(0, np.nan).ffill().fillna(1)

        position = (close - range_low) / range_size
        base_score = 2 * position - 1

        # Amplify during expansion
        expansion_factor = (range_ratio / expansion_multiplier).clip(0, 2)
        scores = base_score * expansion_factor.where(is_expansion, 1.0)

        # Apply tanh to smooth
        scores = self.normalize_tanh(scores, scale=0.8)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "range_breakout",
                "consolidation_period": consolidation_period,
                "atr_period": atr_period,
                "expansion_multiplier": expansion_multiplier,
            },
        )
