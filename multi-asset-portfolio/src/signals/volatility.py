"""
Volatility Signals Module.

Provides volatility-based signals including:
- ATR (Average True Range)
- Volatility Breakout
- Volatility Regime Detection

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
    "atr",
    category="volatility",
    description="ATR-based volatility signal normalized to [-1, +1]",
    tags=["volatility", "atr", "range"],
)
class ATRSignal(Signal):
    """
    Average True Range (ATR) Signal.

    Measures market volatility by decomposing the entire range of an asset
    price for a given period. High ATR indicates high volatility.

    Output interpretation:
    - Positive values: Higher than average volatility
    - Negative values: Lower than average volatility
    - Values near 0: Average volatility
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """ATR: multi-timeframe volatility measure (5-60 days).

        ATR works across short to medium timeframes. Very long periods
        smooth out volatility too much to be useful for timing.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MULTI_TIMEFRAME,
            min_period=5,
            max_period=60,
            supported_variants=["short", "medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=14,
                searchable=True,
                min_value=5,
                max_value=60,  # Extended from 50 to 60 for long variant
                step=1,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="normalization_lookback",
                default=100,
                searchable=False,
                min_value=20,
                max_value=500,
                description="Lookback period for z-score normalization",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute ATR signal.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            SignalResult with ATR-based volatility scores
        """
        self.validate_input(data)
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        period = self._params["period"]
        norm_lookback = self._params["normalization_lookback"]

        # Calculate True Range
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR (EMA of True Range)
        atr = true_range.ewm(span=period, adjust=False).mean()

        # Normalize using z-score and tanh
        scores = self.normalize_zscore_tanh(atr, lookback=norm_lookback, scale=0.5)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "atr",
                "period": period,
                "normalization_lookback": norm_lookback,
            },
        )


@SignalRegistry.register(
    "volatility_breakout",
    category="volatility",
    description="Volatility breakout signal based on ATR bands",
    tags=["volatility", "breakout", "trend"],
)
class VolatilityBreakoutSignal(Signal):
    """
    Volatility Breakout Signal.

    Detects price breakouts from volatility-adjusted bands.
    Uses ATR to define dynamic bands around a moving average.

    Output interpretation:
    - +1: Strong upward breakout (price above upper band)
    - -1: Strong downward breakout (price below lower band)
    - 0: Price within normal volatility range
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Volatility Breakout: medium-term (10-60 days for MA).

        Breakout detection needs enough history for meaningful bands
        but not so long that it loses responsiveness.
        Note: atr_period spec is [5, 30], ma_period spec is [10, 60]
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=10,
            max_period=30,
            # medium(20) is within atr_period and ma_period overlapping range
            supported_variants=["medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="atr_period",
                default=14,
                searchable=True,
                min_value=5,
                max_value=30,
                step=1,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="ma_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=60,  # Extended from 50 to 60 for long variant
                step=5,
                description="Moving average period for center line",
            ),
            ParameterSpec(
                name="multiplier",
                default=2.0,
                searchable=True,
                min_value=1.0,
                max_value=4.0,
                step=0.5,
                description="ATR multiplier for band width",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute volatility breakout signal.

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

        atr_period = self._params["atr_period"]
        ma_period = self._params["ma_period"]
        multiplier = self._params["multiplier"]

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

        # Calculate bands
        ma = close.rolling(window=ma_period, min_periods=1).mean()
        upper_band = ma + multiplier * atr
        lower_band = ma - multiplier * atr

        # Calculate position relative to bands
        band_width = upper_band - lower_band
        # Avoid division by zero
        band_width = band_width.replace(0, np.nan).ffill().fillna(1)

        # Position: -1 (at lower band) to +1 (at upper band)
        position = 2 * (close - lower_band) / band_width - 1

        # Apply tanh to smooth extreme values
        scores = self.normalize_tanh(position, scale=1.5)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "volatility_breakout",
                "atr_period": atr_period,
                "ma_period": ma_period,
                "multiplier": multiplier,
            },
        )


@SignalRegistry.register(
    "volatility_regime",
    category="volatility",
    description="Detects high/low volatility regime changes",
    tags=["volatility", "regime", "state"],
)
class VolatilityRegimeSignal(Signal):
    """
    Volatility Regime Signal.

    Identifies market volatility regimes (high/low volatility states)
    using rolling volatility percentile ranking.

    Output interpretation:
    - +1: High volatility regime
    - -1: Low volatility regime
    - Values around 0: Normal volatility regime
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Volatility Regime: multi-timeframe (captures annual cycles).

        Regime detection benefits from longer lookbacks to capture
        full volatility cycles including annual patterns.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MULTI_TIMEFRAME,
            min_period=5,
            max_period=252,
            supported_variants=["short", "medium", "long", "half_year", "yearly"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="short_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=60,  # Extended for medium-long variants
                step=5,
                description="Short-term volatility measurement period",
            ),
            ParameterSpec(
                name="long_period",
                default=60,
                searchable=True,
                min_value=30,
                max_value=252,  # Extended to yearly for regime detection
                step=10,
                description="Long-term volatility baseline period",
            ),
            ParameterSpec(
                name="percentile_lookback",
                default=252,
                searchable=False,
                min_value=100,
                max_value=504,
                description="Lookback period for percentile ranking",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute volatility regime signal.

        Args:
            data: DataFrame with 'close' column

        Returns:
            SignalResult with regime scores
        """
        self.validate_input(data)

        short_period = self._params["short_period"]
        long_period = self._params["long_period"]
        percentile_lookback = self._params["percentile_lookback"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate short-term and long-term volatility
        short_vol = returns.rolling(window=short_period, min_periods=1).std()
        long_vol = returns.rolling(window=long_period, min_periods=1).std()

        # Volatility ratio (short/long)
        vol_ratio = short_vol / long_vol.replace(0, np.nan).ffill().fillna(1)

        # Calculate percentile rank of current volatility
        def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
            """Calculate rolling percentile rank."""
            result = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                start = max(0, i - window + 1)
                window_data = series.iloc[start : i + 1]
                if len(window_data) > 1:
                    rank = (window_data < series.iloc[i]).sum()
                    result.iloc[i] = rank / (len(window_data) - 1)
                else:
                    result.iloc[i] = 0.5
            return result

        vol_percentile = rolling_percentile(short_vol, percentile_lookback)

        # Combine ratio and percentile for final score
        # Map percentile [0, 1] to [-1, +1]
        scores = 2 * vol_percentile - 1

        # Apply smoothing with vol_ratio influence
        vol_ratio_normalized = self.normalize_zscore_tanh(vol_ratio, lookback=long_period)
        scores = 0.7 * scores + 0.3 * vol_ratio_normalized

        # Ensure bounds
        scores = scores.clip(-1, 1)

        return SignalResult(
            scores=scores,
            metadata={
                "signal_type": "volatility_regime",
                "short_period": short_period,
                "long_period": long_period,
                "percentile_lookback": percentile_lookback,
            },
        )
