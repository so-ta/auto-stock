"""
Advanced Technical Signals - Adaptive and robust technical indicators.

Implements advanced technical signals including:
- KAMA: Kaufman Adaptive Moving Average
- Keltner Channel: ATR-based channel breakout

These signals are designed to be more adaptive to market conditions
and more robust to whipsaws than traditional technical indicators.

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult, TimeframeAffinity, TimeframeConfig
from .registry import SignalRegistry


@SignalRegistry.register(
    "kama",
    category="advanced_technical",
    description="Kaufman Adaptive Moving Average signal",
    tags=["trend", "adaptive", "momentum"],
)
class KAMASignal(Signal):
    """
    Kaufman Adaptive Moving Average (KAMA) Signal.

    KAMA adapts its smoothing based on market noise (efficiency ratio).
    Fast in trending markets, slow in ranging markets.

    Parameters:
        efficiency_period: Period for efficiency ratio (searchable, 5-20)
        fast_period: Fast EMA period (fixed, default 2)
        slow_period: Slow EMA period (fixed, default 30)
        signal_type: Signal generation method (fixed, default 'trend')
        scale: Tanh scaling factor (fixed, default 5.0)

    Formula:
        ER = |change| / sum(|changes|)  (Efficiency Ratio)
        SC = (ER * (fast_alpha - slow_alpha) + slow_alpha)^2  (Smoothing Constant)
        KAMA = KAMA[-1] + SC * (price - KAMA[-1])
        score = tanh((price - KAMA) / ATR * scale)  (trend mode)

    Example:
        signal = KAMASignal(efficiency_period=10)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """KAMA: short-to-medium term adaptive trend (5-20 days efficiency period).

        KAMA's adaptive nature works best with shorter efficiency periods
        that can detect regime changes quickly.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=5,
            max_period=20,
            supported_variants=["short", "medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="efficiency_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=20,
                step=5,
                description="Efficiency ratio calculation period",
            ),
            ParameterSpec(
                name="fast_period",
                default=2,
                searchable=False,
                min_value=2,
                max_value=5,
                description="Fast EMA period for KAMA",
            ),
            ParameterSpec(
                name="slow_period",
                default=30,
                searchable=False,
                min_value=20,
                max_value=50,
                description="Slow EMA period for KAMA",
            ),
            ParameterSpec(
                name="scale",
                default=5.0,
                searchable=False,
                min_value=1.0,
                max_value=20.0,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "efficiency_period": [5, 10, 15, 20],
        }

    def _calculate_kama(
        self,
        close: pd.Series,
        efficiency_period: int,
        fast_period: int,
        slow_period: int,
    ) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average."""
        # Calculate change and volatility
        change = (close - close.shift(efficiency_period)).abs()
        volatility = close.diff().abs().rolling(window=efficiency_period).sum()

        # Efficiency Ratio
        volatility = volatility.replace(0, np.nan).ffill().fillna(1)
        er = change / volatility
        er = er.clip(0, 1)  # Bound between 0 and 1

        # Smoothing constants
        fast_alpha = 2.0 / (fast_period + 1)
        slow_alpha = 2.0 / (slow_period + 1)

        # Smoothing coefficient
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # Calculate KAMA iteratively
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[0] = close.iloc[0]

        for i in range(1, len(close)):
            if pd.isna(sc.iloc[i]):
                kama.iloc[i] = kama.iloc[i - 1]
            else:
                kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
                    close.iloc[i] - kama.iloc[i - 1]
                )

        return kama, er

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        efficiency_period = self._params["efficiency_period"]
        fast_period = self._params["fast_period"]
        slow_period = self._params["slow_period"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate KAMA
        kama, efficiency_ratio = self._calculate_kama(
            close, efficiency_period, fast_period, slow_period
        )

        # Calculate ATR for normalization (use close-based approximation if OHLC not available)
        if "high" in data.columns and "low" in data.columns:
            high = data["high"]
            low = data["low"]
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
        else:
            # Approximate using absolute returns
            tr = close.diff().abs()

        atr = tr.rolling(window=efficiency_period, min_periods=1).mean()
        atr = atr.replace(0, np.nan).ffill().fillna(1)

        # Calculate price deviation from KAMA normalized by ATR
        deviation = (close - kama) / atr

        # Generate trend-following signal
        # Price above KAMA = bullish, Price below KAMA = bearish
        scores = self.normalize_tanh(deviation, scale=scale)

        # Calculate statistics
        above_kama_pct = (close > kama).mean() * 100
        avg_er = efficiency_ratio.mean()

        metadata = {
            "efficiency_period": efficiency_period,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "scale": scale,
            "avg_efficiency_ratio": avg_er,
            "above_kama_pct": above_kama_pct,
            "deviation_mean": deviation.mean(),
            "deviation_std": deviation.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "keltner_channel",
    category="advanced_technical",
    description="Keltner Channel breakout signal",
    tags=["volatility", "breakout", "trend"],
)
class KeltnerChannelSignal(Signal):
    """
    Keltner Channel Signal.

    Keltner Channels use ATR for band width, making them more robust
    to outliers than Bollinger Bands (which use standard deviation).

    Parameters:
        ema_period: EMA period for middle band (searchable, 10-30)
        atr_period: ATR calculation period (fixed, default 10)
        atr_multiplier: ATR multiplier for band width (searchable, 1.0-3.0)
        signal_mode: 'reversion' or 'breakout' (fixed, default 'reversion')
        scale: Tanh scaling factor (fixed, default 1.0)

    Formula:
        Middle = EMA(close, ema_period)
        ATR = Average True Range(atr_period)
        Upper = Middle + atr_multiplier * ATR
        Lower = Middle - atr_multiplier * ATR
        position = (close - Middle) / (atr_multiplier * ATR)
        score = tanh(-position * scale)  (reversion mode)

    Example:
        signal = KeltnerChannelSignal(ema_period=20, atr_multiplier=2.0)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Keltner Channel: medium-term (10-30 day EMA).

        Similar to Bollinger Bands, effective for medium-term trading.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=10,
            max_period=30,
            supported_variants=["medium"],  # Only medium(20) is within range
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="ema_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=30,
                step=5,
                description="EMA period for middle band",
            ),
            ParameterSpec(
                name="atr_period",
                default=10,
                searchable=False,
                min_value=5,
                max_value=20,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="atr_multiplier",
                default=2.0,
                searchable=True,
                min_value=1.0,
                max_value=3.0,
                step=0.5,
                description="ATR multiplier for band width",
            ),
            ParameterSpec(
                name="signal_mode",
                default="reversion",
                searchable=False,
                description="Signal mode: 'reversion' or 'breakout'",
            ),
            ParameterSpec(
                name="scale",
                default=1.0,
                searchable=False,
                min_value=0.5,
                max_value=3.0,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "ema_period": [10, 15, 20, 25, 30],
            "atr_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0],
        }

    def _calculate_atr(
        self, close: pd.Series, high: pd.Series, low: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        ema_period = self._params["ema_period"]
        atr_period = self._params["atr_period"]
        atr_multiplier = self._params["atr_multiplier"]
        signal_mode = self._params["signal_mode"]
        scale = self._params["scale"]

        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)

        # Calculate middle band (EMA)
        middle_band = close.ewm(span=ema_period, min_periods=1, adjust=False).mean()

        # Calculate ATR
        atr = self._calculate_atr(close, high, low, atr_period)
        atr = atr.replace(0, np.nan).ffill().fillna(1)

        # Calculate channels
        upper_band = middle_band + atr_multiplier * atr
        lower_band = middle_band - atr_multiplier * atr

        # Calculate position within channels
        band_width = upper_band - lower_band
        band_width = band_width.replace(0, np.nan).ffill().fillna(1)

        # Position: -1 at lower band, 0 at middle, +1 at upper band
        position = (close - middle_band) / (band_width / 2)

        if signal_mode == "reversion":
            # Mean reversion: buy oversold, sell overbought
            raw_signal = -position
        else:
            # Breakout: buy breakout up, sell breakout down
            raw_signal = position

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate statistics
        above_upper_pct = (close > upper_band).mean() * 100
        below_lower_pct = (close < lower_band).mean() * 100
        within_channel_pct = 100 - above_upper_pct - below_lower_pct

        metadata = {
            "ema_period": ema_period,
            "atr_period": atr_period,
            "atr_multiplier": atr_multiplier,
            "signal_mode": signal_mode,
            "scale": scale,
            "avg_band_width": band_width.mean(),
            "above_upper_pct": above_upper_pct,
            "below_lower_pct": below_lower_pct,
            "within_channel_pct": within_channel_pct,
            "position_mean": position.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)
