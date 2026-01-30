"""
Ensemble Signals Module - Composite and Multi-Indicator Signals.

Provides advanced signals that combine multiple indicators:
- MomentumEnsemble: Multi-timeframe momentum integration
- MeanReversionEnsemble: Multi-indicator mean reversion
- TrendStrength: Trend intensity measurement
- RegimeDetector: Market regime classification

All outputs are normalized to [-1, +1] range.
"""

from typing import List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "momentum_ensemble",
    category="composite",
    description="Multi-timeframe momentum ensemble signal",
    tags=["momentum", "ensemble", "multi-timeframe"],
)
class MomentumEnsemble(Signal):
    """
    Multi-Timeframe Momentum Ensemble Signal.

    Integrates momentum signals from multiple timeframes (5, 10, 20, 60 days)
    using weighted average. Evaluates consistency across short-term and
    long-term trends.

    Output interpretation:
    - Strong positive: All timeframes agree on uptrend
    - Strong negative: All timeframes agree on downtrend
    - Near zero: Mixed signals or ranging market

    Parameters:
        weight_5d: Weight for 5-day momentum (fixed)
        weight_10d: Weight for 10-day momentum (fixed)
        weight_20d: Weight for 20-day momentum (fixed)
        weight_60d: Weight for 60-day momentum (fixed)
        scale: Tanh scaling factor (fixed)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="weight_5d",
                default=0.15,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for 5-day momentum",
            ),
            ParameterSpec(
                name="weight_10d",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for 10-day momentum",
            ),
            ParameterSpec(
                name="weight_20d",
                default=0.35,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for 20-day momentum",
            ),
            ParameterSpec(
                name="weight_60d",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for 60-day momentum",
            ),
            ParameterSpec(
                name="scale",
                default=3.0,
                searchable=False,
                min_value=0.5,
                max_value=10.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        w5 = self._params["weight_5d"]
        w10 = self._params["weight_10d"]
        w20 = self._params["weight_20d"]
        w60 = self._params["weight_60d"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate momentum for each timeframe
        mom_5d = close.pct_change(periods=5)
        mom_10d = close.pct_change(periods=10)
        mom_20d = close.pct_change(periods=20)
        mom_60d = close.pct_change(periods=60)

        # Weighted average of momentums
        weighted_mom = (
            w5 * mom_5d +
            w10 * mom_10d +
            w20 * mom_20d +
            w60 * mom_60d
        )

        # Calculate trend consistency (how many timeframes agree)
        signs = pd.DataFrame({
            "5d": np.sign(mom_5d),
            "10d": np.sign(mom_10d),
            "20d": np.sign(mom_20d),
            "60d": np.sign(mom_60d),
        })
        consistency = signs.mean(axis=1).abs()  # 1.0 = all agree, 0.0 = mixed

        # Adjust signal by consistency
        adjusted_signal = weighted_mom * (0.5 + 0.5 * consistency)

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(adjusted_signal, scale=scale)

        metadata = {
            "weights": {"5d": w5, "10d": w10, "20d": w20, "60d": w60},
            "scale": scale,
            "mom_5d_mean": mom_5d.mean(),
            "mom_60d_mean": mom_60d.mean(),
            "consistency_mean": consistency.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "mean_reversion_ensemble",
    category="composite",
    description="Multi-indicator mean reversion ensemble signal",
    tags=["mean_reversion", "ensemble", "oscillator"],
)
class MeanReversionEnsemble(Signal):
    """
    Multi-Indicator Mean Reversion Ensemble Signal.

    Combines multiple mean reversion indicators:
    - Bollinger Band position (%B)
    - RSI (Relative Strength Index)
    - Z-Score of price

    Output interpretation:
    - Strong positive: Multiple indicators show oversold (buy signal)
    - Strong negative: Multiple indicators show overbought (sell signal)
    - Near zero: No clear extreme

    Parameters:
        bb_period: Bollinger Band period
        bb_std: Bollinger Band standard deviations
        rsi_period: RSI calculation period
        zscore_period: Z-Score lookback period
        weight_bb: Weight for Bollinger signal
        weight_rsi: Weight for RSI signal
        weight_zscore: Weight for Z-Score signal
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="bb_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=50,
                step=5,
                description="Bollinger Band period",
            ),
            ParameterSpec(
                name="bb_std",
                default=2.0,
                searchable=False,
                min_value=1.5,
                max_value=3.0,
                description="Bollinger Band standard deviations",
            ),
            ParameterSpec(
                name="rsi_period",
                default=14,
                searchable=True,
                min_value=7,
                max_value=28,
                step=7,
                description="RSI calculation period",
            ),
            ParameterSpec(
                name="zscore_period",
                default=20,
                searchable=False,
                min_value=10,
                max_value=50,
                description="Z-Score lookback period",
            ),
            ParameterSpec(
                name="weight_bb",
                default=0.4,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for Bollinger signal",
            ),
            ParameterSpec(
                name="weight_rsi",
                default=0.35,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for RSI signal",
            ),
            ParameterSpec(
                name="weight_zscore",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for Z-Score signal",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        bb_period = self._params["bb_period"]
        bb_std = self._params["bb_std"]
        rsi_period = self._params["rsi_period"]
        zscore_period = self._params["zscore_period"]
        w_bb = self._params["weight_bb"]
        w_rsi = self._params["weight_rsi"]
        w_zscore = self._params["weight_zscore"]

        close = data["close"]

        # 1. Bollinger Band signal (inverse %B for mean reversion)
        bb_ma = close.rolling(window=bb_period, min_periods=1).mean()
        bb_rolling_std = close.rolling(window=bb_period, min_periods=1).std()
        bb_rolling_std = bb_rolling_std.replace(0, np.nan).ffill().fillna(1)

        upper_band = bb_ma + bb_std * bb_rolling_std
        lower_band = bb_ma - bb_std * bb_rolling_std
        band_width = upper_band - lower_band
        band_width = band_width.replace(0, np.nan).ffill().fillna(1)

        pct_b = (close - lower_band) / band_width
        bb_signal = 1 - 2 * pct_b  # Inverted: low %B -> buy (+1)

        # 2. RSI signal
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, np.nan).ffill().fillna(1)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # RSI signal: oversold (< 30) -> +1, overbought (> 70) -> -1
        rsi_signal = (50 - rsi) / 50  # Centered at 50

        # 3. Z-Score signal
        zscore_ma = close.rolling(window=zscore_period, min_periods=1).mean()
        zscore_std = close.rolling(window=zscore_period, min_periods=1).std()
        zscore_std = zscore_std.replace(0, np.nan).ffill().fillna(1)

        zscore = (close - zscore_ma) / zscore_std
        zscore_signal = -zscore  # Inverted for mean reversion

        # Weighted combination
        combined = (
            w_bb * bb_signal +
            w_rsi * rsi_signal +
            w_zscore * zscore_signal
        )

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(combined, scale=1.0)

        metadata = {
            "bb_period": bb_period,
            "rsi_period": rsi_period,
            "zscore_period": zscore_period,
            "weights": {"bb": w_bb, "rsi": w_rsi, "zscore": w_zscore},
            "pct_b_mean": pct_b.mean(),
            "rsi_mean": rsi.mean(),
            "zscore_mean": zscore.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "trend_strength",
    category="composite",
    description="Trend strength measurement signal",
    tags=["trend", "strength", "adx"],
)
class TrendStrength(Signal):
    """
    Trend Strength Signal.

    Measures the intensity of the current trend using:
    - ADX (Average Directional Index) approximation
    - Moving average deviation (price vs MA)
    - Price channel position (Donchian)

    Output interpretation:
    - Strong positive: Strong uptrend
    - Strong negative: Strong downtrend
    - Near zero: Weak trend or ranging market

    Parameters:
        adx_period: ADX calculation period
        ma_period: Moving average period for deviation
        channel_period: Donchian channel period
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="adx_period",
                default=14,
                searchable=True,
                min_value=7,
                max_value=28,
                step=7,
                description="ADX calculation period",
            ),
            ParameterSpec(
                name="ma_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=50,
                step=10,
                description="Moving average period",
            ),
            ParameterSpec(
                name="channel_period",
                default=20,
                searchable=False,
                min_value=10,
                max_value=50,
                description="Donchian channel period",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        adx_period = self._params["adx_period"]
        ma_period = self._params["ma_period"]
        channel_period = self._params["channel_period"]

        close = data["close"]
        high = data["high"] if "high" in data.columns else close
        low = data["low"] if "low" in data.columns else close

        # 1. ADX approximation (simplified)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=adx_period, min_periods=1).mean()

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        plus_di = 100 * plus_dm.rolling(window=adx_period, min_periods=1).mean() / atr.replace(0, np.nan).ffill().fillna(1)
        minus_di = 100 * minus_dm.rolling(window=adx_period, min_periods=1).mean() / atr.replace(0, np.nan).ffill().fillna(1)

        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan).ffill().fillna(1)
        dx = 100 * di_diff / di_sum
        adx = dx.rolling(window=adx_period, min_periods=1).mean()

        # Direction: +DI > -DI -> bullish, else bearish
        adx_direction = np.sign(plus_di - minus_di)
        adx_signal = (adx / 50 - 1) * adx_direction  # ADX normalized, with direction

        # 2. Moving average deviation
        ma = close.rolling(window=ma_period, min_periods=1).mean()
        ma_std = close.rolling(window=ma_period, min_periods=1).std()
        ma_std = ma_std.replace(0, np.nan).ffill().fillna(1)
        ma_deviation = (close - ma) / ma_std

        # 3. Price channel position (Donchian)
        highest = high.rolling(window=channel_period, min_periods=1).max()
        lowest = low.rolling(window=channel_period, min_periods=1).min()
        channel_width = highest - lowest
        channel_width = channel_width.replace(0, np.nan).ffill().fillna(1)
        channel_position = 2 * (close - lowest) / channel_width - 1  # -1 to +1

        # Combine signals
        combined = (
            0.4 * adx_signal.clip(-1, 1) +
            0.35 * self.normalize_tanh(ma_deviation, scale=0.5) +
            0.25 * channel_position
        )

        # Final normalization
        scores = combined.clip(-1, 1)

        metadata = {
            "adx_period": adx_period,
            "ma_period": ma_period,
            "channel_period": channel_period,
            "adx_mean": adx.mean(),
            "ma_deviation_mean": ma_deviation.mean(),
            "channel_position_mean": channel_position.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "regime_detector",
    category="composite",
    description="Market regime detection signal",
    tags=["regime", "volatility", "trend"],
)
class RegimeDetector(Signal):
    """
    Market Regime Detection Signal.

    Detects the current market regime based on:
    - Volatility regime (low/medium/high)
    - Trend regime (uptrend/range/downtrend)

    Output interpretation:
    - Strong positive: Low volatility uptrend (favorable for momentum)
    - Positive: Medium volatility with trend
    - Near zero: Ranging or transitional market
    - Negative: High volatility or downtrend
    - Strong negative: High volatility downtrend (risk-off)

    Parameters:
        vol_period: Volatility calculation period
        vol_lookback: Lookback for volatility regime comparison
        trend_period: Trend detection period
        vol_weight: Weight for volatility component
        trend_weight: Weight for trend component
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="vol_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=40,
                step=10,
                description="Volatility calculation period",
            ),
            ParameterSpec(
                name="vol_lookback",
                default=252,
                searchable=False,
                min_value=126,
                max_value=504,
                description="Lookback for volatility percentile",
            ),
            ParameterSpec(
                name="trend_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=50,
                step=10,
                description="Trend detection period",
            ),
            ParameterSpec(
                name="vol_weight",
                default=0.4,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for volatility component",
            ),
            ParameterSpec(
                name="trend_weight",
                default=0.6,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for trend component",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        vol_period = self._params["vol_period"]
        vol_lookback = self._params["vol_lookback"]
        trend_period = self._params["trend_period"]
        vol_weight = self._params["vol_weight"]
        trend_weight = self._params["trend_weight"]

        close = data["close"]
        returns = close.pct_change()

        # 1. Volatility regime
        rolling_vol = returns.rolling(window=vol_period, min_periods=1).std() * np.sqrt(252)

        # Calculate volatility percentile over lookback (O(n) vectorized)
        # Use rolling().apply with raw=True for C-level performance
        def _percentile_in_window(x: np.ndarray) -> float:
            """Calculate percentile of last value within window (excluding itself)."""
            if len(x) <= 1:
                return 0.5  # Neutral when not enough data
            current = x[-1]
            historical = x[:-1]
            return (historical < current).sum() / len(historical)

        # Window = vol_lookback + 1 to match original: [i-vol_lookback : i+1]
        # x[-1] = current (index i), x[:-1] = historical (vol_lookback elements)
        # min_periods = vol_lookback to start calculation at index vol_lookback
        vol_percentile = rolling_vol.rolling(
            window=vol_lookback + 1,
            min_periods=vol_lookback  # Start at index vol_lookback (matches original)
        ).apply(_percentile_in_window, raw=True)

        # Low vol (percentile < 0.3) -> positive, High vol (percentile > 0.7) -> negative
        # Fill NaN (insufficient history) with 0.0 (neutral)
        vol_scores = (0.5 - vol_percentile).fillna(0.0)
        vol_signal = 2 * vol_scores  # Scale to [-1, +1]

        # 2. Trend regime
        ma_short = close.rolling(window=trend_period // 2, min_periods=1).mean()
        ma_long = close.rolling(window=trend_period, min_periods=1).mean()

        # Trend strength: how far short MA is from long MA
        ma_diff = ma_short - ma_long
        ma_std = close.rolling(window=trend_period, min_periods=1).std()
        ma_std = ma_std.replace(0, np.nan).ffill().fillna(1)
        trend_signal = self.normalize_tanh(ma_diff / ma_std, scale=0.5)

        # 3. Combine: favor low vol uptrend, penalize high vol downtrend
        combined = (
            vol_weight * vol_signal +
            trend_weight * trend_signal
        )

        # Final scores
        scores = combined.clip(-1, 1)

        # Determine regime labels for metadata
        last_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0
        last_trend = trend_signal.iloc[-1] if len(trend_signal) > 0 else 0

        if last_vol < rolling_vol.quantile(0.33):
            vol_regime = "low"
        elif last_vol > rolling_vol.quantile(0.67):
            vol_regime = "high"
        else:
            vol_regime = "medium"

        if last_trend > 0.3:
            trend_regime = "uptrend"
        elif last_trend < -0.3:
            trend_regime = "downtrend"
        else:
            trend_regime = "range"

        metadata = {
            "vol_period": vol_period,
            "trend_period": trend_period,
            "weights": {"vol": vol_weight, "trend": trend_weight},
            "current_vol_regime": vol_regime,
            "current_trend_regime": trend_regime,
            "rolling_vol_mean": rolling_vol.mean(),
            "trend_signal_mean": trend_signal.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)
