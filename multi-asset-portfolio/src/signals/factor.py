"""
Factor Signals - Cross-sectional factor-based indicators.

Implements factor-based signals including:
- ValueFactor: Value indicators (distance from 52-week high, long-term average)
- QualityFactor: Quality indicators (volatility stability, return consistency)
- LowVolFactor: Low volatility factor (favoring low-vol assets)
- MomentumFactor: Cross-sectional momentum (relative strength)

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "value_factor",
    category="factor",
    description="Value factor based on distance from 52-week high",
    tags=["factor", "value", "contrarian"],
)
class ValueFactorSignal(Signal):
    """
    Value Factor Signal.

    Measures value based on:
    1. Distance from 52-week high (oversold = value opportunity)
    2. Comparison to long-term average price

    Lower prices relative to historical highs indicate higher value.
    This is a contrarian signal (buys weakness, sells strength).

    Parameters:
        high_lookback: Lookback for 52-week high (searchable, 126-252)
        avg_lookback: Lookback for long-term average (fixed, 200)
        high_weight: Weight for high-based signal (fixed, 0.6)
        avg_weight: Weight for average-based signal (fixed, 0.4)
        scale: Tanh scaling factor (fixed, 2.0)

    Formula:
        distance_from_high = (52w_high - close) / 52w_high
        distance_from_avg = (long_avg - close) / long_avg
        raw_value = high_weight * distance_from_high + avg_weight * distance_from_avg
        score = tanh(raw_value * scale)

    Example:
        signal = ValueFactorSignal(high_lookback=252)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="high_lookback",
                default=252,
                searchable=True,
                min_value=126,
                max_value=252,
                step=21,
                description="Lookback period for 52-week high (days)",
            ),
            ParameterSpec(
                name="avg_lookback",
                default=200,
                searchable=False,
                min_value=50,
                max_value=252,
                description="Lookback period for long-term average",
            ),
            ParameterSpec(
                name="high_weight",
                default=0.6,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for 52-week high component",
            ),
            ParameterSpec(
                name="avg_weight",
                default=0.4,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for long-term average component",
            ),
            ParameterSpec(
                name="scale",
                default=2.0,
                searchable=False,
                min_value=0.5,
                max_value=5.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        high_lookback = self._params["high_lookback"]
        avg_lookback = self._params["avg_lookback"]
        high_weight = self._params["high_weight"]
        avg_weight = self._params["avg_weight"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate 52-week (or specified period) high
        rolling_high = close.rolling(window=high_lookback, min_periods=1).max()

        # Calculate distance from high (positive = below high = value)
        distance_from_high = (rolling_high - close) / rolling_high.replace(0, np.nan)
        distance_from_high = distance_from_high.fillna(0)

        # Calculate long-term average
        long_avg = close.rolling(window=avg_lookback, min_periods=1).mean()

        # Calculate distance from average (positive = below average = value)
        distance_from_avg = (long_avg - close) / long_avg.replace(0, np.nan)
        distance_from_avg = distance_from_avg.fillna(0)

        # Combine signals
        raw_value = high_weight * distance_from_high + avg_weight * distance_from_avg

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_value, scale=scale)

        metadata = {
            "high_lookback": high_lookback,
            "avg_lookback": avg_lookback,
            "distance_from_high_mean": distance_from_high.mean(),
            "distance_from_avg_mean": distance_from_avg.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "quality_factor",
    category="factor",
    description="Quality factor based on volatility stability and return consistency",
    tags=["factor", "quality", "stability"],
)
class QualityFactorSignal(Signal):
    """
    Quality Factor Signal.

    Measures quality based on:
    1. Volatility stability (low vol-of-vol is higher quality)
    2. Return consistency (more consecutive up days = higher quality)

    Higher quality assets get higher scores.

    Parameters:
        vol_lookback: Lookback for volatility calculation (searchable, 20-60)
        consistency_lookback: Lookback for consistency check (fixed, 20)
        vol_weight: Weight for volatility stability (fixed, 0.5)
        consistency_weight: Weight for return consistency (fixed, 0.5)
        scale: Tanh scaling factor (fixed, 1.5)

    Formula:
        vol = rolling_std(returns, vol_lookback)
        vol_of_vol = rolling_std(vol, vol_lookback)
        vol_stability = -vol_of_vol (low vol-of-vol = high quality)
        consistency = sum(returns > 0) / consistency_lookback
        raw_quality = vol_weight * vol_stability_norm + consistency_weight * consistency_norm
        score = tanh(raw_quality * scale)

    Example:
        signal = QualityFactorSignal(vol_lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="vol_lookback",
                default=20,
                searchable=True,
                min_value=20,
                max_value=60,
                step=10,
                description="Lookback period for volatility calculation",
            ),
            ParameterSpec(
                name="consistency_lookback",
                default=20,
                searchable=False,
                min_value=10,
                max_value=40,
                description="Lookback period for consistency check",
            ),
            ParameterSpec(
                name="vol_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for volatility stability component",
            ),
            ParameterSpec(
                name="consistency_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for return consistency component",
            ),
            ParameterSpec(
                name="scale",
                default=1.5,
                searchable=False,
                min_value=0.5,
                max_value=5.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        vol_lookback = self._params["vol_lookback"]
        consistency_lookback = self._params["consistency_lookback"]
        vol_weight = self._params["vol_weight"]
        consistency_weight = self._params["consistency_weight"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=vol_lookback, min_periods=1).std()

        # Calculate volatility of volatility (vol-of-vol)
        vol_of_vol = rolling_vol.rolling(window=vol_lookback, min_periods=1).std()

        # Normalize vol_of_vol using z-score approach
        # Lower vol-of-vol = higher quality = positive signal
        vol_stability = self.normalize_zscore_tanh(-vol_of_vol, lookback=vol_lookback)

        # Calculate return consistency (proportion of up days)
        up_days = (returns > 0).astype(float)
        consistency_ratio = up_days.rolling(
            window=consistency_lookback, min_periods=1
        ).mean()

        # Center around 0.5 (50% up days = neutral)
        consistency_centered = (consistency_ratio - 0.5) * 2  # Scale to [-1, +1]

        # Combine signals
        raw_quality = vol_weight * vol_stability + consistency_weight * consistency_centered

        # Normalize to [-1, +1]
        scores = raw_quality.clip(-1, 1)

        metadata = {
            "vol_lookback": vol_lookback,
            "consistency_lookback": consistency_lookback,
            "vol_of_vol_mean": vol_of_vol.mean(),
            "consistency_ratio_mean": consistency_ratio.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "low_vol_factor",
    category="factor",
    description="Low volatility factor favoring low-vol assets",
    tags=["factor", "low_vol", "defensive"],
)
class LowVolFactorSignal(Signal):
    """
    Low Volatility Factor Signal.

    Favors assets with lower volatility based on the low-volatility anomaly.
    Lower volatility assets historically provide better risk-adjusted returns.

    Parameters:
        vol_lookback: Lookback for volatility calculation (searchable, 21-126)
        vol_halflife: Exponential decay halflife for vol weighting (fixed, 63)
        scale: Tanh scaling factor (fixed, 3.0)

    Formula:
        vol = rolling_std(returns, vol_lookback) * sqrt(252)  # Annualized
        vol_zscore = (vol - rolling_mean(vol)) / rolling_std(vol)
        score = tanh(-vol_zscore * scale)  # Negative = low vol is positive

    Example:
        signal = LowVolFactorSignal(vol_lookback=63)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="vol_lookback",
                default=63,
                searchable=True,
                min_value=21,
                max_value=126,
                step=21,
                description="Lookback period for volatility calculation (days)",
            ),
            ParameterSpec(
                name="vol_halflife",
                default=63,
                searchable=False,
                min_value=21,
                max_value=126,
                description="Exponential decay halflife for volatility",
            ),
            ParameterSpec(
                name="scale",
                default=3.0,
                searchable=False,
                min_value=1.0,
                max_value=10.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        vol_lookback = self._params["vol_lookback"]
        vol_halflife = self._params["vol_halflife"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate exponentially-weighted volatility
        ewm_vol = returns.ewm(halflife=vol_halflife, min_periods=vol_lookback // 2).std()

        # Annualize
        annualized_vol = ewm_vol * np.sqrt(252)

        # Calculate z-score of volatility
        vol_mean = annualized_vol.rolling(window=vol_lookback, min_periods=1).mean()
        vol_std = annualized_vol.rolling(window=vol_lookback, min_periods=1).std()
        vol_std = vol_std.replace(0, np.nan).ffill().fillna(1)

        vol_zscore = (annualized_vol - vol_mean) / vol_std

        # Low volatility = positive signal
        raw_signal = -vol_zscore

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_signal, scale=scale / 3)

        metadata = {
            "vol_lookback": vol_lookback,
            "vol_halflife": vol_halflife,
            "annualized_vol_mean": annualized_vol.mean(),
            "annualized_vol_std": annualized_vol.std(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "momentum_factor",
    category="factor",
    description="Cross-sectional momentum factor (relative strength)",
    tags=["factor", "momentum", "relative"],
)
class MomentumFactorSignal(Signal):
    """
    Cross-Sectional Momentum Factor Signal.

    Measures momentum relative to the asset's own history.
    In cross-sectional context, this would rank assets by their momentum.

    For single-asset use, measures momentum strength across multiple timeframes.

    Parameters:
        short_lookback: Short-term momentum period (searchable, 21-63)
        long_lookback: Long-term momentum period (searchable, 126-252)
        skip_recent: Days to skip for momentum calculation (fixed, 5)
        short_weight: Weight for short-term momentum (fixed, 0.4)
        long_weight: Weight for long-term momentum (fixed, 0.6)
        scale: Tanh scaling factor (fixed, 5.0)

    Formula:
        # Skip most recent days to avoid mean reversion
        mom_short = (close[t-skip] - close[t-short-skip]) / close[t-short-skip]
        mom_long = (close[t-skip] - close[t-long-skip]) / close[t-long-skip]
        raw_momentum = short_weight * mom_short + long_weight * mom_long
        score = tanh(raw_momentum * scale)

    Example:
        signal = MomentumFactorSignal(short_lookback=21, long_lookback=252)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="short_lookback",
                default=21,
                searchable=True,
                min_value=21,
                max_value=63,
                step=21,
                description="Short-term momentum period (days)",
            ),
            ParameterSpec(
                name="long_lookback",
                default=252,
                searchable=True,
                min_value=126,
                max_value=252,
                step=21,
                description="Long-term momentum period (days)",
            ),
            ParameterSpec(
                name="skip_recent",
                default=5,
                searchable=False,
                min_value=1,
                max_value=21,
                description="Days to skip (avoid short-term reversal)",
            ),
            ParameterSpec(
                name="short_weight",
                default=0.4,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for short-term momentum",
            ),
            ParameterSpec(
                name="long_weight",
                default=0.6,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for long-term momentum",
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

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        short_lookback = self._params["short_lookback"]
        long_lookback = self._params["long_lookback"]
        skip_recent = self._params["skip_recent"]
        short_weight = self._params["short_weight"]
        long_weight = self._params["long_weight"]
        scale = self._params["scale"]

        close = data["close"]

        # Skip most recent days to avoid mean reversion effect
        close_lagged = close.shift(skip_recent)

        # Short-term momentum (1 month)
        close_short_ago = close.shift(short_lookback + skip_recent)
        mom_short = (close_lagged - close_short_ago) / close_short_ago.replace(0, np.nan)
        mom_short = mom_short.fillna(0)

        # Long-term momentum (12 months minus 1 month)
        close_long_ago = close.shift(long_lookback + skip_recent)
        mom_long = (close_lagged - close_long_ago) / close_long_ago.replace(0, np.nan)
        mom_long = mom_long.fillna(0)

        # Combine short and long momentum
        total_weight = short_weight + long_weight
        raw_momentum = (short_weight * mom_short + long_weight * mom_long) / total_weight

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_momentum, scale=scale)

        metadata = {
            "short_lookback": short_lookback,
            "long_lookback": long_lookback,
            "skip_recent": skip_recent,
            "mom_short_mean": mom_short.mean(),
            "mom_long_mean": mom_long.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "size_factor",
    category="factor",
    description="Size factor proxy using price level and volatility",
    tags=["factor", "size", "small_cap"],
)
class SizeFactorSignal(Signal):
    """
    Size Factor Signal (Proxy).

    Since we don't have market cap data in OHLCV, use price level and
    volatility as proxies for size. Lower price + higher volatility
    typically indicates smaller market cap.

    This is a simplified proxy and should be replaced with actual
    market cap data when available.

    Parameters:
        price_lookback: Lookback for price normalization (fixed, 252)
        vol_lookback: Lookback for volatility (fixed, 63)
        price_weight: Weight for price component (fixed, 0.5)
        vol_weight: Weight for volatility component (fixed, 0.5)
        favor_small: If True, favors small-cap (positive); else large-cap
        scale: Tanh scaling factor (fixed, 2.0)

    Formula:
        price_rank = rolling_rank(close, price_lookback)
        vol = rolling_std(returns, vol_lookback) * sqrt(252)
        vol_rank = rolling_rank(vol, vol_lookback)
        # Small = low price + high vol
        raw_size = price_weight * (1 - price_rank) + vol_weight * vol_rank
        score = tanh(raw_size * scale) if favor_small else tanh(-raw_size * scale)

    Example:
        signal = SizeFactorSignal(favor_small=True)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="price_lookback",
                default=252,
                searchable=False,
                min_value=63,
                max_value=504,
                description="Lookback for price normalization",
            ),
            ParameterSpec(
                name="vol_lookback",
                default=63,
                searchable=False,
                min_value=21,
                max_value=126,
                description="Lookback for volatility calculation",
            ),
            ParameterSpec(
                name="price_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for price component",
            ),
            ParameterSpec(
                name="vol_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for volatility component",
            ),
            ParameterSpec(
                name="favor_small",
                default=True,
                searchable=False,
                description="If True, favor small-cap proxy; else large-cap",
            ),
            ParameterSpec(
                name="scale",
                default=2.0,
                searchable=False,
                min_value=0.5,
                max_value=5.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        price_lookback = self._params["price_lookback"]
        vol_lookback = self._params["vol_lookback"]
        price_weight = self._params["price_weight"]
        vol_weight = self._params["vol_weight"]
        favor_small = self._params["favor_small"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate price percentile (rank within lookback period)
        def rolling_percentile(series, window):
            return series.rolling(window=window, min_periods=1).apply(
                lambda x: (x[-1:].values[0] - x.min()) / (x.max() - x.min() + 1e-10)
                if len(x) > 1 else 0.5,
                raw=False,
            )

        price_rank = rolling_percentile(close, price_lookback)

        # Calculate annualized volatility percentile
        vol = returns.rolling(window=vol_lookback, min_periods=1).std() * np.sqrt(252)
        vol_rank = rolling_percentile(vol, vol_lookback)

        # Small-cap proxy: low price + high volatility
        # Invert price rank (low price = high value for small)
        small_cap_score = price_weight * (1 - price_rank) + vol_weight * vol_rank

        # Center around 0
        raw_size = (small_cap_score - 0.5) * 2

        # Apply direction preference
        if not favor_small:
            raw_size = -raw_size

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_size, scale=scale)

        metadata = {
            "price_lookback": price_lookback,
            "vol_lookback": vol_lookback,
            "favor_small": favor_small,
            "price_rank_mean": price_rank.mean(),
            "vol_rank_mean": vol_rank.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)
