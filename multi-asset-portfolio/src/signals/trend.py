"""
Trend Following Signals - Advanced trend-based indicators.

Implements sophisticated trend-following signals including:
- DualMomentum: Absolute + Relative momentum combination
- TrendFollowing: MA crossover + ADX + Donchian breakout
- AdaptiveTrend: Volatility-adaptive trend parameters
- CrossAssetMomentum: Cross-asset relative strength

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "dual_momentum",
    category="trend",
    description="Dual momentum signal (absolute + relative)",
    tags=["momentum", "trend", "dual"],
)
class DualMomentumSignal(Signal):
    """
    Dual Momentum Signal.

    Combines absolute momentum (self vs past) and relative momentum
    (self vs benchmark/other assets). Only generates bullish signal
    when BOTH momentums are positive.

    Absolute Momentum: Is the asset trending up vs its own history?
    Relative Momentum: Is the asset outperforming alternatives?

    Parameters:
        abs_lookback: Lookback for absolute momentum (days)
        rel_lookback: Lookback for relative momentum (days)
        abs_weight: Weight for absolute momentum component
        rel_weight: Weight for relative momentum component
        scale: Tanh scaling factor

    Formula:
        abs_mom = (close[t] - close[t-abs_lookback]) / close[t-abs_lookback]
        rel_mom = abs_mom - benchmark_return  (or excess return)
        combined = abs_weight * abs_mom + rel_weight * rel_mom
        score = tanh(combined * scale) if both > 0 else tanh(combined * scale * 0.5)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="abs_lookback",
                default=126,
                searchable=True,
                min_value=20,
                max_value=252,
                step=21,
                description="Absolute momentum lookback (days)",
            ),
            ParameterSpec(
                name="rel_lookback",
                default=126,
                searchable=True,
                min_value=20,
                max_value=252,
                step=21,
                description="Relative momentum lookback (days)",
            ),
            ParameterSpec(
                name="abs_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for absolute momentum",
            ),
            ParameterSpec(
                name="rel_weight",
                default=0.5,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for relative momentum",
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

        abs_lookback = self._params["abs_lookback"]
        rel_lookback = self._params["rel_lookback"]
        abs_weight = self._params["abs_weight"]
        rel_weight = self._params["rel_weight"]
        scale = self._params["scale"]

        close = data["close"]

        # Absolute momentum: return vs own history
        abs_return = close.pct_change(periods=abs_lookback)

        # Relative momentum: excess return vs risk-free proxy (use rolling mean as baseline)
        # In practice, this would compare against a benchmark
        rolling_mean_return = close.pct_change().rolling(window=rel_lookback).mean()
        excess_return = abs_return - rolling_mean_return * rel_lookback

        # Combined score with dual momentum logic
        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            if i < max(abs_lookback, rel_lookback):
                scores.iloc[i] = 0.0
                continue

            abs_mom = abs_return.iloc[i]
            rel_mom = excess_return.iloc[i]

            if pd.isna(abs_mom) or pd.isna(rel_mom):
                scores.iloc[i] = 0.0
                continue

            combined = abs_weight * abs_mom + rel_weight * rel_mom

            # Dual momentum condition: amplify if both positive, dampen otherwise
            if abs_mom > 0 and rel_mom > 0:
                # Both positive: full signal strength
                raw_score = combined * scale
            elif abs_mom < 0 and rel_mom < 0:
                # Both negative: full bearish signal
                raw_score = combined * scale
            else:
                # Mixed signals: reduce confidence
                raw_score = combined * scale * 0.5

            scores.iloc[i] = np.tanh(raw_score)

        return SignalResult(
            scores=scores,
            metadata={
                "abs_lookback": abs_lookback,
                "rel_lookback": rel_lookback,
                "abs_weight": abs_weight,
                "rel_weight": rel_weight,
                "dual_positive_pct": ((abs_return > 0) & (excess_return > 0)).mean(),
                "dual_negative_pct": ((abs_return < 0) & (excess_return < 0)).mean(),
            },
        )


@SignalRegistry.register(
    "trend_following",
    category="trend",
    description="Comprehensive trend following signal (MA cross + ADX + Donchian)",
    tags=["trend", "ma", "adx", "breakout"],
)
class TrendFollowingSignal(Signal):
    """
    Comprehensive Trend Following Signal.

    Combines three trend-following components:
    1. MA Crossover: Short MA vs Long MA direction
    2. ADX (Average Directional Index): Trend strength measurement
    3. Donchian Channel: Breakout detection

    Parameters:
        ma_short: Short MA period
        ma_long: Long MA period
        adx_period: ADX calculation period
        donchian_period: Donchian channel period
        ma_weight: Weight for MA crossover component
        adx_weight: Weight for ADX component
        donchian_weight: Weight for Donchian component

    Output:
        Positive: Strong uptrend detected
        Negative: Strong downtrend detected
        Near zero: No clear trend or weak trend
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="ma_short",
                default=20,
                searchable=True,
                min_value=5,
                max_value=50,
                step=5,
                description="Short MA period",
            ),
            ParameterSpec(
                name="ma_long",
                default=50,
                searchable=True,
                min_value=20,
                max_value=200,
                step=10,
                description="Long MA period",
            ),
            ParameterSpec(
                name="adx_period",
                default=14,
                searchable=False,
                min_value=7,
                max_value=28,
                description="ADX calculation period",
            ),
            ParameterSpec(
                name="donchian_period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=55,
                step=5,
                description="Donchian channel period",
            ),
            ParameterSpec(
                name="ma_weight",
                default=0.4,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for MA crossover",
            ),
            ParameterSpec(
                name="adx_weight",
                default=0.3,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for ADX",
            ),
            ParameterSpec(
                name="donchian_weight",
                default=0.3,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for Donchian breakout",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        ma_short = self._params["ma_short"]
        ma_long = self._params["ma_long"]
        adx_period = self._params["adx_period"]
        donchian_period = self._params["donchian_period"]
        ma_weight = self._params["ma_weight"]
        adx_weight = self._params["adx_weight"]
        donchian_weight = self._params["donchian_weight"]

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # 1. MA Crossover signal
        ma_short_line = close.rolling(window=ma_short).mean()
        ma_long_line = close.rolling(window=ma_long).mean()

        # MA signal: normalized distance between MAs
        ma_diff = (ma_short_line - ma_long_line) / ma_long_line
        ma_signal = np.tanh(ma_diff * 20)  # Scale and normalize

        # 2. ADX calculation (simplified)
        adx, plus_di, minus_di = self._compute_adx(high, low, close, adx_period)

        # ADX signal: trend strength * direction
        # Direction from DI comparison, strength from ADX
        di_diff = (plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_normalized = adx / 50.0  # ADX typically 0-50, normalize to 0-1
        adx_signal = di_diff * adx_normalized

        # 3. Donchian Channel breakout
        upper_band = high.rolling(window=donchian_period).max()
        lower_band = low.rolling(window=donchian_period).min()
        mid_band = (upper_band + lower_band) / 2

        # Position within channel: -1 at lower, +1 at upper
        channel_width = upper_band - lower_band
        donchian_pos = (close - mid_band) / (channel_width / 2 + 1e-10)
        donchian_signal = donchian_pos.clip(-1, 1)

        # Combined score
        scores = (
            ma_weight * ma_signal.fillna(0)
            + adx_weight * adx_signal.fillna(0)
            + donchian_weight * donchian_signal.fillna(0)
        )

        # Final normalization
        scores = scores.clip(-1, 1)

        return SignalResult(
            scores=scores,
            metadata={
                "ma_short": ma_short,
                "ma_long": ma_long,
                "adx_period": adx_period,
                "donchian_period": donchian_period,
                "weights": {"ma": ma_weight, "adx": adx_weight, "donchian": donchian_weight},
                "avg_adx": adx.mean() if not adx.isna().all() else 0,
                "ma_crossover_count": (ma_short_line > ma_long_line).sum(),
            },
        )

    def _compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute ADX (Average Directional Index)."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di


@SignalRegistry.register(
    "adaptive_trend",
    category="trend",
    description="Volatility-adaptive trend signal",
    tags=["trend", "adaptive", "volatility"],
)
class AdaptiveTrendSignal(Signal):
    """
    Adaptive Trend Signal.

    Adjusts trend-following parameters based on current volatility regime:
    - High volatility: Use shorter lookback periods (react faster)
    - Low volatility: Use longer lookback periods (smoother signals)

    Parameters:
        base_lookback: Base lookback period (adjusted by volatility)
        vol_lookback: Volatility calculation period
        min_lookback: Minimum lookback (high vol regime)
        max_lookback: Maximum lookback (low vol regime)
        vol_threshold_high: Volatility percentile for high regime
        vol_threshold_low: Volatility percentile for low regime

    Output:
        Adaptive momentum signal with dynamic parameter adjustment
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="base_lookback",
                default=40,
                searchable=True,
                min_value=20,
                max_value=100,
                step=10,
                description="Base lookback period",
            ),
            ParameterSpec(
                name="vol_lookback",
                default=20,
                searchable=False,
                min_value=10,
                max_value=60,
                description="Volatility calculation period",
            ),
            ParameterSpec(
                name="min_lookback",
                default=10,
                searchable=False,
                min_value=5,
                max_value=30,
                description="Minimum lookback (high vol)",
            ),
            ParameterSpec(
                name="max_lookback",
                default=100,
                searchable=False,
                min_value=50,
                max_value=200,
                description="Maximum lookback (low vol)",
            ),
            ParameterSpec(
                name="vol_threshold_high",
                default=0.75,
                searchable=False,
                min_value=0.5,
                max_value=0.95,
                description="Volatility percentile for high regime",
            ),
            ParameterSpec(
                name="vol_threshold_low",
                default=0.25,
                searchable=False,
                min_value=0.05,
                max_value=0.5,
                description="Volatility percentile for low regime",
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

        base_lookback = self._params["base_lookback"]
        vol_lookback = self._params["vol_lookback"]
        min_lookback = self._params["min_lookback"]
        max_lookback = self._params["max_lookback"]
        vol_threshold_high = self._params["vol_threshold_high"]
        vol_threshold_low = self._params["vol_threshold_low"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=vol_lookback).std() * np.sqrt(252)

        scores = pd.Series(index=data.index, dtype=float)
        adaptive_lookbacks = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            if i < max_lookback + vol_lookback:
                scores.iloc[i] = 0.0
                adaptive_lookbacks.iloc[i] = base_lookback
                continue

            # Determine volatility regime
            current_vol = rolling_vol.iloc[i]
            historical_vol = rolling_vol.iloc[max(0, i - 252) : i]

            if len(historical_vol) < 20 or pd.isna(current_vol):
                adaptive_lookback = base_lookback
            else:
                vol_percentile = (historical_vol < current_vol).mean()

                if vol_percentile > vol_threshold_high:
                    # High volatility: use shorter lookback
                    adaptive_lookback = min_lookback
                elif vol_percentile < vol_threshold_low:
                    # Low volatility: use longer lookback
                    adaptive_lookback = max_lookback
                else:
                    # Normal volatility: interpolate
                    vol_range = vol_threshold_high - vol_threshold_low
                    vol_pos = (vol_percentile - vol_threshold_low) / vol_range
                    adaptive_lookback = int(
                        max_lookback - vol_pos * (max_lookback - min_lookback)
                    )

            adaptive_lookbacks.iloc[i] = adaptive_lookback

            # Calculate momentum with adaptive lookback
            lookback = int(adaptive_lookback)
            if i >= lookback:
                momentum = (close.iloc[i] - close.iloc[i - lookback]) / close.iloc[i - lookback]
                scores.iloc[i] = np.tanh(momentum * scale)
            else:
                scores.iloc[i] = 0.0

        return SignalResult(
            scores=scores,
            metadata={
                "base_lookback": base_lookback,
                "min_lookback": min_lookback,
                "max_lookback": max_lookback,
                "avg_adaptive_lookback": adaptive_lookbacks.mean(),
                "vol_regime_distribution": {
                    "high_vol_pct": (adaptive_lookbacks == min_lookback).mean(),
                    "low_vol_pct": (adaptive_lookbacks == max_lookback).mean(),
                    "normal_pct": (
                        (adaptive_lookbacks > min_lookback) & (adaptive_lookbacks < max_lookback)
                    ).mean(),
                },
            },
        )


@SignalRegistry.register(
    "cross_asset_momentum",
    category="trend",
    description="Cross-asset relative strength momentum",
    tags=["momentum", "cross-asset", "risk-on-off"],
)
class CrossAssetMomentumSignal(Signal):
    """
    Cross-Asset Momentum Signal.

    Evaluates the asset's momentum relative to a broader market context.
    Useful for determining risk-on/risk-off positioning.

    For a single asset, compares short-term vs long-term momentum:
    - Strong short-term AND long-term momentum = risk-on
    - Weak short-term OR long-term momentum = risk-off tendency

    Parameters:
        short_lookback: Short-term momentum period
        long_lookback: Long-term momentum period
        vol_adjust: Whether to adjust for volatility

    Output:
        +1: Strong risk-on signal (both momentums positive and strong)
        -1: Strong risk-off signal (both momentums negative)
         0: Mixed or neutral
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="short_lookback",
                default=21,
                searchable=True,
                min_value=5,
                max_value=63,
                step=7,
                description="Short-term momentum period (days)",
            ),
            ParameterSpec(
                name="long_lookback",
                default=126,
                searchable=True,
                min_value=63,
                max_value=252,
                step=21,
                description="Long-term momentum period (days)",
            ),
            ParameterSpec(
                name="vol_adjust",
                default=True,
                searchable=False,
                description="Adjust momentum for volatility",
            ),
            ParameterSpec(
                name="vol_lookback",
                default=21,
                searchable=False,
                min_value=10,
                max_value=63,
                description="Volatility calculation period",
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

        short_lookback = self._params["short_lookback"]
        long_lookback = self._params["long_lookback"]
        vol_adjust = self._params["vol_adjust"]
        vol_lookback = self._params["vol_lookback"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Short-term momentum
        short_mom = close.pct_change(periods=short_lookback)

        # Long-term momentum
        long_mom = close.pct_change(periods=long_lookback)

        # Volatility adjustment
        if vol_adjust:
            rolling_vol = returns.rolling(window=vol_lookback).std()
            short_mom_adj = short_mom / (rolling_vol * np.sqrt(short_lookback) + 1e-10)
            long_mom_adj = long_mom / (rolling_vol * np.sqrt(long_lookback) + 1e-10)
        else:
            short_mom_adj = short_mom
            long_mom_adj = long_mom

        # Risk-on/risk-off scoring
        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data)):
            if i < long_lookback + vol_lookback:
                scores.iloc[i] = 0.0
                continue

            s_mom = short_mom_adj.iloc[i]
            l_mom = long_mom_adj.iloc[i]

            if pd.isna(s_mom) or pd.isna(l_mom):
                scores.iloc[i] = 0.0
                continue

            # Combined signal with emphasis on agreement
            if s_mom > 0 and l_mom > 0:
                # Risk-on: both positive
                combined = (s_mom + l_mom) / 2
                scores.iloc[i] = np.tanh(combined * scale)
            elif s_mom < 0 and l_mom < 0:
                # Risk-off: both negative
                combined = (s_mom + l_mom) / 2
                scores.iloc[i] = np.tanh(combined * scale)
            else:
                # Mixed signals: reduced confidence
                combined = (s_mom + l_mom) / 2
                scores.iloc[i] = np.tanh(combined * scale * 0.5)

        # Determine risk regime for metadata
        risk_on_pct = ((short_mom > 0) & (long_mom > 0)).mean()
        risk_off_pct = ((short_mom < 0) & (long_mom < 0)).mean()

        return SignalResult(
            scores=scores,
            metadata={
                "short_lookback": short_lookback,
                "long_lookback": long_lookback,
                "vol_adjust": vol_adjust,
                "risk_regime": {
                    "risk_on_pct": risk_on_pct,
                    "risk_off_pct": risk_off_pct,
                    "mixed_pct": 1.0 - risk_on_pct - risk_off_pct,
                },
                "avg_short_mom": short_mom.mean(),
                "avg_long_mom": long_mom.mean(),
            },
        )


@SignalRegistry.register(
    "multi_timeframe_momentum",
    category="trend",
    description="Multi-timeframe momentum signal combining daily, weekly, and monthly views",
    tags=["momentum", "multi-timeframe", "consensus"],
)
class MultiTimeframeMomentumSignal(Signal):
    """
    Multi-Timeframe Momentum Signal.

    Combines momentum signals across three timeframes (daily, weekly, monthly)
    for a more robust trend detection. Each timeframe uses its own lookback
    periods and weights.

    Timeframe philosophy:
    - Daily: Captures short-term price action and recent momentum
    - Weekly: Medium-term trend direction
    - Monthly: Long-term trend and major market regimes

    Parameters:
        daily_lookbacks: List of lookback periods for daily (searchable)
        weekly_lookbacks: List of lookback periods for weekly (searchable)
        monthly_lookbacks: List of lookback periods for monthly (searchable)
        daily_weight: Weight for daily timeframe (fixed, default 0.30)
        weekly_weight: Weight for weekly timeframe (fixed, default 0.35)
        monthly_weight: Weight for monthly timeframe (fixed, default 0.35)
        consensus_threshold: Minimum agreement ratio for strong signal (fixed, default 0.6)

    Output:
        +1.0: Strong uptrend consensus across all timeframes
         0.0: Mixed or conflicting signals
        -1.0: Strong downtrend consensus across all timeframes
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="daily_short",
                default=5,
                searchable=True,
                min_value=3,
                max_value=10,
                step=1,
                description="Short daily lookback",
            ),
            ParameterSpec(
                name="daily_medium",
                default=10,
                searchable=True,
                min_value=8,
                max_value=15,
                step=1,
                description="Medium daily lookback",
            ),
            ParameterSpec(
                name="daily_long",
                default=20,
                searchable=True,
                min_value=15,
                max_value=25,
                step=5,
                description="Long daily lookback",
            ),
            ParameterSpec(
                name="weekly_short",
                default=20,
                searchable=True,
                min_value=15,
                max_value=30,
                step=5,
                description="Short weekly lookback (approx 4 weeks)",
            ),
            ParameterSpec(
                name="weekly_medium",
                default=40,
                searchable=True,
                min_value=30,
                max_value=50,
                step=5,
                description="Medium weekly lookback (approx 8 weeks)",
            ),
            ParameterSpec(
                name="weekly_long",
                default=60,
                searchable=True,
                min_value=50,
                max_value=70,
                step=5,
                description="Long weekly lookback (approx 12 weeks)",
            ),
            ParameterSpec(
                name="monthly_short",
                default=60,
                searchable=True,
                min_value=40,
                max_value=80,
                step=10,
                description="Short monthly lookback (approx 3 months)",
            ),
            ParameterSpec(
                name="monthly_medium",
                default=120,
                searchable=True,
                min_value=100,
                max_value=140,
                step=20,
                description="Medium monthly lookback (approx 6 months)",
            ),
            ParameterSpec(
                name="monthly_long",
                default=252,
                searchable=True,
                min_value=200,
                max_value=280,
                step=20,
                description="Long monthly lookback (approx 12 months)",
            ),
            ParameterSpec(
                name="daily_weight",
                default=0.30,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for daily timeframe",
            ),
            ParameterSpec(
                name="weekly_weight",
                default=0.35,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for weekly timeframe",
            ),
            ParameterSpec(
                name="monthly_weight",
                default=0.35,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for monthly timeframe",
            ),
            ParameterSpec(
                name="consensus_threshold",
                default=0.6,
                searchable=False,
                min_value=0.5,
                max_value=0.9,
                description="Minimum agreement ratio for strong signal",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "daily_short": [3, 5, 7],
            "daily_medium": [10, 12],
            "weekly_short": [15, 20, 25],
            "weekly_medium": [35, 40, 45],
            "monthly_short": [50, 60, 70],
            "monthly_medium": [100, 120],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        # Extract parameters
        daily_lookbacks = [
            self._params["daily_short"],
            self._params["daily_medium"],
            self._params["daily_long"],
        ]
        weekly_lookbacks = [
            self._params["weekly_short"],
            self._params["weekly_medium"],
            self._params["weekly_long"],
        ]
        monthly_lookbacks = [
            self._params["monthly_short"],
            self._params["monthly_medium"],
            self._params["monthly_long"],
        ]

        daily_weight = self._params["daily_weight"]
        weekly_weight = self._params["weekly_weight"]
        monthly_weight = self._params["monthly_weight"]
        consensus_threshold = self._params["consensus_threshold"]

        close = data["close"]

        # Calculate momentum for each lookback period
        def calc_momentum_score(lookback: int) -> pd.Series:
            """Calculate normalized momentum score for a given lookback."""
            returns = close.pct_change(periods=lookback)
            # Scale factor inversely proportional to lookback (shorter = more volatile)
            scale = 5.0 / np.sqrt(lookback / 20)
            return np.tanh(returns * scale)

        # Daily timeframe scores
        daily_scores = pd.concat(
            [calc_momentum_score(lb) for lb in daily_lookbacks], axis=1
        )
        daily_avg = daily_scores.mean(axis=1)

        # Weekly timeframe scores
        weekly_scores = pd.concat(
            [calc_momentum_score(lb) for lb in weekly_lookbacks], axis=1
        )
        weekly_avg = weekly_scores.mean(axis=1)

        # Monthly timeframe scores
        monthly_scores = pd.concat(
            [calc_momentum_score(lb) for lb in monthly_lookbacks], axis=1
        )
        monthly_avg = monthly_scores.mean(axis=1)

        # Calculate consensus metrics
        all_scores = pd.concat([daily_scores, weekly_scores, monthly_scores], axis=1)
        n_signals = all_scores.shape[1]

        # Count how many signals agree on direction
        bullish_count = (all_scores > 0).sum(axis=1)
        bearish_count = (all_scores < 0).sum(axis=1)
        consensus_ratio = (
            pd.concat([bullish_count, bearish_count], axis=1).max(axis=1) / n_signals
        )

        # Weighted combination of timeframes
        total_weight = daily_weight + weekly_weight + monthly_weight
        combined_score = (
            daily_weight * daily_avg
            + weekly_weight * weekly_avg
            + monthly_weight * monthly_avg
        ) / total_weight

        # Apply consensus adjustment: amplify when consensus is high
        consensus_multiplier = np.where(
            consensus_ratio >= consensus_threshold,
            1.0 + (consensus_ratio - consensus_threshold),  # Boost high consensus
            consensus_ratio / consensus_threshold,  # Dampen low consensus
        )

        scores = (combined_score * consensus_multiplier).clip(-1, 1)

        # Calculate metadata
        strong_bull = ((bullish_count / n_signals) >= consensus_threshold).mean()
        strong_bear = ((bearish_count / n_signals) >= consensus_threshold).mean()

        return SignalResult(
            scores=scores,
            metadata={
                "daily_lookbacks": daily_lookbacks,
                "weekly_lookbacks": weekly_lookbacks,
                "monthly_lookbacks": monthly_lookbacks,
                "weights": {
                    "daily": daily_weight,
                    "weekly": weekly_weight,
                    "monthly": monthly_weight,
                },
                "consensus_threshold": consensus_threshold,
                "timeframe_means": {
                    "daily": daily_avg.mean(),
                    "weekly": weekly_avg.mean(),
                    "monthly": monthly_avg.mean(),
                },
                "consensus_stats": {
                    "strong_bull_pct": strong_bull,
                    "strong_bear_pct": strong_bear,
                    "avg_consensus_ratio": consensus_ratio.mean(),
                },
            },
        )


@SignalRegistry.register(
    "timeframe_consensus",
    category="trend",
    description="Timeframe consensus signal based on agreement across lookback periods",
    tags=["consensus", "multi-timeframe", "confirmation"],
)
class TimeframeConsensusSignal(Signal):
    """
    Timeframe Consensus Signal.

    Generates signals based on the level of agreement across multiple
    timeframes. Only produces strong signals when a sufficient number
    of timeframes agree on direction.

    Key difference from MultiTimeframeMomentum:
    - This signal focuses on consensus level rather than weighted average
    - Outputs are discrete levels based on agreement threshold
    - Better for regime detection and position sizing

    Parameters:
        lookbacks: List of lookback periods to check (configurable)
        min_consensus: Minimum fraction of timeframes that must agree
        neutral_zone: Width of neutral zone around zero

    Output:
        +1.0: Strong bullish consensus (>= min_consensus agree on up)
        +0.5: Moderate bullish consensus (majority agree on up)
         0.0: No consensus or conflicting signals
        -0.5: Moderate bearish consensus
        -1.0: Strong bearish consensus
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback_1",
                default=5,
                searchable=True,
                min_value=3,
                max_value=10,
                step=1,
                description="First (shortest) lookback period",
            ),
            ParameterSpec(
                name="lookback_2",
                default=10,
                searchable=True,
                min_value=8,
                max_value=20,
                step=2,
                description="Second lookback period",
            ),
            ParameterSpec(
                name="lookback_3",
                default=20,
                searchable=True,
                min_value=15,
                max_value=30,
                step=5,
                description="Third lookback period",
            ),
            ParameterSpec(
                name="lookback_4",
                default=40,
                searchable=True,
                min_value=30,
                max_value=60,
                step=5,
                description="Fourth lookback period",
            ),
            ParameterSpec(
                name="lookback_5",
                default=60,
                searchable=True,
                min_value=50,
                max_value=90,
                step=10,
                description="Fifth lookback period",
            ),
            ParameterSpec(
                name="lookback_6",
                default=120,
                searchable=True,
                min_value=90,
                max_value=150,
                step=15,
                description="Sixth (longest) lookback period",
            ),
            ParameterSpec(
                name="min_consensus",
                default=0.6,
                searchable=True,
                min_value=0.5,
                max_value=0.9,
                step=0.1,
                description="Minimum consensus ratio for strong signal",
            ),
            ParameterSpec(
                name="neutral_threshold",
                default=0.02,
                searchable=False,
                min_value=0.001,
                max_value=0.05,
                description="Threshold for considering a signal neutral",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback_1": [3, 5, 7],
            "lookback_3": [15, 20, 25],
            "lookback_4": [30, 40, 50],
            "lookback_6": [100, 120, 140],
            "min_consensus": [0.5, 0.6, 0.7],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookbacks = [
            self._params["lookback_1"],
            self._params["lookback_2"],
            self._params["lookback_3"],
            self._params["lookback_4"],
            self._params["lookback_5"],
            self._params["lookback_6"],
        ]
        min_consensus = self._params["min_consensus"]
        neutral_threshold = self._params["neutral_threshold"]

        close = data["close"]
        n_lookbacks = len(lookbacks)

        # Calculate momentum for each lookback
        momentum_signals = pd.DataFrame(index=data.index)
        for i, lb in enumerate(lookbacks):
            returns = close.pct_change(periods=lb)
            # Classify as bullish (+1), bearish (-1), or neutral (0)
            signal = pd.Series(0, index=data.index)
            signal[returns > neutral_threshold] = 1
            signal[returns < -neutral_threshold] = -1
            momentum_signals[f"lb_{lb}"] = signal

        # Count votes
        bullish_votes = (momentum_signals == 1).sum(axis=1)
        bearish_votes = (momentum_signals == -1).sum(axis=1)
        neutral_votes = (momentum_signals == 0).sum(axis=1)

        # Calculate consensus ratios
        bullish_ratio = bullish_votes / n_lookbacks
        bearish_ratio = bearish_votes / n_lookbacks

        # Generate scores based on consensus level
        scores = pd.Series(0.0, index=data.index)

        # Strong bullish consensus
        strong_bull_mask = bullish_ratio >= min_consensus
        scores[strong_bull_mask] = 1.0

        # Moderate bullish (majority but below threshold)
        mod_bull_mask = (bullish_ratio > 0.5) & ~strong_bull_mask
        scores[mod_bull_mask] = 0.5

        # Strong bearish consensus
        strong_bear_mask = bearish_ratio >= min_consensus
        scores[strong_bear_mask] = -1.0

        # Moderate bearish
        mod_bear_mask = (bearish_ratio > 0.5) & ~strong_bear_mask
        scores[mod_bear_mask] = -0.5

        # Calculate transition metrics for metadata
        score_changes = scores.diff().abs()
        avg_hold_period = 1 / (score_changes.mean() + 1e-10)

        return SignalResult(
            scores=scores,
            metadata={
                "lookbacks": lookbacks,
                "min_consensus": min_consensus,
                "neutral_threshold": neutral_threshold,
                "signal_distribution": {
                    "strong_bull_pct": strong_bull_mask.mean(),
                    "mod_bull_pct": mod_bull_mask.mean(),
                    "neutral_pct": (~strong_bull_mask & ~strong_bear_mask & ~mod_bull_mask & ~mod_bear_mask).mean(),
                    "mod_bear_pct": mod_bear_mask.mean(),
                    "strong_bear_pct": strong_bear_mask.mean(),
                },
                "avg_bullish_ratio": bullish_ratio.mean(),
                "avg_bearish_ratio": bearish_ratio.mean(),
                "avg_hold_period": avg_hold_period,
            },
        )
