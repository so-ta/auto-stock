"""
Sentiment Signals - Market sentiment and fear/greed indicators.

Implements sentiment-based signals including:
- VIX Sentiment: Risk-on/off based on VIX levels
- Put/Call Ratio: Contrarian signal from options market
- Market Breadth: Advance-decline based signal
- Fear & Greed Composite: Combined sentiment indicator

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "vix_sentiment",
    category="sentiment",
    description="VIX-based risk-on/risk-off sentiment signal",
    tags=["volatility", "fear", "contrarian"],
)
class VIXSentimentSignal(Signal):
    """
    VIX Sentiment Signal.

    Uses VIX levels to determine risk-on/risk-off sentiment.
    High VIX (fear) generates negative scores (risk-off).
    Low VIX (complacency) generates positive scores (risk-on).

    This is a contrarian signal: extreme fear can indicate buying opportunities,
    while extreme complacency can indicate selling opportunities.

    Parameters:
        vix_high: VIX threshold for high fear (searchable, 20-40)
        vix_low: VIX threshold for low fear (searchable, 10-20)
        lookback: Smoothing period for VIX (fixed, default 5)
        contrarian: If True, high VIX = positive score (buy fear)

    Formula:
        If contrarian=False:
            score = tanh((vix_mid - vix) / scale)
        If contrarian=True:
            score = tanh((vix - vix_mid) / scale)

    Note:
        Requires 'vix' column in input data or uses 'close' as proxy.
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="vix_high",
                default=25.0,
                searchable=True,
                min_value=20.0,
                max_value=40.0,
                step=5.0,
                description="VIX threshold for high fear",
            ),
            ParameterSpec(
                name="vix_low",
                default=15.0,
                searchable=True,
                min_value=10.0,
                max_value=20.0,
                step=2.5,
                description="VIX threshold for low fear/complacency",
            ),
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=False,
                min_value=1,
                max_value=20,
                description="Smoothing period for VIX",
            ),
            ParameterSpec(
                name="contrarian",
                default=False,
                searchable=False,
                description="If True, high VIX generates positive scores (buy fear)",
            ),
            ParameterSpec(
                name="scale",
                default=10.0,
                searchable=False,
                min_value=1.0,
                max_value=30.0,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "vix_high": [20, 25, 30, 35],
            "vix_low": [12, 15, 18],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        vix_high = self._params["vix_high"]
        vix_low = self._params["vix_low"]
        lookback = self._params["lookback"]
        contrarian = self._params["contrarian"]
        scale = self._params["scale"]

        # Use 'vix' column if available, otherwise estimate from realized volatility
        if "vix" in data.columns:
            vix = data["vix"]
        else:
            # Estimate VIX proxy from realized volatility (annualized)
            returns = data["close"].pct_change()
            vix = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100

        # Apply smoothing
        if lookback > 1:
            vix = vix.rolling(window=lookback, min_periods=1).mean()

        # Calculate midpoint
        vix_mid = (vix_high + vix_low) / 2

        # Generate scores
        if contrarian:
            # High VIX = positive (buy the fear)
            raw_score = (vix - vix_mid) / scale
        else:
            # High VIX = negative (risk-off)
            raw_score = (vix_mid - vix) / scale

        scores = self.normalize_tanh(raw_score, scale=1.0)

        metadata = {
            "vix_high": vix_high,
            "vix_low": vix_low,
            "vix_mid": vix_mid,
            "contrarian": contrarian,
            "vix_mean": vix.mean(),
            "vix_current": vix.iloc[-1] if len(vix) > 0 else None,
            "score_mean": scores.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "put_call_ratio",
    category="sentiment",
    description="Put/Call ratio contrarian sentiment signal",
    tags=["options", "contrarian", "fear"],
)
class PutCallRatioSignal(Signal):
    """
    Put/Call Ratio Signal.

    Uses the put/call ratio as a contrarian sentiment indicator.
    High P/C ratio (fear) generates positive scores (contrarian buy).
    Low P/C ratio (greed) generates negative scores (contrarian sell).

    Parameters:
        pc_high: High P/C ratio threshold (searchable, 0.8-1.5)
        pc_low: Low P/C ratio threshold (searchable, 0.5-0.8)
        lookback: Smoothing period (fixed, default 5)

    Note:
        Requires 'put_call_ratio' column in input data.
        If not available, generates neutral scores.
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="pc_high",
                default=1.0,
                searchable=True,
                min_value=0.8,
                max_value=1.5,
                step=0.1,
                description="High P/C ratio threshold (fear)",
            ),
            ParameterSpec(
                name="pc_low",
                default=0.7,
                searchable=True,
                min_value=0.5,
                max_value=0.8,
                step=0.05,
                description="Low P/C ratio threshold (greed)",
            ),
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=False,
                min_value=1,
                max_value=20,
                description="Smoothing period for P/C ratio",
            ),
            ParameterSpec(
                name="scale",
                default=3.0,
                searchable=False,
                min_value=1.0,
                max_value=10.0,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "pc_high": [0.9, 1.0, 1.1, 1.2],
            "pc_low": [0.6, 0.7, 0.8],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        pc_high = self._params["pc_high"]
        pc_low = self._params["pc_low"]
        lookback = self._params["lookback"]
        scale = self._params["scale"]

        # Check for put_call_ratio column
        if "put_call_ratio" not in data.columns:
            # Return neutral scores if data not available
            scores = pd.Series(0.0, index=data.index)
            return SignalResult(
                scores=scores,
                metadata={
                    "warning": "put_call_ratio column not available",
                    "pc_high": pc_high,
                    "pc_low": pc_low,
                },
            )

        pc_ratio = data["put_call_ratio"]

        # Apply smoothing
        if lookback > 1:
            pc_ratio = pc_ratio.rolling(window=lookback, min_periods=1).mean()

        # Calculate midpoint
        pc_mid = (pc_high + pc_low) / 2

        # Contrarian: high P/C = buy signal, low P/C = sell signal
        raw_score = (pc_ratio - pc_mid) * scale

        scores = self.normalize_tanh(raw_score, scale=1.0)

        metadata = {
            "pc_high": pc_high,
            "pc_low": pc_low,
            "pc_mid": pc_mid,
            "pc_ratio_mean": pc_ratio.mean(),
            "pc_ratio_current": pc_ratio.iloc[-1] if len(pc_ratio) > 0 else None,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "market_breadth",
    category="sentiment",
    description="Market breadth (advance/decline) sentiment signal",
    tags=["breadth", "trend", "confirmation"],
)
class MarketBreadthSignal(Signal):
    """
    Market Breadth Signal.

    Uses the percentage of advancing stocks as a market health indicator.
    High breadth (many advancing) generates positive scores.
    Low breadth (many declining) generates negative scores.

    Parameters:
        breadth_high: High breadth threshold (searchable, 55-75%)
        breadth_low: Low breadth threshold (searchable, 25-45%)
        lookback: Smoothing period (fixed, default 5)

    Note:
        Requires 'breadth' or 'advance_decline' column in input data.
        Breadth should be in [0, 100] range (percentage of advancing stocks).
        If not available, estimates from price action.
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="breadth_high",
                default=60.0,
                searchable=True,
                min_value=55.0,
                max_value=75.0,
                step=5.0,
                description="High breadth threshold (%)",
            ),
            ParameterSpec(
                name="breadth_low",
                default=40.0,
                searchable=True,
                min_value=25.0,
                max_value=45.0,
                step=5.0,
                description="Low breadth threshold (%)",
            ),
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=False,
                min_value=1,
                max_value=20,
                description="Smoothing period for breadth",
            ),
            ParameterSpec(
                name="scale",
                default=0.1,
                searchable=False,
                min_value=0.01,
                max_value=0.5,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "breadth_high": [55, 60, 65, 70],
            "breadth_low": [30, 35, 40, 45],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        breadth_high = self._params["breadth_high"]
        breadth_low = self._params["breadth_low"]
        lookback = self._params["lookback"]
        scale = self._params["scale"]

        # Check for breadth column
        if "breadth" in data.columns:
            breadth = data["breadth"]
        elif "advance_decline" in data.columns:
            # Convert A/D ratio to percentage
            ad = data["advance_decline"]
            breadth = (ad / (ad.abs() * 2) + 0.5) * 100
        else:
            # Estimate breadth from price momentum
            # Use rolling positive return percentage as proxy
            returns = data["close"].pct_change()
            breadth = (
                (returns > 0).rolling(window=20, min_periods=5).mean() * 100
            )

        # Apply smoothing
        if lookback > 1:
            breadth = breadth.rolling(window=lookback, min_periods=1).mean()

        # Calculate midpoint (neutral = 50%)
        breadth_mid = (breadth_high + breadth_low) / 2

        # Score: above midpoint = bullish, below = bearish
        raw_score = (breadth - breadth_mid) * scale

        scores = self.normalize_tanh(raw_score, scale=1.0)

        metadata = {
            "breadth_high": breadth_high,
            "breadth_low": breadth_low,
            "breadth_mid": breadth_mid,
            "breadth_mean": breadth.mean(),
            "breadth_current": breadth.iloc[-1] if len(breadth) > 0 else None,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "fear_greed_composite",
    category="sentiment",
    description="Composite fear/greed index combining multiple sentiment indicators",
    tags=["composite", "fear", "greed", "sentiment"],
)
class FearGreedCompositeSignal(Signal):
    """
    Fear & Greed Composite Signal.

    Combines multiple sentiment indicators into a single composite score:
    - VIX level (volatility fear)
    - Price momentum (market trend)
    - Relative strength (market vs safe haven)
    - Volatility trend (fear momentum)

    Parameters:
        vix_weight: Weight for VIX component (fixed, default 0.30)
        momentum_weight: Weight for momentum component (fixed, default 0.25)
        strength_weight: Weight for relative strength (fixed, default 0.25)
        vol_trend_weight: Weight for volatility trend (fixed, default 0.20)
        vix_threshold: VIX neutral level (searchable, 15-25)
        momentum_lookback: Momentum calculation period (searchable, 10-30)

    Output interpretation:
        +1.0 = Extreme greed (market euphoria)
        +0.5 = Greed (bullish sentiment)
         0.0 = Neutral (balanced sentiment)
        -0.5 = Fear (bearish sentiment)
        -1.0 = Extreme fear (panic)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="vix_weight",
                default=0.30,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for VIX component",
            ),
            ParameterSpec(
                name="momentum_weight",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for momentum component",
            ),
            ParameterSpec(
                name="strength_weight",
                default=0.25,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for relative strength component",
            ),
            ParameterSpec(
                name="vol_trend_weight",
                default=0.20,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for volatility trend component",
            ),
            ParameterSpec(
                name="vix_threshold",
                default=20.0,
                searchable=True,
                min_value=15.0,
                max_value=25.0,
                step=2.5,
                description="VIX neutral threshold",
            ),
            ParameterSpec(
                name="momentum_lookback",
                default=20,
                searchable=True,
                min_value=10,
                max_value=30,
                step=5,
                description="Momentum calculation lookback period",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "vix_threshold": [15, 18, 20, 22, 25],
            "momentum_lookback": [10, 15, 20, 25, 30],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        vix_weight = self._params["vix_weight"]
        momentum_weight = self._params["momentum_weight"]
        strength_weight = self._params["strength_weight"]
        vol_trend_weight = self._params["vol_trend_weight"]
        vix_threshold = self._params["vix_threshold"]
        momentum_lookback = self._params["momentum_lookback"]

        close = data["close"]
        n = len(close)

        # 1. VIX Component (inverted: low VIX = greed, high VIX = fear)
        if "vix" in data.columns:
            vix = data["vix"]
        else:
            # Estimate from realized volatility
            returns = close.pct_change()
            vix = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100

        vix_score = self.normalize_tanh(vix_threshold - vix, scale=0.1)

        # 2. Momentum Component (positive momentum = greed)
        momentum = close.pct_change(periods=momentum_lookback)
        momentum_score = self.normalize_tanh(momentum, scale=5.0)

        # 3. Relative Strength Component (above MA = greed)
        ma_50 = close.rolling(window=50, min_periods=20).mean()
        relative_strength = (close - ma_50) / ma_50
        strength_score = self.normalize_tanh(relative_strength, scale=5.0)

        # 4. Volatility Trend Component (decreasing vol = greed)
        vol_short = close.pct_change().rolling(window=5, min_periods=2).std()
        vol_long = close.pct_change().rolling(window=20, min_periods=5).std()
        vol_ratio = vol_short / vol_long.replace(0, np.nan).ffill()
        vol_trend_score = self.normalize_tanh(1 - vol_ratio, scale=2.0)

        # Combine components with weights
        total_weight = vix_weight + momentum_weight + strength_weight + vol_trend_weight

        composite_score = (
            vix_weight * vix_score
            + momentum_weight * momentum_score
            + strength_weight * strength_score
            + vol_trend_weight * vol_trend_score
        ) / total_weight

        # Ensure output is in [-1, +1]
        scores = composite_score.clip(-1, 1)

        metadata = {
            "weights": {
                "vix": vix_weight,
                "momentum": momentum_weight,
                "strength": strength_weight,
                "vol_trend": vol_trend_weight,
            },
            "vix_threshold": vix_threshold,
            "momentum_lookback": momentum_lookback,
            "component_means": {
                "vix": vix_score.mean(),
                "momentum": momentum_score.mean(),
                "strength": strength_score.mean(),
                "vol_trend": vol_trend_score.mean(),
            },
            "composite_mean": scores.mean(),
            "composite_current": scores.iloc[-1] if n > 0 else None,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "vix_term_structure",
    category="sentiment",
    description="VIX term structure (contango/backwardation) signal",
    tags=["volatility", "term_structure", "fear"],
)
class VIXTermStructureSignal(Signal):
    """
    VIX Term Structure Signal.

    Uses the relationship between short-term and long-term VIX to gauge
    market sentiment. Contango (normal) indicates complacency, while
    backwardation indicates fear.

    Parameters:
        lookback: Smoothing period (fixed, default 5)
        threshold: Neutral threshold for term structure ratio (searchable, 0.95-1.05)

    Note:
        Requires 'vix' and 'vix3m' (or similar) columns.
        If not available, estimates from realized volatility term structure.
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=False,
                min_value=1,
                max_value=20,
                description="Smoothing period",
            ),
            ParameterSpec(
                name="threshold",
                default=1.0,
                searchable=True,
                min_value=0.95,
                max_value=1.05,
                step=0.01,
                description="Neutral threshold for VIX ratio",
            ),
            ParameterSpec(
                name="scale",
                default=10.0,
                searchable=False,
                min_value=1.0,
                max_value=30.0,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "threshold": [0.95, 0.97, 1.0, 1.03, 1.05],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        threshold = self._params["threshold"]
        scale = self._params["scale"]

        # Check for VIX term structure columns
        has_term_structure = "vix" in data.columns and "vix3m" in data.columns

        if has_term_structure:
            vix_short = data["vix"]
            vix_long = data["vix3m"]
        else:
            # Estimate from realized volatility
            returns = data["close"].pct_change()
            vix_short = returns.rolling(window=5, min_periods=2).std() * np.sqrt(252) * 100
            vix_long = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100

        # Calculate term structure ratio (short/long)
        # < 1 = contango (normal, greed)
        # > 1 = backwardation (fear)
        term_ratio = vix_short / vix_long.replace(0, np.nan).ffill()

        # Apply smoothing
        if lookback > 1:
            term_ratio = term_ratio.rolling(window=lookback, min_periods=1).mean()

        # Score: backwardation (>1) = fear = negative, contango (<1) = greed = positive
        raw_score = (threshold - term_ratio) * scale

        scores = self.normalize_tanh(raw_score, scale=1.0)

        metadata = {
            "threshold": threshold,
            "has_term_structure_data": has_term_structure,
            "term_ratio_mean": term_ratio.mean(),
            "term_ratio_current": term_ratio.iloc[-1] if len(term_ratio) > 0 else None,
            "in_backwardation": (term_ratio > 1).sum() / len(term_ratio) if len(term_ratio) > 0 else 0,
        }

        return SignalResult(scores=scores, metadata=metadata)
