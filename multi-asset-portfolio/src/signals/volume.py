"""
Volume-based Signals - Order flow and volume pressure indicators.

Implements volume-based signals including:
- OBV Momentum: On-Balance Volume rate of change
- Money Flow Index (MFI): Volume-weighted RSI
- VWAP Deviation: Price deviation from Volume Weighted Average Price
- Accumulation/Distribution: A/D line momentum

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult, TimeframeAffinity, TimeframeConfig
from .registry import SignalRegistry


@SignalRegistry.register(
    "obv_momentum",
    category="volume",
    description="On-Balance Volume momentum signal",
    tags=["volume", "order_flow", "trend"],
)
class OBVMomentumSignal(Signal):
    """
    On-Balance Volume (OBV) Momentum Signal.

    OBV accumulates volume based on price direction. This signal measures
    the rate of change of OBV to detect volume pressure trends.

    Positive OBV momentum suggests accumulation (bullish).
    Negative OBV momentum suggests distribution (bearish).

    Parameters:
        lookback: Lookback period for OBV momentum calculation (searchable, 5-60)
        scale: Tanh scaling factor (fixed, default 0.5)

    Formula:
        OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]
        OBV_momentum = (OBV[t] - OBV[t-lookback]) / OBV_std
        score = tanh(OBV_momentum * scale)

    Example:
        signal = OBVMomentumSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """OBV Momentum: medium-term (5-60 days).

        Volume-based momentum is effective across short to medium timeframes.
        Very long periods smooth out the accumulation/distribution patterns.
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
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=60,
                step=5,
                description="Lookback period for OBV momentum",
            ),
            ParameterSpec(
                name="scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=2.0,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [5, 10, 15, 20, 30, 40, 60],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        scale = self._params["scale"]

        close = data["close"]
        volume = data.get("volume", pd.Series(1, index=close.index))

        # Calculate OBV
        price_direction = np.sign(close.diff())
        obv = (price_direction * volume).cumsum()

        # Calculate OBV momentum (rate of change)
        obv_change = obv.diff(periods=lookback)

        # Normalize by rolling std to make comparable
        obv_std = obv_change.rolling(window=lookback, min_periods=1).std()
        obv_std = obv_std.replace(0, np.nan).ffill().fillna(1)

        obv_momentum = obv_change / obv_std

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(obv_momentum, scale=scale)

        metadata = {
            "lookback": lookback,
            "scale": scale,
            "obv_momentum_mean": obv_momentum.mean(),
            "obv_momentum_std": obv_momentum.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "money_flow_index",
    category="volume",
    description="Money Flow Index (volume-weighted RSI) signal",
    tags=["volume", "oscillator", "reversal"],
)
class MoneyFlowIndexSignal(Signal):
    """
    Money Flow Index (MFI) Signal.

    MFI is a volume-weighted RSI that incorporates both price and volume.
    Measures buying and selling pressure.

    Parameters:
        period: MFI calculation period (searchable, 7-28)
        oversold_level: Oversold threshold (fixed, default 20)
        overbought_level: Overbought threshold (fixed, default 80)
        scale: Tanh scaling factor (fixed, default 0.04)

    Formula:
        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price * Volume
        MFI = 100 - (100 / (1 + Money Flow Ratio))
        score = tanh(-(MFI - 50) * scale)  # Inverted for mean reversion

    Example:
        signal = MoneyFlowIndexSignal(period=14)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Money Flow Index: short-term oscillator (7-28 days).

        Like RSI, MFI is designed as a short-term overbought/oversold indicator.
        Note: period spec is [7, 28]
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=7,
            max_period=28,
            # Only medium(20) is within spec range [7, 28]
            supported_variants=["medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=14,
                searchable=True,
                min_value=7,
                max_value=28,
                step=7,
                description="MFI calculation period",
            ),
            ParameterSpec(
                name="oversold_level",
                default=20.0,
                searchable=False,
                min_value=10.0,
                max_value=30.0,
                description="Oversold threshold",
            ),
            ParameterSpec(
                name="overbought_level",
                default=80.0,
                searchable=False,
                min_value=70.0,
                max_value=90.0,
                description="Overbought threshold",
            ),
            ParameterSpec(
                name="scale",
                default=0.04,
                searchable=False,
                min_value=0.01,
                max_value=0.1,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "period": [7, 14, 21, 28],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        period = self._params["period"]
        oversold_level = self._params["oversold_level"]
        overbought_level = self._params["overbought_level"]
        scale = self._params["scale"]

        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        volume = data.get("volume", pd.Series(1, index=close.index))

        # Calculate Typical Price
        typical_price = (high + low + close) / 3

        # Calculate Raw Money Flow
        raw_money_flow = typical_price * volume

        # Determine positive/negative money flow
        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)

        # Calculate money flow sums over period
        positive_sum = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_sum = negative_flow.rolling(window=period, min_periods=1).sum()

        # Calculate Money Flow Ratio and MFI
        negative_sum = negative_sum.replace(0, np.nan).ffill().fillna(1)
        money_flow_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))

        # Center MFI around 50
        centered_mfi = mfi - 50

        # Invert for mean reversion (high MFI = overbought = sell)
        raw_signal = -centered_mfi

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate zone statistics
        oversold_pct = (mfi < oversold_level).mean() * 100
        overbought_pct = (mfi > overbought_level).mean() * 100

        metadata = {
            "period": period,
            "oversold_level": oversold_level,
            "overbought_level": overbought_level,
            "scale": scale,
            "mfi_mean": mfi.mean(),
            "mfi_std": mfi.std(),
            "oversold_pct": oversold_pct,
            "overbought_pct": overbought_pct,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "vwap_deviation",
    category="volume",
    description="VWAP deviation mean reversion signal",
    tags=["volume", "reversal", "institutional"],
)
class VWAPDeviationSignal(Signal):
    """
    VWAP Deviation Signal.

    Measures price deviation from Volume Weighted Average Price.
    Institutional traders often use VWAP as fair value reference.

    Parameters:
        lookback: VWAP calculation lookback (searchable, 5-60)
        scale: Tanh scaling factor (fixed, default 1.0)

    Formula:
        VWAP = cumsum(price * volume) / cumsum(volume)
        deviation = (close - VWAP) / VWAP_std
        score = tanh(-deviation * scale)  # Inverted for mean reversion

    Example:
        signal = VWAPDeviationSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """VWAP Deviation: medium-term mean reversion (5-60 days).

        VWAP is used by institutional traders for intraday and short-term
        fair value assessment. Longer periods lose effectiveness.
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
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=60,
                step=5,
                description="Rolling VWAP calculation period",
            ),
            ParameterSpec(
                name="scale",
                default=1.0,
                searchable=False,
                min_value=0.1,
                max_value=3.0,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [5, 10, 15, 20, 30, 40, 60],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        scale = self._params["scale"]

        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        volume = data.get("volume", pd.Series(1, index=close.index))

        # Use typical price for VWAP calculation
        typical_price = (high + low + close) / 3

        # Calculate rolling VWAP
        pv = typical_price * volume
        rolling_pv = pv.rolling(window=lookback, min_periods=1).sum()
        rolling_vol = volume.rolling(window=lookback, min_periods=1).sum()
        rolling_vol = rolling_vol.replace(0, np.nan).ffill().fillna(1)

        vwap = rolling_pv / rolling_vol

        # Calculate deviation from VWAP
        deviation = close - vwap

        # Normalize by rolling std
        dev_std = deviation.rolling(window=lookback, min_periods=1).std()
        dev_std = dev_std.replace(0, np.nan).ffill().fillna(1)

        normalized_deviation = deviation / dev_std

        # Invert for mean reversion
        raw_signal = -normalized_deviation

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate deviation statistics
        pct_above = (close > vwap).mean() * 100
        pct_below = (close < vwap).mean() * 100

        metadata = {
            "lookback": lookback,
            "scale": scale,
            "deviation_mean": deviation.mean(),
            "deviation_std": deviation.std(),
            "pct_above_vwap": pct_above,
            "pct_below_vwap": pct_below,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "accumulation_distribution",
    category="volume",
    description="Accumulation/Distribution line momentum signal",
    tags=["volume", "order_flow", "trend"],
)
class AccumulationDistributionSignal(Signal):
    """
    Accumulation/Distribution (A/D) Line Signal.

    A/D line uses the close location within the high-low range,
    providing a more nuanced view than OBV.

    Parameters:
        lookback: Momentum calculation lookback (searchable, 5-60)
        scale: Tanh scaling factor (fixed, default 0.5)

    Formula:
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        A/D = cumsum(CLV * Volume)
        A/D_momentum = (A/D[t] - A/D[t-lookback]) / A/D_std
        score = tanh(A/D_momentum * scale)

    Example:
        signal = AccumulationDistributionSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """A/D Line: medium-term (5-60 days).

        Similar to OBV momentum, effective for medium-term trend detection.
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
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=60,
                step=5,
                description="Lookback for A/D momentum",
            ),
            ParameterSpec(
                name="scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=2.0,
                description="Tanh scaling factor",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [5, 10, 15, 20, 30, 40, 60],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        scale = self._params["scale"]

        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        volume = data.get("volume", pd.Series(1, index=close.index))

        # Calculate Close Location Value (CLV)
        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan).ffill().fillna(1)

        clv = ((close - low) - (high - close)) / hl_range

        # Calculate A/D line
        ad_line = (clv * volume).cumsum()

        # Calculate A/D momentum
        ad_change = ad_line.diff(periods=lookback)

        # Normalize by rolling std
        ad_std = ad_change.rolling(window=lookback, min_periods=1).std()
        ad_std = ad_std.replace(0, np.nan).ffill().fillna(1)

        ad_momentum = ad_change / ad_std

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(ad_momentum, scale=scale)

        # Calculate CLV statistics
        accumulation_pct = (clv > 0).mean() * 100
        distribution_pct = (clv < 0).mean() * 100

        metadata = {
            "lookback": lookback,
            "scale": scale,
            "ad_momentum_mean": ad_momentum.mean(),
            "ad_momentum_std": ad_momentum.std(),
            "accumulation_pct": accumulation_pct,
            "distribution_pct": distribution_pct,
            "clv_mean": clv.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)
