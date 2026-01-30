"""
Macro Economic Signals - Macro-economic regime indicators.

Implements macro-based signals including:
- Yield Curve: Term structure of interest rates
- Inflation Expectations: TIP/IEF or breakeven rates
- Credit Spread: Investment grade vs high yield spread
- Dollar Strength: USD index based signals

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "yield_curve",
    category="macro",
    description="Yield curve (TLT/SHY) shape signal for economic regime",
    tags=["rates", "recession", "regime"],
)
class YieldCurveSignal(Signal):
    """
    Yield Curve Signal.

    Uses the ratio of long-term to short-term bond ETFs (TLT/SHY) as a
    proxy for yield curve shape. A steep curve (high ratio) indicates
    economic expansion, while a flat/inverted curve indicates recession risk.

    Parameters:
        lookback: Smoothing period (searchable, 5-30)
        threshold: Neutral threshold for ratio (searchable, 0.9-1.1)
        long_etf: Column name for long-term bond ETF (fixed, default 'tlt')
        short_etf: Column name for short-term bond ETF (fixed, default 'shy')

    Note:
        Requires price data for both long and short-term bond ETFs.
        If not available, uses 'close' column with moving average proxy.

    Output interpretation:
        +1.0 = Very steep curve (economic expansion)
         0.0 = Neutral curve
        -1.0 = Flat/inverted curve (recession risk)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=30,
                step=5,
                description="Smoothing period for ratio calculation",
            ),
            ParameterSpec(
                name="threshold",
                default=1.0,
                searchable=True,
                min_value=0.9,
                max_value=1.1,
                step=0.05,
                description="Neutral threshold for TLT/SHY ratio",
            ),
            ParameterSpec(
                name="long_etf",
                default="tlt",
                searchable=False,
                description="Column name for long-term bond ETF",
            ),
            ParameterSpec(
                name="short_etf",
                default="shy",
                searchable=False,
                description="Column name for short-term bond ETF",
            ),
            ParameterSpec(
                name="scale",
                default=5.0,
                searchable=False,
                min_value=1.0,
                max_value=20.0,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [10, 15, 20, 25],
            "threshold": [0.95, 1.0, 1.05],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        threshold = self._params["threshold"]
        long_etf = self._params["long_etf"]
        short_etf = self._params["short_etf"]
        scale = self._params["scale"]

        # Check for required columns
        has_bond_data = long_etf in data.columns and short_etf in data.columns

        if has_bond_data:
            long_price = data[long_etf]
            short_price = data[short_etf]
        else:
            # Use rolling ratio of long vs short MA as proxy
            close = data["close"]
            long_price = close.rolling(window=60, min_periods=20).mean()
            short_price = close.rolling(window=10, min_periods=5).mean()

        # Calculate ratio
        ratio = long_price / short_price.replace(0, np.nan).ffill()

        # Apply smoothing
        if lookback > 1:
            ratio = ratio.rolling(window=lookback, min_periods=1).mean()

        # Normalize: steep curve (high ratio) = positive, flat = negative
        raw_score = (ratio - threshold) * scale

        scores = self.normalize_tanh(raw_score, scale=1.0)

        metadata = {
            "lookback": lookback,
            "threshold": threshold,
            "has_bond_data": has_bond_data,
            "ratio_mean": ratio.mean(),
            "ratio_current": ratio.iloc[-1] if len(ratio) > 0 else None,
            "score_mean": scores.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "inflation_expectation",
    category="macro",
    description="Inflation expectation (TIP/IEF) signal",
    tags=["inflation", "rates", "regime"],
)
class InflationExpectationSignal(Signal):
    """
    Inflation Expectation Signal.

    Uses the ratio of TIP (inflation-protected) to IEF (nominal treasury)
    as a proxy for inflation expectations. Rising ratio indicates
    increasing inflation expectations.

    Parameters:
        lookback: Smoothing period (searchable, 5-30)
        threshold: Neutral threshold for ratio (searchable, 0.95-1.05)
        momentum_period: Period for momentum calculation (fixed, default 20)

    Note:
        Requires 'tip' and 'ief' columns in input data.
        Higher inflation expectations typically favor:
        - Commodities, TIPS, Real Estate
        - Hurt: Long-duration bonds, growth stocks

    Output interpretation:
        +1.0 = High/rising inflation expectations
         0.0 = Neutral/stable expectations
        -1.0 = Low/falling inflation expectations
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=5,
                max_value=30,
                step=5,
                description="Smoothing period for ratio",
            ),
            ParameterSpec(
                name="threshold",
                default=1.0,
                searchable=True,
                min_value=0.95,
                max_value=1.05,
                step=0.01,
                description="Neutral threshold for TIP/IEF ratio",
            ),
            ParameterSpec(
                name="momentum_period",
                default=20,
                searchable=False,
                min_value=5,
                max_value=60,
                description="Period for momentum calculation",
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
            "lookback": [10, 15, 20, 25],
            "threshold": [0.98, 1.0, 1.02],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        threshold = self._params["threshold"]
        momentum_period = self._params["momentum_period"]
        scale = self._params["scale"]

        # Check for required columns
        has_inflation_data = "tip" in data.columns and "ief" in data.columns

        if has_inflation_data:
            tip_price = data["tip"]
            ief_price = data["ief"]
        else:
            # Use close price volatility as inflation proxy
            close = data["close"]
            returns = close.pct_change()
            # High volatility often correlates with inflation concerns
            tip_price = returns.rolling(window=20, min_periods=5).std()
            ief_price = returns.rolling(window=60, min_periods=20).std()

        # Calculate ratio
        ratio = tip_price / ief_price.replace(0, np.nan).ffill()

        # Apply smoothing
        if lookback > 1:
            ratio = ratio.rolling(window=lookback, min_periods=1).mean()

        # Calculate ratio momentum (direction of change)
        ratio_momentum = ratio.pct_change(periods=momentum_period)

        # Combine level and momentum
        level_score = self.normalize_tanh((ratio - threshold) * scale, scale=1.0)
        momentum_score = self.normalize_tanh(ratio_momentum * 10, scale=1.0)

        # Weighted combination (70% level, 30% momentum)
        scores = 0.7 * level_score + 0.3 * momentum_score
        scores = scores.clip(-1, 1)

        metadata = {
            "lookback": lookback,
            "threshold": threshold,
            "momentum_period": momentum_period,
            "has_inflation_data": has_inflation_data,
            "ratio_mean": ratio.mean(),
            "ratio_current": ratio.iloc[-1] if len(ratio) > 0 else None,
            "ratio_momentum_mean": ratio_momentum.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "credit_spread",
    category="macro",
    description="Credit spread (HYG/LQD) signal for credit conditions",
    tags=["credit", "risk", "regime"],
)
class CreditSpreadSignal(Signal):
    """
    Credit Spread Signal.

    Uses the ratio of HYG (high yield) to LQD (investment grade) as a
    proxy for credit spread conditions. A tightening spread (high ratio)
    indicates risk-on sentiment, while widening spread indicates risk-off.

    Parameters:
        lookback: Smoothing period (searchable, 5-30)
        threshold: Neutral threshold for ratio (searchable, 0.85-1.0)
        momentum_period: Period for momentum calculation (fixed, default 10)

    Note:
        Requires 'hyg' and 'lqd' columns in input data.
        Credit spreads are a leading indicator for:
        - Economic conditions
        - Default risk
        - Risk appetite

    Output interpretation:
        +1.0 = Tight spreads (risk-on, economic confidence)
         0.0 = Neutral spreads
        -1.0 = Wide spreads (risk-off, credit stress)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=10,
                searchable=True,
                min_value=5,
                max_value=30,
                step=5,
                description="Smoothing period for ratio",
            ),
            ParameterSpec(
                name="threshold",
                default=0.92,
                searchable=True,
                min_value=0.85,
                max_value=1.0,
                step=0.02,
                description="Neutral threshold for HYG/LQD ratio",
            ),
            ParameterSpec(
                name="momentum_period",
                default=10,
                searchable=False,
                min_value=5,
                max_value=30,
                description="Period for momentum calculation",
            ),
            ParameterSpec(
                name="scale",
                default=15.0,
                searchable=False,
                min_value=5.0,
                max_value=30.0,
                description="Scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [5, 10, 15, 20],
            "threshold": [0.88, 0.90, 0.92, 0.95],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        threshold = self._params["threshold"]
        momentum_period = self._params["momentum_period"]
        scale = self._params["scale"]

        # Check for required columns
        has_credit_data = "hyg" in data.columns and "lqd" in data.columns

        if has_credit_data:
            hyg_price = data["hyg"]
            lqd_price = data["lqd"]
        else:
            # Use volatility as credit risk proxy
            close = data["close"]
            returns = close.pct_change()
            # High vol periods often correlate with spread widening
            realized_vol = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
            # Invert: low vol = tight spreads = high ratio
            hyg_price = 1 / (realized_vol + 0.01)
            lqd_price = pd.Series(1.0, index=data.index)

        # Calculate ratio
        ratio = hyg_price / lqd_price.replace(0, np.nan).ffill()

        # Apply smoothing
        if lookback > 1:
            ratio = ratio.rolling(window=lookback, min_periods=1).mean()

        # Calculate momentum
        ratio_momentum = ratio.pct_change(periods=momentum_period)

        # Score: high ratio (tight spreads) = positive
        level_score = self.normalize_tanh((ratio - threshold) * scale, scale=1.0)
        momentum_score = self.normalize_tanh(ratio_momentum * 20, scale=1.0)

        # Combine level and momentum
        scores = 0.6 * level_score + 0.4 * momentum_score
        scores = scores.clip(-1, 1)

        metadata = {
            "lookback": lookback,
            "threshold": threshold,
            "momentum_period": momentum_period,
            "has_credit_data": has_credit_data,
            "ratio_mean": ratio.mean(),
            "ratio_current": ratio.iloc[-1] if len(ratio) > 0 else None,
            "spread_tightening": (ratio_momentum > 0).sum() / len(ratio_momentum) if len(ratio_momentum) > 0 else 0,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "dollar_strength",
    category="macro",
    description="US Dollar strength (UUP/DXY proxy) signal",
    tags=["fx", "dollar", "regime"],
)
class DollarStrengthSignal(Signal):
    """
    Dollar Strength Signal.

    Uses USD index or UUP ETF as a measure of dollar strength.
    Strong dollar typically benefits:
    - US domestic companies
    - Importers
    Hurts:
    - Emerging markets
    - Commodities
    - US multinationals

    Parameters:
        lookback: Smoothing period (searchable, 10-40)
        momentum_period: Period for momentum calculation (searchable, 5-20)
        invert: If True, strong dollar = negative score (EM-friendly)

    Note:
        Requires 'uup' or 'dxy' column in input data.
        If not available, uses price momentum as proxy.

    Output interpretation (invert=False):
        +1.0 = Strong/strengthening dollar
         0.0 = Neutral dollar
        -1.0 = Weak/weakening dollar
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=10,
                max_value=40,
                step=5,
                description="Smoothing period",
            ),
            ParameterSpec(
                name="momentum_period",
                default=10,
                searchable=True,
                min_value=5,
                max_value=20,
                step=5,
                description="Period for momentum calculation",
            ),
            ParameterSpec(
                name="invert",
                default=False,
                searchable=False,
                description="If True, strong dollar generates negative scores",
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
            "lookback": [10, 15, 20, 30],
            "momentum_period": [5, 10, 15],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        momentum_period = self._params["momentum_period"]
        invert = self._params["invert"]
        scale = self._params["scale"]

        # Check for dollar index columns
        if "uup" in data.columns:
            dollar = data["uup"]
            has_dollar_data = True
        elif "dxy" in data.columns:
            dollar = data["dxy"]
            has_dollar_data = True
        else:
            # Use inverse of risky asset performance as dollar proxy
            # Strong dollar often correlates with risk-off
            close = data["close"]
            returns = close.pct_change()
            # Inverse correlation: market down = dollar up
            dollar = -returns.cumsum()
            has_dollar_data = False

        # Calculate moving average level
        dollar_ma = dollar.rolling(window=lookback, min_periods=5).mean()

        # Calculate momentum
        dollar_momentum = dollar.pct_change(periods=momentum_period)

        # Level relative to MA
        level_deviation = (dollar - dollar_ma) / dollar_ma.abs().replace(0, 1)

        # Combine level and momentum
        level_score = self.normalize_tanh(level_deviation * scale, scale=1.0)
        momentum_score = self.normalize_tanh(dollar_momentum * scale * 2, scale=1.0)

        scores = 0.5 * level_score + 0.5 * momentum_score

        # Invert if requested (for EM/commodity-friendly signal)
        if invert:
            scores = -scores

        scores = scores.clip(-1, 1)

        metadata = {
            "lookback": lookback,
            "momentum_period": momentum_period,
            "invert": invert,
            "has_dollar_data": has_dollar_data,
            "dollar_mean": dollar.mean(),
            "dollar_current": dollar.iloc[-1] if len(dollar) > 0 else None,
            "momentum_mean": dollar_momentum.mean(),
            "strengthening": (dollar_momentum > 0).sum() / len(dollar_momentum) if len(dollar_momentum) > 0 else 0,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "macro_regime_composite",
    category="macro",
    description="Composite macro regime signal combining multiple indicators",
    tags=["composite", "regime", "macro"],
)
class MacroRegimeCompositeSignal(Signal):
    """
    Macro Regime Composite Signal.

    Combines multiple macro indicators into a single regime signal:
    - Yield curve shape (recession risk)
    - Credit spreads (risk appetite)
    - Dollar strength (global risk)
    - Volatility regime (fear)

    Parameters:
        yield_weight: Weight for yield curve (fixed, default 0.30)
        credit_weight: Weight for credit spread (fixed, default 0.30)
        dollar_weight: Weight for dollar strength (fixed, default 0.20)
        vol_weight: Weight for volatility (fixed, default 0.20)
        smoothing: Final smoothing period (searchable, 5-20)

    Output interpretation:
        +1.0 = Risk-on macro regime (expansion, tight spreads)
         0.0 = Neutral macro regime
        -1.0 = Risk-off macro regime (recession risk, wide spreads)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="yield_weight",
                default=0.30,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for yield curve component",
            ),
            ParameterSpec(
                name="credit_weight",
                default=0.30,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for credit spread component",
            ),
            ParameterSpec(
                name="dollar_weight",
                default=0.20,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for dollar strength component",
            ),
            ParameterSpec(
                name="vol_weight",
                default=0.20,
                searchable=False,
                min_value=0.0,
                max_value=1.0,
                description="Weight for volatility component",
            ),
            ParameterSpec(
                name="smoothing",
                default=10,
                searchable=True,
                min_value=5,
                max_value=20,
                step=5,
                description="Final smoothing period",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "smoothing": [5, 10, 15, 20],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        yield_weight = self._params["yield_weight"]
        credit_weight = self._params["credit_weight"]
        dollar_weight = self._params["dollar_weight"]
        vol_weight = self._params["vol_weight"]
        smoothing = self._params["smoothing"]

        close = data["close"]
        returns = close.pct_change()

        # 1. Yield Curve Proxy (long MA vs short MA ratio)
        ma_long = close.rolling(window=60, min_periods=20).mean()
        ma_short = close.rolling(window=10, min_periods=5).mean()
        yield_proxy = ma_long / ma_short.replace(0, np.nan).ffill()
        yield_score = self.normalize_zscore_tanh(yield_proxy, lookback=60, scale=0.5)

        # 2. Credit Spread Proxy (inverse volatility)
        realized_vol = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
        # Low vol = tight spreads = risk-on
        vol_percentile = realized_vol.rolling(window=252, min_periods=60).apply(
            lambda x: (x.iloc[-1] < x).mean() if len(x) > 0 else 0.5,
            raw=False
        )
        credit_score = self.normalize_tanh(0.5 - vol_percentile, scale=2.0)

        # 3. Dollar Strength Proxy (momentum of returns - strong market = weak dollar proxy)
        momentum_20 = close.pct_change(periods=20)
        # Assume inverse relationship: strong equity = weak dollar
        dollar_score = self.normalize_tanh(-momentum_20, scale=5.0)

        # 4. Volatility Regime (low vol = risk-on)
        vol_score = self.normalize_tanh(0.2 - realized_vol, scale=3.0)

        # Combine with weights
        total_weight = yield_weight + credit_weight + dollar_weight + vol_weight

        composite = (
            yield_weight * yield_score
            + credit_weight * credit_score
            + dollar_weight * dollar_score
            + vol_weight * vol_score
        ) / total_weight

        # Apply final smoothing
        if smoothing > 1:
            composite = composite.rolling(window=smoothing, min_periods=1).mean()

        scores = composite.clip(-1, 1)

        metadata = {
            "weights": {
                "yield": yield_weight,
                "credit": credit_weight,
                "dollar": dollar_weight,
                "vol": vol_weight,
            },
            "smoothing": smoothing,
            "component_means": {
                "yield": yield_score.mean(),
                "credit": credit_score.mean(),
                "dollar": dollar_score.mean(),
                "vol": vol_score.mean(),
            },
            "composite_mean": scores.mean(),
            "composite_current": scores.iloc[-1] if len(scores) > 0 else None,
        }

        return SignalResult(scores=scores, metadata=metadata)
