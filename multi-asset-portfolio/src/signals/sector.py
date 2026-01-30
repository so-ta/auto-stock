"""
Sector Signals - Sector-based trading indicators.

Implements sector-based signals including:
- SectorMomentum: Sector ETF momentum for sector rotation strategies
- SectorRelativeStrength: Relative strength within sector (outperformers)
- SectorBreadth: Sector breadth (advance/decline ratio)

All outputs are normalized to [-1, +1] using tanh compression.

Sector ETF Mapping (S&P 500 GICS Sectors):
- Technology: XLK
- Financials: XLF
- Energy: XLE
- Healthcare: XLV
- Consumer Discretionary: XLY
- Consumer Staples: XLP
- Industrials: XLI
- Materials: XLB
- Utilities: XLU
- Real Estate: XLRE
- Communication Services: XLC
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


# GICS Sector to ETF mapping
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Reverse mapping: ETF to Sector
ETF_TO_SECTOR: Dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# All sector ETF tickers
SECTOR_ETF_TICKERS: List[str] = list(SECTOR_ETFS.values())


@SignalRegistry.register(
    "sector_momentum",
    category="sector",
    description="Sector momentum based on sector ETF performance",
    tags=["sector", "momentum", "rotation"],
)
class SectorMomentumSignal(Signal):
    """
    Sector Momentum Signal.

    Calculates momentum of sector ETFs for sector rotation strategies.
    When applied to a sector ETF, measures its own momentum.
    When applied to individual stocks, requires sector_data parameter with
    sector ETF prices.

    This signal is useful for:
    - Sector rotation strategies (overweight strong sectors)
    - Relative sector strength comparison
    - Timing sector entry/exit

    Parameters:
        lookback: Momentum lookback period (searchable, 21-126)
        skip_recent: Days to skip to avoid mean reversion (fixed, 5)
        use_ema: Use EMA instead of simple returns (fixed, True)
        ema_span: EMA span if use_ema is True (fixed, 20)
        scale: Tanh scaling factor (fixed, 5.0)

    Formula:
        if use_ema:
            momentum = (ema(close) - ema(close).shift(lookback)) / ema(close).shift(lookback)
        else:
            momentum = (close - close.shift(lookback + skip)) / close.shift(lookback + skip)
        score = tanh(momentum * scale)

    Example:
        # For sector ETF directly
        signal = SectorMomentumSignal(lookback=63)
        result = signal.compute(xlk_data)

        # For individual stock with sector context
        signal = SectorMomentumSignal(lookback=63)
        result = signal.compute(aapl_data, sector_data={"XLK": xlk_prices})
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=63,
                searchable=True,
                min_value=21,
                max_value=126,
                step=21,
                description="Momentum lookback period (days)",
            ),
            ParameterSpec(
                name="skip_recent",
                default=5,
                searchable=False,
                min_value=0,
                max_value=21,
                description="Days to skip (avoid short-term reversal)",
            ),
            ParameterSpec(
                name="use_ema",
                default=True,
                searchable=False,
                description="Use EMA smoothing for momentum calculation",
            ),
            ParameterSpec(
                name="ema_span",
                default=20,
                searchable=False,
                min_value=5,
                max_value=50,
                description="EMA span for smoothing",
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

    def compute(
        self,
        data: pd.DataFrame,
        sector_data: Optional[Dict[str, pd.Series]] = None,
        sector: Optional[str] = None,
    ) -> SignalResult:
        """
        Compute sector momentum signal.

        Args:
            data: OHLCV DataFrame for the asset
            sector_data: Optional dict mapping sector ETF tickers to price Series
                        Used when computing signal for individual stocks
            sector: Optional sector name for the asset (e.g., "Technology")
                   Required when sector_data is provided

        Returns:
            SignalResult with momentum scores in [-1, +1]
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        skip_recent = self._params["skip_recent"]
        use_ema = self._params["use_ema"]
        ema_span = self._params["ema_span"]
        scale = self._params["scale"]

        # Determine which price series to use for momentum
        if sector_data is not None and sector is not None:
            # Use sector ETF prices instead of individual stock
            sector_etf = SECTOR_ETFS.get(sector)
            if sector_etf and sector_etf in sector_data:
                close = sector_data[sector_etf]
            else:
                # Fallback to asset's own price
                close = data["close"]
        else:
            # Use the asset's own price
            close = data["close"]

        # Calculate momentum
        if use_ema:
            ema = close.ewm(span=ema_span, min_periods=ema_span // 2).mean()
            ema_lagged = ema.shift(lookback)
            momentum = (ema - ema_lagged) / ema_lagged.replace(0, np.nan)
        else:
            close_lagged = close.shift(skip_recent)
            close_past = close.shift(lookback + skip_recent)
            momentum = (close_lagged - close_past) / close_past.replace(0, np.nan)

        momentum = momentum.fillna(0)

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(momentum, scale=scale)

        # Align index with input data
        if len(scores) != len(data):
            scores = scores.reindex(data.index).fillna(0)

        metadata = {
            "lookback": lookback,
            "skip_recent": skip_recent,
            "use_ema": use_ema,
            "sector": sector,
            "momentum_mean": momentum.mean(),
            "momentum_std": momentum.std(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "sector_relative_strength",
    category="sector",
    description="Relative strength of asset within its sector",
    tags=["sector", "relative", "ranking"],
)
class SectorRelativeStrengthSignal(Signal):
    """
    Sector Relative Strength Signal.

    Measures how an asset performs relative to its sector benchmark.
    Assets outperforming their sector get positive scores, underperformers
    get negative scores.

    This signal identifies sector leaders and laggards, useful for:
    - Picking best-in-class within each sector
    - Avoiding sector laggards
    - Pair trades within sectors

    Parameters:
        lookback: Return comparison period (searchable, 21-63)
        smooth_period: Smoothing period for relative strength (fixed, 5)
        use_log_returns: Use log returns instead of simple (fixed, False)
        scale: Tanh scaling factor (fixed, 3.0)

    Formula:
        asset_return = (close - close.shift(lookback)) / close.shift(lookback)
        sector_return = (sector_close - sector_close.shift(lookback)) / sector_close.shift(lookback)
        relative_strength = asset_return - sector_return
        smoothed = relative_strength.rolling(smooth_period).mean()
        score = tanh(smoothed * scale)

    Example:
        signal = SectorRelativeStrengthSignal(lookback=21)
        result = signal.compute(
            aapl_data,
            sector_data={"XLK": xlk_prices},
            sector="Technology"
        )
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=21,
                searchable=True,
                min_value=21,
                max_value=63,
                step=21,
                description="Return comparison period (days)",
            ),
            ParameterSpec(
                name="smooth_period",
                default=5,
                searchable=False,
                min_value=1,
                max_value=21,
                description="Smoothing period for relative strength",
            ),
            ParameterSpec(
                name="use_log_returns",
                default=False,
                searchable=False,
                description="Use log returns instead of simple returns",
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

    def compute(
        self,
        data: pd.DataFrame,
        sector_data: Optional[Dict[str, pd.Series]] = None,
        sector: Optional[str] = None,
    ) -> SignalResult:
        """
        Compute sector relative strength signal.

        Args:
            data: OHLCV DataFrame for the asset
            sector_data: Dict mapping sector ETF tickers to price Series
                        Required for relative strength calculation
            sector: Sector name for the asset (e.g., "Technology")
                   Required for relative strength calculation

        Returns:
            SignalResult with relative strength scores in [-1, +1]
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        smooth_period = self._params["smooth_period"]
        use_log_returns = self._params["use_log_returns"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate asset returns
        if use_log_returns:
            asset_return = np.log(close / close.shift(lookback))
        else:
            asset_return = (close - close.shift(lookback)) / close.shift(lookback).replace(
                0, np.nan
            )

        # Calculate sector returns if sector data is provided
        sector_return = pd.Series(0.0, index=close.index)
        sector_etf_used = None

        if sector_data is not None and sector is not None:
            sector_etf = SECTOR_ETFS.get(sector)
            if sector_etf and sector_etf in sector_data:
                sector_close = sector_data[sector_etf]
                # Align index
                sector_close = sector_close.reindex(close.index).ffill()

                if use_log_returns:
                    sector_return = np.log(
                        sector_close / sector_close.shift(lookback)
                    )
                else:
                    sector_return = (
                        sector_close - sector_close.shift(lookback)
                    ) / sector_close.shift(lookback).replace(0, np.nan)

                sector_etf_used = sector_etf

        # Calculate relative strength
        relative_strength = asset_return - sector_return
        relative_strength = relative_strength.fillna(0)

        # Smooth the relative strength
        smoothed = relative_strength.rolling(
            window=smooth_period, min_periods=1
        ).mean()

        # Normalize to [-1, +1]
        scores = self.normalize_zscore_tanh(smoothed, lookback=lookback, scale=scale / 5)

        metadata = {
            "lookback": lookback,
            "smooth_period": smooth_period,
            "sector": sector,
            "sector_etf": sector_etf_used,
            "asset_return_mean": asset_return.mean(),
            "sector_return_mean": sector_return.mean(),
            "relative_strength_mean": relative_strength.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "sector_breadth",
    category="sector",
    description="Sector breadth indicator based on advance/decline ratio",
    tags=["sector", "breadth", "market_health"],
)
class SectorBreadthSignal(Signal):
    """
    Sector Breadth Signal.

    Measures the internal health of a sector by tracking the ratio of
    advancing to declining assets within the sector.

    High breadth (many advancers) indicates healthy sector momentum.
    Low breadth (few advancers) indicates weak sector momentum or distribution.

    This signal is useful for:
    - Confirming sector trends
    - Detecting sector rotation early
    - Identifying divergences (price up, breadth down = warning)

    Parameters:
        lookback: Period for advance/decline calculation (searchable, 5-21)
        threshold: Return threshold for advance/decline classification (fixed, 0.0)
        smooth_period: Smoothing period for breadth ratio (fixed, 5)
        scale: Tanh scaling factor (fixed, 2.0)

    Formula:
        returns = close.pct_change(lookback)
        is_advancing = returns > threshold
        breadth_ratio = advancing_count / total_count
        centered = (breadth_ratio - 0.5) * 2  # Center around 0
        smoothed = centered.rolling(smooth_period).mean()
        score = tanh(smoothed * scale)

    Note:
        For proper breadth calculation, provide multiple assets' data
        via the constituents_data parameter. Without it, the signal
        uses the asset's own advance/decline status.

    Example:
        # With sector constituents
        signal = SectorBreadthSignal(lookback=5)
        result = signal.compute(
            xlk_data,  # Sector ETF for reference
            constituents_data={
                "AAPL": aapl_closes,
                "MSFT": msft_closes,
                "GOOGL": googl_closes,
                ...
            }
        )
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=True,
                min_value=5,
                max_value=21,
                step=5,
                description="Period for advance/decline calculation (days)",
            ),
            ParameterSpec(
                name="threshold",
                default=0.0,
                searchable=False,
                min_value=-0.05,
                max_value=0.05,
                description="Return threshold for advance classification",
            ),
            ParameterSpec(
                name="smooth_period",
                default=5,
                searchable=False,
                min_value=1,
                max_value=21,
                description="Smoothing period for breadth ratio",
            ),
            ParameterSpec(
                name="scale",
                default=2.0,
                searchable=False,
                min_value=1.0,
                max_value=5.0,
                description="Tanh scaling factor",
            ),
        ]

    def compute(
        self,
        data: pd.DataFrame,
        constituents_data: Optional[Dict[str, pd.Series]] = None,
    ) -> SignalResult:
        """
        Compute sector breadth signal.

        Args:
            data: OHLCV DataFrame (typically sector ETF)
            constituents_data: Optional dict mapping ticker to close prices
                             for all sector constituents

        Returns:
            SignalResult with breadth scores in [-1, +1]
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        threshold = self._params["threshold"]
        smooth_period = self._params["smooth_period"]
        scale = self._params["scale"]

        if constituents_data and len(constituents_data) > 1:
            # Calculate breadth from constituents
            # Create DataFrame of returns for all constituents
            returns_dict = {}
            for ticker, prices in constituents_data.items():
                prices_aligned = prices.reindex(data.index).ffill()
                ret = prices_aligned.pct_change(periods=lookback)
                returns_dict[ticker] = ret

            returns_df = pd.DataFrame(returns_dict)

            # Calculate advance/decline
            advancing = (returns_df > threshold).sum(axis=1)
            total = returns_df.notna().sum(axis=1)

            # Avoid division by zero
            total = total.replace(0, np.nan)
            breadth_ratio = advancing / total
            breadth_ratio = breadth_ratio.fillna(0.5)

            constituents_count = len(constituents_data)
        else:
            # Single asset: use its own return as binary breadth
            close = data["close"]
            returns = close.pct_change(periods=lookback)
            breadth_ratio = (returns > threshold).astype(float)

            # Apply rolling mean to simulate breadth accumulation
            breadth_ratio = breadth_ratio.rolling(
                window=lookback, min_periods=1
            ).mean()

            constituents_count = 1

        # Center around 0 (0.5 = neutral)
        centered = (breadth_ratio - 0.5) * 2

        # Smooth the breadth
        smoothed = centered.rolling(window=smooth_period, min_periods=1).mean()

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(smoothed, scale=scale)

        metadata = {
            "lookback": lookback,
            "threshold": threshold,
            "smooth_period": smooth_period,
            "constituents_count": constituents_count,
            "breadth_ratio_mean": breadth_ratio.mean(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)


def get_sector_for_ticker(ticker: str, sector_mapping: Dict[str, str]) -> Optional[str]:
    """
    Get sector name for a given ticker.

    Args:
        ticker: Stock ticker symbol
        sector_mapping: Dict mapping tickers to sector names

    Returns:
        Sector name or None if not found
    """
    return sector_mapping.get(ticker)


def get_sector_etf(sector: str) -> Optional[str]:
    """
    Get sector ETF ticker for a given sector name.

    Args:
        sector: Sector name (e.g., "Technology")

    Returns:
        Sector ETF ticker (e.g., "XLK") or None if not found
    """
    return SECTOR_ETFS.get(sector)


def get_all_sector_etfs() -> List[str]:
    """
    Get list of all sector ETF tickers.

    Returns:
        List of sector ETF tickers
    """
    return SECTOR_ETF_TICKERS.copy()
