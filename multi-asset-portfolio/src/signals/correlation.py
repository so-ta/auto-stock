"""
Correlation-based Signals - Market regime and dispersion indicators.

Implements correlation-based signals including:
- Correlation Regime: Detects correlation regime changes across assets
- Return Dispersion: Cross-sectional return dispersion

All outputs are normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


# Reference assets for correlation regime detection
REFERENCE_ASSETS = ["SPY", "TLT", "GLD", "UUP"]


@SignalRegistry.register(
    "correlation_regime",
    category="correlation",
    description="Correlation regime change detection signal",
    tags=["regime", "risk", "correlation"],
)
class CorrelationRegimeSignal(Signal):
    """
    Correlation Regime Signal.

    Detects changes in market correlation regimes. High correlation
    typically indicates risk-off environment (assets move together).

    Parameters:
        lookback: Rolling correlation lookback (searchable, 20-120)
        reference_ticker: Reference asset for correlation (fixed, default 'SPY')
        threshold_high: High correlation threshold (fixed, default 0.7)
        threshold_low: Low correlation threshold (fixed, default 0.3)
        scale: Tanh scaling factor (fixed, default 2.0)

    Formula:
        rolling_corr = corr(asset_returns, reference_returns, lookback)
        corr_change = rolling_corr - rolling_corr.shift(lookback//2)
        score = -tanh(corr_change * scale)  # High corr increase = risk-off = negative

    Note:
        This signal requires reference asset data to be available in the DataFrame
        or uses the asset's own returns for autocorrelation analysis.

    Example:
        signal = CorrelationRegimeSignal(lookback=60)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=60,
                searchable=True,
                min_value=20,
                max_value=120,
                step=20,
                description="Rolling correlation calculation period",
            ),
            ParameterSpec(
                name="change_lookback",
                default=20,
                searchable=False,
                min_value=5,
                max_value=60,
                description="Lookback for correlation change detection",
            ),
            ParameterSpec(
                name="threshold_high",
                default=0.7,
                searchable=False,
                min_value=0.5,
                max_value=0.9,
                description="High correlation threshold",
            ),
            ParameterSpec(
                name="threshold_low",
                default=0.3,
                searchable=False,
                min_value=0.1,
                max_value=0.5,
                description="Low correlation threshold",
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

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": [20, 40, 60, 90, 120],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        change_lookback = self._params["change_lookback"]
        threshold_high = self._params["threshold_high"]
        threshold_low = self._params["threshold_low"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Use rolling autocorrelation (lag-1) as a proxy for regime
        # In high correlation regimes, returns tend to be more persistent
        lagged_returns = returns.shift(1)

        rolling_corr = returns.rolling(window=lookback, min_periods=lookback // 2).corr(
            lagged_returns
        )

        # Fill NaN with 0 (neutral correlation)
        rolling_corr = rolling_corr.fillna(0)

        # Calculate correlation change
        corr_change = rolling_corr - rolling_corr.shift(change_lookback)
        corr_change = corr_change.fillna(0)

        # Also compute correlation level signal
        # High absolute correlation = unstable regime
        abs_corr = rolling_corr.abs()

        # Combine: penalize both high correlation and rising correlation
        # High corr or rising corr = risk-off = negative score
        raw_signal = -(abs_corr + corr_change)

        # Normalize to [-1, +1]
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate regime statistics
        high_corr_pct = (rolling_corr.abs() > threshold_high).mean() * 100
        low_corr_pct = (rolling_corr.abs() < threshold_low).mean() * 100

        metadata = {
            "lookback": lookback,
            "change_lookback": change_lookback,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "scale": scale,
            "corr_mean": rolling_corr.mean(),
            "corr_std": rolling_corr.std(),
            "high_corr_pct": high_corr_pct,
            "low_corr_pct": low_corr_pct,
            "corr_change_mean": corr_change.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "return_dispersion",
    category="correlation",
    description="Cross-sectional return dispersion signal",
    tags=["dispersion", "regime", "volatility"],
)
class ReturnDispersionSignal(Signal):
    """
    Return Dispersion Signal.

    Measures the cross-sectional dispersion of returns within an asset's
    own return distribution. High dispersion indicates active stock-picking
    environment; low dispersion indicates macro-driven market.

    For single-asset data, uses rolling return volatility as a proxy.
    For multi-asset data, uses cross-sectional standard deviation.

    Parameters:
        lookback: Dispersion calculation lookback (searchable, 10-60)
        normalize_lookback: Lookback for normalization (fixed, default 60)
        scale: Tanh scaling factor (fixed, default 1.0)

    Formula:
        dispersion = rolling_std(returns, lookback)
        dispersion_zscore = (dispersion - dispersion_ma) / dispersion_std
        score = tanh(dispersion_zscore * scale)

    High dispersion score is bullish (more opportunities for alpha).

    Example:
        signal = ReturnDispersionSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=10,
                max_value=60,
                step=10,
                description="Dispersion calculation period",
            ),
            ParameterSpec(
                name="normalize_lookback",
                default=60,
                searchable=False,
                min_value=20,
                max_value=120,
                description="Lookback for dispersion normalization",
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
            "lookback": [10, 20, 30, 40, 60],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        normalize_lookback = self._params["normalize_lookback"]
        scale = self._params["scale"]

        close = data["close"]
        returns = close.pct_change()

        # Calculate rolling dispersion (volatility of returns)
        dispersion = returns.rolling(window=lookback, min_periods=lookback // 2).std()

        # Normalize dispersion relative to its historical distribution
        disp_ma = dispersion.rolling(window=normalize_lookback, min_periods=1).mean()
        disp_std = dispersion.rolling(window=normalize_lookback, min_periods=1).std()
        disp_std = disp_std.replace(0, np.nan).ffill().fillna(1)

        dispersion_zscore = (dispersion - disp_ma) / disp_std

        # Fill NaN
        dispersion_zscore = dispersion_zscore.fillna(0)

        # High dispersion = more alpha opportunities = positive signal
        # Low dispersion = correlated market = may need to adjust
        scores = self.normalize_tanh(dispersion_zscore, scale=scale)

        # Calculate statistics
        high_disp_pct = (dispersion_zscore > 1.0).mean() * 100
        low_disp_pct = (dispersion_zscore < -1.0).mean() * 100

        metadata = {
            "lookback": lookback,
            "normalize_lookback": normalize_lookback,
            "scale": scale,
            "dispersion_mean": dispersion.mean(),
            "dispersion_std": dispersion.std(),
            "zscore_mean": dispersion_zscore.mean(),
            "high_dispersion_pct": high_disp_pct,
            "low_dispersion_pct": low_disp_pct,
        }

        return SignalResult(scores=scores, metadata=metadata)


class CrossAssetCorrelationSignal(Signal):
    """
    Cross-Asset Correlation Signal.

    Computes correlation between multiple assets and generates signals
    based on correlation regime changes.

    This signal requires a multi-asset DataFrame with price columns for
    each asset. Use with portfolio-level data.

    Parameters:
        lookback: Correlation calculation lookback (searchable, 20-120)
        assets: List of asset column names to include
        scale: Tanh scaling factor (fixed, default 2.0)

    Note:
        Not registered in SignalRegistry as it requires special data format.
        Use directly when working with multi-asset portfolios.

    Example:
        signal = CrossAssetCorrelationSignal(
            lookback=60,
            assets=['SPY', 'TLT', 'GLD']
        )
        result = signal.compute(multi_asset_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=60,
                searchable=True,
                min_value=20,
                max_value=120,
                step=20,
                description="Rolling correlation period",
            ),
            ParameterSpec(
                name="assets",
                default=REFERENCE_ASSETS,
                searchable=False,
                description="List of asset columns to analyze",
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
        lookback = self._params["lookback"]
        assets = self._params["assets"]
        scale = self._params["scale"]

        # Check which assets are available
        available_assets = [a for a in assets if a in data.columns]

        if len(available_assets) < 2:
            # Fallback to single-asset analysis
            close = data["close"]
            returns = close.pct_change()
            avg_corr = returns.rolling(window=lookback).corr(returns.shift(1))
            avg_corr = avg_corr.fillna(0)
        else:
            # Calculate pairwise correlations
            returns_df = data[available_assets].pct_change()

            # Rolling average correlation across all pairs
            rolling_corrs = []
            for i, asset1 in enumerate(available_assets):
                for asset2 in available_assets[i + 1 :]:
                    pair_corr = (
                        returns_df[asset1]
                        .rolling(window=lookback, min_periods=lookback // 2)
                        .corr(returns_df[asset2])
                    )
                    rolling_corrs.append(pair_corr)

            # Average correlation
            avg_corr = pd.concat(rolling_corrs, axis=1).mean(axis=1)
            avg_corr = avg_corr.fillna(0)

        # High correlation = risk-off = negative signal
        raw_signal = -avg_corr

        scores = self.normalize_tanh(raw_signal, scale=scale)

        metadata = {
            "lookback": lookback,
            "assets_used": available_assets,
            "num_assets": len(available_assets),
            "avg_correlation_mean": avg_corr.mean(),
            "avg_correlation_std": avg_corr.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)
