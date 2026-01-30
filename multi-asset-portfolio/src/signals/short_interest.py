"""
Short Interest Signal - Academic-based contrarian indicator.

Implements short interest-based signals based on academic research:
- Rapach et al. (2016): Aggregate short interest predicts market returns
- Boehmer et al. (2008): Stock-level short interest predicts returns

Key Insight: High short interest → Future low returns (contrarian signal)

Data Source:
- FINRA API (free, biweekly updates)
- 2014+ archive available

Usage:
    signal = ShortInterestSignal(use_days_to_cover=True, zscore_normalize=True)
    result = signal.compute(price_data, short_interest_data)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "short_interest",
    category="sentiment",
    description="Short interest based contrarian signal (Rapach et al. 2016)",
    tags=["contrarian", "sentiment", "academic"],
)
class ShortInterestSignal(Signal):
    """
    Short Interest Signal.

    Generates contrarian signals based on short interest data.
    High short interest indicates bearish sentiment, which academically
    predicts future low returns.

    Academic Basis:
    - Rapach, Ringgenberg, Zhou (2016): "Short Interest and Aggregate
      Stock Returns" - Journal of Financial Economics
    - Boehmer, Jones, Zhang (2008): "Which Shorts Are Informed?"
      - Journal of Finance

    The signal is INVERTED: High SI → Negative signal (sell)
                          Low SI  → Positive signal (buy)

    Parameters:
        use_days_to_cover: Use days-to-cover ratio instead of raw SI (default: True)
        zscore_normalize: Apply z-score normalization (default: True)
        zscore_lookback: Lookback period for z-score calculation (default: 60)
        smoothing: Smoothing period (1 = no smoothing, default: 1)
        invert: Invert signal (False = high SI = negative, default: False)

    Formula:
        If use_days_to_cover:
            raw = Days to Cover (Short Interest / Avg Daily Volume)
        Else:
            raw = Short Interest Ratio (Short Interest / Float)

        If zscore_normalize:
            normalized = (raw - rolling_mean) / rolling_std

        signal = -tanh(normalized * scale)  # Inverted: high SI = negative

    Example:
        signal = ShortInterestSignal(use_days_to_cover=True)

        # With external short interest data
        result = signal.compute(
            price_data,
            short_interest=si_data,  # DataFrame with days_to_cover or si_ratio
        )
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="use_days_to_cover",
                default=True,
                searchable=True,
                description="Use days-to-cover ratio instead of SI ratio",
            ),
            ParameterSpec(
                name="zscore_normalize",
                default=True,
                searchable=True,
                description="Apply z-score normalization",
            ),
            ParameterSpec(
                name="zscore_lookback",
                default=60,
                searchable=True,
                min_value=20,
                max_value=252,
                step=20,
                description="Lookback period for z-score calculation (days)",
            ),
            ParameterSpec(
                name="smoothing",
                default=1,
                searchable=True,
                min_value=1,
                max_value=10,
                step=2,
                description="Smoothing period (1 = no smoothing)",
            ),
            ParameterSpec(
                name="scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=2.0,
                description="Tanh scaling factor",
            ),
            ParameterSpec(
                name="invert",
                default=False,
                searchable=False,
                description="Invert signal direction (True = high SI = positive)",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Parameter grid for optimization.

        Returns combinations of:
        - use_days_to_cover: True/False
        - zscore_normalize: True/False
        - smoothing: 1, 3, 5
        """
        return {
            "use_days_to_cover": [True, False],
            "zscore_normalize": [True, False],
            "smoothing": [1, 3, 5],
        }

    def compute(
        self,
        data: pd.DataFrame,
        short_interest: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> SignalResult:
        """
        Compute short interest signal.

        Args:
            data: Price DataFrame with ['close', 'volume'] columns
            short_interest: Optional external short interest data with columns:
                - 'date' or DatetimeIndex: Settlement dates
                - 'days_to_cover': Days to cover ratio
                - 'short_interest_ratio': SI / Float ratio
                - 'short_interest': Raw short interest (shares)
                If not provided, uses synthetic proxy from price/volume.

        Returns:
            SignalResult with scores in [-1, +1] range

        Note:
            If short_interest data is not provided, the signal generates
            a synthetic proxy using volume patterns. For production use,
            always provide actual FINRA short interest data.
        """
        self.validate_input(data)

        use_days_to_cover = self._params["use_days_to_cover"]
        zscore_normalize = self._params["zscore_normalize"]
        zscore_lookback = self._params["zscore_lookback"]
        smoothing = self._params["smoothing"]
        scale = self._params["scale"]
        invert = self._params["invert"]

        # Get short interest metric
        if short_interest is not None:
            si_metric = self._extract_si_metric(
                data, short_interest, use_days_to_cover
            )
        else:
            # Fallback: Use volume-based proxy (less accurate)
            si_metric = self._compute_si_proxy(data)

        # Apply smoothing
        if smoothing > 1:
            si_metric = si_metric.rolling(window=smoothing, min_periods=1).mean()

        # Z-score normalization
        if zscore_normalize:
            rolling_mean = si_metric.rolling(
                window=zscore_lookback, min_periods=min(zscore_lookback // 2, 10)
            ).mean()
            rolling_std = si_metric.rolling(
                window=zscore_lookback, min_periods=min(zscore_lookback // 2, 10)
            ).std()
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)
            normalized = (si_metric - rolling_mean) / rolling_std
        else:
            # Simple min-max normalization
            normalized = self.normalize_minmax_scaled(si_metric, lookback=zscore_lookback)

        # Apply tanh compression
        scores = self.normalize_tanh(normalized, scale=scale)

        # Invert signal (high SI = negative signal by default)
        if not invert:
            scores = -scores

        # Ensure scores are in [-1, +1]
        scores = scores.clip(-1, 1)

        metadata = {
            "use_days_to_cover": use_days_to_cover,
            "zscore_normalize": zscore_normalize,
            "zscore_lookback": zscore_lookback,
            "smoothing": smoothing,
            "scale": scale,
            "invert": invert,
            "si_metric_mean": si_metric.mean(),
            "si_metric_std": si_metric.std(),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
            "has_external_data": short_interest is not None,
        }

        return SignalResult(scores=scores, metadata=metadata)

    def _extract_si_metric(
        self,
        price_data: pd.DataFrame,
        short_interest: pd.DataFrame,
        use_days_to_cover: bool,
    ) -> pd.Series:
        """
        Extract short interest metric from external data.

        Args:
            price_data: Price DataFrame
            short_interest: Short interest DataFrame
            use_days_to_cover: Whether to use days-to-cover

        Returns:
            Series aligned with price_data index
        """
        # Ensure datetime index
        if "date" in short_interest.columns:
            si_df = short_interest.set_index("date")
        else:
            si_df = short_interest.copy()

        # Select metric
        if use_days_to_cover and "days_to_cover" in si_df.columns:
            metric = si_df["days_to_cover"]
        elif "short_interest_ratio" in si_df.columns:
            metric = si_df["short_interest_ratio"]
        elif "days_to_cover" in si_df.columns:
            metric = si_df["days_to_cover"]
        elif "short_interest" in si_df.columns:
            # Calculate ratio if volume available
            if "volume" in price_data.columns:
                avg_volume = price_data["volume"].rolling(20, min_periods=1).mean()
                # Align and calculate
                aligned_si = si_df["short_interest"].reindex(
                    price_data.index, method="ffill"
                )
                metric = aligned_si / avg_volume
            else:
                metric = si_df["short_interest"]
        else:
            raise ValueError(
                "short_interest DataFrame must contain one of: "
                "'days_to_cover', 'short_interest_ratio', or 'short_interest'"
            )

        # Align to price data index (forward fill - SI data is biweekly)
        aligned = metric.reindex(price_data.index, method="ffill")

        return aligned

    def _compute_si_proxy(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute a synthetic short interest proxy from price/volume data.

        This is a FALLBACK when actual FINRA data is not available.
        Uses volume patterns as a rough proxy for short activity.

        The proxy is based on the observation that short selling often
        accompanies increased volume on down days.

        Args:
            data: Price DataFrame with 'close' and 'volume' columns

        Returns:
            Synthetic SI proxy series
        """
        close = data["close"]

        if "volume" in data.columns:
            volume = data["volume"]
            returns = close.pct_change()

            # Volume on down days relative to average
            down_volume = volume.where(returns < 0, 0)
            avg_volume = volume.rolling(20, min_periods=1).mean()

            # Ratio of down-day volume to average (higher = more bearish pressure)
            down_volume_ratio = (
                down_volume.rolling(10, min_periods=1).sum()
                / (avg_volume.rolling(10, min_periods=1).sum() + 1e-8)
            )

            return down_volume_ratio
        else:
            # Without volume, use returns volatility as very rough proxy
            returns = close.pct_change()
            rolling_vol = returns.rolling(20, min_periods=1).std()
            return rolling_vol

    def compute_with_finra_client(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> SignalResult:
        """
        Compute signal using FINRA API client directly.

        Args:
            data: Price DataFrame
            symbol: Stock symbol for FINRA lookup

        Returns:
            SignalResult
        """
        try:
            from src.data.finra import FINRAClient

            client = FINRAClient()
            si_data = client.get_short_interest(
                symbol=symbol,
                start_date=data.index[0].to_pydatetime()
                if hasattr(data.index[0], "to_pydatetime")
                else data.index[0],
                end_date=data.index[-1].to_pydatetime()
                if hasattr(data.index[-1], "to_pydatetime")
                else data.index[-1],
            )

            if si_data.empty:
                return self.compute(data)  # Fallback to proxy

            si_data = si_data.set_index("settlement_date")
            return self.compute(data, short_interest=si_data)
        except Exception as e:
            import logging

            logging.warning(f"FINRA client failed: {e}, using proxy")
            return self.compute(data)


@SignalRegistry.register(
    "short_interest_change",
    category="sentiment",
    description="Short interest change momentum signal",
    tags=["contrarian", "sentiment", "momentum"],
)
class ShortInterestChangeSignal(Signal):
    """
    Short Interest Change Signal.

    Signals based on the rate of change in short interest.
    Rapid increases in short interest may precede price declines.
    Rapid decreases (short covering) may precede price increases.

    Parameters:
        change_period: Period for calculating SI change (default: 2, biweekly)
        zscore_normalize: Apply z-score normalization (default: True)
        zscore_lookback: Z-score lookback period (default: 60)

    Formula:
        si_change = (SI[t] - SI[t-period]) / SI[t-period]
        signal = -tanh(zscore(si_change))  # Inverted
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="change_period",
                default=2,
                searchable=True,
                min_value=1,
                max_value=6,
                step=1,
                description="Period for SI change calculation (in reports)",
            ),
            ParameterSpec(
                name="zscore_normalize",
                default=True,
                searchable=False,
                description="Apply z-score normalization",
            ),
            ParameterSpec(
                name="zscore_lookback",
                default=60,
                searchable=True,
                min_value=30,
                max_value=120,
                step=30,
                description="Z-score lookback period",
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

    def compute(
        self,
        data: pd.DataFrame,
        short_interest: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> SignalResult:
        """
        Compute short interest change signal.

        Args:
            data: Price DataFrame
            short_interest: Short interest DataFrame with 'short_interest' column

        Returns:
            SignalResult
        """
        self.validate_input(data)

        change_period = self._params["change_period"]
        zscore_normalize = self._params["zscore_normalize"]
        zscore_lookback = self._params["zscore_lookback"]
        scale = self._params["scale"]

        if short_interest is not None and "short_interest" in short_interest.columns:
            # Ensure datetime index
            if "date" in short_interest.columns:
                si_df = short_interest.set_index("date")
            elif "settlement_date" in short_interest.columns:
                si_df = short_interest.set_index("settlement_date")
            else:
                si_df = short_interest.copy()

            si = si_df["short_interest"]

            # Calculate percentage change
            si_change = si.pct_change(periods=change_period)

            # Align to price data index
            si_change = si_change.reindex(data.index, method="ffill")
        else:
            # Fallback: Use volume proxy
            close = data["close"]
            returns = close.pct_change()

            if "volume" in data.columns:
                volume = data["volume"]
                # Volume on down days as proxy
                down_vol = volume.where(returns < 0, 0)
                down_vol_ma = down_vol.rolling(10, min_periods=1).mean()
                si_change = down_vol_ma.pct_change(periods=change_period * 5)  # ~2 weeks
            else:
                # Use returns volatility change
                vol = returns.rolling(10).std()
                si_change = vol.pct_change(periods=change_period * 5)

        # Z-score normalization
        if zscore_normalize:
            si_change = self.normalize_zscore_tanh(
                si_change, lookback=zscore_lookback, scale=scale
            )
        else:
            si_change = self.normalize_tanh(si_change, scale=scale)

        # Invert (increasing SI = negative signal)
        scores = -si_change

        # Clip to [-1, +1]
        scores = scores.clip(-1, 1)

        metadata = {
            "change_period": change_period,
            "zscore_normalize": zscore_normalize,
            "zscore_lookback": zscore_lookback,
            "scale": scale,
            "has_external_data": short_interest is not None,
        }

        return SignalResult(scores=scores, metadata=metadata)


# Convenience function
def compute_short_interest_signal(
    data: pd.DataFrame,
    short_interest: Optional[pd.DataFrame] = None,
    use_days_to_cover: bool = True,
    zscore_normalize: bool = True,
) -> pd.Series:
    """
    Quick function to compute short interest signal.

    Args:
        data: Price DataFrame
        short_interest: Optional SI data
        use_days_to_cover: Use DTC ratio
        zscore_normalize: Apply z-score

    Returns:
        Signal scores series
    """
    signal = ShortInterestSignal(
        use_days_to_cover=use_days_to_cover,
        zscore_normalize=zscore_normalize,
    )
    result = signal.compute(data, short_interest=short_interest)
    return result.scores
