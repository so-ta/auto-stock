"""
Short-Term Reversal Signal - Counter-trend signal based on recent returns.

Academic Background:
- Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns"
- Short-term (weekly/monthly) returns tend to reverse
- Market maker inventory adjustment / liquidity provision
- Nagel (2012) - Relationship with liquidity supply returns

Key characteristics:
- Past winners tend to underperform (sell signal)
- Past losers tend to outperform (buy signal)
- Works best with high-liquidity large-cap stocks
- Transaction costs are a major concern

Output normalized to [-1, +1] using tanh compression.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "short_term_reversal",
    category="reversal",
    description="Short-term reversal signal based on recent returns (Jegadeesh 1990)",
    tags=["contrarian", "liquidity", "mean_reversion", "cross_sectional"],
)
class ShortTermReversalSignal(Signal):
    """
    Short-Term Reversal Signal.

    Contrarian strategy that buys past losers and sells past winners.
    Based on the empirical observation that short-term returns tend to reverse.

    Academic Foundation:
    - Jegadeesh (1990): "Evidence of Predictable Behavior of Security Returns"
    - Returns over 1-week to 1-month horizons show negative autocorrelation
    - Explanation: Market maker inventory adjustment, liquidity provision
    - Nagel (2012): Links reversal returns to liquidity supply

    Signal Computation:
    1. Calculate past N-day returns
    2. Cross-sectionally normalize (z-score across assets)
    3. Invert (multiply by -1)
    4. Optional: Volume-weight (higher volume = stronger signal)

    Parameters:
        lookback: Return calculation period in days (default: 5)
        holding_period: Expected holding period (for reference, default: 5)
        use_volume_weight: Weight by volume (default: True)
        volume_lookback: Volume averaging period (default: 20)
        scale: Tanh scaling factor (default: 0.5)

    Cautions:
    - Transaction costs significantly impact profitability
    - Recommended for large-cap, liquid stocks only
    - Weekly frequency often more effective than daily

    Example:
        signal = ShortTermReversalSignal(lookback=5, use_volume_weight=True)
        result = signal.compute(price_data)

        # For cross-sectional use
        signal = ShortTermReversalSignal(lookback=5)
        result = signal.compute_cross_sectional(multi_asset_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=5,
                searchable=True,
                min_value=3,
                max_value=20,
                step=1,
                description="Return calculation period (days)",
            ),
            ParameterSpec(
                name="holding_period",
                default=5,
                searchable=False,
                min_value=1,
                max_value=20,
                description="Expected holding period (reference only)",
            ),
            ParameterSpec(
                name="use_volume_weight",
                default=True,
                searchable=True,
                description="Weight signal by trading volume",
            ),
            ParameterSpec(
                name="volume_lookback",
                default=20,
                searchable=False,
                min_value=5,
                max_value=60,
                description="Volume averaging period for weighting",
            ),
            ParameterSpec(
                name="scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=2.0,
                description="Tanh scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization.

        Covers:
        - lookback: [3, 5, 10, 20] (week to month)
        - use_volume_weight: [True, False]

        Total combinations: 4 * 2 = 8
        """
        return {
            "lookback": [3, 5, 10, 20],
            "use_volume_weight": [True, False],
        }

    def _calculate_returns(self, close: pd.Series, lookback: int) -> pd.Series:
        """
        Calculate N-day returns.

        Args:
            close: Close price series
            lookback: Number of days for return calculation

        Returns:
            N-day returns series
        """
        returns = close.pct_change(periods=lookback)
        return returns

    def _calculate_volume_weight(
        self, volume: pd.Series, lookback: int
    ) -> pd.Series:
        """
        Calculate volume-based weight.

        Higher volume stocks get higher weight (more reliable reversal signal).
        Normalized to have mean=1.

        Args:
            volume: Volume series
            lookback: Averaging period

        Returns:
            Volume weight series (mean-normalized)
        """
        # Calculate average volume
        avg_volume = volume.rolling(window=lookback, min_periods=1).mean()

        # Normalize to mean=1
        global_mean = avg_volume.mean()
        if global_mean > 0:
            weight = avg_volume / global_mean
        else:
            weight = pd.Series(1.0, index=volume.index)

        # Clip extreme values
        weight = weight.clip(lower=0.1, upper=5.0)

        return weight

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute short-term reversal signal for a single asset.

        For single-asset use, this computes the time-series reversal signal.
        For cross-sectional use, see compute_cross_sectional().

        Args:
            data: DataFrame with columns ['close', 'volume'] (volume optional)
                  Index should be DatetimeIndex

        Returns:
            SignalResult with reversal scores in [-1, +1]
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        use_volume_weight = self._params["use_volume_weight"]
        volume_lookback = self._params["volume_lookback"]
        scale = self._params["scale"]

        close = data["close"]

        # 1. Calculate past returns
        past_returns = self._calculate_returns(close, lookback)

        # 2. Time-series z-score (for single asset)
        # Note: For cross-sectional, use compute_cross_sectional()
        rolling_mean = past_returns.rolling(window=60, min_periods=10).mean()
        rolling_std = past_returns.rolling(window=60, min_periods=10).std()
        rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)

        z_score = (past_returns - rolling_mean) / rolling_std

        # 3. Invert for reversal (past winners -> sell, past losers -> buy)
        raw_signal = -z_score

        # 4. Apply volume weight if enabled
        if use_volume_weight and "volume" in data.columns:
            volume_weight = self._calculate_volume_weight(
                data["volume"], volume_lookback
            )
            raw_signal = raw_signal * volume_weight

        # 5. Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate metadata
        positive_return_pct = (past_returns > 0).mean() * 100
        metadata = {
            "lookback": lookback,
            "use_volume_weight": use_volume_weight,
            "scale": scale,
            "past_return_mean": past_returns.mean(),
            "past_return_std": past_returns.std(),
            "positive_return_pct": positive_return_pct,
            "z_score_mean": z_score.mean(),
            "z_score_std": z_score.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)

    def compute_cross_sectional(
        self,
        multi_asset_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, SignalResult]:
        """
        Compute cross-sectional short-term reversal signal.

        This is the academically standard approach:
        - Calculate returns for all assets
        - Cross-sectionally normalize (z-score across assets at each time point)
        - Invert for reversal

        Args:
            multi_asset_data: Dictionary mapping ticker to price DataFrame

        Returns:
            Dictionary mapping ticker to SignalResult
        """
        lookback = self._params["lookback"]
        use_volume_weight = self._params["use_volume_weight"]
        volume_lookback = self._params["volume_lookback"]
        scale = self._params["scale"]

        # 1. Calculate returns for all assets
        returns_dict = {}
        volume_weight_dict = {}

        for ticker, data in multi_asset_data.items():
            if "close" not in data.columns:
                continue

            returns_dict[ticker] = self._calculate_returns(data["close"], lookback)

            if use_volume_weight and "volume" in data.columns:
                volume_weight_dict[ticker] = self._calculate_volume_weight(
                    data["volume"], volume_lookback
                )

        if not returns_dict:
            return {}

        # 2. Create returns DataFrame for cross-sectional normalization
        returns_df = pd.DataFrame(returns_dict)

        # 3. Cross-sectional z-score (across assets at each time point)
        cross_mean = returns_df.mean(axis=1)
        cross_std = returns_df.std(axis=1)
        cross_std = cross_std.replace(0, np.nan).ffill().fillna(1)

        z_scores_df = returns_df.sub(cross_mean, axis=0).div(cross_std, axis=0)

        # 4. Invert for reversal
        raw_signals_df = -z_scores_df

        # 5. Apply volume weight if enabled
        if use_volume_weight and volume_weight_dict:
            for ticker in raw_signals_df.columns:
                if ticker in volume_weight_dict:
                    raw_signals_df[ticker] = (
                        raw_signals_df[ticker] * volume_weight_dict[ticker]
                    )

        # 6. Normalize to [-1, +1] using tanh
        scores_df = np.tanh(raw_signals_df * scale)

        # 7. Create results
        results = {}
        for ticker in scores_df.columns:
            metadata = {
                "lookback": lookback,
                "use_volume_weight": use_volume_weight,
                "scale": scale,
                "method": "cross_sectional",
                "past_return_mean": returns_df[ticker].mean(),
                "z_score_mean": z_scores_df[ticker].mean(),
            }
            results[ticker] = SignalResult(
                scores=scores_df[ticker],
                metadata=metadata,
            )

        return results


@SignalRegistry.register(
    "short_term_reversal_weekly",
    category="reversal",
    description="Weekly short-term reversal signal (stronger effect)",
    tags=["contrarian", "weekly", "mean_reversion"],
)
class WeeklyShortTermReversalSignal(ShortTermReversalSignal):
    """
    Weekly Short-Term Reversal Signal.

    Same as ShortTermReversalSignal but with weekly default parameters.
    Weekly frequency often shows stronger reversal effects due to:
    - Lower transaction costs relative to signal strength
    - More time for market maker inventory adjustment

    Default lookback: 5 days (1 week)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        specs = super().parameter_specs()
        # Override default lookback to 5 (weekly)
        for spec in specs:
            if spec.name == "lookback":
                return [
                    ParameterSpec(
                        name="lookback",
                        default=5,
                        searchable=True,
                        min_value=3,
                        max_value=10,
                        step=1,
                        description="Return calculation period (days, weekly focus)",
                    )
                ] + [s for s in specs if s.name != "lookback"]
        return specs


@SignalRegistry.register(
    "short_term_reversal_monthly",
    category="reversal",
    description="Monthly short-term reversal signal",
    tags=["contrarian", "monthly", "mean_reversion"],
)
class MonthlyShortTermReversalSignal(ShortTermReversalSignal):
    """
    Monthly Short-Term Reversal Signal.

    Same as ShortTermReversalSignal but with monthly default parameters.
    Monthly frequency is the classic academic setup from Jegadeesh (1990).

    Default lookback: 20 days (1 month)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        specs = super().parameter_specs()
        # Override default lookback to 20 (monthly)
        for spec in specs:
            if spec.name == "lookback":
                return [
                    ParameterSpec(
                        name="lookback",
                        default=20,
                        searchable=True,
                        min_value=15,
                        max_value=30,
                        step=5,
                        description="Return calculation period (days, monthly focus)",
                    )
                ] + [s for s in specs if s.name != "lookback"]
        return specs
