"""
52-Week High Momentum Signal.

Based on George & Hwang (2004) "The 52-Week High and Momentum Investing"
Journal of Finance.

Key findings:
- Price proximity to 52-week high is a better predictor than traditional momentum
- No long-term reversal effect (unlike standard momentum)
- Based on anchoring bias in behavioral finance
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@SignalRegistry.register(
    "fifty_two_week_high_momentum",
    category="momentum",
    description="52-week high momentum signal based on George & Hwang (2004)",
    tags=["momentum", "academic", "behavioral"],
)
class FiftyTwoWeekHighMomentumSignal(Signal):
    """
    52-Week High Momentum Signal.

    Measures the proximity of current price to its 52-week high.
    Stocks trading near their 52-week high tend to continue rising,
    while stocks far from their high tend to continue falling.

    Academic Reference:
        George, T. J., & Hwang, C. Y. (2004).
        "The 52-Week High and Momentum Investing."
        Journal of Finance, 59(5), 2145-2176.

    Theory:
        Anchoring bias causes investors to use the 52-week high as a
        reference point. When prices approach this anchor, investors
        are slow to update their expectations, creating predictable
        price patterns.

    Parameters:
        lookback: Rolling window for 52-week high calculation (searchable)
                  Default 252 (trading days in a year)
        smoothing: Smoothing period for noise reduction (searchable)
                   Default 5 days

    Formula:
        ratio = current_price / rolling_max(price, lookback)
        signal = ((ratio - 0.5) * 2).clip(-1, 1)

    Signal Interpretation:
        - Close to 1.0: Strong bullish (price at/near 52-week high)
        - Close to 0.0: Neutral
        - Close to -1.0: Strong bearish (price far from 52-week high)

    Example:
        signal = FiftyTwoWeekHighMomentumSignal(lookback=252, smoothing=5)
        result = signal.compute(price_data)
    """

    signal_name = "fifty_two_week_high_momentum"

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=252,
                searchable=True,
                min_value=63,  # ~3 months minimum
                max_value=504,  # ~2 years maximum
                step=21,  # ~1 month increments
                description="Rolling window for 52-week high (trading days)",
            ),
            ParameterSpec(
                name="smoothing",
                default=5,
                searchable=True,
                min_value=1,
                max_value=20,
                step=1,
                description="Smoothing period for noise reduction",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization.

        Lookback periods:
        - 126: ~6 months (half-year high)
        - 189: ~9 months
        - 252: ~12 months (standard 52-week high)

        Smoothing periods:
        - 1: No smoothing
        - 3, 5, 10: Various smoothing levels
        """
        return {
            "lookback": [126, 189, 252],
            "smoothing": [1, 3, 5, 10],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute the 52-week high momentum signal.

        Args:
            data: DataFrame with 'close' column and DatetimeIndex

        Returns:
            SignalResult with scores normalized to [-1, +1]
        """
        self.validate_input(data)

        lookback = self._params["lookback"]
        smoothing = self._params["smoothing"]

        close = data["close"]

        # Calculate rolling maximum (52-week high)
        rolling_high = close.rolling(window=lookback, min_periods=1).max()

        # Calculate ratio: current price / 52-week high
        # This ratio is in range (0, 1] where 1 means price is at high
        ratio = close / rolling_high

        # Handle potential division issues
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)

        # Apply smoothing if specified
        if smoothing > 1:
            ratio = ratio.rolling(window=smoothing, min_periods=1).mean()

        # Normalize to [-1, +1]
        # Original ratio: (0, 1] -> transform to [-1, 1]
        # Using formula: ((ratio - 0.5) * 2).clip(-1, 1)
        # When ratio = 1.0 (at high): score = 1.0
        # When ratio = 0.5: score = 0.0
        # When ratio = 0.0: score = -1.0
        scores = ((ratio - 0.5) * 2).clip(-1, 1)

        metadata = {
            "lookback": lookback,
            "smoothing": smoothing,
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "ratio_min": ratio.min(),
            "ratio_max": ratio.max(),
            "pct_at_high": (ratio > 0.95).mean(),  # % of time within 5% of high
            "pct_far_from_high": (ratio < 0.7).mean(),  # % of time >30% below high
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }

        return SignalResult(scores=scores, metadata=metadata)

    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input DataFrame.

        Extends base validation to check for sufficient data points.
        """
        super().validate_input(data)

        lookback = self._params["lookback"]
        if len(data) < lookback:
            # Warning only - still compute with available data
            pass  # min_periods=1 handles this gracefully
