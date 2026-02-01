"""
Signal Base Class - Abstract base class for all signals.

Design Principles:
- Pure function design: input (historical data) -> output (score)
- Output normalization: [-1, +1] range using tanh compression
- Parameter separation: searchable vs fixed parameters
- Timeframe affinity: each signal declares its valid time horizons

Timeframe Affinity System:
- SHORT_TERM: 3-21 days (oscillators, short-term reversal)
- MEDIUM_TERM: 15-60 days (breakout, Bollinger)
- MULTI_TIMEFRAME: 5-252 days (momentum, volatility, Sharpe)
- LONG_TERM_ONLY: 63-504 days (52-week high, annual effects)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", bound="Signal")


# =============================================================================
# Timeframe Affinity System
# =============================================================================


class TimeframeAffinity(Enum):
    """Signal's timeframe affinity based on financial theory.

    Each signal has an inherent time horizon where it provides meaningful information.
    Using signals outside their natural timeframe often produces noise rather than signal.

    Academic references:
    - SHORT_TERM: Jegadeesh (1990) short-term reversal effect < 1 month
    - MEDIUM_TERM: Bollinger (1983) bands typically 10-50 days
    - MULTI_TIMEFRAME: Jegadeesh & Titman (1993) momentum 1-12 months
    - LONG_TERM_ONLY: George & Hwang (2004) 52-week high momentum
    """

    SHORT_TERM = "short_term"  # 3-21 days: oscillators, short-term reversal
    MEDIUM_TERM = "medium_term"  # 15-60 days: breakout, Bollinger
    MULTI_TIMEFRAME = "multi"  # 5-252 days: momentum, volatility
    LONG_TERM_ONLY = "long_only"  # 63-504 days: 52-week high, annual effects


# Default variant periods (must match settings.py PeriodVariantSettings)
DEFAULT_VARIANT_PERIODS: Dict[str, int] = {
    "short": 5,
    "medium": 20,
    "long": 60,
    "half_year": 126,
    "yearly": 252,
}


@dataclass(frozen=True)
class TimeframeConfig:
    """Configuration declaring a signal's timeframe affinity.

    Each signal should declare which timeframes/variants it supports based on
    the underlying financial theory. Variants outside the supported range will
    not be generated, preventing meaningless signal computations.

    Attributes:
        affinity: The signal's natural timeframe category
        min_period: Minimum valid period for this signal's parameters
        max_period: Maximum valid period for this signal's parameters
        supported_variants: List of variant names this signal supports
                           (subset of: short, medium, long, half_year, yearly)

    Example:
        RSI (7-21 days effective range per Wilder):
            TimeframeConfig(
                affinity=TimeframeAffinity.SHORT_TERM,
                min_period=7,
                max_period=21,
                supported_variants=["short", "medium"]  # 5d and 20d only
            )

        Momentum (1-12 months effective per J&T):
            TimeframeConfig(
                affinity=TimeframeAffinity.MULTI_TIMEFRAME,
                min_period=5,
                max_period=252,
                supported_variants=["short", "medium", "long", "half_year", "yearly"]
            )
    """

    affinity: TimeframeAffinity
    min_period: int
    max_period: int
    supported_variants: List[str]

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_period <= 0:
            raise ValueError("min_period must be positive")
        if self.max_period < self.min_period:
            raise ValueError("max_period must be >= min_period")
        if not self.supported_variants:
            raise ValueError("supported_variants must not be empty")

        valid_variants = {"short", "medium", "long", "half_year", "yearly"}
        for variant in self.supported_variants:
            if variant not in valid_variants:
                raise ValueError(
                    f"Invalid variant: {variant}. Must be one of {valid_variants}"
                )

    def supports_variant(self, variant_name: str) -> bool:
        """Check if this signal supports the given variant."""
        return variant_name in self.supported_variants

    def get_valid_period(self, variant_name: str) -> Optional[int]:
        """Get the period for a variant if supported, else None.

        Returns the standard variant period if the variant is supported AND
        the period falls within min_period/max_period bounds.
        """
        if variant_name not in self.supported_variants:
            return None
        period = DEFAULT_VARIANT_PERIODS.get(variant_name)
        if period is None:
            return None
        if not (self.min_period <= period <= self.max_period):
            return None
        return period


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a signal parameter."""

    name: str
    default: Any
    searchable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""

    def validate(self, value: Any) -> bool:
        """Validate a parameter value against constraints."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def search_range(self) -> Optional[np.ndarray]:
        """Generate search range for optimization."""
        if not self.searchable:
            return None
        if self.min_value is None or self.max_value is None:
            return None
        step = self.step or (self.max_value - self.min_value) / 10
        return np.arange(self.min_value, self.max_value + step, step)


@dataclass
class SignalResult:
    """Container for signal computation results."""

    scores: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the result contains valid scores."""
        return not self.scores.isna().all()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scores": self.scores.to_dict(),
            "metadata": self.metadata,
        }


class Signal(ABC):
    """
    Abstract base class for all trading signals.

    All signals must:
    1. Accept historical data (DataFrame with OHLCV)
    2. Output scores in [-1, +1] range
    3. Separate searchable from fixed parameters
    4. Be pure functions (no side effects)

    Example:
        class MySignal(Signal):
            @classmethod
            def parameter_specs(cls) -> List[ParameterSpec]:
                return [
                    ParameterSpec("lookback", 20, searchable=True, min_value=5, max_value=100),
                ]

            def compute(self, data: pd.DataFrame) -> SignalResult:
                # Implementation here
                pass
    """

    def __init__(self, **params: Any):
        """
        Initialize signal with parameters.

        Args:
            **params: Signal parameters (must match parameter_specs)
        """
        self._params = self._validate_and_set_params(params)
        self._name = self.__class__.__name__

    def _validate_and_set_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set parameters with defaults."""
        specs = {spec.name: spec for spec in self.parameter_specs()}
        validated = {}

        for name, spec in specs.items():
            value = params.get(name, spec.default)
            if not spec.validate(value):
                raise ValueError(
                    f"Parameter '{name}' value {value} is out of range "
                    f"[{spec.min_value}, {spec.max_value}]"
                )
            validated[name] = value

        # Check for unknown parameters
        unknown = set(params.keys()) - set(specs.keys())
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")

        return validated

    @property
    def name(self) -> str:
        """Signal name."""
        return self._name

    @property
    def params(self) -> Dict[str, Any]:
        """Current parameter values."""
        return self._params.copy()

    @classmethod
    @abstractmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        """
        Define parameter specifications for this signal.

        Returns:
            List of ParameterSpec defining all parameters
        """
        pass

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """
        Declare this signal's timeframe affinity.

        Override in subclasses to specify the natural time horizons for this signal.
        Signals with period/lookback parameters should declare which period variants
        they support based on financial theory.

        Default implementation: MULTI_TIMEFRAME supporting all variants (5-252 days).
        This is the most permissive setting and should be overridden for signals
        with more specific requirements.

        Returns:
            TimeframeConfig declaring affinity, period bounds, and supported variants

        Examples:
            # For RSI (Wilder's 14-day, effective 7-21)
            @classmethod
            def timeframe_config(cls) -> TimeframeConfig:
                return TimeframeConfig(
                    affinity=TimeframeAffinity.SHORT_TERM,
                    min_period=7,
                    max_period=21,
                    supported_variants=["short", "medium"],
                )

            # For Momentum (J&T 1-12 months)
            @classmethod
            def timeframe_config(cls) -> TimeframeConfig:
                return TimeframeConfig(
                    affinity=TimeframeAffinity.MULTI_TIMEFRAME,
                    min_period=5,
                    max_period=252,
                    supported_variants=["short", "medium", "long", "half_year", "yearly"],
                )
        """
        # Default: support all variants (most permissive)
        return TimeframeConfig(
            affinity=TimeframeAffinity.MULTI_TIMEFRAME,
            min_period=5,
            max_period=252,
            supported_variants=["short", "medium", "long", "half_year", "yearly"],
        )

    @classmethod
    def searchable_params(cls) -> List[ParameterSpec]:
        """Get only searchable parameters."""
        return [spec for spec in cls.parameter_specs() if spec.searchable]

    @classmethod
    def fixed_params(cls) -> List[ParameterSpec]:
        """Get only fixed (non-searchable) parameters."""
        return [spec for spec in cls.parameter_specs() if not spec.searchable]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for optimization search.

        Override this method in subclasses to provide custom parameter ranges
        that differ from the default ranges defined in parameter_specs.

        Returns:
            Dictionary mapping parameter names to lists of values to search.

        Example:
            @classmethod
            def get_param_grid(cls) -> Dict[str, List[Any]]:
                return {
                    "lookback": [5, 10, 15, 20, 30, 40, 60, 90, 120],
                }
        """
        grid: Dict[str, List[Any]] = {}
        for spec in cls.searchable_params():
            if spec.min_value is not None and spec.max_value is not None:
                step = spec.step or (spec.max_value - spec.min_value) / 10
                values = np.arange(spec.min_value, spec.max_value + step / 2, step)
                grid[spec.name] = values.tolist()
        return grid

    @classmethod
    def get_param_combinations(cls, max_combinations: int = 500) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for grid search.

        Args:
            max_combinations: Maximum number of combinations to return.
                              If grid produces more, samples uniformly.

        Returns:
            List of parameter dictionaries.
        """
        import itertools
        import random

        grid = cls.get_param_grid()
        if not grid:
            return [{}]

        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        # Generate all combinations
        all_combos = list(itertools.product(*values))
        total = len(all_combos)

        # Sample if too many
        if total > max_combinations:
            all_combos = random.sample(all_combos, max_combinations)

        # Convert to list of dicts
        return [dict(zip(keys, combo)) for combo in all_combos]

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute signal scores from historical data.

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  Index should be DatetimeIndex

        Returns:
            SignalResult with scores in [-1, +1] range
        """
        pass

    def __call__(self, data: pd.DataFrame) -> SignalResult:
        """Shorthand for compute()."""
        return self.compute(data)

    @staticmethod
    def normalize_tanh(values: pd.Series, scale: float = 1.0) -> pd.Series:
        """
        Normalize values to [-1, +1] using tanh compression.

        Args:
            values: Raw signal values
            scale: Scaling factor before tanh (higher = more aggressive compression)

        Returns:
            Normalized values in [-1, +1]
        """
        return np.tanh(values * scale)

    @staticmethod
    def normalize_zscore_tanh(
        values: pd.Series, lookback: int = 20, scale: float = 0.5
    ) -> pd.Series:
        """
        Normalize using rolling z-score then tanh compression.

        Args:
            values: Raw signal values
            lookback: Lookback period for z-score calculation
            scale: Scaling factor before tanh

        Returns:
            Normalized values in [-1, +1]
        """
        rolling_mean = values.rolling(window=lookback, min_periods=1).mean()
        rolling_std = values.rolling(window=lookback, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)
        z_score = (values - rolling_mean) / rolling_std
        return np.tanh(z_score * scale)

    @staticmethod
    def normalize_minmax_scaled(
        values: pd.Series, lookback: int = 20
    ) -> pd.Series:
        """
        Normalize using rolling min-max scaling to [-1, +1].

        Args:
            values: Raw signal values
            lookback: Lookback period for min-max calculation

        Returns:
            Normalized values in [-1, +1]
        """
        rolling_min = values.rolling(window=lookback, min_periods=1).min()
        rolling_max = values.rolling(window=lookback, min_periods=1).max()
        range_val = rolling_max - rolling_min
        # Avoid division by zero
        range_val = range_val.replace(0, np.nan).ffill().fillna(1)
        # Scale to [0, 1] then to [-1, +1]
        normalized = (values - rolling_min) / range_val
        return 2 * normalized - 1

    def validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input DataFrame format.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ["close"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if data.empty:
            raise ValueError("Input DataFrame is empty")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"
