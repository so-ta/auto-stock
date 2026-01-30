"""
Signal Base Class - Abstract base class for all signals.

Design Principles:
- Pure function design: input (historical data) -> output (score)
- Output normalization: [-1, +1] range using tanh compression
- Parameter separation: searchable vs fixed parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", bound="Signal")


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
