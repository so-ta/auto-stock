"""Walk-forward validation engine.

This module implements walk-forward validation for time series data,
ensuring no data leakage between train and test periods.

Features:
- Rolling train/test splits with configurable periods
- Embargo period to prevent data leakage
- Support for both calendar days and business days
- Iterator interface for easy integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator, Optional, Protocol, Sequence, TypeVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        train_days: Number of days for training period.
        test_days: Number of days for test period.
        step_days: Number of days to slide forward between folds.
        embargo_days: Gap between train and test to prevent leakage.
        min_train_samples: Minimum required samples in training set.
        use_business_days: Whether to count only business days.
    """

    train_days: int = 730  # ~2 years
    test_days: int = 30  # ~1 month
    step_days: int = 30  # ~1 month
    embargo_days: int = 5  # Gap to prevent leakage
    min_train_samples: int = 200
    use_business_days: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.train_days <= 0:
            raise ValueError("train_days must be positive")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        if self.min_train_samples <= 0:
            raise ValueError("min_train_samples must be positive")


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward validation.

    Attributes:
        fold_id: Unique identifier for this fold.
        train_start: Start date of training period.
        train_end: End date of training period.
        test_start: Start date of test period.
        test_end: End date of test period.
        train_indices: Array indices for training data.
        test_indices: Array indices for test data.
    """

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray = field(repr=False)
    test_indices: np.ndarray = field(repr=False)

    @property
    def train_size(self) -> int:
        """Number of samples in training set."""
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        """Number of samples in test set."""
        return len(self.test_indices)

    @property
    def embargo_gap(self) -> timedelta:
        """Time gap between train_end and test_start."""
        return self.test_start - self.train_end


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation.

    Attributes:
        config: Configuration used for validation.
        folds: List of fold results.
        total_folds: Total number of folds generated.
        skipped_folds: Number of folds skipped due to insufficient data.
        data_start: Start date of input data.
        data_end: End date of input data.
    """

    config: WalkForwardConfig
    folds: list[WalkForwardFold]
    total_folds: int
    skipped_folds: int
    data_start: datetime
    data_end: datetime

    @property
    def valid_folds(self) -> int:
        """Number of valid (non-skipped) folds."""
        return len(self.folds)

    def get_all_train_indices(self) -> np.ndarray:
        """Get unique indices used in any training set."""
        if not self.folds:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([f.train_indices for f in self.folds]))

    def get_all_test_indices(self) -> np.ndarray:
        """Get unique indices used in any test set."""
        if not self.folds:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([f.test_indices for f in self.folds]))


class DateIndexer(Protocol):
    """Protocol for objects that provide date indexing."""

    def get_dates(self) -> pd.DatetimeIndex:
        """Return the datetime index."""
        ...


T = TypeVar("T", bound=DateIndexer)


class WalkForwardValidator:
    """Walk-forward validation engine.

    This class generates train/test splits for time series data while
    maintaining temporal order and preventing data leakage.

    Example:
        >>> config = WalkForwardConfig(train_days=730, test_days=30)
        >>> validator = WalkForwardValidator(config)
        >>> for fold in validator.split(dates):
        ...     train_data = data[fold.train_indices]
        ...     test_data = data[fold.test_indices]
        ...     # Train and evaluate model
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        """Initialize the validator.

        Args:
            config: Validation configuration. Uses defaults if None.
        """
        self.config = config or WalkForwardConfig()

    def split(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> WalkForwardResult:
        """Generate walk-forward folds for the given date index.

        Args:
            dates: DateTime index of the data.
            start_date: Optional start date to begin validation.
            end_date: Optional end date to stop validation.

        Returns:
            WalkForwardResult containing all generated folds.

        Raises:
            ValueError: If dates is empty or has insufficient data.
        """
        if isinstance(dates, pd.DatetimeIndex):
            dates_index = dates
        else:
            dates_index = pd.DatetimeIndex(dates)

        if len(dates_index) == 0:
            raise ValueError("dates cannot be empty")

        # Sort dates to ensure temporal order
        dates_index = dates_index.sort_values()
        date_values = dates_index.to_numpy()

        data_start = pd.Timestamp(date_values[0]).to_pydatetime()
        data_end = pd.Timestamp(date_values[-1]).to_pydatetime()

        # Apply date bounds if specified
        effective_start = start_date if start_date else data_start
        effective_end = end_date if end_date else data_end

        folds: list[WalkForwardFold] = []
        fold_id = 0
        skipped = 0

        # Calculate first possible test_end date
        # train + embargo + test from effective_start
        first_test_end = (
            effective_start
            + timedelta(days=self.config.train_days)
            + timedelta(days=self.config.embargo_days)
            + timedelta(days=self.config.test_days)
        )

        current_test_end = first_test_end

        while current_test_end <= effective_end:
            # Calculate period boundaries
            test_end = current_test_end
            test_start = test_end - timedelta(days=self.config.test_days)
            train_end = test_start - timedelta(days=self.config.embargo_days)
            train_start = train_end - timedelta(days=self.config.train_days)

            # Ensure train_start is not before data_start
            if train_start < data_start:
                skipped += 1
                current_test_end += timedelta(days=self.config.step_days)
                continue

            # Get indices for train and test periods
            train_mask = (date_values >= np.datetime64(train_start)) & (
                date_values < np.datetime64(train_end)
            )
            test_mask = (date_values >= np.datetime64(test_start)) & (
                date_values <= np.datetime64(test_end)
            )

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Check minimum training samples
            if len(train_indices) < self.config.min_train_samples:
                skipped += 1
                current_test_end += timedelta(days=self.config.step_days)
                continue

            # Skip if no test samples
            if len(test_indices) == 0:
                skipped += 1
                current_test_end += timedelta(days=self.config.step_days)
                continue

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            folds.append(fold)
            fold_id += 1

            # Move to next fold
            current_test_end += timedelta(days=self.config.step_days)

        return WalkForwardResult(
            config=self.config,
            folds=folds,
            total_folds=fold_id + skipped,
            skipped_folds=skipped,
            data_start=data_start,
            data_end=data_end,
        )

    def iter_folds(
        self,
        dates: pd.DatetimeIndex | Sequence[datetime],
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Iterator[WalkForwardFold]:
        """Iterate over walk-forward folds.

        This is a convenience method that yields folds one at a time.

        Args:
            dates: DateTime index of the data.
            start_date: Optional start date to begin validation.
            end_date: Optional end date to stop validation.

        Yields:
            WalkForwardFold for each valid fold.
        """
        result = self.split(dates, start_date=start_date, end_date=end_date)
        yield from result.folds

    def validate_no_leakage(self, fold: WalkForwardFold) -> bool:
        """Verify that a fold has no data leakage.

        Args:
            fold: The fold to validate.

        Returns:
            True if no leakage detected, False otherwise.
        """
        # Check that train and test indices don't overlap
        train_set = set(fold.train_indices.tolist())
        test_set = set(fold.test_indices.tolist())

        if train_set & test_set:
            return False

        # Check that test comes after train
        if fold.test_start <= fold.train_end:
            return False

        # Check embargo gap
        expected_gap = timedelta(days=self.config.embargo_days)
        actual_gap = fold.test_start - fold.train_end

        if actual_gap < expected_gap:
            return False

        return True


def create_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_days: int = 730,
    test_days: int = 30,
    step_days: int = 30,
    embargo_days: int = 5,
) -> WalkForwardResult:
    """Convenience function to create walk-forward splits.

    Args:
        dates: DateTime index of the data.
        train_days: Number of days for training period.
        test_days: Number of days for test period.
        step_days: Number of days to slide forward between folds.
        embargo_days: Gap between train and test to prevent leakage.

    Returns:
        WalkForwardResult containing all generated folds.

    Example:
        >>> dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        >>> result = create_walk_forward_splits(dates)
        >>> print(f"Generated {result.valid_folds} folds")
    """
    config = WalkForwardConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
    )
    validator = WalkForwardValidator(config)
    return validator.split(dates)
