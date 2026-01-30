"""Time series cross-validation module.

This module implements time series cross-validation that respects
temporal ordering and prevents data leakage.

Features:
- K-fold splits maintaining temporal order
- No random shuffling (forbidden for time series)
- Purging to prevent information leakage
- Embargo periods after each test set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesCVConfig:
    """Configuration for time series cross-validation.

    Attributes:
        n_splits: Number of CV splits (folds).
        gap_days: Number of days to purge between train and test.
        embargo_days: Days to embargo after test period.
        min_train_size: Minimum samples required in training set.
        max_train_size: Maximum samples in training set (None for expanding).
        test_size: Fixed test size (None for proportional).
    """

    n_splits: int = 5
    gap_days: int = 5
    embargo_days: int = 5
    min_train_size: int = 100
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap_days < 0:
            raise ValueError("gap_days cannot be negative")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        if self.min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if self.max_train_size is not None and self.max_train_size < self.min_train_size:
            raise ValueError("max_train_size cannot be less than min_train_size")
        if self.test_size is not None and self.test_size <= 0:
            raise ValueError("test_size must be positive if specified")


@dataclass
class CVFold:
    """A single fold in time series cross-validation.

    Attributes:
        fold_id: Unique identifier for this fold (0-indexed).
        train_indices: Array indices for training data.
        test_indices: Array indices for test data.
        train_start_date: Start date of training period.
        train_end_date: End date of training period.
        test_start_date: Start date of test period.
        test_end_date: End date of test period.
        gap_indices: Indices in the purge gap (excluded from both).
        embargo_indices: Indices in the embargo period (excluded from train).
    """

    fold_id: int
    train_indices: np.ndarray = field(repr=False)
    test_indices: np.ndarray = field(repr=False)
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None
    gap_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)
    embargo_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64), repr=False)

    @property
    def train_size(self) -> int:
        """Number of samples in training set."""
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        """Number of samples in test set."""
        return len(self.test_indices)

    def has_leakage(self) -> bool:
        """Check if there's any overlap between train and test indices."""
        train_set = set(self.train_indices.tolist())
        test_set = set(self.test_indices.tolist())
        return bool(train_set & test_set)


@dataclass
class CVResult:
    """Result of time series cross-validation split.

    Attributes:
        config: Configuration used for splitting.
        folds: List of CV folds.
        n_samples: Total number of samples in the data.
        data_start: Start date of the data.
        data_end: End date of the data.
    """

    config: TimeSeriesCVConfig
    folds: list[CVFold]
    n_samples: int
    data_start: Optional[datetime] = None
    data_end: Optional[datetime] = None

    @property
    def n_folds(self) -> int:
        """Number of folds generated."""
        return len(self.folds)

    def get_train_sizes(self) -> list[int]:
        """Get training set sizes for all folds."""
        return [f.train_size for f in self.folds]

    def get_test_sizes(self) -> list[int]:
        """Get test set sizes for all folds."""
        return [f.test_size for f in self.folds]


class TimeSeriesCV:
    """Time series cross-validation splitter.

    This class generates train/test splits for time series data while:
    - Maintaining strict temporal order (no shuffling)
    - Applying purge gaps to prevent information leakage
    - Supporting both expanding and sliding window schemes

    Example:
        >>> config = TimeSeriesCVConfig(n_splits=5, gap_days=5)
        >>> cv = TimeSeriesCV(config)
        >>> for fold in cv.split(data):
        ...     X_train, X_test = data[fold.train_indices], data[fold.test_indices]
        ...     # Train and evaluate model
    """

    def __init__(self, config: Optional[TimeSeriesCVConfig] = None) -> None:
        """Initialize the cross-validator.

        Args:
            config: CV configuration. Uses defaults if None.
        """
        self.config = config or TimeSeriesCVConfig()

    def split(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        dates: Optional[pd.DatetimeIndex | Sequence[datetime]] = None,
    ) -> CVResult:
        """Generate time series CV splits.

        Args:
            X: Input data (used to determine sample count).
            dates: Optional date index for date-aware splitting.

        Returns:
            CVResult containing all folds.

        Raises:
            ValueError: If data is too small for the requested splits.
        """
        n_samples = len(X)

        if n_samples < self.config.min_train_size + self.config.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for "
                f"{self.config.n_splits} splits with min_train_size={self.config.min_train_size}"
            )

        # Process dates if provided
        dates_index: Optional[pd.DatetimeIndex] = None
        if dates is not None:
            if isinstance(dates, pd.DatetimeIndex):
                dates_index = dates
            else:
                dates_index = pd.DatetimeIndex(dates)

            if len(dates_index) != n_samples:
                raise ValueError(
                    f"dates length ({len(dates_index)}) must match X length ({n_samples})"
                )

        # Calculate test size
        if self.config.test_size is not None:
            test_size = self.config.test_size
        else:
            # Proportional test size
            test_size = n_samples // (self.config.n_splits + 1)

        if test_size <= 0:
            raise ValueError("Calculated test_size is too small")

        folds: list[CVFold] = []

        for fold_id in range(self.config.n_splits):
            # Calculate test boundaries
            # Folds are arranged so later folds use later test data
            test_end = n_samples - (self.config.n_splits - fold_id - 1) * test_size
            test_start = test_end - test_size

            # Calculate purge gap
            gap_end = test_start
            gap_start = max(0, test_start - self.config.gap_days)

            # Calculate train boundaries
            train_end = gap_start
            train_start = 0

            # Apply max_train_size if specified (sliding window)
            if self.config.max_train_size is not None:
                train_start = max(0, train_end - self.config.max_train_size)

            # Skip if training set is too small
            if train_end - train_start < self.config.min_train_size:
                continue

            # Create indices
            train_indices = np.arange(train_start, train_end, dtype=np.int64)
            test_indices = np.arange(test_start, test_end, dtype=np.int64)
            gap_indices = np.arange(gap_start, gap_end, dtype=np.int64)

            # Calculate embargo indices (after test set)
            embargo_start = test_end
            embargo_end = min(n_samples, test_end + self.config.embargo_days)
            embargo_indices = np.arange(embargo_start, embargo_end, dtype=np.int64)

            # Get dates if available
            train_start_date = None
            train_end_date = None
            test_start_date = None
            test_end_date = None

            if dates_index is not None:
                if train_start < len(dates_index):
                    train_start_date = pd.Timestamp(dates_index[train_start]).to_pydatetime()
                if train_end - 1 < len(dates_index) and train_end > 0:
                    train_end_date = pd.Timestamp(dates_index[train_end - 1]).to_pydatetime()
                if test_start < len(dates_index):
                    test_start_date = pd.Timestamp(dates_index[test_start]).to_pydatetime()
                if test_end - 1 < len(dates_index) and test_end > 0:
                    test_end_date = pd.Timestamp(dates_index[test_end - 1]).to_pydatetime()

            fold = CVFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                gap_indices=gap_indices,
                embargo_indices=embargo_indices,
            )
            folds.append(fold)

        data_start = None
        data_end = None
        if dates_index is not None and len(dates_index) > 0:
            data_start = pd.Timestamp(dates_index[0]).to_pydatetime()
            data_end = pd.Timestamp(dates_index[-1]).to_pydatetime()

        return CVResult(
            config=self.config,
            folds=folds,
            n_samples=n_samples,
            data_start=data_start,
            data_end=data_end,
        )

    def iter_splits(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        dates: Optional[pd.DatetimeIndex | Sequence[datetime]] = None,
    ) -> Iterator[CVFold]:
        """Iterate over CV folds.

        Args:
            X: Input data.
            dates: Optional date index.

        Yields:
            CVFold for each valid fold.
        """
        result = self.split(X, dates)
        yield from result.folds

    def get_n_splits(
        self,
        X: Optional[np.ndarray | pd.DataFrame | pd.Series] = None,
    ) -> int:
        """Return the number of splits.

        Args:
            X: Optional input data (not used, for sklearn compatibility).

        Returns:
            Number of splits configured.
        """
        return self.config.n_splits


class PurgedKFold(TimeSeriesCV):
    """K-Fold CV with purging for financial time series.

    This is an alias for TimeSeriesCV that emphasizes the purging behavior.
    Used in financial ML to prevent information leakage from overlapping
    features (e.g., moving averages).
    """

    pass


class CombinatorialPurgedKFold:
    """Combinatorial purged cross-validation.

    Generates all possible train/test combinations while respecting
    temporal order and applying purging. Useful for strategy selection
    where you want to test all possible out-of-sample scenarios.

    Note: Number of combinations grows rapidly with n_splits.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        gap_days: int = 5,
        embargo_days: int = 5,
    ) -> None:
        """Initialize combinatorial CV.

        Args:
            n_splits: Total number of time periods to create.
            n_test_splits: Number of periods to use for testing in each combination.
            gap_days: Days to purge between train and test.
            embargo_days: Days to embargo after test.
        """
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3")
        if n_test_splits < 1:
            raise ValueError("n_test_splits must be at least 1")
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be less than n_splits")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.gap_days = gap_days
        self.embargo_days = embargo_days

    def get_n_combinations(self) -> int:
        """Calculate number of train/test combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Iterator[CVFold]:
        """Generate combinatorial CV splits.

        Args:
            X: Input data.
            dates: Optional date index.

        Yields:
            CVFold for each combination.
        """
        from itertools import combinations

        n_samples = len(X)
        period_size = n_samples // self.n_splits

        # Generate all combinations of test periods
        all_periods = list(range(self.n_splits))
        fold_id = 0

        for test_periods in combinations(all_periods, self.n_test_splits):
            train_periods = [p for p in all_periods if p not in test_periods]

            # Build train indices (excluding gaps around test periods)
            train_indices_list = []
            test_indices_list = []

            for period in train_periods:
                start = period * period_size
                end = min((period + 1) * period_size, n_samples)

                # Check if adjacent to any test period
                is_adjacent_to_test = False
                for test_period in test_periods:
                    if abs(period - test_period) <= 1:
                        is_adjacent_to_test = True
                        break

                if is_adjacent_to_test:
                    # Apply purging
                    start = start + self.gap_days
                    end = end - self.embargo_days

                if start < end:
                    train_indices_list.append(np.arange(start, end))

            for period in test_periods:
                start = period * period_size
                end = min((period + 1) * period_size, n_samples)
                test_indices_list.append(np.arange(start, end))

            if train_indices_list and test_indices_list:
                train_indices = np.concatenate(train_indices_list)
                test_indices = np.concatenate(test_indices_list)

                yield CVFold(
                    fold_id=fold_id,
                    train_indices=train_indices.astype(np.int64),
                    test_indices=test_indices.astype(np.int64),
                )
                fold_id += 1


def time_series_split(
    n_samples: int,
    n_splits: int = 5,
    gap: int = 0,
    test_size: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Simple function for time series splitting.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of splits.
        gap: Number of samples to skip between train and test.
        test_size: Fixed test size (None for proportional).

    Returns:
        List of (train_indices, test_indices) tuples.

    Example:
        >>> splits = time_series_split(1000, n_splits=5, gap=10)
        >>> for train_idx, test_idx in splits:
        ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """
    config = TimeSeriesCVConfig(
        n_splits=n_splits,
        gap_days=gap,
        embargo_days=0,
        test_size=test_size,
        min_train_size=1,
    )
    cv = TimeSeriesCV(config)
    result = cv.split(np.zeros(n_samples))

    return [(fold.train_indices, fold.test_indices) for fold in result.folds]
