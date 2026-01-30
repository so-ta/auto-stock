"""
Data Quality Checker - Quality Validation for OHLCV Data

This module provides comprehensive data quality checks including:
- Missing data detection
- Anomaly detection (price spikes, volume zeros)
- Duplicate bar detection
- Time series monotonicity check
- Data staleness check

Quality NG data is logged and excluded from evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import structlog

if TYPE_CHECKING:
    from src.config.settings import DataQualityConfig, Settings

logger = structlog.get_logger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""

    check_name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    message: str = ""
    severity: str = "warning"  # info, warning, error


@dataclass
class DataQualityReport:
    """Complete quality report for a dataset."""

    symbol: str
    timestamp: datetime
    overall_passed: bool
    is_excluded: bool
    exclusion_reason: str | None
    checks: list[QualityCheckResult] = field(default_factory=list)
    missing_rate: float = 0.0
    anomaly_count: int = 0
    duplicate_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_passed": self.overall_passed,
            "is_excluded": self.is_excluded,
            "exclusion_reason": self.exclusion_reason,
            "missing_rate": self.missing_rate,
            "anomaly_count": self.anomaly_count,
            "duplicate_count": self.duplicate_count,
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


class DataQualityChecker:
    """
    Data quality checker for OHLCV data.

    Performs comprehensive quality checks and generates detailed reports.
    Assets failing critical checks are excluded from evaluation.

    Example:
        >>> checker = DataQualityChecker(settings)
        >>> report = checker.check(df, symbol="BTCUSD")
        >>> if report.is_excluded:
        ...     print(f"Excluded: {report.exclusion_reason}")
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        """
        Initialize quality checker.

        Args:
            settings: Settings instance (uses global if not provided)
        """
        self._settings = settings
        self._logger = logger.bind(component="quality_checker")

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def config(self) -> "DataQualityConfig":
        """Get data quality config."""
        return self.settings.data_quality

    def check(
        self,
        df: pl.DataFrame,
        symbol: str,
        expected_bars: int | None = None,
    ) -> DataQualityReport:
        """
        Perform all quality checks on a DataFrame.

        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            symbol: Asset symbol for logging
            expected_bars: Expected number of bars (for missing rate calculation)

        Returns:
            DataQualityReport with check results
        """
        checks: list[QualityCheckResult] = []
        is_excluded = False
        exclusion_reason = None

        # Run all checks
        checks.append(self._check_empty(df))
        checks.append(self._check_missing_values(df))
        checks.append(self._check_missing_rate(df, expected_bars))
        checks.append(self._check_consecutive_missing(df))
        checks.append(self._check_duplicates(df))
        checks.append(self._check_monotonicity(df))
        checks.append(self._check_price_anomalies(df))
        checks.append(self._check_volume_anomalies(df))
        checks.append(self._check_ohlc_consistency(df))
        checks.append(self._check_staleness(df))

        # Calculate summary metrics
        missing_rate = self._calculate_missing_rate(df, expected_bars)
        anomaly_count = self._count_anomalies(df)
        duplicate_count = self._count_duplicates(df)

        # Determine if should be excluded (any error-level failure)
        error_checks = [c for c in checks if not c.passed and c.severity == "error"]
        if error_checks:
            is_excluded = True
            exclusion_reason = "; ".join(c.message for c in error_checks)

        # Check if missing rate exceeds threshold
        if missing_rate > self.config.max_missing_rate:
            is_excluded = True
            exclusion_reason = (
                f"Missing rate {missing_rate:.2%} exceeds threshold "
                f"{self.config.max_missing_rate:.2%}"
            )

        overall_passed = all(c.passed for c in checks)

        report = DataQualityReport(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            overall_passed=overall_passed,
            is_excluded=is_excluded,
            exclusion_reason=exclusion_reason,
            checks=checks,
            missing_rate=missing_rate,
            anomaly_count=anomaly_count,
            duplicate_count=duplicate_count,
        )

        # Log the report
        self._log_report(report)

        return report

    def check_batch(
        self,
        data: dict[str, pl.DataFrame],
        expected_bars: int | None = None,
    ) -> dict[str, DataQualityReport]:
        """
        Check multiple assets at once.

        Args:
            data: Dict mapping symbol -> DataFrame
            expected_bars: Expected bars per asset

        Returns:
            Dict mapping symbol -> DataQualityReport
        """
        reports = {}
        for symbol, df in data.items():
            reports[symbol] = self.check(df, symbol, expected_bars)
        return reports

    def filter_excluded(
        self,
        data: dict[str, pl.DataFrame],
        reports: dict[str, DataQualityReport] | None = None,
    ) -> tuple[dict[str, pl.DataFrame], list[str]]:
        """
        Filter out excluded assets.

        Args:
            data: Dict mapping symbol -> DataFrame
            reports: Optional pre-computed reports

        Returns:
            Tuple of (filtered_data, excluded_symbols)
        """
        if reports is None:
            reports = self.check_batch(data)

        filtered = {}
        excluded = []

        for symbol, df in data.items():
            report = reports.get(symbol)
            if report and report.is_excluded:
                excluded.append(symbol)
                self._logger.info(
                    "Asset excluded due to quality issues",
                    symbol=symbol,
                    reason=report.exclusion_reason,
                )
            else:
                filtered[symbol] = df

        return filtered, excluded

    # =========================================================================
    # Individual Quality Checks
    # =========================================================================
    def _check_empty(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check if DataFrame is empty."""
        is_empty = len(df) == 0
        return QualityCheckResult(
            check_name="empty_check",
            passed=not is_empty,
            value=float(len(df)),
            message="DataFrame is empty" if is_empty else "",
            severity="error" if is_empty else "info",
        )

    def _check_missing_values(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check for null/NaN values in critical columns."""
        if len(df) == 0:
            return QualityCheckResult(
                check_name="missing_values",
                passed=True,
                message="Skipped (empty DataFrame)",
            )

        critical_cols = ["open", "high", "low", "close"]
        null_counts = {}

        for col in critical_cols:
            if col in df.columns:
                null_counts[col] = df[col].null_count()

        total_nulls = sum(null_counts.values())
        passed = total_nulls == 0

        return QualityCheckResult(
            check_name="missing_values",
            passed=passed,
            value=float(total_nulls),
            message=f"Found {total_nulls} null values in price columns" if not passed else "",
            severity="error" if not passed else "info",
        )

    def _check_missing_rate(
        self,
        df: pl.DataFrame,
        expected_bars: int | None,
    ) -> QualityCheckResult:
        """Check missing data rate."""
        if expected_bars is None or expected_bars == 0:
            return QualityCheckResult(
                check_name="missing_rate",
                passed=True,
                message="Skipped (no expected_bars provided)",
            )

        actual_bars = len(df)
        missing_rate = 1 - (actual_bars / expected_bars)
        passed = missing_rate <= self.config.max_missing_rate

        return QualityCheckResult(
            check_name="missing_rate",
            passed=passed,
            value=missing_rate,
            threshold=self.config.max_missing_rate,
            message=(
                f"Missing rate {missing_rate:.2%} exceeds {self.config.max_missing_rate:.2%}"
                if not passed
                else ""
            ),
            severity="error" if not passed else "info",
        )

    def _check_consecutive_missing(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check for consecutive missing bars."""
        if len(df) < 2 or "timestamp" not in df.columns:
            return QualityCheckResult(
                check_name="consecutive_missing",
                passed=True,
                message="Skipped (insufficient data)",
            )

        # Calculate time differences
        df_sorted = df.sort("timestamp")
        timestamps = df_sorted["timestamp"].to_list()

        # Infer expected interval from median difference
        diffs = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        if not diffs:
            return QualityCheckResult(
                check_name="consecutive_missing",
                passed=True,
                message="Skipped (single bar)",
            )

        median_diff = sorted(diffs)[len(diffs) // 2]

        # Find gaps larger than 2x median (indicates missing bars)
        max_consecutive = 0
        current_consecutive = 0

        for diff in diffs:
            if diff > median_diff * 1.5:
                consecutive_missing = int(diff / median_diff) - 1
                current_consecutive = max(current_consecutive, consecutive_missing)
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 0

        max_consecutive = max(max_consecutive, current_consecutive)
        passed = max_consecutive <= self.config.max_consecutive_missing

        return QualityCheckResult(
            check_name="consecutive_missing",
            passed=passed,
            value=float(max_consecutive),
            threshold=float(self.config.max_consecutive_missing),
            message=(
                f"Found {max_consecutive} consecutive missing bars"
                if not passed
                else ""
            ),
            severity="warning" if not passed else "info",
        )

    def _check_duplicates(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check for duplicate timestamps."""
        if len(df) == 0 or "timestamp" not in df.columns:
            return QualityCheckResult(
                check_name="duplicates",
                passed=True,
                message="Skipped (no timestamp column)",
            )

        total_rows = len(df)
        unique_rows = df.select("timestamp").n_unique()
        duplicate_count = total_rows - unique_rows
        passed = duplicate_count == 0

        return QualityCheckResult(
            check_name="duplicates",
            passed=passed,
            value=float(duplicate_count),
            message=f"Found {duplicate_count} duplicate timestamps" if not passed else "",
            severity="warning" if not passed else "info",
        )

    def _check_monotonicity(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check if timestamps are monotonically increasing."""
        if len(df) < 2 or "timestamp" not in df.columns:
            return QualityCheckResult(
                check_name="monotonicity",
                passed=True,
                message="Skipped (insufficient data)",
            )

        # Check if any timestamp is earlier than the previous
        df_check = df.with_columns(
            pl.col("timestamp").shift(1).alias("prev_timestamp")
        ).filter(
            pl.col("prev_timestamp").is_not_null()
        )

        non_monotonic = df_check.filter(
            pl.col("timestamp") <= pl.col("prev_timestamp")
        )

        count = len(non_monotonic)
        passed = count == 0

        return QualityCheckResult(
            check_name="monotonicity",
            passed=passed,
            value=float(count),
            message=(
                f"Found {count} non-monotonic timestamps (possible future data leak)"
                if not passed
                else ""
            ),
            severity="error" if not passed else "info",
        )

    def _check_price_anomalies(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check for price anomalies (extreme changes)."""
        if len(df) < 2 or "close" not in df.columns:
            return QualityCheckResult(
                check_name="price_anomalies",
                passed=True,
                message="Skipped (insufficient data)",
            )

        # Calculate returns
        df_returns = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
        ).filter(pl.col("return").is_not_null())

        # Find extreme returns
        threshold = self.config.price_change_threshold
        anomalies = df_returns.filter(pl.col("return").abs() > threshold)
        count = len(anomalies)

        # More than 1% anomalies is suspicious
        anomaly_rate = count / len(df) if len(df) > 0 else 0
        passed = anomaly_rate < 0.01

        return QualityCheckResult(
            check_name="price_anomalies",
            passed=passed,
            value=float(count),
            threshold=threshold,
            message=(
                f"Found {count} price anomalies (>{threshold:.0%} change)"
                if count > 0
                else ""
            ),
            severity="warning" if not passed else "info",
        )

    def _check_volume_anomalies(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check for volume anomalies (zeros, extreme values)."""
        if len(df) == 0 or "volume" not in df.columns:
            return QualityCheckResult(
                check_name="volume_anomalies",
                passed=True,
                message="Skipped (no volume column)",
            )

        # Count zero volumes
        zero_volumes = df.filter(pl.col("volume") <= self.config.min_volume_threshold)
        zero_count = len(zero_volumes)

        # Zero volume rate
        zero_rate = zero_count / len(df) if len(df) > 0 else 0
        passed = zero_rate < 0.1  # Less than 10% zero volumes

        return QualityCheckResult(
            check_name="volume_anomalies",
            passed=passed,
            value=float(zero_count),
            message=(
                f"Found {zero_count} bars with zero/low volume ({zero_rate:.1%})"
                if zero_count > 0
                else ""
            ),
            severity="warning" if not passed else "info",
        )

    def _check_ohlc_consistency(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check OHLC consistency (high >= low, prices within high/low)."""
        if len(df) == 0:
            return QualityCheckResult(
                check_name="ohlc_consistency",
                passed=True,
                message="Skipped (empty DataFrame)",
            )

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(df.columns)):
            return QualityCheckResult(
                check_name="ohlc_consistency",
                passed=True,
                message="Skipped (missing OHLC columns)",
            )

        # Check high >= low
        invalid_hl = df.filter(pl.col("high") < pl.col("low"))

        # Check open/close within high/low
        invalid_open = df.filter(
            (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
        )
        invalid_close = df.filter(
            (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
        )

        total_invalid = len(invalid_hl) + len(invalid_open) + len(invalid_close)
        total_rows = len(df)
        inconsistency_rate = total_invalid / total_rows if total_rows > 0 else 0.0
        threshold = self.config.ohlc_inconsistency_threshold

        # Determine severity based on inconsistency rate
        if total_invalid == 0:
            severity = "info"
            passed = True
        elif inconsistency_rate <= threshold:
            severity = "warning"  # Do not exclude
            passed = True
        else:
            severity = "error"  # Exclude
            passed = False

        return QualityCheckResult(
            check_name="ohlc_consistency",
            passed=passed,
            value=inconsistency_rate,
            threshold=threshold,
            message=(
                f"Found {total_invalid} OHLC inconsistencies ({inconsistency_rate:.2%} rate)"
                if total_invalid > 0
                else ""
            ),
            severity=severity,
        )

    def _check_staleness(self, df: pl.DataFrame) -> QualityCheckResult:
        """Check if data is stale (last timestamp too old)."""
        if len(df) == 0 or "timestamp" not in df.columns:
            return QualityCheckResult(
                check_name="staleness",
                passed=True,
                message="Skipped (no timestamp)",
            )

        last_timestamp = df["timestamp"].max()
        if last_timestamp is None:
            return QualityCheckResult(
                check_name="staleness",
                passed=True,
                message="Skipped (no valid timestamp)",
            )

        # Convert to datetime if needed
        if isinstance(last_timestamp, str):
            last_timestamp = datetime.fromisoformat(last_timestamp)

        age = datetime.utcnow() - last_timestamp
        age_hours = age.total_seconds() / 3600
        threshold_hours = self.config.staleness_hours
        passed = age_hours <= threshold_hours

        return QualityCheckResult(
            check_name="staleness",
            passed=passed,
            value=age_hours,
            threshold=float(threshold_hours),
            message=(
                f"Data is {age_hours:.1f} hours old (threshold: {threshold_hours}h)"
                if not passed
                else ""
            ),
            severity="warning" if not passed else "info",
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _calculate_missing_rate(
        self,
        df: pl.DataFrame,
        expected_bars: int | None,
    ) -> float:
        """Calculate missing data rate."""
        if expected_bars is None or expected_bars == 0:
            return 0.0
        return 1 - (len(df) / expected_bars)

    def _count_anomalies(self, df: pl.DataFrame) -> int:
        """Count total anomalies in data."""
        if len(df) < 2 or "close" not in df.columns:
            return 0

        df_returns = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
        ).filter(pl.col("return").is_not_null())

        threshold = self.config.price_change_threshold
        return len(df_returns.filter(pl.col("return").abs() > threshold))

    def _count_duplicates(self, df: pl.DataFrame) -> int:
        """Count duplicate timestamps."""
        if len(df) == 0 or "timestamp" not in df.columns:
            return 0
        return len(df) - df.select("timestamp").n_unique()

    def _log_report(self, report: DataQualityReport) -> None:
        """Log quality report."""
        if report.is_excluded:
            self._logger.warning(
                "Asset excluded due to quality issues",
                symbol=report.symbol,
                reason=report.exclusion_reason,
                missing_rate=f"{report.missing_rate:.2%}",
                anomaly_count=report.anomaly_count,
            )
        elif not report.overall_passed:
            self._logger.info(
                "Asset passed with warnings",
                symbol=report.symbol,
                missing_rate=f"{report.missing_rate:.2%}",
                anomaly_count=report.anomaly_count,
                failed_checks=[c.check_name for c in report.checks if not c.passed],
            )
        else:
            self._logger.debug(
                "Asset quality check passed",
                symbol=report.symbol,
            )
