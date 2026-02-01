"""
Reliability Assessment Module - Backtest Result Reliability Scoring

This module provides reliability assessment for backtest results based on
execution logs. When errors, warnings, or fallbacks occur during execution,
the reliability score is reduced to indicate that results may be less trustworthy.

Usage:
    from src.backtest.reliability import ReliabilityCalculator
    from src.utils.pipeline_log_collector import PipelineLogCollector

    calculator = ReliabilityCalculator()
    assessment = calculator.calculate(log_collector)

    if not assessment.is_reliable:
        print(f"Warning: Results have low reliability ({assessment.score:.0%})")
        for reason in assessment.reasons:
            print(f"  - {reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.utils.pipeline_log_collector import PipelineLogCollector


@dataclass
class ReliabilityAssessment:
    """
    Reliability assessment result.

    Attributes:
        score: Reliability score from 0.0 to 1.0
        level: Human-readable level ("high", "medium", "low", "unreliable")
        reasons: List of reasons for score deductions
    """

    score: float
    level: str
    reasons: List[str] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        """Check if the result is considered reliable."""
        return self.level in ("high", "medium")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "reasons": self.reasons,
            "is_reliable": self.is_reliable,
        }


class ReliabilityCalculator:
    """
    Calculator for backtest result reliability scores.

    Analyzes execution logs to determine how trustworthy the backtest
    results are. Deducts points for errors, warnings, fallbacks, and
    other issues that may affect result accuracy.
    """

    # Score deduction triggers (negative values)
    TRIGGERS = {
        # Critical issues - major impact on reliability
        "unhandled_exception": -1.0,       # Unrecoverable error
        "data_fetch_failed": -0.5,         # Missing price data
        "insufficient_data": -0.4,         # Not enough data for analysis

        # Significant issues
        "error_recovered": -0.3,           # Error occurred but recovered
        "fallback_activated": -0.25,       # Fallback mode triggered
        "covariance_failed": -0.25,        # Covariance calculation failed

        # Moderate issues
        "data_quality_warnings_high": -0.2,  # Many data quality warnings
        "missing_benchmark": -0.15,        # Benchmark data unavailable
        "signal_computation_failed": -0.15, # Some signals failed to compute

        # Minor issues
        "data_quality_warnings": -0.1,     # Some data quality warnings
        "partial_data": -0.1,              # Some assets have incomplete data
        "cache_miss_high": -0.05,          # High cache miss rate (slow)
    }

    # Keywords to detect issues in log messages
    ISSUE_KEYWORDS = {
        "unhandled_exception": ["unhandled", "traceback", "fatal", "critical"],
        "error_recovered": ["error", "failed", "exception"],
        "fallback_activated": ["fallback", "fallback_mode", "fallback_triggered"],
        "data_fetch_failed": ["fetch failed", "no data", "data unavailable"],
        "insufficient_data": ["insufficient", "not enough data", "minimum samples"],
        "covariance_failed": ["covariance failed", "matrix singular", "not positive"],
        "missing_benchmark": ["benchmark unavailable", "missing benchmark"],
        "signal_computation_failed": ["signal failed", "signal computation"],
        "data_quality_warnings": ["quality warning", "missing values", "outlier"],
        "partial_data": ["partial", "incomplete", "missing"],
    }

    def __init__(self) -> None:
        """Initialize the calculator."""
        pass

    def calculate(self, log_collector: "PipelineLogCollector") -> ReliabilityAssessment:
        """
        Calculate reliability assessment from log collector.

        Args:
            log_collector: PipelineLogCollector with execution logs

        Returns:
            ReliabilityAssessment with score, level, and reasons
        """
        score = 1.0
        reasons: List[str] = []

        # Get log statistics
        stats = log_collector.get_stats()
        error_count = stats["by_level"].get("ERROR", 0)
        warning_count = stats["by_level"].get("WARNING", 0)
        critical_count = stats["by_level"].get("CRITICAL", 0)

        # Check for critical errors
        if critical_count > 0:
            score += self.TRIGGERS["unhandled_exception"]
            reasons.append(f"Critical error occurred ({critical_count} times)")

        # Check for regular errors
        if error_count > 0:
            score += self.TRIGGERS["error_recovered"]
            reasons.append(f"Errors occurred during execution ({error_count} times)")

        # Check warnings threshold
        if warning_count >= 10:
            score += self.TRIGGERS["data_quality_warnings_high"]
            reasons.append(f"High number of warnings ({warning_count})")
        elif warning_count > 0:
            score += self.TRIGGERS["data_quality_warnings"]
            reasons.append(f"Warnings occurred ({warning_count} times)")

        # Analyze log content for specific issues
        logs = log_collector.get_logs()
        detected_issues = self._detect_issues(logs)

        for issue, occurrences in detected_issues.items():
            if issue not in ("error_recovered", "data_quality_warnings", "data_quality_warnings_high"):
                if issue in self.TRIGGERS:
                    score += self.TRIGGERS[issue]
                    reasons.append(f"{issue.replace('_', ' ').title()} ({occurrences} times)")

        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        # Determine level
        level = self._score_to_level(score)

        return ReliabilityAssessment(
            score=score,
            level=level,
            reasons=reasons,
        )

    def _detect_issues(self, logs: List[Any]) -> Dict[str, int]:
        """
        Detect specific issues in log entries.

        Args:
            logs: List of LogEntry objects

        Returns:
            Dictionary mapping issue type to occurrence count
        """
        detected: Dict[str, int] = {}

        for entry in logs:
            # Convert entry to searchable text
            if hasattr(entry, "event"):
                text = f"{entry.event} {entry.message}".lower()
            else:
                text = str(entry).lower()

            # Check for each issue type
            for issue_type, keywords in self.ISSUE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text:
                        detected[issue_type] = detected.get(issue_type, 0) + 1
                        break  # Count each log entry only once per issue type

        return detected

    def _score_to_level(self, score: float) -> str:
        """
        Convert numeric score to human-readable level.

        Args:
            score: Reliability score (0.0 to 1.0)

        Returns:
            Level string: "high", "medium", "low", or "unreliable"
        """
        if score >= 0.9:
            return "high"
        elif score >= 0.7:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "unreliable"

    def calculate_from_counts(
        self,
        error_count: int = 0,
        warning_count: int = 0,
        fallback_triggered: bool = False,
        data_issues: int = 0,
    ) -> ReliabilityAssessment:
        """
        Calculate reliability from simple counts (convenience method).

        Args:
            error_count: Number of errors
            warning_count: Number of warnings
            fallback_triggered: Whether fallback mode was activated
            data_issues: Number of data quality issues

        Returns:
            ReliabilityAssessment
        """
        score = 1.0
        reasons: List[str] = []

        if error_count > 0:
            score += self.TRIGGERS["error_recovered"]
            reasons.append(f"Errors occurred ({error_count})")

        if warning_count >= 10:
            score += self.TRIGGERS["data_quality_warnings_high"]
            reasons.append(f"High warnings ({warning_count})")
        elif warning_count > 0:
            score += self.TRIGGERS["data_quality_warnings"]
            reasons.append(f"Warnings ({warning_count})")

        if fallback_triggered:
            score += self.TRIGGERS["fallback_activated"]
            reasons.append("Fallback mode activated")

        if data_issues > 0:
            score += self.TRIGGERS["partial_data"]
            reasons.append(f"Data issues ({data_issues})")

        score = max(0.0, min(1.0, score))
        level = self._score_to_level(score)

        return ReliabilityAssessment(
            score=score,
            level=level,
            reasons=reasons,
        )
