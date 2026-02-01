"""
Pipeline Log Collector - Unified Log Collection for Pipeline Execution

This module provides centralized log collection for pipeline execution,
enabling log persistence, viewer integration, and reliability assessment.

Features:
- Collects logs from structlog via processor
- Ring buffer to limit memory usage
- Integration with ProgressTracker for real-time updates
- Saves logs to JSONL format for persistence
- Provides log filtering and retrieval

Usage:
    from src.utils.pipeline_log_collector import PipelineLogCollector
    from src.utils.logger import set_log_collector

    collector = PipelineLogCollector(run_id="run_123")
    set_log_collector(collector)

    # Pipeline execution...

    # Save logs
    collector.save_to_file(Path("results/run_123/logs.jsonl"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.utils.progress_tracker import ProgressTracker


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: str
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    event: str
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PipelineLogCollector:
    """
    Centralized log collector for pipeline execution.

    Collects all logs during pipeline execution, integrates with ProgressTracker
    for real-time viewer updates, and saves logs for persistence.

    Attributes:
        run_id: Unique identifier for the pipeline run
        max_buffer_size: Maximum number of log entries to keep in memory
    """

    def __init__(
        self,
        run_id: str,
        max_buffer_size: int = 1000,
    ) -> None:
        """
        Initialize the log collector.

        Args:
            run_id: Unique identifier for this pipeline run
            max_buffer_size: Maximum entries in ring buffer (oldest are dropped)
        """
        self._run_id = run_id
        self._max_size = max_buffer_size
        self._buffer: List[LogEntry] = []
        self._progress_tracker: Optional["ProgressTracker"] = None

        # Counters for statistics
        self._counts: Dict[str, int] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }

        # Recursion guard for ProgressTracker forwarding
        self._forwarding: bool = False

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run_id

    def attach_progress_tracker(self, tracker: "ProgressTracker") -> None:
        """
        Attach a ProgressTracker for real-time viewer updates.

        Args:
            tracker: ProgressTracker instance to receive log notifications
        """
        self._progress_tracker = tracker

    def log(
        self,
        level: str,
        event: str,
        component: str = "",
        **details: Any,
    ) -> None:
        """
        Add a log entry.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            event: Event description or message
            component: Source component (e.g., module name)
            **details: Additional key-value pairs
        """
        # Normalize level
        level = level.upper()
        if level not in self._counts:
            level = "INFO"

        # Create entry
        entry = LogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            level=level,
            event=event,
            component=component,
            message=details.pop("message", ""),
            details=details,
        )

        # Add to buffer (ring buffer behavior)
        self._buffer.append(entry)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

        # Update counts
        self._counts[level] += 1

        # Forward to ProgressTracker if attached
        if self._progress_tracker is not None:
            self._forward_to_tracker(entry)

    def _forward_to_tracker(self, entry: LogEntry) -> None:
        """Forward log entry to ProgressTracker for viewer updates.

        Uses a recursion guard to prevent infinite loops when ProgressTracker
        methods log messages that would be forwarded back to the tracker.
        """
        if self._progress_tracker is None:
            return

        # Prevent recursion: ProgressTracker.add_warning() calls logger.warning()
        # which would be forwarded back here
        if self._forwarding:
            return

        self._forwarding = True
        try:
            message = entry.event
            if entry.message:
                message = f"{entry.event}: {entry.message}"

            if entry.level == "ERROR":
                self._progress_tracker.add_error(message)
            elif entry.level == "WARNING":
                self._progress_tracker.add_warning(message)
            elif entry.level in ("INFO", "DEBUG"):
                # Use add_info if available (we'll add this method)
                if hasattr(self._progress_tracker, "add_info"):
                    self._progress_tracker.add_info(
                        message=message,
                        component=entry.component,
                        **entry.details,
                    )
        finally:
            self._forwarding = False

    def get_logs(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LogEntry]:
        """
        Get log entries with optional filtering.

        Args:
            level: Filter by log level (e.g., "ERROR", "WARNING")
            component: Filter by component name
            limit: Maximum number of entries to return

        Returns:
            List of matching LogEntry objects
        """
        result = self._buffer

        if level is not None:
            level = level.upper()
            result = [e for e in result if e.level == level]

        if component is not None:
            result = [e for e in result if component in e.component]

        if limit is not None and len(result) > limit:
            result = result[-limit:]

        return result

    def get_logs_as_dicts(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get log entries as dictionaries.

        Args:
            level: Filter by log level
            component: Filter by component name
            limit: Maximum number of entries to return

        Returns:
            List of log entry dictionaries
        """
        entries = self.get_logs(level=level, component=component, limit=limit)
        return [e.to_dict() for e in entries]

    def get_errors(self) -> List[str]:
        """Get all error messages."""
        errors = self.get_logs(level="ERROR")
        return [f"{e.event}: {e.message}" if e.message else e.event for e in errors]

    def get_warnings(self) -> List[str]:
        """Get all warning messages."""
        warnings = self.get_logs(level="WARNING")
        return [f"{e.event}: {e.message}" if e.message else e.event for e in warnings]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get log statistics.

        Returns:
            Dictionary with log counts by level and total
        """
        return {
            "total": sum(self._counts.values()),
            "by_level": self._counts.copy(),
            "buffer_size": len(self._buffer),
            "max_buffer_size": self._max_size,
        }

    def has_errors(self) -> bool:
        """Check if any errors were logged."""
        return self._counts["ERROR"] > 0 or self._counts["CRITICAL"] > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were logged."""
        return self._counts["WARNING"] > 0

    def save_to_file(self, path: Path) -> None:
        """
        Save logs to a JSONL file.

        Args:
            path: Path to the output file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for entry in self._buffer:
                line = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

    def save_partial(self, path: Path) -> None:
        """
        Save partial logs (for abnormal termination).

        Same as save_to_file but with a clear marker.

        Args:
            path: Path to the output file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            # Write marker
            marker = {
                "timestamp": datetime.now().isoformat(),
                "level": "WARNING",
                "event": "PARTIAL_LOG",
                "component": "PipelineLogCollector",
                "message": "Logs may be incomplete due to abnormal termination",
                "details": {"run_id": self._run_id},
            }
            f.write(json.dumps(marker, ensure_ascii=False) + "\n")

            # Write logs
            for entry in self._buffer:
                line = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

    @classmethod
    def load_from_file(cls, path: Path) -> List[Dict[str, Any]]:
        """
        Load logs from a JSONL file.

        Args:
            path: Path to the log file

        Returns:
            List of log entry dictionaries
        """
        path = Path(path)
        if not path.exists():
            return []

        logs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return logs

    def clear(self) -> None:
        """Clear all logs from the buffer."""
        self._buffer.clear()
        for level in self._counts:
            self._counts[level] = 0
