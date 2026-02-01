"""
Structured Logging Module - Production-Ready Logging

This module provides comprehensive structured logging with:
- JSON output format for log aggregation
- Contextual logging with bound variables
- Execution timing and tracing
- Audit trail for all decisions

Required log items (from §10):
- Execution time, target period (train/test)
- Data quality inspection results
- Strategy×Asset evaluation metrics
- Strategy adoption/rejection reasons
- Asset allocation calculation basis
- Exception/fallback mode trigger reasons
- Random seed, code/config version
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import structlog

if TYPE_CHECKING:
    from src.config.settings import LoggingConfig, Settings
    from src.utils.pipeline_log_collector import PipelineLogCollector

# Context variables for request tracing
_run_id: ContextVar[str] = ContextVar("run_id", default="")
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

# Context variable for pipeline log collector
_log_collector: ContextVar[Optional["PipelineLogCollector"]] = ContextVar(
    "log_collector", default=None
)


def get_run_id() -> str:
    """Get current run ID."""
    return _run_id.get()


def set_run_id(run_id: str) -> None:
    """Set current run ID."""
    _run_id.set(run_id)


def get_log_collector() -> Optional["PipelineLogCollector"]:
    """Get current pipeline log collector."""
    return _log_collector.get()


def set_log_collector(collector: Optional["PipelineLogCollector"]) -> None:
    """Set current pipeline log collector."""
    _log_collector.set(collector)


def generate_run_id() -> str:
    """Generate a new unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{unique_id}"


class AuditLogger:
    """
    Specialized logger for audit trail logging.

    Records all decisions and their rationale for compliance
    and debugging purposes.
    """

    def __init__(self, base_logger: structlog.BoundLogger) -> None:
        self._logger = base_logger.bind(audit=True)

    def log_data_quality(
        self,
        symbol: str,
        status: str,
        metrics: dict[str, Any],
        excluded: bool = False,
        reason: str | None = None,
    ) -> None:
        """Log data quality inspection result."""
        self._logger.info(
            "data_quality_check",
            symbol=symbol,
            status=status,
            metrics=metrics,
            excluded=excluded,
            exclusion_reason=reason,
        )

    def log_strategy_evaluation(
        self,
        strategy_name: str,
        symbol: str,
        metrics: dict[str, Any],
        status: str,
        reason: str | None = None,
    ) -> None:
        """Log strategy evaluation result."""
        self._logger.info(
            "strategy_evaluation",
            strategy_name=strategy_name,
            symbol=symbol,
            metrics=metrics,
            status=status,
            rejection_reason=reason,
        )

    def log_signal_generation(
        self,
        total_assets: int,
        total_signals: int,
        optimized_signals: int,
        signal_names: list[str],
    ) -> None:
        """Log signal generation summary."""
        self._logger.info(
            "signal_generation",
            total_assets=total_assets,
            total_signals=total_signals,
            optimized_signals=optimized_signals,
            signal_names=signal_names,
        )

    def log_strategy_weight(
        self,
        strategy_name: str,
        symbol: str,
        score: float,
        weight: float,
        calculation_basis: dict[str, Any],
    ) -> None:
        """Log strategy weight calculation."""
        self._logger.info(
            "strategy_weight_calculation",
            strategy_name=strategy_name,
            symbol=symbol,
            score=score,
            weight=weight,
            calculation_basis=calculation_basis,
        )

    def log_asset_allocation(
        self,
        weights: dict[str, float],
        method: str,
        constraints: dict[str, Any],
        risk_metrics: dict[str, Any],
        fallback_reason: str | None = None,
        constraint_violations: int = 0,
        turnover: float = 0.0,
        cash_weight: float = 0.0,
        non_cash_assets: int = 0,
    ) -> None:
        """Log asset allocation decision.

        Args:
            weights: Final asset weights
            method: Allocation method used (HRP, risk_parity, etc.)
            constraints: Constraint configuration
            risk_metrics: Risk metrics from allocation
            fallback_reason: Reason if fallback was triggered
            constraint_violations: Number of constraint violations
            turnover: Portfolio turnover
            cash_weight: Weight allocated to CASH
            non_cash_assets: Number of non-CASH assets
        """
        self._logger.info(
            "asset_allocation",
            weights=weights,
            method=method,
            constraints=constraints,
            risk_metrics=risk_metrics,
            fallback_reason=fallback_reason,
            constraint_violations=constraint_violations,
            turnover=round(turnover, 4),
            cash_weight=round(cash_weight, 4),
            non_cash_assets=non_cash_assets,
        )

    def log_fallback_triggered(
        self,
        mode: str,
        trigger_reason: str,
        previous_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> None:
        """Log fallback mode activation."""
        self._logger.warning(
            "fallback_mode_triggered",
            mode=mode,
            trigger_reason=trigger_reason,
            previous_weights=previous_weights,
            new_weights=new_weights,
        )

    def log_anomaly_detected(
        self,
        anomaly_type: str,
        details: dict[str, Any],
        severity: str,
        action_taken: str,
    ) -> None:
        """Log anomaly detection."""
        log_method = self._logger.error if severity == "critical" else self._logger.warning
        log_method(
            "anomaly_detected",
            anomaly_type=anomaly_type,
            details=details,
            severity=severity,
            action_taken=action_taken,
        )

    def log_pipeline_step(
        self,
        step_name: str,
        step_number: int,
        status: str,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log pipeline step execution."""
        self._logger.info(
            "pipeline_step",
            step_name=step_name,
            step_number=step_number,
            status=status,
            duration_ms=duration_ms,
            details=details or {},
        )

    def log_run_summary(
        self,
        run_id: str,
        status: str,
        duration_seconds: float,
        assets_processed: int,
        strategies_evaluated: int,
        strategies_adopted: int,
        fallback_mode: str | None,
        output_weights: dict[str, float],
    ) -> None:
        """Log complete run summary."""
        self._logger.info(
            "run_summary",
            run_id=run_id,
            status=status,
            duration_seconds=duration_seconds,
            assets_processed=assets_processed,
            strategies_evaluated=strategies_evaluated,
            strategies_adopted=strategies_adopted,
            fallback_mode=fallback_mode,
            output_weights=output_weights,
        )

    def log_gate_rejection(
        self,
        strategy_id: str,
        asset_id: str,
        reasons: list[str],
        metrics: dict[str, Any],
    ) -> None:
        """Log strategy rejection by hard gate."""
        self._logger.warning(
            "gate_rejection",
            strategy_id=strategy_id,
            asset_id=asset_id,
            rejection_reasons=reasons,
            metrics=metrics,
        )

    def log_gate_summary(
        self,
        total_evaluated: int,
        total_passed: int,
        total_rejected: int,
        gate_config: dict[str, Any],
    ) -> None:
        """Log gate check summary."""
        self._logger.info(
            "gate_check_summary",
            total_evaluated=total_evaluated,
            total_passed=total_passed,
            total_rejected=total_rejected,
            pass_rate=round(total_passed / total_evaluated * 100, 1) if total_evaluated > 0 else 0.0,
            gate_config=gate_config,
        )

    def log_strategy_weighting(
        self,
        asset_id: str,
        scores: dict[str, float],
        raw_weights: dict[str, float],
        final_weights: dict[str, float],
        entropy_before: float,
        entropy_after: float,
        was_adjusted: bool,
        adopted_count: int,
    ) -> None:
        """Log strategy weighting calculation for an asset.

        Args:
            asset_id: Asset identifier
            scores: Strategy scores (strategy_id -> score)
            raw_weights: Weights before entropy adjustment
            final_weights: Weights after entropy adjustment
            entropy_before: Normalized entropy before adjustment
            entropy_after: Normalized entropy after adjustment
            was_adjusted: Whether entropy adjustment was applied
            adopted_count: Number of strategies with weight > 0
        """
        self._logger.info(
            "strategy_weighting",
            asset_id=asset_id,
            strategy_count=len(scores),
            scores=scores,
            raw_weights=raw_weights,
            final_weights=final_weights,
            entropy_before=round(entropy_before, 4),
            entropy_after=round(entropy_after, 4),
            was_adjusted=was_adjusted,
            adopted_count=adopted_count,
        )

    def log_top_n_selection(
        self,
        n: int,
        cash_score: float,
        selected_count: int,
        excluded_count: int,
        cash_selected: bool,
        cash_weight: float,
        weights: dict[str, float],
        strategy_weights: dict[str, float],
    ) -> None:
        """Log TopNSelector strategy selection results.

        Args:
            n: Maximum number of strategies to select
            cash_score: Fixed score for CASH position
            selected_count: Number of strategies selected
            excluded_count: Number of strategies excluded
            cash_selected: Whether CASH was selected
            cash_weight: Weight assigned to CASH
            weights: Asset-level weights
            strategy_weights: Strategy-level weights
        """
        self._logger.info(
            "top_n_selection",
            n=n,
            cash_score=cash_score,
            selected_count=selected_count,
            excluded_count=excluded_count,
            cash_selected=cash_selected,
            cash_weight=round(cash_weight, 4),
            asset_count=len(weights),
            strategy_count=len(strategy_weights),
            weights=weights,
        )

    def log_risk_estimation(
        self,
        covariance_method: str,
        n_assets: int,
        n_samples: int,
        shrinkage_intensity: float | None,
        portfolio_volatility: float,
        var_95: float,
        expected_shortfall: float,
    ) -> None:
        """Log risk estimation results.

        Args:
            covariance_method: Method used for covariance estimation
            n_assets: Number of assets
            n_samples: Number of samples used
            shrinkage_intensity: Ledoit-Wolf shrinkage intensity
            portfolio_volatility: Portfolio volatility (equal-weighted baseline)
            var_95: Value at Risk (95%)
            expected_shortfall: Expected Shortfall (95%)
        """
        self._logger.info(
            "risk_estimation",
            covariance_method=covariance_method,
            n_assets=n_assets,
            n_samples=n_samples,
            shrinkage_intensity=round(shrinkage_intensity or 0, 4),
            portfolio_volatility=round(portfolio_volatility, 4),
            var_95=round(var_95, 6),
            expected_shortfall=round(expected_shortfall, 6),
        )


def add_timestamp(
    logger: structlog.BoundLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO 8601 timestamp to log entry."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_run_id(
    logger: structlog.BoundLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add run ID to log entry."""
    run_id = _run_id.get()
    if run_id:
        event_dict["run_id"] = run_id
    return event_dict


def add_log_level(
    logger: structlog.BoundLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


class JSONFileRenderer:
    """Render logs to JSON file."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        logger: structlog.BoundLogger,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> str:
        """Render and write to file."""
        line = json.dumps(event_dict, default=str, ensure_ascii=False)

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        return line


def setup_logging(
    settings: "Settings | None" = None,
    log_file: Path | str | None = None,
    json_format: bool = True,
    level: str = "INFO",
    enable_log_collector: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        settings: Optional settings instance
        log_file: Optional path to log file
        json_format: Whether to use JSON format
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_log_collector: Whether to enable PipelineLogCollector integration
    """
    # Get config from settings if provided
    if settings is not None:
        log_config = settings.logging
        level = log_config.level
        json_format = log_config.json_format
        if log_file is None:
            log_file = log_config.log_dir / "system.log"

    # Set up processors
    shared_processors: list[structlog.types.Processor] = [
        add_log_level,
        add_timestamp,
        add_run_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add log collector processor for PipelineLogCollector integration
    if enable_log_collector:
        shared_processors.insert(0, log_collector_processor)

    if json_format:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(
                serializer=lambda obj, **kw: json.dumps(obj, default=str, ensure_ascii=False, **kw)
            )
        ]
    else:
        # Console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(level.upper(), logging.INFO)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up standard logging bridge
    setup_standard_logging_bridge(level=level)


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        **initial_values: Initial bound values

    Returns:
        Configured structlog BoundLogger
    """
    logger = structlog.get_logger(name)

    if initial_values:
        logger = logger.bind(**initial_values)

    return logger


def get_audit_logger(name: str | None = None) -> AuditLogger:
    """
    Get an audit logger instance.

    Args:
        name: Logger name

    Returns:
        AuditLogger instance
    """
    base_logger = get_logger(name)
    return AuditLogger(base_logger)


class LogContext:
    """
    Context manager for adding temporary log context.

    Example:
        >>> with LogContext(operation="data_fetch", symbol="BTCUSD"):
        ...     logger.info("Fetching data")
        ...     # All logs in this block will have operation and symbol
    """

    def __init__(self, **context: Any) -> None:
        self.context = context
        self._token: structlog.contextvars.Token | None = None

    def __enter__(self) -> "LogContext":
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


class TimedOperation:
    """
    Context manager for timing operations.

    Example:
        >>> with TimedOperation("data_fetch") as timer:
        ...     fetch_data()
        >>> print(f"Duration: {timer.duration_ms}ms")
    """

    def __init__(self, operation_name: str, logger: structlog.BoundLogger | None = None) -> None:
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.duration_ms: float = 0.0

    def __enter__(self) -> "TimedOperation":
        self.start_time = datetime.now(timezone.utc)
        self.logger.debug(f"{self.operation_name}_started")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        status = "success" if exc_type is None else "error"
        self.logger.info(
            f"{self.operation_name}_completed",
            status=status,
            duration_ms=round(self.duration_ms, 2),
            error=str(exc_val) if exc_val else None,
        )


def log_to_file(
    log_entry: dict[str, Any],
    file_path: Path | str,
) -> None:
    """
    Append a log entry to a JSON Lines file.

    Args:
        log_entry: Log entry as dictionary
        file_path: Path to log file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, default=str, ensure_ascii=False) + "\n")


def read_logs(
    file_path: Path | str,
    run_id: str | None = None,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Read and filter logs from a JSON Lines file.

    Args:
        file_path: Path to log file
        run_id: Filter by run ID
        event_type: Filter by event type

    Returns:
        List of matching log entries
    """
    path = Path(file_path)
    if not path.exists():
        return []

    logs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Apply filters
                if run_id and entry.get("run_id") != run_id:
                    continue
                if event_type and entry.get("event") != event_type:
                    continue

                logs.append(entry)
            except json.JSONDecodeError:
                continue

    return logs


# =============================================================================
# Standard logging → structlog bridge (for 60+ files using logging.getLogger)
# =============================================================================


class StructlogHandler(logging.Handler):
    """
    Standard logging handler that forwards logs to structlog.

    This enables automatic integration of logs from modules using
    logging.getLogger(__name__) into the structlog/PipelineLogCollector system.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a log record to structlog."""
        try:
            # Get structlog logger
            logger = structlog.get_logger(record.name)

            # Map log level to structlog method
            level_method_map = {
                logging.DEBUG: logger.debug,
                logging.INFO: logger.info,
                logging.WARNING: logger.warning,
                logging.ERROR: logger.error,
                logging.CRITICAL: logger.critical,
            }

            log_method = level_method_map.get(record.levelno, logger.info)

            # Forward to structlog with component info
            log_method(
                record.getMessage(),
                component=record.name,
                exc_info=record.exc_info if record.exc_info else None,
            )
        except Exception:
            # Avoid infinite recursion if structlog itself fails
            pass


def log_collector_processor(
    logger: structlog.BoundLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Structlog processor that forwards logs to PipelineLogCollector.

    This processor is added to the structlog chain and automatically
    sends all log entries to the current PipelineLogCollector (if set).
    """
    collector = _log_collector.get()
    if collector is not None:
        collector.log(
            level=method_name.upper(),
            event=event_dict.get("event", ""),
            component=event_dict.get("component", ""),
            **{k: v for k, v in event_dict.items() if k not in ("event", "component")}
        )
    return event_dict


def setup_standard_logging_bridge(level: str = "INFO") -> None:
    """
    Configure standard logging to forward to structlog.

    This should be called after setup_logging() to bridge logs from
    modules that use logging.getLogger() instead of structlog.

    Args:
        level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR)
    """
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(level.upper(), logging.INFO)

    # Get root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    # (Keep only our StructlogHandler)
    structlog_handler = None
    for handler in root_logger.handlers[:]:
        if isinstance(handler, StructlogHandler):
            structlog_handler = handler
        else:
            root_logger.removeHandler(handler)

    # Add our handler if not already present
    if structlog_handler is None:
        root_logger.addHandler(StructlogHandler())
