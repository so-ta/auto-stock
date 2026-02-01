"""
Backtest Exception Classes - Custom exceptions for strict error handling.

These exceptions enable the "immediate stop" mode where any error during
backtest execution causes the entire process to halt immediately.
"""

from __future__ import annotations


class BacktestError(Exception):
    """Base class for backtest execution errors.

    All backtest-specific errors should inherit from this class
    to enable unified error handling in UnifiedExecutor.
    """

    pass


class SignalEvaluationError(BacktestError):
    """Raised when signal evaluation fails for a specific asset/signal pair.

    This error captures the symbol, signal name, and original error message
    to provide detailed context for debugging.

    Attributes:
        symbol: The asset ticker symbol (e.g., "AAPL", "TZA")
        signal_name: The signal identifier (e.g., "rsi_14", "momentum_20")
        original_error: The original error message
    """

    def __init__(self, symbol: str, signal_name: str, original_error: str) -> None:
        self.symbol = symbol
        self.signal_name = signal_name
        self.original_error = original_error
        super().__init__(
            f"Signal evaluation failed for {symbol}/{signal_name}: {original_error}"
        )


class DataConversionError(BacktestError):
    """Raised when data conversion fails (e.g., pandas to polars).

    This error indicates issues with data format or type conversion
    that prevent the backtest from proceeding.

    Attributes:
        symbol: The asset ticker symbol (if applicable)
        operation: The operation that failed (e.g., "pandas_to_polars")
        original_error: The original error message
    """

    def __init__(
        self,
        operation: str,
        original_error: str,
        symbol: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.operation = operation
        self.original_error = original_error
        if symbol:
            message = f"Data conversion failed for {symbol} ({operation}): {original_error}"
        else:
            message = f"Data conversion failed ({operation}): {original_error}"
        super().__init__(message)


class PipelineStepError(BacktestError):
    """Raised when a pipeline step fails.

    This error captures which step in the pipeline failed
    to help identify the source of the problem.

    Attributes:
        step_name: The name of the pipeline step that failed
        original_error: The original error message
    """

    def __init__(self, step_name: str, original_error: str) -> None:
        self.step_name = step_name
        self.original_error = original_error
        super().__init__(f"Pipeline step '{step_name}' failed: {original_error}")
