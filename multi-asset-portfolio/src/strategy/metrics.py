"""Performance metrics calculation module.

This module implements evaluation metrics for strategy validation,
including risk-adjusted returns, drawdown analysis, and tail risk measures.

Features:
- Expected value (mean return, cost-adjusted)
- Volatility (annualized)
- Sharpe/Sortino ratios (with configurable annualization)
- Maximum drawdown (MDD)
- Tail risk (VaR/ES at configurable percentile)
- Turnover calculation
- Trade count and effective sample tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# Annualization factors
TRADING_DAYS_PER_YEAR = 252
SQRT_TRADING_DAYS = np.sqrt(TRADING_DAYS_PER_YEAR)


class GateStatus(Enum):
    """Status of gate check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass(frozen=True)
class GateConfig:
    """Configuration for hard gate thresholds.

    These are minimum requirements for strategy adoption.
    Strategies failing any gate are rejected.

    Attributes:
        min_trades: Minimum number of trades in test period.
        max_mdd: Maximum allowed drawdown (as positive decimal, e.g., 0.25 for 25%).
        min_expected_value: Minimum expected value after costs.
        min_sharpe: Minimum Sharpe ratio.
        min_sortino: Minimum Sortino ratio (optional).
        max_instability_periods: Max periods with negative returns out of N.
        instability_window: Window size N for instability check.
    """

    min_trades: int = 30
    max_mdd: float = 0.25
    min_expected_value: float = 0.0
    min_sharpe: Optional[float] = 0.5
    min_sortino: Optional[float] = None
    max_instability_periods: int = 3
    instability_window: int = 5


@dataclass
class GateResult:
    """Result of gate check for a single metric.

    Attributes:
        gate_name: Name of the gate checked.
        status: Pass/Fail/Warning status.
        threshold: Threshold value for the gate.
        actual_value: Actual value of the metric.
        message: Human-readable message about the result.
    """

    gate_name: str
    status: GateStatus
    threshold: float
    actual_value: float
    message: str


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a strategy.

    All return values are in decimal form (0.01 = 1%).

    Attributes:
        expected_value: Mean return (cost-adjusted if costs provided).
        expected_value_gross: Mean return before costs.
        volatility: Annualized volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio.
        max_drawdown: Maximum drawdown (positive value).
        var_95: 5% Value at Risk.
        es_95: 5% Expected Shortfall (CVaR).
        turnover: Average turnover rate.
        n_trades: Number of trades.
        n_samples: Number of return samples.
        win_rate: Proportion of positive returns.
        profit_factor: Ratio of gross profits to gross losses.
        calmar_ratio: Return / Max drawdown.
        total_return: Cumulative return over period.
        annualized_return: Annualized return.
    """

    expected_value: float
    expected_value_gross: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    es_95: float
    turnover: float
    n_trades: int
    n_samples: int
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    total_return: float
    annualized_return: float

    def check_gates(self, config: GateConfig) -> list[GateResult]:
        """Check if metrics pass all gates.

        Args:
            config: Gate configuration with thresholds.

        Returns:
            List of GateResult for each check.
        """
        results: list[GateResult] = []

        # Min trades gate
        results.append(
            GateResult(
                gate_name="min_trades",
                status=GateStatus.PASS if self.n_trades >= config.min_trades else GateStatus.FAIL,
                threshold=float(config.min_trades),
                actual_value=float(self.n_trades),
                message=f"Trades: {self.n_trades} (min: {config.min_trades})",
            )
        )

        # Max MDD gate
        results.append(
            GateResult(
                gate_name="max_mdd",
                status=GateStatus.PASS if self.max_drawdown <= config.max_mdd else GateStatus.FAIL,
                threshold=config.max_mdd,
                actual_value=self.max_drawdown,
                message=f"MDD: {self.max_drawdown:.2%} (max: {config.max_mdd:.2%})",
            )
        )

        # Min expected value gate
        results.append(
            GateResult(
                gate_name="min_expected_value",
                status=GateStatus.PASS if self.expected_value >= config.min_expected_value else GateStatus.FAIL,
                threshold=config.min_expected_value,
                actual_value=self.expected_value,
                message=f"E[R]: {self.expected_value:.4%} (min: {config.min_expected_value:.4%})",
            )
        )

        # Min Sharpe gate (if configured)
        if config.min_sharpe is not None:
            results.append(
                GateResult(
                    gate_name="min_sharpe",
                    status=GateStatus.PASS if self.sharpe_ratio >= config.min_sharpe else GateStatus.FAIL,
                    threshold=config.min_sharpe,
                    actual_value=self.sharpe_ratio,
                    message=f"Sharpe: {self.sharpe_ratio:.2f} (min: {config.min_sharpe:.2f})",
                )
            )

        # Min Sortino gate (if configured)
        if config.min_sortino is not None:
            results.append(
                GateResult(
                    gate_name="min_sortino",
                    status=GateStatus.PASS if self.sortino_ratio >= config.min_sortino else GateStatus.FAIL,
                    threshold=config.min_sortino,
                    actual_value=self.sortino_ratio,
                    message=f"Sortino: {self.sortino_ratio:.2f} (min: {config.min_sortino:.2f})",
                )
            )

        return results

    def passes_all_gates(self, config: GateConfig) -> bool:
        """Check if all gates pass.

        Args:
            config: Gate configuration.

        Returns:
            True if all gates pass.
        """
        results = self.check_gates(config)
        return all(r.status == GateStatus.PASS for r in results)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "expected_value": self.expected_value,
            "expected_value_gross": self.expected_value_gross,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "es_95": self.es_95,
            "turnover": self.turnover,
            "n_trades": self.n_trades,
            "n_samples": self.n_samples,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
        }


class MetricsCalculator:
    """Calculator for performance metrics.

    This class computes various performance metrics from return series,
    with support for cost adjustment and different annualization periods.

    Example:
        >>> calc = MetricsCalculator(annualization_factor=252)
        >>> metrics = calc.calculate(returns, positions)
        >>> if metrics.passes_all_gates(GateConfig()):
        ...     print("Strategy passes all gates")
    """

    def __init__(
        self,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = 0.0,
        var_percentile: float = 0.05,
    ) -> None:
        """Initialize the calculator.

        Args:
            annualization_factor: Number of periods per year.
            risk_free_rate: Risk-free rate (annualized).
            var_percentile: Percentile for VaR/ES (default 5%).
        """
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.var_percentile = var_percentile

        # Daily risk-free rate
        self._rf_daily = risk_free_rate / annualization_factor

    def calculate(
        self,
        returns: np.ndarray | pd.Series | Sequence[float],
        positions: Optional[np.ndarray | pd.Series | Sequence[float]] = None,
        cost_per_trade: float = 0.0,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Args:
            returns: Array of period returns (not cumulative).
            positions: Array of position values (for turnover calculation).
            cost_per_trade: Cost per unit of turnover (in return terms).

        Returns:
            PerformanceMetrics with all calculated values.
        """
        # Convert to numpy array
        if isinstance(returns, pd.Series):
            returns_arr = returns.values.astype(np.float64)
        else:
            returns_arr = np.asarray(returns, dtype=np.float64)

        # Remove NaN values
        valid_mask = ~np.isnan(returns_arr)
        returns_arr = returns_arr[valid_mask]

        n_samples = len(returns_arr)
        if n_samples == 0:
            return self._empty_metrics()

        # Process positions for turnover
        turnover = 0.0
        if positions is not None:
            if isinstance(positions, pd.Series):
                pos_arr = positions.values.astype(np.float64)
            else:
                pos_arr = np.asarray(positions, dtype=np.float64)

            # Apply same mask
            pos_arr = pos_arr[valid_mask[: len(pos_arr)]] if len(pos_arr) >= len(valid_mask) else pos_arr
            turnover = self._calculate_turnover(pos_arr)

        # Calculate cost-adjusted returns
        total_cost = turnover * cost_per_trade
        cost_per_period = total_cost / n_samples if n_samples > 0 else 0.0

        # Gross metrics
        mean_return_gross = float(np.mean(returns_arr))

        # Net metrics (cost-adjusted)
        mean_return_net = mean_return_gross - cost_per_period

        # Volatility (annualized)
        volatility = float(np.std(returns_arr, ddof=1) * np.sqrt(self.annualization_factor))

        # Sharpe ratio
        excess_returns = returns_arr - self._rf_daily
        sharpe_ratio = self._calculate_sharpe(excess_returns)

        # Sortino ratio
        sortino_ratio = self._calculate_sortino(excess_returns)

        # Drawdown analysis
        max_drawdown = self._calculate_max_drawdown(returns_arr)

        # Tail risk
        var_95 = self._calculate_var(returns_arr)
        es_95 = self._calculate_es(returns_arr)

        # Trade statistics
        n_trades = self._count_trades(positions) if positions is not None else n_samples

        # Win rate
        win_rate = float(np.mean(returns_arr > 0))

        # Profit factor
        profit_factor = self._calculate_profit_factor(returns_arr)

        # Total and annualized return
        total_return = self._calculate_total_return(returns_arr)
        annualized_return = self._annualize_return(total_return, n_samples)

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        return PerformanceMetrics(
            expected_value=mean_return_net,
            expected_value_gross=mean_return_gross,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            es_95=es_95,
            turnover=turnover,
            n_trades=n_trades,
            n_samples=n_samples,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
        )

    def _calculate_sharpe(self, excess_returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(excess_returns) < 2:
            return 0.0

        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / std
        return float(sharpe * np.sqrt(self.annualization_factor))

    def _calculate_sortino(self, excess_returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(excess_returns) < 2:
            return 0.0

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std
        return float(sortino * np.sqrt(self.annualization_factor))

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown as positive value."""
        if len(returns) == 0:
            return 0.0

        # Cumulative returns
        cum_returns = np.cumprod(1 + returns)

        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Drawdown series
        drawdowns = (running_max - cum_returns) / running_max

        return float(np.max(drawdowns))

    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk at configured percentile."""
        if len(returns) == 0:
            return 0.0

        # VaR is the negative of the percentile (loss)
        var = -float(np.percentile(returns, self.var_percentile * 100))
        return max(0.0, var)  # VaR should be positive (representing loss)

    def _calculate_es(self, returns: np.ndarray) -> float:
        """Calculate Expected Shortfall (CVaR) at configured percentile."""
        if len(returns) == 0:
            return 0.0

        var_threshold = np.percentile(returns, self.var_percentile * 100)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return self._calculate_var(returns)

        # ES is the negative of mean tail returns (loss)
        es = -float(np.mean(tail_returns))
        return max(0.0, es)

    def _calculate_turnover(self, positions: np.ndarray) -> float:
        """Calculate average turnover from position changes."""
        if len(positions) < 2:
            return 0.0

        # Position changes
        position_changes = np.abs(np.diff(positions))

        # Average turnover per period
        return float(np.mean(position_changes))

    def _count_trades(self, positions: Optional[np.ndarray | pd.Series | Sequence[float]]) -> int:
        """Count number of trades from position changes."""
        if positions is None:
            return 0

        if isinstance(positions, pd.Series):
            pos_arr = positions.values
        else:
            pos_arr = np.asarray(positions)

        if len(pos_arr) < 2:
            return 0

        # Count non-zero changes
        changes = np.diff(pos_arr)
        return int(np.sum(changes != 0))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        profits = returns[returns > 0]
        losses = returns[returns < 0]

        total_profit = np.sum(profits) if len(profits) > 0 else 0.0
        total_loss = -np.sum(losses) if len(losses) > 0 else 0.0

        if total_loss == 0:
            return float("inf") if total_profit > 0 else 0.0

        return float(total_profit / total_loss)

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total cumulative return."""
        if len(returns) == 0:
            return 0.0

        return float(np.prod(1 + returns) - 1)

    def _annualize_return(self, total_return: float, n_periods: int) -> float:
        """Annualize a total return."""
        if n_periods == 0:
            return 0.0

        years = n_periods / self.annualization_factor
        if years <= 0:
            return 0.0

        # Compound annual growth rate
        return float((1 + total_return) ** (1 / years) - 1)

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases."""
        return PerformanceMetrics(
            expected_value=0.0,
            expected_value_gross=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            es_95=0.0,
            turnover=0.0,
            n_trades=0,
            n_samples=0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            total_return=0.0,
            annualized_return=0.0,
        )


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate rolling performance metrics.

    Args:
        returns: Series of returns with datetime index.
        window: Rolling window size in periods.
        min_periods: Minimum periods for valid calculation.

    Returns:
        DataFrame with rolling metrics.
    """
    if min_periods is None:
        min_periods = window // 2

    calc = MetricsCalculator()

    def calc_window_metrics(window_returns: pd.Series) -> pd.Series:
        metrics = calc.calculate(window_returns.values)
        return pd.Series(
            {
                "sharpe": metrics.sharpe_ratio,
                "sortino": metrics.sortino_ratio,
                "volatility": metrics.volatility,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
            }
        )

    # Use rolling apply
    results = returns.rolling(window=window, min_periods=min_periods).apply(
        lambda x: calc.calculate(x).sharpe_ratio, raw=True
    )

    return pd.DataFrame({"rolling_sharpe": results})


def check_stability(
    period_returns: Sequence[float],
    max_negative_periods: int = 3,
    window: int = 5,
) -> tuple[bool, int]:
    """Check return stability across periods.

    Args:
        period_returns: Returns for each evaluation period.
        max_negative_periods: Maximum allowed negative periods.
        window: Window size for stability check.

    Returns:
        Tuple of (is_stable, count_of_negative_periods).
    """
    if len(period_returns) < window:
        return True, 0

    # Check last 'window' periods
    recent = list(period_returns)[-window:]
    negative_count = sum(1 for r in recent if r < 0)

    return negative_count <= max_negative_periods, negative_count
