"""
Backtest Result Module.

Provides data structures for storing and analyzing backtest results,
including daily snapshots, performance metrics, and visualization.

Numba JIT Acceleration (v1.1.0+):
- _calculate_max_drawdown() uses Numba JIT for faster computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# Use unified metrics module
from src.utils.metrics import calculate_max_drawdown


@dataclass
class DailySnapshot:
    """
    Daily portfolio snapshot.

    Stores the state of the portfolio at the end of each trading day,
    including weights, values, and returns.
    """

    date: datetime
    weights: dict[str, float]
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    cash_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "weights": self.weights,
            "portfolio_value": self.portfolio_value,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "cash_weight": self.cash_weight,
        }


@dataclass
class BacktestResult:
    """
    Backtest result container.

    Stores complete backtest results including configuration, time series data,
    performance metrics, and trading statistics. Provides methods for analysis
    and visualization.

    Example:
        >>> result = BacktestResult(
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     rebalance_frequency="monthly",
        ...     initial_capital=100000.0
        ... )
        >>> result.snapshots.append(snapshot)
        >>> result.calculate_metrics()
        >>> print(result.summary())
    """

    # Configuration
    start_date: datetime
    end_date: datetime
    rebalance_frequency: str
    initial_capital: float

    # Time series data
    snapshots: list[DailySnapshot] = field(default_factory=list)

    # Performance metrics (calculated after backtest)
    final_value: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # Trading statistics
    total_trades: int = 0
    turnover: float = 0.0
    transaction_costs: float = 0.0
    num_rebalances: int = 0

    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def calculate_metrics(self, risk_free_rate: float = 0.02) -> None:
        """
        Calculate all performance metrics from snapshots.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        if not self.snapshots:
            return

        # Extract daily returns
        daily_returns = np.array([s.daily_return for s in self.snapshots])
        portfolio_values = np.array([s.portfolio_value for s in self.snapshots])

        # Basic metrics
        self.final_value = portfolio_values[-1]
        self.total_return = (self.final_value - self.initial_capital) / self.initial_capital

        # Number of trading days
        n_days = len(daily_returns)
        if n_days == 0:
            return

        # Annualized return
        # (1 + total_return)^(252/days) - 1
        if n_days > 0:
            self.annualized_return = (1 + self.total_return) ** (252 / n_days) - 1

        # Volatility (annualized)
        daily_std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
        self.volatility = daily_std * np.sqrt(252)

        # Sharpe Ratio
        # (annualized_return - risk_free_rate) / annualized_volatility
        if self.volatility > 0:
            self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.volatility
        else:
            self.sharpe_ratio = 0.0

        # Sortino Ratio (using downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns, ddof=1)
            downside_deviation = downside_std * np.sqrt(252)
            if downside_deviation > 0:
                self.sortino_ratio = (self.annualized_return - risk_free_rate) / downside_deviation
            else:
                self.sortino_ratio = 0.0
        else:
            # No negative returns
            self.sortino_ratio = float("inf") if self.annualized_return > risk_free_rate else 0.0

        # Max Drawdown
        self.max_drawdown = self._calculate_max_drawdown(portfolio_values)

        # Calmar Ratio
        # annualized_return / |max_drawdown|
        if abs(self.max_drawdown) > 0:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
        else:
            self.calmar_ratio = float("inf") if self.annualized_return > 0 else 0.0

        # VaR (95%)
        self.var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0.0

        # Expected Shortfall (CVaR 95%)
        var_threshold = np.percentile(daily_returns, 5)
        tail_returns = daily_returns[daily_returns <= var_threshold]
        self.expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0.0

        # Win Rate
        winning_days = np.sum(daily_returns > 0)
        total_days = len(daily_returns)
        self.win_rate = winning_days / total_days if total_days > 0 else 0.0

        # Profit Factor
        gains = np.sum(daily_returns[daily_returns > 0])
        losses = abs(np.sum(daily_returns[daily_returns < 0]))
        self.profit_factor = gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown from portfolio values.

        Args:
            portfolio_values: Array of portfolio values

        Returns:
            Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
        """
        if len(portfolio_values) == 0:
            return 0.0

        # Use unified metrics module (returns positive value, negate for this interface)
        max_dd = calculate_max_drawdown(portfolio_values=portfolio_values)
        return -max_dd  # Return negative as per original interface

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert daily snapshots to DataFrame.

        Returns:
            DataFrame with columns: date, portfolio_value, daily_return,
            cumulative_return, drawdown
        """
        if not self.snapshots:
            return pd.DataFrame(
                columns=["date", "portfolio_value", "daily_return", "cumulative_return", "drawdown"]
            )

        data = []
        portfolio_values = [s.portfolio_value for s in self.snapshots]
        running_max = np.maximum.accumulate(portfolio_values)

        for i, snapshot in enumerate(self.snapshots):
            drawdown = (snapshot.portfolio_value - running_max[i]) / running_max[i]
            data.append(
                {
                    "date": snapshot.date,
                    "portfolio_value": snapshot.portfolio_value,
                    "daily_return": snapshot.daily_return,
                    "cumulative_return": snapshot.cumulative_return,
                    "drawdown": drawdown,
                }
            )

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve with drawdown visualization.

        Args:
            save_path: If provided, save the plot to this path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        if not self.snapshots:
            print("No snapshots to plot")
            return

        df = self.to_dataframe()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        # Equity curve
        ax1 = axes[0]
        ax1.plot(df.index, df["portfolio_value"], label="Portfolio Value", color="blue", linewidth=1.5)
        ax1.fill_between(df.index, self.initial_capital, df["portfolio_value"], alpha=0.3, color="blue")
        ax1.axhline(y=self.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
        ax1.set_ylabel("Portfolio Value")
        ax1.set_title(f"Backtest Results: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(df.index, 0, df["drawdown"] * 100, color="red", alpha=0.5)
        ax2.plot(df.index, df["drawdown"] * 100, color="red", linewidth=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        # Add metrics text box
        metrics_text = (
            f"Total Return: {self.total_return * 100:.2f}%\n"
            f"Ann. Return: {self.annualized_return * 100:.2f}%\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown * 100:.2f}%"
        )
        ax1.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns from daily data.

        Returns:
            Series of monthly returns indexed by month-end date
        """
        if not self.snapshots:
            return pd.Series(dtype=float)

        df = self.to_dataframe()

        # Resample to monthly and calculate returns
        monthly_values = df["portfolio_value"].resample("ME").last()
        monthly_returns = monthly_values.pct_change()

        # First month return from initial capital
        if len(monthly_values) > 0:
            first_month_return = (monthly_values.iloc[0] - self.initial_capital) / self.initial_capital
            monthly_returns.iloc[0] = first_month_return

        return monthly_returns

    def get_drawdown_series(self) -> pd.Series:
        """
        Get the drawdown time series.

        Returns:
            Series of drawdown values indexed by date
        """
        df = self.to_dataframe()
        return df["drawdown"] if "drawdown" in df.columns else pd.Series(dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dictionary containing all result data
        """
        return {
            # Configuration
            "start_date": self.start_date.isoformat() if isinstance(self.start_date, datetime) else str(self.start_date),
            "end_date": self.end_date.isoformat() if isinstance(self.end_date, datetime) else str(self.end_date),
            "rebalance_frequency": self.rebalance_frequency,
            "initial_capital": self.initial_capital,
            # Performance metrics
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio if self.sortino_ratio != float("inf") else None,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio if self.calmar_ratio != float("inf") else None,
            "volatility": self.volatility,
            # Risk metrics
            "var_95": self.var_95,
            "expected_shortfall": self.expected_shortfall,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor if self.profit_factor != float("inf") else None,
            # Trading statistics
            "total_trades": self.total_trades,
            "turnover": self.turnover,
            "transaction_costs": self.transaction_costs,
            "num_rebalances": self.num_rebalances,
            # Snapshot count (not full snapshots for JSON size)
            "num_snapshots": len(self.snapshots),
        }

    def summary(self) -> str:
        """
        Generate a human-readable summary of the backtest results.

        Returns:
            Formatted string containing key metrics
        """
        lines = [
            "=" * 60,
            "  BACKTEST RESULT SUMMARY",
            "=" * 60,
            "",
            f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            f"  Rebalance: {self.rebalance_frequency}",
            f"  Initial Capital: ${self.initial_capital:,.2f}",
            f"  Final Value: ${self.final_value:,.2f}",
            "",
            "-" * 60,
            "  RETURNS",
            "-" * 60,
            f"  Total Return:      {self.total_return * 100:>10.2f}%",
            f"  Annualized Return: {self.annualized_return * 100:>10.2f}%",
            f"  Volatility (ann.): {self.volatility * 100:>10.2f}%",
            "",
            "-" * 60,
            "  RISK-ADJUSTED METRICS",
            "-" * 60,
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.3f}" if self.sortino_ratio != float("inf") else f"  Sortino Ratio:          inf",
            f"  Calmar Ratio:      {self.calmar_ratio:>10.3f}" if self.calmar_ratio != float("inf") else f"  Calmar Ratio:           inf",
            "",
            "-" * 60,
            "  RISK METRICS",
            "-" * 60,
            f"  Max Drawdown:      {self.max_drawdown * 100:>10.2f}%",
            f"  VaR (95%):         {self.var_95 * 100:>10.2f}%",
            f"  Expected Shortfall:{self.expected_shortfall * 100:>10.2f}%",
            f"  Win Rate:          {self.win_rate * 100:>10.2f}%",
            "",
            "-" * 60,
            "  TRADING STATISTICS",
            "-" * 60,
            f"  Total Trades:      {self.total_trades:>10d}",
            f"  Num Rebalances:    {self.num_rebalances:>10d}",
            f"  Turnover:          {self.turnover * 100:>10.2f}%",
            f"  Transaction Costs: ${self.transaction_costs:>10.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the result."""
        return (
            f"BacktestResult("
            f"period={self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}, "
            f"return={self.total_return * 100:.2f}%, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown * 100:.2f}%)"
        )
