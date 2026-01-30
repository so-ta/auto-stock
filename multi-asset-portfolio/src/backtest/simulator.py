"""
Portfolio Simulator - Time-series simulation for portfolio backtesting.

This module provides a realistic portfolio simulator that handles:
- Position management (shares held per symbol)
- Cash management
- Transaction cost calculation
- Rebalancing execution
- Daily value tracking

Example:
    >>> sim = PortfolioSimulator(initial_capital=10000.0, transaction_cost_bps=10.0)
    >>> prices = {"SPY": 450.0, "TLT": 100.0}
    >>> weights = {"SPY": 0.6, "TLT": 0.4}
    >>> result = sim.rebalance(weights, prices, datetime(2024, 1, 2))
    >>> print(f"Portfolio value: ${sim.get_portfolio_value(prices):.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DividendHandling(Enum):
    """How to handle dividend payments."""

    REINVEST = "reinvest"  # Automatically reinvest dividends in the same stock
    CASH = "cash"  # Keep dividends as cash
    IGNORE = "ignore"  # Ignore dividends (use adjusted prices instead)


@dataclass
class Trade:
    """Record of a single trade execution.

    Attributes:
        symbol: Ticker symbol
        date: Trade execution date
        side: "BUY" or "SELL"
        shares: Number of shares traded (absolute value)
        price: Execution price per share
        notional: Total trade value (shares × price)
        cost: Transaction cost
    """

    symbol: str
    date: datetime
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    notional: float
    cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat() if self.date else None,
            "side": self.side,
            "shares": self.shares,
            "price": self.price,
            "notional": self.notional,
            "cost": self.cost,
        }


@dataclass
class RebalanceResult:
    """Result of a rebalance operation.

    Attributes:
        date: Rebalance date
        trades: List of executed trades
        total_cost: Total transaction costs
        value_before: Portfolio value before rebalance
        value_after: Portfolio value after rebalance
        turnover: Portfolio turnover (0-2, one-way basis)
        old_weights: Weights before rebalance
        new_weights: Weights after rebalance
    """

    date: datetime
    trades: List[Trade] = field(default_factory=list)
    total_cost: float = 0.0
    value_before: float = 0.0
    value_after: float = 0.0
    turnover: float = 0.0
    old_weights: Dict[str, float] = field(default_factory=dict)
    new_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "trades": [t.to_dict() for t in self.trades],
            "total_cost": self.total_cost,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "turnover": self.turnover,
            "old_weights": self.old_weights,
            "new_weights": self.new_weights,
        }


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time.

    Attributes:
        date: Snapshot date
        value: Total portfolio value
        cash: Cash balance
        positions: Position values by symbol
        weights: Position weights by symbol
        daily_return: Return since previous snapshot
    """

    date: datetime
    value: float
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    daily_return: float = 0.0


class PortfolioSimulator:
    """
    Portfolio time-series simulator.

    Simulates portfolio evolution over time including:
    - Position tracking (shares held)
    - Cash management
    - Transaction costs
    - Rebalancing

    Attributes:
        initial_capital: Starting capital amount
        cost_bps: Transaction cost in basis points (1 bps = 0.01%)
        cash: Current cash balance
        positions: Current positions {symbol: shares}
        transaction_costs_total: Cumulative transaction costs
        trades_count: Total number of trades executed

    Example:
        >>> sim = PortfolioSimulator(10000.0, transaction_cost_bps=10.0)
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0}
        >>> weights = {"AAPL": 0.5, "MSFT": 0.5}
        >>> result = sim.rebalance(weights, prices, datetime.now())
        >>> print(f"Value: ${sim.get_portfolio_value(prices):.2f}")
    """

    # Reserved key for cash position in weights
    CASH_KEY = "CASH"

    def __init__(
        self,
        initial_capital: float,
        transaction_cost_bps: float = 10.0,
        dividend_handling: DividendHandling = DividendHandling.REINVEST,
    ):
        """
        Initialize portfolio simulator.

        Args:
            initial_capital: Starting capital amount
            transaction_cost_bps: Transaction cost in basis points (1 bps = 0.01%)
                                  Default: 10 bps (0.1%)
            dividend_handling: How to handle dividend payments
                              Default: REINVEST (auto-reinvest in same stock)
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")

        self.initial_capital = initial_capital
        self.cost_bps = transaction_cost_bps
        self.dividend_handling = dividend_handling
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> shares
        self.transaction_costs_total = 0.0
        self.trades_count = 0

        # Dividend tracking
        self.dividends_received_total = 0.0
        self._dividend_history: List[Dict[str, Any]] = []

        # History tracking
        self._history: List[PortfolioSnapshot] = []
        self._rebalance_history: List[RebalanceResult] = []

        logger.info(
            f"PortfolioSimulator initialized: "
            f"capital=${initial_capital:,.2f}, cost={transaction_cost_bps}bps, "
            f"dividend_handling={dividend_handling.value}"
        )

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        date: datetime,
    ) -> Dict[str, Any]:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Target allocation {symbol: weight}
                           e.g., {"SPY": 0.6, "TLT": 0.4, "CASH": 0.0}
                           Weights should sum to 1.0 (or close to it)
            prices: Current prices {symbol: price}
            date: Rebalance date

        Returns:
            Dictionary with rebalance details:
            - trades: List of executed trades
            - cost: Total transaction cost
            - value_before: Portfolio value before rebalance
            - value_after: Portfolio value after rebalance
            - turnover: Portfolio turnover

        Raises:
            ValueError: If prices are missing for target positions
        """
        # 1. Calculate current portfolio value
        value_before = self.get_portfolio_value(prices)
        old_weights = self.get_current_weights(prices)

        # 2. Validate target weights
        target_weights = self._normalize_weights(target_weights)

        # 3. Check that prices are available for all target positions
        for symbol, weight in target_weights.items():
            if symbol != self.CASH_KEY and weight > 0 and symbol not in prices:
                raise ValueError(f"Missing price for symbol: {symbol}")

        # 4. Calculate target values and required trades
        trades: List[Trade] = []
        total_cost = 0.0

        # Get cash target (default to 0 if not specified)
        target_cash_weight = target_weights.get(self.CASH_KEY, 0.0)

        # Calculate target equity value (excluding cash target)
        target_equity_value = value_before * (1 - target_cash_weight)

        # Calculate trades for each symbol
        for symbol, target_weight in target_weights.items():
            if symbol == self.CASH_KEY:
                continue

            price = prices[symbol]
            current_shares = self.positions.get(symbol, 0.0)
            current_value = current_shares * price

            # Target value for this symbol (proportion of equity value)
            # Recalculate weight relative to equity portion
            if (1 - target_cash_weight) > 0:
                equity_relative_weight = target_weight / (1 - target_cash_weight)
            else:
                equity_relative_weight = 0.0

            target_value = target_equity_value * equity_relative_weight
            target_shares = target_value / price if price > 0 else 0.0

            # Calculate trade
            shares_diff = target_shares - current_shares

            if abs(shares_diff) > 1e-8:  # Minimum trade threshold
                notional = abs(shares_diff * price)
                cost = self._calculate_cost(notional)

                trade = Trade(
                    symbol=symbol,
                    date=date,
                    side="BUY" if shares_diff > 0 else "SELL",
                    shares=abs(shares_diff),
                    price=price,
                    notional=notional,
                    cost=cost,
                )
                trades.append(trade)
                total_cost += cost

        # 5. Execute trades (update positions and cash)
        for trade in trades:
            if trade.side == "BUY":
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0.0) + trade.shares
                self.cash -= trade.notional + trade.cost
            else:  # SELL
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0.0) - trade.shares
                self.cash += trade.notional - trade.cost

            self.trades_count += 1

        # 6. Update totals
        self.transaction_costs_total += total_cost

        # 7. Calculate value after (accounting for costs)
        value_after = self.get_portfolio_value(prices)
        new_weights = self.get_current_weights(prices)

        # 8. Calculate turnover
        turnover = self.calculate_turnover(old_weights, new_weights)

        # 9. Create result
        result = RebalanceResult(
            date=date,
            trades=trades,
            total_cost=total_cost,
            value_before=value_before,
            value_after=value_after,
            turnover=turnover,
            old_weights=old_weights,
            new_weights=new_weights,
        )
        self._rebalance_history.append(result)

        logger.debug(
            f"Rebalance on {date}: {len(trades)} trades, "
            f"cost=${total_cost:.2f}, turnover={turnover:.2%}"
        )

        return result.to_dict()

    def update_value(
        self,
        prices: Dict[str, float],
        date: datetime,
    ) -> float:
        """
        Update and record portfolio value (no rebalancing).

        Args:
            prices: Current prices {symbol: price}
            date: Current date

        Returns:
            Current portfolio value
        """
        value = self.get_portfolio_value(prices)
        weights = self.get_current_weights(prices)

        # Calculate daily return if we have history
        daily_return = 0.0
        if self._history:
            prev_value = self._history[-1].value
            if prev_value > 0:
                daily_return = (value - prev_value) / prev_value

        # Record snapshot
        position_values = {
            symbol: shares * prices.get(symbol, 0.0)
            for symbol, shares in self.positions.items()
        }

        snapshot = PortfolioSnapshot(
            date=date,
            value=value,
            cash=self.cash,
            positions=position_values,
            weights=weights,
            daily_return=daily_return,
        )
        self._history.append(snapshot)

        return value

    def receive_dividend(
        self,
        symbol: str,
        dividend_per_share: float,
        prices: Dict[str, float],
        date: datetime,
    ) -> float:
        """
        Process a dividend payment for a position.

        Args:
            symbol: Stock symbol paying the dividend
            dividend_per_share: Dividend amount per share
            prices: Current prices {symbol: price} for reinvestment
            date: Dividend payment date

        Returns:
            Total dividend amount received
        """
        if self.dividend_handling == DividendHandling.IGNORE:
            # When using adjusted prices, dividends are already reflected
            logger.debug(f"Ignoring dividend for {symbol} (using adjusted prices)")
            return 0.0

        shares = self.positions.get(symbol, 0.0)
        if shares <= 0:
            return 0.0

        dividend_amount = shares * dividend_per_share
        self.dividends_received_total += dividend_amount

        # Record dividend
        dividend_record = {
            "date": date,
            "symbol": symbol,
            "shares": shares,
            "dividend_per_share": dividend_per_share,
            "amount": dividend_amount,
            "handling": self.dividend_handling.value,
        }

        if self.dividend_handling == DividendHandling.CASH:
            # Add dividend to cash balance
            self.cash += dividend_amount
            dividend_record["action"] = "added_to_cash"
            logger.info(
                f"Dividend received: {symbol} ${dividend_amount:.2f} -> cash"
            )

        elif self.dividend_handling == DividendHandling.REINVEST:
            # Reinvest dividend by buying more shares
            price = prices.get(symbol)
            if price and price > 0:
                # Calculate additional shares (allow fractional)
                additional_shares = dividend_amount / price
                self.positions[symbol] = shares + additional_shares
                dividend_record["action"] = "reinvested"
                dividend_record["additional_shares"] = additional_shares
                logger.info(
                    f"Dividend reinvested: {symbol} ${dividend_amount:.2f} "
                    f"-> {additional_shares:.4f} shares"
                )
            else:
                # No price available, add to cash instead
                self.cash += dividend_amount
                dividend_record["action"] = "added_to_cash_no_price"
                logger.warning(
                    f"Dividend for {symbol} added to cash (no price for reinvestment)"
                )

        self._dividend_history.append(dividend_record)
        return dividend_amount

    def process_dividends(
        self,
        dividends: Dict[str, float],
        prices: Dict[str, float],
        date: datetime,
    ) -> float:
        """
        Process multiple dividend payments.

        Args:
            dividends: Dividends per share {symbol: dividend_per_share}
            prices: Current prices {symbol: price}
            date: Dividend payment date

        Returns:
            Total dividend amount received
        """
        total = 0.0
        for symbol, div_per_share in dividends.items():
            if div_per_share > 0:
                total += self.receive_dividend(symbol, div_per_share, prices, date)
        return total

    def get_dividend_history(self) -> List[Dict[str, Any]]:
        """
        Get dividend history.

        Returns:
            List of dividend records
        """
        return self._dividend_history.copy()

    def get_current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current portfolio weights.

        Args:
            prices: Current prices {symbol: price}

        Returns:
            Weights {symbol: weight}, including CASH
        """
        total_value = self.get_portfolio_value(prices)

        if total_value <= 0:
            return {self.CASH_KEY: 1.0}

        weights: Dict[str, float] = {}

        # Calculate position weights
        for symbol, shares in self.positions.items():
            price = prices.get(symbol, 0.0)
            value = shares * price
            weights[symbol] = value / total_value

        # Add cash weight
        weights[self.CASH_KEY] = self.cash / total_value

        return weights

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            prices: Current prices {symbol: price}

        Returns:
            Total value (cash + position values)
        """
        position_value = sum(
            shares * prices.get(symbol, 0.0)
            for symbol, shares in self.positions.items()
        )
        return self.cash + position_value

    def get_daily_return(
        self,
        prices: Dict[str, float],
        prev_value: float,
    ) -> float:
        """
        Calculate daily return.

        Args:
            prices: Current prices {symbol: price}
            prev_value: Previous portfolio value

        Returns:
            Daily return (e.g., 0.005 = 0.5%)
        """
        if prev_value <= 0:
            return 0.0

        current_value = self.get_portfolio_value(prices)
        return (current_value - prev_value) / prev_value

    def calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """
        Calculate portfolio turnover.

        Turnover = Σ |w_new - w_old| / 2

        Args:
            old_weights: Previous weights
            new_weights: New weights

        Returns:
            Turnover (0-1 for one-way, 0-2 for round-trip)
        """
        # Get all symbols
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())

        # Calculate absolute weight changes
        total_change = sum(
            abs(new_weights.get(symbol, 0.0) - old_weights.get(symbol, 0.0))
            for symbol in all_symbols
        )

        # One-way turnover (divide by 2 since changes are counted twice)
        return total_change / 2

    def get_position_shares(self, symbol: str) -> float:
        """
        Get number of shares held for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Number of shares (0 if no position)
        """
        return self.positions.get(symbol, 0.0)

    def get_cash(self) -> float:
        """
        Get current cash balance.

        Returns:
            Cash balance
        """
        return self.cash

    def get_history(self) -> List[PortfolioSnapshot]:
        """
        Get portfolio history.

        Returns:
            List of PortfolioSnapshot objects
        """
        return self._history.copy()

    def get_rebalance_history(self) -> List[RebalanceResult]:
        """
        Get rebalance history.

        Returns:
            List of RebalanceResult objects
        """
        return self._rebalance_history.copy()

    def get_summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.

        Args:
            prices: Current prices

        Returns:
            Summary dictionary
        """
        current_value = self.get_portfolio_value(prices)
        total_return = (current_value - self.initial_capital) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_value": current_value,
            "cash": self.cash,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "transaction_costs_total": self.transaction_costs_total,
            "trades_count": self.trades_count,
            "positions_count": len(self.positions),
            "history_length": len(self._history),
            "rebalances_count": len(self._rebalance_history),
            "dividends_received_total": self.dividends_received_total,
            "dividend_payments_count": len(self._dividend_history),
            "dividend_handling": self.dividend_handling.value,
        }

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.transaction_costs_total = 0.0
        self.trades_count = 0
        self.dividends_received_total = 0.0
        self._dividend_history = []
        self._history = []
        self._rebalance_history = []

        logger.info("PortfolioSimulator reset to initial state")

    def _calculate_cost(self, notional: float) -> float:
        """
        Calculate transaction cost.

        Args:
            notional: Trade notional value

        Returns:
            Transaction cost
        """
        return notional * (self.cost_bps / 10000)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Input weights

        Returns:
            Normalized weights
        """
        total = sum(weights.values())

        if total <= 0:
            return {self.CASH_KEY: 1.0}

        if abs(total - 1.0) < 1e-6:
            return weights

        # Normalize
        return {symbol: weight / total for symbol, weight in weights.items()}
