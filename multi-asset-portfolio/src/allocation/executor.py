"""
Allocation Executor - VIX-integrated portfolio allocation execution.

This module integrates VIX-based risk management into the portfolio
allocation process. It acts as the final step before trade execution,
applying VIX-based cash allocation adjustments to computed weights.

Key Features:
- VIX signal integration for dynamic cash allocation
- Drawdown control integration
- Position limit enforcement
- Smooth transition between allocations
- Emergency risk-off triggers

Based on IMP-003: VIX-based dynamic cash allocation enhancement.

Usage:
    executor = AllocationExecutor()
    adjusted_weights = executor.execute(
        weights=computed_weights,
        vix_data=vix_df,
        portfolio_value=1000000.0,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.signals.vix_signal import (
    EnhancedVIXSignal,
    VIXSignalConfig,
    VIXSignalResult,
    calculate_vix_adjusted_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for allocation executor.

    Attributes:
        vix_adjustment_enabled: Enable VIX-based cash allocation
        drawdown_control_enabled: Enable drawdown-based risk reduction
        position_limits_enabled: Enable position size limits
        smooth_transitions: Enable gradual weight changes
        transition_speed: Speed of transitions (0.0 to 1.0)
        max_single_asset: Maximum weight for single asset
        max_sector: Maximum weight for single sector
        max_cash: Maximum cash allocation
        min_invested: Minimum invested allocation
        cash_symbol: Symbol for cash position
        config_path: Path to risk_params.yaml
    """

    vix_adjustment_enabled: bool = True
    drawdown_control_enabled: bool = True
    position_limits_enabled: bool = True
    smooth_transitions: bool = True
    transition_speed: float = 0.3
    max_single_asset: float = 0.20
    max_sector: float = 0.30
    max_cash: float = 0.80
    min_invested: float = 0.20
    cash_symbol: str = "CASH"
    config_path: str = "config/risk_params.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExecutorConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        position_limits = data.get("position_limits", {})
        drawdown = data.get("drawdown_control", {})

        return cls(
            vix_adjustment_enabled=True,
            drawdown_control_enabled=drawdown.get("trailing_stop", {}).get("enabled", True),
            max_single_asset=position_limits.get("max_single_asset", 0.20),
            max_sector=position_limits.get("max_sector", 0.30),
            max_cash=position_limits.get("max_cash", 0.80),
            min_invested=position_limits.get("min_invested", 0.20),
            config_path=str(path),
        )


@dataclass
class ExecutionResult:
    """Result from allocation execution.

    Attributes:
        weights: Final adjusted weights
        cash_allocation: Final cash allocation
        vix_signal: VIX signal result
        adjustments_applied: List of adjustments applied
        original_weights: Original weights before adjustment
        metadata: Additional execution metadata
    """

    weights: dict[str, float]
    cash_allocation: float
    vix_signal: Optional[VIXSignalResult] = None
    adjustments_applied: list[str] = field(default_factory=list)
    original_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def invested_allocation(self) -> float:
        """Return total invested (non-cash) allocation."""
        return 1.0 - self.cash_allocation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "cash_allocation": self.cash_allocation,
            "invested_allocation": self.invested_allocation,
            "adjustments_applied": self.adjustments_applied,
            "vix_signal": {
                "vix_level": self.vix_signal.vix_level if self.vix_signal else None,
                "vix_change": self.vix_signal.vix_change if self.vix_signal else None,
                "emergency_triggered": self.vix_signal.emergency_triggered if self.vix_signal else False,
            },
            "metadata": self.metadata,
        }


class AllocationExecutor:
    """
    Portfolio allocation executor with VIX integration.

    This class is responsible for the final step of portfolio allocation,
    applying risk management adjustments before trade execution.

    Processing Flow:
    1. Receive computed weights from allocator
    2. Apply VIX-based cash allocation adjustment
    3. Apply drawdown control if triggered
    4. Enforce position limits
    5. Smooth transitions if enabled
    6. Return final executable weights

    Example:
        executor = AllocationExecutor()

        # With VIX data
        result = executor.execute(
            weights={"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
            vix=25.5,
            vix_change=0.15,
        )
        print(f"Adjusted weights: {result.weights}")
        print(f"Cash allocation: {result.cash_allocation:.1%}")

        # With drawdown control
        result = executor.execute(
            weights={"SPY": 0.4, "TLT": 0.3, "GLD": 0.3},
            vix=18.0,
            current_drawdown=-0.18,
        )
    """

    def __init__(self, config: Optional[ExecutorConfig] = None):
        """Initialize executor.

        Args:
            config: Executor configuration
        """
        self.config = config or ExecutorConfig()
        self._vix_signal = EnhancedVIXSignal.from_config(self.config.config_path)
        self._previous_weights: dict[str, float] = {}
        self._high_water_mark: float = 0.0
        self._logger = logger

    @classmethod
    def from_config(cls, config_path: str | Path) -> "AllocationExecutor":
        """Create executor from YAML configuration.

        Args:
            config_path: Path to risk_params.yaml

        Returns:
            Configured AllocationExecutor
        """
        config = ExecutorConfig.from_yaml(config_path)
        return cls(config=config)

    def execute(
        self,
        weights: dict[str, float],
        vix: float = 15.0,
        vix_change: float = 0.0,
        vix_change_weekly: float = 0.0,
        current_drawdown: float = 0.0,
        portfolio_value: float = 0.0,
        portfolio_return: float = 0.0,
        sector_map: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute allocation with risk management adjustments.

        Args:
            weights: Computed portfolio weights (symbol -> weight)
            vix: Current VIX level
            vix_change: Daily VIX percentage change
            vix_change_weekly: Weekly VIX percentage change
            current_drawdown: Current portfolio drawdown from high
            portfolio_value: Current portfolio value (for HWM tracking)
            portfolio_return: Today's portfolio return
            sector_map: Optional mapping of symbol -> sector

        Returns:
            ExecutionResult with adjusted weights
        """
        adjustments_applied = []
        original_weights = weights.copy()
        current_weights = weights.copy()

        # Track high water mark
        if portfolio_value > self._high_water_mark:
            self._high_water_mark = portfolio_value

        # Step 1: VIX-based cash allocation
        vix_result = None
        if self.config.vix_adjustment_enabled:
            vix_result = self._vix_signal.get_cash_allocation(
                vix=vix,
                vix_change=vix_change,
                vix_change_weekly=vix_change_weekly,
                portfolio_return=portfolio_return,
            )

            if vix_result.cash_allocation > 0:
                current_weights = self._apply_cash_allocation(
                    current_weights, vix_result.cash_allocation
                )
                adjustments_applied.append(f"VIX cash allocation: {vix_result.cash_allocation:.1%}")

                if vix_result.emergency_triggered:
                    adjustments_applied.append(
                        f"Emergency trigger: {vix_result.emergency_type}"
                    )

        # Step 2: Drawdown control
        if self.config.drawdown_control_enabled and current_drawdown < -0.15:
            dd_cash = self._calculate_drawdown_cash(current_drawdown)
            current_cash = current_weights.get(self.config.cash_symbol, 0.0)

            if dd_cash > current_cash:
                current_weights = self._apply_cash_allocation(current_weights, dd_cash)
                adjustments_applied.append(f"Drawdown control: {dd_cash:.1%} cash")

        # Step 3: Position limits
        if self.config.position_limits_enabled:
            current_weights, limit_adjustments = self._apply_position_limits(
                current_weights, sector_map
            )
            adjustments_applied.extend(limit_adjustments)

        # Step 4: Smooth transitions
        if self.config.smooth_transitions and self._previous_weights:
            current_weights = self._smooth_transition(current_weights)
            adjustments_applied.append("Smooth transition applied")

        # Step 5: Final normalization
        current_weights = self._normalize_weights(current_weights)

        # Update state
        self._previous_weights = current_weights.copy()

        # Calculate final cash allocation
        final_cash = current_weights.get(self.config.cash_symbol, 0.0)

        return ExecutionResult(
            weights=current_weights,
            cash_allocation=final_cash,
            vix_signal=vix_result,
            adjustments_applied=adjustments_applied,
            original_weights=original_weights,
            metadata={
                "vix": vix,
                "vix_change": vix_change,
                "current_drawdown": current_drawdown,
                "high_water_mark": self._high_water_mark,
            },
        )

    def _apply_cash_allocation(
        self,
        weights: dict[str, float],
        cash_allocation: float,
    ) -> dict[str, float]:
        """Apply cash allocation by scaling down other positions.

        Args:
            weights: Current weights
            cash_allocation: Target cash allocation

        Returns:
            Adjusted weights with cash allocation
        """
        # Enforce maximum cash limit
        cash_allocation = min(cash_allocation, self.config.max_cash)

        # Ensure minimum invested
        cash_allocation = max(0.0, min(cash_allocation, 1.0 - self.config.min_invested))

        # Scale down non-cash positions
        scale_factor = 1.0 - cash_allocation
        adjusted = {}

        for symbol, weight in weights.items():
            if symbol == self.config.cash_symbol:
                continue
            adjusted[symbol] = weight * scale_factor

        # Add cash
        adjusted[self.config.cash_symbol] = cash_allocation

        return adjusted

    def _calculate_drawdown_cash(self, drawdown: float) -> float:
        """Calculate cash allocation based on drawdown.

        Args:
            drawdown: Current drawdown (negative value)

        Returns:
            Recommended cash allocation
        """
        # Graduated response to drawdown
        if drawdown >= -0.15:
            return 0.0
        elif drawdown >= -0.20:
            return 0.25
        elif drawdown >= -0.25:
            return 0.50
        else:
            return 0.80

    def _apply_position_limits(
        self,
        weights: dict[str, float],
        sector_map: Optional[dict[str, str]] = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Apply position size limits.

        Args:
            weights: Current weights
            sector_map: Optional symbol -> sector mapping

        Returns:
            Tuple of (adjusted weights, list of adjustments made)
        """
        adjustments = []
        adjusted = weights.copy()

        # Single asset limit
        for symbol, weight in adjusted.items():
            if symbol == self.config.cash_symbol:
                continue
            if weight > self.config.max_single_asset:
                adjusted[symbol] = self.config.max_single_asset
                adjustments.append(f"{symbol} capped at {self.config.max_single_asset:.1%}")

        # Sector limit (if sector_map provided)
        if sector_map:
            sector_weights: dict[str, float] = {}
            for symbol, weight in adjusted.items():
                if symbol == self.config.cash_symbol:
                    continue
                sector = sector_map.get(symbol, "Other")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

            # Scale down over-concentrated sectors
            for sector, total_weight in sector_weights.items():
                if total_weight > self.config.max_sector:
                    scale = self.config.max_sector / total_weight
                    for symbol, weight in adjusted.items():
                        if sector_map.get(symbol) == sector:
                            adjusted[symbol] = weight * scale
                    adjustments.append(f"Sector {sector} scaled to {self.config.max_sector:.1%}")

        return adjusted, adjustments

    def _smooth_transition(self, target_weights: dict[str, float]) -> dict[str, float]:
        """Apply smooth transition from previous weights.

        Args:
            target_weights: Target weights

        Returns:
            Smoothed weights
        """
        if not self._previous_weights:
            return target_weights

        alpha = self.config.transition_speed
        smoothed = {}

        all_symbols = set(target_weights.keys()) | set(self._previous_weights.keys())
        for symbol in all_symbols:
            target = target_weights.get(symbol, 0.0)
            previous = self._previous_weights.get(symbol, 0.0)
            smoothed[symbol] = alpha * target + (1 - alpha) * previous

        return smoothed

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Weights to normalize

        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        if total <= 0:
            # Return equal weight if no valid weights
            n = len(weights)
            return {symbol: 1.0 / n for symbol in weights}

        return {symbol: weight / total for symbol, weight in weights.items()}

    def reset_state(self) -> None:
        """Reset executor state (high water mark, previous weights)."""
        self._previous_weights = {}
        self._high_water_mark = 0.0


# Convenience functions


def execute_with_vix(
    weights: dict[str, float],
    vix: float,
    vix_change: float = 0.0,
    config_path: Optional[str | Path] = None,
) -> dict[str, float]:
    """
    Quick function to apply VIX adjustment to weights.

    Args:
        weights: Original portfolio weights
        vix: Current VIX level
        vix_change: Daily VIX percentage change
        config_path: Optional path to risk_params.yaml

    Returns:
        Adjusted weights
    """
    if config_path:
        executor = AllocationExecutor.from_config(config_path)
    else:
        executor = AllocationExecutor()

    result = executor.execute(weights=weights, vix=vix, vix_change=vix_change)
    return result.weights
