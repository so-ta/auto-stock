"""
VIX Signal Module - VIX-based dynamic cash allocation.

This module implements enhanced VIX-based risk management signals
for dynamic cash allocation during high volatility periods.

Based on IMP-003 from portfolio improvements analysis.

Key Features:
- Multi-tier VIX thresholds with graduated cash allocations
- Emergency triggers for sudden VIX spikes
- Recovery mode for gradual return to full investment
- Integration with drawdown control

Expected Effect:
- Sharpe ratio improvement: +0.15 to +0.25
- Maximum drawdown reduction: -5% to -8%

Usage:
    signal = EnhancedVIXSignal()
    cash_allocation = signal.get_cash_allocation(vix=25.5, vix_change=0.15)

    # Or use with configuration
    signal = EnhancedVIXSignal.from_config(config_path="config/risk_params.yaml")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import yaml

from .base import ParameterSpec, Signal, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class VIXTier:
    """VIX threshold tier configuration.

    Attributes:
        vix_max: Maximum VIX value for this tier
        cash_allocation: Cash allocation percentage for this tier
        description: Human-readable description
    """

    vix_max: float
    cash_allocation: float
    description: str = ""


@dataclass
class EmergencyTrigger:
    """Emergency trigger configuration.

    Attributes:
        threshold: Threshold value (percentage change or absolute)
        cash_allocation: Minimum cash allocation when triggered
        trigger_type: Type of trigger (daily_change, weekly_change, absolute)
    """

    threshold: float
    cash_allocation: float
    trigger_type: str = "daily_change"


@dataclass
class VIXSignalConfig:
    """Configuration for VIX signal.

    Attributes:
        tiers: List of VIX threshold tiers
        emergency_triggers: List of emergency triggers
        positive_days_for_recovery: Consecutive positive days to start recovery
        cash_reduction_rate: Rate to reduce cash during recovery
        vix_ticker: Ticker symbol for VIX data
        lookback_days: Days for change calculation
    """

    tiers: List[VIXTier] = field(default_factory=list)
    emergency_triggers: List[EmergencyTrigger] = field(default_factory=list)
    positive_days_for_recovery: int = 10
    cash_reduction_rate: float = 0.05
    vix_ticker: str = "^VIX"
    lookback_days: int = 5

    @classmethod
    def default(cls) -> "VIXSignalConfig":
        """Create default configuration."""
        return cls(
            tiers=[
                VIXTier(vix_max=15.0, cash_allocation=0.00, description="Low volatility"),
                VIXTier(vix_max=20.0, cash_allocation=0.10, description="Normal volatility"),
                VIXTier(vix_max=25.0, cash_allocation=0.25, description="Elevated volatility"),
                VIXTier(vix_max=30.0, cash_allocation=0.40, description="High volatility"),
                VIXTier(vix_max=999.0, cash_allocation=0.60, description="Extreme volatility"),
            ],
            emergency_triggers=[
                EmergencyTrigger(threshold=0.20, cash_allocation=0.30, trigger_type="daily_change"),
                EmergencyTrigger(threshold=0.30, cash_allocation=0.40, trigger_type="weekly_change"),
                EmergencyTrigger(threshold=35.0, cash_allocation=0.50, trigger_type="absolute"),
            ],
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VIXSignalConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls.default()

        with open(path) as f:
            data = yaml.safe_load(f)

        vix_config = data.get("vix_cash_allocation", {})

        # Parse tiers
        tiers = []
        for tier_data in vix_config.get("tiers", []):
            tiers.append(
                VIXTier(
                    vix_max=tier_data.get("vix_max", 999.0),
                    cash_allocation=tier_data.get("cash_allocation", 0.0),
                    description=tier_data.get("description", ""),
                )
            )

        # Parse emergency triggers
        triggers = []
        trigger_config = vix_config.get("emergency_triggers", {})
        if trigger_config.get("daily_change_threshold"):
            triggers.append(
                EmergencyTrigger(
                    threshold=trigger_config["daily_change_threshold"],
                    cash_allocation=trigger_config.get("daily_change_cash", 0.30),
                    trigger_type="daily_change",
                )
            )
        if trigger_config.get("weekly_change_threshold"):
            triggers.append(
                EmergencyTrigger(
                    threshold=trigger_config["weekly_change_threshold"],
                    cash_allocation=trigger_config.get("weekly_change_cash", 0.40),
                    trigger_type="weekly_change",
                )
            )
        if trigger_config.get("absolute_spike_threshold"):
            triggers.append(
                EmergencyTrigger(
                    threshold=trigger_config["absolute_spike_threshold"],
                    cash_allocation=trigger_config.get("absolute_spike_cash", 0.50),
                    trigger_type="absolute",
                )
            )

        # Parse recovery settings
        recovery_config = vix_config.get("recovery", {})

        return cls(
            tiers=tiers if tiers else cls.default().tiers,
            emergency_triggers=triggers if triggers else cls.default().emergency_triggers,
            positive_days_for_recovery=recovery_config.get("positive_days_required", 10),
            cash_reduction_rate=recovery_config.get("cash_reduction_rate", 0.05),
            vix_ticker=vix_config.get("data", {}).get("vix_ticker", "^VIX"),
            lookback_days=vix_config.get("data", {}).get("lookback_days", 5),
        )


@dataclass
class VIXSignalResult:
    """Result from VIX signal computation.

    Attributes:
        cash_allocation: Recommended cash allocation (0.0 to 1.0)
        tier_triggered: Which VIX tier was triggered
        emergency_triggered: Whether an emergency trigger was activated
        emergency_type: Type of emergency trigger if activated
        vix_level: Current VIX level
        vix_change: VIX change (daily or weekly)
        metadata: Additional metadata
    """

    cash_allocation: float
    tier_triggered: str = ""
    emergency_triggered: bool = False
    emergency_type: str = ""
    vix_level: float = 0.0
    vix_change: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EnhancedVIXSignal(Signal):
    """
    Enhanced VIX-based cash allocation signal.

    This signal computes recommended cash allocation based on:
    1. Current VIX level (multi-tier thresholds)
    2. VIX rate of change (emergency triggers)
    3. Absolute VIX spikes

    The cash allocation is designed to protect portfolio during
    high volatility periods while allowing full investment during
    calm markets.

    Usage:
        signal = EnhancedVIXSignal()
        result = signal.get_cash_allocation(vix=25.5, vix_change=0.15)
        print(f"Recommended cash: {result.cash_allocation:.1%}")

        # With OHLCV data
        signal_result = signal.compute(vix_data)
    """

    def __init__(self, config: Optional[VIXSignalConfig] = None, **params: Any):
        """Initialize VIX signal.

        Args:
            config: VIX signal configuration
            **params: Additional parameters
        """
        super().__init__(**params)
        self.config = config or VIXSignalConfig.default()
        self._consecutive_positive_days = 0
        self._last_cash_allocation = 0.0

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        """Define parameter specifications."""
        return [
            ParameterSpec(
                name="sensitivity",
                default=1.0,
                searchable=True,
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                description="Sensitivity multiplier for VIX thresholds",
            ),
        ]

    @classmethod
    def from_config(cls, config_path: str | Path) -> "EnhancedVIXSignal":
        """Create signal from YAML configuration file.

        Args:
            config_path: Path to risk_params.yaml

        Returns:
            Configured EnhancedVIXSignal instance
        """
        config = VIXSignalConfig.from_yaml(config_path)
        return cls(config=config)

    def get_cash_allocation(
        self,
        vix: float,
        vix_change: float = 0.0,
        vix_change_weekly: float = 0.0,
        portfolio_return: float = 0.0,
    ) -> VIXSignalResult:
        """
        Compute recommended cash allocation based on VIX.

        This is the main method for getting cash allocation recommendations.

        Args:
            vix: Current VIX level
            vix_change: Daily VIX percentage change (e.g., 0.20 for +20%)
            vix_change_weekly: Weekly VIX percentage change
            portfolio_return: Today's portfolio return (for recovery tracking)

        Returns:
            VIXSignalResult with recommended cash allocation
        """
        sensitivity = self._params.get("sensitivity", 1.0)

        # Step 1: Determine base cash allocation from VIX tier
        base_cash = 0.0
        tier_description = "Unknown"

        for tier in sorted(self.config.tiers, key=lambda t: t.vix_max):
            adjusted_max = tier.vix_max * sensitivity
            if vix < adjusted_max:
                base_cash = tier.cash_allocation
                tier_description = tier.description
                break

        # Step 2: Check emergency triggers
        emergency_triggered = False
        emergency_type = ""
        emergency_cash = 0.0

        for trigger in self.config.emergency_triggers:
            triggered = False

            if trigger.trigger_type == "daily_change" and vix_change > trigger.threshold:
                triggered = True
            elif trigger.trigger_type == "weekly_change" and vix_change_weekly > trigger.threshold:
                triggered = True
            elif trigger.trigger_type == "absolute" and vix > trigger.threshold:
                triggered = True

            if triggered and trigger.cash_allocation > emergency_cash:
                emergency_triggered = True
                emergency_type = trigger.trigger_type
                emergency_cash = trigger.cash_allocation

        # Step 3: Take maximum of base and emergency cash
        final_cash = max(base_cash, emergency_cash)

        # Step 4: Apply recovery logic if in recovery mode
        if portfolio_return > 0:
            self._consecutive_positive_days += 1
        else:
            self._consecutive_positive_days = 0

        if self._consecutive_positive_days >= self.config.positive_days_for_recovery:
            # Gradually reduce cash
            recovery_reduction = (
                self._consecutive_positive_days - self.config.positive_days_for_recovery
            ) * self.config.cash_reduction_rate
            final_cash = max(base_cash, final_cash - recovery_reduction)

        # Step 5: Clamp to valid range
        final_cash = max(0.0, min(1.0, final_cash))

        # Track last allocation for smoothing
        self._last_cash_allocation = final_cash

        return VIXSignalResult(
            cash_allocation=final_cash,
            tier_triggered=tier_description,
            emergency_triggered=emergency_triggered,
            emergency_type=emergency_type,
            vix_level=vix,
            vix_change=vix_change,
            metadata={
                "base_cash": base_cash,
                "emergency_cash": emergency_cash,
                "consecutive_positive_days": self._consecutive_positive_days,
                "sensitivity": sensitivity,
            },
        )

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute VIX signal from OHLCV data.

        The input data should contain VIX prices with 'close' column.

        Args:
            data: DataFrame with VIX OHLCV data

        Returns:
            SignalResult with cash allocation as scores
        """
        self.validate_input(data)

        # Calculate VIX metrics
        vix_close = data["close"]
        vix_change = vix_close.pct_change()
        vix_change_weekly = vix_close.pct_change(periods=5)

        # Compute cash allocation for each day
        cash_allocations = []
        for i in range(len(data)):
            if i == 0:
                change = 0.0
                change_weekly = 0.0
            else:
                change = vix_change.iloc[i] if not pd.isna(vix_change.iloc[i]) else 0.0
                change_weekly = (
                    vix_change_weekly.iloc[i]
                    if i >= 5 and not pd.isna(vix_change_weekly.iloc[i])
                    else 0.0
                )

            result = self.get_cash_allocation(
                vix=vix_close.iloc[i],
                vix_change=change,
                vix_change_weekly=change_weekly,
            )
            cash_allocations.append(result.cash_allocation)

        # Convert to Series (scores represent cash allocation, so higher = more defensive)
        # We return as signal score: -1 = full invest, +1 = max cash
        scores = pd.Series(cash_allocations, index=data.index)
        # Normalize to [-1, +1]: 0% cash = -1, 100% cash = +1
        normalized_scores = 2 * scores - 1

        return SignalResult(
            scores=normalized_scores,
            metadata={
                "raw_cash_allocations": cash_allocations,
                "vix_mean": float(vix_close.mean()),
                "vix_max": float(vix_close.max()),
                "vix_min": float(vix_close.min()),
            },
        )


# Convenience functions


def get_vix_cash_allocation(
    vix: float,
    vix_change: float = 0.0,
    config_path: Optional[str | Path] = None,
) -> float:
    """
    Quick function to get cash allocation for given VIX level.

    Args:
        vix: Current VIX level
        vix_change: Daily VIX percentage change
        config_path: Optional path to risk_params.yaml

    Returns:
        Recommended cash allocation (0.0 to 1.0)
    """
    if config_path:
        signal = EnhancedVIXSignal.from_config(config_path)
    else:
        signal = EnhancedVIXSignal()

    result = signal.get_cash_allocation(vix, vix_change)
    return result.cash_allocation


def calculate_vix_adjusted_weights(
    weights: dict[str, float],
    vix: float,
    vix_change: float = 0.0,
    cash_symbol: str = "CASH",
    config_path: Optional[str | Path] = None,
) -> dict[str, float]:
    """
    Adjust portfolio weights based on VIX signal.

    This function takes existing portfolio weights and adjusts them
    by allocating a portion to cash based on VIX level.

    Args:
        weights: Original portfolio weights (symbol -> weight)
        vix: Current VIX level
        vix_change: Daily VIX percentage change
        cash_symbol: Symbol to use for cash position
        config_path: Optional path to risk_params.yaml

    Returns:
        Adjusted weights with cash allocation
    """
    cash_allocation = get_vix_cash_allocation(vix, vix_change, config_path)

    if cash_allocation <= 0:
        return weights.copy()

    # Scale down existing weights
    scale_factor = 1.0 - cash_allocation
    adjusted = {symbol: weight * scale_factor for symbol, weight in weights.items()}

    # Add cash
    adjusted[cash_symbol] = cash_allocation

    return adjusted
