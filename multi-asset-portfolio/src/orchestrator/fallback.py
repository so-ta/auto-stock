"""
Fallback/Degradation Mode Handler

This module manages fallback modes when anomalies are detected:
- Mode 1: Hold previous weights (HOLD_PREVIOUS)
- Mode 2: Equal weight distribution (EQUAL_WEIGHT)
- Mode 3: Cash evacuation (CASH)

From §9:
Degradation modes (fixed selection):
- Mode 1: Maintain previous day weights
- Mode 2: Retreat to equal distribution
- Mode 3: Cash (w=0) - if permitted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from src.config.settings import FallbackMode

if TYPE_CHECKING:
    from src.config.settings import DegradationConfig, Settings

logger = structlog.get_logger(__name__)


# Note: FallbackMode is now imported from src.config.settings (SSOT)
# Available values: NONE, HOLD_PREVIOUS, EQUAL_WEIGHT, CASH


class DegradationLevel(int, Enum):
    """Degradation levels from settings."""

    LEVEL_0 = 0  # Normal operation
    LEVEL_1 = 1  # Reduced operation
    LEVEL_2 = 2  # Cash evacuation
    LEVEL_3 = 3  # Emergency stop


@dataclass
class FallbackState:
    """Current fallback state."""

    active: bool = False
    mode: FallbackMode = FallbackMode.NONE
    level: DegradationLevel = DegradationLevel.LEVEL_0
    reason: str | None = None
    activated_at: datetime | None = None
    previous_weights: dict[str, float] = field(default_factory=dict)
    applied_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active": self.active,
            "mode": self.mode.value,
            "level": self.level.value,
            "reason": self.reason,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "previous_weights": self.previous_weights,
            "applied_weights": self.applied_weights,
        }


class FallbackHandler:
    """
    Handles fallback/degradation mode transitions.

    Manages the transition between normal operation and various
    fallback modes based on anomaly detection results.

    Example:
        >>> handler = FallbackHandler(settings)
        >>> if anomaly_result.should_trigger_fallback:
        ...     weights = handler.apply_fallback(
        ...         previous_weights=prev_weights,
        ...         new_weights=new_weights,
        ...         reason=anomaly_result.fallback_reason,
        ...     )
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        """
        Initialize fallback handler.

        Args:
            settings: Optional settings instance
        """
        self._settings = settings
        self._logger = logger.bind(component="fallback_handler")
        self._state = FallbackState()
        self._history: list[FallbackState] = []

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def current_state(self) -> FallbackState:
        """Get current fallback state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if fallback mode is active."""
        return self._state.active

    @property
    def current_mode(self) -> FallbackMode:
        """Get current fallback mode."""
        return self._state.mode

    def apply_fallback(
        self,
        previous_weights: dict[str, float],
        new_weights: dict[str, float],
        reason: str,
        mode: FallbackMode | None = None,
        adopted_strategies: int = 0,
    ) -> dict[str, float]:
        """
        Apply fallback mode and return adjusted weights.

        Args:
            previous_weights: Weights from previous period
            new_weights: Newly calculated weights (before fallback)
            reason: Reason for fallback activation
            mode: Specific fallback mode to use (or auto-determine)
            adopted_strategies: Number of adopted strategies (for level determination)

        Returns:
            Adjusted weights based on fallback mode
        """
        # Determine mode based on settings or parameter
        if mode is None:
            mode = self._determine_mode(adopted_strategies)

        # Determine degradation level
        level = self._determine_level(adopted_strategies, mode)

        # Calculate applied weights
        applied_weights = self._calculate_fallback_weights(
            mode=mode,
            level=level,
            previous_weights=previous_weights,
            new_weights=new_weights,
        )

        # Update state
        self._state = FallbackState(
            active=True,
            mode=mode,
            level=level,
            reason=reason,
            activated_at=datetime.now(timezone.utc),
            previous_weights=previous_weights,
            applied_weights=applied_weights,
        )
        self._history.append(self._state)

        # Log activation
        self._logger.warning(
            "Fallback mode activated",
            mode=mode.value,
            level=level.value,
            reason=reason,
            previous_weight_count=len(previous_weights),
            applied_weight_count=len(applied_weights),
        )

        return applied_weights

    def deactivate(self) -> None:
        """Deactivate fallback mode and return to normal operation."""
        if not self._state.active:
            return

        self._logger.info(
            "Fallback mode deactivated",
            previous_mode=self._state.mode.value,
            duration_seconds=(
                (datetime.now(timezone.utc) - self._state.activated_at).total_seconds()
                if self._state.activated_at
                else 0
            ),
        )

        self._state = FallbackState()

    def check_recovery(
        self,
        adopted_strategies: int,
        current_level: DegradationLevel,
    ) -> bool:
        """
        Check if recovery from current degradation level is possible.

        Args:
            adopted_strategies: Current number of adopted strategies
            current_level: Current degradation level

        Returns:
            True if recovery is possible
        """
        degradation_config = self.settings.degradation

        if current_level == DegradationLevel.LEVEL_3:
            # Level 3 requires manual approval
            if degradation_config.require_manual_approval_for_level_3_recovery:
                return False

        if current_level == DegradationLevel.LEVEL_2:
            # Need at least 1 strategy for level 2 -> 1
            level_1_config = degradation_config.level_1
            min_strategies = level_1_config.min_adopted_strategies or 1
            return adopted_strategies >= min_strategies

        if current_level == DegradationLevel.LEVEL_1:
            # Need at least 3 strategies for level 1 -> 0
            level_0_config = degradation_config.level_0
            min_strategies = level_0_config.min_adopted_strategies or 3
            return adopted_strategies >= min_strategies

        return True

    def _determine_mode(self, adopted_strategies: int) -> FallbackMode:
        """Determine fallback mode based on settings and conditions."""
        # Get default mode from settings
        default_mode = self.settings.fallback

        if default_mode == "hold_previous":
            return FallbackMode.HOLD_PREVIOUS
        elif default_mode == "equal_weight":
            return FallbackMode.EQUAL_WEIGHT
        elif default_mode == "cash":
            return FallbackMode.CASH
        else:
            # Auto-determine based on adopted strategies
            if adopted_strategies == 0:
                return FallbackMode.CASH
            elif adopted_strategies < 3:
                return FallbackMode.HOLD_PREVIOUS
            else:
                return FallbackMode.EQUAL_WEIGHT

    def _determine_level(
        self,
        adopted_strategies: int,
        mode: FallbackMode,
    ) -> DegradationLevel:
        """Determine degradation level based on conditions."""
        degradation_config = self.settings.degradation

        # Level 3: Emergency (cash mode forced)
        if mode == FallbackMode.CASH:
            return DegradationLevel.LEVEL_3

        # Level 2: Cash evacuation conditions
        level_2_config = degradation_config.level_2
        if adopted_strategies == 0:
            return DegradationLevel.LEVEL_2

        # Level 1: Reduced operation
        level_1_config = degradation_config.level_1
        min_for_level_0 = degradation_config.level_0.min_adopted_strategies or 3
        if adopted_strategies < min_for_level_0:
            return DegradationLevel.LEVEL_1

        return DegradationLevel.LEVEL_0

    def _calculate_fallback_weights(
        self,
        mode: FallbackMode,
        level: DegradationLevel,
        previous_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate weights based on fallback mode."""
        if mode == FallbackMode.NONE:
            return new_weights

        degradation_config = self.settings.degradation

        if mode == FallbackMode.HOLD_PREVIOUS:
            # Mode 1: Keep previous weights
            if not previous_weights:
                # No previous weights, fall back to equal weight
                return self._equal_weights(list(new_weights.keys()))
            return previous_weights.copy()

        elif mode == FallbackMode.EQUAL_WEIGHT:
            # Mode 2: Equal distribution
            assets = list(new_weights.keys()) or list(previous_weights.keys())
            return self._equal_weights(assets)

        elif mode == FallbackMode.CASH:
            # Mode 3: Full cash
            # Get cash ratio from level config
            if level == DegradationLevel.LEVEL_2:
                cash_ratio = degradation_config.level_2.cash_ratio
            elif level == DegradationLevel.LEVEL_3:
                cash_ratio = degradation_config.level_3.cash_ratio
            else:
                cash_ratio = 1.0

            return self._cash_weights(
                assets=list(new_weights.keys()) or list(previous_weights.keys()),
                cash_ratio=cash_ratio,
            )

        return new_weights

    def _equal_weights(self, assets: list[str]) -> dict[str, float]:
        """Generate equal weights for assets."""
        if not assets:
            return {"CASH": 1.0}

        weight = 1.0 / len(assets)
        return {asset: weight for asset in assets}

    def _cash_weights(
        self,
        assets: list[str],
        cash_ratio: float = 1.0,
    ) -> dict[str, float]:
        """Generate weights with cash allocation."""
        if cash_ratio >= 1.0:
            # Full cash
            return {"CASH": 1.0}

        if not assets:
            return {"CASH": 1.0}

        # Distribute remaining weight equally
        remaining = 1.0 - cash_ratio
        asset_weight = remaining / len(assets)

        weights = {asset: asset_weight for asset in assets}
        weights["CASH"] = cash_ratio

        return weights

    def get_history(self) -> list[dict[str, Any]]:
        """Get fallback activation history."""
        return [state.to_dict() for state in self._history]

    def apply_position_size_multiplier(
        self,
        weights: dict[str, float],
        level: DegradationLevel,
    ) -> dict[str, float]:
        """
        Apply position size multiplier based on degradation level.

        Args:
            weights: Current weights
            level: Degradation level

        Returns:
            Adjusted weights with multiplier applied
        """
        degradation_config = self.settings.degradation

        # Get multiplier from config
        if level == DegradationLevel.LEVEL_1:
            multiplier = degradation_config.level_1.position_size_multiplier
        else:
            multiplier = 1.0

        if multiplier >= 1.0:
            return weights

        # Apply multiplier and add to cash
        adjusted = {}
        cash_addition = 0.0

        for asset, weight in weights.items():
            if asset == "CASH":
                adjusted[asset] = weight
            else:
                adjusted_weight = weight * multiplier
                adjusted[asset] = adjusted_weight
                cash_addition += weight - adjusted_weight

        # Add freed capital to cash
        adjusted["CASH"] = adjusted.get("CASH", 0) + cash_addition

        return adjusted


def create_fallback_handler(settings: "Settings | None" = None) -> FallbackHandler:
    """
    Factory function to create a FallbackHandler.

    Args:
        settings: Optional settings instance

    Returns:
        Configured FallbackHandler instance
    """
    return FallbackHandler(settings)
