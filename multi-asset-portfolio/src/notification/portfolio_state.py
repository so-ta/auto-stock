"""
Portfolio State Manager for Trading Notifications.

Manages the persistence of portfolio weights between runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Portfolio state data."""

    market: str  # "US" or "JP"
    weights: dict[str, float]  # symbol -> weight
    last_update: datetime
    last_rebalance: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "market": self.market,
            "weights": self.weights,
            "last_update": self.last_update.isoformat(),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioState":
        """Create from dictionary."""
        return cls(
            market=data["market"],
            weights=data["weights"],
            last_update=datetime.fromisoformat(data["last_update"]),
            last_rebalance=(
                datetime.fromisoformat(data["last_rebalance"])
                if data.get("last_rebalance")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


class PortfolioStateManager:
    """
    Manages portfolio state persistence.

    Stores and retrieves portfolio weights from JSON files.

    Usage:
        manager = PortfolioStateManager(state_dir="data/portfolio_state")
        state = manager.load_state("US")
        manager.save_state(state)
    """

    def __init__(self, state_dir: str | Path = "data/portfolio_state"):
        """
        Initialize the state manager.

        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PortfolioStateManager initialized: {self.state_dir}")

    def _get_state_path(self, market: str) -> Path:
        """Get the state file path for a market."""
        return self.state_dir / f"{market}_state.json"

    def load_state(self, market: str) -> PortfolioState | None:
        """
        Load portfolio state for a market.

        Args:
            market: Market identifier ("US" or "JP")

        Returns:
            PortfolioState if exists, None otherwise
        """
        state_path = self._get_state_path(market)

        if not state_path.exists():
            logger.info(f"No existing state found for {market}")
            return None

        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)

            state = PortfolioState.from_dict(data)
            logger.info(
                f"Loaded state for {market}: {len(state.weights)} positions, "
                f"last_update={state.last_update}"
            )
            return state

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse state file for {market}: {e}")
            return None

        except (KeyError, ValueError) as e:
            logger.error(f"Invalid state format for {market}: {e}")
            return None

    def save_state(self, state: PortfolioState) -> bool:
        """
        Save portfolio state.

        Args:
            state: PortfolioState to save

        Returns:
            True if saved successfully, False otherwise
        """
        state_path = self._get_state_path(state.market)

        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved state for {state.market}: {len(state.weights)} positions"
            )
            return True

        except OSError as e:
            logger.error(f"Failed to save state for {state.market}: {e}")
            return False

    def get_previous_weights(self, market: str) -> dict[str, float]:
        """
        Get previous weights for a market.

        Args:
            market: Market identifier ("US" or "JP")

        Returns:
            Dictionary of symbol -> weight, empty dict if no previous state
        """
        state = self.load_state(market)
        if state is None:
            return {}
        return state.weights

    def update_weights(
        self,
        market: str,
        weights: dict[str, float],
        is_rebalance: bool = False,
    ) -> bool:
        """
        Update weights for a market.

        Args:
            market: Market identifier ("US" or "JP")
            weights: New weights dictionary
            is_rebalance: Whether this update is from an actual rebalance

        Returns:
            True if updated successfully, False otherwise
        """
        now = datetime.now()

        # Load existing state or create new
        existing = self.load_state(market)
        if existing:
            last_rebalance = existing.last_rebalance
            if is_rebalance:
                last_rebalance = now
        else:
            last_rebalance = now if is_rebalance else None

        state = PortfolioState(
            market=market,
            weights=weights,
            last_update=now,
            last_rebalance=last_rebalance,
        )

        return self.save_state(state)

    def clear_state(self, market: str) -> bool:
        """
        Clear state for a market.

        Args:
            market: Market identifier ("US" or "JP")

        Returns:
            True if cleared successfully, False otherwise
        """
        state_path = self._get_state_path(market)

        if not state_path.exists():
            return True

        try:
            state_path.unlink()
            logger.info(f"Cleared state for {market}")
            return True

        except OSError as e:
            logger.error(f"Failed to clear state for {market}: {e}")
            return False
