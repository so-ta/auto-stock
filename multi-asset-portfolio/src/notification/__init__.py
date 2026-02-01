"""
Notification module for trading signal alerts.

This package provides:
- DiscordNotifier: Send trading signals to Discord via webhooks
- PortfolioStateManager: Track and persist portfolio state between runs
- TradingScheduler: Schedule daily market checks with APScheduler
"""

from src.notification.discord import DiscordNotifier
from src.notification.portfolio_state import PortfolioStateManager

__all__ = [
    "DiscordNotifier",
    "PortfolioStateManager",
]
