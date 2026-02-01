"""
Result Viewer Routers

Modular API routers for the backtest result viewer application.
"""

from .results import router as results_router
from .backtest import router as backtest_router
from .config import router as config_router

__all__ = [
    "results_router",
    "backtest_router",
    "config_router",
]
