"""バックテストモジュール"""

from .fast_engine import BacktestEngine, BacktestConfig
from .factory import create_engine
from .rebalance_tracker import (
    RebalanceTracker,
    RebalanceRecord,
    ForecastMetrics,
)

# 後方互換エイリアス（非推奨）
FastBacktestEngine = BacktestEngine
FastBacktestConfig = BacktestConfig

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "create_engine",
    # Rebalance Tracker
    "RebalanceTracker",
    "RebalanceRecord",
    "ForecastMetrics",
    # 後方互換
    "FastBacktestEngine",
    "FastBacktestConfig",
]
