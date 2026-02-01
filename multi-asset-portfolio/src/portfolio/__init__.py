"""
Portfolio Management Module - 本番運用向けポートフォリオ管理

複数ポートフォリオの管理、保有資産の追跡、スケジュール実行、通知を提供。

主要コンポーネント:
- PortfolioConfig: ポートフォリオ設定
- Holdings: 保有資産データ
- PortfolioManager: ポートフォリオ管理の統合クラス
- PortfolioHistory: パフォーマンス履歴追跡
"""

from __future__ import annotations

from src.portfolio.config import (
    PortfolioConfig,
    ScheduleConfig,
    NotificationConfig,
    UniverseConfig,
    LotSizeConfig,
)
from src.portfolio.holdings import (
    Position,
    Holdings,
)
from src.portfolio.manager import (
    PortfolioManager,
    RebalanceResult,
)
from src.portfolio.history import (
    PositionSnapshot,
    DailySnapshot,
    PerformanceMetrics,
    PortfolioHistory,
)

__all__ = [
    # Config
    "PortfolioConfig",
    "ScheduleConfig",
    "NotificationConfig",
    "UniverseConfig",
    "LotSizeConfig",
    # Holdings
    "Position",
    "Holdings",
    # Manager
    "PortfolioManager",
    "RebalanceResult",
    # History
    "PositionSnapshot",
    "DailySnapshot",
    "PerformanceMetrics",
    "PortfolioHistory",
]
