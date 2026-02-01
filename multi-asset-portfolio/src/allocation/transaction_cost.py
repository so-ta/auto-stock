"""カテゴリ別取引コスト管理

銘柄カテゴリごとに異なる取引コストを設定できるようにする。

対応コストタイプ:
- bps: ベーシスポイント（例: 20 = 20bps = 0.2%）
- percentage: パーセンテージ（例: 0.001 = 0.1%）
- fixed: 固定額（例: 2.0 = $2）

使用例:
    from src.allocation.transaction_cost import (
        TransactionCostConfig,
        TransactionCostSchedule,
        DEFAULT_COST_SCHEDULE,
    )

    # カテゴリ別コストスケジュール
    schedule = DEFAULT_COST_SCHEDULE

    # 米国株式（固定額$2）
    us_cost = schedule.calculate_cost(
        symbol='AAPL',
        weight_change=0.05,
        portfolio_value=100000,
        category='us_stocks'
    )

    # 日本株式（0.1%）
    jp_cost = schedule.calculate_cost(
        symbol='7203.T',
        weight_change=0.05,
        category='japan_stocks'
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend


@dataclass
class TransactionCostConfig:
    """取引コスト設定

    Attributes:
        cost_type: コストの種類
            - "bps": ベーシスポイント（例: 20 = 20bps = 0.2%）
            - "percentage": パーセンテージ（例: 0.001 = 0.1%）
            - "fixed": 固定額（例: 2.0 = $2）
        value: コスト値
        currency: 通貨（fixed typeの場合のみ使用）
    """

    cost_type: str = "bps"
    value: float = 20.0
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.cost_type not in ("bps", "percentage", "fixed"):
            raise ValueError(f"Invalid cost_type: {self.cost_type}")
        if self.value < 0:
            raise ValueError("Cost value must be non-negative")

    def to_rate(self, trade_value: float | None = None) -> float:
        """コストをレート（0-1の比率）に変換

        Args:
            trade_value: 取引額（fixed typeの場合に必要）

        Returns:
            コストレート（0-1）
        """
        if self.cost_type == "bps":
            return self.value / 10000
        elif self.cost_type == "percentage":
            return self.value
        elif self.cost_type == "fixed":
            if trade_value is None or trade_value <= 0:
                # 取引額が不明な場合は保守的に高めのコストを返す
                return 0.01  # 1%
            return self.value / trade_value
        else:
            return 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionCostConfig":
        """辞書から作成"""
        return cls(
            cost_type=data.get("cost_type", "bps"),
            value=data.get("value", 20.0),
            currency=data.get("currency", "USD"),
        )

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "cost_type": self.cost_type,
            "value": self.value,
            "currency": self.currency,
        }


@dataclass
class TransactionCostSchedule:
    """カテゴリ別取引コストスケジュール

    カテゴリごとに異なる取引コストを設定し、
    銘柄のコスト計算を行う。

    Example:
        >>> schedule = TransactionCostSchedule(
        ...     default=TransactionCostConfig(cost_type="bps", value=20),
        ...     category_costs={
        ...         "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
        ...         "japan_stocks": TransactionCostConfig(cost_type="percentage", value=0.001),
        ...     }
        ... )
        >>> cost = schedule.calculate_cost("AAPL", 0.05, 100000, "us_stocks")
    """

    default: TransactionCostConfig = field(
        default_factory=lambda: TransactionCostConfig()
    )
    category_costs: dict[str, TransactionCostConfig] = field(default_factory=dict)

    def get_cost_config(
        self,
        symbol: str,
        category: str | None = None,
    ) -> TransactionCostConfig:
        """銘柄のコスト設定を取得

        Args:
            symbol: ティッカーシンボル
            category: カテゴリ（Noneの場合はデフォルトを使用）

        Returns:
            TransactionCostConfig
        """
        if category and category in self.category_costs:
            return self.category_costs[category]
        return self.default

    def calculate_cost(
        self,
        symbol: str,
        weight_change: float,
        portfolio_value: float | None = None,
        category: str | None = None,
    ) -> float:
        """取引コストを計算

        Args:
            symbol: ティッカーシンボル
            weight_change: ウェイト変化（絶対値）
            portfolio_value: ポートフォリオ総額（固定額コスト用）
            category: カテゴリ

        Returns:
            コスト（ウェイトベース、0-1）
        """
        config = self.get_cost_config(symbol, category)

        if config.cost_type == "fixed" and portfolio_value:
            trade_value = portfolio_value * weight_change
            rate = config.to_rate(trade_value)
        else:
            rate = config.to_rate()

        return weight_change * rate

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionCostSchedule":
        """辞書から作成"""
        default_data = data.get("default", {"cost_type": "bps", "value": 20})
        default = TransactionCostConfig.from_dict(default_data)

        category_costs = {}
        for category, cost_data in data.get("categories", {}).items():
            category_costs[category] = TransactionCostConfig.from_dict(cost_data)

        return cls(default=default, category_costs=category_costs)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        storage_backend: Optional["StorageBackend"] = None,
    ) -> "TransactionCostSchedule":
        """YAMLファイルから読み込み

        Args:
            path: YAMLファイルパス
            storage_backend: オプショナルなStorageBackend（S3サポート用）

        Returns:
            TransactionCostSchedule インスタンス
        """
        if storage_backend:
            try:
                data = storage_backend.read_yaml(str(path))
            except FileNotFoundError:
                return cls()
        else:
            path = Path(path)
            if not path.exists():
                return cls()

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        if data is None:
            return cls()

        return cls.from_dict(data.get("transaction_costs", data))

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "default": self.default.to_dict(),
            "categories": {
                cat: config.to_dict() for cat, config in self.category_costs.items()
            },
        }


# デフォルトスケジュール
DEFAULT_COST_SCHEDULE = TransactionCostSchedule(
    default=TransactionCostConfig(cost_type="bps", value=15),  # 0.15% (15bps)
    category_costs={
        "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
        "japan_stocks": TransactionCostConfig(cost_type="percentage", value=0.001),
        "etfs": TransactionCostConfig(cost_type="percentage", value=0.002),
        "etfs_equity": TransactionCostConfig(cost_type="percentage", value=0.002),
        "etfs_sector": TransactionCostConfig(cost_type="percentage", value=0.002),
        "etfs_bond": TransactionCostConfig(cost_type="percentage", value=0.002),
        "etfs_commodity": TransactionCostConfig(cost_type="percentage", value=0.002),
        "etfs_international": TransactionCostConfig(cost_type="percentage", value=0.002),
    },
)


def load_cost_schedule(path: str | Path | None = None) -> TransactionCostSchedule:
    """取引コストスケジュールを読み込み

    Args:
        path: YAMLファイルパス（Noneでデフォルト）

    Returns:
        TransactionCostSchedule
    """
    if path is None:
        default_path = Path(__file__).parent.parent.parent / "config" / "cost_schedule.yaml"
        if default_path.exists():
            return TransactionCostSchedule.from_yaml(default_path)
        return DEFAULT_COST_SCHEDULE

    return TransactionCostSchedule.from_yaml(path)
