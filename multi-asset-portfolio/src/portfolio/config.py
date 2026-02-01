"""
Portfolio Configuration - ポートフォリオ設定

ポートフォリオの定義、スケジュール、通知設定を管理。

使用例:
    from src.portfolio.config import PortfolioConfig

    config = PortfolioConfig.from_yaml("config/portfolios/japan_stocks.yaml")
    print(config.name)  # "日本株ポートフォリオ"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScheduleConfig:
    """スケジュール設定

    Attributes:
        enabled: スケジュール実行を有効にするか
        hour: 実行時刻（時）
        minute: 実行時刻（分）
        timezone: タイムゾーン
        trading_days_only: 営業日のみ実行するか
    """

    enabled: bool = False
    hour: int = 3
    minute: int = 0
    timezone: str = "Asia/Tokyo"
    trading_days_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "enabled": self.enabled,
            "hour": self.hour,
            "minute": self.minute,
            "timezone": self.timezone,
            "trading_days_only": self.trading_days_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleConfig":
        """辞書から作成"""
        return cls(
            enabled=data.get("enabled", False),
            hour=data.get("hour", 3),
            minute=data.get("minute", 0),
            timezone=data.get("timezone", "Asia/Tokyo"),
            trading_days_only=data.get("trading_days_only", True),
        )


@dataclass
class NotificationConfig:
    """通知設定

    Attributes:
        on_rebalance: リバランス時に通知するか
        on_no_rebalance: リバランス不要時も通知するか
        include_order_details: 発注詳細を含めるか
        webhook_url: Discord Webhook URL（未設定時はグローバル設定を使用）
    """

    on_rebalance: bool = True
    on_no_rebalance: bool = False
    include_order_details: bool = True
    webhook_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "on_rebalance": self.on_rebalance,
            "on_no_rebalance": self.on_no_rebalance,
            "include_order_details": self.include_order_details,
            "webhook_url": self.webhook_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NotificationConfig":
        """辞書から作成"""
        return cls(
            on_rebalance=data.get("on_rebalance", True),
            on_no_rebalance=data.get("on_no_rebalance", False),
            include_order_details=data.get("include_order_details", True),
            webhook_url=data.get("webhook_url"),
        )


@dataclass
class UniverseConfig:
    """ユニバース設定

    Attributes:
        source: ユニバースソース（"asset_master" | "custom"）
        subset: asset_master.yamlのサブセット名
        symbols: カスタムシンボルリスト
    """

    source: str = "asset_master"
    subset: str | None = None
    symbols: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "source": self.source,
            "subset": self.subset,
            "symbols": self.symbols,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniverseConfig":
        """辞書から作成"""
        return cls(
            source=data.get("source", "asset_master"),
            subset=data.get("subset"),
            symbols=data.get("symbols"),
        )


@dataclass
class LotSizeConfig:
    """ロットサイズ設定

    Attributes:
        default: デフォルトのロットサイズ
        fractional_allowed: 端数購入を許可するか
        overrides: 銘柄ごとのロットサイズ上書き
    """

    default: int = 1
    fractional_allowed: bool = False
    overrides: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "default": self.default,
            "fractional_allowed": self.fractional_allowed,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LotSizeConfig":
        """辞書から作成"""
        return cls(
            default=data.get("default", 1),
            fractional_allowed=data.get("fractional_allowed", False),
            overrides=data.get("overrides", {}),
        )


@dataclass
class PortfolioConfig:
    """ポートフォリオ設定

    Attributes:
        id: ポートフォリオID（ファイル名から導出）
        name: 表示名
        description: 説明
        universe: ユニバース設定
        initial_capital: 初期資本
        currency: 通貨
        lot_size: ロットサイズ設定
        schedule: スケジュール設定
        notification: 通知設定
    """

    id: str
    name: str
    description: str = ""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    initial_capital: float = 1_000_000.0
    currency: str = "JPY"
    lot_size: LotSizeConfig = field(default_factory=LotSizeConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "universe": self.universe.to_dict(),
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "lot_size": self.lot_size.to_dict(),
            "schedule": self.schedule.to_dict(),
            "notification": self.notification.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], portfolio_id: str | None = None) -> "PortfolioConfig":
        """辞書から作成

        Args:
            data: 設定データ
            portfolio_id: ポートフォリオID（未指定時はdataから取得）
        """
        return cls(
            id=portfolio_id or data.get("id", "unknown"),
            name=data.get("name", "Unknown Portfolio"),
            description=data.get("description", ""),
            universe=UniverseConfig.from_dict(data.get("universe", {})),
            initial_capital=data.get("initial_capital", 1_000_000.0),
            currency=data.get("currency", "JPY"),
            lot_size=LotSizeConfig.from_dict(data.get("lot_size", {})),
            schedule=ScheduleConfig.from_dict(data.get("schedule", {})),
            notification=NotificationConfig.from_dict(data.get("notification", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PortfolioConfig":
        """YAMLファイルから読み込み

        Args:
            yaml_path: YAMLファイルのパス

        Returns:
            PortfolioConfig: 設定オブジェクト
        """
        path = Path(yaml_path)
        portfolio_id = path.stem  # ファイル名から拡張子を除いたもの

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data, portfolio_id=portfolio_id)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """YAMLファイルに保存

        Args:
            yaml_path: 保存先のパス
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        # idはファイル名で決まるため保存しない
        del data["id"]

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def get_lot_size(self, symbol: str) -> int:
        """銘柄のロットサイズを取得

        Args:
            symbol: ティッカーシンボル

        Returns:
            ロットサイズ
        """
        return self.lot_size.overrides.get(symbol, self.lot_size.default)

    def is_fractional_allowed(self, symbol: str) -> bool:
        """銘柄の端数購入が許可されているか

        Args:
            symbol: ティッカーシンボル

        Returns:
            端数購入可否
        """
        return self.lot_size.fractional_allowed
