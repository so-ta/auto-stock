"""
Holdings Management - 保有資産管理

ポートフォリオの保有資産（株数・金額）を管理。

使用例:
    from src.portfolio.holdings import Holdings, Position

    holdings = Holdings.load("japan_stocks")
    print(holdings.total_value)  # 10500000
    print(holdings.get_position("7203.T").shares)  # 200
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """保有ポジション

    Attributes:
        symbol: ティッカーシンボル
        shares: 保有株数
        avg_cost: 平均取得単価
        current_price: 現在価格
        market_value: 時価
        weight: ポートフォリオ内比率
        lot_size: 最小購入単位
        is_fractional: 端数保有かどうか
    """

    symbol: str
    shares: int | float
    avg_cost: float
    current_price: float
    market_value: float = 0.0
    weight: float = 0.0
    lot_size: int = 1
    is_fractional: bool = False

    def __post_init__(self):
        """時価を自動計算"""
        if self.market_value == 0.0 and self.current_price > 0:
            self.market_value = self.shares * self.current_price

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "weight": self.weight,
            "lot_size": self.lot_size,
            "is_fractional": self.is_fractional,
        }

    @classmethod
    def from_dict(cls, symbol: str, data: dict[str, Any]) -> "Position":
        """辞書から作成"""
        return cls(
            symbol=symbol,
            shares=data.get("shares", 0),
            avg_cost=data.get("avg_cost", 0.0),
            current_price=data.get("current_price", 0.0),
            market_value=data.get("market_value", 0.0),
            weight=data.get("weight", 0.0),
            lot_size=data.get("lot_size", 1),
            is_fractional=data.get("is_fractional", False),
        )

    @property
    def unrealized_pnl(self) -> float:
        """未実現損益"""
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """未実現損益率（%）"""
        if self.avg_cost <= 0:
            return 0.0
        return ((self.current_price / self.avg_cost) - 1) * 100


@dataclass
class Holdings:
    """保有資産

    Attributes:
        portfolio_id: ポートフォリオID
        positions: ポジション辞書（symbol -> Position）
        cash: 現金残高
        total_value: 総資産価値
        currency: 通貨
        updated_at: 最終更新日時
        metadata: メタデータ
    """

    portfolio_id: str
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    total_value: float = 0.0
    currency: str = "JPY"
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """総資産を自動計算"""
        self._recalculate()

    def _recalculate(self) -> None:
        """総資産と重みを再計算"""
        positions_value = sum(p.market_value for p in self.positions.values())
        self.total_value = positions_value + self.cash

        # 各ポジションの重みを再計算
        if self.total_value > 0:
            for position in self.positions.values():
                position.weight = position.market_value / self.total_value

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "portfolio_id": self.portfolio_id,
            "updated_at": self.updated_at.isoformat(),
            "total_value": self.total_value,
            "cash": self.cash,
            "currency": self.currency,
            "positions": {
                symbol: pos.to_dict() for symbol, pos in self.positions.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Holdings":
        """辞書から作成"""
        positions = {}
        for symbol, pos_data in data.get("positions", {}).items():
            positions[symbol] = Position.from_dict(symbol, pos_data)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        holdings = cls(
            portfolio_id=data.get("portfolio_id", "unknown"),
            positions=positions,
            cash=data.get("cash", 0.0),
            total_value=data.get("total_value", 0.0),
            currency=data.get("currency", "JPY"),
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )
        # 総資産を再計算（cashを含む正確な値を使用）
        holdings._recalculate()
        return holdings

    def get_position(self, symbol: str) -> Position | None:
        """ポジションを取得

        Args:
            symbol: ティッカーシンボル

        Returns:
            Position or None
        """
        return self.positions.get(symbol)

    def add_position(
        self,
        symbol: str,
        shares: int | float,
        avg_cost: float,
        current_price: float | None = None,
        lot_size: int = 1,
        is_fractional: bool = False,
    ) -> Position:
        """ポジションを追加

        Args:
            symbol: ティッカーシンボル
            shares: 株数
            avg_cost: 平均取得単価
            current_price: 現在価格（未指定時はavg_costを使用）
            lot_size: ロットサイズ
            is_fractional: 端数保有かどうか

        Returns:
            追加されたPosition
        """
        if current_price is None:
            current_price = avg_cost

        position = Position(
            symbol=symbol,
            shares=shares,
            avg_cost=avg_cost,
            current_price=current_price,
            lot_size=lot_size,
            is_fractional=is_fractional,
        )
        self.positions[symbol] = position
        self._recalculate()
        return position

    def remove_position(self, symbol: str) -> Position | None:
        """ポジションを削除

        Args:
            symbol: ティッカーシンボル

        Returns:
            削除されたPosition or None
        """
        position = self.positions.pop(symbol, None)
        if position:
            self._recalculate()
        return position

    def update_prices(self, prices: dict[str, float]) -> None:
        """価格を一括更新

        Args:
            prices: symbol -> price の辞書
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                self.positions[symbol].market_value = self.positions[symbol].shares * price

        self.updated_at = datetime.now()
        self._recalculate()

    def get_current_shares(self) -> dict[str, int | float]:
        """現在の保有株数を辞書で取得

        Returns:
            symbol -> shares の辞書
        """
        return {symbol: pos.shares for symbol, pos in self.positions.items()}

    def get_current_weights(self) -> dict[str, float]:
        """現在の重みを辞書で取得

        Returns:
            symbol -> weight の辞書
        """
        return {symbol: pos.weight for symbol, pos in self.positions.items()}

    @property
    def cash_weight(self) -> float:
        """現金比率"""
        if self.total_value <= 0:
            return 0.0
        return self.cash / self.total_value

    @property
    def total_unrealized_pnl(self) -> float:
        """総未実現損益"""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def position_count(self) -> int:
        """保有銘柄数"""
        return len(self.positions)

    def save(self, base_dir: str | Path = "data/portfolio_state") -> bool:
        """保有資産をJSONに保存

        Args:
            base_dir: 保存先ベースディレクトリ

        Returns:
            成功したらTrue
        """
        base_path = Path(base_dir)
        portfolio_dir = base_path / self.portfolio_id
        portfolio_dir.mkdir(parents=True, exist_ok=True)

        holdings_path = portfolio_dir / "holdings.json"

        try:
            with open(holdings_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Holdings saved: {holdings_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to save holdings: {e}")
            return False

    @classmethod
    def load(
        cls,
        portfolio_id: str,
        base_dir: str | Path = "data/portfolio_state",
    ) -> "Holdings | None":
        """保有資産をJSONから読み込み

        Args:
            portfolio_id: ポートフォリオID
            base_dir: 保存先ベースディレクトリ

        Returns:
            Holdings or None
        """
        base_path = Path(base_dir)
        holdings_path = base_path / portfolio_id / "holdings.json"

        if not holdings_path.exists():
            logger.info(f"No holdings found for {portfolio_id}")
            return None

        try:
            with open(holdings_path, encoding="utf-8") as f:
                data = json.load(f)
            holdings = cls.from_dict(data)
            logger.info(f"Holdings loaded: {holdings.position_count} positions")
            return holdings
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load holdings: {e}")
            return None

    @classmethod
    def create_empty(
        cls,
        portfolio_id: str,
        initial_capital: float = 0.0,
        currency: str = "JPY",
    ) -> "Holdings":
        """空の保有資産を作成

        Args:
            portfolio_id: ポートフォリオID
            initial_capital: 初期資本（現金として設定）
            currency: 通貨

        Returns:
            Holdings
        """
        return cls(
            portfolio_id=portfolio_id,
            positions={},
            cash=initial_capital,
            total_value=initial_capital,
            currency=currency,
            updated_at=datetime.now(),
            metadata={
                "created_at": datetime.now().isoformat(),
                "initial_capital": initial_capital,
            },
        )
