"""
Order Generator - 発注リスト生成

リバランス結果から実際の発注リスト（売買指示）を生成する。
現在のポジションと目標株数の差分を計算し、BUY/SELL注文を作成。

使用例:
    from src.allocation.order_generator import OrderGenerator, OrderItem
    from src.allocation.lot_adjuster import LotSizeAdjuster

    # ロット調整
    adjuster = LotSizeAdjuster()
    adjustment = adjuster.adjust_to_lot_size(...)

    # 発注リスト生成
    generator = OrderGenerator()
    orders = generator.generate_orders(
        current_positions={"AAPL": 100, "7203.T": 200},
        target_result=adjustment,
        prices={"AAPL": 150.0, "7203.T": 2500.0, "SPY": 450.0},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.allocation.lot_adjuster import LotAdjustmentResult


@dataclass
class OrderItem:
    """発注アイテム

    Attributes:
        symbol: ティッカーシンボル
        action: 売買区分（"BUY" | "SELL"）
        shares: 株数（整数 or 端数）
        price: 参考価格（発注時点の価格）
        amount: 概算金額
        lot_size: 最小購入単位
        is_fractional: 端数注文かどうか
        current_shares: 現在保有株数
        target_shares: 目標株数
    """

    symbol: str
    action: str  # "BUY" | "SELL"
    shares: int | float
    price: float
    amount: float
    lot_size: int = 1
    is_fractional: bool = False
    current_shares: int | float = 0
    target_shares: int | float = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "shares": self.shares,
            "price": self.price,
            "amount": self.amount,
            "lot_size": self.lot_size,
            "is_fractional": self.is_fractional,
            "current_shares": self.current_shares,
            "target_shares": self.target_shares,
        }

    def __str__(self) -> str:
        """表示用文字列"""
        shares_str = f"{self.shares:.4f}" if self.is_fractional else str(int(self.shares))
        return (
            f"{self.action:4s} {self.symbol:10s} "
            f"{shares_str:>10s}株 @ {self.price:,.2f} = {self.amount:,.0f}"
        )


@dataclass
class OrderSummary:
    """発注サマリ

    Attributes:
        orders: 発注リスト
        total_buy_amount: 買い総額
        total_sell_amount: 売り総額
        net_amount: 純売買額（正=買い超過、負=売り超過）
        buy_count: 買い銘柄数
        sell_count: 売り銘柄数
    """

    orders: list[OrderItem] = field(default_factory=list)
    total_buy_amount: float = 0.0
    total_sell_amount: float = 0.0
    net_amount: float = 0.0
    buy_count: int = 0
    sell_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "orders": [o.to_dict() for o in self.orders],
            "total_buy_amount": self.total_buy_amount,
            "total_sell_amount": self.total_sell_amount,
            "net_amount": self.net_amount,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
        }

    def buy_orders(self) -> list[OrderItem]:
        """買い注文のみ取得"""
        return [o for o in self.orders if o.action == "BUY"]

    def sell_orders(self) -> list[OrderItem]:
        """売り注文のみ取得"""
        return [o for o in self.orders if o.action == "SELL"]

    def format_report(self) -> str:
        """発注レポートをフォーマット"""
        lines = []
        lines.append("=" * 60)
        lines.append("発注リスト")
        lines.append("=" * 60)

        # 売り注文
        sell_orders = self.sell_orders()
        if sell_orders:
            lines.append("\n【売り注文】")
            lines.append("-" * 60)
            for order in sell_orders:
                lines.append(str(order))
            lines.append(f"  売り合計: {self.total_sell_amount:,.0f}")

        # 買い注文
        buy_orders = self.buy_orders()
        if buy_orders:
            lines.append("\n【買い注文】")
            lines.append("-" * 60)
            for order in buy_orders:
                lines.append(str(order))
            lines.append(f"  買い合計: {self.total_buy_amount:,.0f}")

        # サマリ
        lines.append("\n" + "=" * 60)
        lines.append(f"売り銘柄数: {self.sell_count}")
        lines.append(f"買い銘柄数: {self.buy_count}")
        lines.append(f"純売買額: {self.net_amount:+,.0f}")
        lines.append("=" * 60)

        return "\n".join(lines)


class OrderGenerator:
    """発注リスト生成クラス

    現在のポジションと目標株数の差分から発注リストを生成する。

    Example:
        >>> generator = OrderGenerator()
        >>> summary = generator.generate_orders(
        ...     current_positions={"AAPL": 100, "7203.T": 200},
        ...     target_result=adjustment_result,
        ...     prices={"AAPL": 150.0, "7203.T": 2500.0},
        ... )
        >>> print(summary.format_report())
    """

    def __init__(
        self,
        min_order_amount: float = 0.0,
        min_order_shares: int = 0,
    ) -> None:
        """初期化

        Args:
            min_order_amount: 最小発注金額（これ以下は発注しない）
            min_order_shares: 最小発注株数（これ以下は発注しない）
        """
        self.min_order_amount = min_order_amount
        self.min_order_shares = min_order_shares

    def generate_orders(
        self,
        current_positions: dict[str, int | float],
        target_result: LotAdjustmentResult,
        prices: dict[str, float],
        lot_sizes: dict[str, int] | None = None,
    ) -> OrderSummary:
        """発注リストを生成

        Args:
            current_positions: 現在保有株数（symbol -> shares）
            target_result: ロット調整結果
            prices: 各銘柄の価格（symbol -> price）
            lot_sizes: 各銘柄の最小購入単位

        Returns:
            OrderSummary: 発注サマリ
        """
        if lot_sizes is None:
            lot_sizes = {}

        orders: list[OrderItem] = []
        total_buy = 0.0
        total_sell = 0.0

        # 全銘柄を列挙（現在保有 + 目標）
        all_symbols = set(current_positions.keys())
        all_symbols.update(target_result.target_shares.keys())
        all_symbols.update(target_result.fractional_shares.keys())

        for symbol in sorted(all_symbols):
            if symbol not in prices:
                continue

            price = prices[symbol]
            current = current_positions.get(symbol, 0)
            lot_size = lot_sizes.get(symbol, 1)

            # 目標株数を取得
            target = target_result.get_shares(symbol)
            is_fractional = symbol in target_result.fractional_shares

            # 差分を計算
            diff = target - current

            # 小さすぎる注文はスキップ
            if abs(diff) < self.min_order_shares:
                continue

            amount = abs(diff) * price
            if amount < self.min_order_amount:
                continue

            if diff > 0:
                # 買い注文
                order = OrderItem(
                    symbol=symbol,
                    action="BUY",
                    shares=diff if is_fractional else int(diff),
                    price=price,
                    amount=amount,
                    lot_size=lot_size,
                    is_fractional=is_fractional,
                    current_shares=current,
                    target_shares=target,
                )
                orders.append(order)
                total_buy += amount

            elif diff < 0:
                # 売り注文
                order = OrderItem(
                    symbol=symbol,
                    action="SELL",
                    shares=abs(diff) if is_fractional else int(abs(diff)),
                    price=price,
                    amount=amount,
                    lot_size=lot_size,
                    is_fractional=is_fractional,
                    current_shares=current,
                    target_shares=target,
                )
                orders.append(order)
                total_sell += amount

        # 売り→買いの順でソート
        orders.sort(key=lambda o: (0 if o.action == "SELL" else 1, o.symbol))

        return OrderSummary(
            orders=orders,
            total_buy_amount=total_buy,
            total_sell_amount=total_sell,
            net_amount=total_buy - total_sell,
            buy_count=sum(1 for o in orders if o.action == "BUY"),
            sell_count=sum(1 for o in orders if o.action == "SELL"),
        )

    def generate_from_weights(
        self,
        current_positions: dict[str, int | float],
        target_weights: dict[str, float],
        prices: dict[str, float],
        portfolio_value: float,
        lot_sizes: dict[str, int] | None = None,
        fractional_allowed: dict[str, bool] | None = None,
        min_position_value: float = 0.0,
    ) -> OrderSummary:
        """重みから直接発注リストを生成

        ロット調整と発注リスト生成を一括で行う便利メソッド。

        Args:
            current_positions: 現在保有株数
            target_weights: 目標重み
            prices: 各銘柄の価格
            portfolio_value: ポートフォリオ総額
            lot_sizes: 各銘柄の最小購入単位
            fractional_allowed: 各銘柄の端数購入可否
            min_position_value: 最小ポジション金額

        Returns:
            OrderSummary: 発注サマリ
        """
        from src.allocation.lot_adjuster import LotSizeAdjuster

        adjuster = LotSizeAdjuster(min_position_value=min_position_value)
        adjustment = adjuster.adjust_to_lot_size(
            target_weights=target_weights,
            prices=prices,
            portfolio_value=portfolio_value,
            lot_sizes=lot_sizes,
            fractional_allowed=fractional_allowed,
        )

        return self.generate_orders(
            current_positions=current_positions,
            target_result=adjustment,
            prices=prices,
            lot_sizes=lot_sizes,
        )
