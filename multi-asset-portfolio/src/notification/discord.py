"""
Discord Notification Client for Trading Signals.

Sends rebalance notifications to Discord via webhooks.

ポートフォリオリバランス通知も対応:
- 保有資産サマリ
- 売買注文リスト（株数・金額）
- 必要追加資金
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from src.portfolio.holdings import Holdings
    from src.allocation.order_generator import OrderSummary
    from src.allocation.lot_adjuster import LotAdjustmentResult

logger = logging.getLogger(__name__)


@dataclass
class RebalanceNotification:
    """Rebalance notification data."""

    market: str  # "US" or "JP"
    date: datetime
    trigger_reasons: list[str]
    buys: dict[str, dict[str, float]]  # symbol -> {old_weight, new_weight, change}
    sells: dict[str, dict[str, float]]  # symbol -> {old_weight, new_weight, change}
    estimated_turnover: float
    cash_weight_old: float
    cash_weight_new: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DiscordNotifier:
    """
    Discord webhook notification client.

    Usage:
        notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/xxx/yyy")
        notifier.send_rebalance_notification(notification)
    """

    def __init__(self, webhook_url: str | None = None):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL. If None, notifications are logged but not sent.
        """
        self.webhook_url = webhook_url
        if not webhook_url:
            logger.warning("Discord webhook URL not configured - notifications will be logged only")

    def send_rebalance_notification(self, notification: RebalanceNotification) -> bool:
        """
        Send a rebalance notification to Discord.

        Args:
            notification: RebalanceNotification object containing trade details

        Returns:
            True if notification was sent successfully, False otherwise
        """
        market_name = "米国市場" if notification.market == "US" else "日本市場"
        date_str = notification.date.strftime("%Y-%m-%d")

        # Build message
        lines = [
            f"📊 **{date_str} {market_name} リバランス通知**",
            "",
        ]

        # Trigger reasons
        if notification.trigger_reasons:
            lines.append(f"🔄 **トリガー**: {' + '.join(notification.trigger_reasons)}")
            lines.append("")

        # Buys
        if notification.buys:
            lines.append(f"📈 **買い ({len(notification.buys)}銘柄)**")
            for symbol, data in sorted(
                notification.buys.items(),
                key=lambda x: x[1].get("change", 0),
                reverse=True,
            ):
                old = data.get("old_weight", 0) * 100
                new = data.get("new_weight", 0) * 100
                change = data.get("change", 0) * 100
                lines.append(f"  {symbol}: {old:.1f}% → {new:.1f}% (+{change:.1f}%)")
            lines.append("")

        # Sells
        if notification.sells:
            lines.append(f"📉 **売り ({len(notification.sells)}銘柄)**")
            for symbol, data in sorted(
                notification.sells.items(),
                key=lambda x: x[1].get("change", 0),
            ):
                old = data.get("old_weight", 0) * 100
                new = data.get("new_weight", 0) * 100
                change = data.get("change", 0) * 100
                lines.append(f"  {symbol}: {old:.1f}% → {new:.1f}% ({change:.1f}%)")
            lines.append("")

        # Summary
        lines.append("💰 **サマリ**")
        lines.append(f"  推定回転率: {notification.estimated_turnover * 100:.1f}%")
        lines.append(
            f"  現金比率: {notification.cash_weight_old * 100:.0f}% → "
            f"{notification.cash_weight_new * 100:.0f}%"
        )

        message = "\n".join(lines)
        return self._send_message(message)

    def send_no_rebalance_notification(self, market: str, date: datetime) -> bool:
        """
        Send a notification that no rebalance is needed.

        Args:
            market: Market identifier ("US" or "JP")
            date: Current date

        Returns:
            True if notification was sent successfully, False otherwise
        """
        market_name = "米国市場" if market == "US" else "日本市場"
        date_str = date.strftime("%Y-%m-%d")

        message = f"✅ **{date_str} {market_name}**: リバランス不要（トリガー条件未達）"
        return self._send_message(message)

    def send_error_notification(self, error: str | Exception, market: str) -> bool:
        """
        Send an error notification.

        Args:
            error: Error message or exception
            market: Market identifier ("US" or "JP")

        Returns:
            True if notification was sent successfully, False otherwise
        """
        market_name = "米国市場" if market == "US" else "日本市場"
        error_msg = str(error)

        message = f"❌ **{market_name} エラー**\n```\n{error_msg[:1500]}\n```"
        return self._send_message(message)

    def send_startup_notification(self, enabled_markets: list[str]) -> bool:
        """
        Send a startup notification.

        Args:
            enabled_markets: List of enabled market identifiers

        Returns:
            True if notification was sent successfully, False otherwise
        """
        markets_str = ", ".join(enabled_markets)
        message = f"🚀 **トレーディングスケジューラ起動**\n有効な市場: {markets_str}"
        return self._send_message(message)

    def send_portfolio_rebalance(
        self,
        portfolio_name: str,
        holdings: "Holdings",
        orders: "OrderSummary",
        adjustment: "LotAdjustmentResult | None" = None,
    ) -> bool:
        """
        Send a portfolio rebalance notification with order details.

        Args:
            portfolio_name: ポートフォリオ名
            holdings: 現在の保有資産
            orders: 発注サマリ
            adjustment: ロット調整結果（オプション）

        Returns:
            True if notification was sent successfully, False otherwise
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        currency = holdings.currency

        # 通貨フォーマット
        def fmt_amount(amount: float) -> str:
            if currency == "JPY":
                return f"¥{amount:,.0f}"
            return f"${amount:,.2f}"

        # ヘッダー
        lines = [
            f"📊 **{date_str} {portfolio_name} リバランス通知**",
            "",
        ]

        # 現在資産サマリ
        lines.append(f"💰 **現在資産**: {fmt_amount(holdings.total_value)} (現金: {fmt_amount(holdings.cash)})")
        lines.append(f"📈 保有銘柄数: {holdings.position_count}")
        lines.append("")

        # 売り注文
        sell_orders = orders.sell_orders()
        if sell_orders:
            lines.append(f"📉 **売り注文 ({len(sell_orders)}銘柄)**")
            for order in sell_orders[:10]:  # 最大10件
                shares_str = f"{order.shares:.2f}" if order.is_fractional else f"{int(order.shares)}"
                lines.append(
                    f"  {order.symbol}: -{shares_str}株 @ {fmt_amount(order.price)} = {fmt_amount(order.amount)}"
                )
            if len(sell_orders) > 10:
                lines.append(f"  ...他 {len(sell_orders) - 10}銘柄")
            lines.append(f"  **売り合計**: {fmt_amount(orders.total_sell_amount)}")
            lines.append("")

        # 買い注文
        buy_orders = orders.buy_orders()
        if buy_orders:
            lines.append(f"📈 **買い注文 ({len(buy_orders)}銘柄)**")
            for order in buy_orders[:10]:  # 最大10件
                shares_str = f"{order.shares:.2f}" if order.is_fractional else f"{int(order.shares)}"
                lines.append(
                    f"  {order.symbol}: +{shares_str}株 @ {fmt_amount(order.price)} = {fmt_amount(order.amount)}"
                )
            if len(buy_orders) > 10:
                lines.append(f"  ...他 {len(buy_orders) - 10}銘柄")
            lines.append(f"  **買い合計**: {fmt_amount(orders.total_buy_amount)}")
            lines.append("")

        # 売買サマリ
        lines.append("💵 **売買サマリ**")
        lines.append(f"  売り合計: {fmt_amount(orders.total_sell_amount)}")
        lines.append(f"  買い合計: {fmt_amount(orders.total_buy_amount)}")

        net = orders.net_amount
        if net > 0:
            lines.append(f"  **必要追加資金**: {fmt_amount(net)}")
            # 現金で賄えるかチェック
            if holdings.cash >= net:
                lines.append(f"  ✅ 現金で賄えます（残高: {fmt_amount(holdings.cash - net)}）")
            else:
                shortfall = net - holdings.cash
                lines.append(f"  ⚠️ 現金不足: {fmt_amount(shortfall)} 追加入金が必要")
        else:
            lines.append(f"  **余剰資金**: {fmt_amount(-net)}")

        # ロット調整情報
        if adjustment and adjustment.cash_remainder > 0:
            lines.append("")
            lines.append(f"🔧 端数現金: {fmt_amount(adjustment.cash_remainder)}")
            if adjustment.weight_deviation > 0:
                lines.append(f"  目標乖離(RMSE): {adjustment.weight_deviation * 100:.2f}%")

        message = "\n".join(lines)
        return self._send_message(message)

    def send_portfolio_no_rebalance(
        self,
        portfolio_name: str,
        reason: str = "トリガー条件未達",
    ) -> bool:
        """
        Send a notification that portfolio rebalance is not needed.

        Args:
            portfolio_name: ポートフォリオ名
            reason: 理由

        Returns:
            True if notification was sent successfully, False otherwise
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        message = f"✅ **{date_str} {portfolio_name}**: リバランス不要（{reason}）"
        return self._send_message(message)

    def send_portfolio_error(
        self,
        portfolio_name: str,
        error: str | Exception,
    ) -> bool:
        """
        Send a portfolio error notification.

        Args:
            portfolio_name: ポートフォリオ名
            error: エラーメッセージまたは例外

        Returns:
            True if notification was sent successfully, False otherwise
        """
        error_msg = str(error)
        message = f"❌ **{portfolio_name} エラー**\n```\n{error_msg[:1500]}\n```"
        return self._send_message(message)

    def send_scheduler_startup(self, portfolios: list[str]) -> bool:
        """
        Send a scheduler startup notification.

        Args:
            portfolios: スケジュール有効なポートフォリオIDリスト

        Returns:
            True if notification was sent successfully, False otherwise
        """
        portfolios_str = ", ".join(portfolios) if portfolios else "(なし)"
        message = (
            f"🚀 **ポートフォリオスケジューラ起動**\n"
            f"有効なポートフォリオ: {portfolios_str}"
        )
        return self._send_message(message)

    def _send_message(self, content: str) -> bool:
        """
        Send a message to Discord webhook.

        Args:
            content: Message content (markdown supported)

        Returns:
            True if message was sent successfully, False otherwise
        """
        logger.info(f"Discord notification:\n{content}")

        if not self.webhook_url:
            logger.debug("Webhook URL not configured - skipping Discord send")
            return True  # Consider it successful for testing

        try:
            response = requests.post(
                self.webhook_url,
                json={"content": content},
                timeout=30,
            )
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
            return True

        except requests.exceptions.Timeout:
            logger.error("Discord notification timed out")
            return False

        except requests.exceptions.HTTPError as e:
            logger.error(f"Discord notification failed: {e.response.status_code} - {e.response.text}")
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Discord notification failed: {e}")
            return False
