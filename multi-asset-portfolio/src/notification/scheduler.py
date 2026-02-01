"""
Trading Scheduler - Foreground Scheduler for Daily Market Checks.

Runs as a foreground process, checking markets at specified times using APScheduler.

Usage:
    # Start scheduler (foreground)
    python -m src.notification.scheduler

    # Force immediate check
    python -m src.notification.scheduler --market US --force

    # Check specific market without sending notification
    python -m src.notification.scheduler --market JP --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Setup logging before imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.notification.discord import DiscordNotifier, RebalanceNotification
from src.notification.portfolio_state import PortfolioState, PortfolioStateManager


# Market configuration
MARKET_CONFIG = {
    "US": {
        "name": "米国市場",
        "calendar": "NYSE",
        "timezone": "America/New_York",
        "universe_key": "us_stocks",
    },
    "JP": {
        "name": "日本市場",
        "calendar": "JPX",
        "timezone": "Asia/Tokyo",
        "universe_key": "japan_stocks",
    },
}


class TradingScheduler:
    """
    Foreground trading scheduler using APScheduler.

    Monitors markets at specified times and sends Discord notifications
    when rebalancing is needed.
    """

    def __init__(self, settings: Any | None = None) -> None:
        """Initialize the trading scheduler.

        Args:
            settings: Optional Settings instance. If not provided, loads from
                      default config and environment variables.
        """
        # Load environment variables
        load_dotenv()

        # Load settings if not provided
        if settings is None:
            try:
                from src.config.settings import get_settings
                settings = get_settings()
            except Exception as e:
                logger.warning(f"Could not load settings, using defaults: {e}")
                settings = None

        # Initialize components - use settings if available, fallback to env vars
        notif = settings.notification if settings else None

        self.webhook_url = (
            (notif.discord_webhook_url if notif else None)
            or os.getenv("DISCORD_WEBHOOK_URL")
        )
        self.notifier = DiscordNotifier(webhook_url=self.webhook_url)
        self.state_manager = PortfolioStateManager(state_dir="data/portfolio_state")

        # Schedule configuration from settings or environment
        self.notify_us_hour = (
            notif.notify_us_hour if notif else int(os.getenv("NOTIFY_US_HOUR", "7"))
        )
        self.notify_us_minute = (
            notif.notify_us_minute if notif else int(os.getenv("NOTIFY_US_MINUTE", "30"))
        )
        self.notify_jp_hour = (
            notif.notify_jp_hour if notif else int(os.getenv("NOTIFY_JP_HOUR", "16"))
        )
        self.notify_jp_minute = (
            notif.notify_jp_minute if notif else int(os.getenv("NOTIFY_JP_MINUTE", "30"))
        )

        # Enabled markets
        if notif:
            self.enabled_markets = notif.enabled_markets
        else:
            enabled_markets_str = os.getenv("ENABLED_MARKETS", "US,JP")
            self.enabled_markets = [m.strip() for m in enabled_markets_str.split(",") if m.strip()]

        # Scheduler instance (lazy init)
        self._scheduler = None

        # Log configuration
        logger.info(
            f"TradingScheduler initialized: "
            f"US={self.notify_us_hour:02d}:{self.notify_us_minute:02d} JST, "
            f"JP={self.notify_jp_hour:02d}:{self.notify_jp_minute:02d} JST, "
            f"markets={self.enabled_markets}"
        )

    def _get_calendar_manager(self) -> Any:
        """Get or create CalendarManager."""
        from src.data.calendar_manager import CalendarManager
        return CalendarManager()

    def _load_universe(self, market: str) -> list[str]:
        """
        Load universe for a market.

        Args:
            market: Market identifier ("US" or "JP")

        Returns:
            List of ticker symbols
        """
        config_path = project_root / "config" / "universe_standard.yaml"
        if not config_path.exists():
            logger.warning(f"Universe config not found: {config_path}")
            return []

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        market_config = MARKET_CONFIG.get(market, {})
        universe_key = market_config.get("universe_key", "")

        if market == "US":
            # US stocks are in us_stocks.symbols
            return config.get("us_stocks", {}).get("symbols", [])
        elif market == "JP":
            # Japan stocks are in japan_stocks.symbols
            return config.get("japan_stocks", {}).get("symbols", [])
        else:
            logger.warning(f"Unknown market: {market}")
            return []

    def _is_trading_day(self, market: str, check_date: date | None = None) -> bool:
        """
        Check if the given date is a trading day.

        Args:
            market: Market identifier
            check_date: Date to check (default: today)

        Returns:
            True if it's a trading day
        """
        if check_date is None:
            check_date = date.today()

        calendar_name = MARKET_CONFIG.get(market, {}).get("calendar", "NYSE")

        try:
            calendar_manager = self._get_calendar_manager()
            return calendar_manager.is_trading_day(calendar_name, check_date)
        except Exception as e:
            logger.warning(f"Calendar check failed, assuming trading day: {e}")
            # Fallback: weekday is a trading day
            return check_date.weekday() < 5

    def _should_rebalance(
        self,
        market: str,
        previous_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """
        Determine if rebalancing is needed using EventDrivenRebalanceScheduler.

        Args:
            market: Market identifier
            previous_weights: Previous portfolio weights
            new_weights: New calculated weights

        Returns:
            Tuple of (should_rebalance, trigger_reasons)
        """
        from src.backtest.rebalance_scheduler import (
            EventDrivenRebalanceScheduler,
            MarketData,
            PortfolioState as RebalancePortfolioState,
            TriggerType,
        )

        # Create rebalance scheduler
        scheduler = EventDrivenRebalanceScheduler(
            base_frequency="monthly",
            min_interval_days=5,
        )

        # Build portfolio state
        portfolio_state = RebalancePortfolioState(
            current_weights=previous_weights,
            target_weights=new_weights,
            portfolio_value=1_000_000,  # Placeholder
        )

        # Build market data (minimal - we don't have VIX etc. in this context)
        market_data = MarketData(
            date=datetime.now(),
            vix=None,  # Could fetch from yfinance if needed
        )

        # Check rebalance decision
        decision = scheduler.should_rebalance(
            date=datetime.now(),
            portfolio_state=portfolio_state,
            market_data=market_data,
        )

        trigger_reasons = []
        if decision.should_rebalance:
            for result in decision.trigger_results:
                if result.triggered:
                    if result.trigger_type == TriggerType.SCHEDULED:
                        trigger_reasons.append("月次定期")
                    elif result.trigger_type == TriggerType.POSITION_DEVIATION:
                        trigger_reasons.append("ポジション乖離")
                    elif result.trigger_type == TriggerType.VIX_SPIKE:
                        trigger_reasons.append("VIXスパイク")
                    elif result.trigger_type == TriggerType.REGIME_CHANGE:
                        trigger_reasons.append("レジーム変化")
                    elif result.trigger_type == TriggerType.DRAWDOWN:
                        trigger_reasons.append("ドローダウン")

        return decision.should_rebalance, trigger_reasons

    def _calculate_new_weights(
        self,
        market: str,
        universe: list[str],
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate new portfolio weights using UnifiedExecutor.

        Args:
            market: Market identifier
            universe: List of ticker symbols
            previous_weights: Previous weights

        Returns:
            Dictionary of symbol -> weight
        """
        from src.orchestrator.unified_executor import UnifiedExecutor

        logger.info(f"Calculating weights for {market}: {len(universe)} assets")

        try:
            executor = UnifiedExecutor()
            result = executor.run_single(
                universe=universe,
                as_of_date=datetime.now(),
                previous_weights=previous_weights,
            )

            if result.weights:
                logger.info(f"Calculated weights: {len(result.weights)} positions")
                return result.weights

        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")

        # Fallback: equal weight for top N assets
        n = min(20, len(universe))
        weight = 1.0 / n
        return {symbol: weight for symbol in universe[:n]}

    def _build_notification(
        self,
        market: str,
        previous_weights: dict[str, float],
        new_weights: dict[str, float],
        trigger_reasons: list[str],
    ) -> RebalanceNotification:
        """
        Build a rebalance notification from weight changes.

        Args:
            market: Market identifier
            previous_weights: Previous weights
            new_weights: New weights
            trigger_reasons: List of trigger reasons

        Returns:
            RebalanceNotification object
        """
        buys: dict[str, dict[str, float]] = {}
        sells: dict[str, dict[str, float]] = {}

        all_symbols = set(previous_weights.keys()) | set(new_weights.keys())
        total_turnover = 0.0

        for symbol in all_symbols:
            old_weight = previous_weights.get(symbol, 0)
            new_weight = new_weights.get(symbol, 0)
            change = new_weight - old_weight

            total_turnover += abs(change)

            # Threshold for significant change
            if abs(change) < 0.005:  # 0.5%
                continue

            data = {
                "old_weight": old_weight,
                "new_weight": new_weight,
                "change": change,
            }

            if change > 0:
                buys[symbol] = data
            else:
                sells[symbol] = data

        # Calculate cash weight (assuming weights should sum to 1.0)
        old_sum = sum(previous_weights.values())
        new_sum = sum(new_weights.values())
        cash_old = max(0, 1.0 - old_sum)
        cash_new = max(0, 1.0 - new_sum)

        return RebalanceNotification(
            market=market,
            date=datetime.now(),
            trigger_reasons=trigger_reasons,
            buys=buys,
            sells=sells,
            estimated_turnover=total_turnover / 2,  # One-way turnover
            cash_weight_old=cash_old,
            cash_weight_new=cash_new,
        )

    def check_market(self, market: str, force: bool = False, dry_run: bool = False) -> bool:
        """
        Check a market for rebalancing needs and send notification.

        Args:
            market: Market identifier ("US" or "JP")
            force: Skip trading day check
            dry_run: Don't send notification

        Returns:
            True if check completed successfully
        """
        logger.info(f"Checking market: {market} (force={force}, dry_run={dry_run})")

        # Trading day check
        if not force and not self._is_trading_day(market):
            logger.info(f"{market}: Not a trading day, skipping")
            return True

        try:
            # Load universe
            universe = self._load_universe(market)
            if not universe:
                logger.error(f"{market}: Failed to load universe")
                if not dry_run:
                    self.notifier.send_error_notification("Universe loading failed", market)
                return False

            logger.info(f"{market}: Loaded {len(universe)} symbols")

            # Load previous weights
            previous_weights = self.state_manager.get_previous_weights(market)
            logger.info(f"{market}: Previous positions: {len(previous_weights)}")

            # Calculate new weights
            new_weights = self._calculate_new_weights(market, universe, previous_weights)
            logger.info(f"{market}: New positions: {len(new_weights)}")

            # Check if rebalance is needed
            should_rebalance, trigger_reasons = self._should_rebalance(
                market, previous_weights, new_weights
            )

            if not should_rebalance:
                logger.info(f"{market}: No rebalance needed")
                if not dry_run:
                    self.notifier.send_no_rebalance_notification(market, datetime.now())
                return True

            # Build and send notification
            notification = self._build_notification(
                market, previous_weights, new_weights, trigger_reasons
            )

            if dry_run:
                logger.info(f"{market}: Dry run - notification not sent")
                logger.info(f"  Buys: {list(notification.buys.keys())}")
                logger.info(f"  Sells: {list(notification.sells.keys())}")
                logger.info(f"  Turnover: {notification.estimated_turnover:.2%}")
            else:
                success = self.notifier.send_rebalance_notification(notification)
                if success:
                    # Update state
                    self.state_manager.update_weights(market, new_weights, is_rebalance=True)
                    logger.info(f"{market}: Notification sent and state updated")
                else:
                    logger.error(f"{market}: Failed to send notification")
                    return False

            return True

        except Exception as e:
            logger.exception(f"{market}: Error during market check: {e}")
            if not dry_run:
                self.notifier.send_error_notification(e, market)
            return False

    def start(self) -> None:
        """
        Start the foreground scheduler.

        This method blocks until interrupted (Ctrl+C).
        """
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Install with: pip install apscheduler")
            sys.exit(1)

        self._scheduler = BlockingScheduler()

        # Schedule US market check (07:30 JST = 17:30 ET)
        if "US" in self.enabled_markets:
            self._scheduler.add_job(
                self.check_market,
                CronTrigger(
                    hour=self.notify_us_hour,
                    minute=self.notify_us_minute,
                    timezone="Asia/Tokyo",
                ),
                args=["US"],
                id="us_market_check",
                name="US Market Check",
            )
            logger.info(
                f"Scheduled US market check at {self.notify_us_hour:02d}:{self.notify_us_minute:02d} JST"
            )

        # Schedule JP market check (16:30 JST)
        if "JP" in self.enabled_markets:
            self._scheduler.add_job(
                self.check_market,
                CronTrigger(
                    hour=self.notify_jp_hour,
                    minute=self.notify_jp_minute,
                    timezone="Asia/Tokyo",
                ),
                args=["JP"],
                id="jp_market_check",
                name="JP Market Check",
            )
            logger.info(
                f"Scheduled JP market check at {self.notify_jp_hour:02d}:{self.notify_jp_minute:02d} JST"
            )

        # Send startup notification
        self.notifier.send_startup_notification(self.enabled_markets)

        logger.info("Scheduler started. Press Ctrl+C to exit.")
        logger.info("Next scheduled runs:")
        for job in self._scheduler.get_jobs():
            # APScheduler 3.x uses next_run_time attribute
            next_run = getattr(job, "next_run_time", None)
            if next_run is None:
                # Fallback for APScheduler 4.x
                try:
                    next_run = job.trigger.get_next_fire_time(None, datetime.now())
                except Exception:
                    next_run = "unknown"
            logger.info(f"  {job.name}: {next_run}")

        try:
            self._scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            if self._scheduler:
                self._scheduler.shutdown()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Scheduler - Daily market check and Discord notification"
    )
    parser.add_argument(
        "--market",
        choices=["US", "JP"],
        help="Market to check immediately (skip scheduler)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force check even if not a trading day",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't send notifications or update state",
    )

    args = parser.parse_args()

    # Setup file logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "scheduler.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    scheduler = TradingScheduler()

    if args.market:
        # Immediate check mode
        success = scheduler.check_market(
            market=args.market,
            force=args.force,
            dry_run=args.dry_run,
        )
        sys.exit(0 if success else 1)
    else:
        # Start scheduler mode
        scheduler.start()


if __name__ == "__main__":
    main()
