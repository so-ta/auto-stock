"""
Portfolio Scheduler Service - ポートフォリオスケジューラ

各ポートフォリオのスケジュール設定に基づいて、
定期的なリバランスチェックを実行する。

使用例:
    from scripts.result_viewer.services.scheduler_service import PortfolioScheduler

    scheduler = PortfolioScheduler(project_root)
    await scheduler.start()
    # ... アプリケーション実行 ...
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# APScheduler is optional
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    AsyncIOScheduler = None
    CronTrigger = None


class PortfolioScheduler:
    """ポートフォリオスケジューラ

    各ポートフォリオのスケジュール設定に基づいて
    定期的なリバランスチェックを実行する。

    使用例:
        scheduler = PortfolioScheduler(project_root)
        await scheduler.start()
    """

    def __init__(
        self,
        project_root: Path,
        config_dir: Path | None = None,
        state_dir: Path | None = None,
        on_rebalance: Optional[Callable] = None,
    ):
        """初期化

        Args:
            project_root: プロジェクトルートディレクトリ
            config_dir: ポートフォリオ設定ディレクトリ
            state_dir: 保有資産状態ディレクトリ
            on_rebalance: リバランス完了時のコールバック
        """
        if not HAS_APSCHEDULER:
            logger.warning("APScheduler not installed - scheduling disabled")

        self.project_root = project_root
        self.config_dir = config_dir or (project_root / "config" / "portfolios")
        self.state_dir = state_dir or (project_root / "data" / "portfolio_state")
        self.on_rebalance = on_rebalance

        self.scheduler: Optional[AsyncIOScheduler] = None
        self.jobs: dict[str, str] = {}  # portfolio_id -> job_id
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """スケジューラが実行中かどうか"""
        return self._is_running

    async def start(self) -> None:
        """スケジューラを開始"""
        if not HAS_APSCHEDULER:
            logger.warning("Cannot start scheduler - APScheduler not installed")
            return

        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        self._is_running = True

        # 全ポートフォリオのスケジュールを登録
        await self._register_all_portfolios()

        # 起動通知
        await self._send_startup_notification()

        logger.info("Portfolio scheduler started")

    async def stop(self) -> None:
        """スケジューラを停止"""
        if not self._is_running or self.scheduler is None:
            return

        self.scheduler.shutdown(wait=False)
        self._is_running = False
        self.jobs.clear()
        logger.info("Portfolio scheduler stopped")

    async def refresh(self) -> None:
        """スケジュールを再読み込み

        ポートフォリオ設定が変更された場合に呼び出す。
        """
        if not self._is_running:
            return

        # 既存ジョブをクリア
        for portfolio_id, job_id in list(self.jobs.items()):
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass
            del self.jobs[portfolio_id]

        # 再登録
        await self._register_all_portfolios()
        logger.info("Portfolio schedules refreshed")

    def register_portfolio(self, portfolio_id: str) -> bool:
        """ポートフォリオのスケジュールを登録

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            成功したらTrue
        """
        if not self._is_running or self.scheduler is None:
            return False

        from src.portfolio.manager import PortfolioManager
        manager = PortfolioManager(self.config_dir, self.state_dir)
        config = manager.get_portfolio_config(portfolio_id)

        if not config:
            logger.warning(f"Portfolio not found: {portfolio_id}")
            return False

        if not config.schedule.enabled:
            logger.debug(f"Schedule disabled for {portfolio_id}")
            return False

        # 既存ジョブがあれば削除
        if portfolio_id in self.jobs:
            try:
                self.scheduler.remove_job(self.jobs[portfolio_id])
            except Exception:
                pass

        # 新規ジョブを登録
        try:
            tz = ZoneInfo(config.schedule.timezone)
            trigger = CronTrigger(
                hour=config.schedule.hour,
                minute=config.schedule.minute,
                timezone=tz,
            )

            job = self.scheduler.add_job(
                self._run_portfolio_check,
                trigger,
                args=[portfolio_id],
                id=f"portfolio_{portfolio_id}",
                name=f"Portfolio check: {config.name}",
                replace_existing=True,
            )

            self.jobs[portfolio_id] = job.id
            logger.info(
                f"Registered schedule for {portfolio_id}: "
                f"{config.schedule.hour:02d}:{config.schedule.minute:02d} ({config.schedule.timezone})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register schedule for {portfolio_id}: {e}")
            return False

    def unregister_portfolio(self, portfolio_id: str) -> bool:
        """ポートフォリオのスケジュールを解除

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            成功したらTrue
        """
        if portfolio_id not in self.jobs:
            return True

        try:
            self.scheduler.remove_job(self.jobs[portfolio_id])
            del self.jobs[portfolio_id]
            logger.info(f"Unregistered schedule for {portfolio_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister schedule for {portfolio_id}: {e}")
            return False

    def get_scheduled_portfolios(self) -> list[str]:
        """スケジュール登録済みポートフォリオ一覧を取得"""
        return list(self.jobs.keys())

    def get_next_run_time(self, portfolio_id: str) -> datetime | None:
        """次回実行時刻を取得

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            次回実行時刻 or None
        """
        if portfolio_id not in self.jobs or self.scheduler is None:
            return None

        job = self.scheduler.get_job(self.jobs[portfolio_id])
        if job and job.next_run_time:
            return job.next_run_time
        return None

    async def run_now(self, portfolio_id: str) -> dict:
        """手動でリバランスチェックを実行

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            実行結果
        """
        return await self._run_portfolio_check(portfolio_id)

    async def _register_all_portfolios(self) -> None:
        """全ポートフォリオのスケジュールを登録"""
        from src.portfolio.manager import PortfolioManager
        manager = PortfolioManager(self.config_dir, self.state_dir)

        for config in manager.list_portfolios():
            if config.schedule.enabled:
                self.register_portfolio(config.id)

    async def _run_portfolio_check(self, portfolio_id: str) -> dict:
        """ポートフォリオのリバランスチェックを実行

        Args:
            portfolio_id: ポートフォリオID

        Returns:
            実行結果の辞書
        """
        from src.portfolio.manager import PortfolioManager
        from src.notification.discord import DiscordNotifier

        logger.info(f"Running portfolio check: {portfolio_id}")
        result = {
            "portfolio_id": portfolio_id,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            manager = PortfolioManager(self.config_dir, self.state_dir)
            config = manager.get_portfolio_config(portfolio_id)

            if not config:
                result["status"] = "error"
                result["error"] = "Portfolio not found"
                return result

            # 保有資産を読み込み
            holdings = manager.load_holdings(portfolio_id)
            if not holdings:
                result["status"] = "error"
                result["error"] = "No holdings found"
                return result

            # yfinanceから最新価格を取得
            symbols = list(holdings.positions.keys())
            if symbols:
                prices = await self._fetch_latest_prices(symbols)
                # 取得できなかった銘柄は既存の価格を使用
                for symbol, pos in holdings.positions.items():
                    if symbol not in prices:
                        prices[symbol] = pos.current_price
                # 保有資産の価格を更新
                manager.update_prices(portfolio_id, prices)
            else:
                prices = {}

            # 日次スナップショット記録（リバランス前）
            manager.record_snapshot(portfolio_id, is_rebalance=False)

            # 目標重みを計算（ここでは現在の重みをそのまま使用）
            # 実際の運用ではパイプラインから目標重みを取得する
            # TODO: パイプライン連携を実装
            target_weights = holdings.get_current_weights()

            # リバランス計算
            rebalance_result = manager.calculate_rebalance(
                portfolio_id=portfolio_id,
                target_weights=target_weights,
                prices=prices,
                threshold=0.01,
            )

            result["needs_rebalance"] = rebalance_result.needs_rebalance
            result["message"] = rebalance_result.message

            # 通知を送信
            notifier = self._get_notifier()

            if rebalance_result.needs_rebalance:
                result["status"] = "rebalance_needed"

                if config.notification.on_rebalance and rebalance_result.orders:
                    notifier.send_portfolio_rebalance(
                        portfolio_name=config.name,
                        holdings=holdings,
                        orders=rebalance_result.orders,
                        adjustment=rebalance_result.adjustment,
                    )

                # リバランス後のスナップショット記録
                # 注: 実際の約定後に記録するべきだが、ここでは通知のみ
                # manager.record_snapshot(portfolio_id, is_rebalance=True)

            else:
                result["status"] = "no_rebalance"

                if config.notification.on_no_rebalance:
                    notifier.send_portfolio_no_rebalance(
                        portfolio_name=config.name,
                        reason=rebalance_result.message,
                    )

            # コールバック実行
            if self.on_rebalance:
                try:
                    if asyncio.iscoroutinefunction(self.on_rebalance):
                        await self.on_rebalance(result)
                    else:
                        self.on_rebalance(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            logger.info(f"Portfolio check completed: {portfolio_id} - {result['status']}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.exception(f"Portfolio check failed: {portfolio_id}")

            # エラー通知
            try:
                notifier = self._get_notifier()
                config = manager.get_portfolio_config(portfolio_id)
                if config:
                    notifier.send_portfolio_error(config.name, e)
            except Exception:
                pass

        return result

    async def _fetch_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """yfinanceから最新価格を取得

        Args:
            symbols: 銘柄シンボルのリスト

        Returns:
            symbol -> price の辞書
        """
        from src.data.batch_fetcher import BatchDataFetcher

        if not symbols:
            return {}

        try:
            fetcher = BatchDataFetcher(
                max_concurrent=5,
                rate_limit_per_sec=2.0,
                cache_max_age_days=0,  # 常に最新を取得
            )

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            # 非同期で取得
            batch_result = await fetcher.fetch_all(symbols, start_date, end_date)

            prices = {}
            for symbol, result in batch_result.results.items():
                if result.data is not None and len(result.data) > 0:
                    df = result.data
                    # Close列を探す
                    close_col = None
                    for col in df.columns:
                        col_str = str(col).lower()
                        if 'close' in col_str:
                            close_col = col
                            break

                    if close_col:
                        prices[symbol] = float(df[close_col].to_list()[-1])
                    else:
                        logger.warning(f"No Close column found for {symbol}")

            logger.info(f"Fetched prices for {len(prices)}/{len(symbols)} symbols")
            return prices

        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return {}

    def _get_notifier(self) -> "DiscordNotifier":
        """DiscordNotifier インスタンスを取得"""
        from src.notification.discord import DiscordNotifier
        import yaml

        notification_config_path = self.project_root / "config" / "notification.yaml"
        webhook_url = None

        if notification_config_path.exists():
            try:
                with open(notification_config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                webhook_url = config.get("discord", {}).get("webhook_url")
            except Exception as e:
                logger.warning(f"Failed to load notification config: {e}")

        return DiscordNotifier(webhook_url=webhook_url)

    async def _send_startup_notification(self) -> None:
        """起動通知を送信"""
        scheduled = self.get_scheduled_portfolios()
        if not scheduled:
            return

        try:
            notifier = self._get_notifier()
            notifier.send_scheduler_startup(scheduled)
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")


# シングルトンインスタンス
_scheduler_instance: Optional[PortfolioScheduler] = None


def get_scheduler(
    project_root: Path | None = None,
    config_dir: Path | None = None,
    state_dir: Path | None = None,
) -> PortfolioScheduler:
    """スケジューラのシングルトンインスタンスを取得

    Args:
        project_root: プロジェクトルート（初回のみ必要）
        config_dir: 設定ディレクトリ
        state_dir: 状態ディレクトリ

    Returns:
        PortfolioScheduler instance
    """
    global _scheduler_instance

    if _scheduler_instance is None:
        if project_root is None:
            raise ValueError("project_root is required for first initialization")
        _scheduler_instance = PortfolioScheduler(
            project_root=project_root,
            config_dir=config_dir,
            state_dir=state_dir,
        )

    return _scheduler_instance
