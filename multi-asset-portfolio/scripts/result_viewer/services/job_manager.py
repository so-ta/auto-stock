"""
Job Manager - Background job management for backtest execution

multiprocessingを使用してバックテストをバックグラウンドで実行し、
ProgressTrackerと連携して進捗を追跡する。
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import threading

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """ジョブステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestJob:
    """バックテストジョブ情報"""
    job_id: str
    run_id: str  # ProgressTracker用のrun_id
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 設定
    universe_file: str = ""
    start_date: str = ""
    end_date: str = ""
    frequency: str = "weekly"
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # 結果
    archive_id: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        result = asdict(self)
        result["status"] = self.status.value
        result["created_at"] = self.created_at.isoformat()
        if self.started_at:
            result["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        return result


def _run_backtest_process(
    job_dict: Dict[str, Any],
    project_root: str,
    results_dir: str,
) -> None:
    """
    子プロセスでバックテストを実行

    Args:
        job_dict: ジョブ情報（BacktestJob.to_dict()の結果）
        project_root: プロジェクトルートパス
        results_dir: 結果保存ディレクトリ
    """
    # プロジェクトルートをパスに追加（統一初期化関数インポート用）
    sys.path.insert(0, project_root)

    run_id = job_dict["run_id"]

    # 統一初期化関数を使用
    # これによりCLIと同じロギング設定が適用される
    from src.orchestrator.backtest_initializer import initialize_backtest_process

    init_result = initialize_backtest_process(
        project_root=Path(project_root),
        run_id=run_id,
        results_dir=Path(results_dir),
        enable_progress_tracking=True,
        enable_log_collector=True,
    )

    settings = init_result["settings"]
    tracker = init_result["progress_tracker"]
    log_collector = init_result["log_collector"]

    try:
        tracker.set_phase("initializing")
        tracker.set_phase_total("initializing", 1)

        # 遅延インポート（Settingsロード前にpathを設定する必要があるため）
        from src.orchestrator.unified_executor import UnifiedExecutor
        from src.analysis.result_store import BacktestResultStore
        from src.data.universe_loader import UniverseLoader

        # ジョブパラメータを取得
        universe_name = job_dict.get("universe_file", "all")
        start_date = job_dict.get("start_date")
        end_date = job_dict.get("end_date")
        frequency = job_dict.get("frequency", "monthly")

        tracker.increment_phase("initializing")
        tracker.complete_phase("initializing")

        # ユニバースのシンボルリストを取得
        tracker.set_phase("data_loading")

        # ConfigServiceからシンボルを取得
        from scripts.result_viewer.services.config_service import get_config_service
        config_service = get_config_service(project_root=Path(project_root))
        symbols = config_service.get_universe_symbols(universe_name)

        if not symbols:
            raise ValueError(f"ユニバース '{universe_name}' にシンボルが見つかりません")

        tracker.set_phase_total("data_loading", len(symbols))
        logger.info(f"Loading data for {len(symbols)} symbols")

        # 価格データを取得
        from src.data.adapters.stock import StockAdapter
        from datetime import datetime as dt
        import pandas as pd

        adapter = StockAdapter()
        prices = {}
        loaded = 0

        # 日付をdatetimeに変換
        start_dt = dt.strptime(start_date, "%Y-%m-%d")
        end_dt = dt.strptime(end_date, "%Y-%m-%d")

        for symbol in symbols:
            try:
                ohlcv = adapter.fetch_ohlcv(symbol, start_dt, end_dt)
                if ohlcv is not None and ohlcv.data is not None and not ohlcv.data.is_empty():
                    # Polars DataFrame を Pandas DataFrame に変換
                    pdf = ohlcv.data.to_pandas()

                    # timestamp列をインデックスに設定
                    if "timestamp" in pdf.columns:
                        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
                        pdf = pdf.set_index("timestamp")

                    prices[symbol] = pdf
                    loaded += 1
                    tracker.increment_phase("data_loading")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")

        if not prices:
            raise ValueError("価格データを取得できませんでした")

        tracker.complete_phase("data_loading")
        logger.info(f"Loaded price data for {loaded}/{len(symbols)} symbols")

        # バックテスト実行
        tracker.set_phase("backtest")

        executor = UnifiedExecutor(settings=settings)

        # config_overridesを適用
        overrides = job_dict.get("config_overrides", {})
        initial_capital = overrides.get("backtest", {}).get("initial_capital", 100000)
        cost_bps = settings.cost_model.spread_bps + settings.cost_model.commission_bps
        slippage_bps = settings.cost_model.slippage_bps

        logger.info(
            f"Running backtest: {start_date} to {end_date}, "
            f"frequency={frequency}, capital=${initial_capital:,.0f}"
        )

        result = executor.run_backtest(
            universe=list(prices.keys()),
            prices=prices,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            initial_capital=initial_capital,
            transaction_cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            progress_tracker=tracker,
        )

        tracker.complete_phase("backtest")

        # 結果を保存
        tracker.set_phase("saving")
        tracker.set_phase_total("saving", 1)

        store = BacktestResultStore(results_dir)
        archive_id = store.save(
            result=result,
            name=f"Web Backtest - {run_id}",
            description=f"Universe: {universe_name}, Period: {start_date} to {end_date}",
            tags=["web", "automated"],
        )

        tracker.increment_phase("saving")
        tracker.complete_phase("saving")

        # ログを保存（信頼性スコア計算用）
        if log_collector is not None:
            logs_file = tracker.progress_dir / f"{run_id}_logs.jsonl"
            log_collector.save_to_file(logs_file)

        # 成功結果をファイルに書き込み
        result_file = tracker.progress_dir / f"{run_id}.result"
        with open(result_file, "w") as f:
            f.write(archive_id)

        tracker.complete()
        logger.info(f"Backtest completed: archive_id={archive_id}")

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception(f"Backtest failed: {error_msg}")
        tracker.fail(error_msg)

        # ログを保存（エラー解析用）
        if log_collector is not None:
            logs_file = tracker.progress_dir / f"{run_id}_logs.jsonl"
            log_collector.save_partial(logs_file)

        # エラーをファイルに書き込み（詳細なスタックトレース付き）
        error_file = tracker.progress_dir / f"{run_id}.error"
        with open(error_file, "w") as f:
            f.write(f"{error_msg}\n\n")
            f.write(traceback.format_exc())


class JobManager:
    """
    バックテストジョブ管理クラス

    multiprocessingを使用してバックテストをバックグラウンドで実行。
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        max_concurrent_jobs: int = 2,
    ):
        """
        初期化

        Args:
            project_root: プロジェクトルートパス
            results_dir: 結果保存ディレクトリ
            max_concurrent_jobs: 最大同時実行ジョブ数
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.results_dir = results_dir or self.project_root / "results"
        self.max_concurrent_jobs = max_concurrent_jobs

        self._jobs: Dict[str, BacktestJob] = {}
        self._processes: Dict[str, multiprocessing.Process] = {}
        self._lock = threading.Lock()

        # 進捗ディレクトリ作成
        (self.results_dir / ".progress").mkdir(parents=True, exist_ok=True)

        logger.info(f"JobManager initialized: project_root={self.project_root}")

    def create_job(
        self,
        universe_file: str,
        start_date: str,
        end_date: str,
        frequency: str = "weekly",
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestJob:
        """
        新規ジョブを作成

        Args:
            universe_file: ユニバースファイル名（例: "universe_sbi.yaml"）
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            frequency: リバランス頻度（daily/weekly/monthly）
            config_overrides: 設定オーバーライド

        Returns:
            作成されたBacktestJob
        """
        job_id = str(uuid.uuid4())[:8]
        run_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"

        job = BacktestJob(
            job_id=job_id,
            run_id=run_id,
            universe_file=universe_file,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            config_overrides=config_overrides or {},
        )

        with self._lock:
            self._jobs[job_id] = job

        logger.info(f"Job created: {job_id} (run_id={run_id})")
        return job

    def start_job(self, job_id: str) -> BacktestJob:
        """
        ジョブを開始

        Args:
            job_id: ジョブID

        Returns:
            更新されたBacktestJob

        Raises:
            ValueError: ジョブが存在しない、または既に実行中の場合
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job not found: {job_id}")

            job = self._jobs[job_id]

            if job.status != JobStatus.PENDING:
                raise ValueError(f"Job is not pending: {job_id} (status={job.status})")

            # 同時実行数チェック
            running_count = sum(
                1 for j in self._jobs.values() if j.status == JobStatus.RUNNING
            )
            if running_count >= self.max_concurrent_jobs:
                raise ValueError(
                    f"Max concurrent jobs reached: {running_count}/{self.max_concurrent_jobs}"
                )

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            # 子プロセスを起動
            process = multiprocessing.Process(
                target=_run_backtest_process,
                args=(
                    job.to_dict(),
                    str(self.project_root),
                    str(self.results_dir),
                ),
            )
            process.start()
            self._processes[job_id] = process

            logger.info(f"Job started: {job_id} (pid={process.pid})")
            return job

    def stop_job(self, job_id: str) -> BacktestJob:
        """
        ジョブを停止

        Args:
            job_id: ジョブID

        Returns:
            更新されたBacktestJob
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job not found: {job_id}")

            job = self._jobs[job_id]

            if job.status != JobStatus.RUNNING:
                raise ValueError(f"Job is not running: {job_id} (status={job.status})")

            # プロセスを終了
            if job_id in self._processes:
                process = self._processes[job_id]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                del self._processes[job_id]

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

            logger.info(f"Job stopped: {job_id}")
            return job

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        """
        ジョブ情報を取得

        Args:
            job_id: ジョブID

        Returns:
            BacktestJobまたはNone
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                self._update_job_status(job)
            return job

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[BacktestJob]:
        """
        ジョブ一覧を取得

        Args:
            status: フィルタするステータス
            limit: 最大取得件数

        Returns:
            BacktestJobのリスト
        """
        with self._lock:
            # ステータス更新
            for job in self._jobs.values():
                self._update_job_status(job)

            jobs = list(self._jobs.values())

            if status:
                jobs = [j for j in jobs if j.status == status]

            # 作成日時でソート（新しい順）
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            return jobs[:limit]

    def _update_job_status(self, job: BacktestJob) -> None:
        """
        ジョブステータスを更新（プロセス状態をチェック）

        Args:
            job: 更新するジョブ
        """
        if job.status != JobStatus.RUNNING:
            return

        job_id = job.job_id

        # プロセスの状態チェック
        if job_id in self._processes:
            process = self._processes[job_id]
            if not process.is_alive():
                # 結果ファイルをチェック
                progress_dir = self.results_dir / ".progress"
                result_file = progress_dir / f"{job.run_id}.result"
                error_file = progress_dir / f"{job.run_id}.error"

                if result_file.exists():
                    job.status = JobStatus.COMPLETED
                    job.archive_id = result_file.read_text().strip()
                elif error_file.exists():
                    job.status = JobStatus.FAILED
                    job.error_message = error_file.read_text().strip()
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = "Process terminated unexpectedly"

                job.completed_at = datetime.now()
                del self._processes[job_id]

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        古いジョブを削除

        Args:
            max_age_hours: 最大保持時間

        Returns:
            削除したジョブ数
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        with self._lock:
            to_delete = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                and job.created_at < cutoff
            ]

            for job_id in to_delete:
                del self._jobs[job_id]
                deleted += 1

        if deleted:
            logger.info(f"Cleaned up {deleted} old jobs")

        return deleted


# グローバルJobManagerインスタンス（シングルトン）
_job_manager: Optional[JobManager] = None


def get_job_manager(
    project_root: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> JobManager:
    """
    JobManagerインスタンスを取得（シングルトン）

    Args:
        project_root: プロジェクトルートパス
        results_dir: 結果保存ディレクトリ

    Returns:
        JobManagerインスタンス
    """
    global _job_manager

    if _job_manager is None:
        _job_manager = JobManager(
            project_root=project_root,
            results_dir=results_dir,
        )

    return _job_manager
