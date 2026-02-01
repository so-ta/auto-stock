"""
Progress Tracker for Backtest Monitoring

バックテストの進捗、キャッシュヒット率、推定完了時間を追跡するためのクラス。
ファイルベースで進捗情報を共有し、Web ビューアからポーリングで読み取る。

Usage:
    from src.utils.progress_tracker import ProgressTracker

    tracker = ProgressTracker(run_id="bt_20260131_143022")
    tracker.set_phase("data_loading")
    tracker.set_phase("backtest", total_steps=180)

    for i, date in enumerate(rebalance_dates):
        tracker.update_rebalance(i, date)
        tracker.update_cache_stats({"signal": {"hits": 100, "misses": 10}})

    tracker.complete()
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimeseriesSnapshot:
    """リバランス時点のスナップショット（エクイティカーブ用）"""
    date: str
    portfolio_value: float
    cumulative_return: float
    weights_top5: Dict[str, float] = field(default_factory=dict)  # 上位5銘柄のみ（軽量化）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "portfolio_value": self.portfolio_value,
            "cumulative_return": self.cumulative_return,
            "weights_top5": self.weights_top5,
        }


@dataclass
class CacheStats:
    """単一キャッシュの統計情報"""
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """キャッシュヒット率を計算"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class PhaseProgress:
    """工程別進捗情報"""
    current: int = 0
    total: int = 0
    status: str = "pending"  # "pending", "running", "completed"

    @property
    def percentage(self) -> float:
        """進捗パーセンテージを計算"""
        return (self.current / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "total": self.total,
            "percentage": round(self.percentage, 1),
            "status": self.status,
        }


@dataclass
class ProgressData:
    """進捗データ構造"""
    run_id: str
    status: str = "running"  # "initializing", "running", "completed", "failed"
    phase: str = "initializing"  # "initializing", "data_loading", "signal_precompute", "backtest", "saving"

    # リバランス進捗
    current_rebalance: int = 0
    total_rebalances: int = 0
    current_date: str = ""

    # キャッシュ統計
    cache_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # タイミング
    start_time: str = ""
    last_update: str = ""
    estimated_completion: Optional[str] = None

    # エラー・警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # INFOログ（最新100件のみ保持）
    info_logs: List[Dict[str, Any]] = field(default_factory=list)

    # 追加メタデータ
    universe_size: int = 0
    frequency: str = ""

    # 工程別進捗（新規追加）
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換（JSON シリアライズ用）"""
        return asdict(self)


class ProgressTracker:
    """
    バックテスト進捗追跡クラス

    ファイルベースで進捗を共有し、Web ビューアからポーリングで読み取る。
    アトミックな書き込みを行い、部分的なファイルを読み取るリスクを軽減。
    """

    DEFAULT_PROGRESS_DIR = "results/.progress"

    def __init__(
        self,
        run_id: Optional[str] = None,
        progress_dir: Optional[Path] = None,
        universe_size: int = 0,
        frequency: str = "",
    ):
        """
        初期化

        Args:
            run_id: 実行 ID（省略時は自動生成）
            progress_dir: 進捗ファイル保存ディレクトリ
            universe_size: ユニバースサイズ
            frequency: リバランス頻度
        """
        self.run_id = run_id or self._generate_run_id()
        self.progress_dir = Path(progress_dir or self.DEFAULT_PROGRESS_DIR)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self._progress = ProgressData(
            run_id=self.run_id,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            universe_size=universe_size,
            frequency=frequency,
        )

        # ETA 計算用
        self._rebalance_times: List[float] = []
        self._last_rebalance_time: Optional[float] = None
        self._phase_start_time: Optional[float] = None

        # 工程別進捗トラッキング用（全フェーズをpending状態で初期化）
        self._phase_progress: Dict[str, PhaseProgress] = {
            "initializing": PhaseProgress(total=1, status="pending"),
            "data_loading": PhaseProgress(total=0, status="pending"),
            "signal_precompute": PhaseProgress(total=0, status="pending"),
            "backtest": PhaseProgress(total=0, status="pending"),
            "saving": PhaseProgress(total=0, status="pending"),
        }

        # === 時系列スナップショット機能（リアルタイムエクイティカーブ用） ===
        self._timeseries_buffer: List[TimeseriesSnapshot] = []
        self._timeseries_file: Path = self.progress_dir / f"{self.run_id}_timeseries.jsonl"
        self._flush_interval: int = 5  # 5リバランスごとにファイル書き込み
        self._initial_capital: float = 0.0  # 初期資本（累積リターン計算用）

        # 初期状態を保存
        self._write_progress()
        logger.info(f"ProgressTracker initialized: {self.progress_file}")

    @property
    def progress_file(self) -> Path:
        """進捗ファイルパス"""
        return self.progress_dir / f"{self.run_id}.json"

    @staticmethod
    def _generate_run_id() -> str:
        """実行 ID を生成"""
        return f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def set_phase(self, phase: str, total_steps: int = 0) -> None:
        """
        フェーズを設定

        Args:
            phase: フェーズ名 ("data_loading", "signal_precompute", "backtest", "saving")
            total_steps: 総ステップ数（backtest フェーズで使用）
        """
        # 前のフェーズを完了状態に
        if self._progress.phase in self._phase_progress:
            self._phase_progress[self._progress.phase].status = "completed"

        self._progress.phase = phase
        self._progress.status = "running"
        self._phase_start_time = time.time()

        if total_steps > 0:
            self._progress.total_rebalances = total_steps
            self._progress.current_rebalance = 0

        # 新しいフェーズを running 状態で初期化
        if phase not in self._phase_progress:
            self._phase_progress[phase] = PhaseProgress(
                total=total_steps,
                status="running"
            )
        else:
            self._phase_progress[phase].status = "running"
            if total_steps > 0:
                self._phase_progress[phase].total = total_steps

        self._sync_phases()
        self._write_progress()
        logger.debug(f"Phase set: {phase} (total_steps={total_steps})")

    def set_phase_total(self, phase: str, total: int) -> None:
        """
        工程の全件数を設定

        Args:
            phase: フェーズ名
            total: 全件数
        """
        if phase not in self._phase_progress:
            self._phase_progress[phase] = PhaseProgress(total=total, status="pending")
        else:
            self._phase_progress[phase].total = total

        self._sync_phases()
        self._write_progress()
        logger.debug(f"Phase total set: {phase} = {total}")

    def increment_phase(self, phase: str, amount: int = 1) -> None:
        """
        工程の進捗件数をインクリメント

        Args:
            phase: フェーズ名
            amount: 増加量（デフォルト1）
        """
        if phase not in self._phase_progress:
            self._phase_progress[phase] = PhaseProgress(current=amount, status="running")
        else:
            self._phase_progress[phase].current += amount
            if self._phase_progress[phase].status == "pending":
                self._phase_progress[phase].status = "running"

        self._sync_phases()
        self._write_progress()

    def complete_phase(self, phase: str) -> None:
        """
        工程を完了としてマーク

        Args:
            phase: フェーズ名
        """
        if phase in self._phase_progress:
            pp = self._phase_progress[phase]
            pp.status = "completed"
            pp.current = pp.total  # 完了時は current = total

        self._sync_phases()
        self._write_progress()
        logger.debug(f"Phase completed: {phase}")

    def get_phase_progress(self, phase: str) -> Optional[PhaseProgress]:
        """
        工程の進捗を取得

        Args:
            phase: フェーズ名

        Returns:
            PhaseProgress または None
        """
        return self._phase_progress.get(phase)

    def _sync_phases(self) -> None:
        """内部の _phase_progress を _progress.phases に同期"""
        self._progress.phases = {
            k: v.to_dict() for k, v in self._phase_progress.items()
        }

    def update_rebalance(self, index: int, date: datetime) -> None:
        """
        リバランス進捗を更新

        Args:
            index: 現在のリバランスインデックス（0 始まり）
            date: 現在の日付
        """
        current_time = time.time()

        # リバランス時間を記録（ETA 計算用）
        if self._last_rebalance_time is not None:
            elapsed = current_time - self._last_rebalance_time
            self._rebalance_times.append(elapsed)
            # 直近 20 回分のみ保持
            if len(self._rebalance_times) > 20:
                self._rebalance_times.pop(0)

        self._last_rebalance_time = current_time

        self._progress.current_rebalance = index + 1  # 1 始まりに変換
        self._progress.current_date = date.strftime("%Y-%m-%d")

        # ETA を計算
        eta = self._estimate_completion()
        if eta:
            self._progress.estimated_completion = eta.isoformat()

        self._write_progress()

    def set_initial_capital(self, capital: float) -> None:
        """
        初期資本を設定（累積リターン計算用）

        Args:
            capital: 初期資本
        """
        self._initial_capital = capital

    def update_rebalance_with_snapshot(
        self,
        index: int,
        date: datetime,
        portfolio_value: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        拡張版リバランス更新（エクイティデータ付き）

        リバランス進捗を更新し、同時にエクイティカーブ用のスナップショットを保存。
        フロントエンドのリアルタイムエクイティカーブ表示に使用。

        Args:
            index: 現在のリバランスインデックス（0 始まり）
            date: 現在の日付
            portfolio_value: 現在のポートフォリオ価値
            weights: 現在のウェイト辞書（省略可）
        """
        # 通常のリバランス更新
        self.update_rebalance(index, date)

        # 累積リターンを計算
        if self._initial_capital > 0:
            cumulative_return = (portfolio_value / self._initial_capital) - 1.0
        else:
            cumulative_return = 0.0

        # 上位5銘柄のウェイトを抽出
        weights_top5: Dict[str, float] = {}
        if weights:
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            weights_top5 = dict(sorted_weights[:5])

        # スナップショットを作成
        snapshot = TimeseriesSnapshot(
            date=date.strftime("%Y-%m-%d"),
            portfolio_value=portfolio_value,
            cumulative_return=cumulative_return,
            weights_top5=weights_top5,
        )

        # バッファに追加
        self._timeseries_buffer.append(snapshot)

        # flush_intervalに達したらファイルに追記
        if len(self._timeseries_buffer) >= self._flush_interval:
            self._flush_timeseries()

    def _flush_timeseries(self) -> None:
        """
        時系列バッファをファイルに追記してクリア

        JSONL形式（1行1JSON）で追記。効率的な追記とストリーミング読み込みに対応。
        """
        if not self._timeseries_buffer:
            return

        try:
            with open(self._timeseries_file, "a") as f:
                for snapshot in self._timeseries_buffer:
                    f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")

            logger.debug(
                f"Flushed {len(self._timeseries_buffer)} snapshots to {self._timeseries_file}"
            )
            self._timeseries_buffer.clear()

        except Exception as e:
            logger.warning(f"Failed to flush timeseries: {e}")

    def get_timeseries(self) -> List[Dict[str, Any]]:
        """
        ファイル + バッファから全時系列を取得

        Returns:
            時系列スナップショットのリスト
        """
        result: List[Dict[str, Any]] = []

        # ファイルから読み込み
        if self._timeseries_file.exists():
            try:
                with open(self._timeseries_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to read timeseries file: {e}")

        # バッファを追加
        for snapshot in self._timeseries_buffer:
            result.append(snapshot.to_dict())

        return result

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        最新のスナップショットを取得

        Returns:
            最新のスナップショット辞書、またはNone
        """
        # バッファに何かあればその最後を返す
        if self._timeseries_buffer:
            return self._timeseries_buffer[-1].to_dict()

        # ファイルから最後の行を読む
        if self._timeseries_file.exists():
            try:
                with open(self._timeseries_file, "r") as f:
                    last_line = None
                    for line in f:
                        if line.strip():
                            last_line = line.strip()
                    if last_line:
                        return json.loads(last_line)
            except Exception as e:
                logger.warning(f"Failed to read latest snapshot: {e}")

        return None

    @classmethod
    def load_timeseries(cls, progress_dir: Path, run_id: str) -> List[Dict[str, Any]]:
        """
        外部から時系列データを読み込む（クラスメソッド）

        Args:
            progress_dir: 進捗ディレクトリ
            run_id: 実行ID

        Returns:
            時系列スナップショットのリスト
        """
        timeseries_file = progress_dir / f"{run_id}_timeseries.jsonl"
        result: List[Dict[str, Any]] = []

        if timeseries_file.exists():
            try:
                with open(timeseries_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to load timeseries for {run_id}: {e}")

        return result

    def update_cache_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """
        キャッシュ統計を更新

        Args:
            stats: キャッシュ統計辞書
                例: {"signal": {"hits": 100, "misses": 10}, ...}
        """
        for cache_name, cache_data in stats.items():
            if isinstance(cache_data, dict):
                hits = cache_data.get("hits", 0)
                misses = cache_data.get("misses", 0)
                total = hits + misses
                hit_rate = hits / total if total > 0 else 0.0

                self._progress.cache_stats[cache_name] = {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": round(hit_rate, 4),
                }

        # 書き込み頻度を制限（最大 2 秒に 1 回）
        # update_rebalance で既に書き込んでいる場合はスキップ
        # ただし、初回は必ず書き込む
        if not self._progress.cache_stats or len(self._progress.cache_stats) == len(stats):
            self._write_progress()

    def add_warning(self, message: str) -> None:
        """警告を追加"""
        self._progress.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self._write_progress()
        logger.warning(f"Progress warning: {message}")

    def add_error(self, message: str) -> None:
        """エラーを追加"""
        self._progress.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self._write_progress()
        logger.error(f"Progress error: {message}")

    def add_info(self, message: str, component: str = "", **details: Any) -> None:
        """
        INFOログを追加（最新100件のみ保持）

        Args:
            message: ログメッセージ
            component: ソースコンポーネント名
            **details: 追加のキーバリューペア
        """
        log_entry = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "message": message,
            "component": component,
            **details
        }
        self._progress.info_logs.append(log_entry)

        # 最新100件のみ保持（メモリ節約）
        if len(self._progress.info_logs) > 100:
            self._progress.info_logs = self._progress.info_logs[-100:]

        self._write_progress()

    def complete(self) -> None:
        """正常完了としてマーク"""
        # 残りの時系列バッファをフラッシュ
        self._flush_timeseries()

        self._progress.status = "completed"
        self._progress.phase = "completed"
        self._progress.estimated_completion = None
        self._write_progress()
        logger.info(f"Progress completed: {self.run_id}")

    def fail(self, error: str) -> None:
        """
        失敗としてマーク

        Args:
            error: エラーメッセージ
        """
        # 残りの時系列バッファをフラッシュ（失敗時も保存）
        self._flush_timeseries()

        self._progress.status = "failed"
        self.add_error(error)
        self._write_progress()
        logger.error(f"Progress failed: {self.run_id} - {error}")

    def _estimate_completion(self) -> Optional[datetime]:
        """
        推定完了時間を計算（指数移動平均）

        Returns:
            推定完了時刻、または計算不可の場合は None
        """
        if len(self._rebalance_times) < 3:
            return None

        # EMA (alpha=0.3) で平均リバランス時間を推定
        alpha = 0.3
        ema = self._rebalance_times[0]
        for t in self._rebalance_times[1:]:
            ema = alpha * t + (1 - alpha) * ema

        remaining = self._progress.total_rebalances - self._progress.current_rebalance
        if remaining <= 0:
            return None

        # 10% バッファを追加
        remaining_seconds = ema * remaining * 1.1

        return datetime.now() + timedelta(seconds=remaining_seconds)

    def _write_progress(self) -> None:
        """
        進捗をファイルに書き込み（アトミック）

        一時ファイルに書き込んでからリネームすることで、
        部分的なファイルを読み取るリスクを軽減。
        """
        self._progress.last_update = datetime.now().isoformat()

        try:
            # 一時ファイルに書き込み
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                dir=str(self.progress_dir),
            )

            try:
                with open(temp_fd, "w") as f:
                    json.dump(self._progress.to_dict(), f, indent=2, ensure_ascii=False)

                # アトミックにリネーム
                Path(temp_path).replace(self.progress_file)

            except Exception:
                # リネーム失敗時は一時ファイルを削除
                Path(temp_path).unlink(missing_ok=True)
                raise

        except Exception as e:
            logger.warning(f"Failed to write progress file: {e}")

    @classmethod
    def load_progress(cls, progress_file: Path) -> Optional[ProgressData]:
        """
        進捗ファイルを読み込み

        Args:
            progress_file: 進捗ファイルパス

        Returns:
            ProgressData または読み込み失敗時は None
        """
        # ファイルが存在しない場合は None を返す（ログなし、初期化中は正常）
        if not progress_file.exists():
            return None

        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
            return ProgressData(**data)
        except Exception as e:
            logger.warning(f"Failed to load progress file {progress_file}: {e}")
            return None

    @classmethod
    def list_active_progress(
        cls,
        progress_dir: Optional[Path] = None,
    ) -> List[ProgressData]:
        """
        アクティブな進捗一覧を取得

        Args:
            progress_dir: 進捗ディレクトリ

        Returns:
            アクティブな ProgressData リスト
        """
        dir_path = Path(progress_dir or cls.DEFAULT_PROGRESS_DIR)
        if not dir_path.exists():
            return []

        results = []
        for json_file in dir_path.glob("*.json"):
            progress = cls.load_progress(json_file)
            if progress:
                results.append(progress)

        # 最終更新時刻でソート（新しい順）
        results.sort(key=lambda p: p.last_update, reverse=True)
        return results

    @classmethod
    def cleanup_old_progress(
        cls,
        progress_dir: Optional[Path] = None,
        max_age_hours: int = 24,
    ) -> int:
        """
        古い進捗ファイルをクリーンアップ

        Args:
            progress_dir: 進捗ディレクトリ
            max_age_hours: 保持する最大時間

        Returns:
            削除したファイル数
        """
        dir_path = Path(progress_dir or cls.DEFAULT_PROGRESS_DIR)
        if not dir_path.exists():
            return 0

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        for json_file in dir_path.glob("*.json"):
            try:
                progress = cls.load_progress(json_file)
                if progress is None:
                    continue

                # 完了・失敗かつ古いファイルを削除
                if progress.status in ("completed", "failed"):
                    last_update = datetime.fromisoformat(progress.last_update)
                    if last_update < cutoff:
                        json_file.unlink()
                        deleted += 1
                        logger.debug(f"Deleted old progress file: {json_file}")

                        # 対応する時系列ファイルも削除
                        run_id = progress.run_id
                        timeseries_file = dir_path / f"{run_id}_timeseries.jsonl"
                        if timeseries_file.exists():
                            timeseries_file.unlink()
                            logger.debug(f"Deleted old timeseries file: {timeseries_file}")

            except Exception as e:
                logger.warning(f"Failed to cleanup {json_file}: {e}")

        if deleted:
            logger.info(f"Cleaned up {deleted} old progress files")

        return deleted
