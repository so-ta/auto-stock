"""
Backtest View Service - 統一ビューのためのサービス層

実行中バックテストとアーカイブ済みバックテストを統一的に扱うサービス。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


class BacktestState(Enum):
    """バックテストの状態"""
    INITIALIZING = "initializing"  # 初期化中（progressファイルなし or status=initializing）
    RUNNING = "running"            # 実行中
    COMPLETED = "completed"        # 正常完了
    FAILED = "failed"              # 失敗
    ARCHIVED = "archived"          # アーカイブ済み（progressなし、archiveあり）
    NOT_FOUND = "not_found"        # 見つからない


@dataclass
class BacktestViewData:
    """統一ビュー用データ構造"""
    id: str
    state: BacktestState

    # 進捗情報（実行中の場合）
    progress: Optional[Dict[str, Any]] = None

    # アーカイブ情報（完了後）
    archive_id: Optional[str] = None

    # 共通メトリクス
    metrics: Optional[Dict[str, Any]] = None
    trading_stats: Optional[Dict[str, Any]] = None

    # 設定情報
    config_snapshot: Optional[Dict[str, Any]] = None

    # 実行情報
    execution_info: Optional[Dict[str, Any]] = None

    # 時系列データ参照
    has_timeseries: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "progress": self.progress,
            "archive_id": self.archive_id,
            "metrics": self.metrics,
            "trading_stats": self.trading_stats,
            "config_snapshot": self.config_snapshot,
            "execution_info": self.execution_info,
            "has_timeseries": self.has_timeseries,
        }


class BacktestViewService:
    """
    バックテスト統一ビューサービス

    実行中バックテスト（.progress）とアーカイブ（results/）を
    統一的に扱うためのサービス層。
    """

    def __init__(
        self,
        results_dir: Path,
        progress_dir: Optional[Path] = None,
    ):
        """
        初期化

        Args:
            results_dir: 結果ディレクトリ（results/）
            progress_dir: 進捗ディレクトリ（results/.progress/）
        """
        self.results_dir = Path(results_dir)
        self.progress_dir = progress_dir or (self.results_dir / ".progress")

    def get_state(self, id: str, allow_initializing: bool = True) -> BacktestState:
        """
        バックテストの状態を判定

        Args:
            id: バックテストID（run_id or archive_id）
            allow_initializing: IDが bt_ で始まる場合、ファイルがなくても
                               INITIALIZING を返すか（デフォルト: True）

        Returns:
            BacktestState enum
        """
        # 1. .progress/{id}.json が存在するか確認
        progress_file = self.progress_dir / f"{id}.json"
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)
                status = data.get("status", "running")

                if status == "initializing":
                    return BacktestState.INITIALIZING
                elif status == "running":
                    return BacktestState.RUNNING
                elif status == "completed":
                    return BacktestState.COMPLETED
                elif status == "failed":
                    return BacktestState.FAILED
                else:
                    return BacktestState.RUNNING
            except Exception as e:
                logger.warning(f"Failed to read progress file {progress_file}: {e}")
                return BacktestState.INITIALIZING

        # 2. results/{id}/metadata.json が存在するか確認
        archive_dir = self.results_dir / id
        metadata_file = archive_dir / "metadata.json"
        if metadata_file.exists():
            return BacktestState.ARCHIVED

        # 3. どちらも存在しない場合
        # IDが bt_ で始まる場合はバックテスト実行中の可能性があるため
        # progressファイルがまだ作成されていない初期化中と判断
        if allow_initializing and id.startswith("bt_"):
            return BacktestState.INITIALIZING

        return BacktestState.NOT_FOUND

    def get_view_data(self, id: str) -> Optional[BacktestViewData]:
        """
        統一ビュー用データを取得

        Args:
            id: バックテストID

        Returns:
            BacktestViewData or None
        """
        state = self.get_state(id)

        if state == BacktestState.NOT_FOUND:
            return None

        view_data = BacktestViewData(id=id, state=state)

        # 進捗ファイルから情報を取得（実行中の場合）
        if state in (BacktestState.INITIALIZING, BacktestState.RUNNING,
                     BacktestState.COMPLETED, BacktestState.FAILED):
            progress_file = self.progress_dir / f"{id}.json"
            if progress_file.exists():
                try:
                    with open(progress_file, "r") as f:
                        view_data.progress = json.load(f)
                except Exception:
                    pass

            # INITIALIZING/RUNNING でファイルがない場合はデフォルト進捗を生成
            if view_data.progress is None and state in (
                BacktestState.INITIALIZING, BacktestState.RUNNING
            ):
                view_data.progress = {
                    "run_id": id,
                    "status": "initializing",
                    "phase": "initializing",
                    "start_time": datetime.now().isoformat(),
                    "phases": {
                        "initializing": {"status": "running", "current": 0, "total": 1},
                        "data_loading": {"status": "pending", "current": 0, "total": 0},
                        "signal_precompute": {"status": "pending", "current": 0, "total": 0},
                        "backtest": {"status": "pending", "current": 0, "total": 0},
                        "saving": {"status": "pending", "current": 0, "total": 0},
                    },
                    "current_rebalance": 0,
                    "total_rebalances": 0,
                    "cache_stats": {},
                }

            # 完了時は.resultファイルからarchive_idを取得
            if state == BacktestState.COMPLETED:
                result_file = self.progress_dir / f"{id}.result"
                if result_file.exists():
                    try:
                        with open(result_file, "r") as f:
                            content = f.read().strip()
                            # JSONフォーマットか単純テキストかを判定
                            if content.startswith("{"):
                                result_data = json.loads(content)
                                view_data.archive_id = result_data.get("archive_id")
                            else:
                                # 単純テキスト（archive_idのみ）
                                view_data.archive_id = content
                    except Exception:
                        pass

            # 時系列ファイル存在確認
            timeseries_file = self.progress_dir / f"{id}_timeseries.jsonl"
            view_data.has_timeseries = timeseries_file.exists()

        # アーカイブから情報を取得（アーカイブの場合）
        if state == BacktestState.ARCHIVED or view_data.archive_id:
            archive_id = view_data.archive_id or id
            self._load_archive_data(archive_id, view_data)

        return view_data

    def _load_archive_data(self, archive_id: str, view_data: BacktestViewData) -> None:
        """アーカイブからデータを読み込み"""
        archive_dir = self.results_dir / archive_id
        metadata_file = archive_dir / "metadata.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            view_data.archive_id = archive_id
            view_data.metrics = metadata.get("metrics", {})
            view_data.trading_stats = metadata.get("trading_stats", {})
            view_data.execution_info = metadata.get("execution_info")

            # config_snapshot.yaml から設定を読み込み
            config_file = archive_dir / "config_snapshot.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, "r") as f:
                    view_data.config_snapshot = yaml.safe_load(f)

            # 時系列ファイル存在確認
            timeseries_file = archive_dir / "timeseries.parquet"
            view_data.has_timeseries = timeseries_file.exists()

        except Exception as e:
            logger.warning(f"Failed to load archive data for {archive_id}: {e}")

    def get_summary(self, id: str) -> Optional[Dict[str, Any]]:
        """
        概要情報を取得（メトリクス + 進捗）

        Args:
            id: バックテストID

        Returns:
            概要情報辞書
        """
        view_data = self.get_view_data(id)
        if not view_data:
            return None

        summary: Dict[str, Any] = {
            "id": id,
            "state": view_data.state.value,
        }

        # 進捗情報
        if view_data.progress:
            summary["progress"] = {
                "phase": view_data.progress.get("phase"),
                "current_rebalance": view_data.progress.get("current_rebalance", 0),
                "total_rebalances": view_data.progress.get("total_rebalances", 0),
                "current_date": view_data.progress.get("current_date"),
                "estimated_completion": view_data.progress.get("estimated_completion"),
                "cache_stats": view_data.progress.get("cache_stats", {}),
                "phases": view_data.progress.get("phases", {}),
            }

        # メトリクス
        if view_data.metrics:
            summary["metrics"] = view_data.metrics

        # 取引統計
        if view_data.trading_stats:
            summary["trading_stats"] = view_data.trading_stats

        # アーカイブID
        if view_data.archive_id:
            summary["archive_id"] = view_data.archive_id

        # 実行情報
        if view_data.execution_info:
            summary["execution_info"] = view_data.execution_info

        return summary

    def get_timeseries(self, id: str) -> Dict[str, Any]:
        """
        時系列データを取得（実行中 or アーカイブ）

        Args:
            id: バックテストID

        Returns:
            時系列データ辞書
        """
        state = self.get_state(id)

        # 実行中の場合は .progress から JSONL を読み込み
        if state in (BacktestState.INITIALIZING, BacktestState.RUNNING,
                     BacktestState.COMPLETED, BacktestState.FAILED):
            timeseries_file = self.progress_dir / f"{id}_timeseries.jsonl"
            if timeseries_file.exists():
                timeseries = []
                try:
                    with open(timeseries_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                timeseries.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Failed to read timeseries file: {e}")

                return {
                    "source": "progress",
                    "timeseries": timeseries,
                    "count": len(timeseries),
                }

        # アーカイブの場合は parquet から読み込み
        view_data = self.get_view_data(id)
        if view_data and view_data.archive_id:
            archive_id = view_data.archive_id
        elif state == BacktestState.ARCHIVED:
            archive_id = id
        else:
            return {"source": "none", "timeseries": [], "count": 0}

        timeseries_file = self.results_dir / archive_id / "timeseries.parquet"
        if timeseries_file.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(timeseries_file)

                # timestamp をインデックスに
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    dates = df["timestamp"].dt.strftime("%Y-%m-%d").tolist()
                else:
                    dates = df.index.strftime("%Y-%m-%d").tolist()

                return {
                    "source": "archive",
                    "dates": dates,
                    "portfolio_value": df["portfolio_value"].tolist(),
                    "daily_return": df["daily_return"].tolist() if "daily_return" in df else [],
                    "cumulative_return": df["cumulative_return"].tolist() if "cumulative_return" in df else [],
                    "drawdown": df["drawdown"].tolist() if "drawdown" in df else [],
                    "count": len(df),
                }
            except Exception as e:
                logger.warning(f"Failed to read parquet timeseries: {e}")

        return {"source": "none", "timeseries": [], "count": 0}

    def find_archive_id(self, id: str) -> Optional[str]:
        """
        IDに対応するarchive_idを検索

        Args:
            id: バックテストID（run_id or archive_id）

        Returns:
            archive_id or None
        """
        # アーカイブディレクトリが直接存在する場合
        archive_dir = self.results_dir / id
        if (archive_dir / "metadata.json").exists():
            return id

        # 完了済みの場合は .result ファイルから取得
        result_file = self.progress_dir / f"{id}.result"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    content = f.read().strip()
                    # JSONフォーマットか単純テキストかを判定
                    if content.startswith("{"):
                        result_data = json.loads(content)
                        return result_data.get("archive_id")
                    else:
                        # 単純テキスト（archive_idのみ）
                        return content
            except Exception:
                pass

        return None


# シングルトンインスタンスを管理
_service_instance: Optional[BacktestViewService] = None


def get_backtest_view_service(
    results_dir: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
) -> BacktestViewService:
    """
    BacktestViewService のシングルトンインスタンスを取得

    Args:
        results_dir: 結果ディレクトリ
        progress_dir: 進捗ディレクトリ

    Returns:
        BacktestViewService インスタンス
    """
    global _service_instance

    if _service_instance is None or results_dir is not None:
        if results_dir is None:
            results_dir = Path("results")
        _service_instance = BacktestViewService(
            results_dir=results_dir,
            progress_dir=progress_dir,
        )

    return _service_instance
