"""
Backtest Result Store - バックテスト結果の永続保存・読み込み

バックテスト結果をファイルシステムに保存し、後から読み込み・検索・比較できるようにする。

ディレクトリ構造:
    results/
    ├── index.json                    # 全アーカイブのインデックス
    ├── bt_20260131_123456_abc123/    # 個別アーカイブ
    │   ├── metadata.json             # メタデータ・メトリクス
    │   ├── config_snapshot.yaml      # 設定のスナップショット
    │   ├── timeseries.parquet        # 日次時系列データ
    │   └── rebalances.parquet        # リバランス詳細データ
    └── comparisons/                  # 比較結果
        └── compare_abc123_def456.html
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
import yaml

from src.backtest.base import UnifiedBacktestResult, RebalanceRecord
from src.analysis.backtest_archive import (
    BacktestArchive,
    generate_archive_id,
    generate_config_hash,
    generate_universe_hash,
    get_code_version,
)

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend
    from src.utils.pipeline_log_collector import PipelineLogCollector

logger = logging.getLogger(__name__)


class BacktestResultStore:
    """
    バックテスト結果の保存・読み込み・検索を行うクラス

    Attributes:
        base_dir: 結果保存ディレクトリ
        _index_path: インデックスファイルパス
        _backend: オプショナルなStorageBackend（S3サポート用）
    """

    def __init__(
        self,
        base_dir: str = "results",
        storage_backend: Optional["StorageBackend"] = None,
    ) -> None:
        """
        初期化

        Args:
            base_dir: 結果保存ディレクトリ
            storage_backend: オプショナルなStorageBackend（S3サポート用）
        """
        self.base_dir = Path(base_dir)
        self._index_path = self.base_dir / "index.json"
        self._comparisons_dir = self.base_dir / "comparisons"
        self._backend = storage_backend

        # ディレクトリ作成（ローカルモードの場合のみ）
        if self._backend is None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._comparisons_dir.mkdir(parents=True, exist_ok=True)

        # インデックスが存在しない場合は初期化
        if not self._index_exists():
            self._save_index([])

    def _index_exists(self) -> bool:
        """インデックスファイルが存在するかチェック"""
        if self._backend:
            return self._backend.exists("index.json")
        return self._index_path.exists()

    def save(
        self,
        result: UnifiedBacktestResult,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        universe: Optional[List[str]] = None,
        execution_info: Optional[Dict[str, Any]] = None,
        log_collector: Optional["PipelineLogCollector"] = None,
    ) -> str:
        """
        バックテスト結果を保存

        Args:
            result: UnifiedBacktestResult インスタンス
            name: 人間可読な名前
            description: 説明文
            tags: タグリスト
            universe: 銘柄リスト（resultから取得できない場合に指定）
            execution_info: 実行情報メタデータ
            log_collector: パイプラインログコレクター（logs.jsonl 保存用）

        Returns:
            archive_id: 保存されたアーカイブのID
        """
        tags = tags or []

        # ユニバース取得
        if universe is None:
            # リバランス記録から銘柄を抽出
            symbols = set()
            for rebalance in result.rebalances:
                symbols.update(rebalance.weights_after.keys())
            universe = sorted(symbols)

        # 設定辞書を取得
        config_dict = result.config.to_dict() if result.config else {}

        # ハッシュ生成
        config_hash = generate_config_hash(config_dict, universe)
        universe_hash = generate_universe_hash(universe)
        code_version = get_code_version()

        # アーカイブID生成
        archive_id = generate_archive_id(config_hash)

        # アーカイブオブジェクト作成
        archive = BacktestArchive(
            archive_id=archive_id,
            created_at=datetime.now(),
            config_hash=config_hash,
            universe_hash=universe_hash,
            code_version=code_version,
            name=name,
            description=description,
            tags=tags,
            config_snapshot=config_dict,
            universe=universe,
            metrics={
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "volatility": result.volatility,
                "calmar_ratio": result.calmar_ratio,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "var_95": result.var_95,
                "expected_shortfall": result.expected_shortfall,
            },
            trading_stats={
                "n_days": result.n_days,
                "n_rebalances": result.n_rebalances,
                "n_trades": result.n_trades,
                "total_turnover": result.total_turnover,
                "avg_turnover": result.avg_turnover,
                "total_transaction_costs": result.total_transaction_costs,
                "initial_value": result.initial_value,
                "final_value": result.final_value,
            },
            start_date=result.start_date,
            end_date=result.end_date,
            engine_name=result.engine_name,
            warnings=result.warnings,
            errors=result.errors,
            execution_info=execution_info,
        )

        # アーカイブディレクトリ作成
        archive_dir = self.base_dir / archive_id
        if self._backend is None:
            archive_dir.mkdir(parents=True, exist_ok=True)

        # メタデータ保存
        if self._backend:
            self._backend.write_json(
                archive.to_metadata_dict(), f"{archive_id}/metadata.json"
            )
        else:
            metadata_path = archive_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(archive.to_metadata_dict(), f, indent=2, ensure_ascii=False)

        # 設定スナップショット保存
        if self._backend:
            self._backend.write_yaml(config_dict, f"{archive_id}/config_snapshot.yaml")
        else:
            config_path = archive_dir / "config_snapshot.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        # 銘柄リスト保存
        if self._backend:
            self._backend.write_json(universe, f"{archive_id}/universe.json")
        else:
            universe_path = archive_dir / "universe.json"
            with open(universe_path, "w", encoding="utf-8") as f:
                json.dump(universe, f, indent=2)

        # 時系列データ保存（Parquet）
        self._save_timeseries(archive_id if self._backend else archive_dir, result)

        # リバランスデータ保存（Parquet）
        self._save_rebalances(archive_id if self._backend else archive_dir, result)

        # ログデータ保存（JSONL）
        if log_collector is not None:
            self._save_logs(archive_id if self._backend else archive_dir, log_collector)

        # インデックス更新
        self._add_to_index(archive)

        logger.info(f"Saved backtest archive: {archive_id}")
        return archive_id

    def _save_timeseries(
        self, archive_dir_or_id: Path | str, result: UnifiedBacktestResult
    ) -> None:
        """時系列データをParquet形式で保存"""
        if len(result.portfolio_values) == 0:
            return

        # DataFrameを構築
        df = pd.DataFrame({
            "portfolio_value": result.portfolio_values,
            "daily_return": result.daily_returns,
        })

        # 累積リターンとドローダウンを計算
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

        running_max = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - running_max) / running_max

        # インデックスをカラムに変換
        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp"})

        # 保存
        if self._backend and isinstance(archive_dir_or_id, str):
            self._backend.write_parquet(
                df, f"{archive_dir_or_id}/timeseries.parquet"
            )
        else:
            archive_dir = archive_dir_or_id
            timeseries_path = archive_dir / "timeseries.parquet"
            df.to_parquet(timeseries_path, index=False)

    def _save_rebalances(
        self, archive_dir_or_id: Path | str, result: UnifiedBacktestResult
    ) -> None:
        """リバランスデータをParquet形式で保存"""
        if len(result.rebalances) == 0:
            return

        rows = []
        for rebalance in result.rebalances:
            rows.append({
                "timestamp": rebalance.date,
                "weights_before": json.dumps(rebalance.weights_before),
                "weights_after": json.dumps(rebalance.weights_after),
                "turnover": rebalance.turnover,
                "transaction_cost": rebalance.transaction_cost,
                "portfolio_value": rebalance.portfolio_value,
            })

        df = pd.DataFrame(rows)

        if self._backend and isinstance(archive_dir_or_id, str):
            self._backend.write_parquet(
                df, f"{archive_dir_or_id}/rebalances.parquet"
            )
        else:
            archive_dir = archive_dir_or_id
            rebalances_path = archive_dir / "rebalances.parquet"
            df.to_parquet(rebalances_path, index=False)

    def _save_logs(
        self, archive_dir_or_id: Path | str, log_collector: "PipelineLogCollector"
    ) -> None:
        """パイプラインログをJSONL形式で保存"""
        if self._backend and isinstance(archive_dir_or_id, str):
            # S3保存: ログを文字列として書き込み
            logs = log_collector.get_logs_as_dicts()
            log_lines = [json.dumps(entry, ensure_ascii=False) for entry in logs]
            log_content = "\n".join(log_lines)
            # StorageBackendにwrite_textがなければwrite_jsonを応用
            logs_path = f"{archive_dir_or_id}/logs.jsonl"
            try:
                # 直接ファイルとして書き込む（StorageBackendはJSONL非対応のため）
                from pathlib import Path as PathLib
                local_path = PathLib(self.base_dir) / archive_dir_or_id / "logs.jsonl"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception as e:
                logger.warning(f"Failed to save logs to S3: {e}")
        else:
            # ローカル保存
            archive_dir = archive_dir_or_id
            logs_path = archive_dir / "logs.jsonl"
            log_collector.save_to_file(logs_path)

    def _add_to_index(self, archive: BacktestArchive) -> None:
        """インデックスにエントリを追加"""
        index = self._load_index()
        entry = archive.to_index_entry()

        # 既存のエントリを更新（同一IDがある場合）
        index = [e for e in index if e["archive_id"] != archive.archive_id]
        index.append(entry)

        # 作成日時でソート（新しい順）
        index.sort(key=lambda x: x["created_at"], reverse=True)

        self._save_index(index)

    def _load_index(self) -> List[Dict[str, Any]]:
        """インデックスを読み込み"""
        if self._backend:
            try:
                return self._backend.read_json("index.json")
            except FileNotFoundError:
                return []
        if not self._index_path.exists():
            return []
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: List[Dict[str, Any]]) -> None:
        """インデックスを保存"""
        if self._backend:
            self._backend.write_json(index, "index.json")
            return
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def load(self, archive_id: str) -> BacktestArchive:
        """
        アーカイブを読み込み

        Args:
            archive_id: アーカイブID

        Returns:
            BacktestArchive インスタンス

        Raises:
            FileNotFoundError: アーカイブが存在しない場合
        """
        if self._backend:
            if not self._backend.exists(f"{archive_id}/metadata.json"):
                raise FileNotFoundError(f"Archive not found: {archive_id}")
            metadata = self._backend.read_json(f"{archive_id}/metadata.json")
            try:
                config_snapshot = self._backend.read_yaml(
                    f"{archive_id}/config_snapshot.yaml"
                )
            except FileNotFoundError:
                config_snapshot = {}
            try:
                universe = self._backend.read_json(f"{archive_id}/universe.json")
            except FileNotFoundError:
                universe = []
            return BacktestArchive.from_metadata_dict(
                metadata, config_snapshot=config_snapshot, universe=universe
            )

        archive_dir = self.base_dir / archive_id
        if not archive_dir.exists():
            raise FileNotFoundError(f"Archive not found: {archive_id}")

        # メタデータ読み込み
        metadata_path = archive_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 設定スナップショット読み込み
        config_path = archive_dir / "config_snapshot.yaml"
        config_snapshot = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_snapshot = yaml.safe_load(f) or {}

        # 銘柄リスト読み込み
        universe_path = archive_dir / "universe.json"
        universe = []
        if universe_path.exists():
            with open(universe_path, "r", encoding="utf-8") as f:
                universe = json.load(f)

        return BacktestArchive.from_metadata_dict(
            metadata, config_snapshot=config_snapshot, universe=universe
        )

    def load_timeseries(self, archive_id: str) -> pd.DataFrame:
        """
        時系列データを読み込み

        Args:
            archive_id: アーカイブID

        Returns:
            時系列DataFrame

        Raises:
            FileNotFoundError: アーカイブまたはファイルが存在しない場合
        """
        if self._backend:
            df = self._backend.read_parquet(f"{archive_id}/timeseries.parquet")
        else:
            archive_dir = self.base_dir / archive_id
            timeseries_path = archive_dir / "timeseries.parquet"

            if not timeseries_path.exists():
                raise FileNotFoundError(f"Timeseries not found: {archive_id}")

            df = pd.read_parquet(timeseries_path)

        # timestampをインデックスに設定
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        return df

    def load_rebalances(self, archive_id: str) -> pd.DataFrame:
        """
        リバランスデータを読み込み

        Args:
            archive_id: アーカイブID

        Returns:
            リバランスDataFrame

        Raises:
            FileNotFoundError: アーカイブまたはファイルが存在しない場合
        """
        if self._backend:
            df = self._backend.read_parquet(f"{archive_id}/rebalances.parquet")
        else:
            archive_dir = self.base_dir / archive_id
            rebalances_path = archive_dir / "rebalances.parquet"

            if not rebalances_path.exists():
                raise FileNotFoundError(f"Rebalances not found: {archive_id}")

            df = pd.read_parquet(rebalances_path)

        # JSON文字列を辞書に変換
        if "weights_before" in df.columns:
            df["weights_before"] = df["weights_before"].apply(json.loads)
        if "weights_after" in df.columns:
            df["weights_after"] = df["weights_after"].apply(json.loads)

        return df

    def list_archives(
        self,
        tags: Optional[List[str]] = None,
        config_hash: Optional[str] = None,
        name_contains: Optional[str] = None,
        start_date_after: Optional[datetime] = None,
        start_date_before: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        条件に合うアーカイブを検索

        Args:
            tags: フィルタするタグ（ANDで絞り込み）
            config_hash: 設定ハッシュでフィルタ
            name_contains: 名前に含まれる文字列
            start_date_after: この日付以降のバックテスト
            start_date_before: この日付以前のバックテスト
            limit: 最大取得件数

        Returns:
            マッチするアーカイブのリスト
        """
        index = self._load_index()
        results = []

        for entry in index:
            # タグフィルタ
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not set(tags).issubset(entry_tags):
                    continue

            # 設定ハッシュフィルタ
            if config_hash and entry.get("config_hash") != config_hash:
                continue

            # 名前フィルタ
            if name_contains and name_contains.lower() not in entry.get("name", "").lower():
                continue

            # 日付フィルタ
            if start_date_after or start_date_before:
                entry_start = entry.get("start_date")
                if entry_start:
                    entry_start_dt = datetime.fromisoformat(entry_start)
                    if start_date_after and entry_start_dt < start_date_after:
                        continue
                    if start_date_before and entry_start_dt > start_date_before:
                        continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def find_by_config_hash(self, config_hash: str) -> List[str]:
        """
        同一設定のアーカイブIDを検索

        Args:
            config_hash: 設定ハッシュ

        Returns:
            マッチするアーカイブIDのリスト
        """
        index = self._load_index()
        return [
            entry["archive_id"]
            for entry in index
            if entry.get("config_hash") == config_hash
        ]

    def find_by_universe_hash(self, universe_hash: str) -> List[str]:
        """
        同一ユニバースのアーカイブIDを検索

        Args:
            universe_hash: ユニバースハッシュ

        Returns:
            マッチするアーカイブIDのリスト
        """
        index = self._load_index()
        return [
            entry["archive_id"]
            for entry in index
            if entry.get("universe_hash") == universe_hash
        ]

    def delete(self, archive_id: str) -> bool:
        """
        アーカイブを削除

        Args:
            archive_id: 削除するアーカイブID

        Returns:
            削除成功したらTrue
        """
        if self._backend:
            if not self._backend.exists(f"{archive_id}/metadata.json"):
                logger.warning(f"Archive not found: {archive_id}")
                return False
            self._backend.delete_directory(archive_id)
        else:
            archive_dir = self.base_dir / archive_id
            if not archive_dir.exists():
                logger.warning(f"Archive not found: {archive_id}")
                return False
            shutil.rmtree(archive_dir)

        # インデックスから削除
        index = self._load_index()
        index = [e for e in index if e["archive_id"] != archive_id]
        self._save_index(index)

        logger.info(f"Deleted archive: {archive_id}")
        return True

    def get_archive_path(self, archive_id: str) -> Path:
        """
        アーカイブディレクトリのパスを取得

        Args:
            archive_id: アーカイブID

        Returns:
            アーカイブディレクトリのPath
        """
        return self.base_dir / archive_id

    def exists(self, archive_id: str) -> bool:
        """
        アーカイブが存在するか確認

        Args:
            archive_id: アーカイブID

        Returns:
            存在すればTrue
        """
        return (self.base_dir / archive_id).exists()

    def get_stats(self) -> Dict[str, Any]:
        """
        ストアの統計情報を取得

        Returns:
            統計情報の辞書
        """
        index = self._load_index()

        if not index:
            return {
                "total_archives": 0,
                "total_size_mb": 0.0,
                "oldest_archive": None,
                "newest_archive": None,
            }

        # サイズ計算
        total_size = 0
        for entry in index:
            archive_dir = self.base_dir / entry["archive_id"]
            if archive_dir.exists():
                for file in archive_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "total_archives": len(index),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_archive": index[-1]["archive_id"] if index else None,
            "newest_archive": index[0]["archive_id"] if index else None,
            "unique_config_hashes": len(set(e.get("config_hash") for e in index)),
            "unique_tags": sorted(set(tag for e in index for tag in e.get("tags", []))),
        }
