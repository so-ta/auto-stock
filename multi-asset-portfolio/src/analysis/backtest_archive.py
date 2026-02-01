"""
Backtest Archive - バックテスト結果のアーカイブデータ構造

バックテスト結果を永続保存し、再現性検証・比較を可能にするためのデータクラス。

設計方針:
- メタデータ・メトリクス: JSON形式（人間可読、diff可能）
- 時系列データ: Parquet形式（高圧縮、高速読み込み）
- 設定スナップショット: YAML形式（既存設定ファイルと互換）
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.hash_utils import compute_config_hash, compute_universe_hash


def generate_config_hash(config: Dict[str, Any], universe: List[str]) -> str:
    """
    結果に影響する設定項目のみでハッシュ生成

    Args:
        config: バックテスト設定辞書
        universe: 銘柄リスト

    Returns:
        16文字のハッシュ文字列
    """
    relevant_keys = [
        "initial_capital",
        "transaction_cost_bps",
        "slippage_bps",
        "rebalance_frequency",
        "start_date",
        "end_date",
        "max_weight",
        "min_weight",
        "allow_short",
        "risk_free_rate",
    ]
    config_subset = {k: config.get(k) for k in relevant_keys if k in config}
    config_subset["universe"] = sorted(universe)

    # 日付をISO形式に統一
    for date_key in ["start_date", "end_date"]:
        if date_key in config_subset and config_subset[date_key]:
            val = config_subset[date_key]
            if isinstance(val, datetime):
                config_subset[date_key] = val.isoformat()
            elif hasattr(val, "isoformat"):
                config_subset[date_key] = val.isoformat()

    return compute_config_hash(config_subset)


def generate_universe_hash(universe: List[str]) -> str:
    """
    銘柄リストのハッシュを生成

    Args:
        universe: 銘柄リスト

    Returns:
        16文字のハッシュ文字列

    Note:
        sha256ではなくmd5を使用（compute_universe_hashのデフォルト）。
        既存互換性のためこの関数を維持。
    """
    # Note: This uses md5 via compute_universe_hash for consistency
    # with other universe hash implementations in the codebase
    return compute_universe_hash(universe)


def get_code_version() -> str:
    """
    現在のコードバージョンを取得（gitコミットハッシュ）

    Returns:
        gitコミットハッシュ（短縮形）、取得できない場合は"unknown"
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def generate_archive_id(config_hash: str) -> str:
    """
    アーカイブIDを生成

    フォーマット: bt_YYYYMMDD_HHMMSS_<microseconds>_<hash[:8]>

    Args:
        config_hash: 設定ハッシュ

    Returns:
        アーカイブID文字列
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # マイクロ秒を追加して一意性を保証
    micro = f"{now.microsecond:06d}"[:4]
    return f"bt_{timestamp}_{micro}_{config_hash[:8]}"


@dataclass
class BacktestArchive:
    """
    バックテスト結果のアーカイブ

    Attributes:
        archive_id: アーカイブ識別子（bt_YYYYMMDD_HHMMSS_hash[:8]）
        created_at: アーカイブ作成日時

        # 再現性情報
        config_hash: 設定のSHA256ハッシュ（16文字）
        universe_hash: 銘柄リストのハッシュ（16文字）
        code_version: gitコミットハッシュまたはバージョン

        # メタデータ
        name: 人間可読な名前
        description: 説明文
        tags: タグリスト（検索用）

        # 設定スナップショット
        config_snapshot: 完全な設定辞書
        universe: 銘柄リスト

        # 結果メトリクス
        metrics: パフォーマンスメトリクス
        trading_stats: 取引統計

        # 時系列ファイル参照（Parquetファイル名）
        timeseries_file: 日次時系列データファイル
        rebalances_file: リバランス詳細データファイル
    """

    # 識別子
    archive_id: str
    created_at: datetime

    # 再現性情報
    config_hash: str
    universe_hash: str
    code_version: str

    # メタデータ
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 設定スナップショット
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    universe: List[str] = field(default_factory=list)

    # 結果メトリクス
    metrics: Dict[str, float] = field(default_factory=dict)
    trading_stats: Dict[str, Any] = field(default_factory=dict)

    # 期間情報
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # 時系列ファイル参照
    timeseries_file: str = "timeseries.parquet"
    rebalances_file: str = "rebalances.parquet"

    # エンジン情報
    engine_name: str = "unknown"

    # 警告・エラー
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # 実行情報（進捗トラッキングデータ）
    execution_info: Optional[Dict[str, Any]] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        メタデータ用の辞書に変換（JSON保存用）

        Returns:
            metadata.jsonに保存する辞書
        """
        return {
            "archive_id": self.archive_id,
            "created_at": self.created_at.isoformat(),
            "config_hash": self.config_hash,
            "universe_hash": self.universe_hash,
            "code_version": self.code_version,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "metrics": self.metrics,
            "trading_stats": self.trading_stats,
            "engine_name": self.engine_name,
            "timeseries_file": self.timeseries_file,
            "rebalances_file": self.rebalances_file,
            "warnings": self.warnings,
            "errors": self.errors,
            "universe_count": len(self.universe),
            "execution_info": self.execution_info,
        }

    @classmethod
    def from_metadata_dict(
        cls,
        data: Dict[str, Any],
        config_snapshot: Optional[Dict[str, Any]] = None,
        universe: Optional[List[str]] = None,
    ) -> "BacktestArchive":
        """
        メタデータ辞書からインスタンスを作成

        Args:
            data: metadata.jsonから読み込んだ辞書
            config_snapshot: 設定スナップショット（別ファイルから読み込み）
            universe: 銘柄リスト（別ファイルから読み込み）

        Returns:
            BacktestArchiveインスタンス
        """
        return cls(
            archive_id=data["archive_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config_hash=data["config_hash"],
            universe_hash=data["universe_hash"],
            code_version=data.get("code_version", "unknown"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            config_snapshot=config_snapshot or {},
            universe=universe or [],
            metrics=data.get("metrics", {}),
            trading_stats=data.get("trading_stats", {}),
            start_date=(
                datetime.fromisoformat(data["start_date"])
                if data.get("start_date")
                else None
            ),
            end_date=(
                datetime.fromisoformat(data["end_date"])
                if data.get("end_date")
                else None
            ),
            engine_name=data.get("engine_name", "unknown"),
            timeseries_file=data.get("timeseries_file", "timeseries.parquet"),
            rebalances_file=data.get("rebalances_file", "rebalances.parquet"),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            execution_info=data.get("execution_info"),
        )

    def to_index_entry(self) -> Dict[str, Any]:
        """
        インデックス用のコンパクトなエントリを生成

        Returns:
            index.jsonに追加するエントリ
        """
        return {
            "archive_id": self.archive_id,
            "created_at": self.created_at.isoformat(),
            "name": self.name,
            "tags": self.tags,
            "config_hash": self.config_hash,
            "universe_hash": self.universe_hash,
            "total_return": self.metrics.get("total_return", 0.0),
            "sharpe_ratio": self.metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": self.metrics.get("max_drawdown", 0.0),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }

    @property
    def is_successful(self) -> bool:
        """バックテストが成功したか"""
        return len(self.errors) == 0

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"BacktestArchive("
            f"id={self.archive_id}, "
            f"name='{self.name}', "
            f"return={self.metrics.get('total_return', 0) * 100:.2f}%, "
            f"sharpe={self.metrics.get('sharpe_ratio', 0):.2f})"
        )


@dataclass
class ReproducibilityReport:
    """
    再現性検証レポート

    Attributes:
        is_reproducible: 再現性があるかどうか
        config_hash_match: 設定ハッシュが一致するか
        universe_hash_match: 銘柄ハッシュが一致するか
        metric_diffs: 各メトリクスの差分
        max_diff: 最大差分
        mismatches: 不一致項目リスト
        tolerance: 使用した許容誤差
    """

    is_reproducible: bool
    config_hash_match: bool
    universe_hash_match: bool
    metric_diffs: Dict[str, float] = field(default_factory=dict)
    max_diff: float = 0.0
    mismatches: List[str] = field(default_factory=list)
    tolerance: float = 1e-6

    def summary(self) -> str:
        """レポートのサマリを生成"""
        lines = [
            "=" * 60,
            "  REPRODUCIBILITY REPORT",
            "=" * 60,
            "",
            f"  Reproducible: {'Yes' if self.is_reproducible else 'No'}",
            f"  Config Hash Match: {'Yes' if self.config_hash_match else 'No'}",
            f"  Universe Hash Match: {'Yes' if self.universe_hash_match else 'No'}",
            f"  Max Metric Diff: {self.max_diff:.2e}",
            f"  Tolerance: {self.tolerance:.2e}",
            "",
        ]

        if self.mismatches:
            lines.extend([
                "-" * 60,
                "  MISMATCHES",
                "-" * 60,
            ])
            for mismatch in self.mismatches:
                lines.append(f"  - {mismatch}")
            lines.append("")

        if self.metric_diffs:
            lines.extend([
                "-" * 60,
                "  METRIC DIFFERENCES",
                "-" * 60,
            ])
            for metric, diff in sorted(self.metric_diffs.items()):
                status = "OK" if abs(diff) <= self.tolerance else "FAIL"
                lines.append(f"  {metric:20s}: {diff:+.6e} [{status}]")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """
    複数アーカイブの比較結果

    Attributes:
        archive_ids: 比較対象のアーカイブID
        archives: アーカイブオブジェクト
        metric_comparison: メトリクス比較テーブル
        config_diffs: 設定の差分
    """

    archive_ids: List[str]
    archives: List[BacktestArchive]
    metric_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    config_diffs: Dict[str, List[Any]] = field(default_factory=dict)

    def summary(self) -> str:
        """比較サマリを生成"""
        if not self.archives:
            return "No archives to compare"

        lines = [
            "=" * 80,
            "  BACKTEST COMPARISON",
            "=" * 80,
            "",
        ]

        # ヘッダー行
        header = "  {:25s}".format("Metric")
        for archive in self.archives:
            header += " | {:17s}".format(archive.archive_id[-12:])
        if len(self.archives) == 2:
            header += " | {:10s}".format("Diff")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        # メトリクス行
        metrics_to_show = [
            ("name", "Name", "{:s}"),
            ("config_hash", "Config Hash", "{:s}"),
            ("total_return", "Total Return", "{:+.2%}"),
            ("annual_return", "Annual Return", "{:+.2%}"),
            ("sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
            ("sortino_ratio", "Sortino Ratio", "{:.3f}"),
            ("max_drawdown", "Max Drawdown", "{:.2%}"),
            ("volatility", "Volatility", "{:.2%}"),
            ("calmar_ratio", "Calmar Ratio", "{:.3f}"),
            ("total_turnover", "Total Turnover", "{:.2%}"),
        ]

        for key, label, fmt in metrics_to_show:
            row = "  {:25s}".format(label)
            values = []
            for archive in self.archives:
                if key == "name":
                    val = archive.name[:15] if archive.name else "-"
                elif key == "config_hash":
                    val = archive.config_hash[:8]
                elif key in archive.metrics:
                    val = archive.metrics[key]
                elif key in archive.trading_stats:
                    val = archive.trading_stats[key]
                else:
                    val = None
                values.append(val)

                if val is None:
                    row += " | {:>17s}".format("-")
                elif isinstance(val, str):
                    row += " | {:>17s}".format(val)
                else:
                    row += " | {:>17s}".format(fmt.format(val))

            # 差分（2つの場合のみ）
            if len(values) == 2 and all(isinstance(v, (int, float)) for v in values if v is not None):
                if values[0] is not None and values[1] is not None:
                    diff = values[0] - values[1]
                    row += " | {:>+10.4f}".format(diff)

            lines.append(row)

        lines.append("=" * 80)
        return "\n".join(lines)
