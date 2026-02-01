"""
Backtest Comparator - バックテスト結果の比較・再現性検証

複数のバックテスト結果を比較し、再現性を検証するためのクラス。

主要機能:
- 複数アーカイブのメトリクス比較
- 再現性検証（同一設定で同一結果が得られるか）
- 2つの結果の詳細差分分析
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.base import UnifiedBacktestResult
from src.analysis.backtest_archive import (
    BacktestArchive,
    ReproducibilityReport,
    ComparisonResult,
    generate_config_hash,
    generate_universe_hash,
)
from src.analysis.result_store import BacktestResultStore

logger = logging.getLogger(__name__)


@dataclass
class DiffReport:
    """
    2つのアーカイブの詳細差分レポート

    Attributes:
        archive_id_1: 1つ目のアーカイブID
        archive_id_2: 2つ目のアーカイブID
        config_diffs: 設定の差分
        metric_diffs: メトリクスの差分
        timeseries_correlation: 時系列の相関係数
        rebalance_diffs: リバランスの差分サマリ
    """

    archive_id_1: str
    archive_id_2: str
    config_diffs: Dict[str, tuple] = field(default_factory=dict)
    metric_diffs: Dict[str, float] = field(default_factory=dict)
    timeseries_correlation: float = 0.0
    timeseries_rmse: float = 0.0
    rebalance_count_diff: int = 0
    weight_correlation: float = 0.0

    def summary(self) -> str:
        """差分サマリを生成"""
        lines = [
            "=" * 70,
            "  DIFF REPORT",
            "=" * 70,
            "",
            f"  Archive 1: {self.archive_id_1}",
            f"  Archive 2: {self.archive_id_2}",
            "",
        ]

        # 設定差分
        if self.config_diffs:
            lines.extend([
                "-" * 70,
                "  CONFIG DIFFERENCES",
                "-" * 70,
            ])
            for key, (val1, val2) in self.config_diffs.items():
                lines.append(f"  {key}:")
                lines.append(f"    Archive 1: {val1}")
                lines.append(f"    Archive 2: {val2}")
            lines.append("")

        # メトリクス差分
        lines.extend([
            "-" * 70,
            "  METRIC DIFFERENCES",
            "-" * 70,
        ])
        for metric, diff in sorted(self.metric_diffs.items()):
            lines.append(f"  {metric:25s}: {diff:+.6f}")
        lines.append("")

        # 時系列比較
        lines.extend([
            "-" * 70,
            "  TIMESERIES COMPARISON",
            "-" * 70,
            f"  Correlation: {self.timeseries_correlation:.6f}",
            f"  RMSE: {self.timeseries_rmse:.6f}",
            "",
        ])

        # リバランス比較
        lines.extend([
            "-" * 70,
            "  REBALANCE COMPARISON",
            "-" * 70,
            f"  Rebalance Count Diff: {self.rebalance_count_diff}",
            f"  Weight Correlation: {self.weight_correlation:.6f}",
            "",
        ])

        lines.append("=" * 70)
        return "\n".join(lines)


class BacktestComparator:
    """
    バックテスト結果の比較・検証クラス

    Attributes:
        store: バックテスト結果ストア
    """

    def __init__(self, store: BacktestResultStore) -> None:
        """
        初期化

        Args:
            store: BacktestResultStore インスタンス
        """
        self.store = store

    def compare(
        self,
        archive_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """
        複数アーカイブを比較

        Args:
            archive_ids: 比較するアーカイブIDのリスト
            metrics: 比較するメトリクス（Noneの場合は全メトリクス）

        Returns:
            ComparisonResult インスタンス
        """
        if len(archive_ids) < 2:
            raise ValueError("At least 2 archive IDs required for comparison")

        # アーカイブ読み込み
        archives = []
        for archive_id in archive_ids:
            try:
                archive = self.store.load(archive_id)
                archives.append(archive)
            except FileNotFoundError:
                logger.warning(f"Archive not found: {archive_id}")
                raise

        # デフォルトメトリクス
        if metrics is None:
            metrics = [
                "total_return",
                "annual_return",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "volatility",
                "calmar_ratio",
                "win_rate",
            ]

        # メトリクス比較テーブル作成
        comparison_data = {}
        for archive in archives:
            comparison_data[archive.archive_id] = {
                metric: archive.metrics.get(metric, None)
                for metric in metrics
            }

        metric_df = pd.DataFrame(comparison_data).T
        metric_df.index.name = "archive_id"

        # 設定差分を検出
        config_diffs = self._find_config_diffs(archives)

        return ComparisonResult(
            archive_ids=archive_ids,
            archives=archives,
            metric_comparison=metric_df,
            config_diffs=config_diffs,
        )

    def _find_config_diffs(
        self, archives: List[BacktestArchive]
    ) -> Dict[str, List[Any]]:
        """設定の差分を検出"""
        if len(archives) < 2:
            return {}

        diffs = {}
        all_keys = set()
        for archive in archives:
            all_keys.update(archive.config_snapshot.keys())

        for key in all_keys:
            values = [archive.config_snapshot.get(key) for archive in archives]
            # 値が全て同じでない場合は差分として記録
            if len(set(str(v) for v in values)) > 1:
                diffs[key] = values

        return diffs

    def verify_reproducibility(
        self,
        archive_id: str,
        new_result: UnifiedBacktestResult,
        universe: Optional[List[str]] = None,
        tolerance: float = 1e-6,
    ) -> ReproducibilityReport:
        """
        再現性を検証

        保存されたアーカイブと新しいバックテスト結果を比較し、
        同一の結果が再現されるかを検証する。

        Args:
            archive_id: 比較対象のアーカイブID
            new_result: 新しいバックテスト結果
            universe: 銘柄リスト（new_resultから取得できない場合）
            tolerance: 許容誤差（デフォルト: 1e-6）

        Returns:
            ReproducibilityReport インスタンス
        """
        # アーカイブ読み込み
        archive = self.store.load(archive_id)

        # ユニバース取得
        if universe is None:
            symbols = set()
            for rebalance in new_result.rebalances:
                symbols.update(rebalance.weights_after.keys())
            universe = sorted(symbols)

        # 新しい結果の設定ハッシュを計算
        new_config_dict = new_result.config.to_dict() if new_result.config else {}
        new_config_hash = generate_config_hash(new_config_dict, universe)
        new_universe_hash = generate_universe_hash(universe)

        # ハッシュ比較
        config_hash_match = archive.config_hash == new_config_hash
        universe_hash_match = archive.universe_hash == new_universe_hash

        # メトリクス差分計算
        metric_diffs = {}
        mismatches = []

        metrics_to_compare = [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "volatility",
            "calmar_ratio",
        ]

        new_metrics = {
            "total_return": new_result.total_return,
            "annual_return": new_result.annual_return,
            "sharpe_ratio": new_result.sharpe_ratio,
            "sortino_ratio": new_result.sortino_ratio,
            "max_drawdown": new_result.max_drawdown,
            "volatility": new_result.volatility,
            "calmar_ratio": new_result.calmar_ratio,
        }

        max_diff = 0.0
        for metric in metrics_to_compare:
            old_val = archive.metrics.get(metric, 0.0)
            new_val = new_metrics.get(metric, 0.0)
            diff = abs(new_val - old_val)
            metric_diffs[metric] = new_val - old_val

            if diff > tolerance:
                mismatches.append(f"{metric}: {old_val:.6f} -> {new_val:.6f} (diff: {diff:.2e})")

            max_diff = max(max_diff, diff)

        # ハッシュ不一致も記録
        if not config_hash_match:
            mismatches.insert(0, f"Config hash mismatch: {archive.config_hash} vs {new_config_hash}")
        if not universe_hash_match:
            mismatches.insert(0, f"Universe hash mismatch: {archive.universe_hash} vs {new_universe_hash}")

        is_reproducible = (
            config_hash_match
            and universe_hash_match
            and max_diff <= tolerance
        )

        return ReproducibilityReport(
            is_reproducible=is_reproducible,
            config_hash_match=config_hash_match,
            universe_hash_match=universe_hash_match,
            metric_diffs=metric_diffs,
            max_diff=max_diff,
            mismatches=mismatches,
            tolerance=tolerance,
        )

    def diff(
        self,
        archive_id_1: str,
        archive_id_2: str,
    ) -> DiffReport:
        """
        2つの結果の差分を詳細分析

        Args:
            archive_id_1: 1つ目のアーカイブID
            archive_id_2: 2つ目のアーカイブID

        Returns:
            DiffReport インスタンス
        """
        # アーカイブ読み込み
        archive1 = self.store.load(archive_id_1)
        archive2 = self.store.load(archive_id_2)

        # 設定差分
        config_diffs = {}
        all_keys = set(archive1.config_snapshot.keys()) | set(archive2.config_snapshot.keys())
        for key in all_keys:
            val1 = archive1.config_snapshot.get(key)
            val2 = archive2.config_snapshot.get(key)
            if str(val1) != str(val2):
                config_diffs[key] = (val1, val2)

        # メトリクス差分
        metric_diffs = {}
        all_metrics = set(archive1.metrics.keys()) | set(archive2.metrics.keys())
        for metric in all_metrics:
            val1 = archive1.metrics.get(metric, 0.0)
            val2 = archive2.metrics.get(metric, 0.0)
            metric_diffs[metric] = val1 - val2

        # 時系列比較
        timeseries_correlation = 0.0
        timeseries_rmse = 0.0
        try:
            ts1 = self.store.load_timeseries(archive_id_1)
            ts2 = self.store.load_timeseries(archive_id_2)

            # 共通期間で比較
            common_idx = ts1.index.intersection(ts2.index)
            if len(common_idx) > 0:
                pv1 = ts1.loc[common_idx, "portfolio_value"]
                pv2 = ts2.loc[common_idx, "portfolio_value"]

                # 相関係数
                if len(pv1) > 1:
                    timeseries_correlation = float(pv1.corr(pv2))

                # RMSE（正規化）
                norm1 = pv1 / pv1.iloc[0]
                norm2 = pv2 / pv2.iloc[0]
                timeseries_rmse = float(np.sqrt(np.mean((norm1 - norm2) ** 2)))
        except (FileNotFoundError, KeyError):
            pass

        # リバランス比較
        rebalance_count_diff = archive1.trading_stats.get("n_rebalances", 0) - archive2.trading_stats.get("n_rebalances", 0)

        weight_correlation = 0.0
        try:
            reb1 = self.store.load_rebalances(archive_id_1)
            reb2 = self.store.load_rebalances(archive_id_2)

            # 共通日付でウェイトを比較
            reb1["date"] = pd.to_datetime(reb1["timestamp"]).dt.date
            reb2["date"] = pd.to_datetime(reb2["timestamp"]).dt.date

            common_dates = set(reb1["date"]) & set(reb2["date"])
            if common_dates:
                weight_correlations = []
                for date in common_dates:
                    w1 = reb1[reb1["date"] == date]["weights_after"].iloc[0]
                    w2 = reb2[reb2["date"] == date]["weights_after"].iloc[0]

                    # 共通銘柄のウェイトで相関計算
                    common_symbols = set(w1.keys()) & set(w2.keys())
                    if common_symbols:
                        v1 = [w1[s] for s in common_symbols]
                        v2 = [w2[s] for s in common_symbols]
                        if len(v1) > 1:
                            corr = np.corrcoef(v1, v2)[0, 1]
                            if not np.isnan(corr):
                                weight_correlations.append(corr)

                if weight_correlations:
                    weight_correlation = float(np.mean(weight_correlations))
        except (FileNotFoundError, KeyError, IndexError):
            pass

        return DiffReport(
            archive_id_1=archive_id_1,
            archive_id_2=archive_id_2,
            config_diffs=config_diffs,
            metric_diffs=metric_diffs,
            timeseries_correlation=timeseries_correlation,
            timeseries_rmse=timeseries_rmse,
            rebalance_count_diff=rebalance_count_diff,
            weight_correlation=weight_correlation,
        )

    def find_similar(
        self,
        archive_id: str,
        min_return_diff: float = 0.05,
        min_sharpe_diff: float = 0.3,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        類似のアーカイブを検索

        Args:
            archive_id: 基準となるアーカイブID
            min_return_diff: リターン差分の最大値
            min_sharpe_diff: シャープレシオ差分の最大値
            limit: 最大取得件数

        Returns:
            類似アーカイブのリスト
        """
        archive = self.store.load(archive_id)
        base_return = archive.metrics.get("total_return", 0.0)
        base_sharpe = archive.metrics.get("sharpe_ratio", 0.0)

        all_archives = self.store.list_archives(limit=1000)
        similar = []

        for entry in all_archives:
            if entry["archive_id"] == archive_id:
                continue

            entry_return = entry.get("total_return", 0.0)
            entry_sharpe = entry.get("sharpe_ratio", 0.0)

            return_diff = abs(entry_return - base_return)
            sharpe_diff = abs(entry_sharpe - base_sharpe)

            if return_diff <= min_return_diff and sharpe_diff <= min_sharpe_diff:
                entry["return_diff"] = return_diff
                entry["sharpe_diff"] = sharpe_diff
                similar.append(entry)

            if len(similar) >= limit:
                break

        # 類似度でソート
        similar.sort(key=lambda x: x["return_diff"] + x["sharpe_diff"])
        return similar
