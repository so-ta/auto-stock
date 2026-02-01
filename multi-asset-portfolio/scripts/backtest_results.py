#!/usr/bin/env python3
"""
Backtest Results CLI - バックテスト結果の管理・比較ツール

保存されたバックテスト結果を一覧表示、比較、再現性検証、削除するためのCLI。

Usage:
    # アーカイブ一覧
    python scripts/backtest_results.py list
    python scripts/backtest_results.py list --tags monthly
    python scripts/backtest_results.py list --name "Test"

    # 詳細表示
    python scripts/backtest_results.py show bt_20260131_123456_abc123

    # 比較
    python scripts/backtest_results.py compare bt_abc123 bt_def456
    python scripts/backtest_results.py compare bt_abc123 bt_def456 --html

    # 再現性検証
    python scripts/backtest_results.py verify bt_abc123

    # 削除
    python scripts/backtest_results.py delete bt_abc123
    python scripts/backtest_results.py delete bt_abc123 --force

    # 統計情報
    python scripts/backtest_results.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.result_store import BacktestResultStore
from src.analysis.backtest_comparator import BacktestComparator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> int:
    """アーカイブ一覧を表示"""
    store = BacktestResultStore(args.results_dir)

    # フィルタ条件
    tags = args.tags.split(",") if args.tags else None

    archives = store.list_archives(
        tags=tags,
        config_hash=args.config_hash,
        name_contains=args.name,
        limit=args.limit,
    )

    if not archives:
        print("No archives found.")
        return 0

    if args.json:
        print(json.dumps(archives, indent=2, ensure_ascii=False))
        return 0

    # テーブル形式で表示
    print()
    print("=" * 100)
    print("  BACKTEST ARCHIVES")
    print("=" * 100)
    print()
    print(f"  {'ID':<30} {'Name':<20} {'Return':>10} {'Sharpe':>8} {'Created':<20}")
    print("  " + "-" * 96)

    for entry in archives:
        archive_id = entry["archive_id"]
        name = entry.get("name", "-")[:18]
        total_return = entry.get("total_return", 0.0)
        sharpe = entry.get("sharpe_ratio", 0.0)
        created = entry.get("created_at", "")[:19]

        print(f"  {archive_id:<30} {name:<20} {total_return:>+9.2%} {sharpe:>8.3f} {created:<20}")

    print()
    print(f"  Total: {len(archives)} archive(s)")
    print("=" * 100)

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """アーカイブ詳細を表示"""
    store = BacktestResultStore(args.results_dir)

    try:
        archive = store.load(args.archive_id)
    except FileNotFoundError:
        print(f"Error: Archive not found: {args.archive_id}")
        return 1

    if args.json:
        print(json.dumps(archive.to_metadata_dict(), indent=2, ensure_ascii=False))
        return 0

    # 詳細表示
    print()
    print("=" * 70)
    print(f"  ARCHIVE DETAILS: {archive.archive_id}")
    print("=" * 70)
    print()
    print(f"  Name:           {archive.name}")
    print(f"  Description:    {archive.description or '-'}")
    print(f"  Tags:           {', '.join(archive.tags) if archive.tags else '-'}")
    print(f"  Created:        {archive.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Code Version:   {archive.code_version}")
    print()
    print("-" * 70)
    print("  HASHES")
    print("-" * 70)
    print(f"  Config Hash:    {archive.config_hash}")
    print(f"  Universe Hash:  {archive.universe_hash}")
    print(f"  Universe Size:  {len(archive.universe)} symbols")
    print()
    print("-" * 70)
    print("  PERIOD")
    print("-" * 70)
    print(f"  Start Date:     {archive.start_date.strftime('%Y-%m-%d') if archive.start_date else '-'}")
    print(f"  End Date:       {archive.end_date.strftime('%Y-%m-%d') if archive.end_date else '-'}")
    print(f"  Engine:         {archive.engine_name}")
    print()
    print("-" * 70)
    print("  METRICS")
    print("-" * 70)
    for metric, value in sorted(archive.metrics.items()):
        if "return" in metric or "drawdown" in metric or "volatility" in metric:
            print(f"  {metric:<25}: {value:>+12.2%}")
        else:
            print(f"  {metric:<25}: {value:>12.4f}")
    print()
    print("-" * 70)
    print("  TRADING STATS")
    print("-" * 70)
    for stat, value in sorted(archive.trading_stats.items()):
        if isinstance(value, float):
            if "turnover" in stat:
                print(f"  {stat:<25}: {value:>12.2%}")
            elif "value" in stat or "cost" in stat:
                print(f"  {stat:<25}: ${value:>12,.2f}")
            else:
                print(f"  {stat:<25}: {value:>12.4f}")
        else:
            print(f"  {stat:<25}: {value:>12}")
    print()
    print("=" * 70)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """複数アーカイブを比較"""
    store = BacktestResultStore(args.results_dir)
    comparator = BacktestComparator(store)

    archive_ids = args.archive_ids

    if len(archive_ids) < 2:
        print("Error: At least 2 archive IDs required for comparison")
        return 1

    try:
        result = comparator.compare(archive_ids)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.json:
        output = {
            "archive_ids": result.archive_ids,
            "metric_comparison": result.metric_comparison.to_dict(),
            "config_diffs": result.config_diffs,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0

    # サマリ表示
    print(result.summary())

    # 詳細差分（2つの場合）
    if len(archive_ids) == 2 and args.detailed:
        diff_report = comparator.diff(archive_ids[0], archive_ids[1])
        print()
        print(diff_report.summary())

    # HTML出力
    if args.html:
        html_path = _generate_comparison_html(store, result, args.output)
        print(f"\nHTML report saved: {html_path}")

    return 0


def _generate_comparison_html(
    store: BacktestResultStore,
    comparison: "ComparisonResult",
    output_path: Optional[str] = None,
) -> str:
    """比較HTMLレポートを生成"""
    from src.analysis.report_generator import ReportGenerator

    if output_path is None:
        ids_str = "_".join(a.archive_id[-8:] for a in comparison.archives)
        output_path = str(store._comparisons_dir / f"compare_{ids_str}.html")

    # 時系列データ読み込み
    timeseries_list = []
    for archive in comparison.archives:
        try:
            ts = store.load_timeseries(archive.archive_id)
            timeseries_list.append((archive.name or archive.archive_id, ts))
        except FileNotFoundError:
            pass

    # HTMLレポート生成
    generator = ReportGenerator()
    generator.generate_comparison_html(
        comparison=comparison,
        timeseries_list=timeseries_list,
        output_path=output_path,
    )

    return output_path


def cmd_diff(args: argparse.Namespace) -> int:
    """2つのアーカイブの詳細差分を表示"""
    store = BacktestResultStore(args.results_dir)
    comparator = BacktestComparator(store)

    try:
        diff_report = comparator.diff(args.archive_id_1, args.archive_id_2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.json:
        output = {
            "archive_id_1": diff_report.archive_id_1,
            "archive_id_2": diff_report.archive_id_2,
            "config_diffs": {k: list(v) for k, v in diff_report.config_diffs.items()},
            "metric_diffs": diff_report.metric_diffs,
            "timeseries_correlation": diff_report.timeseries_correlation,
            "timeseries_rmse": diff_report.timeseries_rmse,
            "rebalance_count_diff": diff_report.rebalance_count_diff,
            "weight_correlation": diff_report.weight_correlation,
        }
        print(json.dumps(output, indent=2))
        return 0

    print(diff_report.summary())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """再現性を検証

    同一設定でバックテストを再実行し、結果が一致するか確認する。
    """
    store = BacktestResultStore(args.results_dir)

    try:
        archive = store.load(args.archive_id)
    except FileNotFoundError:
        print(f"Error: Archive not found: {args.archive_id}")
        return 1

    print(f"Verifying reproducibility for: {args.archive_id}")
    print(f"  Config Hash: {archive.config_hash}")
    print(f"  Universe: {len(archive.universe)} symbols")
    print()

    # 同一設定のアーカイブを検索
    same_config = store.find_by_config_hash(archive.config_hash)
    same_config = [aid for aid in same_config if aid != args.archive_id]

    if not same_config:
        print("No other archives with the same config hash found.")
        print("Run a new backtest with --save to compare:")
        print(f"  python scripts/run_backtest.py -f monthly --save --name 'Verify'")
        return 0

    print(f"Found {len(same_config)} archive(s) with same config:")
    for aid in same_config:
        print(f"  - {aid}")
    print()

    # 最新のものと比較
    comparator = BacktestComparator(store)

    for other_id in same_config[:3]:  # 最大3つまで
        print(f"Comparing with: {other_id}")

        other_archive = store.load(other_id)

        # メトリクス比較
        metric_diffs = {}
        for metric in ["total_return", "sharpe_ratio", "max_drawdown"]:
            val1 = archive.metrics.get(metric, 0.0)
            val2 = other_archive.metrics.get(metric, 0.0)
            diff = abs(val1 - val2)
            metric_diffs[metric] = diff
            status = "OK" if diff < args.tolerance else "DIFF"
            print(f"  {metric:<20}: {val1:.6f} vs {val2:.6f} (diff: {diff:.2e}) [{status}]")

        max_diff = max(metric_diffs.values())
        is_reproducible = max_diff < args.tolerance
        print(f"  Reproducible: {'Yes' if is_reproducible else 'No'} (max diff: {max_diff:.2e})")
        print()

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """アーカイブを削除"""
    store = BacktestResultStore(args.results_dir)

    if not store.exists(args.archive_id):
        print(f"Error: Archive not found: {args.archive_id}")
        return 1

    if not args.force:
        # 確認
        archive = store.load(args.archive_id)
        print(f"Archive to delete: {args.archive_id}")
        print(f"  Name: {archive.name}")
        print(f"  Created: {archive.created_at}")
        print(f"  Total Return: {archive.metrics.get('total_return', 0):.2%}")
        print()
        confirm = input("Delete this archive? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    success = store.delete(args.archive_id)
    if success:
        print(f"Deleted: {args.archive_id}")
        return 0
    else:
        print(f"Failed to delete: {args.archive_id}")
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """ストアの統計情報を表示"""
    store = BacktestResultStore(args.results_dir)
    stats = store.get_stats()

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    print()
    print("=" * 50)
    print("  BACKTEST RESULTS STORE STATS")
    print("=" * 50)
    print()
    print(f"  Total Archives:      {stats['total_archives']}")
    print(f"  Total Size:          {stats['total_size_mb']:.2f} MB")
    if stats.get('unique_config_hashes'):
        print(f"  Unique Configs:      {stats['unique_config_hashes']}")
    print(f"  Oldest Archive:      {stats.get('oldest_archive') or '-'}")
    print(f"  Newest Archive:      {stats.get('newest_archive') or '-'}")
    print()
    if stats.get("unique_tags"):
        print(f"  Tags: {', '.join(stats['unique_tags'])}")
    print("=" * 50)

    return 0


def cmd_find_similar(args: argparse.Namespace) -> int:
    """類似のアーカイブを検索"""
    store = BacktestResultStore(args.results_dir)
    comparator = BacktestComparator(store)

    try:
        similar = comparator.find_similar(
            args.archive_id,
            min_return_diff=args.return_diff,
            min_sharpe_diff=args.sharpe_diff,
            limit=args.limit,
        )
    except FileNotFoundError:
        print(f"Error: Archive not found: {args.archive_id}")
        return 1

    if not similar:
        print("No similar archives found.")
        return 0

    if args.json:
        print(json.dumps(similar, indent=2, ensure_ascii=False))
        return 0

    print()
    print(f"Similar archives to: {args.archive_id}")
    print("=" * 80)
    print(f"  {'ID':<30} {'Name':<20} {'Return Diff':>12} {'Sharpe Diff':>12}")
    print("  " + "-" * 76)

    for entry in similar:
        print(f"  {entry['archive_id']:<30} {entry.get('name', '-')[:18]:<20} "
              f"{entry['return_diff']:>+11.2%} {entry['sharpe_diff']:>+12.3f}")

    print("=" * 80)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest Results CLI - Manage and compare saved backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results-dir", "-d",
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List saved archives")
    list_parser.add_argument("--tags", type=str, help="Filter by tags (comma-separated)")
    list_parser.add_argument("--name", type=str, help="Filter by name (contains)")
    list_parser.add_argument("--config-hash", type=str, help="Filter by config hash")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")

    # show command
    show_parser = subparsers.add_parser("show", help="Show archive details")
    show_parser.add_argument("archive_id", help="Archive ID to show")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple archives")
    compare_parser.add_argument("archive_ids", nargs="+", help="Archive IDs to compare")
    compare_parser.add_argument("--html", action="store_true", help="Generate HTML report")
    compare_parser.add_argument("--output", "-o", type=str, help="Output path for HTML")
    compare_parser.add_argument("--detailed", action="store_true", help="Show detailed diff")

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Show detailed diff between two archives")
    diff_parser.add_argument("archive_id_1", help="First archive ID")
    diff_parser.add_argument("archive_id_2", help="Second archive ID")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify reproducibility")
    verify_parser.add_argument("archive_id", help="Archive ID to verify")
    verify_parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance (default: 1e-6)")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an archive")
    delete_parser.add_argument("archive_id", help="Archive ID to delete")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # stats command
    subparsers.add_parser("stats", help="Show store statistics")

    # find-similar command
    similar_parser = subparsers.add_parser("find-similar", help="Find similar archives")
    similar_parser.add_argument("archive_id", help="Reference archive ID")
    similar_parser.add_argument("--return-diff", type=float, default=0.05, help="Max return diff (default: 0.05)")
    similar_parser.add_argument("--sharpe-diff", type=float, default=0.3, help="Max Sharpe diff (default: 0.3)")
    similar_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # コマンド実行
    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "compare": cmd_compare,
        "diff": cmd_diff,
        "verify": cmd_verify,
        "delete": cmd_delete,
        "stats": cmd_stats,
        "find-similar": cmd_find_similar,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
