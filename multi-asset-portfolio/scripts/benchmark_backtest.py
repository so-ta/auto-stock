#!/usr/bin/env python3
"""
Backtest Performance Benchmark Script (cmd_036 Task 5)

高速化最適化の効果を定量的に検証するベンチマークスクリプト。

テストケース:
- 小規模: 10銘柄 × 1年
- 中規模: 100銘柄 × 3年
- 大規模: 777銘柄 × 5年

計測項目:
- 総実行時間
- リバランス1回あたりの時間
- メモリ使用量
- キャッシュヒット率

成功基準:
- 5年バックテスト: 3時間以内 → 30分以内（10倍高速化）

Usage:
    python scripts/benchmark_backtest.py [--quick] [--report]
    python scripts/benchmark_backtest.py --scale small
    python scripts/benchmark_backtest.py --scale all --report
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkConfig:
    """Benchmark configuration for a single test case."""
    name: str
    n_assets: int
    years: int
    frequency: str = "monthly"
    description: str = ""


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark."""
    total_time_sec: float = 0.0
    rebalance_count: int = 0
    time_per_rebalance_sec: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    signal_cache_hits: int = 0
    signal_cache_misses: int = 0
    signal_cache_hit_rate: float = 0.0
    cov_cache_hits: int = 0
    cov_cache_misses: int = 0
    cov_cache_hit_rate: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete result for one benchmark test."""
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
        }


# Standard benchmark configurations
BENCHMARK_CONFIGS = {
    "small": BenchmarkConfig(
        name="small",
        n_assets=10,
        years=1,
        frequency="monthly",
        description="小規模: 10銘柄 × 1年",
    ),
    "medium": BenchmarkConfig(
        name="medium",
        n_assets=100,
        years=3,
        frequency="monthly",
        description="中規模: 100銘柄 × 3年",
    ),
    "large": BenchmarkConfig(
        name="large",
        n_assets=777,
        years=5,
        frequency="monthly",
        description="大規模: 777銘柄 × 5年",
    ),
}


def measure_memory() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def load_universe(n_assets: int) -> List[str]:
    """Load universe symbols (up to n_assets)."""
    import yaml

    universe_path = PROJECT_ROOT / "config" / "universe_standard.yaml"
    if not universe_path.exists():
        universe_path = PROJECT_ROOT / "config" / "universe.yaml"

    if not universe_path.exists():
        # Generate synthetic symbols
        return [f"ASSET_{i:04d}" for i in range(n_assets)]

    with open(universe_path) as f:
        data = yaml.safe_load(f)

    # Collect all symbols
    symbols = []
    for section in ["us_stocks", "japan_stocks", "etfs", "forex"]:
        if section in data and "symbols" in data[section]:
            symbols.extend(data[section]["symbols"])

    # Also check top-level keys
    if "tickers" in data:
        symbols = data["tickers"]
    elif "passed_tickers" in data:
        symbols = data["passed_tickers"]

    return symbols[:n_assets]


def generate_mock_prices(
    symbols: List[str],
    years: int,
    frequency: str = "daily",
) -> Dict[str, pd.DataFrame]:
    """Generate mock price data for benchmarking."""
    np.random.seed(42)

    if frequency == "daily":
        n_days = years * 252
    elif frequency == "weekly":
        n_days = years * 52
    else:  # monthly
        n_days = years * 12 * 21  # Approximate trading days

    # Generate dates (business days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(n_days * 1.5))  # Buffer for weekends
    dates = pd.bdate_range(start=start_date, end=end_date)[:n_days]

    prices = {}
    for symbol in symbols:
        # Random walk with drift
        returns = np.random.randn(len(dates)) * 0.02 + 0.0003
        price_series = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "open": price_series * (1 + np.random.randn(len(dates)) * 0.005),
            "high": price_series * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            "low": price_series * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            "close": price_series,
            "volume": np.random.randint(100000, 10000000, len(dates)),
        }, index=dates)

        prices[symbol] = df

    return prices


def load_real_prices(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load real price data from cache."""
    import polars as pl

    cache_dir = PROJECT_ROOT / "cache" / "price_data"
    prices = {}

    for symbol in symbols:
        safe_symbol = symbol.replace("=", "_").replace("/", "_")
        path = cache_dir / f"{safe_symbol}.parquet"

        if path.exists():
            try:
                df = pl.read_parquet(path).to_pandas()
                if "date" in df.columns:
                    df = df.set_index("date")
                prices[symbol] = df
            except Exception:
                pass

    return prices


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics from all caches."""
    stats = {
        "signal_cache": {"hits": 0, "misses": 0, "hit_rate": 0.0},
        "cov_cache": {"hits": 0, "misses": 0, "hit_rate": 0.0},
    }

    # Try to get SignalCache stats
    try:
        from src.orchestrator.signal_cache import get_signal_cache
        signal_cache = get_signal_cache()
        cache_stats = signal_cache.stats
        stats["signal_cache"] = {
            "hits": cache_stats.hits,
            "misses": cache_stats.misses,
            "hit_rate": cache_stats.hit_rate,
        }
    except Exception:
        pass

    # Try to get CovarianceCache stats
    try:
        from src.allocation.covariance_cache import CovarianceCache
        # CovarianceCache is typically instantiated per RiskEstimator
        # For benchmark, we track via global variable if available
        if hasattr(CovarianceCache, "_global_instance"):
            cov_cache = CovarianceCache._global_instance
            if cov_cache:
                cov_stats = cov_cache.stats
                stats["cov_cache"] = {
                    "hits": cov_stats.hits,
                    "misses": cov_stats.misses,
                    "hit_rate": cov_stats.hit_rate,
                }
    except Exception:
        pass

    return stats


def clear_caches() -> None:
    """Clear all caches before benchmark."""
    try:
        from src.orchestrator.signal_cache import get_signal_cache
        get_signal_cache().clear()
    except Exception:
        pass


def run_benchmark(config: BenchmarkConfig, use_mock_data: bool = True) -> BenchmarkResult:
    """Run a single benchmark test."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {config.description}")
    print(f"  Assets: {config.n_assets}, Years: {config.years}, Frequency: {config.frequency}")
    print(f"{'='*60}")

    metrics = BenchmarkMetrics()

    # Clear caches
    clear_caches()
    gc.collect()

    # Load or generate data
    print("  Loading data...")
    symbols = load_universe(config.n_assets)
    symbols = symbols[:config.n_assets]  # Ensure exact count

    if use_mock_data or len(symbols) < config.n_assets:
        if len(symbols) < config.n_assets:
            # Supplement with synthetic symbols
            additional = [f"SYN_{i:04d}" for i in range(config.n_assets - len(symbols))]
            symbols.extend(additional)
        prices = generate_mock_prices(symbols, config.years)
    else:
        prices = load_real_prices(symbols)
        if len(prices) < config.n_assets * 0.5:
            # Not enough real data, use mock
            prices = generate_mock_prices(symbols, config.years)

    print(f"  Loaded {len(prices)} assets")

    # Prepare dates
    first_prices = next(iter(prices.values()))
    all_dates = first_prices.index.tolist()
    start_date = all_dates[0].strftime("%Y-%m-%d")
    end_date = all_dates[-1].strftime("%Y-%m-%d")

    # Memory before
    gc.collect()
    mem_before = measure_memory()

    # Initial cache stats
    cache_stats_before = get_cache_stats()

    # Run benchmark
    print("  Running backtest...")
    start_time = time.perf_counter()

    try:
        from src.orchestrator.unified_executor import UnifiedExecutor

        executor = UnifiedExecutor()
        result = executor.run_backtest(
            universe=symbols,
            prices=prices,
            start_date=start_date,
            end_date=end_date,
            frequency=config.frequency,
            initial_capital=1_000_000,
        )

        metrics.rebalance_count = len(result.portfolio_values) if hasattr(result, "portfolio_values") else 0

        # Try to get more accurate rebalance count
        if hasattr(result, "weights_history"):
            metrics.rebalance_count = len(result.weights_history)
        elif hasattr(result, "trades"):
            # Estimate from trades
            metrics.rebalance_count = max(1, len(set(t.get("date", t.get("timestamp", "")) for t in result.trades if isinstance(t, dict))))

        metrics.success = True

    except Exception as e:
        metrics.success = False
        metrics.error = f"{type(e).__name__}: {str(e)}"
        print(f"  ERROR: {metrics.error}")
        traceback.print_exc()

    elapsed = time.perf_counter() - start_time
    metrics.total_time_sec = elapsed

    # Memory after
    mem_after = measure_memory()
    metrics.memory_peak_mb = mem_after
    metrics.memory_delta_mb = mem_after - mem_before

    # Cache stats after
    cache_stats_after = get_cache_stats()

    # Calculate cache metrics (delta)
    metrics.signal_cache_hits = cache_stats_after["signal_cache"]["hits"] - cache_stats_before["signal_cache"]["hits"]
    metrics.signal_cache_misses = cache_stats_after["signal_cache"]["misses"] - cache_stats_before["signal_cache"]["misses"]
    total_signal = metrics.signal_cache_hits + metrics.signal_cache_misses
    metrics.signal_cache_hit_rate = metrics.signal_cache_hits / total_signal if total_signal > 0 else 0.0

    metrics.cov_cache_hits = cache_stats_after["cov_cache"]["hits"] - cache_stats_before["cov_cache"]["hits"]
    metrics.cov_cache_misses = cache_stats_after["cov_cache"]["misses"] - cache_stats_before["cov_cache"]["misses"]
    total_cov = metrics.cov_cache_hits + metrics.cov_cache_misses
    metrics.cov_cache_hit_rate = metrics.cov_cache_hits / total_cov if total_cov > 0 else 0.0

    # Calculate per-rebalance time
    if metrics.rebalance_count > 0:
        metrics.time_per_rebalance_sec = metrics.total_time_sec / metrics.rebalance_count

    # Print results
    print(f"\n  Results:")
    print(f"    Total time:          {metrics.total_time_sec:.2f} sec ({metrics.total_time_sec/60:.1f} min)")
    print(f"    Rebalances:          {metrics.rebalance_count}")
    print(f"    Time/rebalance:      {metrics.time_per_rebalance_sec*1000:.1f} ms")
    print(f"    Memory delta:        {metrics.memory_delta_mb:.1f} MB")
    print(f"    Signal cache:        {metrics.signal_cache_hit_rate*100:.1f}% hit rate ({metrics.signal_cache_hits}/{metrics.signal_cache_hits + metrics.signal_cache_misses})")
    print(f"    Cov cache:           {metrics.cov_cache_hit_rate*100:.1f}% hit rate ({metrics.cov_cache_hits}/{metrics.cov_cache_hits + metrics.cov_cache_misses})")

    return BenchmarkResult(config=config, metrics=metrics)


def generate_report(results: List[BenchmarkResult]) -> str:
    """Generate markdown report."""
    lines = [
        "# Backtest Optimization Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Scale | Assets | Years | Time (min) | Time/Rebalance | Memory | Signal Cache | Cov Cache | Status |",
        "|-------|--------|-------|------------|----------------|--------|--------------|-----------|--------|",
    ]

    for r in results:
        c = r.config
        m = r.metrics
        status = "PASS" if m.success else "FAIL"
        lines.append(
            f"| {c.name} | {c.n_assets} | {c.years} | "
            f"{m.total_time_sec/60:.1f} | "
            f"{m.time_per_rebalance_sec*1000:.0f}ms | "
            f"{m.memory_delta_mb:.0f}MB | "
            f"{m.signal_cache_hit_rate*100:.0f}% | "
            f"{m.cov_cache_hit_rate*100:.0f}% | "
            f"{status} |"
        )

    lines.extend([
        "",
        "## Performance Target",
        "",
        "| Metric | Target | Baseline | Current |",
        "|--------|--------|----------|---------|",
    ])

    # Find large benchmark result
    large_result = next((r for r in results if r.config.name == "large"), None)
    if large_result:
        current_min = large_result.metrics.total_time_sec / 60
        target_status = "PASS" if current_min <= 30 else "FAIL"
        lines.append(
            f"| 5-year backtest | 30 min | 180 min (est.) | {current_min:.1f} min ({target_status}) |"
        )
    else:
        lines.append("| 5-year backtest | 30 min | 180 min (est.) | Not tested |")

    lines.extend([
        "",
        "## Optimization Features (cmd_036)",
        "",
        "- **task_036_1**: Signal computation caching (SignalCache)",
        "- **task_036_2**: Covariance matrix caching (CovarianceCache)",
        "- **task_036_3**: Parallel signal generation (ParallelSignalGenerator)",
        "- **task_036_4**: Lightweight pipeline mode for backtest",
        "",
        "## Detailed Results",
        "",
    ])

    for r in results:
        c = r.config
        m = r.metrics
        lines.extend([
            f"### {c.name}: {c.description}",
            "",
            f"- **Total time:** {m.total_time_sec:.2f} sec ({m.total_time_sec/60:.1f} min)",
            f"- **Rebalance count:** {m.rebalance_count}",
            f"- **Time per rebalance:** {m.time_per_rebalance_sec*1000:.1f} ms",
            f"- **Memory usage:** {m.memory_delta_mb:.1f} MB",
            f"- **Signal cache:** {m.signal_cache_hits} hits / {m.signal_cache_misses} misses ({m.signal_cache_hit_rate*100:.1f}%)",
            f"- **Covariance cache:** {m.cov_cache_hits} hits / {m.cov_cache_misses} misses ({m.cov_cache_hit_rate*100:.1f}%)",
            f"- **Status:** {'SUCCESS' if m.success else 'FAILED - ' + (m.error or 'Unknown error')}",
            "",
        ])

    lines.extend([
        "## Success Criteria",
        "",
        "- 5年バックテスト（777銘柄）: 3時間 → 30分以内 (10倍高速化)",
        "",
    ])

    if large_result and large_result.metrics.success:
        current_min = large_result.metrics.total_time_sec / 60
        if current_min <= 30:
            lines.append("**Result: PASS - 目標達成**")
            speedup = 180 / current_min  # Estimated baseline
            lines.append(f"  - 推定高速化倍率: {speedup:.1f}x")
        else:
            lines.append(f"**Result: FAIL - 現在 {current_min:.1f} 分 (目標: 30分以内)**")
    else:
        lines.append("**Result: Large benchmark not executed or failed**")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Backtest Performance Benchmark")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Benchmark scale to run",
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (small only)")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock data (default: True)")
    parser.add_argument("--real", action="store_true", help="Use real price data if available")
    args = parser.parse_args()

    use_mock = not args.real

    print("=" * 60)
    print("BACKTEST PERFORMANCE BENCHMARK (cmd_036 Task 5)")
    print("=" * 60)
    print(f"Data source: {'Mock' if use_mock else 'Real (with fallback to mock)'}")

    results: List[BenchmarkResult] = []

    # Determine which benchmarks to run
    if args.quick:
        configs = [BENCHMARK_CONFIGS["small"]]
    elif args.scale == "all":
        configs = list(BENCHMARK_CONFIGS.values())
    else:
        configs = [BENCHMARK_CONFIGS[args.scale]]

    # Run benchmarks
    for config in configs:
        result = run_benchmark(config, use_mock_data=use_mock)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for r in results:
        status = "PASS" if r.metrics.success else "FAIL"
        print(f"  {r.config.name}: {r.metrics.total_time_sec:.1f}s ({status})")

    # Check success criteria
    large_result = next((r for r in results if r.config.name == "large"), None)
    if large_result and large_result.metrics.success:
        minutes = large_result.metrics.total_time_sec / 60
        if minutes <= 30:
            print(f"\n  SUCCESS: Large benchmark completed in {minutes:.1f} min (target: 30 min)")
        else:
            print(f"\n  TARGET NOT MET: Large benchmark took {minutes:.1f} min (target: 30 min)")

    # Generate report
    if args.report or len(results) > 1:
        report_path = PROJECT_ROOT / "results" / "benchmark_optimization_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = generate_report(results)
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\n  Report saved to: {report_path}")

    # Save JSON results
    json_path = PROJECT_ROOT / "results" / "benchmark_optimization_results.json"
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print(f"  JSON results saved to: {json_path}")

    # Exit code based on success
    failed = sum(1 for r in results if not r.metrics.success)
    if failed > 0:
        print(f"\n  {failed} benchmark(s) failed")
        sys.exit(1)

    print("\n  All benchmarks completed successfully")


if __name__ == "__main__":
    main()
