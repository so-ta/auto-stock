#!/usr/bin/env python3
"""
Performance Benchmark Script

Automated performance benchmarking for the Multi-Asset Portfolio system.
Measures execution time, memory usage, and throughput for various scenarios.

Usage:
    python scripts/benchmark.py [--quick] [--save] [--compare]

Options:
    --quick     Run quick benchmark (fewer iterations)
    --save      Save results to results/benchmark_history.json
    --compare   Compare with previous benchmark results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    scenario: str
    execution_time_sec: float
    memory_peak_mb: float
    iterations: int
    throughput: Optional[float] = None  # items/sec
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    timestamp: str
    git_commit: Optional[str]
    python_version: str
    platform: str
    results: List[BenchmarkResult]
    total_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "platform": self.platform,
            "results": [asdict(r) for r in self.results],
            "total_time_sec": self.total_time_sec,
        }


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def measure_memory() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback without psutil
        return 0.0


def run_benchmark(
    name: str,
    scenario: str,
    func: Callable,
    iterations: int = 3,
    warmup: int = 1,
) -> BenchmarkResult:
    """Run a benchmark and collect metrics."""
    # Warmup
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass
        gc.collect()

    # Measure
    times = []
    peak_memory = 0.0
    error = None
    success = True

    for i in range(iterations):
        gc.collect()
        mem_before = measure_memory()

        start = time.perf_counter()
        try:
            result = func()
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            success = False
            break
        elapsed = time.perf_counter() - start

        mem_after = measure_memory()
        peak_memory = max(peak_memory, mem_after - mem_before)
        times.append(elapsed)

    avg_time = np.mean(times) if times else 0.0

    return BenchmarkResult(
        name=name,
        scenario=scenario,
        execution_time_sec=avg_time,
        memory_peak_mb=peak_memory,
        iterations=len(times),
        success=success,
        error=error,
    )


class BacktestBenchmarks:
    """Backtest engine benchmarks."""

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.iterations = 2 if quick else 5

    def _generate_mock_prices(self, n_assets: int, n_days: int) -> pd.DataFrame:
        """Generate mock price data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
        symbols = [f"ASSET_{i:03d}" for i in range(n_assets)]

        # Random walk prices
        returns = np.random.randn(n_days, n_assets) * 0.02
        prices = 100 * np.exp(np.cumsum(returns, axis=0))

        return pd.DataFrame(prices, index=dates, columns=symbols)

    def benchmark_fast_engine(self, n_assets: int) -> BenchmarkResult:
        """Benchmark FastBacktestEngine."""
        from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig

        n_days = 252 if self.quick else 756  # 1 or 3 years
        prices = self._generate_mock_prices(n_assets, n_days)

        def run():
            config = FastBacktestConfig(
                start_date=prices.index[0],
                end_date=prices.index[-1],
                initial_capital=1_000_000,
                rebalance_frequency="weekly",
            )
            engine = FastBacktestEngine(config)
            result = engine.run(prices)
            return result

        result = run_benchmark(
            name="FastBacktestEngine",
            scenario=f"{n_assets}_assets_{n_days}_days",
            func=run,
            iterations=self.iterations,
        )
        result.throughput = n_assets * n_days / result.execution_time_sec
        result.metadata = {"n_assets": n_assets, "n_days": n_days}
        return result

    def benchmark_signal_computation(self, n_assets: int) -> BenchmarkResult:
        """Benchmark signal computation."""
        from src.signals.momentum import MomentumReturnSignal

        n_days = 252 if self.quick else 504
        prices = self._generate_mock_prices(n_assets, n_days)

        def run():
            signal = MomentumReturnSignal(lookback=20)
            results = []
            for col in prices.columns[:min(10, n_assets)]:  # Limit for speed
                df = pd.DataFrame({"close": prices[col]})
                result = signal.compute(df)
                results.append(result)
            return results

        result = run_benchmark(
            name="SignalComputation",
            scenario=f"{n_assets}_assets_{n_days}_days",
            func=run,
            iterations=self.iterations,
        )
        result.metadata = {"n_assets": n_assets, "n_days": n_days}
        return result

    def benchmark_covariance_estimation(self, n_assets: int) -> BenchmarkResult:
        """Benchmark covariance matrix estimation."""
        from src.backtest.incremental_cov import IncrementalCovarianceEstimator

        n_days = 252

        def run():
            estimator = IncrementalCovarianceEstimator(
                n_assets=n_assets,
                halflife=60,
            )
            np.random.seed(42)
            for _ in range(n_days):
                returns = np.random.randn(n_assets) * 0.02
                estimator.update(returns)
            cov = estimator.get_covariance()
            return cov

        result = run_benchmark(
            name="CovarianceEstimation",
            scenario=f"{n_assets}_assets_{n_days}_updates",
            func=run,
            iterations=self.iterations,
        )
        result.metadata = {"n_assets": n_assets, "n_days": n_days}
        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all backtest benchmarks."""
        results = []

        # Different asset counts
        asset_counts = [10, 50] if self.quick else [10, 50, 100, 200]

        for n_assets in asset_counts:
            print(f"  Benchmarking with {n_assets} assets...")

            try:
                results.append(self.benchmark_fast_engine(n_assets))
            except Exception as e:
                print(f"    FastBacktestEngine failed: {e}")
                results.append(BenchmarkResult(
                    name="FastBacktestEngine",
                    scenario=f"{n_assets}_assets",
                    execution_time_sec=0,
                    memory_peak_mb=0,
                    iterations=0,
                    success=False,
                    error=str(e),
                ))

            try:
                results.append(self.benchmark_signal_computation(n_assets))
            except Exception as e:
                print(f"    SignalComputation failed: {e}")

            try:
                results.append(self.benchmark_covariance_estimation(n_assets))
            except Exception as e:
                print(f"    CovarianceEstimation failed: {e}")

        return results


class CacheBenchmarks:
    """Cache system benchmarks."""

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.iterations = 2 if quick else 5

    def benchmark_signal_cache(self) -> BenchmarkResult:
        """Benchmark SignalCache operations."""
        from src.backtest.cache import SignalCache
        import tempfile

        n_ops = 100 if self.quick else 500

        def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = SignalCache(cache_dir=tmpdir, enable_disk_cache=False)

                # Write
                for i in range(n_ops):
                    scores = pd.Series(np.random.randn(252), name=f"signal_{i}")
                    cache.put(
                        symbol=f"ASSET_{i}",
                        signal_name="momentum",
                        params={"lookback": 20},
                        start_date=datetime(2024, 1, 1),
                        end_date=datetime(2024, 12, 31),
                        scores=scores,
                    )

                # Read
                hits = 0
                for i in range(n_ops):
                    result = cache.get(
                        symbol=f"ASSET_{i}",
                        signal_name="momentum",
                        params={"lookback": 20},
                        start_date=datetime(2024, 1, 1),
                        end_date=datetime(2024, 12, 31),
                    )
                    if result is not None:
                        hits += 1

                return hits

        result = run_benchmark(
            name="SignalCache",
            scenario=f"{n_ops}_operations",
            func=run,
            iterations=self.iterations,
        )
        result.throughput = n_ops * 2 / result.execution_time_sec  # read + write
        result.metadata = {"n_operations": n_ops}
        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run all cache benchmarks."""
        results = []

        try:
            results.append(self.benchmark_signal_cache())
        except Exception as e:
            print(f"    SignalCache benchmark failed: {e}")

        return results


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load benchmark history from file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_history(path: Path, history: List[Dict[str, Any]]) -> None:
    """Save benchmark history to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def compare_results(current: BenchmarkSuite, history: List[Dict[str, Any]]) -> None:
    """Compare current results with history."""
    if not history:
        print("\nNo previous benchmark history to compare.")
        return

    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS RUN")
    print("=" * 60)

    prev = history[-1]
    prev_results = {r["name"] + "_" + r["scenario"]: r for r in prev["results"]}

    for result in current.results:
        key = f"{result.name}_{result.scenario}"
        if key in prev_results:
            prev_time = prev_results[key]["execution_time_sec"]
            curr_time = result.execution_time_sec

            if prev_time > 0:
                change = (curr_time - prev_time) / prev_time * 100
                status = "🟢" if change < -5 else ("🔴" if change > 5 else "⚪")
                print(f"{status} {result.name} ({result.scenario}): "
                      f"{prev_time:.3f}s -> {curr_time:.3f}s ({change:+.1f}%)")


def print_results(suite: BenchmarkSuite) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Timestamp: {suite.timestamp}")
    print(f"Git Commit: {suite.git_commit or 'N/A'}")
    print(f"Python: {suite.python_version}")
    print(f"Platform: {suite.platform}")
    print(f"Total Time: {suite.total_time_sec:.2f}s")
    print("-" * 60)

    for result in suite.results:
        status = "✓" if result.success else "✗"
        print(f"\n{status} {result.name} ({result.scenario})")
        print(f"   Time: {result.execution_time_sec:.3f}s")
        print(f"   Memory: {result.memory_peak_mb:.1f}MB")
        if result.throughput:
            print(f"   Throughput: {result.throughput:.0f} ops/sec")
        if result.error:
            print(f"   Error: {result.error}")


def main():
    parser = argparse.ArgumentParser(description="Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--save", action="store_true", help="Save results to history")
    parser.add_argument("--compare", action="store_true", help="Compare with previous")
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-ASSET PORTFOLIO PERFORMANCE BENCHMARK")
    print("=" * 60)

    start_time = time.perf_counter()
    results: List[BenchmarkResult] = []

    # Run benchmarks
    print("\n[1/2] Running Backtest Benchmarks...")
    backtest_benchmarks = BacktestBenchmarks(quick=args.quick)
    results.extend(backtest_benchmarks.run_all())

    print("\n[2/2] Running Cache Benchmarks...")
    cache_benchmarks = CacheBenchmarks(quick=args.quick)
    results.extend(cache_benchmarks.run_all())

    total_time = time.perf_counter() - start_time

    # Create suite
    suite = BenchmarkSuite(
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        results=results,
        total_time_sec=total_time,
    )

    # Print results
    print_results(suite)

    # Compare with history
    history_path = PROJECT_ROOT / "results" / "benchmark_history.json"
    history = load_history(history_path)

    if args.compare:
        compare_results(suite, history)

    # Save to history
    if args.save:
        history.append(suite.to_dict())
        # Keep last 50 runs
        if len(history) > 50:
            history = history[-50:]
        save_history(history_path, history)
        print(f"\nResults saved to {history_path}")

    # Exit with error if any benchmark failed
    failed = sum(1 for r in results if not r.success)
    if failed > 0:
        print(f"\n⚠️  {failed} benchmark(s) failed")
        sys.exit(1)

    print("\n✓ All benchmarks completed successfully")


if __name__ == "__main__":
    main()
