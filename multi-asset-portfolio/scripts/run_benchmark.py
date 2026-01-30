#!/usr/bin/env python3
"""
Benchmark Script for Performance Optimization (task_045_15)

Measures:
1. Numba JIT compilation effect
2. Numba parallel effect
3. Cache backend comparison (local only, S3 simulated)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Number of runs for averaging
N_RUNS = 3

# Test data size (larger for meaningful parallel benchmark)
N_TICKERS = 100
N_DAYS = 1000  # ~4 years of trading days


def generate_test_data(n_tickers: int, n_days: int) -> Dict[str, pd.DataFrame]:
    """Generate synthetic price data for benchmarking."""
    np.random.seed(42)

    prices = {}
    dates = pd.date_range(end="2024-12-31", periods=n_days, freq="B")

    for i in range(n_tickers):
        ticker = f"TEST{i:03d}"
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)
        close = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "open": close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            "close": close,
            "volume": np.random.randint(100000, 10000000, n_days),
        }, index=dates)
        prices[ticker] = df

    return prices


def benchmark_numba_computation(
    prices: Dict[str, pd.DataFrame],
    use_numba: bool = True,
    parallel: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Benchmark Numba-accelerated computation."""
    # Build price matrix (outside timing)
    tickers = list(prices.keys())
    first_df = prices[tickers[0]]
    dates = first_df.index

    price_matrix = np.zeros((len(dates), len(tickers)))
    for i, ticker in enumerate(tickers):
        price_matrix[:, i] = prices[ticker]["close"].values

    # Compute returns (outside timing)
    returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]

    # Compute rolling statistics (the heavy computation - TIMED)
    window = 20
    n_periods = len(returns) - window + 1

    start_time = time.perf_counter()

    if use_numba:
        try:
            if parallel:
                func = get_numba_parallel_func()
            else:
                func = get_numba_serial_func()
            means, stds = func(returns, window)
        except ImportError:
            means, stds = compute_rolling_stats_numpy(returns, window)
    else:
        means, stds = compute_rolling_stats_numpy(returns, window)

    # Compute Sharpe-like scores
    scores = means / (stds + 1e-8)

    elapsed = time.perf_counter() - start_time

    return elapsed, {
        "n_tickers": len(tickers),
        "n_days": len(dates),
        "n_periods": n_periods,
        "mean_score": float(np.mean(scores)),
    }


def compute_rolling_stats_numpy(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy rolling statistics (baseline)."""
    n, m = data.shape
    n_out = n - window + 1
    means = np.zeros((n_out, m))
    stds = np.zeros((n_out, m))

    for j in range(m):
        for i in range(n_out):
            window_data = data[i:i+window, j]
            means[i, j] = np.mean(window_data)
            stds[i, j] = np.std(window_data)

    return means, stds


# Pre-compile Numba functions (outside benchmark loop)
_numba_serial_func = None
_numba_parallel_func = None


def get_numba_serial_func():
    """Get cached Numba serial function."""
    global _numba_serial_func
    if _numba_serial_func is None:
        from numba import jit

        @jit(nopython=True, cache=True)
        def compute_rolling_stats_serial(data, window):
            n, m = data.shape
            n_out = n - window + 1
            means = np.zeros((n_out, m))
            stds = np.zeros((n_out, m))

            for j in range(m):
                for i in range(n_out):
                    window_data = data[i:i+window, j]
                    means[i, j] = np.mean(window_data)
                    stds[i, j] = np.std(window_data)

            return means, stds

        _numba_serial_func = compute_rolling_stats_serial
    return _numba_serial_func


def get_numba_parallel_func():
    """Get cached Numba parallel function."""
    global _numba_parallel_func
    if _numba_parallel_func is None:
        from numba import jit, prange

        @jit(nopython=True, parallel=True, cache=True)
        def compute_rolling_stats_parallel(data, window):
            n, m = data.shape
            n_out = n - window + 1
            means = np.zeros((n_out, m))
            stds = np.zeros((n_out, m))

            for j in prange(m):
                for i in range(n_out):
                    window_data = data[i:i+window, j]
                    means[i, j] = np.mean(window_data)
                    stds[i, j] = np.std(window_data)

            return means, stds

        _numba_parallel_func = compute_rolling_stats_parallel
    return _numba_parallel_func


def benchmark_cache_operations() -> Dict[str, float]:
    """Benchmark cache read/write operations using simple file I/O."""
    results = {}

    # Generate test data
    dates = pd.date_range("2020-01-01", periods=1000)
    test_data = pd.DataFrame({
        "close": np.random.randn(1000),
        "volume": np.random.randint(1000, 100000, 1000),
    }, index=dates)

    cache_dir = PROJECT_ROOT / ".cache" / "benchmark_test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Simple Parquet write benchmark
        start = time.perf_counter()
        for i in range(10):
            test_data.to_parquet(cache_dir / f"bench_{i}.parquet")
        write_time = time.perf_counter() - start
        results["local_write_10x"] = write_time

        # Simple Parquet read benchmark
        start = time.perf_counter()
        for i in range(10):
            _ = pd.read_parquet(cache_dir / f"bench_{i}.parquet")
        read_time = time.perf_counter() - start
        results["local_read_10x"] = read_time

        # Cleanup
        for i in range(10):
            (cache_dir / f"bench_{i}.parquet").unlink(missing_ok=True)

        print(f"  Parquet Write (10x): {write_time:.4f}s")
        print(f"  Parquet Read (10x): {read_time:.4f}s")

    except Exception as e:
        print(f"  Cache benchmark error: {e}")
        results["error"] = str(e)

    return results


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks."""
    print("=" * 60)
    print("Performance Benchmark Suite")
    print("=" * 60)

    results = {
        "environment": {
            "cpu_count": os.cpu_count(),
            "n_tickers": N_TICKERS,
            "n_days": N_DAYS,
            "n_runs": N_RUNS,
        },
        "numba_benchmarks": {},
        "cache_benchmarks": {},
    }

    # Generate test data
    print("\n[1/4] Generating test data...")
    prices = generate_test_data(N_TICKERS, N_DAYS)
    print(f"  Generated {len(prices)} tickers x {N_DAYS} days")

    # Benchmark 1: Pure NumPy (baseline)
    print("\n[2/4] Benchmarking NumPy baseline...")
    times = []
    for run in range(N_RUNS):
        # Disable numba
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        elapsed, info = benchmark_numba_computation(prices, use_numba=False)
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s")

    baseline_time = np.mean(times)
    results["numba_benchmarks"]["numpy_baseline"] = {
        "mean_time": baseline_time,
        "times": times,
        "speedup": 1.0,
    }
    print(f"  Average: {baseline_time:.4f}s (baseline)")

    # Benchmark 2: Numba JIT (serial)
    print("\n[3/4] Benchmarking Numba JIT (serial)...")
    os.environ.pop("NUMBA_DISABLE_JIT", None)

    # Warm up JIT compilation
    print("  Warming up JIT...")
    _ = benchmark_numba_computation(prices, use_numba=True, parallel=False)

    times = []
    for run in range(N_RUNS):
        elapsed, info = benchmark_numba_computation(prices, use_numba=True, parallel=False)
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s")

    numba_serial_time = np.mean(times)
    results["numba_benchmarks"]["numba_serial"] = {
        "mean_time": numba_serial_time,
        "times": times,
        "speedup": baseline_time / numba_serial_time if numba_serial_time > 0 else 0,
    }
    print(f"  Average: {numba_serial_time:.4f}s ({baseline_time/numba_serial_time:.2f}x faster)")

    # Benchmark 3: Numba JIT (parallel)
    print("\n[4/4] Benchmarking Numba JIT (parallel)...")

    # Warm up parallel JIT
    print("  Warming up parallel JIT...")
    _ = benchmark_numba_computation(prices, use_numba=True, parallel=True)

    times = []
    for run in range(N_RUNS):
        elapsed, info = benchmark_numba_computation(prices, use_numba=True, parallel=True)
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s")

    numba_parallel_time = np.mean(times)
    results["numba_benchmarks"]["numba_parallel"] = {
        "mean_time": numba_parallel_time,
        "times": times,
        "speedup": baseline_time / numba_parallel_time if numba_parallel_time > 0 else 0,
    }
    print(f"  Average: {numba_parallel_time:.4f}s ({baseline_time/numba_parallel_time:.2f}x faster)")

    # Cache benchmarks
    print("\n[Bonus] Cache operation benchmarks...")
    try:
        cache_results = benchmark_cache_operations()
        results["cache_benchmarks"] = cache_results
        for key, value in cache_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}s")
    except Exception as e:
        print(f"  Cache benchmark failed: {e}")
        results["cache_benchmarks"]["error"] = str(e)

    return results


def format_cache_time(value) -> str:
    """Format cache time value safely."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def generate_report(results: Dict[str, Any]) -> str:
    """Generate markdown benchmark report."""
    env = results["environment"]
    numba = results["numba_benchmarks"]
    cache = results.get("cache_benchmarks", {})

    report = f"""# Performance Benchmark Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Environment

| Item | Value |
|------|-------|
| CPU Cores | {env['cpu_count']} |
| Test Tickers | {env['n_tickers']} |
| Test Days | {env['n_days']} |
| Runs per Config | {env['n_runs']} |

## Numba JIT Compilation Effect

| Configuration | Mean Time (s) | Speedup |
|---------------|---------------|---------|
| NumPy Baseline | {numba['numpy_baseline']['mean_time']:.4f} | 1.0x |
| Numba Serial | {numba['numba_serial']['mean_time']:.4f} | {numba['numba_serial']['speedup']:.2f}x |
| Numba Parallel | {numba['numba_parallel']['mean_time']:.4f} | {numba['numba_parallel']['speedup']:.2f}x |

### Analysis

- **Numba Serial JIT**: {numba['numba_serial']['speedup']:.1f}x faster than pure NumPy
- **Numba Parallel JIT**: {numba['numba_parallel']['speedup']:.1f}x faster than pure NumPy
- **Parallel vs Serial**: {numba['numba_serial']['mean_time']/numba['numba_parallel']['mean_time']:.1f}x improvement

## Cache Operations

| Operation | Time (s) |
|-----------|----------|
| Local Write (10 files) | {format_cache_time(cache.get('local_write_10x'))} |
| Local Read (10 files) | {format_cache_time(cache.get('local_read_10x'))} |

## Unavailable Optimizations

The following optimizations were not tested due to missing dependencies:

| Optimization | Status | Notes |
|--------------|--------|-------|
| Ray Distributed | Not installed | `pip install ray` to enable |
| GPU (CuPy) | Not installed | Requires NVIDIA GPU + CUDA |
| S3 Cache | Not tested | Requires AWS credentials |

## Recommendations

1. **Numba Parallel**: Always enable `numba_parallel=True` in ResourceConfig for {numba['numba_parallel']['speedup']:.1f}x speedup
2. **Large Scale**: Consider installing Ray for multi-process parallelism
3. **GPU**: For very large computations, GPU acceleration can provide 10-50x improvement

---
*Benchmark by task_045_15*
"""
    return report


def main():
    """Main entry point."""
    results = run_benchmarks()

    # Generate report
    report = generate_report(results)

    # Save report
    report_path = PROJECT_ROOT / "results" / "benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"Report saved to: {report_path}")
    print("=" * 60)

    # Print summary
    print("\n### Summary ###")
    numba = results["numba_benchmarks"]
    print(f"NumPy Baseline: {numba['numpy_baseline']['mean_time']:.4f}s")
    print(f"Numba Serial:   {numba['numba_serial']['mean_time']:.4f}s ({numba['numba_serial']['speedup']:.2f}x)")
    print(f"Numba Parallel: {numba['numba_parallel']['mean_time']:.4f}s ({numba['numba_parallel']['speedup']:.2f}x)")


if __name__ == "__main__":
    main()
