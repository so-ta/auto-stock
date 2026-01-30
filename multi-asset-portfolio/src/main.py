"""
Main Entry Point - CLI for Multi-Asset Portfolio System

This module provides the command-line interface for running the
portfolio optimization pipeline.

Usage:
    python -m src.main --config config/default.yaml
    python -m src.main --universe BTCUSD,ETHUSD,AAPL --seed 42
    python -m src.main --dry-run

    # Backtest mode (performance testing)
    python -m src.main --backtest
    python -m src.main --backtest --universe AAPL,MSFT,GOOGL,AMZN,META
    python -m src.main --backtest --output results.json

    # Walk-forward backtest with BacktestEngine
    python -m src.main --backtest --start 2023-01-01 --end 2023-12-31 --rebalance monthly
    python -m src.main --backtest --start 2022-01-01 --end 2023-12-31 \\
        --rebalance weekly --train-days 756 --capital 100000 --cost-bps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.backtest import BacktestConfig, BacktestEngine
from src.config.settings import Settings, load_settings_from_yaml
from src.orchestrator.pipeline import Pipeline, PipelineConfig, PipelineResult
from src.utils.logger import setup_logging
from src.utils.reproducibility import get_run_info, save_run_info


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Asset Portfolio Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file
    python -m src.main --config config/default.yaml

    # Run with specific universe
    python -m src.main --universe BTCUSD,ETHUSD,AAPL

    # Dry run (no data fetch)
    python -m src.main --dry-run --universe BTCUSD,ETHUSD

    # With specific seed for reproducibility
    python -m src.main --seed 12345 --config config/default.yaml

    # Backtest mode (performance testing with summary output)
    python -m src.main --backtest
    python -m src.main --backtest --universe AAPL,MSFT,GOOGL,AMZN,META --output results.json

    # Walk-forward backtest with BacktestEngine
    python -m src.main --backtest --start 2023-01-01 --end 2023-12-31 --rebalance monthly
    python -m src.main --backtest --start 2022-01-01 --end 2023-12-31 \\
        --rebalance weekly --train-days 756 --capital 100000

    # Fast mode (default) - 5-20x faster with vectorized computation
    python -m src.main --backtest --fast --precompute
    python -m src.main --backtest --no-fast  # Disable fast mode
    python -m src.main --backtest --no-cache  # Disable caching
    python -m src.main --backtest --cache-dir /tmp/cache  # Custom cache dir
    python -m src.main --backtest -j 4  # Use 4 parallel workers

    # Performance report generation
    python -m src.main --report --start 2010-01-01 --end 2025-01-01 \\
        --benchmarks SPY,QQQ,DIA --report-output reports/performance.html

    # Interactive dashboard
    python -m src.main --dashboard --port 8050 \\
        --start 2020-01-01 --end 2025-01-01
""",
    )

    # Backtest mode
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode with performance summary output",
    )

    # Backtest options (for BacktestEngine)
    parser.add_argument(
        "--start",
        type=str,
        metavar="DATE",
        help="Backtest start date (e.g., 2023-01-01). Enables BacktestEngine mode.",
    )
    parser.add_argument(
        "--end",
        type=str,
        metavar="DATE",
        help="Backtest end date (e.g., 2023-12-31). Enables BacktestEngine mode.",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        choices=["daily", "weekly", "monthly", "quarterly"],
        default="monthly",
        metavar="FREQ",
        help="Rebalance frequency: daily | weekly | monthly | quarterly (default: monthly)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=504,
        metavar="N",
        help="Training period in days (default: 504 = ~2 years)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        metavar="N",
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        metavar="N",
        help="Transaction cost in basis points (default: 10)",
    )

    # Performance optimization options
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        dest="fast_mode",
        help="Enable fast backtest mode with vectorized computation (default: enabled)",
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast_mode",
        help="Disable fast mode, use traditional computation",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Force signal precomputation before backtest (auto-enabled with --fast)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (recompute everything)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/backtest"),
        metavar="DIR",
        help="Cache directory for precomputed signals (default: .cache/backtest)",
    )
    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: auto, -1 for all cores)",
    )

    # Numba/GPU acceleration options
    parser.add_argument(
        "--numba",
        action="store_true",
        default=True,
        dest="use_numba",
        help="Enable Numba JIT acceleration (default: enabled)",
    )
    parser.add_argument(
        "--no-numba",
        action="store_false",
        dest="use_numba",
        help="Disable Numba JIT acceleration",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        dest="use_gpu",
        help="Enable GPU acceleration with CuPy (requires CuPy installed)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        help="Disable GPU acceleration",
    )

    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file (YAML or JSON)",
    )

    # Universe
    parser.add_argument(
        "--universe",
        "-u",
        type=str,
        help="Comma-separated list of asset symbols",
    )

    # Previous weights
    parser.add_argument(
        "--previous-weights",
        "-p",
        type=Path,
        help="Path to previous weights JSON file",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/output"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output file path (for backtest mode)",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory",
    )

    # Execution options
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip data fetching (for testing)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run ID (auto-generated if not provided)",
    )

    # Report generation
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report with benchmark comparison",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="SPY,QQQ,DIA",
        help="Benchmark tickers for comparison (comma-separated, default: SPY,QQQ,DIA)",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="reports/performance_report.html",
        help="Report output path (default: reports/performance_report.html)",
    )

    # Dashboard
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive performance dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port (default: 8050)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level",
    )

    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format",
    )

    # Output format
    parser.add_argument(
        "--output-format",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format for results",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (only log to file)",
    )

    return parser.parse_args()


# =============================================================================
# Default Backtest Universe
# =============================================================================
DEFAULT_BACKTEST_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


# =============================================================================
# Backtest Summary Functions
# =============================================================================
def print_backtest_summary(result: PipelineResult) -> None:
    """Print formatted backtest summary to console."""
    print("\n" + "=" * 70)
    print("  BACKTEST SUMMARY")
    print("=" * 70)

    # Basic info
    print(f"\n  Run ID:    {result.run_id}")
    print(f"  Status:    {result.status.value.upper()}")
    print(f"  Duration:  {result.duration_seconds:.2f} seconds")
    print(f"  Period:    {result.start_time.strftime('%Y-%m-%d %H:%M')} - {result.end_time.strftime('%Y-%m-%d %H:%M')}")

    # Portfolio Weights
    print("\n" + "-" * 70)
    print("  PORTFOLIO WEIGHTS")
    print("-" * 70)
    print(f"  {'Asset':<15} {'Weight':>12} {'Allocation':>15}")
    print(f"  {'-'*15} {'-'*12} {'-'*15}")

    for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
        bar_length = int(weight * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {asset:<15} {weight:>11.2%} {bar}")

    total_weight = sum(result.weights.values())
    print(f"  {'-'*15} {'-'*12}")
    print(f"  {'TOTAL':<15} {total_weight:>11.2%}")

    # Performance Metrics
    print("\n" + "-" * 70)
    print("  PERFORMANCE METRICS")
    print("-" * 70)

    diagnostics = result.diagnostics
    risk_metrics = diagnostics.get("risk_metrics", {})

    sharpe = risk_metrics.get("sharpe_ratio", "N/A")
    sortino = risk_metrics.get("sortino_ratio", "N/A")
    volatility = risk_metrics.get("volatility", "N/A")
    max_dd = risk_metrics.get("max_drawdown", "N/A")
    var_95 = risk_metrics.get("var_95", "N/A")
    es_95 = risk_metrics.get("expected_shortfall", "N/A")

    if isinstance(sharpe, (int, float)):
        print(f"  Sharpe Ratio:        {sharpe:>10.3f}")
    else:
        print(f"  Sharpe Ratio:        {sharpe:>10}")

    if isinstance(sortino, (int, float)):
        print(f"  Sortino Ratio:       {sortino:>10.3f}")
    else:
        print(f"  Sortino Ratio:       {sortino:>10}")

    if isinstance(volatility, (int, float)):
        print(f"  Volatility (ann.):   {volatility:>10.2%}")
    else:
        print(f"  Volatility (ann.):   {volatility:>10}")

    if isinstance(max_dd, (int, float)):
        print(f"  Max Drawdown:        {max_dd:>10.2%}")
    else:
        print(f"  Max Drawdown:        {max_dd:>10}")

    if isinstance(var_95, (int, float)):
        print(f"  VaR (95%):           {var_95:>10.4f}")
    else:
        print(f"  VaR (95%):           {var_95:>10}")

    if isinstance(es_95, (int, float)):
        print(f"  Expected Shortfall:  {es_95:>10.4f}")
    else:
        print(f"  Expected Shortfall:  {es_95:>10}")

    # Strategy Stats
    print("\n" + "-" * 70)
    print("  STRATEGY STATISTICS")
    print("-" * 70)

    strategies_evaluated = diagnostics.get("strategies_evaluated", 0)
    excluded_assets = diagnostics.get("excluded_assets", [])

    print(f"  Strategies Evaluated:  {strategies_evaluated}")
    print(f"  Assets Processed:      {len(result.weights)}")
    print(f"  Assets Excluded:       {len(excluded_assets)}")

    if excluded_assets:
        print(f"  Excluded List:         {', '.join(excluded_assets)}")

    # Fallback Status
    if result.fallback_state and result.fallback_state.active:
        print("\n" + "-" * 70)
        print("  ⚠️  FALLBACK MODE ACTIVE")
        print("-" * 70)
        print(f"  Mode:    {result.fallback_state.mode.value}")
        print(f"  Reason:  {result.fallback_state.reason}")

    # Warnings & Errors
    if result.warnings:
        print("\n" + "-" * 70)
        print("  ⚠️  WARNINGS")
        print("-" * 70)
        for i, warning in enumerate(result.warnings[:5], 1):
            print(f"  {i}. {warning[:65]}...")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more warnings")

    if result.errors:
        print("\n" + "-" * 70)
        print("  ❌ ERRORS")
        print("-" * 70)
        for i, error in enumerate(result.errors[:5], 1):
            print(f"  {i}. {error[:65]}...")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")

    print("\n" + "=" * 70 + "\n")


def save_backtest_json(result: PipelineResult, output_path: Path) -> None:
    """Save backtest result to JSON file."""
    output = {
        "run_id": result.run_id,
        "status": result.status.value,
        "timestamp": {
            "start": result.start_time.isoformat(),
            "end": result.end_time.isoformat(),
            "duration_seconds": result.duration_seconds,
        },
        "weights": result.weights,
        "performance": {
            "risk_metrics": result.diagnostics.get("risk_metrics", {}),
        },
        "statistics": {
            "strategies_evaluated": result.diagnostics.get("strategies_evaluated", 0),
            "excluded_assets": result.diagnostics.get("excluded_assets", []),
        },
        "fallback": (
            {
                "active": result.fallback_state.active,
                "mode": result.fallback_state.mode.value,
                "reason": result.fallback_state.reason,
            }
            if result.fallback_state
            else None
        ),
        "warnings": result.warnings,
        "errors": result.errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def run_backtest(args: argparse.Namespace, logger: Any) -> int:
    """Run backtest mode with summary output.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    # Require --start and --end for backtest mode
    if args.start is None or args.end is None:
        logger.error("Backtest mode requires both --start and --end flags")
        print("Error: Backtest mode requires --start and --end date arguments")
        print("Example: --backtest --start 2024-01-01 --end 2025-01-01")
        return 1

    return run_backtest_engine(args, logger)


def run_backtest_engine(args: argparse.Namespace, logger: Any) -> int:
    """Run walk-forward backtest using BacktestEngine.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        logger.error("Invalid start date format", error=str(e))
        print("Error: --start must be in YYYY-MM-DD format")
        return 1

    try:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        logger.error("Invalid end date format", error=str(e))
        print("Error: --end must be in YYYY-MM-DD format")
        return 1

    # Set universe
    if args.universe:
        universe = [s.strip() for s in args.universe.split(",")]
    else:
        universe = DEFAULT_BACKTEST_UNIVERSE
        logger.info("Using default backtest universe", universe=universe)

    # Performance options
    use_fast_mode = args.fast_mode
    precompute_signals = args.precompute or use_fast_mode
    use_cache = not args.no_cache
    cache_dir = args.cache_dir if use_cache else None
    n_jobs = args.parallel
    use_numba = args.use_numba
    use_gpu = args.use_gpu

    logger.info(
        "Starting BacktestEngine mode",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        rebalance=args.rebalance,
        train_days=args.train_days,
        capital=args.capital,
        cost_bps=args.cost_bps,
        universe=universe,
        fast_mode=use_fast_mode,
        precompute=precompute_signals,
        use_cache=use_cache,
        cache_dir=str(cache_dir) if cache_dir else None,
        n_jobs=n_jobs,
    )

    # Load settings
    if args.config and args.config.exists():
        settings = load_settings_from_yaml(args.config)
        logger.info("Loaded config", path=str(args.config))
    else:
        settings = Settings()
        if args.config:
            logger.warning("Config file not found, using defaults", path=str(args.config))

    # Create BacktestConfig with performance options
    # Note: train_period_days is not supported by BacktestConfig
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=args.rebalance,
        initial_capital=args.capital,
        transaction_cost_bps=args.cost_bps,
    )

    # Add performance options if BacktestConfig supports them
    if hasattr(backtest_config, 'use_fast_mode'):
        backtest_config.use_fast_mode = use_fast_mode
    if hasattr(backtest_config, 'precompute_signals'):
        backtest_config.precompute_signals = precompute_signals
    if hasattr(backtest_config, 'use_incremental_cov'):
        backtest_config.use_incremental_cov = use_fast_mode
    if hasattr(backtest_config, 'cache_dir'):
        backtest_config.cache_dir = cache_dir
    if hasattr(backtest_config, 'n_jobs') and n_jobs is not None:
        backtest_config.n_jobs = n_jobs
    # Numba/GPU acceleration options
    if hasattr(backtest_config, 'use_numba'):
        backtest_config.use_numba = use_numba
    if hasattr(backtest_config, 'use_gpu'):
        backtest_config.use_gpu = use_gpu

    # Display fast mode status
    if use_fast_mode:
        print(f"\n🚀 Fast mode enabled (vectorized computation, 5-20x speedup)")
        if precompute_signals:
            print(f"   Signal precomputation: ON")
        if use_cache and cache_dir:
            print(f"   Cache: {cache_dir}")
        if n_jobs:
            print(f"   Parallel workers: {n_jobs}")
        # Display Numba/GPU status
        if use_numba:
            print(f"   Numba JIT: ON (5-10x additional speedup)")
        if use_gpu:
            print(f"   GPU acceleration: ON (10-50x additional speedup)")
        print()

    # Fetch price data
    from src.data import BatchDataFetcher
    import pandas as pd
    print("   Fetching price data...")
    fetcher = BatchDataFetcher()
    prices_raw = fetcher.fetch_all_sync(
        tickers=universe,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    # Convert Dict[str, pl.DataFrame] to pd.DataFrame (columns=assets, index=timestamp)
    price_series = {}
    for symbol, df in prices_raw.items():
        pdf = df.to_pandas()
        if 'close' in pdf.columns:
            pdf = pdf.set_index('timestamp') if 'timestamp' in pdf.columns else pdf
            price_series[symbol] = pdf['close']
    prices = pd.DataFrame(price_series)
    if prices.empty:
        logger.error("No price data fetched")
        print("Error: Could not fetch price data")
        return 1
    logger.info("Fetched price data", symbols=len(prices.columns), rows=len(prices))

    # Run BacktestEngine
    try:
        engine = BacktestEngine(backtest_config)
        result = engine.run(prices=prices)
    except Exception as e:
        logger.exception("BacktestEngine failed", error=str(e))
        print(f"Error: Backtest failed - {e}")
        return 1

    # Print summary
    print_backtest_engine_summary(result, backtest_config)

    # Save JSON output if requested
    if args.output:
        save_backtest_engine_json(result, backtest_config, args.output)
        logger.info("Backtest results saved", path=str(args.output))
        print(f"Results saved to: {args.output}")

    return 0


def print_backtest_engine_summary(result: Any, config: BacktestConfig) -> None:
    """Print formatted BacktestEngine summary to console."""
    print("\n" + "=" * 70)
    print("  BACKTEST ENGINE SUMMARY")
    print("=" * 70)

    # Period info
    print(f"\n  Period:       {config.start_date.strftime('%Y-%m-%d')} - {config.end_date.strftime('%Y-%m-%d')}")
    print(f"  Rebalance:    {config.rebalance_frequency}")
    print(f"  Train Days:   {config.train_period_days}")
    print(f"  Init Capital: ${config.initial_capital:,.0f}")
    print(f"  Cost (bps):   {config.transaction_cost_bps}")

    # Performance metrics
    print("\n" + "-" * 70)
    print("  PERFORMANCE METRICS")
    print("-" * 70)

    print(f"  Total Return:      {result.total_return:>10.2%}")
    print(f"  Annualized Return: {result.annualized_return:>10.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.3f}")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"  Volatility (ann.): {result.volatility:>10.2%}")

    # Rebalance stats
    print("\n" + "-" * 70)
    print("  REBALANCE STATISTICS")
    print("-" * 70)

    print(f"  Total Rebalances:     {result.rebalance_count}")
    print(f"  Avg Turnover:         {result.avg_turnover:>10.2%}")
    print(f"  Total Costs:          ${result.total_costs:>10,.2f}")
    print(f"  Final Value:          ${result.final_value:>10,.2f}")

    # Final weights
    if hasattr(result, 'final_weights') and result.final_weights:
        print("\n" + "-" * 70)
        print("  FINAL PORTFOLIO WEIGHTS")
        print("-" * 70)
        print(f"  {'Asset':<15} {'Weight':>12}")
        print(f"  {'-'*15} {'-'*12}")

        for asset, weight in sorted(result.final_weights.items(), key=lambda x: -x[1]):
            if weight > 0.001:  # Only show meaningful weights
                print(f"  {asset:<15} {weight:>11.2%}")

    print("\n" + "=" * 70 + "\n")


def save_backtest_engine_json(result: Any, config: BacktestConfig, output_path: Path) -> None:
    """Save BacktestEngine result to JSON file."""
    output = {
        "config": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "rebalance_frequency": config.rebalance_frequency,
            "train_period_days": config.train_period_days,
            "initial_capital": config.initial_capital,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        "performance": {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "volatility": result.volatility,
        },
        "statistics": {
            "rebalance_count": result.rebalance_count,
            "avg_turnover": result.avg_turnover,
            "total_costs": result.total_costs,
            "final_value": result.final_value,
        },
        "final_weights": result.final_weights if hasattr(result, 'final_weights') else {},
        "equity_curve": result.equity_curve.to_dict() if hasattr(result, 'equity_curve') else {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def run_report_generation(args: argparse.Namespace, logger: Any) -> int:
    """Generate performance report with benchmark comparison.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    # Require --start and --end
    if args.start is None or args.end is None:
        logger.error("Report generation requires both --start and --end flags")
        print("Error: Report generation requires --start and --end date arguments")
        print("Example: --report --start 2010-01-01 --end 2025-01-01")
        return 1

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        logger.error("Invalid date format", error=str(e))
        print("Error: Dates must be in YYYY-MM-DD format")
        return 1

    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    logger.info(
        "Starting report generation",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        benchmarks=benchmarks,
        output=args.report_output,
    )

    print(f"\n📊 Generating Performance Report")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Benchmarks: {', '.join(benchmarks)}")
    print(f"   Output: {args.report_output}")
    print()

    try:
        # Import required modules
        from src.analysis.benchmark_fetcher import BenchmarkFetcher
        from src.analysis.performance_comparator import PerformanceComparator
        from src.analysis.report_generator import ReportGenerator
        from src.analysis.chart_generator import ChartGenerator

        # Run backtest to get portfolio returns
        # Set universe
        if args.universe:
            universe = [s.strip() for s in args.universe.split(",")]
        else:
            universe = DEFAULT_BACKTEST_UNIVERSE
            logger.info("Using default universe", universe=universe)

        # Load settings
        if args.config and args.config.exists():
            settings = load_settings_from_yaml(args.config)
        else:
            settings = Settings()

        # Create BacktestConfig
        backtest_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=args.rebalance,
            train_period_days=args.train_days,
            initial_capital=args.capital,
            transaction_cost_bps=args.cost_bps,
        )

        # Fetch price data for backtest
        from src.data import BatchDataFetcher
        import pandas as pd
        print("   Fetching price data...")
        price_fetcher = BatchDataFetcher()
        prices_raw = price_fetcher.fetch_all_sync(
            tickers=universe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        # Convert Dict[str, pl.DataFrame] to pd.DataFrame
        price_series = {}
        for symbol, df in prices_raw.items():
            pdf = df.to_pandas()
            if 'close' in pdf.columns:
                pdf = pdf.set_index('timestamp') if 'timestamp' in pdf.columns else pdf
                price_series[symbol] = pdf['close']
        prices = pd.DataFrame(price_series)
        if prices.empty:
            logger.error("No price data fetched for report")
            print("Error: Could not fetch price data")
            return 1

        print("   Running backtest...")
        engine = BacktestEngine(backtest_config)
        result = engine.run(prices=prices)

        # Get portfolio returns from equity curve
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            portfolio_values = result.equity_curve
            portfolio_returns = portfolio_values.pct_change().dropna()
        else:
            logger.error("Backtest result has no equity curve")
            print("Error: Could not extract portfolio returns from backtest")
            return 1

        # Fetch benchmark data
        print("   Fetching benchmark data...")
        fetcher = BenchmarkFetcher()
        benchmark_prices = fetcher.fetch_benchmarks(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            benchmarks=benchmarks,
        )
        benchmark_returns = fetcher.calculate_returns(benchmark_prices, frequency="daily")

        # Align indices
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]

        # Compare performance
        print("   Analyzing performance...")
        comparator = PerformanceComparator()
        comparison = comparator.compare(portfolio_returns, benchmark_returns)

        # Generate report
        print("   Generating report...")
        report_generator = ReportGenerator()

        # Create output directory
        output_path = Path(args.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate charts
        chart_generator = ChartGenerator()

        # Save equity comparison chart
        charts_dir = output_path.parent / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        equity_fig = chart_generator.plot_equity_comparison(
            portfolio=portfolio_returns,
            benchmarks=benchmark_returns,
            title="資産推移比較",
        )
        chart_generator.save_chart(
            equity_fig,
            charts_dir / "equity_comparison.html",
        )

        # Generate HTML report
        html = report_generator.generate_html_report(
            comparison=comparison,
            portfolio_name="Portfolio",
            start_date=args.start,
            end_date=args.end,
            output_path=str(output_path),
        )

        print(f"\n✅ Report generated: {output_path}")
        print(f"   Charts saved to: {charts_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("  PERFORMANCE SUMMARY")
        print("=" * 60)
        pm = comparison.portfolio_metrics
        print(f"  Total Return:      {pm.get('total_return', 0):>10.2%}")
        print(f"  Annualized Return: {pm.get('annualized_return', 0):>10.2%}")
        print(f"  Sharpe Ratio:      {pm.get('sharpe_ratio', 0):>10.3f}")
        print(f"  Max Drawdown:      {pm.get('max_drawdown', 0):>10.2%}")
        print(f"  Volatility (ann.): {pm.get('volatility', 0):>10.2%}")
        print("=" * 60 + "\n")

        return 0

    except ImportError as e:
        logger.error("Missing required module", error=str(e))
        print(f"Error: Missing required module - {e}")
        print("Install with: pip install plotly jinja2")
        return 1
    except Exception as e:
        logger.exception("Report generation failed", error=str(e))
        print(f"Error: Report generation failed - {e}")
        return 1


def run_dashboard(args: argparse.Namespace, logger: Any) -> int:
    """Launch interactive performance dashboard.

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Exit code (0=success, 1=failure)
    """
    # Require --start and --end
    if args.start is None or args.end is None:
        logger.error("Dashboard requires both --start and --end flags")
        print("Error: Dashboard requires --start and --end date arguments")
        print("Example: --dashboard --start 2020-01-01 --end 2025-01-01")
        return 1

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        logger.error("Invalid date format", error=str(e))
        print("Error: Dates must be in YYYY-MM-DD format")
        return 1

    # Parse benchmarks
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    logger.info(
        "Starting dashboard",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        benchmarks=benchmarks,
        port=args.port,
    )

    print(f"\n🚀 Launching Performance Dashboard")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Benchmarks: {', '.join(benchmarks)}")
    print(f"   Port: {args.port}")
    print()

    try:
        # Import required modules
        from src.analysis.benchmark_fetcher import BenchmarkFetcher
        from src.analysis.dashboard import create_dashboard

        # Run backtest to get portfolio returns
        if args.universe:
            universe = [s.strip() for s in args.universe.split(",")]
        else:
            universe = DEFAULT_BACKTEST_UNIVERSE
            logger.info("Using default universe", universe=universe)

        # Load settings
        if args.config and args.config.exists():
            settings = load_settings_from_yaml(args.config)
        else:
            settings = Settings()

        # Create BacktestConfig
        backtest_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=args.rebalance,
            train_period_days=args.train_days,
            initial_capital=args.capital,
            transaction_cost_bps=args.cost_bps,
        )

        # Fetch price data for backtest
        from src.data import BatchDataFetcher
        import pandas as pd
        print("   Fetching price data...")
        price_fetcher = BatchDataFetcher()
        prices_raw = price_fetcher.fetch_all_sync(
            tickers=universe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        # Convert Dict[str, pl.DataFrame] to pd.DataFrame
        price_series = {}
        for symbol, df in prices_raw.items():
            pdf = df.to_pandas()
            if 'close' in pdf.columns:
                pdf = pdf.set_index('timestamp') if 'timestamp' in pdf.columns else pdf
                price_series[symbol] = pdf['close']
        prices = pd.DataFrame(price_series)
        if prices.empty:
            logger.error("No price data fetched for dashboard")
            print("Error: Could not fetch price data")
            return 1

        print("   Running backtest...")
        engine = BacktestEngine(backtest_config)
        result = engine.run(prices=prices)

        # Get portfolio returns from equity curve
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            portfolio_values = result.equity_curve
            portfolio_returns = portfolio_values.pct_change().dropna()
        else:
            logger.error("Backtest result has no equity curve")
            print("Error: Could not extract portfolio returns from backtest")
            return 1

        # Fetch benchmark data
        print("   Fetching benchmark data...")
        fetcher = BenchmarkFetcher()
        benchmark_prices = fetcher.fetch_benchmarks(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            benchmarks=benchmarks,
        )
        benchmark_returns = fetcher.calculate_returns(benchmark_prices, frequency="daily")

        # Align indices
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]

        # Create and run dashboard
        print(f"\n   Starting dashboard at http://127.0.0.1:{args.port}")
        print("   Press Ctrl+C to stop\n")

        dashboard = create_dashboard(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            title="Portfolio Performance Dashboard",
        )
        dashboard.run(port=args.port)

        return 0

    except ImportError as e:
        logger.error("Missing required module", error=str(e))
        print(f"Error: Missing required module - {e}")
        print("Install with: pip install dash dash-bootstrap-components plotly")
        return 1
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        return 0
    except Exception as e:
        logger.exception("Dashboard failed", error=str(e))
        print(f"Error: Dashboard failed - {e}")
        return 1


def load_previous_weights(path: Path) -> dict[str, float]:
    """Load previous weights from JSON file."""
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    # Handle both direct weights dict and wrapped format
    if "weights" in data:
        return data["weights"]
    return data


def format_output(result: PipelineResult, format: str) -> str:
    """Format pipeline result for output."""
    if format == "json":
        return json.dumps(result.to_dict(), indent=2, default=str)

    elif format == "yaml":
        try:
            import yaml

            return yaml.dump(result.to_dict(), default_flow_style=False)
        except ImportError:
            return json.dumps(result.to_dict(), indent=2, default=str)

    elif format == "table":
        lines = [
            "=" * 60,
            f"Run ID: {result.run_id}",
            f"Status: {result.status.value}",
            f"Duration: {result.duration_seconds:.2f}s",
            "=" * 60,
            "",
            "Weights:",
            "-" * 40,
        ]

        for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {asset:20s} {weight:8.2%}")

        lines.append("-" * 40)
        lines.append(f"  {'Total':20s} {sum(result.weights.values()):8.2%}")
        lines.append("")

        if result.fallback_state and result.fallback_state.active:
            lines.append(f"Fallback Mode: {result.fallback_state.mode.value}")
            lines.append(f"Reason: {result.fallback_state.reason}")
            lines.append("")

        if result.warnings:
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        if result.errors:
            lines.append("Errors:")
            for e in result.errors:
                lines.append(f"  - {e}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    return str(result.to_dict())


def save_output(result: PipelineResult, output_dir: Path) -> Path:
    """Save pipeline result to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main output
    output_path = output_dir / f"result_{result.run_id}.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    # Save weights separately for easy access
    weights_path = output_dir / f"weights_{result.run_id}.json"
    weights_output = {
        "as_of": result.end_time.isoformat(),
        "run_id": result.run_id,
        "weights": result.weights,
        "diagnostics": {
            "fallback_mode": result.fallback_state.mode.value if result.fallback_state else None,
            "status": result.status.value,
        },
    }
    with open(weights_path, "w") as f:
        json.dump(weights_output, f, indent=2, default=str)

    return output_path


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(
        log_file=args.log_dir / "system.log" if not args.quiet else None,
        json_format=args.json_logs,
        level=args.log_level,
    )

    logger = structlog.get_logger("main")
    logger.info(
        "Starting Multi-Asset Portfolio System",
        config=str(args.config) if args.config else None,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    try:
        # Check for report mode
        if args.report:
            return run_report_generation(args, logger)

        # Check for dashboard mode
        if args.dashboard:
            return run_dashboard(args, logger)

        # Check for backtest mode
        if args.backtest:
            return run_backtest(args, logger)

        # Load settings
        if args.config and args.config.exists():
            settings = load_settings_from_yaml(args.config)
            logger.info("Loaded config", path=str(args.config))
        else:
            settings = Settings()
            if args.config:
                logger.warning("Config file not found, using defaults", path=str(args.config))

        # Override settings with CLI args
        if args.universe:
            universe = [s.strip() for s in args.universe.split(",")]
        else:
            universe = settings.universe

        if not universe:
            logger.error("No assets specified. Use --universe or set in config.")
            return 1

        # Load previous weights
        previous_weights = {}
        if args.previous_weights and args.previous_weights.exists():
            previous_weights = load_previous_weights(args.previous_weights)
            logger.info(
                "Loaded previous weights",
                path=str(args.previous_weights),
                asset_count=len(previous_weights),
            )

        # Create pipeline config
        pipeline_config = PipelineConfig(
            run_id=args.run_id,
            seed=args.seed,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
        )

        # Run pipeline
        pipeline = Pipeline(settings=settings, config=pipeline_config)
        result = pipeline.run(
            universe=universe,
            previous_weights=previous_weights,
        )

        # Save output
        output_path = save_output(result, args.output_dir)
        logger.info("Results saved", path=str(output_path))

        # Print output
        if not args.quiet:
            print(format_output(result, args.output_format))

        # Return appropriate exit code
        if result.status.value == "failed":
            return 1
        elif result.status.value == "fallback":
            return 2  # Partial success
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Fatal error", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
