#!/usr/bin/env python3
"""
Bayesian Optimization Script - 100 trials with 10-year data

Execute bayesian hyperparameter optimization to find optimal parameters
for the multi-asset portfolio system.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta import BayesianOptimizer, OptimizerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_historical_data(
    symbols: list[str],
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for multiple symbols.

    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with MultiIndex columns (symbol, ohlcv)
    """
    logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")

    all_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            df = df.reset_index()
            df.columns = df.columns.str.lower()

            # Rename 'date' to 'timestamp' for consistency
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            all_data[symbol] = df
            logger.info(f"  {symbol}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")

    # Combine into a single DataFrame with 'close' prices
    # Create a DataFrame with all close prices
    close_data = {}
    for symbol, df in all_data.items():
        close_data[symbol] = df.set_index('timestamp')['close']

    combined = pd.DataFrame(close_data)
    combined = combined.dropna()

    # Add 'close' column as equal-weighted average for simple backtest
    # This allows the optimizer to use a portfolio-level backtest
    combined['close'] = combined.mean(axis=1)

    # Normalize each column to start at 100 for comparable returns
    for col in combined.columns:
        if col != 'close':
            combined[col] = combined[col] / combined[col].iloc[0] * 100

    combined['close'] = combined.drop('close', axis=1).mean(axis=1)

    logger.info(f"Combined data: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


def run_optimization():
    """Run the Bayesian optimization."""
    # Configuration
    UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "EFA", "EEM", "IWM", "VNQ"]
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    N_CALLS = 100
    N_RANDOM_STARTS = 20
    OUTPUT_PATH = project_root / "results" / "bayesian_optimization_result.json"

    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION - 100 TRIALS")
    logger.info("=" * 60)
    logger.info(f"Universe: {UNIVERSE}")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Trials: {N_CALLS}")
    logger.info("=" * 60)

    # Fetch data
    logger.info("\n[Step 1/3] Fetching 10-year historical data...")
    train_data = fetch_historical_data(UNIVERSE, START_DATE, END_DATE)

    if len(train_data) < 252:
        logger.error("Insufficient data for optimization")
        return None

    logger.info(f"Data loaded: {len(train_data)} days, {len(train_data.columns)} assets")

    # Create optimizer
    logger.info("\n[Step 2/3] Configuring Bayesian optimizer...")
    config = OptimizerConfig(
        n_calls=N_CALLS,
        n_random_starts=N_RANDOM_STARTS,
        acq_func="EI",
        cv_folds=5,
        overfitting_penalty=0.1,
        early_stop_patience=15,
        random_state=42,
        verbose=True,
        n_jobs=1,  # Single thread for stability
    )

    optimizer = BayesianOptimizer(config=config)
    logger.info(f"  n_calls: {config.n_calls}")
    logger.info(f"  n_random_starts: {config.n_random_starts}")
    logger.info(f"  cv_folds: {config.cv_folds}")
    logger.info(f"  acq_func: {config.acq_func}")

    # Run optimization
    logger.info("\n[Step 3/3] Running Bayesian optimization...")
    logger.info("This may take a while (expect 30-60 minutes)...")

    try:
        result = optimizer.optimize(
            train_data=train_data,
            universe=list(train_data.columns),
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Output results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Score (Sharpe): {result.best_score:.4f}")
    logger.info(f"Iterations: {result.n_iterations}")
    logger.info(f"Elapsed Time: {result.elapsed_time:.1f} seconds")

    logger.info("\nBest Parameters:")
    for param, value in result.best_params.items():
        logger.info(f"  {param}: {value}")

    logger.info("\nParameter Importance:")
    for param, importance in sorted(
        result.param_importance.items(), key=lambda x: -x[1]
    ):
        logger.info(f"  {param}: {importance:.4f}")

    # Save results
    output_data = {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "convergence_curve": result.convergence_curve,
        "param_importance": result.param_importance,
        "n_iterations": result.n_iterations,
        "elapsed_time": result.elapsed_time,
        "cv_scores": result.cv_scores,
        "metadata": {
            "universe": UNIVERSE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "n_calls": N_CALLS,
            "n_random_starts": N_RANDOM_STARTS,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output_data = convert_numpy(output_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    result = run_optimization()
    if result:
        print("\n" + "=" * 60)
        print("SUCCESS: Optimization completed successfully")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Optimization did not complete")
        print("=" * 60)
        sys.exit(1)
