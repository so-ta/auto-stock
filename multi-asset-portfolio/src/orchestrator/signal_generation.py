"""
Signal Generation Module - Handles signal computation and strategy evaluation.

Extracted from pipeline.py for better modularity (QA-003-P1).
This module handles:
1. Signal generation with parameter optimization
2. Strategy evaluation on test data (delegated to strategy_evaluation.py)
3. Signal quality assessment
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.config.resource_config import get_current_resource_config

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from src.config.settings import Settings
    from src.strategy.gate_checker import StrategyMetrics
    from src.utils.logger import AuditLogger

# Re-export StrategyEvaluator from strategy_evaluation module
from src.orchestrator.strategy_evaluation import StrategyEvaluator

# Lazy import for DataFrame conversion utilities (task_013_6)
_ensure_pandas = None

def _get_ensure_pandas():
    """Lazy import of ensure_pandas to avoid circular imports"""
    global _ensure_pandas
    if _ensure_pandas is None:
        from src.utils.dataframe_utils import ensure_pandas
        _ensure_pandas = ensure_pandas
    return _ensure_pandas

logger = structlog.get_logger(__name__)


# =============================================================================
# Module-level worker function for ProcessPoolExecutor (task_039_2)
# =============================================================================
def _process_single_asset_worker(
    symbol: str,
    df_dict: dict,  # DataFrame as dict for pickle compatibility
    signal_names: list[str],
    train_days: int,
    test_days: int,
    seed: int,
) -> tuple[str, dict[str, Any] | None, int, int]:
    """Process a single asset for signal generation (ProcessPool worker).

    This is a module-level function for ProcessPoolExecutor compatibility.
    Uses a deterministic seed for reproducibility.

    Args:
        symbol: Asset symbol
        df_dict: Price data as dict (for pickle)
        signal_names: List of signal names to compute
        train_days: Training period days
        test_days: Test period days
        seed: Random seed for this asset (deterministic)

    Returns:
        Tuple of (symbol, signals_dict, computed_count, optimized_count)
    """
    import pandas as pd
    from itertools import product
    from src.backtest.numba_compute import spearmanr_numba

    # Import SignalRegistry in worker process
    from src.signals import SignalRegistry

    # Set seed for reproducibility
    rng = np.random.RandomState(seed)

    # Convert dict back to DataFrame
    df = pd.DataFrame(df_dict)

    # Ensure index is datetime
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    # Split into train/test periods
    total_rows = len(df)
    test_rows = min(test_days, total_rows // 3)
    train_end_idx = total_rows - test_rows

    if train_end_idx < 50:
        return (symbol, None, 0, 0)

    train_data = df.iloc[:train_end_idx].copy()

    asset_signals: dict[str, Any] = {}
    computed = 0
    optimized = 0

    def evaluate_signal_quality(scores: pd.Series, prices: pd.Series) -> float:
        """Evaluate signal quality using correlation with future returns."""
        returns = prices.pct_change().shift(-1)
        valid_mask = ~(scores.isna() | returns.isna())
        valid_scores = scores[valid_mask]
        valid_returns = returns[valid_mask]

        if len(valid_scores) < 20:
            return float("-inf")

        corr = spearmanr_numba(
            valid_scores.values.astype(np.float64),
            valid_returns.values.astype(np.float64)
        )
        return corr if not pd.isna(corr) else float("-inf")

    for signal_name in signal_names:
        try:
            signal_cls = SignalRegistry.get(signal_name)

            # Get parameter specifications
            specs = signal_cls.parameter_specs()
            searchable_specs = [s for s in specs if s.searchable]

            # Build default parameters
            default_params = {s.name: s.default for s in specs}

            if not searchable_specs:
                # No searchable params - compute with defaults
                signal = signal_cls(**default_params)
                result = signal.compute(train_data)

                signal_entry = {
                    "result": result,
                    "params": default_params,
                    "optimized": False,
                }
                asset_signals[signal_name] = signal_entry
                computed += 1
            else:
                # Grid search over searchable parameters
                best_params = default_params.copy()
                best_score = float("-inf")

                # Generate search grid
                param_grids = []
                param_names = []
                for spec in searchable_specs:
                    grid = spec.search_range()
                    if grid is not None and len(grid) > 0:
                        param_grids.append(grid)
                        param_names.append(spec.name)

                if param_grids:
                    # Limit combinations to prevent memory explosion
                    max_combinations = 100
                    grid_product = list(product(*param_grids))

                    if len(grid_product) > max_combinations:
                        # Random sampling with deterministic seed
                        indices = rng.choice(
                            len(grid_product),
                            max_combinations,
                            replace=False,
                        )
                        grid_product = [grid_product[i] for i in indices]

                    for values in grid_product:
                        params = default_params.copy()
                        for name, value in zip(param_names, values):
                            params[name] = value

                        try:
                            signal = signal_cls(**params)
                            result = signal.compute(train_data)

                            # Evaluate signal quality
                            score = evaluate_signal_quality(
                                result.scores,
                                train_data["close"],
                            )

                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                        except Exception:
                            continue

                # Compute final result with best parameters
                signal = signal_cls(**best_params)
                result = signal.compute(train_data)

                signal_entry = {
                    "result": result,
                    "params": best_params,
                    "optimized": True,
                    "optimization_score": (
                        best_score if best_score > float("-inf") else None
                    ),
                }
                asset_signals[signal_name] = signal_entry
                computed += 1
                optimized += 1

        except Exception:
            continue

    return (symbol, asset_signals, computed, optimized)


@dataclass
class SignalGenerationResult:
    """Result of signal generation step."""

    signals: dict[str, dict[str, Any]]
    total_computed: int
    total_optimized: int


@dataclass
class StrategyEvaluationSummary:
    """Summary of strategy evaluation results."""

    evaluations: list[Any]
    total_evaluated: int
    total_errors: int


class SignalGenerator:
    """
    Handles signal generation for the pipeline (DEPRECATED).

    .. deprecated:: 2.0.0
        SignalGenerator is deprecated and will be removed in a future version.
        Use SignalPrecomputer instead, which pre-computes all 64 registered signals
        and provides 40x faster backtest performance.

    Responsible for:
    - Loading signals from SignalRegistry
    - Parameter optimization on train data
    - Computing signal scores
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
        enable_cache: bool = False,  # Cache disabled - not effective for backtest
    ) -> None:
        """
        Initialize SignalGenerator.

        .. deprecated:: 2.0.0
            Use SignalPrecomputer instead for better performance.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
            enable_cache: Deprecated, ignored (cache removed in v2.1)
        """
        import warnings
        warnings.warn(
            "SignalGenerator is deprecated and will be removed in a future version. "
            "Use SignalPrecomputer instead for all 64 signals and 40x faster backtests.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="signal_generator")

        # Signal quality evaluation cache (memoization)
        self._quality_cache: dict[str, float] = {}
        self._quality_cache_max_size: int = 10000
        self._quality_cache_hits: int = 0
        self._quality_cache_misses: int = 0

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def generate_signals(
        self,
        raw_data: dict[str, "pl.DataFrame"],
        excluded_assets: list[str],
    ) -> SignalGenerationResult:
        """
        Generate signals for each asset.

        This method:
        1. Loads all registered signals from SignalRegistry
        2. For each asset × signal combination:
           - Performs parameter optimization on train data (searchable params only)
           - Computes signal scores with optimized parameters
        3. Returns results

        Args:
            raw_data: Dictionary mapping symbol to DataFrame
            excluded_assets: List of assets to exclude

        Returns:
            SignalGenerationResult with signals data
        """
        import pandas as pd

        # Import signals module to ensure all signals are registered
        from src.signals import SignalRegistry

        signals: dict[str, dict[str, Any]] = {}

        # Get valid (non-excluded) assets
        valid_symbols = [
            symbol
            for symbol in raw_data.keys()
            if symbol not in excluded_assets
        ]

        if not valid_symbols:
            self._logger.warning("No valid assets for signal generation")
            return SignalGenerationResult(
                signals={},
                total_computed=0,
                total_optimized=0,
            )

        # Get all registered signals
        signal_names = SignalRegistry.list_all()

        if not signal_names:
            self._logger.warning("No signals registered in SignalRegistry")
            return SignalGenerationResult(
                signals={},
                total_computed=0,
                total_optimized=0,
            )

        # Get train/test split configuration
        train_days = self.settings.walk_forward.train_period_days
        test_days = self.settings.walk_forward.test_period_days

        self._logger.info(
            "Starting signal generation (parallel)",
            assets=len(valid_symbols),
            signals=len(signal_names),
            train_days=train_days,
            test_days=test_days,
        )

        # Parallel processing of assets (task_013_5)
        # Each asset gets a deterministic seed based on symbol hash for reproducibility
        base_seed = getattr(self._settings, "seed", 42) if hasattr(self._settings, "seed") else 42

        max_workers = self._get_workers()
        self._logger.debug(f"Using {max_workers} ProcessPool workers for parallel signal generation")

        # Prepare data for ProcessPool (convert to dict for pickle compatibility)
        # task_039_2: Use module-level worker function for ProcessPool compatibility
        prepared_data = {}
        for symbol in valid_symbols:
            df = raw_data[symbol]
            # Convert polars to pandas if needed
            df = _get_ensure_pandas()(df)
            # Convert to dict for pickle
            prepared_data[symbol] = df.to_dict()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each symbol using module-level worker function
            futures = {
                executor.submit(
                    _process_single_asset_worker,
                    symbol,
                    prepared_data[symbol],  # Dict for pickle
                    signal_names,
                    train_days,
                    test_days,
                    base_seed + hash(symbol) % (2**31 - 1),  # Deterministic seed per symbol
                ): symbol
                for symbol in valid_symbols
            }

            # Collect results
            total_computed = 0
            total_optimized = 0

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        result_symbol, sym_signals, computed, optimized = result
                        if sym_signals is not None:
                            signals[result_symbol] = sym_signals
                            total_computed += computed
                            total_optimized += optimized
                except Exception as e:
                    self._logger.warning(
                        f"Parallel processing failed for {symbol}",
                        error=str(e),
                    )

        # Log audit info
        if self._audit_logger:
            self._audit_logger.log_signal_generation(
                total_assets=len(signals),
                total_signals=total_computed,
                optimized_signals=total_optimized,
                signal_names=signal_names,
            )

        self._logger.info(
            "Signal generation completed (parallel)",
            assets=len(signals),
            total_signals=total_computed,
            optimized_signals=total_optimized,
            workers=max_workers,
        )

        return SignalGenerationResult(
            signals=signals,
            total_computed=total_computed,
            total_optimized=total_optimized,
        )

    def _get_workers(self) -> int:
        """Get optimal number of workers for ProcessPool parallel processing.

        Returns:
            Number of workers from ResourceConfig (auto-detected based on system resources)
        """
        rc = get_current_resource_config()
        # Use ResourceConfig's max_workers (auto-detected or configured)
        return rc.max_workers

    def _process_single_asset(
        self,
        symbol: str,
        df: "pl.DataFrame",
        signal_names: list[str],
        train_days: int,
        test_days: int,
        SignalRegistry: Any,
        seed: int,
    ) -> tuple[dict[str, Any], int, int] | None:
        """Process a single asset for signal generation (task_013_5).

        This method is designed to be called in parallel.
        Uses a deterministic seed for reproducibility.

        Args:
            symbol: Asset symbol
            df: Price data DataFrame
            signal_names: List of signal names to compute
            train_days: Training period days
            test_days: Test period days
            SignalRegistry: Signal registry class
            seed: Random seed for this asset (deterministic)

        Returns:
            Tuple of (signals_dict, computed_count, optimized_count) or None if failed
        """
        # Set seed for reproducibility in this thread
        rng = np.random.RandomState(seed)

        # Convert polars DataFrame to pandas if needed (task_013_6: 変換削減)
        df = _get_ensure_pandas()(df)

        # Ensure index is datetime
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Split into train/test periods
        total_rows = len(df)
        test_rows = min(test_days, total_rows // 3)
        train_end_idx = total_rows - test_rows

        if train_end_idx < 50:
            self._logger.debug(f"Insufficient data for {symbol}: {total_rows} rows")
            return None

        train_data = df.iloc[:train_end_idx].copy()

        asset_signals: dict[str, Any] = {}
        computed = 0
        optimized = 0

        for signal_name in signal_names:
            try:
                signal_cls = SignalRegistry.get(signal_name)

                # Get parameter specifications
                specs = signal_cls.parameter_specs()
                searchable_specs = [s for s in specs if s.searchable]

                # Build default parameters
                default_params = {s.name: s.default for s in specs}

                if not searchable_specs:
                    # No searchable params - compute with defaults
                    signal = signal_cls(**default_params)
                    result = signal.compute(train_data)

                    signal_entry = {
                        "result": result,
                        "params": default_params,
                        "optimized": False,
                    }
                    asset_signals[signal_name] = signal_entry
                    computed += 1
                else:
                    # Grid search over searchable parameters
                    best_params = default_params.copy()
                    best_score = float("-inf")

                    # Generate search grid
                    param_grids = []
                    param_names = []
                    for spec in searchable_specs:
                        grid = spec.search_range()
                        if grid is not None and len(grid) > 0:
                            param_grids.append(grid)
                            param_names.append(spec.name)

                    if param_grids:
                        # Limit combinations to prevent memory explosion
                        max_combinations = 100
                        grid_product = list(product(*param_grids))

                        if len(grid_product) > max_combinations:
                            # Random sampling with deterministic seed
                            indices = rng.choice(
                                len(grid_product),
                                max_combinations,
                                replace=False,
                            )
                            grid_product = [grid_product[i] for i in indices]

                        for values in grid_product:
                            params = default_params.copy()
                            for name, value in zip(param_names, values):
                                params[name] = value

                            try:
                                signal = signal_cls(**params)
                                result = signal.compute(train_data)

                                # Evaluate signal quality
                                score = self._evaluate_signal_quality(
                                    result.scores,
                                    train_data["close"],
                                )

                                if score > best_score:
                                    best_score = score
                                    best_params = params.copy()
                            except Exception:
                                continue

                    # Compute final result with best parameters
                    signal = signal_cls(**best_params)
                    result = signal.compute(train_data)

                    signal_entry = {
                        "result": result,
                        "params": best_params,
                        "optimized": True,
                        "optimization_score": (
                            best_score if best_score > float("-inf") else None
                        ),
                    }
                    asset_signals[signal_name] = signal_entry
                    computed += 1
                    optimized += 1

            except Exception as e:
                self._logger.debug(
                    f"Signal {signal_name} failed for {symbol}: {e}"
                )
                continue

        return (asset_signals, computed, optimized)

    def _hash_array(self, arr: "np.ndarray") -> str:
        """
        Generate hash for numpy array.

        Args:
            arr: Numpy array to hash

        Returns:
            MD5 hash string
        """
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _manage_quality_cache_size(self) -> None:
        """Manage cache size (remove oldest entries when full)."""
        if len(self._quality_cache) > self._quality_cache_max_size:
            # Remove first 10% of entries
            n_remove = self._quality_cache_max_size // 10
            keys_to_remove = list(self._quality_cache.keys())[:n_remove]
            for key in keys_to_remove:
                self._quality_cache.pop(key, None)

    @property
    def quality_cache_stats(self) -> dict[str, Any]:
        """Get quality cache statistics."""
        total = self._quality_cache_hits + self._quality_cache_misses
        return {
            "cache_size": len(self._quality_cache),
            "hits": self._quality_cache_hits,
            "misses": self._quality_cache_misses,
            "hit_rate": self._quality_cache_hits / total if total > 0 else 0.0,
        }

    def _evaluate_signal_quality(
        self,
        scores: "pd.Series",
        prices: "pd.Series",
    ) -> float:
        """
        Evaluate signal quality using correlation with future returns.

        Memoized version: caches results based on input array hashes
        to avoid redundant computations. Guarantees 100% precision
        (identical inputs always return identical outputs).

        Args:
            scores: Signal scores series
            prices: Close price series

        Returns:
            Evaluation score (higher is better)
        """
        import pandas as pd
        from src.backtest.numba_compute import spearmanr_numba

        # Generate cache key from input arrays
        scores_hash = self._hash_array(scores.values)
        prices_hash = self._hash_array(prices.values)
        cache_key = f"{scores_hash}_{prices_hash}"

        # Check cache
        if cache_key in self._quality_cache:
            self._quality_cache_hits += 1
            return self._quality_cache[cache_key]

        self._quality_cache_misses += 1

        # Calculate 1-day forward returns
        returns = prices.pct_change().shift(-1)

        # Align and drop NaN
        valid_mask = ~(scores.isna() | returns.isna())
        valid_scores = scores[valid_mask]
        valid_returns = returns[valid_mask]

        if len(valid_scores) < 20:
            result = float("-inf")
        else:
            # Use Spearman correlation (rank-based, robust to outliers)
            # Numba JIT version for 6.3x speedup
            corr = spearmanr_numba(
                valid_scores.values.astype(np.float64),
                valid_returns.values.astype(np.float64)
            )
            result = corr if not pd.isna(corr) else float("-inf")

        # Store in cache
        self._quality_cache[cache_key] = result
        self._manage_quality_cache_size()

        return result


# Note: StrategyEvaluator class has been moved to strategy_evaluation.py
# and is re-exported at the top of this file for backward compatibility
