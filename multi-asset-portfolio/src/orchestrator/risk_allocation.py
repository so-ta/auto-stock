"""
Risk Allocation Module - Handles risk estimation and asset allocation.

Extracted from pipeline.py for better modularity (QA-003-P2).
This module handles:
1. Risk estimation (covariance, expected returns)
2. Asset allocation (HRP, Risk Parity, etc.)

v2.2 Performance Optimization (task_040_3):
- Incremental covariance estimation via IncrementalCovarianceEstimator
- Cache key: universe_hash + halflife + date
- 10-20% speedup expected
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import structlog

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from src.allocation.covariance import CovarianceResult
    from src.backtest.covariance_cache import (
        CovarianceState,
        IncrementalCovarianceEstimator,
    )
    from src.config.settings import Settings
    from src.utils.logger import AuditLogger

logger = structlog.get_logger(__name__)


@dataclass
class RiskEstimationResult:
    """Result of risk estimation step."""

    risk_metrics: dict[str, Any]
    covariance: "pd.DataFrame | None"
    correlation: "pd.DataFrame | None"
    expected_returns: dict[str, float]


@dataclass
class AllocationResult:
    """Result of asset allocation step."""

    weights: dict[str, float]
    method: str
    turnover: float
    constraint_violations: int
    cash_weight: float


class RiskEstimator:
    """
    Handles risk estimation for the pipeline.

    Responsible for:
    - Building returns matrix from raw data
    - Estimating covariance using Ledoit-Wolf shrinkage
    - Computing expected returns for each asset
    - Calculating portfolio risk metrics
    - Caching covariance results for performance optimization
    - Incremental covariance estimation (v2.2 task_040_3)
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
        use_cache: bool = False,  # Cache disabled - not effective for backtest
        use_incremental: bool = False,  # v2.2: Enable incremental covariance
        halflife: int = 60,  # v2.2: Halflife for exponential weighting
    ) -> None:
        """
        Initialize RiskEstimator.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
            use_cache: Deprecated, ignored (cache removed in v2.1)
            use_incremental: Enable incremental covariance estimation (v2.2)
            halflife: Halflife for exponential weighting (v2.2)
        """
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="risk_estimator")
        self._use_cache = False
        self._cov_cache = None

        # v2.2 Incremental covariance support (task_040_3)
        self._use_incremental = use_incremental
        self._halflife = halflife
        self._incremental_estimator: Optional["IncrementalCovarianceEstimator"] = None
        self._incremental_cache_key: str = ""
        self._last_update_date: Optional[datetime] = None
        self._incremental_hits: int = 0
        self._incremental_misses: int = 0

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def _generate_universe_hash(self, universe: list[str]) -> str:
        """Generate hash for universe (v2.2).

        Args:
            universe: List of asset symbols (sorted for consistency)

        Returns:
            MD5 hash of sorted universe
        """
        sorted_universe = sorted(universe)
        universe_str = ",".join(sorted_universe)
        return hashlib.md5(universe_str.encode()).hexdigest()[:12]

    def _generate_cache_key(self, universe: list[str], halflife: int) -> str:
        """Generate cache key for incremental estimator (v2.2).

        Key format: {universe_hash}_{halflife}_{n_assets}

        Args:
            universe: List of asset symbols
            halflife: Halflife for exponential weighting

        Returns:
            Cache key string
        """
        universe_hash = self._generate_universe_hash(universe)
        return f"{universe_hash}_{halflife}_{len(universe)}"

    def _should_reset_estimator(self, cache_key: str) -> bool:
        """Check if estimator should be reset (v2.2).

        Reset when universe changes (different cache key).

        Args:
            cache_key: New cache key

        Returns:
            True if estimator should be reset
        """
        return cache_key != self._incremental_cache_key

    def get_incremental_state(self) -> Optional["CovarianceState"]:
        """Get current incremental estimator state (v2.2).

        Returns:
            CovarianceState or None if not initialized
        """
        if self._incremental_estimator is None:
            return None
        return self._incremental_estimator.get_state()

    def set_incremental_state(self, state: "CovarianceState") -> None:
        """Set incremental estimator state (v2.2).

        Used to restore state from checkpoint or cache.

        Args:
            state: CovarianceState to restore
        """
        from src.backtest.covariance_cache import IncrementalCovarianceEstimator

        self._incremental_estimator = IncrementalCovarianceEstimator(
            n_assets=state.n_assets,
            halflife=state.halflife,
            asset_names=state.asset_names,
        )
        self._incremental_estimator.set_state(state)
        self._logger.info(
            "Incremental estimator state restored",
            n_assets=state.n_assets,
            n_updates=state.n_updates,
        )

    @property
    def incremental_stats(self) -> dict[str, Any]:
        """Get incremental covariance statistics (v2.2).

        Returns:
            Dictionary with hit rate and estimator info
        """
        total = self._incremental_hits + self._incremental_misses
        hit_rate = self._incremental_hits / total if total > 0 else 0.0

        stats = {
            "enabled": self._use_incremental,
            "hits": self._incremental_hits,
            "misses": self._incremental_misses,
            "hit_rate": hit_rate,
            "cache_key": self._incremental_cache_key,
            "last_update_date": self._last_update_date,
        }

        if self._incremental_estimator is not None:
            stats["n_updates"] = self._incremental_estimator.n_updates
            stats["n_assets"] = self._incremental_estimator.n_assets
            stats["halflife"] = self._incremental_estimator.halflife

        return stats

    def estimate(
        self,
        raw_data: dict[str, "pl.DataFrame"],
        excluded_assets: list[str],
    ) -> RiskEstimationResult:
        """
        Estimate risk metrics.

        Args:
            raw_data: Dictionary mapping symbol to DataFrame
            excluded_assets: List of assets to exclude

        Returns:
            RiskEstimationResult with risk metrics and covariance
        """
        import pandas as pd

        from src.allocation.covariance import (
            CovarianceConfig,
            CovarianceEstimator,
            CovarianceMethod,
        )

        # Get valid assets (non-excluded)
        valid_symbols = [
            symbol
            for symbol in raw_data.keys()
            if symbol not in excluded_assets
        ]

        empty_result = RiskEstimationResult(
            risk_metrics={
                "volatility": 0.0,
                "var_95": 0.0,
                "expected_shortfall": 0.0,
            },
            covariance=None,
            correlation=None,
            expected_returns={},
        )

        if not valid_symbols:
            self._logger.warning("No valid assets for risk estimation")
            return empty_result

        # Step 1: Build returns matrix
        returns_dict: dict[str, pd.Series] = {}

        for symbol in valid_symbols:
            df = raw_data.get(symbol)
            if df is None:
                continue

            # Convert polars to pandas if needed
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()

            # Ensure we have a close price column
            if "close" not in df.columns:
                self._logger.warning(f"No close price for {symbol}")
                continue

            # Calculate returns
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            returns = df["close"].pct_change().dropna()
            returns_dict[symbol] = returns

        if not returns_dict:
            self._logger.warning("No returns data available")
            return empty_result

        # Build returns DataFrame (align by date)
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if returns_df.empty or len(returns_df) < 30:
            self._logger.warning(
                "Insufficient aligned returns: %d rows",
                len(returns_df) if not returns_df.empty else 0,
            )
            return empty_result

        # Step 2: Estimate covariance using Ledoit-Wolf (with caching)
        cov_result = self._estimate_covariance_with_cache(
            returns_df, valid_symbols
        )

        if cov_result is None or not cov_result.is_valid:
            self._logger.warning(
                "Covariance estimation failed, using sample covariance"
            )
            # Fallback to sample covariance
            cov_config_sample = CovarianceConfig(
                method=CovarianceMethod.SAMPLE,
                annualization_factor=252,
            )
            cov_estimator_sample = CovarianceEstimator(cov_config_sample)
            cov_result = cov_estimator_sample.estimate(returns_df)

        covariance = cov_result.covariance
        correlation = cov_result.correlation

        # Step 3: Estimate expected returns for each asset
        expected_returns: dict[str, float] = {}
        for symbol in returns_df.columns:
            returns_series = returns_df[symbol]
            # Use historical mean as expected return (annualized)
            # Apply shrinkage toward zero for robustness
            shrinkage_factor = 0.5
            raw_mean = returns_series.mean() * 252  # Annualize
            shrunk_mean = raw_mean * (1 - shrinkage_factor)
            expected_returns[symbol] = float(shrunk_mean)

        # Step 4: Calculate portfolio risk metrics
        n_assets = len(returns_df.columns)
        equal_weights = np.ones(n_assets) / n_assets

        # Portfolio volatility (equal-weighted baseline)
        cov_matrix = covariance.values
        port_var = equal_weights @ cov_matrix @ equal_weights
        port_vol = np.sqrt(max(0, port_var))

        # VaR and ES (historical method on equal-weighted portfolio)
        portfolio_returns = (returns_df * equal_weights).sum(axis=1)
        var_95 = float(np.percentile(portfolio_returns, 5))
        es_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())

        risk_metrics = {
            "volatility": float(port_vol),
            "var_95": var_95,
            "expected_shortfall": es_95 if not np.isnan(es_95) else var_95,
            "n_assets": n_assets,
            "n_samples": len(returns_df),
            "shrinkage_intensity": cov_result.shrinkage_intensity,
            "covariance_method": cov_result.method_used,
        }

        # Log to AuditLogger
        if self._audit_logger:
            self._audit_logger.log_risk_estimation(
                covariance_method=cov_result.method_used,
                n_assets=n_assets,
                n_samples=len(returns_df),
                shrinkage_intensity=cov_result.shrinkage_intensity,
                portfolio_volatility=port_vol,
                var_95=var_95,
                expected_shortfall=risk_metrics["expected_shortfall"],
            )

        self._logger.info(
            "Risk estimation completed",
            n_assets=n_assets,
            n_samples=len(returns_df),
            volatility=round(port_vol, 4),
            shrinkage=round(cov_result.shrinkage_intensity or 0, 4),
        )

        return RiskEstimationResult(
            risk_metrics=risk_metrics,
            covariance=covariance,
            correlation=correlation,
            expected_returns=expected_returns,
        )

    def _estimate_covariance_with_cache(
        self,
        returns_df: "pd.DataFrame",
        universe: list[str],
        lookback_days: int = 252,
        method: str = "ledoit_wolf",
    ) -> "CovarianceResult | None":
        """
        Estimate covariance with caching support and incremental update (v2.2).

        Args:
            returns_df: Returns DataFrame
            universe: List of asset symbols
            lookback_days: Lookback period for covariance estimation
            method: Covariance estimation method

        Returns:
            CovarianceResult or None if estimation fails

        v2.2 (task_040_3): Added incremental covariance estimation.
        When use_incremental=True, uses IncrementalCovarianceEstimator
        to update covariance with new returns only, avoiding full recalculation.
        """
        import pandas as pd

        from src.allocation.covariance import (
            CovarianceConfig,
            CovarianceEstimator,
            CovarianceMethod,
            CovarianceResult,
        )

        # Get end date from returns
        if returns_df.empty:
            return None

        end_date = returns_df.index[-1]
        if hasattr(end_date, "to_pydatetime"):
            end_date = end_date.to_pydatetime()
        elif not isinstance(end_date, datetime):
            end_date = datetime.now()

        # v2.2: Try incremental estimation first
        if self._use_incremental:
            incremental_result = self._try_incremental_estimation(
                returns_df, universe, end_date
            )
            if incremental_result is not None:
                return incremental_result

        # Legacy cache (kept for compatibility but disabled)
        if self._use_cache and self._cov_cache is not None:
            cached_result = self._cov_cache.get(
                universe, lookback_days, method, end_date
            )
            if cached_result is not None:
                self._logger.info(
                    "Covariance cache hit",
                    n_assets=len(universe),
                    method=method,
                )
                return cached_result

            # Check for incremental update opportunity
            if not self._cov_cache.needs_full_recalculation(
                universe, lookback_days, method, end_date
            ):
                base_result, base_date = self._cov_cache.find_nearest_cache(
                    universe, lookback_days, method, end_date
                )
                if base_result is not None and base_date is not None:
                    # Get new returns since base_date
                    new_returns = returns_df[returns_df.index > base_date]
                    if len(new_returns) > 0 and len(new_returns) < len(returns_df) * 0.3:
                        # Incremental update if new data is < 30% of total
                        updated_result = self._cov_cache.incremental_update(
                            universe, lookback_days, method,
                            base_result, new_returns, end_date,
                        )
                        self._logger.info(
                            "Covariance incremental update",
                            new_samples=len(new_returns),
                            n_assets=len(universe),
                        )
                        return updated_result

        # Full calculation
        cov_config = CovarianceConfig(
            method=CovarianceMethod.LEDOIT_WOLF,
            min_periods=30,
            annualization_factor=252,
        )
        cov_estimator = CovarianceEstimator(cov_config)
        cov_result = cov_estimator.estimate(returns_df)

        # Cache the result
        if self._use_cache and self._cov_cache is not None and cov_result.is_valid:
            self._cov_cache.put(
                universe, lookback_days, method, end_date,
                cov_result, is_incremental=False,
            )
            self._logger.info(
                "Covariance cached",
                n_assets=len(universe),
                method=method,
            )

        return cov_result

    def _try_incremental_estimation(
        self,
        returns_df: "pd.DataFrame",
        universe: list[str],
        end_date: datetime,
    ) -> "CovarianceResult | None":
        """Try incremental covariance estimation (v2.2 task_040_3).

        This method implements the core incremental update logic:
        1. Check if estimator needs reset (universe change)
        2. If new estimator, initialize with historical data
        3. If existing estimator, update with new returns only
        4. Convert IncrementalCovarianceEstimator output to CovarianceResult

        Args:
            returns_df: Returns DataFrame
            universe: List of asset symbols
            end_date: End date for estimation

        Returns:
            CovarianceResult or None if incremental estimation not possible
        """
        import pandas as pd

        from src.allocation.covariance import CovarianceResult
        from src.backtest.covariance_cache import IncrementalCovarianceEstimator

        # Ensure columns match universe (sorted for consistency)
        valid_cols = [c for c in returns_df.columns if c in universe]
        if not valid_cols:
            self._incremental_misses += 1
            return None

        returns_df = returns_df[sorted(valid_cols)]
        n_assets = len(returns_df.columns)

        # Generate cache key
        cache_key = self._generate_cache_key(list(returns_df.columns), self._halflife)

        # Check if we need to reset (universe changed)
        if self._should_reset_estimator(cache_key):
            self._logger.info(
                "Incremental estimator reset (universe change)",
                old_key=self._incremental_cache_key,
                new_key=cache_key,
            )
            self._incremental_estimator = None
            self._incremental_cache_key = cache_key
            self._last_update_date = None

        # Initialize estimator if needed
        if self._incremental_estimator is None:
            self._incremental_estimator = IncrementalCovarianceEstimator(
                n_assets=n_assets,
                halflife=self._halflife,
                asset_names=list(returns_df.columns),
            )
            # Initialize with all historical data
            self._incremental_estimator.update_batch(returns_df.values)
            self._last_update_date = end_date
            self._incremental_misses += 1

            self._logger.info(
                "Incremental estimator initialized",
                n_assets=n_assets,
                n_samples=len(returns_df),
                halflife=self._halflife,
            )
        else:
            # Update with new returns only
            if self._last_update_date is not None:
                new_returns = returns_df[returns_df.index > self._last_update_date]
                if len(new_returns) > 0:
                    self._incremental_estimator.update_batch(new_returns.values)
                    self._logger.debug(
                        "Incremental update",
                        new_samples=len(new_returns),
                    )
            self._last_update_date = end_date
            self._incremental_hits += 1

        # Convert to CovarianceResult
        cov_matrix = self._incremental_estimator.get_covariance()
        corr_matrix = self._incremental_estimator.get_correlation()
        vol = self._incremental_estimator.get_volatility()

        # Annualize (assuming daily returns)
        cov_matrix_annualized = cov_matrix * 252

        # Create DataFrames with proper index/columns
        columns = list(returns_df.columns)
        cov_df = pd.DataFrame(cov_matrix_annualized, index=columns, columns=columns)
        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        vol_series = pd.Series(vol * np.sqrt(252), index=columns)

        return CovarianceResult(
            covariance=cov_df,
            correlation=corr_df,
            volatilities=vol_series,
            method_used="incremental_ewm",
            shrinkage_intensity=0.0,  # Not applicable for incremental
            effective_samples=self._incremental_estimator.n_updates,
            metadata={
                "incremental": True,
                "halflife": self._halflife,
                "n_assets": n_assets,
            },
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get covariance cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if self._cov_cache is not None:
            return self._cov_cache.get_stats()
        return {"caching_enabled": False}


class AssetAllocator:
    """
    Handles asset allocation for the pipeline.

    Responsible for:
    - Executing HRP/Risk Parity/other allocation methods
    - Handling CASH weight from TopNSelector
    - Applying constraints (max weight, turnover limit)
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
    ) -> None:
        """
        Initialize AssetAllocator.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
        """
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="asset_allocator")

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def _prepare_covariance(
        self,
        covariance: "pd.DataFrame | None",
        valid_assets: list[str],
    ) -> "pd.DataFrame | None":
        """共分散行列をnon-CASHアセットのみにフィルタリング"""
        if covariance is None or not hasattr(covariance, "columns"):
            return None
        cov_cols = [c for c in covariance.columns if c in valid_assets]
        if cov_cols:
            return covariance.loc[cov_cols, cov_cols]
        return None

    def _build_returns_dataframe(
        self,
        raw_data: dict[str, "pl.DataFrame"],
        valid_assets: list[str],
    ) -> "pd.DataFrame":
        """リターンDataFrameを構築"""
        import pandas as pd

        returns_dict: dict[str, pd.Series] = {}
        for symbol in valid_assets:
            df = raw_data.get(symbol)
            if df is None:
                continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            if "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                returns_dict[symbol] = returns

        return pd.DataFrame(returns_dict).dropna()

    def _prepare_series_for_allocation(
        self,
        data: dict[str, float] | None,
        valid_assets: list[str],
        exclude_cash: bool = True,
    ) -> "pd.Series | None":
        """配分用のSeriesを準備"""
        import pandas as pd

        if not data:
            return None
        filtered = {
            k: v for k, v in data.items()
            if k in valid_assets and (not exclude_cash or k != "CASH")
        }
        if not filtered:
            return None
        series = pd.Series(filtered)
        return series.reindex(valid_assets, fill_value=0.0)

    def _execute_allocation(
        self,
        allocator,
        returns_df: "pd.DataFrame",
        expected_returns_series: "pd.Series | None",
        prev_weights_series: "pd.Series | None",
        cov_for_allocation: "pd.DataFrame | None",
        valid_assets: list[str],
        risk_metrics: dict[str, Any],
    ) -> tuple[dict[str, float], Any]:
        """配分を実行"""
        allocation_result = None
        raw_non_cash_weights: dict[str, float] = {}

        try:
            allocation_result = allocator.allocate(
                returns=returns_df,
                expected_returns=expected_returns_series,
                quality_flags=None,
                previous_weights=prev_weights_series,
                covariance=cov_for_allocation,
            )
            raw_non_cash_weights = allocation_result.weights.to_dict()
            if not allocation_result.is_valid:
                self._logger.warning(
                    "Allocation result invalid, fallback=%s",
                    allocation_result.fallback_reason.value,
                )
            if allocation_result.portfolio_metrics:
                risk_metrics.update(allocation_result.portfolio_metrics)
        except Exception as e:
            self._logger.error("Allocation failed: %s", e)
            weight = 1.0 / len(valid_assets)
            raw_non_cash_weights = {asset: weight for asset in valid_assets}

        return raw_non_cash_weights, allocation_result

    def _apply_constraints(
        self,
        scaled_weights: dict[str, float],
        prev_weights: dict[str, float] | None,
        cash_weight: float,
    ) -> tuple[dict[str, float], int, float]:
        """制約を適用"""
        import pandas as pd

        from src.allocation.constraints import ConstraintConfig, ConstraintProcessor

        constraint_config = ConstraintConfig(
            w_max=self.settings.asset_allocation.w_asset_max,
            w_min=getattr(self.settings.asset_allocation, "w_asset_min", 0.0),
            delta_max=self.settings.asset_allocation.delta_max,
        )
        constraint_processor = ConstraintProcessor(constraint_config)

        non_cash_weights_series = pd.Series({
            k: v for k, v in scaled_weights.items() if k != "CASH"
        })

        non_cash_prev_series = None
        if prev_weights:
            non_cash_prev_series = pd.Series({
                k: v for k, v in prev_weights.items() if k != "CASH"
            })
            non_cash_prev_series = non_cash_prev_series.reindex(
                non_cash_weights_series.index, fill_value=0.0
            )

        constraint_result = constraint_processor.apply(
            non_cash_weights_series,
            non_cash_prev_series,
        )

        result_weights = constraint_result.weights.to_dict()
        if cash_weight > 0:
            result_weights["CASH"] = cash_weight

        return result_weights, len(constraint_result.violations), constraint_result.turnover

    def allocate(
        self,
        raw_data: dict[str, "pl.DataFrame"],
        excluded_assets: list[str],
        covariance: "pd.DataFrame | None",
        expected_returns: dict[str, float],
        risk_metrics: dict[str, Any],
        cash_weight: float,
        prev_weights: dict[str, float] | None = None,
    ) -> AllocationResult:
        """
        Execute asset allocation.

        Args:
            raw_data: Dictionary mapping symbol to DataFrame
            excluded_assets: List of assets to exclude
            covariance: Covariance matrix from risk estimation
            expected_returns: Expected returns from risk estimation
            risk_metrics: Risk metrics dictionary (will be updated)
            cash_weight: CASH weight from TopNSelector
            prev_weights: Previous period weights

        Returns:
            AllocationResult with final weights
        """
        from src.allocation.allocator import (
            AllocationMethod,
            AllocatorConfig,
            AssetAllocator as AA,
        )

        non_cash_weight = 1.0 - cash_weight
        self._logger.debug(
            "CASH handling",
            cash_weight=round(cash_weight, 4),
            non_cash_weight=round(non_cash_weight, 4),
        )

        valid_assets = [
            s for s in raw_data.keys()
            if s not in excluded_assets and s != "CASH"
        ]

        # Edge case: all CASH
        if non_cash_weight <= 0 or not valid_assets:
            self._logger.info("Full CASH allocation")
            return AllocationResult(
                weights={"CASH": 1.0}, method="full_cash",
                turnover=0.0, constraint_violations=0, cash_weight=1.0,
            )

        # Prepare covariance
        cov_for_allocation = self._prepare_covariance(covariance, valid_assets)
        if cov_for_allocation is None or (
            hasattr(cov_for_allocation, "empty") and cov_for_allocation.empty
        ):
            self._logger.warning("No covariance matrix, using equal weights")
            weight = non_cash_weight / len(valid_assets)
            weights = {asset: weight for asset in valid_assets}
            weights["CASH"] = cash_weight
            return AllocationResult(
                weights=weights, method="equal_weight_fallback",
                turnover=0.0, constraint_violations=0, cash_weight=cash_weight,
            )

        # Determine allocation method
        settings_method = self.settings.asset_allocation.method.value
        method_map = {
            "HRP": AllocationMethod.HRP, "hrp": AllocationMethod.HRP,
            "risk_parity": AllocationMethod.RISK_PARITY,
            "RISK_PARITY": AllocationMethod.RISK_PARITY,
            "mean_variance": AllocationMethod.MEAN_VARIANCE,
            "equal_weight": AllocationMethod.EQUAL_WEIGHT,
        }
        allocation_method = method_map.get(settings_method, AllocationMethod.HRP)

        # Initialize allocator
        allocator_config = AllocatorConfig(
            method=allocation_method,
            w_asset_max=self.settings.asset_allocation.w_asset_max,
            w_asset_min=getattr(self.settings.asset_allocation, "w_asset_min", 0.0),
            delta_max=self.settings.asset_allocation.delta_max,
            smooth_alpha=self.settings.asset_allocation.smooth_alpha,
            allow_short=getattr(self.settings.asset_allocation, "allow_short", False),
            fallback_to_previous=True,
            fallback_to_equal=True,
        )
        allocator = AA(allocator_config)

        # Build returns DataFrame
        returns_df = self._build_returns_dataframe(raw_data, valid_assets)
        if returns_df.empty:
            self._logger.warning("No returns data, using equal weights")
            weight = non_cash_weight / len(valid_assets)
            weights = {asset: weight for asset in valid_assets}
            weights["CASH"] = cash_weight
            return AllocationResult(
                weights=weights, method="equal_weight_fallback",
                turnover=0.0, constraint_violations=0, cash_weight=cash_weight,
            )

        # Prepare series
        expected_returns_series = self._prepare_series_for_allocation(
            expected_returns, valid_assets
        )
        prev_weights_series = self._prepare_series_for_allocation(
            prev_weights, valid_assets
        )

        # Execute allocation
        raw_non_cash_weights, allocation_result = self._execute_allocation(
            allocator, returns_df, expected_returns_series,
            prev_weights_series, cov_for_allocation, valid_assets, risk_metrics,
        )

        # Scale by non_cash_weight and add CASH
        scaled_weights = {
            asset: weight * non_cash_weight
            for asset, weight in raw_non_cash_weights.items()
        }
        if cash_weight > 0:
            scaled_weights["CASH"] = cash_weight

        # Apply constraints
        constraint_violations = 0
        turnover = 0.0
        if valid_assets:
            scaled_weights, constraint_violations, turnover = self._apply_constraints(
                scaled_weights, prev_weights, cash_weight
            )

        # Normalize if needed
        total_weight = sum(scaled_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            self._logger.warning("Weight sum deviation", total=round(total_weight, 4))
            scaled_weights = {k: v / total_weight for k, v in scaled_weights.items()}

        # Log to AuditLogger
        if self._audit_logger:
            self._audit_logger.log_asset_allocation(
                weights=scaled_weights, method=allocation_method.value,
                constraints={
                    "w_asset_max": self.settings.asset_allocation.w_asset_max,
                    "delta_max": self.settings.asset_allocation.delta_max,
                },
                risk_metrics=risk_metrics,
                fallback_reason=(
                    allocation_result.fallback_reason.value
                    if allocation_result and allocation_result.is_fallback else None
                ),
                constraint_violations=constraint_violations, turnover=turnover,
                cash_weight=cash_weight, non_cash_assets=len(valid_assets),
            )

        self._logger.info(
            "Asset allocation completed", method=allocation_method.value,
            asset_count=len(scaled_weights), cash_weight=round(cash_weight, 4),
            non_cash_assets=len(valid_assets), turnover=round(turnover, 4),
            violations=constraint_violations,
        )

        return AllocationResult(
            weights=scaled_weights, method=allocation_method.value,
            turnover=turnover, constraint_violations=constraint_violations,
            cash_weight=cash_weight,
        )
