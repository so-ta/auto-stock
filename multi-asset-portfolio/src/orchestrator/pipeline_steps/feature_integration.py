"""
Feature Integration Module - CMD016/017 Feature Integration

This module handles the integration of CMD016 and CMD017 features
into the portfolio pipeline:
- CMD016: VIX-based cash allocation, correlation break detection,
          drawdown protection, sector rotation, signal filtering
- CMD017: NCO, Black-Litterman, CVaR, Transaction Cost optimization,
          Walk-Forward enhancement, Validation (synthetic data, Purged K-Fold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd
    from src.config.settings import Settings
    from src.orchestrator.cmd016_integrator import CMD016Integrator

logger = logging.getLogger(__name__)

# CMD_016 features integration (Phase 5)
try:
    from src.orchestrator.cmd016_integrator import (
        CMD016Integrator,
        CMD016Config,
        IntegrationResult,
    )
    CMD016_AVAILABLE = True
except ImportError:
    CMD016_AVAILABLE = False

# CMD_017 features integration (Phase 6)
try:
    from src.allocation.nco import NestedClusteredOptimization, NCOConfig
    from src.allocation.black_litterman import BlackLittermanModel
    from src.allocation.cvar_optimizer import CVaROptimizer
    from src.allocation.transaction_cost_optimizer import TransactionCostOptimizer
    from src.backtest.adaptive_window import AdaptiveWindowSelector, VolatilityRegime
    from src.backtest.synthetic_data import StatisticalSignificanceTester
    from src.backtest.purged_kfold import PurgedKFold
    CMD017_AVAILABLE = True
except ImportError:
    CMD017_AVAILABLE = False


@dataclass
class FeatureIntegrationContext:
    """Context data for feature integration."""
    raw_data: Dict[str, Any]
    raw_weights: Dict[str, float]
    signals: Dict[str, Any]
    excluded_assets: set
    portfolio_value: float
    vix_value: Optional[float]
    settings: "Settings"


@dataclass
class FeatureIntegrationResult:
    """Result of feature integration."""
    adjusted_weights: Dict[str, float]
    cmd016_result: Dict[str, Any]
    cmd017_result: Dict[str, Any]
    warnings: List[str]


class FeatureIntegrator:
    """
    Integrates CMD016 and CMD017 features into portfolio allocation.

    This class encapsulates the complex feature integration logic that was
    previously in the Pipeline class, making it more modular and testable.
    """

    def __init__(
        self,
        settings: "Settings",
        cmd016_integrator: Optional["CMD016Integrator"] = None,
    ):
        """
        Initialize the feature integrator.

        Args:
            settings: Application settings
            cmd016_integrator: CMD016 integrator instance (optional)
        """
        self._settings = settings
        self._cmd016_integrator = cmd016_integrator
        self._logger = logger

    @property
    def cmd016_available(self) -> bool:
        """Check if CMD016 features are available."""
        return CMD016_AVAILABLE and self._cmd016_integrator is not None

    @property
    def cmd017_available(self) -> bool:
        """Check if CMD017 features are available."""
        return CMD017_AVAILABLE

    def integrate_cmd016(
        self,
        context: FeatureIntegrationContext,
        get_asset_signal_score: callable,
    ) -> Dict[str, Any]:
        """
        Apply CMD016 feature integration.

        Args:
            context: Feature integration context
            get_asset_signal_score: Function to get signal score for an asset

        Returns:
            CMD016 integration result
        """
        import pandas as pd

        if not self.cmd016_available:
            self._logger.info("CMD_016 integration skipped (not available)")
            return {"enabled": False}

        try:
            # Prepare weights
            weights = pd.Series(context.raw_weights) if context.raw_weights else pd.Series()
            if weights.empty:
                return {"enabled": False, "reason": "No weights to adjust"}

            # Prepare signals
            signals = pd.Series()
            for symbol in context.signals:
                score = get_asset_signal_score(symbol)
                if score is not None:
                    signals[symbol] = score

            # Calculate returns DataFrame
            returns_df = self._prepare_returns_df(context)

            # Calculate portfolio value from returns if available
            portfolio_value = context.portfolio_value
            if returns_df is not None and not returns_df.empty:
                portfolio_returns = returns_df.mean(axis=1)
                portfolio_value = 100000.0 * (1 + portfolio_returns).cumprod().iloc[-1]

            # Get VIX value
            vix_value = context.vix_value
            if vix_value is None and "^VIX" in context.raw_data:
                vix_df = context.raw_data["^VIX"]
                if hasattr(vix_df, "to_pandas"):
                    vix_df = vix_df.to_pandas()
                if "close" in vix_df.columns:
                    vix_value = vix_df["close"].iloc[-1]

            # Apply full integration
            result = self._cmd016_integrator.integrate_all(
                base_weights=weights,
                signals=signals if not signals.empty else None,
                portfolio_value=portfolio_value,
                vix_value=vix_value,
                returns=returns_df,
                macro_indicators=None,
            )

            # Log feature status
            feature_status = self._cmd016_integrator.get_feature_status()
            active_features = [k for k, v in feature_status.items() if v]

            self._logger.info(
                f"CMD_016 integration completed: {len(active_features)} features, "
                f"cash_ratio={result.cash_ratio}, vix={vix_value}"
            )

            return {
                "enabled": True,
                "adjusted_weights": result.adjusted_weights.to_dict(),
                "cash_ratio": result.cash_ratio,
                "active_features": active_features,
                "warnings": result.warnings,
            }

        except Exception as e:
            self._logger.warning(f"CMD_016 integration failed: {e}")
            return {"enabled": True, "error": str(e)}

    def integrate_cmd017(
        self,
        context: FeatureIntegrationContext,
    ) -> Dict[str, Any]:
        """
        Apply CMD017 feature integration.

        Args:
            context: Feature integration context

        Returns:
            CMD017 integration result
        """
        import pandas as pd

        if not self.cmd017_available:
            self._logger.info("CMD_017 integration skipped (not available)")
            return {"enabled": False}

        try:
            cmd017_config = getattr(self._settings, "cmd_017_features", None)
            if cmd017_config is None:
                self._logger.info("CMD_017 config not found, using defaults")
                cmd017_config = {}

            result = {
                "enabled": True,
                "allocation": {},
                "ml": {},
                "walkforward": {},
                "execution": {},
                "risk": {},
                "validation": {},
            }

            # Prepare returns data
            returns_df = self._prepare_returns_df(context)
            if returns_df is None:
                self._logger.warning("No returns data for CMD_017 integration")
                return {"enabled": False, "reason": "No returns data"}

            # Prepare weights series
            current_weights = context.raw_weights or {}
            current_weights_series = pd.Series(current_weights)
            current_weights_series = current_weights_series.reindex(returns_df.columns).fillna(0)

            # 1. Allocation Optimization
            allocation_config = cmd017_config.get("allocation", {}) if isinstance(cmd017_config, dict) else {}
            current_weights_series = self._apply_allocation(
                allocation_config, returns_df, current_weights_series, result
            )

            # 2. Walk-Forward Enhancement
            wf_config = cmd017_config.get("walkforward", {}) if isinstance(cmd017_config, dict) else {}
            self._apply_walkforward(wf_config, returns_df, current_weights_series, result)

            # 3. Validation
            val_config = cmd017_config.get("validation", {}) if isinstance(cmd017_config, dict) else {}
            self._apply_validation(val_config, returns_df, current_weights_series, result)

            self._logger.info(
                f"CMD_017 integration completed: method={allocation_config.get('method', 'nco')}"
            )

            result["adjusted_weights"] = current_weights_series.to_dict()
            return result

        except Exception as e:
            self._logger.warning(f"CMD_017 integration failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _prepare_returns_df(self, context: FeatureIntegrationContext) -> "pd.DataFrame | None":
        """Prepare returns DataFrame from raw data."""
        import pandas as pd

        returns_list = []
        for symbol, df in context.raw_data.items():
            if symbol in context.excluded_assets or symbol == "CASH":
                continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            if "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                returns.name = symbol
                returns_list.append(returns)

        if not returns_list:
            return None

        returns_df = pd.concat(returns_list, axis=1).dropna()
        return returns_df if not returns_df.empty else None

    def _apply_allocation(
        self,
        allocation_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> "pd.Series":
        """Apply allocation optimization (NCO, BL, CVaR, TC)."""
        allocation_method = allocation_config.get("method", "nco")

        # NCO
        if allocation_method == "nco" and allocation_config.get("nco", {}).get("enabled", True):
            try:
                nco_params = allocation_config.get("nco", {})
                nco_cfg = NCOConfig(
                    n_clusters=nco_params.get("n_clusters", 5),
                    intra_method=nco_params.get("intra_method", "min_variance"),
                )
                nco = NestedClusteredOptimization(config=nco_cfg)
                nco_result = nco.fit(returns_df)
                if nco_result.is_valid:
                    blend_factor = 0.5
                    for symbol in nco_result.weights.index:
                        if symbol in current_weights_series.index:
                            current_weights_series[symbol] = (
                                blend_factor * nco_result.weights[symbol] +
                                (1 - blend_factor) * current_weights_series[symbol]
                            )
                    result["allocation"]["nco"] = {
                        "status": "applied",
                        "n_clusters": nco_params.get("n_clusters", 5),
                    }
                else:
                    result["allocation"]["nco"] = {"status": "invalid", "reason": "optimization failed"}
            except Exception as e:
                result["allocation"]["nco"] = {"status": "error", "message": str(e)}

        # Black-Litterman
        if allocation_config.get("black_litterman", {}).get("enabled", True):
            try:
                bl_params = allocation_config.get("black_litterman", {})
                bl_model = BlackLittermanModel(
                    tau=bl_params.get("tau", 0.05),
                    risk_aversion=bl_params.get("risk_aversion", 2.5),
                )
                cov_matrix = returns_df.cov() * 252
                equilibrium_returns = bl_model.compute_equilibrium_returns(
                    cov_matrix, current_weights_series
                )
                result["allocation"]["black_litterman"] = {
                    "status": "computed",
                    "tau": bl_params.get("tau", 0.05),
                    "equilibrium_returns_mean": float(equilibrium_returns.mean()),
                }
            except Exception as e:
                result["allocation"]["black_litterman"] = {"status": "error", "message": str(e)}

        # CVaR
        if allocation_config.get("cvar", {}).get("enabled", True):
            try:
                cvar_params = allocation_config.get("cvar", {})
                cvar_optimizer = CVaROptimizer(alpha=cvar_params.get("alpha", 0.05))
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                cvar_value = cvar_optimizer.compute_cvar(portfolio_returns.values)
                result["allocation"]["cvar"] = {
                    "status": "computed",
                    "cvar": float(cvar_value),
                    "alpha": cvar_params.get("alpha", 0.05),
                }
            except Exception as e:
                result["allocation"]["cvar"] = {"status": "error", "message": str(e)}

        # Transaction Cost
        if allocation_config.get("transaction_cost", {}).get("enabled", True):
            try:
                tc_params = allocation_config.get("transaction_cost", {})
                tc_optimizer = TransactionCostOptimizer(
                    cost_aversion=tc_params.get("cost_aversion", 1.0),
                    max_weight=0.20,
                )
                tc_result = tc_optimizer.optimize(returns_df, current_weights_series.to_dict())
                if tc_result.converged or tc_result.turnover < tc_params.get("max_turnover", 0.20):
                    for symbol, weight in tc_result.optimal_weights.items():
                        current_weights_series[symbol] = weight
                    result["allocation"]["transaction_cost"] = {
                        "status": "applied",
                        "turnover": float(tc_result.turnover),
                        "cost": float(tc_result.transaction_cost),
                    }
                else:
                    result["allocation"]["transaction_cost"] = {
                        "status": "skipped",
                        "reason": "turnover_limit_exceeded",
                    }
            except Exception as e:
                result["allocation"]["transaction_cost"] = {"status": "error", "message": str(e)}

        return current_weights_series

    def _apply_walkforward(
        self,
        wf_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> None:
        """Apply Walk-Forward enhancement."""
        if wf_config.get("adaptive_window", {}).get("enabled", True):
            try:
                aw_params = wf_config.get("adaptive_window", {})
                selector = AdaptiveWindowSelector(
                    min_window=aw_params.get("min_window", 126),
                    max_window=aw_params.get("max_window", 756),
                    default_window=aw_params.get("default_window", 504),
                )
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                regime_change = selector.detect_regime_change(portfolio_returns)
                vol_regime = VolatilityRegime.HIGH_VOL if regime_change.detected else VolatilityRegime.NORMAL
                optimal_window = selector.compute_optimal_window(
                    returns=portfolio_returns,
                    volatility_regime=vol_regime,
                    regime_change_detected=regime_change.detected,
                )
                result["walkforward"]["adaptive_window"] = {
                    "status": "computed",
                    "optimal_window": optimal_window,
                    "regime_change_detected": regime_change.detected,
                }
            except Exception as e:
                result["walkforward"]["adaptive_window"] = {"status": "error", "message": str(e)}

    def _apply_validation(
        self,
        val_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> None:
        """Apply validation (statistical significance, Purged K-Fold)."""
        import numpy as np

        # Statistical Significance
        if val_config.get("synthetic_data", {}).get("enabled", True):
            try:
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                tester = StatisticalSignificanceTester(
                    n_bootstrap=val_config.get("synthetic_data", {}).get("n_simulations", 500),
                    confidence_level=0.95,
                )
                sig_result = tester.test_sharpe_significance(portfolio_returns.values)
                result["validation"]["sharpe_significance"] = {
                    "status": "computed",
                    "observed_sharpe": float(sig_result["observed_sharpe"]),
                    "ci_lower": float(sig_result["ci_lower"]),
                    "ci_upper": float(sig_result["ci_upper"]),
                    "p_value": float(sig_result["p_value"]),
                    "significant": sig_result["significant"],
                }
            except Exception as e:
                result["validation"]["sharpe_significance"] = {"status": "error", "message": str(e)}

        # Purged K-Fold
        if val_config.get("purged_kfold", {}).get("enabled", True):
            try:
                pkf_params = val_config.get("purged_kfold", {})
                n_splits = pkf_params.get("n_splits", 5)
                purge_gap = pkf_params.get("purge_gap", 5)
                pkf = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
                n_samples = len(returns_df)
                splits = list(pkf.split(np.arange(n_samples)))
                result["validation"]["purged_kfold"] = {
                    "status": "configured",
                    "n_splits": n_splits,
                    "actual_splits": len(splits),
                }
            except Exception as e:
                result["validation"]["purged_kfold"] = {"status": "error", "message": str(e)}
