"""
Regime Detection Module - Market Regime Detection and Dynamic Weighting

This module handles market regime detection and dynamic weight adjustments:
- Regime detection (volatility/trend regimes)
- Ensemble score combination
- Dynamic weight adjustments based on market conditions
- Return maximization features (Kelly allocation, hysteresis, macro timing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd
    from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Return maximization support (optional - Phase 4 integration)
try:
    from src.allocation.kelly_allocator import KellyAllocator
    from src.strategy.entry_exit_optimizer import HysteresisFilter
    from src.signals.macro_timing import EconomicCycleAllocator
    RETURN_MAXIMIZATION_AVAILABLE = True
except ImportError:
    RETURN_MAXIMIZATION_AVAILABLE = False


@dataclass
class RegimeInfo:
    """Market regime information."""
    current_vol_regime: str = "medium"
    current_trend_regime: str = "range"
    regime_scores: Dict[str, float] = None
    representative_symbol: Optional[str] = None

    def __post_init__(self):
        if self.regime_scores is None:
            self.regime_scores = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_vol_regime": self.current_vol_regime,
            "current_trend_regime": self.current_trend_regime,
            "regime_scores": self.regime_scores,
            "representative_symbol": self.representative_symbol,
        }


@dataclass
class DynamicWeightingResult:
    """Result of dynamic weighting adjustments."""
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    market_data: Dict[str, float]
    regime_info: Dict[str, Any]


class RegimeDetector:
    """
    Detects market regimes and applies dynamic weight adjustments.

    This class encapsulates the regime detection and dynamic weighting logic
    that was previously in the Pipeline class.
    """

    def __init__(self, settings: "Settings"):
        """
        Initialize the regime detector.

        Args:
            settings: Application settings
        """
        self._settings = settings
        self._logger = logger

    def detect_regime(
        self,
        raw_data: Dict[str, Any],
        excluded_assets: set,
    ) -> RegimeInfo:
        """
        Detect market regime using RegimeDetector signal.

        Args:
            raw_data: Raw price data by symbol
            excluded_assets: Set of excluded asset symbols

        Returns:
            RegimeInfo with detected regime information
        """
        import pandas as pd
        from src.signals import SignalRegistry

        regime_info = RegimeInfo()

        # Check if dynamic weighting is enabled
        dynamic_weighting_config = getattr(self._settings, "dynamic_weighting", None)
        if dynamic_weighting_config is None or not getattr(dynamic_weighting_config, "enabled", False):
            self._logger.info("Regime detection skipped (dynamic_weighting disabled)")
            return regime_info

        # Get valid symbols
        valid_symbols = [
            symbol for symbol in raw_data.keys()
            if symbol not in excluded_assets
        ]

        if not valid_symbols:
            self._logger.warning("No valid assets for regime detection")
            return regime_info

        # Get RegimeDetector signal
        try:
            regime_detector_cls = SignalRegistry.get("regime_detector")
        except KeyError:
            self._logger.warning("RegimeDetector signal not registered")
            return regime_info

        # Use first valid asset as representative
        representative_symbol = valid_symbols[0]
        df = raw_data.get(representative_symbol)

        if df is None:
            return regime_info

        # Convert polars to pandas if needed
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        try:
            # Run regime detection
            lookback = getattr(dynamic_weighting_config, "regime_lookback_days", 60)
            regime_signal = regime_detector_cls(vol_period=20, trend_period=lookback)
            result = regime_signal.compute(df)

            # Extract regime info from metadata
            regime_info = RegimeInfo(
                current_vol_regime=result.metadata.get("current_vol_regime", "medium"),
                current_trend_regime=result.metadata.get("current_trend_regime", "range"),
                regime_scores={
                    "vol_mean": result.metadata.get("rolling_vol_mean", 0.0),
                    "trend_mean": result.metadata.get("trend_signal_mean", 0.0),
                },
                representative_symbol=representative_symbol,
            )

            self._logger.info(
                f"Regime detection completed: vol={regime_info.current_vol_regime}, "
                f"trend={regime_info.current_trend_regime}, symbol={representative_symbol}"
            )

        except Exception as e:
            self._logger.warning(f"Regime detection failed: {e}")

        return regime_info

    def combine_ensemble_scores(
        self,
        evaluations: List[Any],
    ) -> Dict[str, Any]:
        """
        Combine strategy scores using EnsembleCombiner.

        Args:
            evaluations: List of strategy evaluations

        Returns:
            Ensemble combined scores by asset
        """
        import pandas as pd

        ensemble_scores = {}

        # Check if ensemble combiner is enabled
        ensemble_config = getattr(self._settings, "ensemble_combiner", None)
        if ensemble_config is None or not getattr(ensemble_config, "enabled", False):
            self._logger.info("Ensemble combine skipped (ensemble_combiner disabled)")
            return ensemble_scores

        if not evaluations:
            self._logger.warning("No evaluations available for ensemble combine")
            return ensemble_scores

        try:
            from src.meta.ensemble_combiner import EnsembleCombiner, EnsembleCombinerConfig

            # Initialize combiner from settings
            combiner_config = EnsembleCombinerConfig(
                method=getattr(ensemble_config, "method", "weighted_avg"),
                beta=getattr(ensemble_config, "beta", 2.0),
            )
            combiner = EnsembleCombiner(combiner_config)

            # Group evaluations by asset
            evaluations_by_asset: Dict[str, list] = {}
            for evaluation in evaluations:
                asset_id = evaluation.asset_id
                if asset_id not in evaluations_by_asset:
                    evaluations_by_asset[asset_id] = []
                evaluations_by_asset[asset_id].append(evaluation)

            for asset_id, asset_evaluations in evaluations_by_asset.items():
                # Prepare strategy scores as dict
                strategy_scores: Dict[str, pd.Series] = {}
                past_performance: Dict[str, float] = {}

                for evaluation in asset_evaluations:
                    if evaluation.metrics is None:
                        continue

                    strategy_id = evaluation.strategy_id
                    score_value = evaluation.score if evaluation.score else 0.0
                    strategy_scores[strategy_id] = pd.Series([score_value], index=[asset_id])
                    past_performance[strategy_id] = evaluation.metrics.sharpe_ratio

                if not strategy_scores:
                    continue

                # Combine
                combine_result = combiner.combine(strategy_scores, past_performance)

                ensemble_scores[asset_id] = {
                    "combined_scores": combine_result.combined_scores.to_dict() if combine_result.is_valid else {},
                    "strategy_weights": combine_result.strategy_weights,
                    "method_used": combine_result.method_used,
                }

            self._logger.info(
                f"Ensemble combine completed: {len(ensemble_scores)} assets, method={combiner_config.method}"
            )

        except ImportError as e:
            self._logger.warning(f"EnsembleCombiner not available: {e}")
        except Exception as e:
            self._logger.warning(f"Ensemble combine failed: {e}")

        return ensemble_scores

    def apply_dynamic_weighting(
        self,
        raw_weights: Dict[str, float],
        raw_data: Dict[str, Any],
        excluded_assets: set,
        regime_info: RegimeInfo,
        previous_weights: Dict[str, float],
        get_asset_signal_score: callable,
    ) -> DynamicWeightingResult:
        """
        Apply dynamic weight adjustments based on market conditions.

        Args:
            raw_weights: Base weights to adjust
            raw_data: Raw price data by symbol
            excluded_assets: Set of excluded asset symbols
            regime_info: Current regime information
            previous_weights: Previous period weights
            get_asset_signal_score: Function to get signal score for an asset

        Returns:
            DynamicWeightingResult with adjusted weights
        """
        import numpy as np
        import pandas as pd

        result = DynamicWeightingResult(
            original_weights=raw_weights.copy(),
            adjusted_weights=raw_weights.copy(),
            market_data={},
            regime_info=regime_info.to_dict() if regime_info else {},
        )

        # Check if dynamic weighting is enabled
        dynamic_config = getattr(self._settings, "dynamic_weighting", None)
        if dynamic_config is None or not getattr(dynamic_config, "enabled", False):
            self._logger.info("Dynamic weighting skipped (disabled)")
            return result

        try:
            from src.meta.dynamic_weighter import DynamicWeighter, DynamicWeightingConfig

            # Initialize DynamicWeighter from settings
            weighter_config = DynamicWeightingConfig(
                target_volatility=getattr(dynamic_config, "target_volatility", 0.15),
                max_drawdown_trigger=getattr(dynamic_config, "max_drawdown_trigger", 0.10),
                regime_lookback_days=getattr(dynamic_config, "regime_lookback_days", 60),
                vol_scaling_enabled=True,
                dd_protection_enabled=True,
                regime_weighting_enabled=bool(regime_info),
            )
            weighter = DynamicWeighter(weighter_config)

            # Prepare returns data
            returns_list = []
            for symbol, df in raw_data.items():
                if symbol in excluded_assets or symbol == "CASH":
                    continue

                if hasattr(df, "to_pandas"):
                    df = df.to_pandas()

                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")

                if "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    returns_list.append(returns)

            if not returns_list:
                self._logger.warning("No returns data for dynamic weighting")
                return result

            # Calculate equal-weight portfolio returns
            returns_df = pd.concat(returns_list, axis=1).dropna()
            if returns_df.empty:
                return result

            portfolio_returns = returns_df.mean(axis=1)

            # Calculate portfolio value (cumulative returns)
            portfolio_value = 100.0 * (1 + portfolio_returns).cumprod()
            peak_value = portfolio_value.cummax()

            current_value = portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 100.0
            current_peak = peak_value.iloc[-1] if len(peak_value) > 0 else 100.0

            market_data = {
                "returns": portfolio_returns,
                "portfolio_value": current_value,
                "peak_value": current_peak,
            }

            # Apply dynamic weight adjustment
            adjusted_weights = weighter.adjust_weights(
                weights=raw_weights,
                market_data=market_data,
                regime_info=regime_info.to_dict() if regime_info else None,
            )

            result.adjusted_weights = adjusted_weights
            result.market_data = {
                "portfolio_value": current_value,
                "peak_value": current_peak,
                "drawdown": (current_peak - current_value) / current_peak if current_peak > 0 else 0,
            }

            self._logger.info(
                f"Dynamic weighting completed: vol_regime={regime_info.current_vol_regime}, "
                f"trend_regime={regime_info.current_trend_regime}"
            )

            # Apply return maximization if enabled
            return_max_config = getattr(self._settings, "return_maximization", None)
            if return_max_config is not None and getattr(return_max_config, "enabled", False):
                result.adjusted_weights = self._apply_return_maximization(
                    weights=result.adjusted_weights,
                    returns_df=returns_df,
                    previous_weights=previous_weights,
                    raw_data=raw_data,
                    excluded_assets=excluded_assets,
                    get_asset_signal_score=get_asset_signal_score,
                )

        except ImportError as e:
            self._logger.warning(f"DynamicWeighter not available: {e}")
        except Exception as e:
            self._logger.warning(f"Dynamic weighting failed: {e}")

        return result

    def _apply_return_maximization(
        self,
        weights: Dict[str, float],
        returns_df: "pd.DataFrame | None",
        previous_weights: Dict[str, float],
        raw_data: Dict[str, Any],
        excluded_assets: set,
        get_asset_signal_score: callable,
    ) -> Dict[str, float]:
        """
        Apply return maximization features.

        Integrates:
        - Kelly allocation sizing
        - Hysteresis filtering
        - Macro timing adjustments
        """
        if not RETURN_MAXIMIZATION_AVAILABLE:
            self._logger.warning("Return maximization modules not available")
            return weights

        return_max_config = getattr(self._settings, "return_maximization", None)
        if return_max_config is None:
            return weights

        adjusted_weights = weights.copy()

        # 1. Kelly Allocation
        kelly_config = getattr(return_max_config, "kelly", None)
        if kelly_config is not None and getattr(kelly_config, "enabled", False):
            try:
                adjusted_weights = self._apply_kelly_allocation(
                    adjusted_weights, kelly_config, returns_df, raw_data, excluded_assets
                )
            except Exception as e:
                self._logger.warning(f"Kelly allocation failed: {e}")

        # 2. Hysteresis Filter
        entry_exit_config = getattr(return_max_config, "entry_exit", None)
        if entry_exit_config is not None and getattr(entry_exit_config, "use_hysteresis", False):
            try:
                adjusted_weights = self._apply_hysteresis_filter(
                    adjusted_weights, entry_exit_config, previous_weights, get_asset_signal_score
                )
            except Exception as e:
                self._logger.warning(f"Hysteresis filter failed: {e}")

        # 3. Macro Timing
        macro_config = getattr(return_max_config, "macro_timing", None)
        if macro_config is not None and getattr(macro_config, "enabled", False):
            try:
                adjusted_weights = self._apply_macro_timing(adjusted_weights, macro_config)
            except Exception as e:
                self._logger.warning(f"Macro timing failed: {e}")

        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        self._logger.info(
            f"Return maximization completed: kelly={getattr(kelly_config, 'enabled', False) if kelly_config else False}, "
            f"hysteresis={getattr(entry_exit_config, 'use_hysteresis', False) if entry_exit_config else False}, "
            f"macro={getattr(macro_config, 'enabled', False) if macro_config else False}"
        )

        return adjusted_weights

    def _apply_kelly_allocation(
        self,
        adjusted_weights: Dict[str, float],
        kelly_config,
        returns_df: "pd.DataFrame | None",
        raw_data: Dict[str, Any],
        excluded_assets: set,
    ) -> Dict[str, float]:
        """Apply Kelly allocation sizing."""
        kelly_fraction = getattr(kelly_config, "fraction", 0.25)
        kelly_max_weight = getattr(kelly_config, "max_weight", 0.25)
        kelly_weight_in_final = getattr(kelly_config, "weight_in_final", 0.5)
        min_trades = getattr(kelly_config, "min_trades", 20)

        allocator = KellyAllocator(fraction=kelly_fraction, max_weight=kelly_max_weight)

        kelly_weights = {}
        for symbol, weight in adjusted_weights.items():
            if symbol == "CASH" or symbol in excluded_assets:
                kelly_weights[symbol] = weight
                continue

            if symbol in raw_data and returns_df is not None:
                try:
                    df = raw_data[symbol]
                    if hasattr(df, "to_pandas"):
                        df = df.to_pandas()
                    if "close" in df.columns:
                        asset_returns = df["close"].pct_change().dropna()
                        if len(asset_returns) >= min_trades:
                            kelly_result = allocator.calculate_strategy_kelly(asset_returns)
                            kelly_weights[symbol] = kelly_result.adjusted_kelly
                        else:
                            kelly_weights[symbol] = weight
                    else:
                        kelly_weights[symbol] = weight
                except Exception as e:
                    logger.debug(f"Kelly calculation failed for {symbol}: {e}")
                    kelly_weights[symbol] = weight
            else:
                kelly_weights[symbol] = weight

        # Blend Kelly weights with base weights
        for symbol in adjusted_weights:
            if symbol in kelly_weights:
                base_weight = adjusted_weights[symbol]
                kelly_weight = kelly_weights[symbol]
                adjusted_weights[symbol] = (
                    (1 - kelly_weight_in_final) * base_weight +
                    kelly_weight_in_final * kelly_weight
                )

        self._logger.debug(f"Kelly allocation applied: {len(kelly_weights)} assets")
        return adjusted_weights

    def _apply_hysteresis_filter(
        self,
        adjusted_weights: Dict[str, float],
        entry_exit_config,
        previous_weights: Dict[str, float],
        get_asset_signal_score: callable,
    ) -> Dict[str, float]:
        """Apply hysteresis filter."""
        entry_threshold = getattr(entry_exit_config, "entry_threshold", 0.3)
        exit_threshold = getattr(entry_exit_config, "exit_threshold", 0.1)

        hysteresis = HysteresisFilter(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        for symbol in list(adjusted_weights.keys()):
            if symbol == "CASH":
                continue

            signal_score = get_asset_signal_score(symbol)
            if signal_score is not None:
                was_holding = previous_weights.get(symbol, 0) > 0.01
                filtered_score = hysteresis.filter_signal(
                    asset_id=symbol, raw_score=signal_score,
                )
                if filtered_score < exit_threshold and was_holding:
                    adjusted_weights[symbol] *= 0.5

        self._logger.debug("Hysteresis filter applied")
        return adjusted_weights

    def _apply_macro_timing(
        self,
        adjusted_weights: Dict[str, float],
        macro_config,
    ) -> Dict[str, float]:
        """Apply macro timing adjustments."""
        cycle_weight = getattr(macro_config, "cycle_allocation_weight", 0.3)
        cycle_allocator = EconomicCycleAllocator()

        phase_result = cycle_allocator.get_current_phase()
        if phase_result is not None:
            cycle_weights = cycle_allocator.get_recommended_weights(phase_result.phase)
            for symbol in adjusted_weights:
                if symbol in cycle_weights:
                    adjusted_weights[symbol] = (
                        (1 - cycle_weight) * adjusted_weights[symbol] +
                        cycle_weight * cycle_weights[symbol]
                    )
            self._logger.debug(
                f"Macro timing applied: phase={phase_result.phase.value if phase_result else 'unknown'}"
            )

        return adjusted_weights
