"""
Weight Calculation Module - Handles gate checks and strategy weighting.

Extracted from pipeline.py for better modularity (QA-003-P2).
This module handles:
1. Gate check (TOP_N and GATE modes)
2. Strategy weighting (TopNSelector and Scorer/Weighter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.utils.logger import AuditLogger

logger = structlog.get_logger(__name__)


@dataclass
class GateCheckResult:
    """Result of gate check step."""

    passed_evaluations: list[Any]
    filtered_count: int
    mode: str


@dataclass
class WeightingResult:
    """Result of strategy weighting step."""

    strategy_weights: dict[str, dict[str, float]]
    cash_weight: float
    adopted_count: int
    mode: str


class GateChecker:
    """
    Handles gate check for filtering strategies.

    Supports two modes:
    - TOP_N: Only filters out failed/incomplete evaluations
    - GATE: Applies hard gate thresholds from settings
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
    ) -> None:
        """
        Initialize GateChecker.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
        """
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="gate_checker")
        self._warnings: list[str] = []

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def check(
        self,
        evaluations: list[Any],
    ) -> tuple[list[Any], list[str]]:
        """
        Apply gate check to filter strategies.

        Args:
            evaluations: List of evaluation results

        Returns:
            Tuple of (passed_evaluations, warnings)
        """
        from src.config.settings import StrategySelectionMethod

        selection_method = self.settings.strategy_selection.method

        if selection_method == StrategySelectionMethod.TOP_N:
            result = self._check_top_n_mode(evaluations)
        else:
            result = self._check_gate_mode(evaluations)

        return result, self._warnings

    def _check_top_n_mode(self, evaluations: list[Any]) -> list[Any]:
        """
        Gate check for TOP_N selection mode.

        Only filters out:
        - Failed evaluations
        - Evaluations without metrics
        """
        from src.strategy.evaluator import EvaluationStatus

        passed_evaluations: list[Any] = []
        filtered_count = 0

        for evaluation in evaluations:
            # Skip failed evaluations
            if evaluation.status == EvaluationStatus.FAILED:
                filtered_count += 1
                continue

            # Skip evaluations without metrics
            if evaluation.metrics is None:
                filtered_count += 1
                continue

            passed_evaluations.append(evaluation)

        self._logger.info(
            "Gate check completed (top_n mode)",
            total=len(passed_evaluations) + filtered_count,
            passed=len(passed_evaluations),
            filtered=filtered_count,
            mode="top_n",
        )

        return passed_evaluations

    def _check_gate_mode(self, evaluations: list[Any]) -> list[Any]:
        """
        Gate check for traditional GATE selection mode.

        Applies hard gate thresholds from settings.hard_gates.
        """
        from src.strategy.evaluator import EvaluationStatus
        from src.strategy.gate_checker import GateCheckResult as GCR
        from src.strategy.gate_checker import GateChecker as GC
        from src.strategy.gate_checker import GateConfig

        # Initialize GateChecker with settings
        hard_gates = self.settings.hard_gates
        gate_config = GateConfig(
            min_trades=hard_gates.min_trades,
            max_drawdown_pct=hard_gates.max_drawdown_pct,
            min_expected_value=hard_gates.min_expected_value,
            min_sharpe_ratio=hard_gates.min_sharpe_ratio,
            min_win_rate_pct=hard_gates.min_win_rate_pct,
            min_profit_factor=hard_gates.min_profit_factor,
        )
        gate_checker = GC(config=gate_config)

        # Check each evaluation against gates
        passed_evaluations: list[Any] = []
        rejected_count = 0
        gate_results: list[GCR] = []

        for evaluation in evaluations:
            # Skip failed evaluations
            if evaluation.status == EvaluationStatus.FAILED:
                continue

            # Skip evaluations without metrics
            if evaluation.metrics is None:
                continue

            # Run gate check
            gate_result = gate_checker.check(evaluation.metrics)
            gate_results.append(gate_result)

            # Update evaluation with gate result
            evaluation.gate_result = gate_result

            if gate_result.passed:
                passed_evaluations.append(evaluation)
                self._logger.debug(
                    "Strategy passed gates",
                    strategy=evaluation.strategy_id,
                    asset=evaluation.asset_id,
                    sharpe=evaluation.metrics.sharpe_ratio,
                )
            else:
                rejected_count += 1
                rejection_reasons = "; ".join(gate_result.rejection_reasons)

                self._logger.info(
                    "Strategy rejected",
                    strategy=evaluation.strategy_id,
                    asset=evaluation.asset_id,
                    reasons=rejection_reasons,
                )

                # Log to AuditLogger for traceability
                if self._audit_logger:
                    self._audit_logger.log_gate_rejection(
                        strategy_id=evaluation.strategy_id,
                        asset_id=evaluation.asset_id,
                        reasons=gate_result.rejection_reasons,
                        metrics={
                            "sharpe_ratio": evaluation.metrics.sharpe_ratio,
                            "max_drawdown_pct": evaluation.metrics.max_drawdown_pct,
                            "win_rate_pct": evaluation.metrics.win_rate_pct,
                            "profit_factor": evaluation.metrics.profit_factor,
                            "trade_count": evaluation.metrics.trade_count,
                            "expected_value": evaluation.metrics.expected_value,
                        },
                    )

        # Handle case where all strategies are rejected
        if not passed_evaluations and evaluations:
            self._warnings.append(
                f"All {len(evaluations)} strategies rejected by gates. "
                "Consider relaxing gate thresholds or improving strategies."
            )
            self._logger.warning(
                "All strategies rejected by gates",
                total_evaluated=len(evaluations),
                gate_config={
                    "min_trades": gate_config.min_trades,
                    "max_drawdown_pct": gate_config.max_drawdown_pct,
                    "min_sharpe": gate_config.min_sharpe_ratio,
                    "min_win_rate_pct": gate_config.min_win_rate_pct,
                },
            )

        # Log summary to AuditLogger
        if self._audit_logger:
            self._audit_logger.log_gate_summary(
                total_evaluated=len(gate_results),
                total_passed=len(passed_evaluations),
                total_rejected=rejected_count,
                gate_config={
                    "min_trades": gate_config.min_trades,
                    "max_drawdown_pct": gate_config.max_drawdown_pct,
                    "min_expected_value": gate_config.min_expected_value,
                    "min_sharpe_ratio": gate_config.min_sharpe_ratio,
                    "min_win_rate_pct": gate_config.min_win_rate_pct,
                    "min_profit_factor": gate_config.min_profit_factor,
                },
            )

        self._logger.info(
            "Gate check completed (gate mode)",
            evaluated=len(gate_results),
            passed=len(passed_evaluations),
            rejected=rejected_count,
            mode="gate",
        )

        return passed_evaluations


class StrategyWeighter:
    """
    Handles strategy weighting calculation.

    Supports two modes:
    - TOP_N: Uses TopNSelector to select top N strategies
    - GATE: Uses traditional Scorer/Weighter per asset
    """

    def __init__(
        self,
        settings: "Settings",
        audit_logger: "AuditLogger | None" = None,
    ) -> None:
        """
        Initialize StrategyWeighter.

        Args:
            settings: Application settings
            audit_logger: Optional audit logger for detailed logging
        """
        self._settings = settings
        self._audit_logger = audit_logger
        self._logger = logger.bind(component="strategy_weighter")

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        return self._settings

    def calculate_weights(
        self,
        evaluations: list[Any],
    ) -> WeightingResult:
        """
        Calculate strategy weights.

        Args:
            evaluations: List of evaluation results (passed gate check)

        Returns:
            WeightingResult with weights and adopted count
        """
        from src.config.settings import StrategySelectionMethod

        selection_method = self.settings.strategy_selection.method

        if selection_method == StrategySelectionMethod.TOP_N:
            return self._weighting_top_n_mode(evaluations)
        else:
            return self._weighting_gate_mode(evaluations)

    def _weighting_top_n_mode(self, evaluations: list[Any]) -> WeightingResult:
        """
        Strategy weighting using TopNSelector.

        Selects top N strategies across all assets and calculates weights
        using softmax. CASH is included as a candidate.

        Returns:
            WeightingResult with weights
        """
        from src.meta.top_n_selector import TopNSelector, TopNSelectorConfig

        strategy_weights: dict[str, dict[str, float]] = {}
        cash_weight = 0.0

        if not evaluations:
            self._logger.warning("No evaluations available for strategy weighting")
            return WeightingResult(
                strategy_weights={},
                cash_weight=0.0,
                adopted_count=0,
                mode="top_n",
            )

        # Get settings
        sel_config = self.settings.strategy_selection

        # Initialize TopNSelector
        selector_config = TopNSelectorConfig(
            n=sel_config.top_n,
            cash_score=sel_config.cash_score,
            min_score=sel_config.min_score,
            softmax_temperature=sel_config.softmax_temperature,
            include_cash=True,
        )
        selector = TopNSelector(config=selector_config)

        # Use select_from_evaluation_results for StrategyEvaluationResult list
        result = selector.select_from_evaluation_results(evaluations)

        # Store strategy weights (asset -> strategy -> weight)
        for strategy in result.selected:
            asset = strategy.asset
            signal = strategy.signal_name
            weight = result.strategy_weights.get(strategy.strategy_key, 0.0)

            if asset not in strategy_weights:
                strategy_weights[asset] = {}
            strategy_weights[asset][signal] = weight

        # Store CASH weight
        cash_weight = result.cash_weight

        # Count adopted strategies (excluding CASH)
        total_adopted = sum(1 for s in result.selected if not s.is_cash)

        # Log to AuditLogger
        if self._audit_logger:
            self._audit_logger.log_top_n_selection(
                n=sel_config.top_n,
                cash_score=sel_config.cash_score,
                selected_count=len(result.selected),
                excluded_count=result.excluded_count,
                cash_selected=result.cash_selected,
                cash_weight=result.cash_weight,
                weights=result.weights,
                strategy_weights=result.strategy_weights,
            )

        self._logger.info(
            "Strategy weighting completed (top_n mode)",
            n=sel_config.top_n,
            selected=len(result.selected),
            adopted=total_adopted,
            cash_selected=result.cash_selected,
            cash_weight=round(result.cash_weight, 4),
            mode="top_n",
        )

        return WeightingResult(
            strategy_weights=strategy_weights,
            cash_weight=cash_weight,
            adopted_count=total_adopted,
            mode="top_n",
        )

    def _weighting_gate_mode(self, evaluations: list[Any]) -> WeightingResult:
        """
        Strategy weighting using traditional Scorer/Weighter per asset.

        Uses StrategyScorer, StrategyWeighter, and EntropyController.

        Returns:
            WeightingResult with weights
        """
        from src.meta.entropy_controller import EntropyConfig, EntropyController
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer
        from src.meta.weighter import StrategyWeighter as SW
        from src.meta.weighter import WeighterConfig

        strategy_weights: dict[str, dict[str, float]] = {}

        if not evaluations:
            self._logger.warning("No evaluations available for strategy weighting")
            return WeightingResult(
                strategy_weights={},
                cash_weight=0.0,
                adopted_count=0,
                mode="gate",
            )

        # Initialize Scorer/Weighter/EntropyController from settings
        sw_config = self.settings.strategy_weighting

        scorer = StrategyScorer(
            config=ScorerConfig(
                penalty_turnover=sw_config.penalty_turnover,
                penalty_mdd=sw_config.penalty_mdd,
                penalty_instability=sw_config.penalty_instability,
            )
        )

        weighter = SW(
            config=WeighterConfig(
                beta=sw_config.beta,
                w_strategy_max=sw_config.w_strategy_max,
            )
        )

        entropy_controller = EntropyController(
            config=EntropyConfig(
                entropy_min=sw_config.entropy_min,
            )
        )

        # Group evaluations by asset
        evaluations_by_asset: dict[str, list] = {}
        for evaluation in evaluations:
            asset_id = evaluation.asset_id
            if asset_id not in evaluations_by_asset:
                evaluations_by_asset[asset_id] = []
            evaluations_by_asset[asset_id].append(evaluation)

        # Calculate weights for each asset
        total_adopted = 0
        total_scored = 0
        entropy_adjusted_count = 0

        for asset_id, asset_evaluations in evaluations_by_asset.items():
            # Create StrategyMetricsInput and score each strategy
            score_results = []

            for evaluation in asset_evaluations:
                if evaluation.metrics is None:
                    continue

                # Create metrics input for scorer
                metrics_input = StrategyMetricsInput(
                    strategy_id=evaluation.strategy_id,
                    asset_id=asset_id,
                    sharpe_ratio=evaluation.metrics.sharpe_ratio,
                    max_drawdown_pct=evaluation.metrics.max_drawdown_pct,
                    turnover=getattr(evaluation.metrics, "turnover", 0.0),
                    period_returns=evaluation.metrics.period_returns or [],
                )

                # Calculate score
                score_result = scorer.score(metrics_input)
                score_results.append(score_result)
                total_scored += 1

            if not score_results:
                self._logger.debug(f"No valid scores for asset {asset_id}")
                continue

            # Calculate weights using softmax
            weighting_result = weighter.calculate_weights(score_results)

            # Apply entropy control for diversity
            raw_weights = weighting_result.get_weight_dict()
            entropy_result = entropy_controller.control(raw_weights)

            if entropy_result.was_adjusted:
                entropy_adjusted_count += 1
                final_weights = entropy_result.adjusted_weights
                self._logger.debug(
                    f"Entropy adjusted for {asset_id}: "
                    f"{entropy_result.original_entropy:.3f} -> "
                    f"{entropy_result.adjusted_entropy:.3f}"
                )
            else:
                final_weights = entropy_result.original_weights

            # Store weights
            strategy_weights[asset_id] = final_weights

            # Count adopted strategies
            asset_adopted = sum(1 for w in final_weights.values() if w > 0)
            total_adopted += asset_adopted

            # Log to AuditLogger
            if self._audit_logger:
                self._audit_logger.log_strategy_weighting(
                    asset_id=asset_id,
                    scores={r.strategy_id: r.final_score for r in score_results},
                    raw_weights=raw_weights,
                    final_weights=final_weights,
                    entropy_before=entropy_result.original_entropy,
                    entropy_after=entropy_result.adjusted_entropy,
                    was_adjusted=entropy_result.was_adjusted,
                    adopted_count=asset_adopted,
                )

        self._logger.info(
            "Strategy weighting completed (gate mode)",
            assets=len(evaluations_by_asset),
            strategies_scored=total_scored,
            adopted_strategies=total_adopted,
            entropy_adjustments=entropy_adjusted_count,
            mode="gate",
        )

        return WeightingResult(
            strategy_weights=strategy_weights,
            cash_weight=0.0,
            adopted_count=total_adopted,
            mode="gate",
        )
