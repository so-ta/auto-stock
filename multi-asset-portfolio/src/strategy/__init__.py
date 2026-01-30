"""Strategy module for validation, metrics, and evaluation.

This module provides:
- Strategy base class for position proposal generation (Phase 2B)
- Walk-forward validation engine (Phase 3A)
- Time series cross-validation (Phase 3A)
- Gate checker for strategy adoption (Phase 3B)
- Strategy evaluator integrating all components (Phase 3B)
"""

from .base import (
    CompositeStrategy,
    PositionMode,
    PositionProposal,
    SingleSignalStrategy,
    Strategy,
    StrategyConfig,
    StrategyRegistry,
)
from .walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardFold,
    create_walk_forward_splits,
)
from .time_series_cv import (
    TimeSeriesCV,
    TimeSeriesCVConfig,
    CVFold,
    CVResult,
    PurgedKFold,
    CombinatorialPurgedKFold,
    time_series_split,
)
from .gate_checker import (
    GateChecker,
    GateCheckResult,
    GateConfig,
    GateResult,
    GateType,
    StrategyMetrics,
)
from .evaluator import (
    AssetEvaluationResult,
    DefaultMetricsCalculator,
    EvaluationReport,
    EvaluationStatus,
    EvaluatorConfig,
    StrategyEvaluationResult,
    StrategyEvaluator,
)
from .entry_exit_optimizer import (
    HysteresisFilter,
    GradualEntryExit,
    StopLossManager,
    EntryExitOptimizer,
    EntryExitConfig,
    PositionState,
    StopType,
    PositionInfo,
    FilterResult,
    StopLossResult,
)
from .sector_rotation import (
    EconomicCycleSectorRotator,
    MomentumSectorRotator,
    EconomicPhase,
    SectorCategory,
    MacroIndicators,
    PhaseDetectionResult,
    SectorAdjustment,
    RotationResult,
    SECTOR_ETFS,
    CYCLE_SECTOR_RECOMMENDATIONS,
    create_economic_cycle_rotator,
    create_momentum_rotator,
    get_all_sector_etfs,
    get_sector_category,
)
from .pairs_trading import (
    CointegrationPairsFinder,
    PairsTrader,
    PairsTradingStrategy,
    PairsTraderConfig,
    CointegrationResult,
    PairPosition,
    PairsTradingSignal,
    PairSignal,
    CointegrationMethod,
    RECOMMENDED_PAIRS,
    find_cointegrated_pairs,
    generate_pairs_signals,
    get_recommended_pairs,
)
from .min_holding_period import (
    MinHoldingPeriodFilter,
    MinHoldingPeriodConfig,
    HoldingInfo,
    TradeDecision,
    TradeAction,
    PositionDirection,
    apply_min_holding,
    create_min_holding_filter,
    get_filtered_signals,
)

__all__ = [
    # Strategy base (Phase 2B)
    "Strategy",
    "StrategyConfig",
    "StrategyRegistry",
    "PositionMode",
    "PositionProposal",
    "SingleSignalStrategy",
    "CompositeStrategy",
    # Walk-forward (Phase 3A)
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardFold",
    "create_walk_forward_splits",
    # Time series CV (Phase 3A)
    "TimeSeriesCV",
    "TimeSeriesCVConfig",
    "CVFold",
    "CVResult",
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "time_series_split",
    # Gate checker (Phase 3B)
    "GateChecker",
    "GateCheckResult",
    "GateConfig",
    "GateResult",
    "GateType",
    "StrategyMetrics",
    # Evaluator (Phase 3B)
    "AssetEvaluationResult",
    "DefaultMetricsCalculator",
    "EvaluationReport",
    "EvaluationStatus",
    "EvaluatorConfig",
    "StrategyEvaluationResult",
    "StrategyEvaluator",
    # Entry/Exit Optimizer
    "HysteresisFilter",
    "GradualEntryExit",
    "StopLossManager",
    "EntryExitOptimizer",
    "EntryExitConfig",
    "PositionState",
    "StopType",
    "PositionInfo",
    "FilterResult",
    "StopLossResult",
    # Sector Rotation
    "EconomicCycleSectorRotator",
    "MomentumSectorRotator",
    "EconomicPhase",
    "SectorCategory",
    "MacroIndicators",
    "PhaseDetectionResult",
    "SectorAdjustment",
    "RotationResult",
    "SECTOR_ETFS",
    "CYCLE_SECTOR_RECOMMENDATIONS",
    "create_economic_cycle_rotator",
    "create_momentum_rotator",
    "get_all_sector_etfs",
    "get_sector_category",
    # Pairs Trading
    "CointegrationPairsFinder",
    "PairsTrader",
    "PairsTradingStrategy",
    "PairsTraderConfig",
    "CointegrationResult",
    "PairPosition",
    "PairsTradingSignal",
    "PairSignal",
    "CointegrationMethod",
    "RECOMMENDED_PAIRS",
    "find_cointegrated_pairs",
    "generate_pairs_signals",
    "get_recommended_pairs",
    # Min Holding Period
    "MinHoldingPeriodFilter",
    "MinHoldingPeriodConfig",
    "HoldingInfo",
    "TradeDecision",
    "TradeAction",
    "PositionDirection",
    "apply_min_holding",
    "create_min_holding_filter",
    "get_filtered_signals",
]
