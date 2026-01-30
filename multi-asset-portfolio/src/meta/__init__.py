"""
Meta Layer Package - 戦略アンサンブル（メタ学習層）

このパッケージは、複数の戦略を統合してポートフォリオを構築するための
Meta層コンポーネントを提供する。

主要コンポーネント:
- StrategyScorer: 戦略スコアリング（Sharpe調整 - ペナルティ）
- StrategyWeighter: 戦略重み計算（softmax + 上限制約）
- EntropyController: エントロピー下限制御（多様性維持）
- MetaLearner: Meta層統合（Scorer + Weighter + EntropyController）
- EnsembleCombiner: アンサンブル統合（stacking/voting/weighted_avg）

設計根拠:
- 要求.md §7: Strategyの採用と重み
- 過剰適応対策: β固定または上限付き
- エントロピー下限で多様性維持
- 「直近で偶然勝っただけ」を避ける
"""

from .ensemble_combiner import (
    EnsembleCombiner,
    EnsembleCombinerConfig,
    EnsembleCombineResult,
    EnsembleMethod,
)
from .entropy_controller import EntropyController, EntropyControlResult
from .learner import (
    MetaLearner,
    MetaLearnerConfig,
    MetaLearnerResult,
    StrategyWeight,
    create_meta_learner_from_settings,
    create_hierarchical_meta_learner,
)
from .scorer import ScorerConfig, StrategyScorer, StrategyScoreResult
from .weighter import StrategyWeighter, WeighterConfig, WeightingResult
from .dynamic_weighter import (
    DynamicWeighter,
    DynamicWeightingConfig,
    DynamicWeightingResult,
    RegimeAdaptiveAllocator,
    AdaptiveRegimeConfig,
    RegimeCondition,
    RegimeAdjustment,
    compute_market_state_from_prices,
)
from .adaptive_lookback import (
    AdaptiveLookback,
    AdaptiveLookbackResult,
    LookbackConfig,
    MarketRegime,
    DEFAULT_REGIME_CONFIGS,
)
from .bayesian_optimizer import (
    BayesianOptimizer,
    OptimizerConfig,
    OptimizationResult,
    CrossValidationResult,
    get_default_search_space,
    get_signal_search_space,
    get_allocation_search_space,
    create_optimizer_from_settings,
    quick_optimize,
)
from .hierarchical_ensemble import (
    HierarchicalEnsemble,
    HierarchicalEnsembleConfig,
    HierarchicalEnsembleResult,
    SignalLayer,
    LayerConfig,
    LayerResult,
    LayerWeights,
    StackingModelType,
    REGIME_LAYER_WEIGHTS,
    create_default_hierarchical_ensemble,
    create_hierarchical_ensemble_with_signals,
)
from .dynamic_scorer_params import (
    DynamicScorerParams,
    DynamicScorerParamsCalculator,
    StrategyMetrics,
    MarketConditions,
    create_dynamic_scorer_params,
    update_scorer_config_dynamically,
)
from .param_consistency import (
    ParameterConsistencyChecker,
    ConsistencyResult,
    ConsistencyIssue,
    ConsistencyLevel,
    check_consistency,
    ensure_consistency,
    create_consistency_checker,
)
from .dynamic_params import (
    DynamicParamsManager,
    DynamicParamsBundle,
    RegimeAdaptiveParams,
    ThresholdType,
    VolatilityRegime,
    ThresholdResult,
    RebalanceThresholdResult,
    SmoothingAlphaResult,
    PositionLimitResult,
    VixThresholdResult,
    CorrelationThresholdResult,
    KellyResult,
    calculate_rebalance_threshold,
    calculate_smoothing_alpha,
    calculate_position_limit,
    get_regime_params,
    create_params_manager,
    detect_volatility_regime,
    detect_market_regime,
)

__all__ = [
    # Scorer
    "StrategyScorer",
    "ScorerConfig",
    "StrategyScoreResult",
    # Weighter
    "StrategyWeighter",
    "WeighterConfig",
    "WeightingResult",
    # Entropy Controller
    "EntropyController",
    "EntropyControlResult",
    # Meta Learner
    "MetaLearner",
    "MetaLearnerConfig",
    "MetaLearnerResult",
    "StrategyWeight",
    "create_meta_learner_from_settings",
    "create_hierarchical_meta_learner",
    # Ensemble Combiner
    "EnsembleCombiner",
    "EnsembleCombinerConfig",
    "EnsembleCombineResult",
    "EnsembleMethod",
    # Dynamic Weighter
    "DynamicWeighter",
    "DynamicWeightingConfig",
    "DynamicWeightingResult",
    # Regime Adaptive Allocator
    "RegimeAdaptiveAllocator",
    "AdaptiveRegimeConfig",
    "RegimeCondition",
    "RegimeAdjustment",
    "compute_market_state_from_prices",
    # Adaptive Lookback
    "AdaptiveLookback",
    "AdaptiveLookbackResult",
    "LookbackConfig",
    "MarketRegime",
    "DEFAULT_REGIME_CONFIGS",
    # Bayesian Optimizer
    "BayesianOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "CrossValidationResult",
    "get_default_search_space",
    "get_signal_search_space",
    "get_allocation_search_space",
    "create_optimizer_from_settings",
    "quick_optimize",
    # Hierarchical Ensemble
    "HierarchicalEnsemble",
    "HierarchicalEnsembleConfig",
    "HierarchicalEnsembleResult",
    "SignalLayer",
    "LayerConfig",
    "LayerResult",
    "LayerWeights",
    "StackingModelType",
    "REGIME_LAYER_WEIGHTS",
    "create_default_hierarchical_ensemble",
    "create_hierarchical_ensemble_with_signals",
    # Dynamic Scorer Params
    "DynamicScorerParams",
    "DynamicScorerParamsCalculator",
    "StrategyMetrics",
    "MarketConditions",
    "create_dynamic_scorer_params",
    "update_scorer_config_dynamically",
    # Parameter Consistency
    "ParameterConsistencyChecker",
    "ConsistencyResult",
    "ConsistencyIssue",
    "ConsistencyLevel",
    "check_consistency",
    "ensure_consistency",
    "create_consistency_checker",
    # Dynamic Params Manager
    "DynamicParamsManager",
    "DynamicParamsBundle",
    "RegimeAdaptiveParams",
    "ThresholdType",
    "VolatilityRegime",
    "ThresholdResult",
    "RebalanceThresholdResult",
    "SmoothingAlphaResult",
    "PositionLimitResult",
    "VixThresholdResult",
    "CorrelationThresholdResult",
    "KellyResult",
    "calculate_rebalance_threshold",
    "calculate_smoothing_alpha",
    "calculate_position_limit",
    "get_regime_params",
    "create_params_manager",
    "detect_volatility_regime",
    "detect_market_regime",
]
