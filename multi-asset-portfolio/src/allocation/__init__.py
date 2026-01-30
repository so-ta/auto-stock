"""
Allocation Module - Asset配分層

ポートフォリオのアセット配分を計算するモジュール群。

主要コンポーネント:
- CovarianceEstimator: 共分散推定（Ledoit-Wolf, EWMA, Sample）
- HierarchicalRiskParity: HRP配分
- RiskParity: リスクパリティ配分
- ConstraintProcessor: 制約処理
- WeightSmoother: スムージング
- AssetAllocator: 統合配分クラス

設計根拠:
- 要求.md §8: 全アセットの重み付け
- HRP/RP推奨、推定誤差に比較的強い

使用例:
    from src.allocation import AssetAllocator, AllocatorConfig, AllocationMethod

    config = AllocatorConfig(
        method=AllocationMethod.HRP,
        w_asset_max=0.2,
        smooth_alpha=0.3,
    )
    allocator = AssetAllocator(config)

    result = allocator.allocate(
        returns=daily_returns_df,
        previous_weights=prev_weights,
    )

    print(result.weights)
"""

from .allocator import (
    AllocationMethod,
    AllocationResult,
    AllocatorConfig,
    AssetAllocator,
    FallbackReason,
    create_allocator_from_settings,
)
from .constraints import (
    ConstraintConfig,
    ConstraintProcessor,
    ConstraintResult,
    ConstraintViolation,
    ConstraintViolationType,
    create_processor_from_settings,
)
from .covariance import (
    CovarianceConfig,
    CovarianceEstimator,
    CovarianceMethod,
    CovarianceResult,
    create_estimator_from_settings,
)
from .hrp import (
    HierarchicalRiskParity,
    HRPConfig,
    HRPResult,
    create_hrp_from_settings,
)
from .risk_parity import (
    NaiveRiskParity,
    RiskParity,
    RiskParityConfig,
    RiskParityResult,
    create_risk_parity_from_settings,
)
from .smoother import (
    AdaptiveSmoother,
    SmootherConfig,
    SmoothingResult,
    WeightSmoother,
    create_smoother_from_settings,
)
from .kelly_allocator import (
    KellyAllocator,
    KellyConfig,
    KellyResult,
    KellyAllocationResult,
    create_kelly_allocator_from_settings,
)
from .dynamic_allocation_params import (
    AllocationDynamicParamsCalculator,
    DynamicAllocationParams,
    WAssetMaxResult,
    DeltaMaxResult,
    SmoothAlphaResult,
    VolatilityRegime as AllocationVolatilityRegime,
    calculate_allocation_params,
    get_w_asset_max,
    get_delta_max,
    get_smooth_alpha,
    create_dynamic_allocator_config,
    detect_volatility_regime as detect_allocation_volatility_regime,
)
from .return_estimator import (
    DynamicReturnEstimator,
    DynamicReturnEstimatorConfig,
    ReturnEstimate,
    MarketRegime as ReturnMarketRegime,
    CrossSectionalMomentum,
    ImpliedReturns,
    MeanReversionForecast,
    FactorPremiumTiming,
    ReturnEstimatorBase,
    create_return_estimator,
    quick_estimate_returns,
)
from .cvar_optimizer import (
    CVaROptimizer,
    CVaRConfig,
    CVaRResult,
    OptimizationResult as CVaROptimizationResult,
    PortfolioMetrics,
    compute_cvar,
    optimize_cvar,
    create_cvar_optimizer,
)
from .nco import (
    NestedClusteredOptimization,
    NCOConfig,
    NCOResult,
    ClusterInfo,
    IntraClusterMethod,
    InterClusterMethod,
    create_nco_from_settings,
    quick_nco_allocation,
)
from .black_litterman import (
    BlackLittermanModel,
    BlackLittermanConfig,
    BlackLittermanResult,
    ViewSet,
    ViewGenerator,
    create_black_litterman_model,
    quick_bl_allocation,
)
from .executor import (
    AllocationExecutor,
    ExecutorConfig,
    ExecutionResult,
    execute_with_vix,
)

__all__ = [
    # Allocator (Main)
    "AssetAllocator",
    "AllocatorConfig",
    "AllocationResult",
    "AllocationMethod",
    "FallbackReason",
    "create_allocator_from_settings",
    # Covariance
    "CovarianceEstimator",
    "CovarianceConfig",
    "CovarianceResult",
    "CovarianceMethod",
    "create_estimator_from_settings",
    # HRP
    "HierarchicalRiskParity",
    "HRPConfig",
    "HRPResult",
    "create_hrp_from_settings",
    # Risk Parity
    "RiskParity",
    "RiskParityConfig",
    "RiskParityResult",
    "NaiveRiskParity",
    "create_risk_parity_from_settings",
    # Constraints
    "ConstraintProcessor",
    "ConstraintConfig",
    "ConstraintResult",
    "ConstraintViolation",
    "ConstraintViolationType",
    "create_processor_from_settings",
    # Smoother
    "WeightSmoother",
    "SmootherConfig",
    "SmoothingResult",
    "AdaptiveSmoother",
    "create_smoother_from_settings",
    # Kelly Allocator
    "KellyAllocator",
    "KellyConfig",
    "KellyResult",
    "KellyAllocationResult",
    "create_kelly_allocator_from_settings",
    # Dynamic Allocation Params
    "AllocationDynamicParamsCalculator",
    "DynamicAllocationParams",
    "WAssetMaxResult",
    "DeltaMaxResult",
    "SmoothAlphaResult",
    "AllocationVolatilityRegime",
    "calculate_allocation_params",
    "get_w_asset_max",
    "get_delta_max",
    "get_smooth_alpha",
    "create_dynamic_allocator_config",
    "detect_allocation_volatility_regime",
    # Return Estimator
    "DynamicReturnEstimator",
    "DynamicReturnEstimatorConfig",
    "ReturnEstimate",
    "ReturnMarketRegime",
    "CrossSectionalMomentum",
    "ImpliedReturns",
    "MeanReversionForecast",
    "FactorPremiumTiming",
    "ReturnEstimatorBase",
    "create_return_estimator",
    "quick_estimate_returns",
    # CVaR Optimizer
    "CVaROptimizer",
    "CVaRConfig",
    "CVaRResult",
    "CVaROptimizationResult",
    "PortfolioMetrics",
    "compute_cvar",
    "optimize_cvar",
    "create_cvar_optimizer",
    # NCO (Nested Clustered Optimization)
    "NestedClusteredOptimization",
    "NCOConfig",
    "NCOResult",
    "ClusterInfo",
    "IntraClusterMethod",
    "InterClusterMethod",
    "create_nco_from_settings",
    "quick_nco_allocation",
    # Black-Litterman
    "BlackLittermanModel",
    "BlackLittermanConfig",
    "BlackLittermanResult",
    "ViewSet",
    "ViewGenerator",
    "create_black_litterman_model",
    "quick_bl_allocation",
    # Executor (VIX Integration)
    "AllocationExecutor",
    "ExecutorConfig",
    "ExecutionResult",
    "execute_with_vix",
]
