"""Backtest module for multi-asset portfolio system.

This module provides tools for backtesting portfolio strategies:
- PortfolioSimulator: Time-series portfolio simulation with transaction costs
- BacktestResult: Result container for backtest runs
- DailySnapshot: Daily portfolio state snapshot
- BacktestEngine: Walk-forward backtest engine with parallelization
- SignalCache: Signal computation caching for performance
"""

from .cache import (
    DataFrameCache,
    IncrementalCache,
    LRUCache,
    SignalCache,
    batch_compute_signals,
    vectorized_momentum,
    vectorized_sharpe,
    vectorized_volatility,
)
from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult as EngineBacktestResult,
    RebalanceFrequency,
    RebalanceRecord,
)
from .result import BacktestResult, DailySnapshot
from .simulator import (
    PortfolioSimulator,
    PortfolioSnapshot,
    RebalanceResult,
    Trade,
)
from .long_term_validation import (
    LongTermValidator,
    ValidationResult,
    SubPeriodResult,
    BenchmarkComparison,
    SUB_PERIODS,
    run_long_term_validation,
    generate_validation_report,
)
from .adaptive_window import (
    AdaptiveWindowSelector,
    ExpandingWindowOptimizer,
    VolatilityRegime,
    WindowResult,
    RegimeChangeResult,
    WindowSplit,
    compute_adaptive_window,
    generate_expanding_splits,
)
from .synthetic_data import (
    SyntheticDataGenerator,
    StatisticalSignificanceTester,
    BootstrapResult,
    SimulationResult,
    SignificanceTestResult,
    bootstrap_returns,
    test_sharpe_significance,
)
from .purged_kfold import (
    PurgedKFold,
    PurgedKFoldConfig,
    CombinatorialPurgedCV,
    FoldInfo,
    CVSummary,
    create_purged_kfold,
    create_combinatorial_purged_cv,
    calculate_optimal_n_splits,
    validate_no_leakage,
)
from .vectorized_compute import (
    compute_all_momentum_vectorized,
    compute_all_volatility_vectorized,
    compute_all_zscore_vectorized,
    compute_covariance_ewm,
    compute_correlation_matrix,
    compute_rolling_correlation_matrix,
    compute_all_rsi_vectorized,
    compute_all_bollinger_vectorized,
    compute_returns_matrix,
    compute_sharpe_vectorized,
    fast_ewm_covariance,
    fast_rolling_std,
)
from .incremental_signal import (
    SignalState,
    MomentumState,
    ROCState,
    ZScoreState,
    RSIState,
    BollingerState,
    EMAState,
    IncrementalSignalEngine,
    SignalConfig,
    SIGNAL_STATE_REGISTRY,
    create_signal_state,
    create_incremental_engine,
    get_available_signals,
)
from .signal_precompute import SignalPrecomputer
from .gpu_compute import (
    GPU_AVAILABLE,
    is_gpu_available,
    get_device_info,
    covariance_gpu,
    correlation_gpu,
    matrix_multiply_gpu,
    matrix_inverse_gpu,
    solve_linear_gpu,
    eigendecomposition_gpu,
    cholesky_gpu,
    rolling_mean_gpu,
    rolling_std_gpu,
    ewm_covariance_gpu,
    portfolio_variance_gpu,
    batch_portfolio_variance_gpu,
    GPUComputeContext,
)
# Numba compute is optional
try:
    from .numba_compute import (
        NUMBA_AVAILABLE,
        momentum_batch,
        momentum_single,
        volatility_batch,
        volatility_ewm,
        covariance_matrix as numba_covariance_matrix,
        correlation_matrix as numba_correlation_matrix,
        zscore_batch,
        rsi_batch,
        sharpe_batch,
        returns_from_prices,
        check_numba_available,
        warmup_jit,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    momentum_batch = None
    momentum_single = None
    volatility_batch = None
    volatility_ewm = None
    numba_covariance_matrix = None
    numba_correlation_matrix = None
    zscore_batch = None
    rsi_batch = None
    sharpe_batch = None
    returns_from_prices = None
    check_numba_available = lambda: False
    warmup_jit = lambda: None
from .score_vectorized import (
    ScoringConfig,
    compute_scores_vectorized,
    compute_scores_at_date,
    VectorizedScorer,
    IncrementalScorer,
    scores_to_weights,
)
from .fast_engine import (
    FastBacktestConfig,
    FastBacktestEngine,
    SimulationState,
    SimulationResult as FastSimulationResult,
    run_fast_backtest,
)
from .streaming_engine import (
    StreamingBacktestEngine,
    StreamingBacktestResult,
    run_streaming_backtest,
)
# Unified base interface (INT-001)
from .base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    TradeRecord,
    RebalanceRecord as UnifiedRebalanceRecord,
    WeightsFuncProtocol,
    RebalanceFrequency as UnifiedRebalanceFrequency,
)
# Engine Factory (INT-005)
from .factory import (
    BacktestEngineFactory,
    EngineMode,
    EngineInfo,
    ENGINE_REGISTRY,
    create_engine,
    list_engines,
    recommend_engine,
)
from .streaming_covariance import (
    StreamingCovariance,
    StreamingCovarianceConfig,
    StreamingCovarianceResult,
    create_streaming_covariance,
    streaming_covariance_from_returns,
    compare_memory_usage,
)
# Ray engine is optional (HI-004)
try:
    from .ray_engine import (
        RAY_AVAILABLE,
        RayBacktestConfig,
        BacktestChunkResult,
        RayBacktestEngine,
        run_benchmark,
    )
except ImportError:
    RAY_AVAILABLE = False
    RayBacktestConfig = None
    BacktestChunkResult = None
    RayBacktestEngine = None
    run_benchmark = None
# VectorBT-style engine (HI-007)
from .vectorbt_engine import (
    VectorBTStyleEngine,
    VectorBTConfig,
    VectorBTResult,
    NUMBA_AVAILABLE as VECTORBT_NUMBA_AVAILABLE,
    create_vectorbt_engine,
    quick_backtest,
)
# Memory optimizer (HI-006)
from .memory_optimizer import (
    MemoryOptimizer,
    MemoryTracker,
    MemoryStats,
    StreamingDataProcessor,
    memory_efficient_context,
    estimate_memory_usage,
    optimize_backtest_memory,
)
# Numba accelerate is optional
try:
    from .numba_accelerate import (
        calculate_max_drawdown_numba,
        calculate_max_drawdown_pct_numba,
        normalize_scores_rank_numba,
        normalize_scores_zscore_numba,
        normalize_scores_minmax_numba,
        calculate_sharpe_ratio_numba,
        calculate_sortino_ratio_numba,
        calculate_calmar_ratio_numba,
        calculate_volatility_numba,
    )
except ImportError:
    calculate_max_drawdown_numba = None
    calculate_max_drawdown_pct_numba = None
    normalize_scores_rank_numba = None
    normalize_scores_zscore_numba = None
    normalize_scores_minmax_numba = None
    calculate_sharpe_ratio_numba = None
    calculate_sortino_ratio_numba = None
    calculate_calmar_ratio_numba = None
    calculate_volatility_numba = None

__all__ = [
    # Original exports
    "BacktestResult",
    "DailySnapshot",
    "PortfolioSimulator",
    "PortfolioSnapshot",
    "RebalanceResult",
    "Trade",
    # Engine exports
    "BacktestConfig",
    "BacktestEngine",
    "EngineBacktestResult",
    "RebalanceFrequency",
    "RebalanceRecord",
    # Cache exports
    "SignalCache",
    "LRUCache",
    "DataFrameCache",
    "IncrementalCache",
    "vectorized_momentum",
    "vectorized_volatility",
    "vectorized_sharpe",
    "batch_compute_signals",
    # Long-term validation exports
    "LongTermValidator",
    "ValidationResult",
    "SubPeriodResult",
    "BenchmarkComparison",
    "SUB_PERIODS",
    "run_long_term_validation",
    "generate_validation_report",
    # Adaptive window exports
    "AdaptiveWindowSelector",
    "ExpandingWindowOptimizer",
    "VolatilityRegime",
    "WindowResult",
    "RegimeChangeResult",
    "WindowSplit",
    "compute_adaptive_window",
    "generate_expanding_splits",
    # Synthetic data exports
    "SyntheticDataGenerator",
    "StatisticalSignificanceTester",
    "BootstrapResult",
    "SimulationResult",
    "SignificanceTestResult",
    "bootstrap_returns",
    "test_sharpe_significance",
    # Purged K-Fold CV exports
    "PurgedKFold",
    "PurgedKFoldConfig",
    "CombinatorialPurgedCV",
    "FoldInfo",
    "CVSummary",
    "create_purged_kfold",
    "create_combinatorial_purged_cv",
    "calculate_optimal_n_splits",
    "validate_no_leakage",
    # Vectorized compute exports
    "compute_all_momentum_vectorized",
    "compute_all_volatility_vectorized",
    "compute_all_zscore_vectorized",
    "compute_covariance_ewm",
    "compute_correlation_matrix",
    "compute_rolling_correlation_matrix",
    "compute_all_rsi_vectorized",
    "compute_all_bollinger_vectorized",
    "compute_returns_matrix",
    "compute_sharpe_vectorized",
    "fast_ewm_covariance",
    "fast_rolling_std",
    # Signal precompute exports
    "SignalPrecomputer",
    # Incremental signal exports
    "SignalState",
    "MomentumState",
    "ROCState",
    "ZScoreState",
    "RSIState",
    "BollingerState",
    "EMAState",
    "IncrementalSignalEngine",
    "SignalConfig",
    "SIGNAL_STATE_REGISTRY",
    "create_signal_state",
    "create_incremental_engine",
    "get_available_signals",
    # GPU compute exports
    "GPU_AVAILABLE",
    "is_gpu_available",
    "get_device_info",
    "covariance_gpu",
    "correlation_gpu",
    "matrix_multiply_gpu",
    "matrix_inverse_gpu",
    "solve_linear_gpu",
    "eigendecomposition_gpu",
    "cholesky_gpu",
    "rolling_mean_gpu",
    "rolling_std_gpu",
    "ewm_covariance_gpu",
    "portfolio_variance_gpu",
    "batch_portfolio_variance_gpu",
    "GPUComputeContext",
    # Numba compute exports
    "NUMBA_AVAILABLE",
    "momentum_batch",
    "momentum_single",
    "volatility_batch",
    "volatility_ewm",
    "numba_covariance_matrix",
    "numba_correlation_matrix",
    "zscore_batch",
    "rsi_batch",
    "sharpe_batch",
    "returns_from_prices",
    "check_numba_available",
    "warmup_jit",
    # Score vectorized exports
    "ScoringConfig",
    "compute_scores_vectorized",
    "compute_scores_at_date",
    "VectorizedScorer",
    "IncrementalScorer",
    "scores_to_weights",
    # Fast engine exports (Numba/GPU integrated)
    "FastBacktestConfig",
    "FastBacktestEngine",
    "SimulationState",
    "FastSimulationResult",
    "run_fast_backtest",
    # Streaming engine exports
    "StreamingBacktestEngine",
    "StreamingBacktestResult",
    "run_streaming_backtest",
    # Streaming covariance exports (HI-005)
    "StreamingCovariance",
    "StreamingCovarianceConfig",
    "StreamingCovarianceResult",
    "create_streaming_covariance",
    "streaming_covariance_from_returns",
    "compare_memory_usage",
    # Numba accelerate exports (JIT高速化関数)
    "calculate_max_drawdown_numba",
    "calculate_max_drawdown_pct_numba",
    "normalize_scores_rank_numba",
    "normalize_scores_zscore_numba",
    "normalize_scores_minmax_numba",
    "calculate_sharpe_ratio_numba",
    "calculate_sortino_ratio_numba",
    "calculate_calmar_ratio_numba",
    "calculate_volatility_numba",
    # Ray distributed engine exports (HI-004)
    "RAY_AVAILABLE",
    "RayBacktestConfig",
    "BacktestChunkResult",
    "RayBacktestEngine",
    "run_benchmark",
    # VectorBT-style engine exports (HI-007)
    "VectorBTStyleEngine",
    "VectorBTConfig",
    "VectorBTResult",
    "VECTORBT_NUMBA_AVAILABLE",
    "create_vectorbt_engine",
    "quick_backtest",
    # Memory optimizer exports (HI-006)
    "MemoryOptimizer",
    "MemoryTracker",
    "MemoryStats",
    "StreamingDataProcessor",
    "memory_efficient_context",
    "estimate_memory_usage",
    "optimize_backtest_memory",
    # Unified base interface exports (INT-001)
    "BacktestEngineBase",
    "UnifiedBacktestConfig",
    "UnifiedBacktestResult",
    "TradeRecord",
    "UnifiedRebalanceRecord",
    "WeightsFuncProtocol",
    "UnifiedRebalanceFrequency",
    # Engine Factory exports (INT-005)
    "BacktestEngineFactory",
    "EngineMode",
    "EngineInfo",
    "ENGINE_REGISTRY",
    "create_engine",
    "list_engines",
    "recommend_engine",
]
