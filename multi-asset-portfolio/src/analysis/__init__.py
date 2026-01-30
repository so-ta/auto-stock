"""
Analysis Package - パフォーマンス分析とチューニング

このパッケージは、ポートフォリオのパフォーマンス分析とチューニング機能を提供する。

主要コンポーネント:
- PerformanceAnalyzer: 統合分析クラス
- StrategyContributionAnalyzer: 戦略別貢献度分析
- DrawdownAnalyzer: ドローダウン分析
- RegimeAnalyzer: 市場レジーム別分析
- ParameterOptimizer: シグナルパラメータ最適化
- AllocationOptimizer: 戦略配分の最適化
- RebalanceOptimizer: リバランス頻度の最適化
"""

from .performance_analyzer import (
    AllocationOptimizer,
    DrawdownAnalyzer,
    ParameterOptimizer,
    PerformanceAnalyzer,
    RebalanceOptimizer,
    RegimeAnalyzer,
    StrategyContributionAnalyzer,
)
from .parameter_optimizer import (
    AdvancedParameterOptimizer,
    TimeSeriesCV,
    SharpeEvaluator,
    ParameterGridConfig,
    OptimizationResult,
    CVFold,
    create_default_param_grid,
    optimize_with_defaults,
)
from .portfolio_specific_params import (
    PortfolioSpecificParams,
    AdaptiveParameterManager,
    DefaultThresholdCalculator,
    ThresholdCalculatorProtocol,
    create_adaptive_parameter_manager,
)
from .benchmark_fetcher import (
    BenchmarkFetcher,
    BenchmarkFetcherError,
)
from .report_generator import (
    PortfolioMetrics,
    ComparisonResult,
    ReportGenerator,
)
# Static charts (requires matplotlib)
try:
    from .static_charts import StaticChartGenerator
    HAS_STATIC_CHARTS = True
except ImportError:
    StaticChartGenerator = None
    HAS_STATIC_CHARTS = False
# Interactive charts (requires plotly)
try:
    from .chart_generator import ChartGenerator, ChartGeneratorError
    HAS_PLOTLY_CHARTS = True
except ImportError:
    ChartGenerator = None
    ChartGeneratorError = None
    HAS_PLOTLY_CHARTS = False
# Note: dynamic_threshold.py と dynamic_thresholds.py は非推奨・削除済み
# 代わりに src.meta.dynamic_params.DynamicParamsManager を使用してください
# 互換性のため一部クラスを再エクスポート
from src.meta.dynamic_params import (
    DynamicParamsManager,
    ThresholdResult,
    RebalanceThresholdResult,
    SmoothingAlphaResult,
    PositionLimitResult,
    calculate_rebalance_threshold,
    calculate_smoothing_alpha,
    calculate_position_limit,
)

__all__ = [
    "PerformanceAnalyzer",
    "StrategyContributionAnalyzer",
    "DrawdownAnalyzer",
    "RegimeAnalyzer",
    "ParameterOptimizer",
    "AllocationOptimizer",
    "RebalanceOptimizer",
    # Advanced optimization
    "AdvancedParameterOptimizer",
    "TimeSeriesCV",
    "SharpeEvaluator",
    "ParameterGridConfig",
    "OptimizationResult",
    "CVFold",
    "create_default_param_grid",
    "optimize_with_defaults",
    # Portfolio specific params
    "PortfolioSpecificParams",
    "AdaptiveParameterManager",
    "DefaultThresholdCalculator",
    "ThresholdCalculatorProtocol",
    "create_adaptive_parameter_manager",
    # Dynamic params (from src.meta.dynamic_params - 互換性のため再エクスポート)
    "DynamicParamsManager",
    "ThresholdResult",
    "RebalanceThresholdResult",
    "SmoothingAlphaResult",
    "PositionLimitResult",
    "calculate_rebalance_threshold",
    "calculate_smoothing_alpha",
    "calculate_position_limit",
    # Benchmark fetcher
    "BenchmarkFetcher",
    "BenchmarkFetcherError",
    # Report generator
    "PortfolioMetrics",
    "ComparisonResult",
    "ReportGenerator",
    # Static charts
    "StaticChartGenerator",
    # Interactive charts (Plotly)
    "ChartGenerator",
    "ChartGeneratorError",
]
