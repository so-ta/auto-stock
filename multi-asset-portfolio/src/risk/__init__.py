"""
Risk Management Module - リスク管理

VIXベースのキャッシュ配分など、リスク管理機能を提供する。
"""

from .vix_cash_allocation import (
    VIXCashAllocator,
    VIXCashConfig,
    VIXCashResult,
    compute_vix_cash_ratio,
    fetch_vix_data,
)
from .correlation_break import (
    CorrelationBreakDetector,
    CorrelationBreakConfig,
    CorrelationBreakResult,
    CorrelationChangeResult,
    AdjustmentResult,
    WarningLevel,
    create_correlation_break_detector,
    quick_detect_correlation_break,
)
from .drawdown_protection import (
    DrawdownProtector,
    DrawdownProtectorConfig,
    DrawdownState,
    ProtectionLevel,
    ProtectionResult,
    ProtectionStatus,
    RecoveryMode,
    create_drawdown_protector,
    quick_adjust_weights,
    calculate_protection_multiplier,
)
from .risk_budgeting import (
    RiskBudgetingAllocator,
    RiskBudgetingConfig,
    RiskBudgetingResult,
    RiskContribution,
    create_risk_budgeting_allocator,
    quick_risk_budgeting,
    compute_risk_contribution,
)
from .stress_testing import (
    StressTester,
    StressScenario,
    StressTestResult,
    HedgeRecommendation,
    HISTORICAL_SCENARIOS,
    HEDGE_ASSETS,
    run_stress_test,
    get_worst_case_loss,
    list_scenarios,
    create_stress_tester,
)
from .drawdown_controller import (
    DrawdownController,
    DrawdownControllerConfig,
    DrawdownState as ControllerDrawdownState,
    ControllerResult,
    create_drawdown_controller,
    quick_position_multiplier,
    adjust_weights_for_drawdown,
)

__all__ = [
    # VIX Cash Allocation
    "VIXCashAllocator",
    "VIXCashConfig",
    "VIXCashResult",
    "compute_vix_cash_ratio",
    "fetch_vix_data",
    # Correlation Break Detection
    "CorrelationBreakDetector",
    "CorrelationBreakConfig",
    "CorrelationBreakResult",
    "CorrelationChangeResult",
    "AdjustmentResult",
    "WarningLevel",
    "create_correlation_break_detector",
    "quick_detect_correlation_break",
    # Drawdown Protection
    "DrawdownProtector",
    "DrawdownProtectorConfig",
    "DrawdownState",
    "ProtectionLevel",
    "ProtectionResult",
    "ProtectionStatus",
    "RecoveryMode",
    "create_drawdown_protector",
    "quick_adjust_weights",
    "calculate_protection_multiplier",
    # Risk Budgeting
    "RiskBudgetingAllocator",
    "RiskBudgetingConfig",
    "RiskBudgetingResult",
    "RiskContribution",
    "create_risk_budgeting_allocator",
    "quick_risk_budgeting",
    "compute_risk_contribution",
    # Stress Testing
    "StressTester",
    "StressScenario",
    "StressTestResult",
    "HedgeRecommendation",
    "HISTORICAL_SCENARIOS",
    "HEDGE_ASSETS",
    "run_stress_test",
    "get_worst_case_loss",
    "list_scenarios",
    "create_stress_tester",
    # Drawdown Controller (IMP-004)
    "DrawdownController",
    "DrawdownControllerConfig",
    "ControllerDrawdownState",
    "ControllerResult",
    "create_drawdown_controller",
    "quick_position_multiplier",
    "adjust_weights_for_drawdown",
]
