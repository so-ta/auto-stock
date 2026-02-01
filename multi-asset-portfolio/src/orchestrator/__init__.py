"""
Orchestrator modules for the Multi-Asset Portfolio System.

Architecture: Single Responsibility Principle
=============================================

Each module has ONE clear responsibility:

1. **pipeline.py** - Orchestration Layer (COORDINATION ONLY)
   - Coordinates execution flow between modules
   - NO business logic - delegates to specialized modules

2. **data_preparation.py** - Data Layer
   - Fetch market data
   - Quality checks and asset exclusion
   - Delegates to: src/data/

3. **signal_generation.py** - Signal Layer
   - Generate trading signals
   - Strategy evaluation
   - Delegates to: src/signals/

4. **risk_allocation.py** - Allocation Layer
   - Risk estimation
   - Asset weight allocation (HRP, RP, etc.)
   - Delegates to: src/allocation/

5. **weight_calculation.py** - Weight Layer
   - Strategy gate checks
   - Weight calculation and validation
   - Delegates to: src/strategy/

6. **anomaly_detector.py** - Monitoring Layer
   - Anomaly detection in data/outputs
   - Alert generation

7. **fallback.py** - Resilience Layer
   - Fallback mode management
   - Recovery handling

8. **cmd016_integrator.py** - Feature Integration Layer
   - CMD016 advanced feature integration

See docs/ORCHESTRATOR_RESPONSIBILITIES.md for detailed documentation.
"""

from src.config.settings import FallbackMode  # SSOT: FallbackMode is defined in settings
from src.orchestrator.anomaly_detector import AnomalyDetector, AnomalyType
from src.orchestrator.data_preparation import DataPreparation, DataPreparationResult
from src.orchestrator.fallback import FallbackHandler
from src.orchestrator.pipeline import Pipeline, PipelineConfig, PipelineResult
from src.orchestrator.risk_allocation import (
    AllocationResult,
    AssetAllocator,
    RiskEstimationResult,
    RiskEstimator,
)
from src.orchestrator.signal_generation import (
    SignalGenerator,
    SignalGenerationResult,
    StrategyEvaluator,
)
from src.orchestrator.weight_calculation import (
    GateCheckResult,
    GateChecker,
    StrategyWeighter,
    WeightingResult,
)
from src.orchestrator.unified_executor import UnifiedExecutor

# CMD_016 integration (optional)
try:
    from src.orchestrator.cmd016_integrator import (
        CMD016Integrator,
        CMD016Config,
        IntegrationResult,
        create_integrator,
        get_available_features,
    )
    _CMD016_EXPORTS = [
        "CMD016Integrator",
        "CMD016Config",
        "IntegrationResult",
        "create_integrator",
        "get_available_features",
    ]
except ImportError:
    _CMD016_EXPORTS = []

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    # Unified Executor
    "UnifiedExecutor",
    # Data Preparation (QA-003-P1)
    "DataPreparation",
    "DataPreparationResult",
    # Signal Generation (QA-003-P1)
    "SignalGenerator",
    "SignalGenerationResult",
    "StrategyEvaluator",
    # Weight Calculation (QA-003-P2)
    "GateChecker",
    "GateCheckResult",
    "StrategyWeighter",
    "WeightingResult",
    # Risk Allocation (QA-003-P2)
    "RiskEstimator",
    "RiskEstimationResult",
    "AssetAllocator",
    "AllocationResult",
    # Anomaly & Fallback
    "AnomalyDetector",
    "AnomalyType",
    "FallbackHandler",
    "FallbackMode",
] + _CMD016_EXPORTS
