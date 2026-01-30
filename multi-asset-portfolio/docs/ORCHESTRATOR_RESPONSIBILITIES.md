# Orchestrator Module Responsibilities

> **Version**: 1.0.0
> **Last Updated**: 2026-01-29
> **Task**: task_032_12 (cmd_032)

## Overview

The `src/orchestrator/` module implements the main execution pipeline for the Multi-Asset Portfolio System. This document defines the **single responsibility** for each sub-module and establishes clear boundaries to prevent responsibility overlap.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      pipeline.py                                 │
│                   (Orchestration Only)                           │
│  - Coordinates execution flow                                    │
│  - No business logic                                            │
│  - Delegates to specialized modules                             │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ data_         │    │ signal_       │    │ risk_         │
│ preparation   │    │ generation    │    │ allocation    │
│               │    │               │    │               │
│ - Fetch data  │    │ - Compute     │    │ - Risk        │
│ - Quality     │    │   signals     │    │   estimation  │
│   check       │    │ - Strategy    │    │ - Asset       │
│ - Cutoff      │    │   evaluation  │    │   allocation  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────┐
                    │ weight_       │
                    │ calculation   │
                    │               │
                    │ - Gate check  │
                    │ - Strategy    │
                    │   weighting   │
                    └───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ anomaly_      │    │ fallback      │    │ cmd016_       │
│ detector      │    │               │    │ integrator    │
│               │    │ - Fallback    │    │               │
│ - Anomaly     │    │   modes       │    │ - CMD016      │
│   detection   │    │ - Recovery    │    │   features    │
└───────────────┘    └───────────────┘    └───────────────┘
```

## Module Responsibilities

### 1. `pipeline.py` - Orchestration Layer

**Single Responsibility**: Coordinate the execution flow between modules.

**DOES**:
- Initialize and configure sub-modules
- Call modules in the correct sequence
- Pass data between modules
- Handle top-level errors
- Log pipeline progress
- Manage pipeline state

**DOES NOT**:
- Implement business logic
- Transform data (delegates to modules)
- Make allocation decisions
- Compute signals
- Check data quality

**Key Classes**:
- `PipelineOrchestrator`: Main orchestrator class
- `PipelineConfig`: Configuration for the pipeline
- `PipelineResult`: Result of pipeline execution

---

### 2. `data_preparation.py` - Data Layer

**Single Responsibility**: Fetch and prepare data for downstream processing.

**DOES**:
- Fetch market data from adapters
- Apply date cutoff (prevent lookahead bias)
- Run quality checks
- Mark/exclude NG assets
- Handle data caching

**DOES NOT**:
- Compute signals
- Allocate weights
- Evaluate strategies

**Key Classes**:
- `DataPreparation`: Main data preparation class
- `DataPreparationResult`: Result with raw data and quality reports

**Delegates To**:
- `src/data/` for data fetching
- `src/data/quality/` for quality checks

---

### 3. `signal_generation.py` - Signal Layer

**Single Responsibility**: Generate trading signals and evaluate strategies.

**DOES**:
- Generate signals using signal modules
- Optimize signal parameters on training data
- Evaluate signal performance on test data
- Track signal quality metrics

**DOES NOT**:
- Fetch data
- Allocate final portfolio weights
- Handle fallback logic

**Key Classes**:
- `SignalGenerator`: Generates signals with parameter optimization
- `StrategyEvaluator`: Evaluates strategy performance
- `SignalGenerationResult`: Result with generated signals

**Delegates To**:
- `src/signals/` for signal computation

---

### 4. `risk_allocation.py` - Allocation Layer

**Single Responsibility**: Estimate risk and allocate weights across assets.

**DOES**:
- Estimate expected returns
- Estimate covariance/risk
- Optimize portfolio weights (HRP, Risk Parity, etc.)
- Apply allocation constraints

**DOES NOT**:
- Generate signals
- Check strategy gates
- Handle fallback

**Key Classes**:
- `RiskEstimator`: Estimates risk metrics
- `AssetAllocator`: Allocates weights across assets

**Delegates To**:
- `src/allocation/` for allocation algorithms

---

### 5. `weight_calculation.py` - Weight Layer

**Single Responsibility**: Calculate and validate strategy weights.

**DOES**:
- Check strategy gates (Hard Gates)
- Calculate strategy weights (softmax, cap, diversity)
- Apply weight constraints
- Validate final weights

**DOES NOT**:
- Generate signals
- Estimate risk
- Handle anomalies

**Key Classes**:
- `GateChecker`: Checks strategy against gates
- `StrategyWeighter`: Calculates strategy weights

**Delegates To**:
- `src/strategy/gate_checker.py` for gate logic

---

### 6. `anomaly_detector.py` - Monitoring Layer

**Single Responsibility**: Detect anomalies in data and outputs.

**DOES**:
- Detect data anomalies (stale data, price spikes)
- Detect portfolio anomalies (concentration, extreme weights)
- Generate anomaly reports
- Trigger alerts

**DOES NOT**:
- Handle fallback (delegates to fallback.py)
- Modify weights
- Fetch data

**Key Classes**:
- `AnomalyDetector`: Main anomaly detection class
- `AnomalyDetectionResult`: Result with detected anomalies

---

### 7. `fallback.py` - Resilience Layer

**Single Responsibility**: Handle system degradation and fallback modes.

**DOES**:
- Manage fallback state
- Apply fallback weights (hold, equal, cash)
- Handle recovery from fallback
- Log fallback events

**DOES NOT**:
- Detect anomalies (receives from anomaly_detector)
- Generate signals
- Fetch data

**Key Classes**:
- `FallbackHandler`: Manages fallback logic
- `FallbackMode`: Enum of fallback modes
- `FallbackState`: Current fallback state

---

### 8. `cmd016_integrator.py` - Feature Integration Layer

**Single Responsibility**: Integrate CMD016 advanced features.

**DOES**:
- Integrate pairs trading
- Integrate cross-asset momentum
- Integrate sector rotation
- Integrate drawdown protection
- Integrate dynamic thresholds

**DOES NOT**:
- Core pipeline logic
- Base signal generation
- Base allocation

**Key Classes**:
- `CMD016Integrator`: Feature integrator
- `CMD016Config`: Feature configuration

---

## Dependency Direction

```
Lower Level (no orchestrator dependencies)
├── src/data/          → Data fetching, quality
├── src/signals/       → Signal computation
├── src/allocation/    → Allocation algorithms
├── src/strategy/      → Strategy evaluation
│
Middle Level (orchestrator modules)
├── data_preparation   → Uses src/data/
├── signal_generation  → Uses src/signals/
├── risk_allocation    → Uses src/allocation/
├── weight_calculation → Uses src/strategy/
├── anomaly_detector   → Uses src/data/quality/
├── fallback           → Standalone
├── cmd016_integrator  → Uses src/signals/, src/allocation/
│
Top Level (orchestration only)
└── pipeline.py        → Uses all middle level modules
```

## Anti-Patterns to Avoid

### 1. ❌ Business Logic in pipeline.py

```python
# BAD: Business logic in pipeline
class PipelineOrchestrator:
    def run(self):
        for asset in assets:
            if returns.mean() > 0.01:  # Business logic!
                weights[asset] = 0.1
```

```python
# GOOD: Delegate to specialized module
class PipelineOrchestrator:
    def run(self):
        weights = self.allocator.allocate(returns)
```

### 2. ❌ Cross-Module Data Fetching

```python
# BAD: signal_generation fetching data
class SignalGenerator:
    def generate(self):
        data = self.fetch_from_yahoo()  # Wrong responsibility!
```

```python
# GOOD: Receive data from pipeline
class SignalGenerator:
    def generate(self, data: dict[str, pl.DataFrame]):
        # Work with provided data
```

### 3. ❌ Circular Dependencies

```python
# BAD: Circular import
# data_preparation.py
from src.orchestrator.signal_generation import SignalGenerator  # Circular!
```

```python
# GOOD: Only import from lower levels
# data_preparation.py
from src.data.fetcher import DataFetcher  # OK
```

## Testing Guidelines

Each module should have independent unit tests:

```
tests/unit/orchestrator/
├── test_data_preparation.py
├── test_signal_generation.py
├── test_risk_allocation.py
├── test_weight_calculation.py
├── test_anomaly_detector.py
├── test_fallback.py
└── test_pipeline.py  # Integration tests
```

## Migration Notes

If moving logic between modules:

1. Identify the logic's primary responsibility
2. Move to the appropriate module
3. Update imports in pipeline.py
4. Add/update unit tests
5. Run integration tests

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-29 | task_032_12 | Initial documentation |
