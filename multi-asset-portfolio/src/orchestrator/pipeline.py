"""
Pipeline Orchestrator - Main Execution Pipeline

This module implements the complete execution pipeline from §12:
1. Data fetch → Quality check → Mark NG assets
2. Generate strategies for each asset (parameter search on train only)
3. Calculate evaluation metrics on test (cost deduction, turnover measurement)
4. Reject strategies via gates
5. Calculate strategy weights within asset (softmax + cap + diversity)
6. Estimate expected value and risk per asset (including Σ)
7. Optimize all asset weights (HRP/RP recommended) + apply constraints
8. Smoothing / change limit
9. Anomaly detection → Apply fallback mode (if needed)
10. Generate output (weights + diagnostics) & save logs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog

if TYPE_CHECKING:
    import polars as pl

    from src.config.schemas import PortfolioWeights, ValidationReport
    from src.config.settings import Settings

from src.orchestrator.anomaly_detector import AnomalyDetectionResult, AnomalyDetector
from src.orchestrator.data_preparation import DataPreparation, DataPreparationResult
from src.orchestrator.fallback import FallbackHandler, FallbackMode, FallbackState
from src.orchestrator.risk_allocation import AssetAllocator, RiskEstimator
from src.orchestrator.signal_generation import SignalGenerator, StrategyEvaluator
from src.orchestrator.weight_calculation import GateChecker, StrategyWeighter

# Universe expansion support (optional - imported on demand)
try:
    from src.data.universe_loader import UniverseLoader
    from src.data.adapters.multi_source_adapter import MultiSourceAdapter
    from src.data.currency_converter import CurrencyConverter
    from src.data.calendar_manager import CalendarManager
    UNIVERSE_EXPANSION_AVAILABLE = True
except ImportError:
    UNIVERSE_EXPANSION_AVAILABLE = False

# Return maximization support (optional - Phase 4 integration)
try:
    from src.allocation.return_estimator import DynamicReturnEstimator
    from src.allocation.kelly_allocator import KellyAllocator
    from src.signals.regime_adaptive_params import RegimeAdaptiveParams, adjust_signal_params
    from src.strategy.entry_exit_optimizer import HysteresisFilter, GradualEntryExit
    from src.signals.macro_timing import EconomicCycleAllocator
    RETURN_MAXIMIZATION_AVAILABLE = True
except ImportError:
    RETURN_MAXIMIZATION_AVAILABLE = False

# CMD_016 features integration (Phase 5)
try:
    from src.orchestrator.cmd016_integrator import (
        CMD016Integrator,
        CMD016Config,
        IntegrationResult,
        create_integrator,
        get_available_features,
    )
    CMD016_AVAILABLE = True
except ImportError:
    CMD016_AVAILABLE = False

# CMD_017 features integration (Phase 6)
try:
    from src.allocation.nco import NestedClusteredOptimization, NCOConfig
    from src.allocation.black_litterman import BlackLittermanModel, BlackLittermanConfig
    from src.allocation.cvar_optimizer import CVaROptimizer
    from src.allocation.transaction_cost_optimizer import TransactionCostOptimizer
    from src.backtest.adaptive_window import AdaptiveWindowSelector, VolatilityRegime
    from src.backtest.synthetic_data import SyntheticDataGenerator, StatisticalSignificanceTester
    from src.backtest.purged_kfold import PurgedKFold, CombinatorialPurgedCV
    CMD017_AVAILABLE = True
except ImportError:
    CMD017_AVAILABLE = False

from src.utils.logger import (
    AuditLogger,
    TimedOperation,
    generate_run_id,
    get_audit_logger,
    get_logger,
    set_run_id,
)
from src.utils.reproducibility import SeedManager, get_run_info, save_run_info

logger = structlog.get_logger(__name__)


class PipelineStep(str, Enum):
    """Pipeline execution steps."""

    INITIALIZE = "initialize"
    DATA_FETCH = "data_fetch"
    QUALITY_CHECK = "quality_check"
    REGIME_DETECTION = "regime_detection"  # 新規: レジーム検出
    SIGNAL_GENERATION = "signal_generation"
    STRATEGY_EVALUATION = "strategy_evaluation"
    GATE_CHECK = "gate_check"
    ENSEMBLE_COMBINE = "ensemble_combine"  # 新規: アンサンブル統合
    STRATEGY_WEIGHTING = "strategy_weighting"
    RISK_ESTIMATION = "risk_estimation"
    ASSET_ALLOCATION = "asset_allocation"
    DYNAMIC_WEIGHTING = "dynamic_weighting"  # 新規: 動的重み調整
    CMD016_INTEGRATION = "cmd016_integration"  # 新規: CMD_016機能統合
    CMD017_INTEGRATION = "cmd017_integration"  # 新規: CMD_017機能統合（配分最適化・ML・検証）
    SMOOTHING = "smoothing"
    ANOMALY_DETECTION = "anomaly_detection"
    FALLBACK_CHECK = "fallback_check"
    OUTPUT_GENERATION = "output_generation"
    LOGGING = "logging"
    FINALIZE = "finalize"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    run_id: str | None = None
    seed: int = 42
    dry_run: bool = False
    skip_data_fetch: bool = False
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Lightweight mode for backtest (task_036_4)
    # Reduces overhead by 20-30% by skipping non-essential operations
    lightweight_mode: bool = False
    skip_diagnostics: bool = False  # Skip detailed diagnostics generation
    skip_audit_log: bool = False  # Skip audit log writing

    # Incremental covariance estimation (task_040_3)
    # Enables exponential-weighted incremental update for 10-20% speedup
    use_incremental_covariance: bool = False
    covariance_halflife: int = 60  # Halflife for exponential weighting


@dataclass
class StepResult:
    """Result of a pipeline step."""

    step: PipelineStep
    status: str  # success, error, skipped
    duration_ms: float
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    run_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    weights: dict[str, float]
    diagnostics: dict[str, Any]
    fallback_state: FallbackState | None
    step_results: list[StepResult]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "weights": self.weights,
            "diagnostics": self.diagnostics,
            "fallback_state": self.fallback_state.to_dict() if self.fallback_state else None,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class Pipeline:
    """
    Main pipeline orchestrator.

    Executes the complete portfolio construction pipeline from
    data fetching to final weight output.

    Example:
        >>> pipeline = Pipeline(settings)
        >>> result = pipeline.run(
        ...     universe=["BTCUSD", "ETHUSD", "AAPL"],
        ...     previous_weights={"BTCUSD": 0.3, "ETHUSD": 0.3, "AAPL": 0.4},
        ... )
        >>> print(result.weights)
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            settings: Application settings
            config: Pipeline-specific configuration
        """
        self._settings = settings
        self._config = config or PipelineConfig()
        self._logger = logger.bind(component="pipeline")
        self._audit_logger: AuditLogger | None = None

        # Components
        self._anomaly_detector: AnomalyDetector | None = None
        self._fallback_handler: FallbackHandler | None = None
        self._cmd016_integrator: "CMD016Integrator | None" = None
        self._data_preparation: DataPreparation | None = None
        self._signal_generator: SignalGenerator | None = None
        self._strategy_evaluator: StrategyEvaluator | None = None
        self._gate_checker: GateChecker | None = None
        self._strategy_weighter: StrategyWeighter | None = None
        self._risk_estimator: RiskEstimator | None = None
        self._asset_allocator: AssetAllocator | None = None

        # Runtime state
        self._step_results: list[StepResult] = []
        self._errors: list[str] = []
        self._warnings: list[str] = []

        # Data stores (populated during execution)
        self._raw_data: dict[str, "pl.DataFrame"] = {}
        self._quality_reports: dict[str, Any] = {}
        self._excluded_assets: list[str] = []
        self._signals: dict[str, dict[str, Any]] = {}
        self._evaluations: list[Any] = []
        self._strategy_weights: dict[str, dict[str, float]] = {}
        self._risk_metrics: dict[str, Any] = {}
        self._correlation_matrix: Any = None
        self._covariance: Any = None  # pd.DataFrame - 共分散行列
        self._expected_returns: dict[str, float] = {}  # アセット別期待リターン
        self._raw_weights: dict[str, float] = {}
        self._smoothed_weights: dict[str, float] = {}
        self._final_weights: dict[str, float] = {}
        self._prev_weights: dict[str, float] = {}  # 前期の重み
        self._cash_weight: float = 0.0  # TopNSelector使用時のCASH重み

        # 新機能用データストア
        self._regime_info: dict[str, Any] = {}  # レジーム検出結果
        self._ensemble_scores: dict[str, Any] = {}  # アンサンブル統合結果
        self._dynamic_weighting_result: Any = None  # 動的重み調整結果
        self._cmd016_result: Any = None  # CMD_016統合結果
        self._cmd017_result: Any = None  # CMD_017統合結果
        self._portfolio_value: float = 100000.0  # ポートフォリオ価値（初期値）
        self._vix_value: float | None = None  # VIX値

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Get anomaly detector."""
        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetector(self._settings)
        return self._anomaly_detector

    @property
    def fallback_handler(self) -> FallbackHandler:
        """Get fallback handler."""
        if self._fallback_handler is None:
            self._fallback_handler = FallbackHandler(self._settings)
        return self._fallback_handler

    @property
    def cmd016_integrator(self) -> "CMD016Integrator | None":
        """Get CMD_016 feature integrator."""
        if not CMD016_AVAILABLE:
            return None
        if self._cmd016_integrator is None:
            self._cmd016_integrator = create_integrator(self._settings)
        return self._cmd016_integrator

    @property
    def data_preparation(self) -> DataPreparation:
        """Get data preparation component."""
        if self._data_preparation is None:
            self._data_preparation = DataPreparation(
                settings=self.settings,
                output_dir=self._config.output_dir,
                audit_logger=self._audit_logger,
            )
        return self._data_preparation

    @property
    def signal_generator(self) -> SignalGenerator:
        """Get signal generator component."""
        if self._signal_generator is None:
            self._signal_generator = SignalGenerator(
                settings=self.settings,
                audit_logger=self._audit_logger,
            )
        return self._signal_generator

    @property
    def strategy_evaluator(self) -> StrategyEvaluator:
        """Get strategy evaluator component."""
        if self._strategy_evaluator is None:
            self._strategy_evaluator = StrategyEvaluator(
                settings=self.settings,
                audit_logger=self._audit_logger,
            )
        return self._strategy_evaluator

    @property
    def gate_checker(self) -> GateChecker:
        """Get gate checker component."""
        if self._gate_checker is None:
            self._gate_checker = GateChecker(
                settings=self.settings,
                audit_logger=self._audit_logger,
            )
        return self._gate_checker

    @property
    def strategy_weighter(self) -> StrategyWeighter:
        """Get strategy weighter component."""
        if self._strategy_weighter is None:
            self._strategy_weighter = StrategyWeighter(
                settings=self.settings,
                audit_logger=self._audit_logger,
            )
        return self._strategy_weighter

    @property
    def risk_estimator(self) -> RiskEstimator:
        """Get risk estimator component."""
        if self._risk_estimator is None:
            self._risk_estimator = RiskEstimator(
                settings=self.settings,
                audit_logger=self._audit_logger,
                use_incremental=self.config.use_incremental_covariance,
                halflife=self.config.covariance_halflife,
            )
        return self._risk_estimator

    @property
    def asset_allocator(self) -> AssetAllocator:
        """Get asset allocator component."""
        if self._asset_allocator is None:
            self._asset_allocator = AssetAllocator(
                settings=self.settings,
                audit_logger=self._audit_logger,
            )
        return self._asset_allocator

    def run(
        self,
        universe: list[str] | None = None,
        previous_weights: dict[str, float] | None = None,
        as_of_date: datetime | None = None,
        data_cutoff_date: datetime | None = None,
    ) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            universe: List of asset symbols to process
            previous_weights: Weights from previous period
            as_of_date: Date for evaluation (default: now)
            data_cutoff_date: If specified, filter all data to this date.
                              Used for backtesting to prevent future data leakage.
                              Data after this date is NEVER used.

        Returns:
            PipelineResult with weights and diagnostics
        """
        # Initialize run
        run_id = self._config.run_id or generate_run_id()
        set_run_id(run_id)
        start_time = datetime.now(timezone.utc)

        self._logger.info("Pipeline started", run_id=run_id)

        # Lightweight mode: skip audit logging for better performance (task_036_4)
        if self._config.lightweight_mode or self._config.skip_audit_log:
            self._audit_logger = None
        else:
            self._audit_logger = get_audit_logger("pipeline")

        # Initialize expanded mode attributes
        self._fetch_summary: dict[str, list] = {}
        self._quality_summary: dict[str, list] = {}

        # Use settings universe if not provided
        if universe is None:
            # Try UniverseLoader first if available
            universe = self._load_universe_from_config()
            if not universe:
                universe = self.settings.universe

        if not universe:
            self._errors.append("No assets in universe")
            return self._create_error_result(run_id, start_time, "Empty universe")

        previous_weights = previous_weights or {}
        as_of_date = as_of_date or datetime.now(timezone.utc)

        # For backtesting: use cutoff_date if specified, otherwise use as_of_date
        effective_cutoff = data_cutoff_date or as_of_date

        # Execute with seed management
        try:
            with SeedManager(seed=self._config.seed):
                result = self._execute_pipeline(
                    run_id=run_id,
                    universe=universe,
                    previous_weights=previous_weights,
                    as_of_date=as_of_date,
                    data_cutoff_date=effective_cutoff,
                    start_time=start_time,
                )
        except Exception as e:
            self._logger.exception("Pipeline failed", error=str(e))
            return self._create_error_result(run_id, start_time, str(e))

        return result

    def _load_universe_from_config(self) -> list[str] | None:
        """
        Load universe from config file using UniverseLoader.

        Returns:
            List of ticker symbols or None if not available
        """
        if not UNIVERSE_EXPANSION_AVAILABLE:
            return None

        # Check if universe config is specified
        universe_config = getattr(self.settings, 'universe', None)
        if isinstance(universe_config, dict) and 'config_file' in universe_config:
            config_file = universe_config['config_file']
        else:
            # Try default location
            config_file = "config/universe.yaml"

        try:
            from pathlib import Path
            config_path = Path(config_file)
            if not config_path.exists():
                return None

            loader = UniverseLoader(config_path)
            tickers = loader.get_all_tickers()

            if tickers:
                self._logger.info(
                    "Loaded universe from config",
                    config_file=str(config_path),
                    ticker_count=len(tickers),
                )
                return tickers

        except Exception as e:
            self._logger.warning(f"Failed to load universe from config: {e}")

        return None

    def _execute_pipeline(
        self,
        run_id: str,
        universe: list[str],
        previous_weights: dict[str, float],
        as_of_date: datetime,
        data_cutoff_date: datetime,
        start_time: datetime,
    ) -> PipelineResult:
        """Execute all pipeline steps."""
        status = PipelineStatus.RUNNING
        fallback_state: FallbackState | None = None

        # Store cutoff date for use in steps
        self._data_cutoff_date = data_cutoff_date

        # Step 1: Data Fetch
        self._run_step(
            PipelineStep.DATA_FETCH,
            lambda: self._step_data_fetch(universe, as_of_date),
        )

        # Step 1.5: Apply data cutoff filter (for backtesting)
        # This MUST happen immediately after data fetch to prevent any future data leakage
        if data_cutoff_date:
            self._apply_data_cutoff(data_cutoff_date)

        # Step 2: Quality Check
        self._run_step(
            PipelineStep.QUALITY_CHECK,
            lambda: self._step_quality_check(),
        )

        # Step 3: Regime Detection（新規）
        self._run_step(
            PipelineStep.REGIME_DETECTION,
            lambda: self._step_regime_detection(),
        )

        # Step 4: Signal Generation（基本 + アンサンブル + ファクター）
        self._run_step(
            PipelineStep.SIGNAL_GENERATION,
            lambda: self._step_signal_generation(),
        )

        # Step 5: Strategy Evaluation
        self._run_step(
            PipelineStep.STRATEGY_EVALUATION,
            lambda: self._step_strategy_evaluation(),
        )

        # Step 6: Gate Check
        self._run_step(
            PipelineStep.GATE_CHECK,
            lambda: self._step_gate_check(),
        )

        # Step 7: Ensemble Combine（新規）
        self._run_step(
            PipelineStep.ENSEMBLE_COMBINE,
            lambda: self._step_ensemble_combine(),
        )

        # Step 8: Strategy Weighting
        adopted_count = self._run_step(
            PipelineStep.STRATEGY_WEIGHTING,
            lambda: self._step_strategy_weighting(),
        )

        # Step 9: Risk Estimation
        self._run_step(
            PipelineStep.RISK_ESTIMATION,
            lambda: self._step_risk_estimation(),
        )

        # Step 10: Asset Allocation
        self._run_step(
            PipelineStep.ASSET_ALLOCATION,
            lambda: self._step_asset_allocation(),
        )

        # Step 11: Dynamic Weighting（新規）
        self._run_step(
            PipelineStep.DYNAMIC_WEIGHTING,
            lambda: self._step_dynamic_weighting(previous_weights),
        )

        # Step 12: CMD_016 Integration（Phase 5 全機能統合）
        self._run_step(
            PipelineStep.CMD016_INTEGRATION,
            lambda: self._step_cmd016_integration(),
        )

        # Step 13: CMD_017 Integration（Phase 6 配分最適化・ML・検証）
        self._run_step(
            PipelineStep.CMD017_INTEGRATION,
            lambda: self._step_cmd017_integration(),
        )

        # Step 14: Smoothing
        self._run_step(
            PipelineStep.SMOOTHING,
            lambda: self._step_smoothing(previous_weights),
        )

        # Step 10: Anomaly Detection
        anomaly_result = self._run_step(
            PipelineStep.ANOMALY_DETECTION,
            lambda: self._step_anomaly_detection(),
        )

        # Step 11: Fallback Check
        if anomaly_result and anomaly_result.should_trigger_fallback:
            fallback_state = self._run_step(
                PipelineStep.FALLBACK_CHECK,
                lambda: self._step_fallback(
                    previous_weights=previous_weights,
                    reason=anomaly_result.fallback_reason or "Unknown",
                    adopted_strategies=adopted_count or 0,
                ),
            )
            status = PipelineStatus.FALLBACK
            self._final_weights = fallback_state.applied_weights if fallback_state else self._smoothed_weights
        else:
            self._final_weights = self._smoothed_weights
            status = PipelineStatus.COMPLETED

        # Step 12: Output Generation
        self._run_step(
            PipelineStep.OUTPUT_GENERATION,
            lambda: self._step_output_generation(run_id),
        )

        # Step 13: Logging
        self._run_step(
            PipelineStep.LOGGING,
            lambda: self._step_logging(run_id),
        )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Create result
        result = PipelineResult(
            run_id=run_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            weights=self._final_weights,
            diagnostics=self._create_diagnostics(fallback_state),
            fallback_state=fallback_state,
            step_results=self._step_results,
            errors=self._errors,
            warnings=self._warnings,
        )

        # Log summary
        if self._audit_logger:
            self._audit_logger.log_run_summary(
                run_id=run_id,
                status=status.value,
                duration_seconds=duration,
                assets_processed=len(self._raw_data),
                strategies_evaluated=len(self._evaluations),
                strategies_adopted=adopted_count or 0,
                fallback_mode=fallback_state.mode.value if fallback_state else None,
                output_weights=self._final_weights,
            )

        self._logger.info(
            "Pipeline completed",
            run_id=run_id,
            status=status.value,
            duration_seconds=round(duration, 2),
        )

        return result

    def _run_step(
        self,
        step: PipelineStep,
        func: Callable[[], Any],
    ) -> Any:
        """Run a pipeline step with timing and error handling.

        In lightweight mode (task_036_4), skips detailed step result recording
        and audit logging for improved backtest performance.
        """
        result = None
        error = None
        status = "success"

        # Lightweight mode: skip timing wrapper overhead
        if self._config.lightweight_mode:
            try:
                result = func()
            except Exception as e:
                error = str(e)
                status = "error"
                self._errors.append(f"{step.value}: {error}")
            return result

        # Full mode: detailed timing and logging
        start_time = time.time()

        try:
            with TimedOperation(step.value, self._logger):
                result = func()
        except Exception as e:
            error = str(e)
            status = "error"
            self._errors.append(f"{step.value}: {error}")
            self._logger.error(f"Step {step.value} failed", error=error)

        duration_ms = (time.time() - start_time) * 1000

        self._step_results.append(
            StepResult(
                step=step,
                status=status,
                duration_ms=duration_ms,
                data={"result_type": type(result).__name__ if result else None},
                error=error,
            )
        )

        if self._audit_logger:
            self._audit_logger.log_pipeline_step(
                step_name=step.value,
                step_number=len(self._step_results),
                status=status,
                duration_ms=round(duration_ms, 2),
            )

        return result

    # =========================================================================
    # Pipeline Steps Implementation
    # =========================================================================
    def _step_data_fetch(
        self,
        universe: list[str],
        as_of_date: datetime,
    ) -> dict[str, "pl.DataFrame"]:
        """
        Step 1: Fetch data for all assets.

        Delegates to DataPreparation module (QA-003-P1 refactoring).

        Args:
            universe: List of asset symbols to fetch
            as_of_date: Reference date for data fetching

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        # Sync any pre-injected data to DataPreparation (e.g., from BacktestEngine)
        # This ensures skip_data_fetch mode can use externally provided data
        if self._raw_data and self._config.skip_data_fetch:
            self.data_preparation._raw_data = self._raw_data

        # Delegate to DataPreparation module
        self._raw_data = self.data_preparation.fetch_data(
            universe=universe,
            as_of_date=as_of_date,
            skip_fetch=self._config.skip_data_fetch,
        )
        # Copy warnings from data preparation
        self._warnings.extend(self.data_preparation._warnings)
        # Store fetch summary
        self._fetch_summary = self.data_preparation._fetch_summary
        return self._raw_data

    # NOTE: _step_data_fetch_expanded moved to DataPreparation module (QA-003-P1)
    # See: src/orchestrator/data_preparation.py

    def _apply_data_cutoff(self, cutoff_date: datetime) -> None:
        """
        Apply data cutoff filter to all raw data.

        Delegates to DataPreparation module (QA-003-P1 refactoring).

        Args:
            cutoff_date: All data after this date will be removed
        """
        # Sync raw_data to data_preparation
        self.data_preparation._raw_data = self._raw_data
        # Apply cutoff via DataPreparation
        self.data_preparation.apply_data_cutoff(cutoff_date)
        # Sync back
        self._raw_data = self.data_preparation._raw_data

    def _step_quality_check(self) -> dict[str, Any]:
        """
        Step 2: Check data quality.

        Delegates to DataPreparation module (QA-003-P1 refactoring).
        """
        # Sync raw_data to data_preparation
        self.data_preparation._raw_data = self._raw_data
        # Run quality check via DataPreparation
        self._quality_reports = self.data_preparation.run_quality_check()
        # Sync results back
        self._excluded_assets = self.data_preparation._excluded_assets
        self._quality_summary = self.data_preparation._quality_summary
        return self._quality_reports

    def _step_signal_generation(self) -> dict[str, dict[str, Any]]:
        """
        Step 3: Generate signals for each asset.

        Delegates to SignalGenerator module (QA-003-P1 refactoring).
        """
        result = self.signal_generator.generate_signals(
            raw_data=self._raw_data,
            excluded_assets=self._excluded_assets,
        )
        self._signals = result.signals
        return self._signals

    def _step_strategy_evaluation(self) -> list[Any]:
        """
        Step 4: Evaluate strategies on test data.

        Delegates to StrategyEvaluator module (QA-003-P1 refactoring).
        """
        self._evaluations = self.strategy_evaluator.evaluate_strategies(
            raw_data=self._raw_data,
            signals=self._signals,
        )
        return self._evaluations

    # NOTE: Legacy methods below moved to respective modules (QA-003-P1):
    # - _step_data_fetch_legacy -> DataPreparation
    # - _is_crypto_symbol -> DataPreparation
    # - _normalize_crypto_symbol -> DataPreparation
    # - _evaluate_signal_quality -> SignalGenerator
    # - _compute_signal_returns -> StrategyEvaluator
    # - _get_cost_per_trade -> StrategyEvaluator
    # - _calculate_strategy_score -> StrategyEvaluator

    # =========================================================================
    # =========================================================================
    # Gate Check, Weighting, Risk, Allocation (delegated to modules - QA-003-P2)
    # =========================================================================

    def _step_gate_check(self) -> list[Any]:
        """
        Step 5: Apply gate check to filter strategies.

        Delegates to GateChecker module (QA-003-P2 refactoring).
        """
        self._evaluations, warnings = self.gate_checker.check(self._evaluations)
        self._warnings.extend(warnings)
        return self._evaluations

    def _step_strategy_weighting(self) -> int:
        """
        Step 6: Calculate strategy weights.

        Delegates to StrategyWeighter module (QA-003-P2 refactoring).
        """
        result = self.strategy_weighter.calculate_weights(self._evaluations)
        self._strategy_weights = result.strategy_weights
        self._cash_weight = result.cash_weight
        return result.adopted_count

    def _step_risk_estimation(self) -> dict[str, Any]:
        """
        Step 7: Estimate risk metrics.

        Delegates to RiskEstimator module (QA-003-P2 refactoring).
        """
        result = self.risk_estimator.estimate(
            raw_data=self._raw_data,
            excluded_assets=self._excluded_assets,
        )
        self._risk_metrics = result.risk_metrics
        self._covariance = result.covariance
        self._correlation_matrix = result.correlation
        self._expected_returns = result.expected_returns
        return self._risk_metrics

    def _step_asset_allocation(self) -> dict[str, float]:
        """
        Step 8: Optimize asset allocation.

        Delegates to AssetAllocator module (QA-003-P2 refactoring).
        """
        result = self.asset_allocator.allocate(
            raw_data=self._raw_data,
            excluded_assets=self._excluded_assets,
            covariance=self._covariance,
            expected_returns=self._expected_returns,
            risk_metrics=self._risk_metrics,
            cash_weight=self._cash_weight,
            prev_weights=self._prev_weights,
        )
        self._raw_weights = result.weights
        return self._raw_weights

    def _step_smoothing(
        self,
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """Step 9: Apply smoothing and change limits."""
        alpha = self.settings.asset_allocation.smooth_alpha
        delta_max = self.settings.asset_allocation.delta_max

        self._smoothed_weights = {}

        for asset, new_weight in self._raw_weights.items():
            prev_weight = previous_weights.get(asset, 0.0)

            # Apply smoothing: w_final = α * w_new + (1-α) * w_prev
            smoothed = alpha * new_weight + (1 - alpha) * prev_weight

            # Apply change limit
            change = smoothed - prev_weight
            if abs(change) > delta_max:
                smoothed = prev_weight + (delta_max if change > 0 else -delta_max)

            self._smoothed_weights[asset] = smoothed

        # Normalize to sum to 1
        total = sum(self._smoothed_weights.values())
        if total > 0:
            self._smoothed_weights = {
                k: v / total for k, v in self._smoothed_weights.items()
            }

        self._logger.info("Smoothing completed")
        return self._smoothed_weights

    # =========================================================================
    # 新規ステップ: Regime Detection, Ensemble Combine, Dynamic Weighting
    # =========================================================================
    def _step_regime_detection(self) -> dict[str, Any]:
        """
        Step 3: Detect market regime using RegimeDetector signal.

        RegimeDetectorシグナルを使用して市場のレジーム（ボラティリティ・トレンド）を検出。
        結果はDynamicWeighterで使用される。

        Returns:
            レジーム情報辞書（vol_regime, trend_regime等）
        """
        import pandas as pd

        from src.signals import SignalRegistry

        self._regime_info = {
            "current_vol_regime": "medium",
            "current_trend_regime": "range",
            "regime_scores": {},
        }

        # 動的重み調整が無効の場合はスキップ
        dynamic_weighting_config = getattr(self.settings, "dynamic_weighting", None)
        if dynamic_weighting_config is None or not getattr(dynamic_weighting_config, "enabled", False):
            self._logger.info("Regime detection skipped (dynamic_weighting disabled)")
            return self._regime_info

        # 最初の有効アセットでレジーム検出
        valid_symbols = [
            symbol for symbol in self._raw_data.keys()
            if symbol not in self._excluded_assets
        ]

        if not valid_symbols:
            self._logger.warning("No valid assets for regime detection")
            return self._regime_info

        # RegimeDetectorシグナルを取得
        try:
            regime_detector_cls = SignalRegistry.get("regime_detector")
        except KeyError:
            self._logger.warning("RegimeDetector signal not registered")
            return self._regime_info

        # 代表アセットでレジーム検出（最初の有効アセット）
        representative_symbol = valid_symbols[0]
        df = self._raw_data.get(representative_symbol)

        if df is None:
            return self._regime_info

        # Convert polars to pandas if needed
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        try:
            # レジーム検出
            lookback = getattr(dynamic_weighting_config, "regime_lookback_days", 60)
            regime_signal = regime_detector_cls(vol_period=20, trend_period=lookback)
            result = regime_signal.compute(df)

            # メタデータからレジーム情報を取得
            self._regime_info = {
                "current_vol_regime": result.metadata.get("current_vol_regime", "medium"),
                "current_trend_regime": result.metadata.get("current_trend_regime", "range"),
                "regime_scores": {
                    "vol_mean": result.metadata.get("rolling_vol_mean", 0.0),
                    "trend_mean": result.metadata.get("trend_signal_mean", 0.0),
                },
                "representative_symbol": representative_symbol,
            }

            self._logger.info(
                "Regime detection completed",
                vol_regime=self._regime_info["current_vol_regime"],
                trend_regime=self._regime_info["current_trend_regime"],
                representative=representative_symbol,
            )

        except Exception as e:
            self._logger.warning(f"Regime detection failed: {e}")

        return self._regime_info

    def _step_ensemble_combine(self) -> dict[str, Any]:
        """
        Step 7: Combine strategy scores using EnsembleCombiner.

        複数の戦略スコアをアンサンブル手法で統合する。
        設定でenabledがtrueの場合のみ実行。

        Returns:
            アンサンブル統合結果
        """
        import pandas as pd

        self._ensemble_scores = {}

        # アンサンブル統合が無効の場合はスキップ
        ensemble_config = getattr(self.settings, "ensemble_combiner", None)
        if ensemble_config is None or not getattr(ensemble_config, "enabled", False):
            self._logger.info("Ensemble combine skipped (ensemble_combiner disabled)")
            return self._ensemble_scores

        if not self._evaluations:
            self._logger.warning("No evaluations available for ensemble combine")
            return self._ensemble_scores

        try:
            from src.meta.ensemble_combiner import EnsembleCombiner, EnsembleCombinerConfig

            # 設定からCombinerを初期化
            combiner_config = EnsembleCombinerConfig(
                method=getattr(ensemble_config, "method", "weighted_avg"),
                beta=getattr(ensemble_config, "beta", 2.0),
            )
            combiner = EnsembleCombiner(combiner_config)

            # アセット別に戦略スコアを統合
            evaluations_by_asset: dict[str, list] = {}
            for evaluation in self._evaluations:
                asset_id = evaluation.asset_id
                if asset_id not in evaluations_by_asset:
                    evaluations_by_asset[asset_id] = []
                evaluations_by_asset[asset_id].append(evaluation)

            for asset_id, evaluations in evaluations_by_asset.items():
                # 戦略スコアを辞書形式で準備
                strategy_scores: dict[str, pd.Series] = {}
                past_performance: dict[str, float] = {}

                for evaluation in evaluations:
                    if evaluation.metrics is None:
                        continue

                    strategy_id = evaluation.strategy_id
                    # スコアを単一値からSeriesに変換（簡易実装）
                    score_value = evaluation.score if evaluation.score else 0.0
                    strategy_scores[strategy_id] = pd.Series([score_value], index=[asset_id])
                    past_performance[strategy_id] = evaluation.metrics.sharpe_ratio

                if not strategy_scores:
                    continue

                # アンサンブル統合
                combine_result = combiner.combine(strategy_scores, past_performance)

                self._ensemble_scores[asset_id] = {
                    "combined_scores": combine_result.combined_scores.to_dict() if combine_result.is_valid else {},
                    "strategy_weights": combine_result.strategy_weights,
                    "method_used": combine_result.method_used,
                }

            self._logger.info(
                "Ensemble combine completed",
                assets=len(self._ensemble_scores),
                method=combiner_config.method,
            )

        except ImportError as e:
            self._logger.warning(f"EnsembleCombiner not available: {e}")
        except Exception as e:
            self._logger.warning(f"Ensemble combine failed: {e}")

        return self._ensemble_scores

    def _step_dynamic_weighting(
        self,
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Step 11: Apply dynamic weight adjustments.

        DynamicWeighterを使用して、市場状況に応じた動的重み調整を適用。
        - ボラティリティスケーリング
        - ドローダウン保護
        - レジームベース重み付け

        Args:
            previous_weights: 前期の重み

        Returns:
            動的調整後の重み
        """
        import numpy as np
        import pandas as pd

        # 動的重み調整が無効の場合はスキップ
        dynamic_config = getattr(self.settings, "dynamic_weighting", None)
        if dynamic_config is None or not getattr(dynamic_config, "enabled", False):
            self._logger.info("Dynamic weighting skipped (disabled)")
            return self._raw_weights

        try:
            from src.meta.dynamic_weighter import DynamicWeighter, DynamicWeightingConfig

            # 設定からDynamicWeighterを初期化
            weighter_config = DynamicWeightingConfig(
                target_volatility=getattr(dynamic_config, "target_volatility", 0.15),
                max_drawdown_trigger=getattr(dynamic_config, "max_drawdown_trigger", 0.10),
                regime_lookback_days=getattr(dynamic_config, "regime_lookback_days", 60),
                vol_scaling_enabled=True,
                dd_protection_enabled=True,
                regime_weighting_enabled=bool(self._regime_info),
            )
            weighter = DynamicWeighter(weighter_config)

            # 市場データを準備
            # ポートフォリオリターンを計算（equal-weight proxy）
            returns_list = []
            for symbol, df in self._raw_data.items():
                if symbol in self._excluded_assets or symbol == "CASH":
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
                return self._raw_weights

            # 等重みポートフォリオリターン
            returns_df = pd.concat(returns_list, axis=1).dropna()
            if returns_df.empty:
                return self._raw_weights

            portfolio_returns = returns_df.mean(axis=1)

            # ポートフォリオ価値を計算（累積リターン）
            portfolio_value = 100.0 * (1 + portfolio_returns).cumprod()
            peak_value = portfolio_value.cummax()

            current_value = portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 100.0
            current_peak = peak_value.iloc[-1] if len(peak_value) > 0 else 100.0

            market_data = {
                "returns": portfolio_returns,
                "portfolio_value": current_value,
                "peak_value": current_peak,
            }

            # 動的重み調整を適用
            adjusted_weights = weighter.adjust_weights(
                weights=self._raw_weights,
                market_data=market_data,
                regime_info=self._regime_info if self._regime_info else None,
            )

            # 結果を保存
            self._dynamic_weighting_result = {
                "original_weights": self._raw_weights.copy(),
                "adjusted_weights": adjusted_weights,
                "market_data": {
                    "portfolio_value": current_value,
                    "peak_value": current_peak,
                    "drawdown": (current_peak - current_value) / current_peak if current_peak > 0 else 0,
                },
                "regime_info": self._regime_info,
            }

            # raw_weightsを更新
            self._raw_weights = adjusted_weights

            self._logger.info(
                "Dynamic weighting completed",
                vol_regime=self._regime_info.get("current_vol_regime", "unknown"),
                trend_regime=self._regime_info.get("current_trend_regime", "unknown"),
            )

        except ImportError as e:
            self._logger.warning(f"DynamicWeighter not available: {e}")
        except Exception as e:
            self._logger.warning(f"Dynamic weighting failed: {e}")

        # =================================================================
        # Return Maximization Integration (Phase 4)
        # =================================================================
        return_max_config = getattr(self.settings, "return_maximization", None)
        if return_max_config is not None and getattr(return_max_config, "enabled", False):
            self._raw_weights = self._apply_return_maximization(
                weights=self._raw_weights,
                returns_df=returns_df if 'returns_df' in dir() else None,
                previous_weights=previous_weights,
            )

        return self._raw_weights

    def _apply_kelly_allocation(
        self,
        adjusted_weights: dict[str, float],
        kelly_config,
        returns_df: "pd.DataFrame | None",
    ) -> dict[str, float]:
        """Kelly配分を適用"""
        kelly_fraction = getattr(kelly_config, "fraction", 0.25)
        kelly_max_weight = getattr(kelly_config, "max_weight", 0.25)
        kelly_weight_in_final = getattr(kelly_config, "weight_in_final", 0.5)
        min_trades = getattr(kelly_config, "min_trades", 20)

        allocator = KellyAllocator(fraction=kelly_fraction, max_weight=kelly_max_weight)

        kelly_weights = {}
        for symbol, weight in adjusted_weights.items():
            if symbol == "CASH" or symbol in self._excluded_assets:
                kelly_weights[symbol] = weight
                continue

            if symbol in self._raw_data and returns_df is not None:
                try:
                    df = self._raw_data[symbol]
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
                except Exception:
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

        self._logger.debug("Kelly allocation applied", assets=len(kelly_weights))
        return adjusted_weights

    def _apply_hysteresis_filter(
        self,
        adjusted_weights: dict[str, float],
        entry_exit_config,
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """ヒステリシスフィルタを適用"""
        entry_threshold = getattr(entry_exit_config, "entry_threshold", 0.3)
        exit_threshold = getattr(entry_exit_config, "exit_threshold", 0.1)

        hysteresis = HysteresisFilter(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        for symbol in list(adjusted_weights.keys()):
            if symbol == "CASH":
                continue

            signal_score = self._get_asset_signal_score(symbol)
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
        adjusted_weights: dict[str, float],
        macro_config,
    ) -> dict[str, float]:
        """マクロタイミングを適用"""
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
                "Macro timing applied",
                phase=phase_result.phase.value if phase_result else "unknown",
            )

        return adjusted_weights

    def _apply_return_maximization(
        self,
        weights: dict[str, float],
        returns_df: "pd.DataFrame | None",
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
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

        return_max_config = getattr(self.settings, "return_maximization", None)
        if return_max_config is None:
            return weights

        adjusted_weights = weights.copy()

        # 1. Kelly Allocation
        kelly_config = getattr(return_max_config, "kelly", None)
        if kelly_config is not None and getattr(kelly_config, "enabled", False):
            try:
                adjusted_weights = self._apply_kelly_allocation(
                    adjusted_weights, kelly_config, returns_df
                )
            except Exception as e:
                self._logger.warning(f"Kelly allocation failed: {e}")

        # 2. Hysteresis Filter
        entry_exit_config = getattr(return_max_config, "entry_exit", None)
        if entry_exit_config is not None and getattr(entry_exit_config, "use_hysteresis", False):
            try:
                adjusted_weights = self._apply_hysteresis_filter(
                    adjusted_weights, entry_exit_config, previous_weights
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
            "Return maximization completed",
            kelly_enabled=getattr(kelly_config, "enabled", False) if kelly_config else False,
            hysteresis_enabled=getattr(entry_exit_config, "use_hysteresis", False) if entry_exit_config else False,
            macro_enabled=getattr(macro_config, "enabled", False) if macro_config else False,
        )

        return adjusted_weights

    def _get_asset_signal_score(self, symbol: str) -> float | None:
        """
        Get the aggregate signal score for an asset.

        Args:
            symbol: Asset symbol

        Returns:
            Signal score (-1 to 1) or None if not available
        """
        # Check ensemble scores first
        if symbol in self._ensemble_scores:
            score_data = self._ensemble_scores[symbol]
            if isinstance(score_data, dict) and "combined_score" in score_data:
                return score_data["combined_score"]
            elif isinstance(score_data, (int, float)):
                return float(score_data)

        # Check strategy signals
        if symbol in self._signals:
            signal_data = self._signals[symbol]
            if isinstance(signal_data, dict):
                # Average all signal values
                scores = [v for v in signal_data.values() if isinstance(v, (int, float))]
                if scores:
                    return sum(scores) / len(scores)

        return None

    def _step_cmd016_integration(self) -> dict[str, Any]:
        """
        Step 12: CMD_016 Feature Integration (Phase 5).

        全cmd_016機能を統合して適用:
        - VIXキャッシュ配分
        - 相関ブレイク検出
        - ドローダウンプロテクション
        - セクターローテーション
        - シグナルフィルター（ヒステリシス、減衰、最低保有期間）

        Returns:
            統合結果の辞書
        """
        import pandas as pd

        if not CMD016_AVAILABLE or self.cmd016_integrator is None:
            self._logger.info("CMD_016 integration skipped (not available)")
            return {"enabled": False}

        try:
            # Prepare data
            weights = pd.Series(self._raw_weights) if self._raw_weights else pd.Series()
            if weights.empty:
                return {"enabled": False, "reason": "No weights to adjust"}

            # Prepare signals
            signals = pd.Series()
            for symbol in self._signals:
                score = self._get_asset_signal_score(symbol)
                if score is not None:
                    signals[symbol] = score

            # Calculate portfolio value from returns if available
            portfolio_value = self._portfolio_value
            returns_df = None

            returns_list = []
            for symbol, df in self._raw_data.items():
                if symbol in self._excluded_assets or symbol == "CASH":
                    continue
                if hasattr(df, "to_pandas"):
                    df = df.to_pandas()
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                if "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    returns.name = symbol
                    returns_list.append(returns)

            if returns_list:
                returns_df = pd.concat(returns_list, axis=1).dropna()
                # Calculate portfolio value
                if not returns_df.empty:
                    portfolio_returns = returns_df.mean(axis=1)
                    portfolio_value = 100000.0 * (1 + portfolio_returns).cumprod().iloc[-1]
                    self._portfolio_value = portfolio_value

            # Get VIX value if available
            vix_value = self._vix_value
            if vix_value is None and "^VIX" in self._raw_data:
                vix_df = self._raw_data["^VIX"]
                if hasattr(vix_df, "to_pandas"):
                    vix_df = vix_df.to_pandas()
                if "close" in vix_df.columns:
                    vix_value = vix_df["close"].iloc[-1]
                    self._vix_value = vix_value

            # Apply full integration
            result = self.cmd016_integrator.integrate_all(
                base_weights=weights,
                signals=signals if not signals.empty else None,
                portfolio_value=portfolio_value,
                vix_value=vix_value,
                returns=returns_df,
                macro_indicators=None,  # Could be fetched from FRED
            )

            # Update raw_weights with adjusted weights
            self._raw_weights = result.adjusted_weights.to_dict()
            self._cmd016_result = result.to_dict()

            # Log feature status
            feature_status = self.cmd016_integrator.get_feature_status()
            active_features = [k for k, v in feature_status.items() if v]

            self._logger.info(
                "CMD_016 integration completed",
                active_features=len(active_features),
                cash_ratio=result.cash_ratio,
                vix_value=vix_value,
                portfolio_value=portfolio_value,
            )

            # Add warnings to pipeline warnings
            if result.warnings:
                self._warnings.extend(result.warnings)

            return self._cmd016_result

        except Exception as e:
            self._logger.warning(f"CMD_016 integration failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _cmd017_prepare_returns(self) -> "pd.DataFrame | None":
        """CMD_017: リターンデータを準備"""
        import pandas as pd

        returns_list = []
        for symbol, df in self._raw_data.items():
            if symbol in self._excluded_assets or symbol == "CASH":
                continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            if "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                returns.name = symbol
                returns_list.append(returns)

        if not returns_list:
            return None
        returns_df = pd.concat(returns_list, axis=1).dropna()
        return returns_df if not returns_df.empty else None

    def _cmd017_apply_allocation(
        self,
        allocation_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> "pd.Series":
        """CMD_017: 配分最適化を適用（NCO, BL, CVaR, TC）"""
        import pandas as pd

        allocation_method = allocation_config.get("method", "nco")

        # NCO
        if allocation_method == "nco" and allocation_config.get("nco", {}).get("enabled", True):
            try:
                nco_params = allocation_config.get("nco", {})
                nco_cfg = NCOConfig(
                    n_clusters=nco_params.get("n_clusters", 5),
                    intra_method=nco_params.get("intra_method", "min_variance"),
                )
                nco = NestedClusteredOptimization(config=nco_cfg)
                nco_result = nco.fit(returns_df)
                if nco_result.is_valid:
                    blend_factor = 0.5
                    for symbol in nco_result.weights.index:
                        if symbol in current_weights_series.index:
                            current_weights_series[symbol] = (
                                blend_factor * nco_result.weights[symbol] +
                                (1 - blend_factor) * current_weights_series[symbol]
                            )
                    result["allocation"]["nco"] = {"status": "applied", "n_clusters": nco_params.get("n_clusters", 5)}
                else:
                    result["allocation"]["nco"] = {"status": "invalid", "reason": "optimization failed"}
            except Exception as e:
                result["allocation"]["nco"] = {"status": "error", "message": str(e)}

        # Black-Litterman
        if allocation_config.get("black_litterman", {}).get("enabled", True):
            try:
                bl_params = allocation_config.get("black_litterman", {})
                bl_model = BlackLittermanModel(
                    tau=bl_params.get("tau", 0.05),
                    risk_aversion=bl_params.get("risk_aversion", 2.5),
                )
                cov_matrix = returns_df.cov() * 252
                equilibrium_returns = bl_model.compute_equilibrium_returns(cov_matrix, current_weights_series)
                result["allocation"]["black_litterman"] = {
                    "status": "computed", "tau": bl_params.get("tau", 0.05),
                    "equilibrium_returns_mean": float(equilibrium_returns.mean()),
                }
            except Exception as e:
                result["allocation"]["black_litterman"] = {"status": "error", "message": str(e)}

        # CVaR
        if allocation_config.get("cvar", {}).get("enabled", True):
            try:
                cvar_params = allocation_config.get("cvar", {})
                cvar_optimizer = CVaROptimizer(alpha=cvar_params.get("alpha", 0.05))
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                cvar_value = cvar_optimizer.compute_cvar(portfolio_returns.values)
                result["allocation"]["cvar"] = {
                    "status": "computed", "cvar": float(cvar_value),
                    "alpha": cvar_params.get("alpha", 0.05),
                }
            except Exception as e:
                result["allocation"]["cvar"] = {"status": "error", "message": str(e)}

        # Transaction Cost
        if allocation_config.get("transaction_cost", {}).get("enabled", True):
            try:
                tc_params = allocation_config.get("transaction_cost", {})
                tc_optimizer = TransactionCostOptimizer(
                    cost_aversion=tc_params.get("cost_aversion", 1.0), max_weight=0.20,
                )
                tc_result = tc_optimizer.optimize(returns_df, current_weights_series.to_dict())
                if tc_result.converged or tc_result.turnover < tc_params.get("max_turnover", 0.20):
                    for symbol, weight in tc_result.optimal_weights.items():
                        current_weights_series[symbol] = weight
                    result["allocation"]["transaction_cost"] = {
                        "status": "applied", "turnover": float(tc_result.turnover),
                        "cost": float(tc_result.transaction_cost),
                    }
                else:
                    result["allocation"]["transaction_cost"] = {"status": "skipped", "reason": "turnover_limit_exceeded"}
            except Exception as e:
                result["allocation"]["transaction_cost"] = {"status": "error", "message": str(e)}

        return current_weights_series

    def _cmd017_apply_walkforward(
        self,
        wf_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> None:
        """CMD_017: Walk-Forward強化を適用"""
        if wf_config.get("adaptive_window", {}).get("enabled", True):
            try:
                aw_params = wf_config.get("adaptive_window", {})
                selector = AdaptiveWindowSelector(
                    min_window=aw_params.get("min_window", 126),
                    max_window=aw_params.get("max_window", 756),
                    default_window=aw_params.get("default_window", 504),
                )
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                regime_change = selector.detect_regime_change(portfolio_returns)
                vol_regime = VolatilityRegime.HIGH_VOL if regime_change.detected else VolatilityRegime.NORMAL
                optimal_window = selector.compute_optimal_window(
                    returns=portfolio_returns, volatility_regime=vol_regime,
                    regime_change_detected=regime_change.detected,
                )
                result["walkforward"]["adaptive_window"] = {
                    "status": "computed", "optimal_window": optimal_window,
                    "regime_change_detected": regime_change.detected,
                }
            except Exception as e:
                result["walkforward"]["adaptive_window"] = {"status": "error", "message": str(e)}

    def _cmd017_apply_validation(
        self,
        val_config: dict,
        returns_df: "pd.DataFrame",
        current_weights_series: "pd.Series",
        result: dict,
    ) -> None:
        """CMD_017: 検証を適用"""
        import numpy as np

        # Statistical Significance
        if val_config.get("synthetic_data", {}).get("enabled", True):
            try:
                portfolio_returns = (returns_df * current_weights_series).sum(axis=1)
                tester = StatisticalSignificanceTester(
                    n_bootstrap=val_config.get("synthetic_data", {}).get("n_simulations", 500),
                    confidence_level=0.95,
                )
                sig_result = tester.test_sharpe_significance(portfolio_returns.values)
                result["validation"]["sharpe_significance"] = {
                    "status": "computed",
                    "observed_sharpe": float(sig_result["observed_sharpe"]),
                    "ci_lower": float(sig_result["ci_lower"]),
                    "ci_upper": float(sig_result["ci_upper"]),
                    "p_value": float(sig_result["p_value"]),
                    "significant": sig_result["significant"],
                }
            except Exception as e:
                result["validation"]["sharpe_significance"] = {"status": "error", "message": str(e)}

        # Purged K-Fold
        if val_config.get("purged_kfold", {}).get("enabled", True):
            try:
                pkf_params = val_config.get("purged_kfold", {})
                n_splits = pkf_params.get("n_splits", 5)
                purge_gap = pkf_params.get("purge_gap", 5)
                pkf = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
                n_samples = len(returns_df)
                splits = list(pkf.split(np.arange(n_samples)))
                result["validation"]["purged_kfold"] = {
                    "status": "configured", "n_splits": n_splits, "actual_splits": len(splits),
                }
            except Exception as e:
                result["validation"]["purged_kfold"] = {"status": "error", "message": str(e)}

    def _step_cmd017_integration(self) -> dict[str, Any]:
        """
        Step 13: CMD_017 Feature Integration (Phase 6).

        全cmd_017機能を統合して適用:
        - 配分最適化: NCO, Black-Litterman, CVaR, トランザクションコスト
        - Walk-Forward強化: 適応ウィンドウ
        - 検証: 合成データ, Purged K-Fold

        Returns:
            統合結果の辞書
        """
        import pandas as pd

        if not CMD017_AVAILABLE:
            self._logger.info("CMD_017 integration skipped (not available)")
            return {"enabled": False}

        try:
            cmd017_config = getattr(self.settings, "cmd_017_features", None)
            if cmd017_config is None:
                self._logger.info("CMD_017 config not found, using defaults")
                cmd017_config = {}

            result = {
                "enabled": True, "allocation": {}, "ml": {},
                "walkforward": {}, "execution": {}, "risk": {}, "validation": {},
            }

            # Prepare returns data
            returns_df = self._cmd017_prepare_returns()
            if returns_df is None:
                self._logger.warning("No returns data for CMD_017 integration")
                return {"enabled": False, "reason": "No returns data"}

            # Prepare weights series
            current_weights = self._raw_weights or {}
            current_weights_series = pd.Series(current_weights)
            current_weights_series = current_weights_series.reindex(returns_df.columns).fillna(0)

            # 1. Allocation Optimization
            allocation_config = cmd017_config.get("allocation", {})
            current_weights_series = self._cmd017_apply_allocation(
                allocation_config, returns_df, current_weights_series, result
            )

            # 2. Walk-Forward Enhancement
            wf_config = cmd017_config.get("walkforward", {})
            self._cmd017_apply_walkforward(wf_config, returns_df, current_weights_series, result)

            # 3. Validation
            val_config = cmd017_config.get("validation", {})
            self._cmd017_apply_validation(val_config, returns_df, current_weights_series, result)

            # Update raw weights
            self._raw_weights = current_weights_series.to_dict()
            self._cmd017_result = result

            self._logger.info(
                "CMD_017 integration completed",
                allocation_method=allocation_method,
                features_applied=sum(
                    1 for cat in result.values()
                    if isinstance(cat, dict)
                    for item in cat.values()
                    if isinstance(item, dict) and item.get("status") in ("applied", "computed", "configured")
                ),
            )

            return result

        except Exception as e:
            self._logger.warning(f"CMD_017 integration failed: {e}")
            return {"enabled": True, "error": str(e)}

    def _step_anomaly_detection(self) -> AnomalyDetectionResult:
        """Step 10: Detect anomalies."""
        result = self.anomaly_detector.check_all(
            quality_reports=self._quality_reports,
            portfolio_risk=self._risk_metrics,
            correlation_matrix=self._correlation_matrix,
            price_data=self._raw_data,
        )

        if result.has_warnings:
            self._warnings.extend([a.description for a in result.anomalies])

        return result

    def _step_fallback(
        self,
        previous_weights: dict[str, float],
        reason: str,
        adopted_strategies: int,
    ) -> FallbackState:
        """Step 11: Apply fallback mode."""
        applied_weights = self.fallback_handler.apply_fallback(
            previous_weights=previous_weights,
            new_weights=self._smoothed_weights,
            reason=reason,
            adopted_strategies=adopted_strategies,
        )

        if self._audit_logger:
            self._audit_logger.log_fallback_triggered(
                mode=self.fallback_handler.current_mode.value,
                trigger_reason=reason,
                previous_weights=previous_weights,
                new_weights=applied_weights,
            )

        return self.fallback_handler.current_state

    def _step_output_generation(self, run_id: str) -> None:
        """Step 12: Generate output files."""
        import json

        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        output = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "weights": self._final_weights,
            "diagnostics": self._create_diagnostics(None),
        }

        output_path = output_dir / f"weights_{run_id}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        self._logger.info("Output generated", path=str(output_path))

    def _step_logging(self, run_id: str) -> None:
        """Step 13: Save run info and logs."""
        log_dir = self._config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save run info for reproducibility
        run_info = get_run_info(
            run_id=run_id,
            seed=self._config.seed,
            config=self.settings.model_dump() if hasattr(self.settings, "model_dump") else {},
        )
        save_run_info(run_info, log_dir / f"run_info_{run_id}.json")

        self._logger.info("Logging completed", run_id=run_id)

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _create_diagnostics(
        self,
        fallback_state: FallbackState | None,
    ) -> dict[str, Any]:
        """Create diagnostics dictionary.

        In lightweight mode (task_036_4), returns minimal diagnostics for performance.
        """
        # Lightweight mode: minimal diagnostics for backtest performance
        if self._config.lightweight_mode or self._config.skip_diagnostics:
            return {
                "lightweight_mode": True,
                "fallback_mode": fallback_state.mode.value if fallback_state else None,
                "errors": self._errors if self._errors else [],
            }

        # Full diagnostics for production mode
        diagnostics = {
            "excluded_assets": self._excluded_assets,
            "quality_reports_count": len(self._quality_reports),
            "strategies_evaluated": len(self._evaluations),
            "risk_metrics": self._risk_metrics,
            "fallback_mode": fallback_state.mode.value if fallback_state else None,
            "warnings": self._warnings,
            "errors": self._errors,
        }

        # Add expanded mode summaries if available
        if hasattr(self, '_fetch_summary') and self._fetch_summary:
            diagnostics["fetch_summary"] = {
                k: len(v) for k, v in self._fetch_summary.items()
            }

        if hasattr(self, '_quality_summary') and self._quality_summary:
            diagnostics["quality_summary"] = {
                k: len(v) for k, v in self._quality_summary.items()
            }

        return diagnostics

    def _create_error_result(
        self,
        run_id: str,
        start_time: datetime,
        error: str,
    ) -> PipelineResult:
        """Create error result."""
        end_time = datetime.now(timezone.utc)
        return PipelineResult(
            run_id=run_id,
            status=PipelineStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            weights={},
            diagnostics={"error": error},
            fallback_state=None,
            step_results=self._step_results,
            errors=[error] + self._errors,
            warnings=self._warnings,
        )
