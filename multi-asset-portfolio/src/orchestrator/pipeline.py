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
from typing import TYPE_CHECKING, Any, Callable, Optional

import structlog

if TYPE_CHECKING:
    import polars as pl

    from src.config.schemas import PortfolioWeights, ValidationReport
    from src.config.settings import Settings
    from src.utils.storage_backend import StorageBackend

from src.orchestrator.anomaly_detector import AnomalyDetectionResult, AnomalyDetector
from src.orchestrator.data_preparation import DataPreparation, DataPreparationResult
from src.orchestrator.fallback import FallbackHandler, FallbackMode, FallbackState
from src.orchestrator.risk_allocation import AssetAllocator, RiskEstimator
from src.orchestrator.signal_generation import SignalGenerator, StrategyEvaluator
from src.orchestrator.weight_calculation import GateChecker, StrategyWeighter

# Pipeline step modules (extracted from Pipeline class for modularity)
from src.orchestrator.pipeline_steps import FeatureIntegrator, RegimeDetector, OutputHandler
from src.orchestrator.pipeline_steps.output_handlers import OutputConfig, DiagnosticsData
from src.orchestrator.pipeline_steps.regime_detection import RegimeInfo

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
    set_log_collector,
)
from src.utils.pipeline_log_collector import PipelineLogCollector
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

    # NOTE: use_precomputed_signals removed in v2.0
    # Precomputed signals are now ALWAYS used (SignalPrecomputer is required)
    # Legacy SignalGenerator on-the-fly computation has been removed


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
        storage_backend: "Optional[StorageBackend]" = None,
        signal_precomputer: Any = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            settings: Application settings
            config: Pipeline-specific configuration
            storage_backend: Optional StorageBackend for S3 price cache support
            signal_precomputer: Optional SignalPrecomputer for precomputed signals
        """
        self._settings = settings
        self._config = config or PipelineConfig()
        self._logger = logger.bind(component="pipeline")
        self._audit_logger: AuditLogger | None = None
        self._storage_backend = storage_backend
        # Signal precomputer must be set at construction time for consistent injection
        self._signal_precomputer = signal_precomputer

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

        # Pipeline step modules (extracted for modularity)
        self._feature_integrator: FeatureIntegrator | None = None
        self._regime_detector: RegimeDetector | None = None
        self._output_handler: OutputHandler | None = None

        # Precomputed signals support (15-year backtest optimization)
        # Note: _signal_precomputer is set in constructor for consistent injection
        # Only initialize if not already set (for backward compatibility)
        if not hasattr(self, '_signal_precomputer') or self._signal_precomputer is None:
            self._signal_precomputer = signal_precomputer
        self._as_of_date: datetime | None = None  # Current evaluation date for precomputed lookup

        # Runtime state
        self._step_results: list[StepResult] = []
        self._errors: list[str] = []
        self._warnings: list[str] = []

        # Log collector for unified logging (Phase 1 logging infrastructure)
        self._log_collector: PipelineLogCollector | None = None
        self._progress_tracker: Any = None  # Optional ProgressTracker for viewer integration

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
    def config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config

    @property
    def log_collector(self) -> PipelineLogCollector | None:
        """Get the log collector for this pipeline run."""
        return self._log_collector

    def set_progress_tracker(self, tracker: Any) -> None:
        """
        Set the progress tracker for viewer integration.

        Args:
            tracker: ProgressTracker instance for real-time updates
        """
        self._progress_tracker = tracker

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
                storage_backend=self._storage_backend,
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
                signal_precomputer=self._signal_precomputer,
            )
        # Update precomputer reference if it was set after evaluator creation
        elif self._signal_precomputer is not None:
            self._strategy_evaluator._signal_precomputer = self._signal_precomputer
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

    @property
    def feature_integrator(self) -> FeatureIntegrator:
        """Get feature integrator for CMD016/CMD017 integration."""
        if self._feature_integrator is None:
            self._feature_integrator = FeatureIntegrator(
                settings=self.settings,
                cmd016_integrator=self.cmd016_integrator,
            )
        return self._feature_integrator

    @property
    def regime_detector(self) -> RegimeDetector:
        """Get regime detector for regime detection and dynamic weighting."""
        if self._regime_detector is None:
            self._regime_detector = RegimeDetector(settings=self.settings)
        return self._regime_detector

    @property
    def output_handler(self) -> OutputHandler:
        """Get output handler for output generation and logging."""
        if self._output_handler is None:
            output_config = OutputConfig(
                output_dir=self._config.output_dir,
                log_dir=self._config.log_dir,
                lightweight_mode=self._config.lightweight_mode,
                skip_diagnostics=self._config.skip_diagnostics,
                seed=self._config.seed,
            )
            self._output_handler = OutputHandler(
                config=output_config,
                settings=self.settings,
            )
        return self._output_handler

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

        # Initialize log collector for unified logging
        self._log_collector = PipelineLogCollector(run_id=run_id)
        set_log_collector(self._log_collector)

        # Attach ProgressTracker if available (for viewer integration)
        if self._progress_tracker is not None:
            self._log_collector.attach_progress_tracker(self._progress_tracker)

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
        finally:
            # Clear global log collector reference
            set_log_collector(None)

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
        # Store as_of_date for precomputed signal lookup
        self._as_of_date = as_of_date

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
        Step 3: Load precomputed signals for each asset.

        v2.0: Precomputed signals are now REQUIRED (legacy on-the-fly computation removed).
        All 64 registered signals are pre-computed by SignalPrecomputer and loaded from cache.

        Raises:
            ValueError: If SignalPrecomputer is not set
        """
        if self._signal_precomputer is None:
            # Fallback for backward compatibility: use SignalGenerator
            # This path should be deprecated and removed in future versions
            self._logger.warning(
                "SignalPrecomputer not set - falling back to legacy SignalGenerator. "
                "This is deprecated and will be removed in a future version."
            )
            result = self.signal_generator.generate_signals(
                raw_data=self._raw_data,
                excluded_assets=self._excluded_assets,
            )
            self._signals = result.signals
            return self._signals

        # Load precomputed signals (primary path)
        self._signals = self._load_precomputed_signals()
        self._logger.debug(
            "Using precomputed signals",
            as_of_date=self._as_of_date,
            n_signals=len(self._signals),
        )
        return self._signals

    def _load_precomputed_signals(self) -> dict[str, dict[str, Any]]:
        """
        Load precomputed signals from SignalPrecomputer cache.

        This method retrieves all pre-calculated signals for the current as_of_date
        from the SignalPrecomputer cache, providing a 40x speedup for backtests.

        Returns:
            Dictionary mapping symbol -> {signal_name: value, ...}
            (Same format as SignalGenerator.generate_signals() for compatibility)
        """
        if self._signal_precomputer is None or self._as_of_date is None:
            self._logger.warning("Precomputer or as_of_date not set, returning empty signals")
            return {}

        # SignalPrecomputer returns: {signal_name: {ticker: value}}
        # StrategyEvaluator expects: {symbol: {signal_name: value}}
        # We need to transpose the structure

        raw_signals: dict[str, dict[str, float]] = {}  # signal_name -> {ticker: value}
        signals: dict[str, dict[str, Any]] = {}  # symbol -> {signal_name: value}

        try:
            # Get list of available cached signals
            cached_signal_names = self._signal_precomputer.list_cached_signals()

            if not cached_signal_names:
                self._logger.warning("No cached signals found in precomputer")
                return {}

            # Load each signal at the current date
            for signal_name in cached_signal_names:
                try:
                    signal_values = self._signal_precomputer.get_signals_at_date(
                        signal_name=signal_name,
                        date=self._as_of_date,
                    )
                    if signal_values:
                        raw_signals[signal_name] = signal_values
                except Exception as e:
                    self._logger.debug(f"Failed to load signal {signal_name}: {e}")

            # Transpose: {signal_name: {ticker: value}} -> {symbol: {signal_name: value}}
            all_symbols: set[str] = set()
            for signal_values in raw_signals.values():
                all_symbols.update(signal_values.keys())

            for symbol in all_symbols:
                signals[symbol] = {}
                for signal_name, signal_values in raw_signals.items():
                    if symbol in signal_values:
                        signals[symbol][signal_name] = signal_values[symbol]

            self._logger.debug(
                "Loaded precomputed signals",
                n_signals=len(raw_signals),
                n_symbols=len(signals),
                signals=list(raw_signals.keys())[:5],  # Log first 5 signal names
            )

        except Exception as e:
            self._logger.warning(f"Failed to load precomputed signals: {e}")

        return signals

    def _step_strategy_evaluation(self) -> list[Any]:
        """
        Step 4: Evaluate strategies on test data.

        Supports two modes:
        - Standard mode: Uses SignalResult objects from SignalGenerator
        - Precomputed mode: Uses cached signal values from SignalPrecomputer

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

        Delegates to RegimeDetector module for regime detection.

        Returns:
            レジーム情報辞書（vol_regime, trend_regime等）
        """
        # Delegate to RegimeDetector module
        regime_info = self.regime_detector.detect_regime(
            raw_data=self._raw_data,
            excluded_assets=set(self._excluded_assets),
        )

        # Update internal state
        self._regime_info = regime_info.to_dict()

        return self._regime_info

    def _step_ensemble_combine(self) -> dict[str, Any]:
        """
        Step 7: Combine strategy scores using EnsembleCombiner.

        Delegates to RegimeDetector module for ensemble combination.

        Returns:
            アンサンブル統合結果
        """
        # Delegate to RegimeDetector module
        self._ensemble_scores = self.regime_detector.combine_ensemble_scores(
            evaluations=self._evaluations,
        )

        return self._ensemble_scores

    def _step_dynamic_weighting(
        self,
        previous_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Step 11: Apply dynamic weight adjustments.

        Delegates to RegimeDetector module for dynamic weighting.
        Return maximization integration (Phase 4) is handled separately.

        Args:
            previous_weights: 前期の重み

        Returns:
            動的調整後の重み
        """
        import pandas as pd

        # Delegate to RegimeDetector module
        result = self.regime_detector.apply_dynamic_weighting(
            raw_weights=self._raw_weights,
            raw_data=self._raw_data,
            excluded_assets=set(self._excluded_assets),
            regime_info=self._regime_info,
        )

        # Update internal state
        self._raw_weights = result.adjusted_weights
        self._dynamic_weighting_result = result.to_dict()

        # Prepare returns_df for return maximization
        returns_df: pd.DataFrame | None = None
        if RETURN_MAXIMIZATION_AVAILABLE:
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
            if returns_list:
                returns_df = pd.concat(returns_list, axis=1).dropna()

        # Return Maximization Integration (Phase 4) - kept in pipeline
        return_max_config = getattr(self.settings, "return_maximization", None)
        if return_max_config is not None and getattr(return_max_config, "enabled", False):
            self._raw_weights = self._apply_return_maximization(
                weights=self._raw_weights,
                returns_df=returns_df,
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

        Delegates to FeatureIntegrator module for CMD016 integration.

        Returns:
            統合結果の辞書
        """
        from src.orchestrator.pipeline_steps.feature_integration import FeatureIntegrationContext

        # Build context for feature integration
        context = FeatureIntegrationContext(
            raw_data=self._raw_data,
            raw_weights=self._raw_weights,
            signals=self._signals,
            excluded_assets=set(self._excluded_assets),
            portfolio_value=self._portfolio_value,
            vix_value=self._vix_value,
            settings=self.settings,
        )

        # Delegate to FeatureIntegrator
        result = self.feature_integrator.integrate_cmd016(
            context=context,
            get_asset_signal_score=self._get_asset_signal_score,
        )

        # Update internal state based on result
        if result.get("enabled") and "adjusted_weights" in result:
            self._raw_weights = result["adjusted_weights"]

        # Add warnings to pipeline warnings
        if result.get("warnings"):
            self._warnings.extend(result["warnings"])

        self._cmd016_result = result
        return result

    def _step_cmd017_integration(self) -> dict[str, Any]:
        """
        Step 13: CMD_017 Feature Integration (Phase 6).

        Delegates to FeatureIntegrator module for CMD017 integration.

        Returns:
            統合結果の辞書
        """
        from src.orchestrator.pipeline_steps.feature_integration import FeatureIntegrationContext

        # Build context for feature integration
        context = FeatureIntegrationContext(
            raw_data=self._raw_data,
            raw_weights=self._raw_weights,
            signals=self._signals,
            excluded_assets=set(self._excluded_assets),
            portfolio_value=self._portfolio_value,
            vix_value=self._vix_value,
            settings=self.settings,
        )

        # Delegate to FeatureIntegrator
        result = self.feature_integrator.integrate_cmd017(context=context)

        # Update internal state based on result
        if result.get("enabled") and "adjusted_weights" in result:
            self._raw_weights = result["adjusted_weights"]

        self._cmd017_result = result
        return result

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
        """Step 12: Generate output files.

        Delegates to OutputHandler module for output generation.
        """
        # Build diagnostics data
        diagnostics_data = DiagnosticsData(
            excluded_assets=set(self._excluded_assets),
            quality_reports=self._quality_reports,
            evaluations=self._evaluations,
            risk_metrics=self._risk_metrics,
            warnings=self._warnings,
            errors=self._errors,
            fetch_summary=getattr(self, '_fetch_summary', None),
            quality_summary=getattr(self, '_quality_summary', None),
        )

        # Delegate to OutputHandler
        self.output_handler.generate_output(
            run_id=run_id,
            final_weights=self._final_weights,
            diagnostics_data=diagnostics_data,
            fallback_state=self.fallback_handler.current_state if self._fallback_handler else None,
        )

    def _step_logging(self, run_id: str) -> None:
        """Step 13: Save run info and logs.

        Delegates to OutputHandler module for logging.
        """
        # Delegate to OutputHandler
        self.output_handler.save_run_logs(run_id=run_id)

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _create_diagnostics(
        self,
        fallback_state: FallbackState | None,
    ) -> dict[str, Any]:
        """Create diagnostics dictionary.

        Delegates to OutputHandler module for diagnostics creation.
        """
        # Build diagnostics data
        diagnostics_data = DiagnosticsData(
            excluded_assets=set(self._excluded_assets),
            quality_reports=self._quality_reports,
            evaluations=self._evaluations,
            risk_metrics=self._risk_metrics,
            warnings=self._warnings,
            errors=self._errors,
            fetch_summary=getattr(self, '_fetch_summary', None),
            quality_summary=getattr(self, '_quality_summary', None),
        )

        # Delegate to OutputHandler
        return self.output_handler.create_diagnostics(
            data=diagnostics_data,
            fallback_state=fallback_state,
        )

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
