"""
Global Settings - Pydantic v2 Configuration Schema

This module defines the complete configuration schema for the Multi-Asset Portfolio System.
All settings are validated at runtime using Pydantic v2.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageConfig


class RebalanceFrequency(str, Enum):
    """Rebalance frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ExecutionDay(str, Enum):
    """Execution day options for rebalancing."""

    FIRST_BUSINESS_DAY = "first_business_day"
    LAST_BUSINESS_DAY = "last_business_day"
    SPECIFIC_DAY = "specific_day"


class DegradationLevel(str, Enum):
    """System degradation levels."""

    LEVEL_0 = "level_0"  # Normal operation
    LEVEL_1 = "level_1"  # Reduced operation
    LEVEL_2 = "level_2"  # Cash evacuation
    LEVEL_3 = "level_3"  # Emergency stop


class FallbackMode(str, Enum):
    """Fallback modes for anomaly situations.

    NONE: Normal operation, no fallback active
    HOLD_PREVIOUS: Mode 1 - Maintain previous day weights
    EQUAL_WEIGHT: Mode 2 - Retreat to equal distribution
    CASH: Mode 3 - Cash evacuation (w=0)
    """

    NONE = "none"  # Normal operation
    HOLD_PREVIOUS = "hold_previous"  # Mode 1
    EQUAL_WEIGHT = "equal_weight"  # Mode 2
    CASH = "cash"  # Mode 3


class AllocationMethod(str, Enum):
    """Asset allocation methods."""

    HRP = "HRP"  # Hierarchical Risk Parity
    NCO = "nco"  # Nested Clustered Optimization
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    EQUAL_WEIGHT = "equal_weight"


# =============================================================================
# Rebalance Settings
# =============================================================================
class RebalanceConfig(BaseModel):
    """Rebalance configuration."""

    model_config = ConfigDict(frozen=True)

    frequency: RebalanceFrequency = Field(
        default=RebalanceFrequency.MONTHLY,
        description="Rebalance frequency",
    )
    execution_day: ExecutionDay = Field(
        default=ExecutionDay.LAST_BUSINESS_DAY,
        description="Which day to execute rebalancing",
    )
    specific_day: int | None = Field(
        default=None,
        ge=1,
        le=28,
        description="Specific day of month (1-28) if execution_day is SPECIFIC_DAY",
    )
    time_utc: str = Field(
        default="00:00:00",
        description="Execution time in UTC (HH:MM:SS format)",
    )
    min_trade_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.02,
        description="Minimum position change to trigger a trade (0.02 = 2%)",
    )
    max_turnover_pct: Annotated[float, Field(ge=0.0, le=200.0)] = Field(
        default=100.0,
        description="Maximum monthly turnover percentage",
    )

    @model_validator(mode="after")
    def validate_specific_day(self) -> "RebalanceConfig":
        if self.execution_day == ExecutionDay.SPECIFIC_DAY and self.specific_day is None:
            raise ValueError("specific_day is required when execution_day is SPECIFIC_DAY")
        return self


# =============================================================================
# Walk-Forward Validation Settings
# =============================================================================
class WalkForwardConfig(BaseModel):
    """Walk-forward validation parameters."""

    model_config = ConfigDict(frozen=True)

    train_period_days: Annotated[int, Field(gt=0)] = Field(
        default=504,
        description="Training period in business days (504 = ~2 years)",
    )
    test_period_days: Annotated[int, Field(gt=0)] = Field(
        default=126,
        description="Test period in business days (126 = ~6 months)",
    )
    step_period_days: Annotated[int, Field(gt=0)] = Field(
        default=21,
        description="Step/slide period in business days (21 = ~1 month)",
    )
    min_train_samples: Annotated[int, Field(gt=0)] = Field(
        default=200,
        description="Minimum number of training samples required",
    )
    purge_gap_days: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Gap between train and test to prevent data leakage",
    )
    embargo_days: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Embargo period after test",
    )


# =============================================================================
# Cost Model Settings
# =============================================================================
class MarketCapAdjustment(BaseModel):
    """Cost adjustment by market cap."""

    model_config = ConfigDict(frozen=True)

    threshold_jpy: int | None = Field(
        default=None,
        description="Market cap threshold in JPY",
    )
    spread_multiplier: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Multiplier for spread cost",
    )
    slippage_multiplier: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Multiplier for slippage cost",
    )


class CostModelConfig(BaseModel):
    """Transaction cost model configuration."""

    model_config = ConfigDict(frozen=True)

    spread_bps: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Bid-ask spread in basis points",
    )
    commission_bps: Annotated[float, Field(ge=0.0)] = Field(
        default=5.0,
        description="Trading commission in basis points",
    )
    slippage_bps: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Expected slippage in basis points",
    )
    large_cap: MarketCapAdjustment = Field(
        default_factory=lambda: MarketCapAdjustment(
            threshold_jpy=1_000_000_000_000,
            spread_multiplier=0.5,
            slippage_multiplier=0.5,
        )
    )
    mid_cap: MarketCapAdjustment = Field(
        default_factory=lambda: MarketCapAdjustment(
            threshold_jpy=100_000_000_000,
            spread_multiplier=1.0,
            slippage_multiplier=1.0,
        )
    )
    small_cap: MarketCapAdjustment = Field(
        default_factory=lambda: MarketCapAdjustment(
            spread_multiplier=1.5,
            slippage_multiplier=2.0,
        )
    )
    rebalance_overhead_jpy: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Fixed cost per rebalance in JPY",
    )

    @property
    def total_one_way_bps(self) -> float:
        """Total one-way transaction cost in basis points."""
        return self.spread_bps + self.commission_bps + self.slippage_bps


# =============================================================================
# Hard Gates Settings
# =============================================================================
class HardGatesConfig(BaseModel):
    """Hard gate thresholds for strategy adoption."""

    model_config = ConfigDict(frozen=True)

    min_sharpe_ratio: float = Field(
        default=0.5,
        description="Minimum Sharpe ratio required",
    )
    max_drawdown_pct: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        default=25.0,
        description="Maximum drawdown percentage allowed",
    )
    min_win_rate_pct: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        default=45.0,
        description="Minimum win rate percentage",
    )
    min_profit_factor: Annotated[float, Field(gt=0.0)] = Field(
        default=1.2,
        description="Minimum profit factor (total profit / total loss)",
    )
    min_trades: Annotated[int, Field(ge=1)] = Field(
        default=30,
        description="Minimum number of trades for statistical significance",
    )
    min_expected_value: float = Field(
        default=0.0,
        description="Minimum expected value after costs",
    )

    # Optional gates
    min_calmar_ratio: float | None = Field(
        default=0.3,
        description="Minimum Calmar ratio (return / max DD)",
    )
    max_volatility_pct: float | None = Field(
        default=30.0,
        description="Maximum annualized volatility percentage",
    )
    min_recovery_factor: float | None = Field(
        default=1.0,
        description="Minimum recovery factor (cumulative profit / max DD)",
    )

    # Warning thresholds (softer than hard gates)
    warn_sharpe_ratio: float = Field(default=1.0)
    warn_drawdown_pct: float = Field(default=15.0)
    warn_win_rate_pct: float = Field(default=50.0)


# =============================================================================
# Strategy Selection Settings (Top-N or Gate-based)
# =============================================================================
class StrategySelectionMethod(str, Enum):
    """Strategy selection method options."""

    TOP_N = "top_n"
    GATE = "gate"


class WeightMethod(str, Enum):
    """Weight calculation method options."""

    SOFTMAX = "softmax"
    EQUAL = "equal"


class StrategySelectionConfig(BaseModel):
    """Strategy selection configuration.

    Controls how strategies are selected for the portfolio:
    - top_n: Select top N strategies by score
    - gate: Use hard gates (legacy mode)
    """

    model_config = ConfigDict(frozen=True)

    method: StrategySelectionMethod = Field(
        default=StrategySelectionMethod.TOP_N,
        description="Selection method: 'top_n' or 'gate'",
    )
    top_n: Annotated[int, Field(ge=1)] = Field(
        default=10,
        description="Number of strategies to adopt (for top_n method)",
    )
    cash_score: float = Field(
        default=0.0,
        description="Fixed score for cash position (baseline)",
    )
    min_score: float = Field(
        default=-999.0,
        description="Minimum score threshold (strategies below this are excluded)",
    )
    weight_method: WeightMethod = Field(
        default=WeightMethod.SOFTMAX,
        description="Weight calculation method: 'softmax' or 'equal'",
    )
    softmax_temperature: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Temperature for softmax (higher = more uniform distribution)",
    )


# =============================================================================
# Strategy Weighting Settings
# =============================================================================
class StrategyWeightingConfig(BaseModel):
    """Strategy weighting configuration (Meta layer)."""

    model_config = ConfigDict(frozen=True)

    beta: Annotated[float, Field(gt=0.0)] = Field(
        default=2.0,
        description="Softmax temperature for strategy weights",
    )
    w_strategy_max: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        default=0.5,
        description="Maximum weight for a single strategy",
    )
    entropy_min: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.8,
        description="Minimum entropy for diversity (0=concentrated, 1=uniform)",
    )
    penalty_turnover: Annotated[float, Field(ge=0.0)] = Field(
        default=0.1,
        description="Penalty coefficient for turnover",
    )
    penalty_mdd: Annotated[float, Field(ge=0.0)] = Field(
        default=0.2,
        description="Penalty coefficient for max drawdown",
    )
    penalty_instability: Annotated[float, Field(ge=0.0)] = Field(
        default=0.15,
        description="Penalty coefficient for performance instability",
    )


# =============================================================================
# Asset Allocation Settings
# =============================================================================
class AssetAllocationConfig(BaseModel):
    """Asset allocation configuration."""

    model_config = ConfigDict(frozen=True)

    method: AllocationMethod = Field(
        default=AllocationMethod.HRP,
        description="Allocation method to use",
    )
    w_asset_max: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        default=0.2,
        description="Maximum weight for a single asset",
    )
    w_asset_min: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Minimum weight for a single asset",
    )
    delta_max: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        default=0.05,
        description="Maximum weight change per rebalance",
    )
    smooth_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="Smoothing coefficient (0=keep previous, 1=use new)",
    )
    allow_short: bool = Field(
        default=False,
        description="Allow short positions",
    )
    max_cash_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="Maximum cash allocation",
    )


# =============================================================================
# Degradation Mode Settings
# =============================================================================
class DegradationLevelConfig(BaseModel):
    """Configuration for a specific degradation level."""

    model_config = ConfigDict(frozen=True)

    min_adopted_strategies: int | None = Field(default=None)
    max_adopted_strategies: int | None = Field(default=None)
    cash_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.0)
    position_size_multiplier: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=1.0)
    allow_defensive_only: bool = Field(default=False)
    close_all_positions: bool = Field(default=False)


class DegradationConfig(BaseModel):
    """System degradation mode configuration."""

    model_config = ConfigDict(frozen=True)

    level_0: DegradationLevelConfig = Field(
        default_factory=lambda: DegradationLevelConfig(
            min_adopted_strategies=3,
            cash_ratio=0.0,
        )
    )
    level_1: DegradationLevelConfig = Field(
        default_factory=lambda: DegradationLevelConfig(
            min_adopted_strategies=1,
            max_adopted_strategies=2,
            cash_ratio=0.3,
            position_size_multiplier=0.7,
        )
    )
    level_2: DegradationLevelConfig = Field(
        default_factory=lambda: DegradationLevelConfig(
            min_adopted_strategies=0,
            cash_ratio=0.8,
            allow_defensive_only=True,
        )
    )
    level_3: DegradationLevelConfig = Field(
        default_factory=lambda: DegradationLevelConfig(
            cash_ratio=1.0,
            close_all_positions=True,
        )
    )
    vix_threshold: Annotated[float, Field(gt=0.0)] = Field(
        default=30.0,
        description="VIX threshold for volatility-based degradation",
    )
    cooldown_days: Annotated[int, Field(ge=0)] = Field(
        default=5,
        description="Cooldown period before recovery",
    )
    require_manual_approval_for_level_3_recovery: bool = Field(
        default=True,
        description="Require manual approval to recover from level 3",
    )


# =============================================================================
# Data Quality Settings
# =============================================================================
class DataQualityConfig(BaseModel):
    """Data quality check thresholds."""

    model_config = ConfigDict(frozen=True)

    max_missing_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.05,
        description="Maximum allowed missing data rate (0.05 = 5%)",
    )
    max_consecutive_missing: Annotated[int, Field(ge=1)] = Field(
        default=5,
        description="Maximum consecutive missing bars",
    )
    price_change_threshold: Annotated[float, Field(gt=0.0)] = Field(
        default=0.5,
        description="Threshold for detecting price anomalies (50% change)",
    )
    min_volume_threshold: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Minimum volume threshold",
    )
    staleness_hours: Annotated[int, Field(ge=1)] = Field(
        default=24,
        description="Data staleness threshold in hours",
    )
    ohlc_inconsistency_threshold: Annotated[float, Field(ge=0.0)] = Field(
        default=0.01,
        description="OHLC inconsistency threshold (0.01 = 1.0%, EEM 0.54%対応)",
    )


# =============================================================================
# Storage Backend Settings
# =============================================================================
class StorageSettings(BaseModel):
    """Storage backend configuration for S3/local storage."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["local", "s3"] = Field(
        default="local",
        description="Storage backend type: 'local' or 's3'",
    )
    base_path: str = Field(
        default=".cache",
        description="Base path for local storage",
    )
    s3_bucket: str = Field(
        default="",
        description="S3 bucket name (required for s3 backend)",
    )
    s3_prefix: str = Field(
        default=".cache",
        description="S3 prefix/folder path",
    )
    s3_region: str = Field(
        default="ap-northeast-1",
        description="AWS region for S3",
    )
    local_cache_enabled: bool = Field(
        default=True,
        description="Enable local cache for S3 backend",
    )
    local_cache_path: str = Field(
        default="/tmp/.backtest_cache",
        description="Local cache path for S3 backend",
    )
    local_cache_ttl_hours: Annotated[int, Field(ge=1)] = Field(
        default=24,
        description="Local cache TTL in hours",
    )

    def to_storage_config(self) -> "StorageConfig":
        """Convert to StorageConfig dataclass for StorageBackend."""
        from src.utils.storage_backend import StorageConfig

        return StorageConfig(
            backend=self.backend,
            base_path=self.base_path,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            s3_region=self.s3_region,
            local_cache_enabled=self.local_cache_enabled,
            local_cache_path=self.local_cache_path,
            local_cache_ttl_hours=self.local_cache_ttl_hours,
        )


# =============================================================================
# Data Source Settings
# =============================================================================
class DataConfig(BaseModel):
    """Data source configuration."""

    model_config = ConfigDict(frozen=True)

    batch_size: Annotated[int, Field(ge=1)] = Field(
        default=50,
        description="Number of symbols per batch for data fetching",
    )
    parallel_workers: Annotated[int, Field(ge=1)] = Field(
        default=4,
        description="Number of parallel workers for data fetching",
    )
    base_currency: str = Field(
        default="USD",
        description="Base currency for normalization",
    )
    retry_count: Annotated[int, Field(ge=0)] = Field(
        default=3,
        description="Number of retries for failed API calls",
    )
    rate_limit_delay: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Delay between API calls (seconds)",
    )
    use_multi_source: bool = Field(
        default=False,
        description="Use multi-source adapter for data fetching",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable data caching",
    )
    cache_max_age_days: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Maximum age of cached data in days",
    )


# =============================================================================
# Logging Settings
# =============================================================================
class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )
    json_format: bool = Field(
        default=True,
        description="Use JSON format for structured logging",
    )
    include_timestamp: bool = Field(default=True)
    include_run_id: bool = Field(default=True)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


# =============================================================================
# Universe Settings (C4修正: 型統一)
# =============================================================================
class UniverseFilters(BaseModel):
    """Universe filtering rules."""

    model_config = ConfigDict(frozen=True)

    min_history_days: int = Field(default=252, description="Minimum history days required")
    min_avg_volume: int = Field(default=1000000, description="Minimum average volume")
    exclude_delisted: bool = Field(default=True, description="Exclude delisted assets")


class UniverseConfig(BaseModel):
    """Universe configuration."""

    model_config = ConfigDict(frozen=True)

    assets: list[str] = Field(
        default_factory=list,
        description="List of asset symbols in the universe",
    )
    config_file: str | None = Field(
        default="config/universe.yaml",
        description="Path to universe config file",
    )
    enable_us_stocks: bool = Field(default=True, description="Enable US equities")
    enable_japan_stocks: bool = Field(default=True, description="Enable Japanese equities")
    enable_etfs: bool = Field(default=True, description="Enable ETFs")
    enable_commodities: bool = Field(default=True, description="Enable commodities")
    enable_forex: bool = Field(default=True, description="Enable FX pairs")
    enable_crypto: bool = Field(default=False, description="Enable cryptocurrency")
    max_assets: int = Field(default=500, description="Maximum assets to process")
    filters: UniverseFilters = Field(default_factory=UniverseFilters)


# =============================================================================
# Dynamic Parameters Configuration
# =============================================================================
class DynamicParamsComponentsConfig(BaseModel):
    """Component-level toggles for dynamic parameters."""

    model_config = ConfigDict(frozen=True)

    scorer: bool = Field(default=True, description="Dynamic ScorerConfig parameters")
    weighter: bool = Field(default=True, description="Dynamic WeighterConfig parameters")
    allocator: bool = Field(default=True, description="Dynamic AllocatorConfig parameters")
    covariance: bool = Field(default=True, description="Dynamic CovarianceConfig parameters")
    signals: bool = Field(default=True, description="Dynamic signal parameters")


class DynamicParamsFallbackConfig(BaseModel):
    """Fallback behavior when data is insufficient."""

    model_config = ConfigDict(frozen=True)

    use_defaults: bool = Field(default=True, description="Use static defaults if dynamic calc fails")
    min_observations: int = Field(default=60, ge=1, description="Minimum observations for dynamic calc")
    log_fallback: bool = Field(default=True, description="Log when fallback is used")


class DynamicParamsConfig(BaseModel):
    """Dynamic parameters configuration for adaptive behavior."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Master switch for dynamic parameters")
    lookback_days: int = Field(default=252, ge=1, description="Historical data for parameter calculation")
    update_frequency: str = Field(default="monthly", description="Update frequency: daily | weekly | monthly")
    components: DynamicParamsComponentsConfig = Field(default_factory=DynamicParamsComponentsConfig)
    fallback: DynamicParamsFallbackConfig = Field(default_factory=DynamicParamsFallbackConfig)


# =============================================================================
# System Settings (Root)
# =============================================================================
class SystemConfig(BaseModel):
    """System-wide configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        default="multi-asset-portfolio",
        description="System name",
    )
    version: str = Field(
        default="1.0.0",
        description="System version",
    )
    timezone: str = Field(
        default="UTC",
        description="System timezone",
    )
    # NOTE: base_currency は DataConfig.base_currency に統一 (C1修正)
    random_seed: int | None = Field(
        default=42,
        description="Random seed for reproducibility",
    )


# =============================================================================
# Main Settings Class
# =============================================================================
class Settings(BaseSettings):
    """
    Main settings class that combines all configuration sections.

    Configuration can be loaded from:
    1. Environment variables (with PORTFOLIO_ prefix)
    2. YAML/JSON configuration files
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="PORTFOLIO_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    system: SystemConfig = Field(default_factory=SystemConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    hard_gates: HardGatesConfig = Field(default_factory=HardGatesConfig)
    strategy_selection: StrategySelectionConfig = Field(default_factory=StrategySelectionConfig)
    strategy_weighting: StrategyWeightingConfig = Field(default_factory=StrategyWeightingConfig)
    asset_allocation: AssetAllocationConfig = Field(default_factory=AssetAllocationConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    fallback: FallbackMode = Field(default=FallbackMode.HOLD_PREVIOUS)
    universe: UniverseConfig = Field(
        default_factory=UniverseConfig,
        description="Universe configuration (C4修正: 型統一)",
    )
    dynamic_params: DynamicParamsConfig = Field(
        default_factory=DynamicParamsConfig,
        description="Dynamic parameters configuration",
    )
    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description="Storage backend configuration for S3/local",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        """Load settings from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save settings to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_json(self, path: str | Path) -> None:
        """Save settings to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# =============================================================================
# Global settings instance (lazy loading)
# =============================================================================
_settings: Settings | None = None

# Path to default YAML configuration (single source of truth)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


def get_settings() -> Settings:
    """Get the global settings instance.

    By default, loads from config/default.yaml to ensure YAML is the single source of truth.
    """
    global _settings
    if _settings is None:
        _settings = load_settings_from_yaml()
    return _settings


def load_settings_from_yaml(
    path: str | Path | None = None,
    **overrides: Any,
) -> Settings:
    """
    Load settings from YAML file (recommended method).

    YAML is the single source of truth for configuration.
    Pydantic models provide validation and type safety.

    Args:
        path: Path to YAML config file (defaults to config/default.yaml)
        **overrides: Override specific settings

    Returns:
        Settings instance loaded from YAML
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        # Fallback to Pydantic defaults if YAML not found
        import logging
        logging.warning(f"Config file not found: {config_path}, using Pydantic defaults")
        return Settings(**overrides) if overrides else Settings()

    settings = Settings.from_yaml(config_path)

    # Apply overrides if any
    if overrides:
        data = settings.model_dump()
        _deep_update(data, overrides)
        settings = Settings(**data)

    return settings


def _deep_update(base: dict, updates: dict) -> dict:
    """Deep update a dictionary with another dictionary."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_yaml_pydantic_sync(yaml_path: str | Path | None = None) -> list[str]:
    """
    Validate that YAML configuration and Pydantic defaults are in sync.

    This function checks for discrepancies between:
    - Values in YAML (source of truth)
    - Default values in Pydantic models

    Args:
        yaml_path: Path to YAML config file

    Returns:
        List of discrepancy warnings (empty if in sync)
    """
    import yaml

    config_path = Path(yaml_path) if yaml_path else DEFAULT_CONFIG_PATH
    warnings: list[str] = []

    if not config_path.exists():
        warnings.append(f"YAML config not found: {config_path}")
        return warnings

    # Load YAML
    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    # Create Pydantic defaults
    pydantic_defaults = Settings()

    # Check key sections for sync
    sections_to_check = [
        ("data_quality", "data_quality"),
        ("rebalance", "rebalance"),
        ("walk_forward", "walk_forward"),
        ("cost_model", "cost_model"),
    ]

    for yaml_key, pydantic_attr in sections_to_check:
        if yaml_key not in yaml_data:
            warnings.append(f"Section '{yaml_key}' missing from YAML")
            continue

        yaml_section = yaml_data[yaml_key]
        pydantic_section = getattr(pydantic_defaults, pydantic_attr)

        if pydantic_section is None:
            continue

        pydantic_dict = pydantic_section.model_dump()

        for field_name, yaml_value in yaml_section.items():
            if field_name in pydantic_dict:
                pydantic_value = pydantic_dict[field_name]
                # Skip complex nested structures
                if isinstance(yaml_value, dict) or isinstance(pydantic_value, dict):
                    continue
                if yaml_value != pydantic_value:
                    warnings.append(
                        f"Sync mismatch in {yaml_key}.{field_name}: "
                        f"YAML={yaml_value}, Pydantic={pydantic_value}"
                    )

    return warnings
