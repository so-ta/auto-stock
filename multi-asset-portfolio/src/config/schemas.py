"""
Input/Output Schemas - Pydantic v2 Data Schemas

This module defines schemas for all data structures used in the system:
- OHLCV data
- Signal outputs
- Strategy metrics
- Portfolio weights
- Validation reports
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================
class AssetClass(str, Enum):
    """Asset class types."""

    CRYPTO = "crypto"
    STOCK = "stock"
    FX = "fx"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"


class DataQualityStatus(str, Enum):
    """Data quality check status."""

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"
    EXCLUDED = "excluded"


class SignalDirection(str, Enum):
    """Signal direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class StrategyStatus(str, Enum):
    """Strategy adoption status."""

    ADOPTED = "adopted"
    REJECTED = "rejected"
    WARNING = "warning"
    PENDING = "pending"


class MarketRegime(str, Enum):
    """Market regime classification."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# =============================================================================
# Base Schemas
# =============================================================================
class TimestampedModel(BaseModel):
    """Base model with timestamp."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        description="Timestamp of the record",
    )


class IdentifiedModel(BaseModel):
    """Base model with identifier."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        description="Unique identifier",
    )


# =============================================================================
# OHLCV Data Schemas
# =============================================================================
class OHLCVBar(TimestampedModel):
    """Single OHLCV bar."""

    open: float = Field(description="Open price")
    high: float = Field(description="High price")
    low: float = Field(description="Low price")
    close: float = Field(description="Close price")
    volume: Annotated[float, Field(ge=0.0)] = Field(description="Volume")
    adjusted_close: float | None = Field(
        default=None,
        description="Adjusted close price (for dividends/splits)",
    )

    @model_validator(mode="after")
    def validate_ohlc(self) -> "OHLCVBar":
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) must be >= Low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Open ({self.open}) must be between Low and High")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Close ({self.close}) must be between Low and High")
        return self


class OHLCVData(BaseModel):
    """OHLCV data for an asset."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Asset symbol (e.g., 'BTCUSD', 'AAPL')")
    asset_class: AssetClass = Field(description="Asset class")
    currency: str = Field(default="USD", description="Quote currency")
    timezone: str = Field(default="UTC", description="Data timezone")
    bars: list[OHLCVBar] = Field(description="List of OHLCV bars")

    @property
    def start_date(self) -> datetime | None:
        return self.bars[0].timestamp if self.bars else None

    @property
    def end_date(self) -> datetime | None:
        return self.bars[-1].timestamp if self.bars else None

    @property
    def bar_count(self) -> int:
        return len(self.bars)


class AssetMetadata(BaseModel):
    """Asset metadata."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Asset symbol")
    name: str = Field(description="Asset name")
    asset_class: AssetClass = Field(description="Asset class")
    exchange: str | None = Field(default=None, description="Exchange")
    currency: str = Field(default="USD", description="Quote currency")
    market_cap: float | None = Field(default=None, description="Market cap")
    sector: str | None = Field(default=None, description="Sector")
    is_tradable: bool = Field(default=True, description="Is currently tradable")


# =============================================================================
# Data Quality Schemas
# =============================================================================
class QualityCheckResult(BaseModel):
    """Result of a single quality check."""

    model_config = ConfigDict(frozen=True)

    check_name: str = Field(description="Name of the quality check")
    status: DataQualityStatus = Field(description="Check status")
    value: float | None = Field(default=None, description="Metric value")
    threshold: float | None = Field(default=None, description="Threshold value")
    message: str | None = Field(default=None, description="Additional message")


class DataQualityReport(TimestampedModel):
    """Complete data quality report for an asset."""

    symbol: str = Field(description="Asset symbol")
    overall_status: DataQualityStatus = Field(description="Overall quality status")
    checks: list[QualityCheckResult] = Field(description="Individual check results")
    missing_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Missing data rate"
    )
    anomaly_count: Annotated[int, Field(ge=0)] = Field(
        description="Number of detected anomalies"
    )
    duplicate_count: Annotated[int, Field(ge=0)] = Field(
        description="Number of duplicate bars"
    )
    is_excluded: bool = Field(
        default=False,
        description="Whether asset is excluded from evaluation",
    )
    exclusion_reason: str | None = Field(
        default=None,
        description="Reason for exclusion",
    )


# =============================================================================
# Signal Schemas
# =============================================================================
class SignalValue(TimestampedModel):
    """Single signal value."""

    value: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        description="Signal value normalized to [-1, 1]"
    )
    raw_value: float | None = Field(
        default=None,
        description="Raw signal value before normalization",
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Signal confidence (0-1)",
    )


class SignalOutput(BaseModel):
    """Signal output for an asset."""

    model_config = ConfigDict(frozen=True)

    signal_name: str = Field(description="Signal/strategy name")
    symbol: str = Field(description="Asset symbol")
    values: list[SignalValue] = Field(description="Signal values over time")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Signal parameters used",
    )
    generated_at: datetime = Field(description="Generation timestamp")

    @property
    def latest_value(self) -> float | None:
        return self.values[-1].value if self.values else None

    @property
    def direction(self) -> SignalDirection:
        if not self.values:
            return SignalDirection.NEUTRAL
        v = self.values[-1].value
        if v > 0.1:
            return SignalDirection.LONG
        elif v < -0.1:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL


# =============================================================================
# Strategy Metrics Schemas
# =============================================================================
class PerformanceMetrics(BaseModel):
    """Performance metrics for a strategy."""

    model_config = ConfigDict(frozen=True)

    # Core metrics
    total_return: float = Field(description="Total return")
    annualized_return: float = Field(description="Annualized return")
    volatility: Annotated[float, Field(ge=0.0)] = Field(
        description="Annualized volatility"
    )
    sharpe_ratio: float = Field(description="Sharpe ratio")
    sortino_ratio: float = Field(description="Sortino ratio")

    # Drawdown metrics
    max_drawdown: Annotated[float, Field(le=0.0)] = Field(
        description="Maximum drawdown (negative value)"
    )
    calmar_ratio: float | None = Field(
        default=None,
        description="Calmar ratio (return / max DD)",
    )

    # Trade metrics
    win_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(description="Win rate")
    profit_factor: Annotated[float, Field(ge=0.0)] = Field(description="Profit factor")
    trade_count: Annotated[int, Field(ge=0)] = Field(description="Total trade count")
    avg_trade_return: float = Field(description="Average return per trade")

    # Risk metrics
    var_95: float | None = Field(default=None, description="95% Value at Risk")
    es_95: float | None = Field(default=None, description="95% Expected Shortfall")
    turnover: Annotated[float, Field(ge=0.0)] | None = Field(
        default=None,
        description="Average turnover rate",
    )

    # Cost-adjusted metrics
    return_after_costs: float | None = Field(
        default=None,
        description="Return after transaction costs",
    )
    sharpe_after_costs: float | None = Field(
        default=None,
        description="Sharpe ratio after costs",
    )


class StrategyEvaluation(BaseModel):
    """Complete strategy evaluation result."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str = Field(description="Strategy name")
    symbol: str = Field(description="Asset symbol")
    evaluation_period: str = Field(description="Evaluation period description")
    train_start: datetime = Field(description="Training start date")
    train_end: datetime = Field(description="Training end date")
    test_start: datetime = Field(description="Test start date")
    test_end: datetime = Field(description="Test end date")

    # Metrics
    train_metrics: PerformanceMetrics = Field(description="Training period metrics")
    test_metrics: PerformanceMetrics = Field(description="Test period metrics")

    # Status
    status: StrategyStatus = Field(description="Adoption status")
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for rejection (if rejected)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages",
    )

    # Score
    score: float | None = Field(
        default=None,
        description="Final score for weighting",
    )
    weight: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Assigned weight within asset",
    )


# =============================================================================
# Portfolio Weight Schemas
# =============================================================================
class AssetWeight(BaseModel):
    """Weight for a single asset."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Asset symbol")
    weight: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        description="Portfolio weight"
    )
    previous_weight: float | None = Field(
        default=None,
        description="Previous period weight",
    )
    weight_change: float | None = Field(
        default=None,
        description="Weight change from previous",
    )
    expected_return: float | None = Field(
        default=None,
        description="Expected return estimate",
    )
    risk_contribution: float | None = Field(
        default=None,
        description="Contribution to portfolio risk",
    )


class PortfolioWeights(TimestampedModel):
    """Portfolio weights output."""

    rebalance_date: datetime = Field(description="Rebalance execution date")
    weights: list[AssetWeight] = Field(description="Asset weights")
    cash_weight: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Cash allocation weight",
    )
    total_weight: float = Field(description="Sum of all weights (should be 1.0)")
    regime: MarketRegime = Field(
        default=MarketRegime.NEUTRAL,
        description="Detected market regime",
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Confidence in weight estimates",
    )
    degradation_level: int = Field(
        default=0,
        description="Current degradation level (0-3)",
    )

    @model_validator(mode="after")
    def validate_total_weight(self) -> "PortfolioWeights":
        total = sum(w.weight for w in self.weights) + self.cash_weight
        if abs(total - self.total_weight) > 0.001:
            raise ValueError(
                f"Weight sum ({total}) doesn't match total_weight ({self.total_weight})"
            )
        return self

    def to_dict(self) -> dict[str, float]:
        """Convert to simple symbol -> weight dict."""
        result = {w.symbol: w.weight for w in self.weights}
        if self.cash_weight > 0:
            result["CASH"] = self.cash_weight
        return result


# =============================================================================
# Validation Report Schemas
# =============================================================================
class ValidationReport(TimestampedModel):
    """Complete validation report."""

    run_id: str = Field(description="Unique run identifier")
    config_version: str = Field(description="Config version used")

    # Data quality
    data_quality_results: list[DataQualityReport] = Field(
        description="Data quality results per asset"
    )
    excluded_assets: list[str] = Field(
        default_factory=list,
        description="Assets excluded due to quality issues",
    )

    # Strategy evaluation
    strategy_evaluations: list[StrategyEvaluation] = Field(
        description="Strategy evaluation results"
    )
    adopted_strategies: int = Field(description="Number of adopted strategies")
    rejected_strategies: int = Field(description="Number of rejected strategies")

    # Portfolio
    portfolio_weights: PortfolioWeights = Field(description="Final portfolio weights")
    portfolio_risk: dict[str, float] = Field(
        default_factory=dict,
        description="Portfolio risk metrics (vol, var, etc.)",
    )

    # Status
    fallback_mode: str | None = Field(
        default=None,
        description="Fallback mode if activated",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages (non-fatal)",
    )


# =============================================================================
# API Response Schemas
# =============================================================================
class RebalanceOutput(BaseModel):
    """Output format for rebalance operation."""

    model_config = ConfigDict(frozen=True)

    as_of: datetime = Field(description="Timestamp of output")
    weights: dict[str, float] = Field(description="Symbol -> weight mapping")
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic information",
    )
    validation_report_path: str | None = Field(
        default=None,
        description="Path to full validation report",
    )

    @classmethod
    def from_portfolio_weights(
        cls,
        weights: PortfolioWeights,
        diagnostics: dict[str, Any] | None = None,
    ) -> "RebalanceOutput":
        """Create from PortfolioWeights."""
        return cls(
            as_of=weights.timestamp,
            weights=weights.to_dict(),
            diagnostics=diagnostics or {},
        )


class HealthCheckResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="Health status (ok, warning, error)")
    timestamp: datetime = Field(description="Check timestamp")
    version: str = Field(description="System version")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Component status",
    )
    last_rebalance: datetime | None = Field(
        default=None,
        description="Last rebalance timestamp",
    )
    degradation_level: int = Field(
        default=0,
        description="Current degradation level",
    )
