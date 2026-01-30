"""
Anomaly Detection Module - Production Monitoring

This module detects anomalies that trigger fallback/degradation modes:
- Data source failures (missing rate threshold exceeded)
- VaR deterioration (portfolio expected loss threshold exceeded)
- Correlation surge (multi-asset crash signal)
- Backtest assumption violations

From §9:
- Major data source failure, missing rate exceeds threshold
- Recent portfolio expected loss exceeds threshold (e.g., VaR deterioration)
- Correlation surges, diversification effect disappears (multi-asset crash signal)
- Suspected collapse of backtest assumptions (cost, liquidity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import polars as pl

    from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    DATA_SOURCE_FAILURE = "data_source_failure"
    HIGH_MISSING_RATE = "high_missing_rate"
    VAR_DETERIORATION = "var_deterioration"
    CORRELATION_SURGE = "correlation_surge"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    COST_ASSUMPTION_VIOLATION = "cost_assumption_violation"
    PRICE_SPIKE = "price_spike"
    VOLUME_COLLAPSE = "volume_collapse"
    SYSTEM_ERROR = "system_error"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Monitor closely
    CRITICAL = "critical"  # Trigger fallback mode


@dataclass
class Anomaly:
    """Detected anomaly."""

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_assets: list[str] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "details": self.details,
            "affected_assets": self.affected_assets,
            "recommended_action": self.recommended_action,
        }


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""

    anomalies: list[Anomaly] = field(default_factory=list)
    should_trigger_fallback: bool = False
    fallback_reason: str | None = None

    @property
    def has_critical(self) -> bool:
        """Check if any critical anomalies were detected."""
        return any(a.severity == AnomalySeverity.CRITICAL for a in self.anomalies)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were detected."""
        return any(a.severity == AnomalySeverity.WARNING for a in self.anomalies)


class AnomalyDetector:
    """
    Detects anomalies in data and portfolio state.

    Monitors for conditions that should trigger risk-off/fallback modes:
    - Data quality issues
    - Risk metric deterioration
    - Market stress indicators

    Example:
        >>> detector = AnomalyDetector(settings)
        >>> result = detector.check_all(
        ...     quality_reports=quality_reports,
        ...     portfolio_risk=risk_metrics,
        ...     correlation_matrix=corr_matrix,
        ... )
        >>> if result.should_trigger_fallback:
        ...     fallback_handler.activate(result.fallback_reason)
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        """
        Initialize anomaly detector.

        Args:
            settings: Optional settings instance
        """
        self._settings = settings
        self._logger = logger.bind(component="anomaly_detector")

        # Thresholds (can be overridden via settings)
        self._missing_rate_threshold = 0.1  # 10%
        self._var_threshold = 0.05  # 5% daily VaR
        self._correlation_threshold = 0.8  # Average correlation
        self._volume_collapse_threshold = 0.3  # 30% of average

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        if self._settings is None:
            from src.config.settings import get_settings

            self._settings = get_settings()
        return self._settings

    def check_all(
        self,
        quality_reports: dict[str, Any] | None = None,
        portfolio_risk: dict[str, float] | None = None,
        correlation_matrix: Any | None = None,
        price_data: dict[str, "pl.DataFrame"] | None = None,
        volume_data: dict[str, "pl.DataFrame"] | None = None,
    ) -> AnomalyDetectionResult:
        """
        Run all anomaly checks.

        Args:
            quality_reports: Data quality reports per asset
            portfolio_risk: Portfolio risk metrics (var_95, vol, etc.)
            correlation_matrix: Asset correlation matrix
            price_data: Recent price data per asset
            volume_data: Recent volume data per asset

        Returns:
            AnomalyDetectionResult with all detected anomalies
        """
        anomalies: list[Anomaly] = []

        # Check data quality
        if quality_reports:
            anomalies.extend(self._check_data_quality(quality_reports))

        # Check portfolio risk
        if portfolio_risk:
            anomalies.extend(self._check_portfolio_risk(portfolio_risk))

        # Check correlation
        if correlation_matrix is not None:
            anomalies.extend(self._check_correlation(correlation_matrix))

        # Check price anomalies
        if price_data:
            anomalies.extend(self._check_price_anomalies(price_data))

        # Check volume
        if volume_data:
            anomalies.extend(self._check_volume_anomalies(volume_data))

        # Determine if fallback should be triggered
        should_fallback = any(a.severity == AnomalySeverity.CRITICAL for a in anomalies)
        fallback_reason = None
        if should_fallback:
            critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
            fallback_reason = "; ".join(a.description for a in critical_anomalies)

        result = AnomalyDetectionResult(
            anomalies=anomalies,
            should_trigger_fallback=should_fallback,
            fallback_reason=fallback_reason,
        )

        # Log result
        self._log_result(result)

        return result

    def _check_data_quality(
        self,
        quality_reports: dict[str, Any],
    ) -> list[Anomaly]:
        """Check for data quality anomalies."""
        anomalies: list[Anomaly] = []
        excluded_count = 0
        high_missing_assets: list[str] = []

        for symbol, report in quality_reports.items():
            # Check if asset is excluded
            if isinstance(report, dict):
                is_excluded = report.get("is_excluded", False)
                missing_rate = report.get("missing_rate", 0)
            else:
                is_excluded = getattr(report, "is_excluded", False)
                missing_rate = getattr(report, "missing_rate", 0)

            if is_excluded:
                excluded_count += 1

            if missing_rate > self._missing_rate_threshold:
                high_missing_assets.append(symbol)

        # Too many excluded assets
        total_assets = len(quality_reports)
        excluded_ratio = excluded_count / total_assets if total_assets > 0 else 0

        if excluded_ratio > 0.3:  # More than 30% excluded
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.DATA_SOURCE_FAILURE,
                    severity=AnomalySeverity.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    description=f"High asset exclusion rate: {excluded_ratio:.1%} ({excluded_count}/{total_assets})",
                    details={
                        "excluded_count": excluded_count,
                        "total_assets": total_assets,
                        "excluded_ratio": excluded_ratio,
                    },
                    recommended_action="Activate fallback mode, investigate data sources",
                )
            )
        elif excluded_ratio > 0.1:  # More than 10% excluded
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.DATA_SOURCE_FAILURE,
                    severity=AnomalySeverity.WARNING,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Elevated asset exclusion rate: {excluded_ratio:.1%}",
                    details={
                        "excluded_count": excluded_count,
                        "total_assets": total_assets,
                    },
                    recommended_action="Monitor data sources closely",
                )
            )

        # High missing rate
        if high_missing_assets:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.HIGH_MISSING_RATE,
                    severity=AnomalySeverity.WARNING,
                    timestamp=datetime.now(timezone.utc),
                    description=f"High missing rate detected in {len(high_missing_assets)} assets",
                    details={"threshold": self._missing_rate_threshold},
                    affected_assets=high_missing_assets,
                    recommended_action="Review data quality, consider excluding affected assets",
                )
            )

        return anomalies

    def _check_portfolio_risk(
        self,
        portfolio_risk: dict[str, float],
    ) -> list[Anomaly]:
        """Check for portfolio risk anomalies."""
        anomalies: list[Anomaly] = []

        # Check VaR
        var_95 = portfolio_risk.get("var_95", 0)
        if var_95 > self._var_threshold:
            severity = (
                AnomalySeverity.CRITICAL
                if var_95 > self._var_threshold * 1.5
                else AnomalySeverity.WARNING
            )
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.VAR_DETERIORATION,
                    severity=severity,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Portfolio VaR ({var_95:.2%}) exceeds threshold ({self._var_threshold:.2%})",
                    details={
                        "var_95": var_95,
                        "threshold": self._var_threshold,
                        "excess": var_95 - self._var_threshold,
                    },
                    recommended_action="Reduce position sizes or activate cash evacuation",
                )
            )

        # Check volatility
        vol = portfolio_risk.get("volatility", 0)
        vol_threshold = 0.3  # 30% annualized
        if vol > vol_threshold:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.VAR_DETERIORATION,
                    severity=AnomalySeverity.WARNING,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Portfolio volatility ({vol:.2%}) is elevated",
                    details={"volatility": vol, "threshold": vol_threshold},
                    recommended_action="Consider reducing exposure",
                )
            )

        return anomalies

    def _check_correlation(
        self,
        correlation_matrix: Any,
    ) -> list[Anomaly]:
        """Check for correlation anomalies (multi-asset crash signal)."""
        anomalies: list[Anomaly] = []

        try:
            import numpy as np

            # Convert to numpy if needed
            if hasattr(correlation_matrix, "to_numpy"):
                corr = correlation_matrix.to_numpy()
            else:
                corr = np.array(correlation_matrix)

            # Calculate average off-diagonal correlation
            n = corr.shape[0]
            if n > 1:
                mask = ~np.eye(n, dtype=bool)
                avg_corr = np.abs(corr[mask]).mean()

                if avg_corr > self._correlation_threshold:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.CORRELATION_SURGE,
                            severity=AnomalySeverity.CRITICAL,
                            timestamp=datetime.now(timezone.utc),
                            description=f"Correlation surge detected: average correlation = {avg_corr:.2f}",
                            details={
                                "average_correlation": float(avg_corr),
                                "threshold": self._correlation_threshold,
                            },
                            recommended_action="Multi-asset crash risk, activate fallback mode",
                        )
                    )
                elif avg_corr > self._correlation_threshold * 0.8:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.CORRELATION_SURGE,
                            severity=AnomalySeverity.WARNING,
                            timestamp=datetime.now(timezone.utc),
                            description=f"Elevated correlation: average = {avg_corr:.2f}",
                            details={"average_correlation": float(avg_corr)},
                            recommended_action="Monitor correlation levels",
                        )
                    )

        except Exception as e:
            self._logger.warning("Failed to check correlation", error=str(e))

        return anomalies

    def _check_price_anomalies(
        self,
        price_data: dict[str, "pl.DataFrame"],
    ) -> list[Anomaly]:
        """Check for price anomalies (spikes)."""
        anomalies: list[Anomaly] = []
        spike_assets: list[str] = []

        for symbol, df in price_data.items():
            try:
                if len(df) < 2:
                    continue

                # Calculate returns
                closes = df["close"].to_numpy()
                returns = closes[1:] / closes[:-1] - 1

                # Check for extreme returns (> 20% in one bar)
                max_return = abs(returns).max()
                if max_return > 0.2:
                    spike_assets.append(symbol)

            except Exception:
                continue

        if spike_assets:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    severity=AnomalySeverity.WARNING,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Price spikes detected in {len(spike_assets)} assets",
                    affected_assets=spike_assets,
                    recommended_action="Verify data quality, consider excluding affected assets",
                )
            )

        return anomalies

    def _check_volume_anomalies(
        self,
        volume_data: dict[str, "pl.DataFrame"],
    ) -> list[Anomaly]:
        """Check for volume anomalies (collapse)."""
        anomalies: list[Anomaly] = []
        low_volume_assets: list[str] = []

        for symbol, df in volume_data.items():
            try:
                if len(df) < 20:
                    continue

                volumes = df["volume"].to_numpy()
                avg_volume = volumes[:-1].mean()
                recent_volume = volumes[-1]

                if avg_volume > 0 and recent_volume < avg_volume * self._volume_collapse_threshold:
                    low_volume_assets.append(symbol)

            except Exception:
                continue

        if len(low_volume_assets) > len(volume_data) * 0.3:  # More than 30% with low volume
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.VOLUME_COLLAPSE,
                    severity=AnomalySeverity.WARNING,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Volume collapse in {len(low_volume_assets)} assets",
                    affected_assets=low_volume_assets,
                    recommended_action="Check for market holidays or liquidity issues",
                )
            )

        return anomalies

    def _log_result(self, result: AnomalyDetectionResult) -> None:
        """Log anomaly detection result."""
        if result.should_trigger_fallback:
            self._logger.warning(
                "Fallback triggered",
                reason=result.fallback_reason,
                anomaly_count=len(result.anomalies),
            )
        elif result.has_warnings:
            self._logger.info(
                "Anomalies detected",
                warning_count=sum(1 for a in result.anomalies if a.severity == AnomalySeverity.WARNING),
            )
        else:
            self._logger.debug("No anomalies detected")
