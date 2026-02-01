"""
Output Handlers Module - Output Generation and Logging

This module handles output generation and logging for the pipeline:
- Output file generation (weights, diagnostics)
- Run info logging for reproducibility
- Diagnostics creation
- Error result creation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.orchestrator.fallback import FallbackState

from src.utils.reproducibility import get_run_info, save_run_info

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    output_dir: Path
    log_dir: Path
    lightweight_mode: bool = False
    skip_diagnostics: bool = False
    seed: int = 42


@dataclass
class DiagnosticsData:
    """Data for creating diagnostics."""
    excluded_assets: set
    quality_reports: Dict[str, Any]
    evaluations: List[Any]
    risk_metrics: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    fetch_summary: Optional[Dict[str, Any]] = None
    quality_summary: Optional[Dict[str, Any]] = None


class OutputHandler:
    """
    Handles output generation and logging for the pipeline.

    This class encapsulates the output generation and logging logic
    that was previously in the Pipeline class.
    """

    def __init__(
        self,
        config: OutputConfig,
        settings: "Settings",
    ):
        """
        Initialize the output handler.

        Args:
            config: Output configuration
            settings: Application settings
        """
        self._config = config
        self._settings = settings
        self._logger = logger

    def generate_output(
        self,
        run_id: str,
        final_weights: Dict[str, float],
        diagnostics_data: DiagnosticsData,
        fallback_state: Optional["FallbackState"] = None,
    ) -> Path:
        """
        Generate output files (weights + diagnostics).

        Args:
            run_id: Unique run identifier
            final_weights: Final portfolio weights
            diagnostics_data: Data for creating diagnostics
            fallback_state: Current fallback state (if any)

        Returns:
            Path to the generated output file
        """
        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create diagnostics
        diagnostics = self.create_diagnostics(diagnostics_data, fallback_state)

        # Build output
        output = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "weights": final_weights,
            "diagnostics": diagnostics,
        }

        # Save to file
        output_path = output_dir / f"weights_{run_id}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        self._logger.info(f"Output generated: {output_path}")
        return output_path

    def save_run_logs(
        self,
        run_id: str,
    ) -> Path:
        """
        Save run info and logs for reproducibility.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to the saved run info file
        """
        log_dir = self._config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get run info
        config_dict = {}
        if hasattr(self._settings, "model_dump"):
            config_dict = self._settings.model_dump()
        elif hasattr(self._settings, "dict"):
            config_dict = self._settings.dict()

        run_info = get_run_info(
            run_id=run_id,
            seed=self._config.seed,
            config=config_dict,
        )

        # Save run info
        run_info_path = log_dir / f"run_info_{run_id}.json"
        save_run_info(run_info, run_info_path)

        self._logger.info(f"Logging completed: run_id={run_id}")
        return run_info_path

    def create_diagnostics(
        self,
        data: DiagnosticsData,
        fallback_state: Optional["FallbackState"] = None,
    ) -> Dict[str, Any]:
        """
        Create diagnostics dictionary.

        In lightweight mode, returns minimal diagnostics for performance.

        Args:
            data: Diagnostics data
            fallback_state: Current fallback state (if any)

        Returns:
            Diagnostics dictionary
        """
        # Lightweight mode: minimal diagnostics for backtest performance
        if self._config.lightweight_mode or self._config.skip_diagnostics:
            return {
                "lightweight_mode": True,
                "fallback_mode": fallback_state.mode.value if fallback_state else None,
                "errors": data.errors if data.errors else [],
            }

        # Full diagnostics for production mode
        diagnostics = {
            "excluded_assets": list(data.excluded_assets) if data.excluded_assets else [],
            "quality_reports_count": len(data.quality_reports) if data.quality_reports else 0,
            "strategies_evaluated": len(data.evaluations) if data.evaluations else 0,
            "risk_metrics": data.risk_metrics,
            "fallback_mode": fallback_state.mode.value if fallback_state else None,
            "warnings": data.warnings,
            "errors": data.errors,
        }

        # Add expanded mode summaries if available
        if data.fetch_summary:
            diagnostics["fetch_summary"] = {
                k: len(v) if isinstance(v, (list, dict)) else v
                for k, v in data.fetch_summary.items()
            }

        if data.quality_summary:
            diagnostics["quality_summary"] = {
                k: len(v) if isinstance(v, (list, dict)) else v
                for k, v in data.quality_summary.items()
            }

        return diagnostics

    def create_error_result(
        self,
        run_id: str,
        start_time: datetime,
        error: str,
        step_results: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Create error result dictionary.

        Args:
            run_id: Unique run identifier
            start_time: Pipeline start time
            error: Error message
            step_results: Results from completed steps
            errors: List of errors
            warnings: List of warnings

        Returns:
            Error result dictionary
        """
        end_time = datetime.now(timezone.utc)

        return {
            "run_id": run_id,
            "status": "FAILED",
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds() if start_time else 0,
            "weights": {},
            "diagnostics": {"error": error},
            "fallback_state": None,
            "step_results": step_results,
            "errors": [error] + errors,
            "warnings": warnings,
        }


def generate_run_output(
    run_id: str,
    output_dir: Path,
    weights: Dict[str, float],
    diagnostics: Dict[str, Any],
) -> Path:
    """
    Convenience function to generate run output.

    Args:
        run_id: Unique run identifier
        output_dir: Output directory path
        weights: Portfolio weights
        diagnostics: Diagnostics dictionary

    Returns:
        Path to the generated output file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "weights": weights,
        "diagnostics": diagnostics,
    }

    output_path = output_dir / f"weights_{run_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path
