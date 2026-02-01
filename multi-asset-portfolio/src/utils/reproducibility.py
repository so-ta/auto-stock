"""
Reproducibility Management Module

This module ensures reproducibility of all computations:
- Seed management for numpy, random, and torch
- Version tracking for code and configuration
- Run information recording

Requirement (from §10):
- Same input data + same config + same seed → same output weights
- Config (parameter ranges, thresholds) must be fixed in config files
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import structlog

from src.utils.hash_utils import compute_config_hash as _compute_config_hash

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RunInfo:
    """Information about a pipeline run for reproducibility."""

    run_id: str
    timestamp: datetime
    seed: int
    python_version: str
    platform: str
    git_commit: str | None
    git_branch: str | None
    config_hash: str
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "seed": self.seed,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "config_hash": self.config_hash,
            "package_versions": self.package_versions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunInfo":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            seed=data["seed"],
            python_version=data["python_version"],
            platform=data["platform"],
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
            config_hash=data["config_hash"],
            package_versions=data.get("package_versions", {}),
        )


class SeedManager:
    """
    Manages random seeds for reproducibility.

    Provides a context manager that sets seeds for all random number
    generators and restores them after the context exits.

    Example:
        >>> with SeedManager(seed=42) as sm:
        ...     # All random operations here use seed 42
        ...     data = generate_random_data()
        >>> # Seeds restored to original state

    Supports:
        - Python's random module
        - NumPy's random generator
        - PyTorch (if available)
    """

    def __init__(
        self,
        seed: int = 42,
        set_numpy: bool = True,
        set_random: bool = True,
        set_torch: bool = True,
        deterministic_torch: bool = True,
    ) -> None:
        """
        Initialize seed manager.

        Args:
            seed: Base seed value
            set_numpy: Whether to set numpy seed
            set_random: Whether to set random module seed
            set_torch: Whether to set torch seed (if available)
            deterministic_torch: Whether to enable torch deterministic mode
        """
        self.seed = seed
        self.set_numpy = set_numpy
        self.set_random = set_random
        self.set_torch = set_torch
        self.deterministic_torch = deterministic_torch

        # Store original states for restoration
        self._original_random_state: tuple[Any, ...] | None = None
        self._original_numpy_state: dict[str, Any] | None = None
        self._original_torch_state: Any = None
        self._original_cuda_state: Any = None

    def __enter__(self) -> "SeedManager":
        """Enter context and set all seeds."""
        self._set_seeds()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original states."""
        self._restore_states()

    def _set_seeds(self) -> None:
        """Set seeds for all random number generators."""
        # Python random
        if self.set_random:
            self._original_random_state = random.getstate()
            random.seed(self.seed)
            logger.debug("Set random seed", seed=self.seed)

        # NumPy
        if self.set_numpy:
            try:
                import numpy as np

                self._original_numpy_state = np.random.get_state()
                np.random.seed(self.seed)
                logger.debug("Set numpy seed", seed=self.seed)
            except ImportError:
                pass

        # PyTorch
        if self.set_torch:
            try:
                import torch

                self._original_torch_state = torch.get_rng_state()
                torch.manual_seed(self.seed)

                if torch.cuda.is_available():
                    self._original_cuda_state = torch.cuda.get_rng_state_all()
                    torch.cuda.manual_seed_all(self.seed)

                if self.deterministic_torch:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                logger.debug("Set torch seed", seed=self.seed)
            except ImportError:
                pass

    def _restore_states(self) -> None:
        """Restore original random states."""
        # Python random
        if self._original_random_state is not None:
            random.setstate(self._original_random_state)

        # NumPy
        if self._original_numpy_state is not None:
            try:
                import numpy as np

                np.random.set_state(self._original_numpy_state)
            except ImportError:
                pass

        # PyTorch
        if self._original_torch_state is not None:
            try:
                import torch

                torch.set_rng_state(self._original_torch_state)

                if self._original_cuda_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(self._original_cuda_state)
            except ImportError:
                pass

    @staticmethod
    def get_derived_seed(base_seed: int, component: str) -> int:
        """
        Generate a derived seed for a specific component.

        This allows different parts of the pipeline to have
        different but reproducible seeds.

        Args:
            base_seed: Base seed value
            component: Component name (e.g., "signal_momentum")

        Returns:
            Derived seed value
        """
        hash_input = f"{base_seed}_{component}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_value[:8], 16)


@contextmanager
def reproducible_context(
    seed: int = 42,
    components: list[str] | None = None,
) -> Iterator[SeedManager]:
    """
    Context manager for reproducible computations.

    Args:
        seed: Base seed value
        components: Optional list of components to set seeds for

    Yields:
        SeedManager instance
    """
    manager = SeedManager(seed=seed)
    try:
        manager._set_seeds()
        yield manager
    finally:
        manager._restore_states()


def get_git_info() -> tuple[str | None, str | None]:
    """
    Get current git commit hash and branch name.

    Returns:
        Tuple of (commit_hash, branch_name)
    """
    commit_hash = None
    branch_name = None

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch_name = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return commit_hash, branch_name


def get_package_versions() -> dict[str, str]:
    """
    Get versions of key packages.

    Returns:
        Dict mapping package name to version
    """
    packages = [
        "polars",
        "numpy",
        "pandas",
        "pydantic",
        "structlog",
        "scipy",
        "scikit-learn",
    ]

    versions = {}
    for package in packages:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass

    return versions


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for reproducibility tracking.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hash of the configuration (16 characters)
    """
    return _compute_config_hash(config)


def get_run_info(
    run_id: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> RunInfo:
    """
    Generate complete run information for reproducibility.

    Args:
        run_id: Unique run identifier
        seed: Random seed used
        config: Configuration dictionary

    Returns:
        RunInfo instance with all metadata
    """
    git_commit, git_branch = get_git_info()
    config_hash = compute_config_hash(config or {})
    package_versions = get_package_versions()

    return RunInfo(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        seed=seed,
        python_version=sys.version,
        platform=platform.platform(),
        git_commit=git_commit,
        git_branch=git_branch,
        config_hash=config_hash,
        package_versions=package_versions,
    )


def save_run_info(run_info: RunInfo, path: Path | str) -> None:
    """
    Save run information to a JSON file.

    Args:
        run_info: RunInfo instance
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_info.to_dict(), f, indent=2, ensure_ascii=False)


def load_run_info(path: Path | str) -> RunInfo:
    """
    Load run information from a JSON file.

    Args:
        path: Input file path

    Returns:
        RunInfo instance
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunInfo.from_dict(data)


def verify_reproducibility(
    run_info_1: RunInfo,
    run_info_2: RunInfo,
) -> tuple[bool, list[str]]:
    """
    Verify if two runs should produce identical results.

    Args:
        run_info_1: First run info
        run_info_2: Second run info

    Returns:
        Tuple of (is_reproducible, list_of_differences)
    """
    differences = []

    if run_info_1.seed != run_info_2.seed:
        differences.append(f"Seed differs: {run_info_1.seed} vs {run_info_2.seed}")

    if run_info_1.config_hash != run_info_2.config_hash:
        differences.append(
            f"Config hash differs: {run_info_1.config_hash} vs {run_info_2.config_hash}"
        )

    if run_info_1.git_commit != run_info_2.git_commit:
        differences.append(
            f"Git commit differs: {run_info_1.git_commit} vs {run_info_2.git_commit}"
        )

    # Check critical package versions
    critical_packages = ["polars", "numpy", "scipy"]
    for pkg in critical_packages:
        v1 = run_info_1.package_versions.get(pkg)
        v2 = run_info_2.package_versions.get(pkg)
        if v1 != v2:
            differences.append(f"Package {pkg} version differs: {v1} vs {v2}")

    is_reproducible = len(differences) == 0
    return is_reproducible, differences


class ReproducibilityChecker:
    """
    Checker for verifying reproducibility requirements.

    Example:
        >>> checker = ReproducibilityChecker()
        >>> checker.record_output("weights", {"BTCUSD": 0.3, "ETH": 0.7})
        >>> # Later, verify same output
        >>> checker.verify_output("weights", {"BTCUSD": 0.3, "ETH": 0.7})
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        """
        Initialize checker.

        Args:
            tolerance: Numerical tolerance for float comparisons
        """
        self.tolerance = tolerance
        self._recorded_outputs: dict[str, Any] = {}

    def record_output(self, name: str, value: Any) -> None:
        """Record an output for later verification."""
        self._recorded_outputs[name] = self._serialize(value)

    def verify_output(self, name: str, value: Any) -> tuple[bool, str]:
        """
        Verify an output matches recorded value.

        Returns:
            Tuple of (matches, difference_message)
        """
        if name not in self._recorded_outputs:
            return False, f"No recorded output for '{name}'"

        recorded = self._recorded_outputs[name]
        current = self._serialize(value)

        if recorded == current:
            return True, ""

        return False, f"Output '{name}' differs from recorded value"

    def _serialize(self, value: Any) -> str:
        """Serialize a value for comparison."""
        if isinstance(value, dict):
            # Round floats for comparison
            processed = {
                k: round(v, 10) if isinstance(v, float) else v
                for k, v in sorted(value.items())
            }
            return json.dumps(processed, sort_keys=True)
        elif isinstance(value, (list, tuple)):
            processed = [
                round(v, 10) if isinstance(v, float) else v for v in value
            ]
            return json.dumps(processed)
        else:
            return str(value)

    def get_recorded_outputs(self) -> dict[str, Any]:
        """Get all recorded outputs."""
        return self._recorded_outputs.copy()
