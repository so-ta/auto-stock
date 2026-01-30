"""
Test configuration synchronization between YAML and Pydantic models.

Ensures that config/default.yaml (source of truth) and settings.py are in sync.
"""

import pytest
import yaml
from pathlib import Path

from src.config.settings import (
    Settings,
    load_settings_from_yaml,
    validate_yaml_pydantic_sync,
    DEFAULT_CONFIG_PATH,
    DataQualityConfig,
    RebalanceConfig,
    WalkForwardConfig,
    CostModelConfig,
)


class TestConfigSync:
    """Test YAML-Pydantic configuration synchronization."""

    def test_default_yaml_exists(self):
        """Verify default.yaml exists."""
        assert DEFAULT_CONFIG_PATH.exists(), f"Default config not found: {DEFAULT_CONFIG_PATH}"

    def test_yaml_loads_without_error(self):
        """Verify YAML can be loaded without parsing errors."""
        settings = load_settings_from_yaml()
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_validate_sync_no_critical_errors(self):
        """Verify no critical sync errors between YAML and Pydantic."""
        warnings = validate_yaml_pydantic_sync()
        # Allow some warnings but no critical mismatches in key fields
        critical_fields = [
            "data_quality.max_missing_rate",
            "rebalance.frequency",
            "walk_forward.train_period_days",
            "cost_model.spread_bps",
        ]
        critical_warnings = [w for w in warnings if any(f in w for f in critical_fields)]
        assert len(critical_warnings) == 0, f"Critical sync issues: {critical_warnings}"

    def test_data_quality_sync(self):
        """Test data_quality section sync."""
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        yaml_dq = yaml_data.get("data_quality", {})
        pydantic_dq = DataQualityConfig()

        # Check key fields match
        assert yaml_dq.get("max_missing_rate", 0.05) == pydantic_dq.max_missing_rate
        assert yaml_dq.get("max_consecutive_missing", 5) == pydantic_dq.max_consecutive_missing

    def test_rebalance_sync(self):
        """Test rebalance section sync."""
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        yaml_rb = yaml_data.get("rebalance", {})
        pydantic_rb = RebalanceConfig()

        # Check frequency matches (YAML uses string, Pydantic uses enum)
        yaml_freq = yaml_rb.get("frequency", "monthly")
        assert yaml_freq == pydantic_rb.frequency.value

    def test_walk_forward_sync(self):
        """Test walk_forward section sync."""
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        yaml_wf = yaml_data.get("walk_forward", {})
        pydantic_wf = WalkForwardConfig()

        # Check key fields match
        assert yaml_wf.get("train_period_days", 504) == pydantic_wf.train_period_days
        assert yaml_wf.get("test_period_days", 126) == pydantic_wf.test_period_days
        assert yaml_wf.get("purge_gap_days", 5) == pydantic_wf.purge_gap_days

    def test_cost_model_sync(self):
        """Test cost_model section sync."""
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        yaml_cm = yaml_data.get("cost_model", {})
        pydantic_cm = CostModelConfig()

        # Check key fields match
        assert yaml_cm.get("spread_bps", 10.0) == pydantic_cm.spread_bps
        assert yaml_cm.get("commission_bps", 5.0) == pydantic_cm.commission_bps
        assert yaml_cm.get("slippage_bps", 10.0) == pydantic_cm.slippage_bps


class TestLoadFromYaml:
    """Test YAML loading functionality."""

    def test_load_from_yaml_returns_settings(self):
        """Verify load_settings_from_yaml returns valid Settings."""
        settings = load_settings_from_yaml()
        assert isinstance(settings, Settings)

    def test_load_with_overrides(self):
        """Verify overrides work correctly."""
        settings = load_settings_from_yaml(
            system={"name": "test-system"}
        )
        assert settings.system.name == "test-system"

    def test_yaml_values_take_precedence(self):
        """Verify YAML values are used instead of Pydantic defaults."""
        settings = load_settings_from_yaml()

        # Read YAML directly
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        # Data quality from YAML should be used
        yaml_dq = yaml_data.get("data_quality", {})
        if yaml_dq:
            # If YAML has a different value than Pydantic default, verify YAML wins
            yaml_max_missing = yaml_dq.get("max_missing_rate")
            if yaml_max_missing is not None:
                assert settings.data_quality.max_missing_rate == yaml_max_missing


class TestYamlSourceOfTruth:
    """Test that YAML is treated as the single source of truth."""

    def test_get_settings_loads_from_yaml(self):
        """Verify get_settings() loads from YAML by default."""
        from src.config.settings import get_settings, _settings

        # Reset global settings
        import src.config.settings as settings_module
        settings_module._settings = None

        settings = get_settings()
        assert settings is not None

        # Should match YAML content
        with open(DEFAULT_CONFIG_PATH) as f:
            yaml_data = yaml.safe_load(f)

        yaml_system = yaml_data.get("system", {})
        if yaml_system.get("name"):
            assert settings.system.name == yaml_system["name"]

    def test_fallback_to_pydantic_if_yaml_missing(self, tmp_path):
        """Verify fallback to Pydantic defaults if YAML missing."""
        nonexistent_path = tmp_path / "nonexistent.yaml"
        settings = load_settings_from_yaml(nonexistent_path)
        assert isinstance(settings, Settings)
        # Should use Pydantic defaults
        assert settings.system.name == "multi-asset-portfolio"
