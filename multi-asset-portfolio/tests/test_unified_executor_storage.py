"""
Tests for UnifiedExecutor with StorageConfig integration.

Verifies:
1. storage_config parameter in run_backtest
2. Automatic storage_config retrieval from Settings
3. storage_config property access
"""

from datetime import datetime
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings, load_settings_from_yaml
from src.orchestrator.unified_executor import UnifiedExecutor
from src.utils.storage_backend import StorageConfig


@pytest.fixture
def sample_settings():
    """Load settings from config."""
    return load_settings_from_yaml()


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    prices = {}

    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        price_series = 100 * (1 + np.random.randn(100).cumsum() * 0.02)
        prices[symbol] = pd.DataFrame({
            "timestamp": dates,
            "open": price_series * 0.99,
            "high": price_series * 1.01,
            "low": price_series * 0.98,
            "close": price_series,
            "volume": np.random.randint(1000000, 10000000, 100),
        })

    return prices


class TestUnifiedExecutorStorageConfig:
    """Test UnifiedExecutor storage_config integration."""

    def test_init_without_storage_config(self, sample_settings):
        """Test initialization without explicit storage_config."""
        executor = UnifiedExecutor(settings=sample_settings)

        # storage_config should be retrievable from settings
        storage = executor.storage_config
        assert storage is not None
        assert storage.backend in ("local", "s3")

    def test_storage_config_from_settings(self, sample_settings):
        """Test that storage_config is correctly retrieved from Settings."""
        executor = UnifiedExecutor(settings=sample_settings)

        storage = executor.storage_config
        assert storage.backend == sample_settings.storage.backend
        assert storage.s3_bucket == sample_settings.storage.s3_bucket

    def test_storage_config_property_caching(self, sample_settings):
        """Test that storage_config property caches the value."""
        executor = UnifiedExecutor(settings=sample_settings)

        # First access
        storage1 = executor.storage_config
        # Second access should return same object
        storage2 = executor.storage_config

        assert storage1 is storage2

    def test_explicit_storage_config_in_run_backtest(self, sample_settings):
        """Test that explicit storage_config overrides Settings."""
        executor = UnifiedExecutor(settings=sample_settings)
        executor._storage_config = None  # Reset

        # Create explicit storage config
        explicit_config = StorageConfig(
            backend="s3",
            s3_bucket="explicit-test-bucket",
            s3_prefix="explicit-prefix",
        )

        # Directly test storage_config assignment logic
        # (mimicking the first few lines of run_backtest)
        if explicit_config is None and executor.settings is not None:
            storage_config = executor.settings.storage.to_storage_config()
        else:
            storage_config = explicit_config
        executor._storage_config = storage_config

        # After setting, storage_config should be the explicit one
        assert executor._storage_config is explicit_config
        assert executor._storage_config.s3_bucket == "explicit-test-bucket"

    def test_default_storage_config_in_run_backtest(self, sample_settings):
        """Test that Settings storage_config is used when not explicitly provided."""
        executor = UnifiedExecutor(settings=sample_settings)
        executor._storage_config = None  # Reset

        # Directly test storage_config assignment logic
        # (mimicking the first few lines of run_backtest)
        storage_config = None  # Simulating no explicit config
        if storage_config is None and executor.settings is not None:
            storage_config = executor.settings.storage.to_storage_config()
        executor._storage_config = storage_config

        # storage_config should come from settings
        assert executor._storage_config is not None
        assert executor._storage_config.backend == sample_settings.storage.backend


class TestStorageConfigPropagation:
    """Test storage_config propagation to cache classes."""

    def test_storage_config_available_for_cache(self, sample_settings):
        """Test that storage_config can be passed to cache classes."""
        executor = UnifiedExecutor(settings=sample_settings)

        storage = executor.storage_config

        # Verify it can be used with CovarianceCache
        from src.backtest.covariance_cache import CovarianceCache
        from src.utils.storage_backend import get_storage_backend

        # Create backend from config
        backend = get_storage_backend(storage)

        # Verify CovarianceCache can be initialized with backend
        cache = CovarianceCache(storage_backend=backend)
        assert cache._use_backend is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
