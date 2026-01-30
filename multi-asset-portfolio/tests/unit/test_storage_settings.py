"""
Test StorageSettings configuration (task_045_6).

Tests:
1. StorageSettings default values
2. StorageSettings with custom values
3. to_storage_config() conversion
4. Settings.storage attribute
"""

import pytest

from src.config.settings import Settings, StorageSettings


class TestStorageSettings:
    """Test StorageSettings configuration."""

    def test_default_values(self):
        """Test default StorageSettings values."""
        storage = StorageSettings()

        assert storage.backend == "local"
        assert storage.base_path == ".cache"
        assert storage.s3_bucket == ""
        assert storage.s3_prefix == ".cache"
        assert storage.s3_region == "ap-northeast-1"
        assert storage.local_cache_enabled is True
        assert storage.local_cache_path == "/tmp/.backtest_cache"
        assert storage.local_cache_ttl_hours == 24

    def test_custom_values(self):
        """Test StorageSettings with custom values."""
        storage = StorageSettings(
            backend="s3",
            s3_bucket="my-bucket",
            s3_prefix="cache/data",
            s3_region="us-west-2",
            local_cache_enabled=False,
            local_cache_ttl_hours=48,
        )

        assert storage.backend == "s3"
        assert storage.s3_bucket == "my-bucket"
        assert storage.s3_prefix == "cache/data"
        assert storage.s3_region == "us-west-2"
        assert storage.local_cache_enabled is False
        assert storage.local_cache_ttl_hours == 48

    def test_to_storage_config_local(self):
        """Test to_storage_config() for local backend."""
        storage = StorageSettings(
            backend="local",
            base_path="/data/cache",
        )

        config = storage.to_storage_config()

        assert config.backend == "local"
        assert config.base_path == "/data/cache"

    def test_to_storage_config_s3(self):
        """Test to_storage_config() for S3 backend."""
        storage = StorageSettings(
            backend="s3",
            s3_bucket="test-bucket",
            s3_prefix="backtest",
            s3_region="eu-west-1",
            local_cache_enabled=True,
            local_cache_path="/tmp/cache",
            local_cache_ttl_hours=12,
        )

        config = storage.to_storage_config()

        assert config.backend == "s3"
        assert config.s3_bucket == "test-bucket"
        assert config.s3_prefix == "backtest"
        assert config.s3_region == "eu-west-1"
        assert config.local_cache_enabled is True
        assert config.local_cache_path == "/tmp/cache"
        assert config.local_cache_ttl_hours == 12


class TestSettingsStorageAttribute:
    """Test Settings.storage attribute."""

    def test_settings_has_storage_attribute(self):
        """Test that Settings has storage attribute."""
        settings = Settings()

        assert hasattr(settings, "storage")
        assert isinstance(settings.storage, StorageSettings)

    def test_settings_storage_default(self):
        """Test Settings.storage default values."""
        settings = Settings()

        assert settings.storage.backend == "local"
        assert settings.storage.base_path == ".cache"

    def test_settings_storage_from_dict(self):
        """Test Settings with storage from dict."""
        settings = Settings(
            storage={
                "backend": "s3",
                "s3_bucket": "my-bucket",
                "local_cache_enabled": False,
            }
        )

        assert settings.storage.backend == "s3"
        assert settings.storage.s3_bucket == "my-bucket"
        assert settings.storage.local_cache_enabled is False

    def test_settings_storage_to_config(self):
        """Test Settings.storage.to_storage_config()."""
        settings = Settings()
        config = settings.storage.to_storage_config()

        # Verify it's a StorageConfig dataclass
        from src.utils.storage_backend import StorageConfig

        assert isinstance(config, StorageConfig)
        assert config.backend == "local"


class TestStorageSettingsValidation:
    """Test StorageSettings validation."""

    def test_backend_literal_validation(self):
        """Test that backend must be 'local' or 's3'."""
        # Valid values
        StorageSettings(backend="local")
        StorageSettings(backend="s3")

        # Invalid value should raise error
        with pytest.raises(Exception):  # Pydantic ValidationError
            StorageSettings(backend="invalid")

    def test_local_cache_ttl_positive(self):
        """Test that local_cache_ttl_hours must be positive."""
        with pytest.raises(Exception):
            StorageSettings(local_cache_ttl_hours=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
