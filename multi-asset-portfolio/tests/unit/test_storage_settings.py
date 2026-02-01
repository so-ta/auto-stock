"""
Test StorageSettings configuration (S3 required mode).

Tests:
1. StorageSettings requires s3_bucket
2. StorageSettings with valid values
3. to_storage_config() conversion
4. Settings.storage attribute
"""

import pytest

from src.config.settings import Settings, StorageSettings


class TestStorageSettings:
    """Test StorageSettings configuration (S3 required for production)."""

    def test_allows_empty_s3_bucket_for_testing(self):
        """Test that s3_bucket can be empty (for testing)."""
        # Empty s3_bucket is now allowed to support testing scenarios
        storage = StorageSettings()
        assert storage.s3_bucket == ""

    def test_valid_values(self):
        """Test StorageSettings with valid values."""
        storage = StorageSettings(
            s3_bucket="my-bucket",
            s3_prefix="cache/data",
            s3_region="us-west-2",
            base_path="/data/cache",
            local_cache_ttl_hours=48,
        )

        assert storage.s3_bucket == "my-bucket"
        assert storage.s3_prefix == "cache/data"
        assert storage.s3_region == "us-west-2"
        assert storage.base_path == "/data/cache"
        assert storage.local_cache_ttl_hours == 48

    def test_default_values(self):
        """Test StorageSettings default values (except s3_bucket)."""
        storage = StorageSettings(s3_bucket="test-bucket")

        assert storage.s3_bucket == "test-bucket"
        assert storage.s3_prefix == ".cache"
        assert storage.s3_region == "ap-northeast-1"
        assert storage.base_path == ".cache"
        assert storage.local_cache_ttl_hours == 24

    def test_to_storage_config(self):
        """Test to_storage_config() conversion."""
        storage = StorageSettings(
            s3_bucket="test-bucket",
            s3_prefix="backtest",
            s3_region="eu-west-1",
            base_path="/tmp/cache",
            local_cache_ttl_hours=12,
        )

        config = storage.to_storage_config()

        assert config.s3_bucket == "test-bucket"
        assert config.s3_prefix == "backtest"
        assert config.s3_region == "eu-west-1"
        assert config.base_path == "/tmp/cache"
        assert config.local_cache_ttl_hours == 12


class TestSettingsStorageAttribute:
    """Test Settings.storage attribute."""

    def test_settings_has_storage_attribute(self):
        """Test that Settings has storage attribute."""
        settings = Settings(storage={"s3_bucket": "test-bucket"})

        assert hasattr(settings, "storage")
        assert isinstance(settings.storage, StorageSettings)

    def test_settings_storage_allows_empty_bucket_for_testing(self):
        """Test Settings.storage allows empty s3_bucket for testing."""
        # Empty s3_bucket is now allowed to support testing scenarios
        settings = Settings()
        assert settings.storage.s3_bucket == ""

    def test_settings_storage_from_dict(self):
        """Test Settings with storage from dict."""
        settings = Settings(
            storage={
                "s3_bucket": "my-bucket",
                "s3_prefix": "custom-prefix",
            }
        )

        assert settings.storage.s3_bucket == "my-bucket"
        assert settings.storage.s3_prefix == "custom-prefix"

    def test_settings_storage_to_config(self):
        """Test Settings.storage.to_storage_config()."""
        settings = Settings(storage={"s3_bucket": "test-bucket"})
        config = settings.storage.to_storage_config()

        # Verify it's a StorageConfig dataclass
        from src.utils.storage_backend import StorageConfig

        assert isinstance(config, StorageConfig)
        assert config.s3_bucket == "test-bucket"


class TestStorageSettingsValidation:
    """Test StorageSettings validation."""

    def test_s3_bucket_accepts_empty_for_testing(self):
        """Test that s3_bucket accepts empty string for testing scenarios."""
        # Empty string is now allowed to support testing
        storage = StorageSettings(s3_bucket="")
        assert storage.s3_bucket == ""

    def test_local_cache_ttl_positive(self):
        """Test that local_cache_ttl_hours must be positive."""
        with pytest.raises(Exception):
            StorageSettings(s3_bucket="test", local_cache_ttl_hours=0)


class TestSignalPrecomputeSettings:
    """Test SignalPrecomputeSettings configuration."""

    def test_default_values(self):
        """Test default period variants."""
        from src.config.settings import SignalPrecomputeSettings

        settings = SignalPrecomputeSettings()

        assert settings.period_variants.short == 5
        assert settings.period_variants.medium == 20
        assert settings.period_variants.long == 60
        assert settings.period_variants.half_year == 126
        assert settings.period_variants.yearly == 252

    def test_enabled_variants_default(self):
        """Test default enabled variants."""
        from src.config.settings import SignalPrecomputeSettings

        settings = SignalPrecomputeSettings()

        assert "short" in settings.enabled_variants
        assert "medium" in settings.enabled_variants
        assert "long" in settings.enabled_variants
        assert "half_year" in settings.enabled_variants
        assert "yearly" in settings.enabled_variants

    def test_custom_periods(self):
        """Test custom periods for specific signals."""
        from src.config.settings import SignalPrecomputeSettings

        settings = SignalPrecomputeSettings(
            custom_periods={"rsi": [7, 14, 21]}
        )

        assert settings.custom_periods["rsi"] == [7, 14, 21]

    def test_period_variants_to_dict(self):
        """Test PeriodVariantSettings.to_dict()."""
        from src.config.settings import PeriodVariantSettings

        variants = PeriodVariantSettings()
        d = variants.to_dict()

        assert d["short"] == 5
        assert d["medium"] == 20
        assert d["long"] == 60
        assert d["half_year"] == 126
        assert d["yearly"] == 252

    def test_invalid_enabled_variant(self):
        """Test that invalid variant names are rejected."""
        from src.config.settings import SignalPrecomputeSettings

        with pytest.raises(ValueError):
            SignalPrecomputeSettings(enabled_variants=["invalid_variant"])

    def test_settings_has_signal_precompute(self):
        """Test that Settings has signal_precompute attribute."""
        from src.config.settings import SignalPrecomputeSettings

        settings = Settings()

        assert hasattr(settings, "signal_precompute")
        assert isinstance(settings.signal_precompute, SignalPrecomputeSettings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
