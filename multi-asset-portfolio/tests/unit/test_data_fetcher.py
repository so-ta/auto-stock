"""Unit tests for data fetcher and quality checker modules."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


class TestDataFetcher:
    """Tests for DataFetcher base class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 24
        return settings

    @pytest.fixture
    def sample_ohlcv_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pl.date_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            eager=True,
        )
        n = len(dates)
        np.random.seed(42)

        return pl.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, n),
            "high": np.random.uniform(110, 120, n),
            "low": np.random.uniform(90, 100, n),
            "close": np.random.uniform(100, 110, n),
            "volume": np.random.uniform(1000, 10000, n),
        })

    def test_mock_fetcher_generates_valid_data(self, mock_settings):
        """Test that MockDataFetcher generates valid OHLCV data."""
        from src.config.schemas import AssetClass
        from src.data.fetcher import MockDataFetcher

        fetcher = MockDataFetcher(
            asset_class=AssetClass.CRYPTO,
            symbols=["TEST1", "TEST2"],
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        df = fetcher.fetch_ohlcv("TEST1", start, end, "1d")

        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_mock_fetcher_available_symbols(self):
        """Test getting available symbols from MockDataFetcher."""
        from src.config.schemas import AssetClass
        from src.data.fetcher import MockDataFetcher

        symbols = ["SYM1", "SYM2", "SYM3"]
        fetcher = MockDataFetcher(
            asset_class=AssetClass.STOCK,
            symbols=symbols,
        )

        available = fetcher.get_available_symbols()
        assert available == symbols

    def test_mock_fetcher_invalid_symbol_raises(self):
        """Test that fetching invalid symbol raises DataNotFoundError."""
        from src.config.schemas import AssetClass
        from src.data.fetcher import DataNotFoundError, MockDataFetcher

        fetcher = MockDataFetcher(
            asset_class=AssetClass.CRYPTO,
            symbols=["VALID"],
        )

        with pytest.raises(DataNotFoundError):
            fetcher.fetch_ohlcv(
                "INVALID",
                datetime(2024, 1, 1),
                datetime(2024, 1, 31),
            )

    def test_cached_fetcher_caches_data(self, mock_settings):
        """Test that CachedDataFetcher caches data properly."""
        from src.config.schemas import AssetClass
        from src.data.fetcher import CachedDataFetcher, MockDataFetcher

        base_fetcher = MockDataFetcher(
            asset_class=AssetClass.CRYPTO,
            symbols=["TEST"],
        )
        cached_fetcher = CachedDataFetcher(base_fetcher, cache_ttl_hours=24)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # First fetch
        df1 = cached_fetcher.fetch_ohlcv("TEST", start, end)

        # Second fetch should use cache
        df2 = cached_fetcher.fetch_ohlcv("TEST", start, end)

        assert len(df1) == len(df2)

    def test_validate_dataframe_checks_columns(self):
        """Test DataFrame validation checks required columns."""
        from src.config.schemas import AssetClass
        from src.data.fetcher import MockDataFetcher

        fetcher = MockDataFetcher(
            asset_class=AssetClass.CRYPTO,
            symbols=["TEST"],
        )

        # Valid DataFrame
        valid_df = pl.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000.0],
        })
        assert fetcher.validate_dataframe(valid_df) is True

        # Invalid DataFrame (missing column)
        invalid_df = pl.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [110.0],
        })
        assert fetcher.validate_dataframe(invalid_df) is False


class TestQualityChecker:
    """Tests for DataQualityChecker."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for quality checker."""
        settings = MagicMock()
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 24
        settings.data_quality.ohlc_inconsistency_threshold = 0.01
        return settings

    @pytest.fixture
    def valid_ohlcv_df(self):
        """Create valid OHLCV DataFrame."""
        n = 30
        # Use 23 hours ago instead of n days to be within staleness threshold
        base_date = datetime.utcnow() - timedelta(hours=23)
        dates = [base_date - timedelta(days=n-1-i) for i in range(n)]
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))

        return pl.DataFrame({
            "timestamp": dates,
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        })

    def test_quality_check_passes_valid_data(self, mock_settings, valid_ohlcv_df):
        """Test that quality check passes for valid data."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)
        report = checker.check(valid_ohlcv_df, "TEST", expected_bars=30)

        assert report.overall_passed is True
        assert report.is_excluded is False

    def test_quality_check_detects_empty_data(self, mock_settings):
        """Test that empty DataFrame is detected."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)
        empty_df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })

        report = checker.check(empty_df, "TEST")
        empty_check = next(c for c in report.checks if c.check_name == "empty_check")
        assert empty_check.passed is False

    def test_quality_check_detects_missing_values(self, mock_settings, valid_ohlcv_df):
        """Test that null values in price columns are detected."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)

        # Add null values
        df_with_nulls = valid_ohlcv_df.with_columns(
            pl.when(pl.col("close").is_first_distinct())
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )

        report = checker.check(df_with_nulls, "TEST")
        missing_check = next(c for c in report.checks if c.check_name == "missing_values")
        assert missing_check.passed is False

    def test_quality_check_detects_duplicates(self, mock_settings, valid_ohlcv_df):
        """Test that duplicate timestamps are detected."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)

        # Create duplicate by concatenating
        df_with_dups = pl.concat([valid_ohlcv_df, valid_ohlcv_df.head(5)])

        report = checker.check(df_with_dups, "TEST")
        dup_check = next(c for c in report.checks if c.check_name == "duplicates")
        assert dup_check.passed is False
        assert report.duplicate_count > 0

    def test_quality_check_detects_ohlc_inconsistency(self, mock_settings):
        """Test that OHLC inconsistencies are detected."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)

        # Create data where high < low
        df = pl.DataFrame({
            "timestamp": [datetime.utcnow() - timedelta(days=i) for i in range(5)],
            "open": [100.0] * 5,
            "high": [90.0] * 5,  # Invalid: high < low
            "low": [110.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        })

        report = checker.check(df, "TEST")
        consistency_check = next(c for c in report.checks if c.check_name == "ohlc_consistency")
        assert consistency_check.passed is False

    def test_quality_check_batch_processing(self, mock_settings, valid_ohlcv_df):
        """Test batch quality check processing."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)

        data = {
            "ASSET1": valid_ohlcv_df,
            "ASSET2": valid_ohlcv_df,
        }

        reports = checker.check_batch(data)

        assert len(reports) == 2
        assert "ASSET1" in reports
        assert "ASSET2" in reports

    def test_filter_excluded_assets(self, mock_settings, valid_ohlcv_df):
        """Test filtering excluded assets."""
        from src.data.quality_checker import DataQualityChecker

        checker = DataQualityChecker(mock_settings)

        # Create one valid and one empty DataFrame
        empty_df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })

        data = {
            "VALID": valid_ohlcv_df,
            "INVALID": empty_df,
        }

        filtered, excluded = checker.filter_excluded(data)

        assert "VALID" in filtered
        assert "INVALID" not in filtered
        assert "INVALID" in excluded
