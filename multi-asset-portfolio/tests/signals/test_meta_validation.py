"""
Tests for Meta Validation module.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.signals.meta_validation import (
    AdaptiveParameterCalculator,
    LevelValidationResult,
    MetaValidationCache,
    MetaValidationResult,
    MetaValidator,
    OptimizationLevel,
    create_adaptive_calculator,
)


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(dates)

    # Random walk with drift
    returns = np.random.normal(0.0003, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "close": prices,
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_level_values(self):
        """Test optimization level values."""
        assert OptimizationLevel.FIXED == 0
        assert OptimizationLevel.STATISTICAL == 1
        assert OptimizationLevel.CONSTRAINED == 2
        assert OptimizationLevel.FULL == 3

    def test_level_ordering(self):
        """Test that levels are properly ordered."""
        assert OptimizationLevel.FIXED < OptimizationLevel.STATISTICAL
        assert OptimizationLevel.STATISTICAL < OptimizationLevel.CONSTRAINED
        assert OptimizationLevel.CONSTRAINED < OptimizationLevel.FULL


class TestMetaValidationResult:
    """Tests for MetaValidationResult dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        result = MetaValidationResult(
            level=OptimizationLevel.STATISTICAL,
            oos_sharpe=1.05,
            oos_sharpe_std=0.15,
            is_sharpe=1.10,
            overfitting_score=0.05,
            stability_score=6.67,
            validated_at=datetime(2024, 12, 25),
            data_end_date=datetime(2024, 12, 25),
            n_folds=3,
            metadata={"test": "value"},
        )

        # Serialize
        data = result.to_dict()
        assert data["level"] == 1
        assert data["level_name"] == "STATISTICAL"
        assert data["oos_sharpe"] == 1.05
        assert "metadata_json" in data  # metadata stored as JSON string

        # Deserialize
        restored = MetaValidationResult.from_dict(data)
        assert restored.level == OptimizationLevel.STATISTICAL
        assert restored.oos_sharpe == 1.05
        assert restored.metadata == {"test": "value"}


class TestLevelValidationResult:
    """Tests for LevelValidationResult dataclass."""

    def test_properties(self):
        """Test computed properties."""
        result = LevelValidationResult(
            level=OptimizationLevel.STATISTICAL,
            oos_sharpes=[0.8, 1.0, 1.2],
            is_sharpes=[1.0, 1.2, 1.4],
        )

        assert abs(result.oos_sharpe - 1.0) < 0.01
        assert abs(result.is_sharpe - 1.2) < 0.01
        assert result.overfitting_score > 0  # IS > OOS
        assert result.stability_score > 0
        assert result.composite_score != 0

    def test_empty_results(self):
        """Test with empty results."""
        result = LevelValidationResult(
            level=OptimizationLevel.FIXED,
            oos_sharpes=[],
            is_sharpes=[],
        )

        assert result.oos_sharpe == 0.0
        assert result.is_sharpe == 0.0


class TestMetaValidationCache:
    """Tests for MetaValidationCache."""

    def test_cache_put_and_get(self, temp_cache_dir):
        """Test cache storage and retrieval."""
        cache = MetaValidationCache(cache_dir=temp_cache_dir, signal_type="test")

        result = MetaValidationResult(
            level=OptimizationLevel.STATISTICAL,
            oos_sharpe=1.0,
            oos_sharpe_std=0.1,
            is_sharpe=1.1,
            overfitting_score=0.1,
            stability_score=10.0,
            validated_at=datetime.now(),
            data_end_date=datetime(2024, 12, 25),
            n_folds=3,
        )

        # Put
        cache.put(datetime(2024, 12, 25), result)

        # Get (should hit memory cache)
        retrieved = cache.get(datetime(2024, 12, 25))
        assert retrieved is not None
        assert retrieved.level == OptimizationLevel.STATISTICAL

        # Get with different date in same year (should hit same cache)
        retrieved2 = cache.get(datetime(2024, 6, 15))
        assert retrieved2 is not None
        assert retrieved2.level == OptimizationLevel.STATISTICAL

    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss."""
        cache = MetaValidationCache(cache_dir=temp_cache_dir, signal_type="test")

        # Should return None for uncached year
        result = cache.get(datetime(2024, 12, 25))
        assert result is None

    def test_disk_persistence(self, temp_cache_dir):
        """Test disk persistence."""
        # Create cache and store result
        cache1 = MetaValidationCache(cache_dir=temp_cache_dir, signal_type="test")
        result = MetaValidationResult(
            level=OptimizationLevel.CONSTRAINED,
            oos_sharpe=0.9,
            oos_sharpe_std=0.2,
            is_sharpe=1.0,
            overfitting_score=0.1,
            stability_score=5.0,
            validated_at=datetime.now(),
            data_end_date=datetime(2024, 12, 25),
            n_folds=3,
        )
        cache1.put(datetime(2024, 12, 25), result)

        # Create new cache instance (simulating restart)
        cache2 = MetaValidationCache(cache_dir=temp_cache_dir, signal_type="test")

        # Should retrieve from disk
        retrieved = cache2.get(datetime(2024, 12, 25))
        assert retrieved is not None
        assert retrieved.level == OptimizationLevel.CONSTRAINED


class TestMetaValidator:
    """Tests for MetaValidator."""

    def test_validate_all_levels(self, sample_prices):
        """Test validation of all levels."""
        validator = MetaValidator(n_folds=2)
        results = validator.validate_all_levels(sample_prices)

        assert len(results) == 4
        assert OptimizationLevel.FIXED in results
        assert OptimizationLevel.STATISTICAL in results
        assert OptimizationLevel.CONSTRAINED in results
        assert OptimizationLevel.FULL in results

    def test_recommend_level(self, sample_prices):
        """Test level recommendation."""
        validator = MetaValidator(n_folds=2)
        results = validator.validate_all_levels(sample_prices)
        recommended = validator.recommend_level(results)

        assert isinstance(recommended, OptimizationLevel)

    def test_short_data(self):
        """Test with insufficient data."""
        dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")
        prices = pd.DataFrame({
            "close": np.random.randn(len(dates)).cumsum() + 100,
        }, index=dates)

        validator = MetaValidator(n_folds=2)
        results = validator.validate_all_levels(prices)

        # Should return results but with default values
        assert len(results) == 4


class TestAdaptiveParameterCalculator:
    """Tests for AdaptiveParameterCalculator."""

    def test_initial_level(self, temp_cache_dir):
        """Test initial optimization level."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
        )

        assert calc.current_level == OptimizationLevel.STATISTICAL

    def test_get_params_before_validation(self, sample_prices, temp_cache_dir):
        """Test parameter calculation before meta validation."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
            min_history_years=2,
        )

        # Use date before December (no meta validation)
        params = calc.get_params(
            prices=sample_prices.loc[:"2020-06-15"],
            current_date=datetime(2020, 6, 15),
        )

        assert "lookback" in params
        assert "scale" in params
        assert calc.current_level == OptimizationLevel.STATISTICAL

    def test_meta_validation_triggered(self, sample_prices, temp_cache_dir):
        """Test that meta validation is triggered in December."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
            min_history_years=2,
            validation_month=12,
        )

        # First call in late December should trigger meta validation
        params = calc.get_params(
            prices=sample_prices.loc[:"2022-12-25"],
            current_date=datetime(2022, 12, 25),
        )

        assert "lookback" in params
        assert "scale" in params
        # Level history should be updated
        assert datetime(2022, 12, 25) in calc.level_history

    def test_cache_prevents_revalidation(self, sample_prices, temp_cache_dir):
        """Test that cache prevents redundant meta validation."""
        # First calculator runs meta validation
        calc1 = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
            min_history_years=2,
        )
        calc1.get_params(
            prices=sample_prices.loc[:"2022-12-25"],
            current_date=datetime(2022, 12, 25),
        )

        # Second calculator should use cache
        calc2 = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
            min_history_years=2,
        )
        calc2.get_params(
            prices=sample_prices.loc[:"2022-12-25"],
            current_date=datetime(2022, 12, 25),
        )

        # Both should have same level
        assert calc1.current_level == calc2.current_level

    def test_daily_backtest_simulation(self, sample_prices, temp_cache_dir):
        """Test simulation of daily backtest with yearly validation."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
            min_history_years=2,
            validation_month=12,
        )

        # Simulate daily backtest for 2022
        dates_2022 = sample_prices.loc["2022-01-01":"2022-12-31"].index

        validation_count = 0
        for date in dates_2022:
            params = calc.get_params(
                prices=sample_prices.loc[:date],
                current_date=date.to_pydatetime(),
            )

            # Check validation triggered only in late December
            if date.to_pydatetime() in calc.level_history:
                validation_count += 1

        # Should have exactly one validation in December
        assert validation_count == 1

    def test_get_validation_summary(self, sample_prices, temp_cache_dir):
        """Test validation summary generation."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="test",
        )

        # Run a validation
        calc.get_params(
            prices=sample_prices.loc[:"2022-12-25"],
            current_date=datetime(2022, 12, 25),
        )

        summary = calc.get_validation_summary()
        assert "META VALIDATION SUMMARY" in summary
        assert "Current Level" in summary


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_adaptive_calculator(self, temp_cache_dir):
        """Test factory function."""
        calc = create_adaptive_calculator(
            signal_type="momentum",
            cache_dir=temp_cache_dir,
        )

        assert isinstance(calc, AdaptiveParameterCalculator)
        assert calc.current_level == OptimizationLevel.STATISTICAL


class TestIntegration:
    """Integration tests."""

    def test_multi_year_backtest(self, sample_prices, temp_cache_dir):
        """Test multi-year backtest simulation."""
        calc = AdaptiveParameterCalculator(
            cache_dir=temp_cache_dir,
            signal_type="integration_test",
            min_history_years=2,
        )

        # Monthly rebalancing simulation: 2020-2024
        monthly_dates = pd.date_range("2020-01-31", "2024-12-31", freq="ME")

        params_history = []
        for date in monthly_dates:
            if date > sample_prices.index[-1]:
                break

            params = calc.get_params(
                prices=sample_prices.loc[:date],
                current_date=date.to_pydatetime(),
            )
            params_history.append({
                "date": date,
                "lookback": params["lookback"],
                "scale": params["scale"],
                "level": calc.current_level.name,
            })

        # Should have multiple years of history
        assert len(params_history) > 40

        # Should have validations in December
        december_entries = [
            p for p in params_history
            if p["date"].month == 12 and p["date"].day >= 25
        ]
        # Validations should be reflected in level history
        assert len(calc.level_history) >= 3  # 2020, 2021, 2022, ...
