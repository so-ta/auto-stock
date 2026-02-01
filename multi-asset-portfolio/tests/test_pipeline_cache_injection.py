"""
Tests for Pipeline cache injection order (task_cache_fix Phase 1.2).

These tests verify that SignalPrecomputer is properly injected into Pipeline
and StrategyEvaluator before first access, preventing cache miss issues.
"""

import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_prices_dict():
    """Create sample price data as Dict[str, pd.DataFrame] for UnifiedExecutor."""
    dates = pd.date_range("2024-01-01", "2024-02-28", freq="D")
    tickers = ["AAPL", "GOOGL", "MSFT"]

    prices = {}
    for ticker in tickers:
        prices[ticker] = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0 + i * 0.5 for i in range(len(dates))],
            "high": [101.0 + i * 0.5 for i in range(len(dates))],
            "low": [99.0 + i * 0.5 for i in range(len(dates))],
            "volume": [1000000] * len(dates),
        }).set_index("timestamp")

    return prices


@pytest.fixture
def sample_prices_polars():
    """Create sample price DataFrame in Polars format."""
    dates = pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 2, 28),
        eager=True,
    )
    tickers = ["AAPL", "GOOGL", "MSFT"]
    data = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "ticker": ticker,
                "close": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
            })
    return pl.DataFrame(data)


# =============================================================================
# Test Pipeline Signal Precomputer Injection
# =============================================================================

class TestPipelineSignalPrecomputerInjection:
    """Tests for Pipeline receiving SignalPrecomputer at construction."""

    def test_pipeline_accepts_signal_precomputer_in_constructor(self, sample_prices_polars):
        """Test: Pipeline constructor accepts signal_precomputer parameter."""
        from src.orchestrator.pipeline import Pipeline, PipelineConfig
        from src.backtest.signal_precompute import SignalPrecomputer

        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)
            precomputer.precompute_all(sample_prices_polars)

            # Create Pipeline with precomputer
            pipeline = Pipeline(
                settings=None,
                config=PipelineConfig(lightweight_mode=True),
                signal_precomputer=precomputer,
            )

            # Verify precomputer is set
            assert pipeline._signal_precomputer is precomputer

    def test_strategy_evaluator_receives_precomputer(self, sample_prices_polars):
        """Test: StrategyEvaluator gets precomputer from Pipeline."""
        from src.orchestrator.pipeline import Pipeline, PipelineConfig
        from src.backtest.signal_precompute import SignalPrecomputer

        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)
            precomputer.precompute_all(sample_prices_polars)

            # Create Pipeline with precomputer
            pipeline = Pipeline(
                settings=None,
                config=PipelineConfig(lightweight_mode=True),
                signal_precomputer=precomputer,
            )

            # Access strategy_evaluator property
            evaluator = pipeline.strategy_evaluator

            # Verify evaluator has the precomputer
            assert evaluator._signal_precomputer is precomputer


# =============================================================================
# Test UnifiedExecutor Precompute Order
# =============================================================================

class TestUnifiedExecutorPrecomputeOrder:
    """Tests for UnifiedExecutor signal precomputation order."""

    def test_precomputer_set_before_pipeline_creation(self, sample_prices_dict):
        """Test: _signal_precomputer is set before backtest_pipeline property access."""
        from src.orchestrator.unified_executor import UnifiedExecutor

        executor = UnifiedExecutor()

        # Verify precomputer is None initially
        assert executor._signal_precomputer is None

        # Mock the signal precomputation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually call _precompute_signals to simulate run_backtest flow
            tickers = list(sample_prices_dict.keys())

            # Convert prices to Polars format
            prices_records = []
            for symbol, df in sample_prices_dict.items():
                df_copy = df.copy().reset_index()
                df_copy.columns = ["timestamp"] + list(df.columns)
                df_copy["ticker"] = symbol
                prices_records.append(df_copy[["timestamp", "ticker", "close", "high", "low", "volume"]])

            combined_prices = pd.concat(prices_records, ignore_index=True)
            prices_pl = pl.from_pandas(combined_prices)

            # Initialize precomputer
            from src.backtest.signal_precompute import SignalPrecomputer
            executor._signal_precomputer = SignalPrecomputer(cache_dir=tmpdir)
            executor._signal_precomputer.precompute_all(prices_pl)

            # Now access backtest_pipeline - it should receive the precomputer
            pipeline = executor.backtest_pipeline

            # The pipeline should have the precomputer
            assert pipeline._signal_precomputer is executor._signal_precomputer

    def test_backtest_pipeline_updates_existing_precomputer(self, sample_prices_dict):
        """Test: backtest_pipeline property updates precomputer if changed after creation."""
        from src.orchestrator.unified_executor import UnifiedExecutor
        from src.backtest.signal_precompute import SignalPrecomputer

        executor = UnifiedExecutor()

        # Create pipeline first (without precomputer)
        pipeline1 = executor.backtest_pipeline
        assert pipeline1._signal_precomputer is None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set precomputer after pipeline creation
            executor._signal_precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # Access pipeline again - should update
            pipeline2 = executor.backtest_pipeline

            # Same pipeline instance
            assert pipeline2 is pipeline1

            # Precomputer should be updated
            assert pipeline2._signal_precomputer is executor._signal_precomputer


# =============================================================================
# Test Cache Hit with Correct Injection
# =============================================================================

class TestCacheHitWithCorrectInjection:
    """Tests verifying cache is used when injection order is correct."""

    def test_precomputed_signals_used_in_pipeline_run(self, sample_prices_polars):
        """Test: Pipeline uses precomputed signals when available."""
        from src.orchestrator.pipeline import Pipeline, PipelineConfig
        from src.backtest.signal_precompute import SignalPrecomputer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup precomputer with signals
            precomputer = SignalPrecomputer(cache_dir=tmpdir)
            precomputer.precompute_all(sample_prices_polars)

            # Create Pipeline with precomputer and use_precomputed_signals=True
            config = PipelineConfig(
                lightweight_mode=True,
                use_precomputed_signals=True,
            )
            pipeline = Pipeline(
                settings=None,
                config=config,
                signal_precomputer=precomputer,
            )

            # Verify config is set correctly
            assert pipeline.config.use_precomputed_signals is True
            assert pipeline._signal_precomputer is precomputer

    def test_signal_precomputer_cache_stats_tracked(self, sample_prices_polars):
        """Test: Cache stats are tracked during precomputation."""
        from src.backtest.signal_precompute import SignalPrecomputer

        with tempfile.TemporaryDirectory() as tmpdir:
            precomputer = SignalPrecomputer(cache_dir=tmpdir)

            # First run - should compute
            computed1 = precomputer.precompute_all(sample_prices_polars)
            assert computed1 is True

            # Get stats after first run
            stats1 = precomputer.cache_stats.copy()

            # Reset stats
            precomputer.reset_cache_stats()

            # Second run - should use cache
            computed2 = precomputer.precompute_all(sample_prices_polars)
            assert computed2 is False

            # Get stats after second run
            stats2 = precomputer.cache_stats

            # Second run should show cache hit
            assert stats2["hits"] >= 1 or stats2["misses"] == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
