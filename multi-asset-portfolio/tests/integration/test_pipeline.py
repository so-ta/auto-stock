"""Integration tests for the pipeline orchestrator and reproducibility."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest


class TestPipelineExecution:
    """Integration tests for Pipeline execution."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for pipeline."""
        settings = MagicMock()
        settings.universe = ["TEST1", "TEST2", "TEST3"]
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 24
        settings.asset_allocation.method.value = "HRP"
        settings.asset_allocation.w_asset_max = 0.4
        settings.asset_allocation.delta_max = 0.1
        settings.asset_allocation.smooth_alpha = 0.3
        settings.hard_gates.min_trades = 30
        settings.hard_gates.max_mdd = 0.25
        return settings

    @pytest.fixture
    def pipeline_config(self, tmp_path):
        """Create pipeline configuration."""
        from src.orchestrator.pipeline import PipelineConfig

        return PipelineConfig(
            run_id="test_run_001",
            seed=42,
            dry_run=False,
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )

    def test_pipeline_initialization(self, mock_settings, pipeline_config):
        """Test pipeline initialization."""
        from src.orchestrator.pipeline import Pipeline

        pipeline = Pipeline(settings=mock_settings, config=pipeline_config)

        assert pipeline._config.run_id == "test_run_001"
        assert pipeline._config.seed == 42

    def test_pipeline_empty_universe_error(self, mock_settings, pipeline_config):
        """Test pipeline returns error for empty universe."""
        from src.orchestrator.pipeline import Pipeline, PipelineStatus

        mock_settings.universe = []
        pipeline = Pipeline(settings=mock_settings, config=pipeline_config)

        result = pipeline.run(universe=[])

        assert result.status == PipelineStatus.FAILED
        assert "Empty universe" in result.errors[0] or len(result.errors) > 0

    def test_pipeline_step_results_tracking(self, mock_settings, pipeline_config):
        """Test that pipeline tracks step results."""
        from src.orchestrator.pipeline import Pipeline, PipelineStep

        pipeline = Pipeline(settings=mock_settings, config=pipeline_config)
        pipeline._config.skip_data_fetch = True

        result = pipeline.run(
            universe=["TEST1", "TEST2"],
            previous_weights={"TEST1": 0.5, "TEST2": 0.5},
        )

        # Should have step results
        assert len(result.step_results) > 0

        # Check that key steps were recorded
        step_names = [sr.step.value for sr in result.step_results]
        assert "data_fetch" in step_names or "quality_check" in step_names

    def test_pipeline_output_generation(self, mock_settings, pipeline_config, tmp_path):
        """Test pipeline output file generation."""
        from src.orchestrator.pipeline import Pipeline

        pipeline = Pipeline(settings=mock_settings, config=pipeline_config)
        pipeline._config.skip_data_fetch = True

        result = pipeline.run(
            universe=["TEST1", "TEST2"],
            previous_weights={"TEST1": 0.5, "TEST2": 0.5},
        )

        # Check output directory was created
        output_dir = tmp_path / "output"
        assert output_dir.exists()

    def test_pipeline_result_to_dict(self, mock_settings, pipeline_config):
        """Test PipelineResult serialization."""
        from src.orchestrator.pipeline import Pipeline

        pipeline = Pipeline(settings=mock_settings, config=pipeline_config)
        pipeline._config.skip_data_fetch = True

        result = pipeline.run(
            universe=["TEST1", "TEST2"],
            previous_weights={},
        )

        result_dict = result.to_dict()

        assert "run_id" in result_dict
        assert "status" in result_dict
        assert "weights" in result_dict
        assert "diagnostics" in result_dict


class TestReproducibility:
    """Tests for reproducibility (same seed -> same output)."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.universe = ["A", "B", "C"]
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 24
        settings.asset_allocation.method.value = "HRP"
        settings.asset_allocation.w_asset_max = 0.4
        settings.asset_allocation.delta_max = 0.1
        settings.asset_allocation.smooth_alpha = 0.3
        return settings

    def test_seed_manager_reproducibility(self):
        """Test that SeedManager produces reproducible results."""
        from src.utils.reproducibility import SeedManager

        results1 = []
        results2 = []

        # First run
        with SeedManager(seed=42):
            results1.append(np.random.rand())
            results1.append(np.random.rand())

        # Second run with same seed
        with SeedManager(seed=42):
            results2.append(np.random.rand())
            results2.append(np.random.rand())

        assert results1 == results2

    def test_seed_manager_different_seeds(self):
        """Test that different seeds produce different results."""
        from src.utils.reproducibility import SeedManager

        with SeedManager(seed=42):
            result1 = np.random.rand()

        with SeedManager(seed=123):
            result2 = np.random.rand()

        assert result1 != result2

    def test_metrics_reproducibility(self):
        """Test that metrics calculation is reproducible."""
        from src.strategy.metrics import MetricsCalculator
        from src.utils.reproducibility import SeedManager

        calc = MetricsCalculator()

        def compute_with_seed(seed):
            with SeedManager(seed=seed):
                returns = np.random.normal(0.001, 0.02, 252)
                return calc.calculate(returns)

        metrics1 = compute_with_seed(42)
        metrics2 = compute_with_seed(42)

        assert metrics1.sharpe_ratio == metrics2.sharpe_ratio
        assert metrics1.max_drawdown == metrics2.max_drawdown
        assert metrics1.sortino_ratio == metrics2.sortino_ratio

    def test_allocation_reproducibility(self):
        """Test that allocation is reproducible with same seed."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.utils.reproducibility import SeedManager

        def allocate_with_seed(seed):
            with SeedManager(seed=seed):
                np.random.seed(seed)
                returns = pd.DataFrame(
                    np.random.normal(0, 0.02, (252, 3)),
                    columns=["A", "B", "C"],
                )

                allocator = AssetAllocator(AllocatorConfig())
                return allocator.allocate(returns)

        result1 = allocate_with_seed(42)
        result2 = allocate_with_seed(42)

        # Weights should be identical
        for asset in ["A", "B", "C"]:
            assert result1.weights[asset] == pytest.approx(
                result2.weights[asset], rel=0.0001
            )

    def test_scorer_reproducibility(self):
        """Test that scoring is reproducible."""
        from src.meta.scorer import ScorerConfig, StrategyMetricsInput, StrategyScorer
        from src.utils.reproducibility import SeedManager

        def score_with_seed(seed):
            with SeedManager(seed=seed):
                np.random.seed(seed)
                metrics = StrategyMetricsInput(
                    strategy_id="test",
                    asset_id="ASSET",
                    sharpe_ratio=1.0 + np.random.rand(),
                    max_drawdown_pct=10 + np.random.rand() * 10,
                    turnover=np.random.rand(),
                    period_returns=list(np.random.normal(0.01, 0.02, 10)),
                )

                scorer = StrategyScorer(ScorerConfig())
                return scorer.score(metrics)

        result1 = score_with_seed(42)
        result2 = score_with_seed(42)

        assert result1.final_score == pytest.approx(result2.final_score, rel=0.0001)
        assert result1.total_penalty == pytest.approx(result2.total_penalty, rel=0.0001)


class TestAnomalyDetection:
    """Integration tests for anomaly detection."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.degradation_mode.correlation_threshold = 0.8
        settings.degradation_mode.vix_threshold = 30
        settings.data_quality.max_missing_rate = 0.05
        return settings

    def test_anomaly_detector_no_anomalies(self, mock_settings):
        """Test anomaly detector with clean data."""
        from src.orchestrator.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(mock_settings)

        result = detector.check_all(
            quality_reports={},
            portfolio_risk={"volatility": 0.15, "var_95": 0.02},
            correlation_matrix=None,
            price_data={},
        )

        assert not result.should_trigger_fallback
        assert len(result.anomalies) == 0

    def test_anomaly_detector_high_correlation(self, mock_settings):
        """Test anomaly detector detects high correlation."""
        from src.orchestrator.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(mock_settings)

        # High correlation matrix
        corr = pd.DataFrame(
            [[1.0, 0.95, 0.90], [0.95, 1.0, 0.92], [0.90, 0.92, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        result = detector.check_all(
            quality_reports={},
            portfolio_risk={"volatility": 0.15},
            correlation_matrix=corr,
            price_data={},
        )

        # Should detect high correlation
        assert result.has_warnings or result.should_trigger_fallback


class TestFallbackHandler:
    """Integration tests for fallback handling."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.degradation_mode.levels = {
            "level_0": {"min_adopted_strategies": 3, "cash_ratio": 0.0},
            "level_1": {"min_adopted_strategies": 1, "cash_ratio": 0.3},
            "level_2": {"adopted_strategies": 0, "cash_ratio": 0.8},
        }
        return settings

    def test_fallback_hold_previous(self, mock_settings):
        """Test fallback to previous weights."""
        from src.orchestrator.fallback import FallbackHandler, FallbackMode

        handler = FallbackHandler(mock_settings)

        previous = {"A": 0.4, "B": 0.4, "C": 0.2}
        new = {"A": 0.6, "B": 0.3, "C": 0.1}

        applied = handler.apply_fallback(
            previous_weights=previous,
            new_weights=new,
            reason="test_fallback",
            adopted_strategies=0,
        )

        # Should apply some fallback mode
        assert handler.current_mode is not None


class TestEndToEndPipeline:
    """End-to-end pipeline tests with mock data."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for multiple assets."""
        np.random.seed(42)
        n = 252
        assets = ["ASSET1", "ASSET2", "ASSET3"]

        data = {}
        for asset in assets:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
            dates = [datetime(2024, 1, 1) + pd.Timedelta(days=i) for i in range(n)]

            data[asset] = pl.DataFrame({
                "timestamp": dates,
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n),
            })

        return data

    def test_data_quality_to_allocation_flow(self, sample_ohlcv_data):
        """Test flow from data quality check to allocation."""
        from src.allocation.allocator import AllocatorConfig, AssetAllocator
        from src.data.quality_checker import DataQualityChecker

        # Mock settings for quality checker
        settings = MagicMock()
        settings.data_quality.max_missing_rate = 0.05
        settings.data_quality.max_consecutive_missing = 5
        settings.data_quality.price_change_threshold = 0.5
        settings.data_quality.min_volume_threshold = 0
        settings.data_quality.staleness_hours = 48

        # Quality check
        checker = DataQualityChecker(settings)
        filtered_data, excluded = checker.filter_excluded(sample_ohlcv_data)

        assert len(filtered_data) > 0

        # Convert to returns for allocation
        returns_data = {}
        for symbol, df in filtered_data.items():
            close = df["close"].to_numpy()
            rets = np.diff(close) / close[:-1]
            returns_data[symbol] = rets

        returns_df = pd.DataFrame(returns_data)

        # Allocate
        allocator = AssetAllocator(AllocatorConfig())
        result = allocator.allocate(returns_df)

        assert result.is_valid
        assert result.weights.sum() == pytest.approx(1.0, rel=0.01)

    def test_full_pipeline_reproducibility(self, sample_ohlcv_data, tmp_path):
        """Test that full pipeline produces reproducible results."""
        from src.utils.reproducibility import SeedManager

        def run_pipeline_with_seed(seed):
            with SeedManager(seed=seed):
                # Quality check (deterministic)
                settings = MagicMock()
                settings.data_quality.max_missing_rate = 0.05
                settings.data_quality.max_consecutive_missing = 5
                settings.data_quality.price_change_threshold = 0.5
                settings.data_quality.min_volume_threshold = 0
                settings.data_quality.staleness_hours = 48

                from src.data.quality_checker import DataQualityChecker
                checker = DataQualityChecker(settings)
                filtered_data, _ = checker.filter_excluded(sample_ohlcv_data)

                # Returns
                returns_data = {}
                for symbol, df in filtered_data.items():
                    close = df["close"].to_numpy()
                    rets = np.diff(close) / close[:-1]
                    returns_data[symbol] = rets

                returns_df = pd.DataFrame(returns_data)

                # Allocation
                from src.allocation.allocator import AllocatorConfig, AssetAllocator
                allocator = AssetAllocator(AllocatorConfig())
                result = allocator.allocate(returns_df)

                return dict(result.weights)

        weights1 = run_pipeline_with_seed(42)
        weights2 = run_pipeline_with_seed(42)

        for asset in weights1:
            assert weights1[asset] == pytest.approx(weights2[asset], rel=0.0001)


class TestRunInfo:
    """Tests for run info and logging."""

    def test_generate_run_id(self):
        """Test run ID generation."""
        from src.utils.logger import generate_run_id

        run_id1 = generate_run_id()
        run_id2 = generate_run_id()

        assert run_id1 != run_id2
        assert len(run_id1) > 0

    def test_get_run_info(self):
        """Test getting run info."""
        from src.utils.reproducibility import get_run_info

        info = get_run_info(
            run_id="test_123",
            seed=42,
            config={"key": "value"},
        )

        assert info["run_id"] == "test_123"
        assert info["seed"] == 42
        assert "timestamp" in info

    def test_save_and_load_run_info(self, tmp_path):
        """Test saving and loading run info."""
        import json

        from src.utils.reproducibility import get_run_info, save_run_info

        info = get_run_info(run_id="test_456", seed=123, config={})
        path = tmp_path / "run_info.json"

        save_run_info(info, path)

        assert path.exists()

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["run_id"] == "test_456"
        assert loaded["seed"] == 123
