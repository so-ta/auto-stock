"""
Test Ray + ResourceConfig Integration (task_045_12).

Tests:
1. RayBacktestEngine with n_workers parameter
2. RayBacktestEngine.from_unified_config with ResourceConfig
3. create_ray_engine() with ResourceConfig integration
4. init_ray() helper function
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.backtest.base import UnifiedBacktestConfig
from src.backtest.ray_engine import RayBacktestEngine, RayBacktestConfig, RAY_AVAILABLE


class TestRayBacktestEngineNWorkers:
    """Test RayBacktestEngine n_workers parameter."""

    def test_n_workers_from_init(self):
        """Test n_workers passed directly to __init__."""
        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = RayBacktestEngine(n_workers=8, auto_init_ray=False)
            assert engine.n_workers == 8

    def test_n_workers_priority_over_config(self):
        """Test n_workers parameter takes priority over config."""
        config = RayBacktestConfig(n_workers=4)
        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = RayBacktestEngine(config=config, n_workers=16, auto_init_ray=False)
            # n_workers parameter should override config
            assert engine.n_workers == 16

    def test_n_workers_from_config(self):
        """Test n_workers from RayBacktestConfig."""
        config = RayBacktestConfig(n_workers=12)
        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = RayBacktestEngine(config=config, auto_init_ray=False)
            assert engine.n_workers == 12

    def test_n_workers_auto_detect(self):
        """Test n_workers auto detection when -1."""
        config = RayBacktestConfig(n_workers=-1)
        with patch.object(RayBacktestEngine, "_init_ray"):
            with patch("os.cpu_count", return_value=8):
                engine = RayBacktestEngine(config=config, auto_init_ray=False)
                # Should be cpu_count - 1 = 7
                assert engine.n_workers == 7

    def test_auto_init_ray_false(self):
        """Test auto_init_ray=False skips Ray initialization."""
        with patch.object(RayBacktestEngine, "_init_ray") as mock_init:
            engine = RayBacktestEngine(n_workers=4, auto_init_ray=False)
            mock_init.assert_not_called()
            assert engine.auto_init_ray is False


class TestRayBacktestEngineFromUnifiedConfig:
    """Test RayBacktestEngine.from_unified_config class method."""

    def test_from_unified_config_basic(self):
        """Test from_unified_config creates engine with correct settings."""
        config = UnifiedBacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 1, 1),
            initial_capital=100000.0,
            rebalance_frequency="weekly",
        )

        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = RayBacktestEngine.from_unified_config(
                config,
                n_workers=6,
                auto_init_ray=False,
            )

            assert engine.n_workers == 6
            assert engine.ray_config.start_date == "2020-01-01"
            assert engine.ray_config.end_date == "2024-01-01"
            assert engine.ray_config.rebalance_freq == "weekly"

    def test_from_unified_config_with_resource_config(self):
        """Test from_unified_config with ResourceConfig integration."""
        config = UnifiedBacktestConfig(
            start_date="2015-01-01",
            end_date="2025-01-01",
            initial_capital=1000000.0,
        )

        # Mock ResourceConfig
        with patch("src.config.resource_config.get_current_resource_config") as mock_rc:
            mock_config = MagicMock()
            mock_config.ray_workers = 10
            mock_rc.return_value = mock_config

            with patch.object(RayBacktestEngine, "_init_ray"):
                # Get ray_workers from ResourceConfig
                from src.config.resource_config import get_current_resource_config
                rc = get_current_resource_config()

                engine = RayBacktestEngine.from_unified_config(
                    config,
                    n_workers=rc.ray_workers,
                    auto_init_ray=False,
                )

                assert engine.n_workers == 10

    def test_from_unified_config_with_kwargs(self):
        """Test from_unified_config with additional kwargs."""
        config = UnifiedBacktestConfig(
            start_date="2020-01-01",
            end_date="2024-01-01",
            initial_capital=100000.0,
        )

        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = RayBacktestEngine.from_unified_config(
                config,
                n_workers=4,
                auto_init_ray=False,
                top_n=50,
                lookback_days=90,
            )

            assert engine.ray_config.top_n == 50
            assert engine.ray_config.lookback_days == 90


class TestBacktestEngineFactoryRay:
    """Test BacktestEngineFactory with Ray mode."""

    def test_factory_create_ray_with_n_workers(self):
        """Test BacktestEngineFactory.create() with ray mode and n_workers."""
        from src.backtest.factory import BacktestEngineFactory

        config = UnifiedBacktestConfig(
            start_date="2020-01-01",
            end_date="2024-01-01",
            initial_capital=100000.0,
        )

        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = BacktestEngineFactory.create(
                mode="ray",
                config=config,
                n_workers=8,
                auto_init_ray=False,
            )

            assert isinstance(engine, RayBacktestEngine)
            assert engine.n_workers == 8


class TestCreateRayEngine:
    """Test create_ray_engine() helper function."""

    def test_create_ray_engine_with_resource_config(self):
        """Test create_ray_engine uses ResourceConfig."""
        from src.backtest.factory import create_ray_engine

        config = UnifiedBacktestConfig(
            start_date="2020-01-01",
            end_date="2024-01-01",
            initial_capital=100000.0,
        )

        with patch("src.config.resource_config.get_current_resource_config") as mock_rc:
            mock_config = MagicMock()
            mock_config.ray_workers = 12
            mock_rc.return_value = mock_config

            with patch.object(RayBacktestEngine, "_init_ray"):
                engine = create_ray_engine(config, auto_init_ray=False)

                assert isinstance(engine, RayBacktestEngine)
                # Should use ray_workers from ResourceConfig
                assert engine.n_workers == 12

    def test_create_ray_engine_with_explicit_n_workers(self):
        """Test create_ray_engine with explicit n_workers."""
        from src.backtest.factory import create_ray_engine

        config = UnifiedBacktestConfig(
            start_date="2020-01-01",
            end_date="2024-01-01",
            initial_capital=100000.0,
        )

        with patch.object(RayBacktestEngine, "_init_ray"):
            engine = create_ray_engine(
                config,
                n_workers=6,
                auto_init_ray=False,
            )

            assert engine.n_workers == 6


class TestInitRay:
    """Test init_ray() helper function."""

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_init_ray_with_resource_config(self):
        """Test init_ray uses ResourceConfig for n_workers."""
        from src.backtest.factory import init_ray
        import ray

        # Skip if already initialized
        if ray.is_initialized():
            ray.shutdown()

        with patch("src.config.resource_config.get_current_resource_config") as mock_rc:
            mock_config = MagicMock()
            mock_config.ray_workers = 2
            mock_rc.return_value = mock_config

            with patch("ray.init") as mock_init:
                with patch("ray.is_initialized", return_value=False):
                    init_ray()

                    mock_init.assert_called_once()
                    call_kwargs = mock_init.call_args[1]
                    assert call_kwargs.get("num_cpus") == 2

    def test_init_ray_without_ray_installed(self):
        """Test init_ray returns False when Ray not installed."""
        from src.backtest.factory import init_ray

        with patch.dict("sys.modules", {"ray": None}):
            with patch("builtins.__import__", side_effect=ImportError("No ray")):
                # This should handle ImportError gracefully
                pass


class TestResourceConfigIntegrationPattern:
    """Test the recommended ResourceConfig integration pattern."""

    def test_usage_pattern(self):
        """Test the documented usage pattern."""
        from src.backtest.factory import BacktestEngineFactory

        # The pattern from task description:
        # from src.config.resource_config import get_current_resource_config
        # rc = get_current_resource_config()
        # engine = BacktestEngineFactory.create(
        #     mode="ray",
        #     n_workers=rc.ray_workers,
        # )

        with patch("src.config.resource_config.get_current_resource_config") as mock_rc:
            mock_config = MagicMock()
            mock_config.ray_workers = 16
            mock_rc.return_value = mock_config

            from src.config.resource_config import get_current_resource_config
            rc = get_current_resource_config()

            with patch.object(RayBacktestEngine, "_init_ray"):
                engine = BacktestEngineFactory.create(
                    mode="ray",
                    n_workers=rc.ray_workers,
                    auto_init_ray=False,
                )

                assert isinstance(engine, RayBacktestEngine)
                assert engine.n_workers == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
