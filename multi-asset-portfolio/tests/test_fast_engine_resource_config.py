"""
FastBacktestEngine ResourceConfig統合テスト（task_045_10）

ResourceConfigからのFastBacktestConfig生成をテスト。
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.backtest.fast_engine import FastBacktestConfig, FastBacktestEngine
from src.config.resource_config import (
    ResourceConfig,
    get_current_resource_config,
    init_resource_config,
)


@pytest.fixture
def temp_cache_dir():
    """一時キャッシュディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_resource_config():
    """モックResourceConfig"""
    return ResourceConfig(
        use_numba=True,
        numba_parallel=True,
        use_gpu=False,
        max_workers=8,
    )


class TestFastBacktestConfigFromResourceConfig:
    """FastBacktestConfig.from_resource_config()テスト"""

    def test_from_resource_config_basic(self, temp_cache_dir):
        """基本的なfrom_resource_config()テスト"""
        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
        )

        assert config.start_date == datetime(2023, 1, 1)
        assert config.end_date == datetime(2023, 12, 31)
        # ResourceConfigからの設定が適用されている
        assert config.use_numba is True
        assert config.numba_parallel is True

    def test_from_resource_config_with_override(self, temp_cache_dir):
        """オーバーライド付きfrom_resource_config()テスト"""
        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            rebalance_frequency="weekly",
            initial_capital=500000.0,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
        )

        assert config.rebalance_frequency == "weekly"
        assert config.initial_capital == 500000.0

    def test_from_resource_config_numba_settings(self, temp_cache_dir):
        """Numba設定がResourceConfigから取得されることを確認"""
        rc = get_current_resource_config()

        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
        )

        # ResourceConfigの値と一致
        assert config.use_numba == rc.use_numba
        assert config.numba_parallel == rc.numba_parallel
        assert config.use_gpu == rc.use_gpu

    def test_from_resource_config_can_override_numba(self, temp_cache_dir):
        """Numba設定をオーバーライドできることを確認"""
        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            use_numba=False,
            numba_parallel=False,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
        )

        assert config.use_numba is False
        assert config.numba_parallel is False


class TestFastBacktestConfigNumbaDefaults:
    """FastBacktestConfigのNumbaデフォルト値テスト"""

    def test_numba_parallel_default_is_true(self):
        """numba_parallelのデフォルトがTrueであることを確認"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        assert config.numba_parallel is True
        assert config.use_numba is True

    def test_explicit_numba_settings(self):
        """明示的なNumba設定"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            use_numba=False,
            numba_parallel=False,
        )

        assert config.use_numba is False
        assert config.numba_parallel is False


class TestFastBacktestEngineWithResourceConfig:
    """FastBacktestEngineとResourceConfigの統合テスト"""

    def test_engine_with_resource_config(self, temp_cache_dir):
        """ResourceConfigベースの設定でエンジンを初期化"""
        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,  # テスト環境ではNumba無効化
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        assert engine.config.start_date == datetime(2023, 1, 1)
        assert engine.config.end_date == datetime(2023, 3, 31)

    def test_engine_inherits_resource_config_settings(self, temp_cache_dir):
        """エンジンがResourceConfig設定を継承していることを確認"""
        config = FastBacktestConfig.from_resource_config(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        # 設定が正しく伝播していることを確認
        assert engine.config.use_numba is False
        assert engine.config.warmup_jit is False


class TestResourceConfigIntegration:
    """ResourceConfigシステム統合テスト"""

    def test_resource_config_singleton(self):
        """ResourceConfigがシングルトンとして動作することを確認"""
        rc1 = get_current_resource_config()
        rc2 = get_current_resource_config()

        # 同じインスタンスが返される
        assert rc1 is rc2

    def test_resource_config_has_numba_settings(self):
        """ResourceConfigにNumba設定が含まれていることを確認"""
        rc = get_current_resource_config()

        assert hasattr(rc, "use_numba")
        assert hasattr(rc, "numba_parallel")
        assert hasattr(rc, "use_gpu")

    def test_resource_config_default_values(self):
        """ResourceConfigのデフォルト値を確認"""
        rc = ResourceConfig()

        assert rc.use_numba is True
        assert rc.numba_parallel is True
        assert rc.use_gpu is False
