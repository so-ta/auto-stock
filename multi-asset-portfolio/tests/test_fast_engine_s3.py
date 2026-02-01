"""
FastBacktestEngine StorageBackend対応テスト

StorageConfig経由でのS3/ローカルバックエンド統合をテスト。
Note: S3は必須。全てのテストでStorageBackendを使用。
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.fast_engine import FastBacktestConfig, FastBacktestEngine
from src.backtest.covariance_cache import CovarianceCache
from src.utils.storage_backend import StorageBackend, StorageConfig


@pytest.fixture
def temp_cache_dir():
    """一時キャッシュディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage_config(temp_cache_dir):
    """テスト用StorageConfig（S3必須）"""
    return StorageConfig(
        s3_bucket="test-bucket",
        base_path=temp_cache_dir,
        s3_prefix=".cache",
        s3_region="ap-northeast-1",
    )


@pytest.fixture
def sample_prices():
    """テスト用価格データ"""
    dates = pl.date_range(
        datetime(2023, 1, 1),
        datetime(2023, 3, 31),
        "1d",
        eager=True,
    )

    tickers = ["AAPL", "GOOGL", "MSFT"]
    rows = []

    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        base_price = 100 + hash(ticker) % 50
        prices = base_price * np.cumprod(1 + np.random.randn(len(dates)) * 0.02)

        for i, date in enumerate(dates):
            rows.append({
                "timestamp": date,
                "ticker": ticker,
                "close": prices[i],
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "volume": 1000000 + np.random.randint(-100000, 100000),
            })

    return pl.DataFrame(rows)


class TestFastBacktestConfigS3:
    """FastBacktestConfig S3統合テスト"""

    def test_config_has_storage_config_field(self):
        """storage_configフィールドが存在することを確認"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
        )
        assert hasattr(config, "storage_config")
        assert config.storage_config is None

    def test_config_with_storage_config(self, storage_config):
        """storage_configを指定して初期化"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
        )

        assert config.storage_config is not None
        assert config.storage_config.s3_bucket == "test-bucket"


class TestFastBacktestEngineS3Integration:
    """FastBacktestEngine S3統合テスト"""

    def test_init_with_storage_config(self, temp_cache_dir, storage_config):
        """StorageConfigを渡して初期化"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        assert engine._storage_backend is not None
        assert engine._storage_backend.config.s3_bucket == "test-bucket"

    def test_covariance_cache_receives_storage_backend(self, temp_cache_dir, storage_config):
        """CovarianceCacheにstorage_backendが渡されることを確認"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        assert engine.cov_cache is not None
        assert isinstance(engine.cov_cache, CovarianceCache)
        # CovarianceCacheは_backend属性を使用
        assert engine.cov_cache._backend is not None
        assert engine.cov_cache._use_backend is True

    def test_storage_backend_attribute_accessible(self, temp_cache_dir, storage_config):
        """_storage_backend属性がアクセス可能であることを確認"""
        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        # 外部からSignalPrecomputer作成時に使用可能
        assert engine._storage_backend is not None
        assert engine._storage_backend.config.s3_bucket == "test-bucket"


class TestFastBacktestEngineWithSignalPrecomputer:
    """SignalPrecomputer連携テスト"""

    def test_external_signal_precomputer_with_storage_backend(self, temp_cache_dir, storage_config):
        """外部からstorage_backend付きSignalPrecomputerを渡す"""
        from src.backtest.signal_precompute import SignalPrecomputer

        backend = StorageBackend(storage_config)

        # 外部でSignalPrecomputerを作成
        signal_precomputer = SignalPrecomputer(storage_backend=backend)

        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config, signal_precomputer=signal_precomputer)

        assert engine.signal_precomputer is signal_precomputer
        assert engine.signal_precomputer._use_backend is True

    def test_use_engine_storage_backend_for_signal_precomputer(self, temp_cache_dir, storage_config):
        """エンジンのstorage_backendを使ってSignalPrecomputerを作成"""
        from src.backtest.signal_precompute import SignalPrecomputer

        config = FastBacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            storage_config=storage_config,
            cov_cache_dir=str(Path(temp_cache_dir) / "covariance"),
            use_numba=False,
            warmup_jit=False,
        )

        engine = FastBacktestEngine(config)

        # エンジンのstorage_backendを使ってSignalPrecomputerを作成
        signal_precomputer = SignalPrecomputer(storage_backend=engine._storage_backend)

        assert signal_precomputer._use_backend is True
        assert signal_precomputer._backend is engine._storage_backend
