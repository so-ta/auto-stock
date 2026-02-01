"""
UnifiedCacheManager StorageBackend対応テスト

StorageConfig経由でのS3/ローカルバックエンド統合をテスト。
Note: S3は必須。全てのテストでStorageBackendを使用。
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.utils.cache_manager import (
    UnifiedCacheManager,
    CacheType,
    DataCacheAdapter,
    CachePolicy,
)
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


class TestUnifiedCacheManagerS3Integration:
    """UnifiedCacheManager S3統合テスト"""

    def test_init_with_storage_config(self, temp_cache_dir, storage_config):
        """StorageConfigを渡して初期化"""
        manager = UnifiedCacheManager(
            cache_base_dir=temp_cache_dir,
            storage_config=storage_config,
        )

        assert manager._storage_backend is not None
        assert manager._storage_backend.config.s3_bucket == "test-bucket"

    def test_data_cache_receives_storage_backend(self, temp_cache_dir, storage_config):
        """DataCacheAdapterにstorage_backendが渡されることを確認"""
        manager = UnifiedCacheManager(
            cache_base_dir=temp_cache_dir,
            storage_config=storage_config,
        )

        # デフォルトキャッシュを初期化
        data_cache = manager.get_cache(CacheType.DATA)

        assert data_cache is not None
        assert isinstance(data_cache, DataCacheAdapter)
        assert data_cache._storage_backend is not None

    def test_get_all_stats_with_storage_config(self, temp_cache_dir, storage_config):
        """StorageConfig有りで全統計を取得"""
        manager = UnifiedCacheManager(
            cache_base_dir=temp_cache_dir,
            storage_config=storage_config,
        )

        stats = manager.get_all_stats()

        assert len(stats) > 0
        assert CacheType.DATA.value in stats

    def test_clear_all_with_storage_config(self, temp_cache_dir, storage_config):
        """StorageConfig有りで全キャッシュをクリア"""
        manager = UnifiedCacheManager(
            cache_base_dir=temp_cache_dir,
            storage_config=storage_config,
        )

        # キャッシュを初期化
        _ = manager.get_cache(CacheType.DATA)

        # クリア
        cleared = manager.clear_all()

        assert cleared >= 0

    def test_list_caches(self, temp_cache_dir, storage_config):
        """キャッシュ一覧取得"""
        manager = UnifiedCacheManager(
            cache_base_dir=temp_cache_dir,
            storage_config=storage_config,
        )

        caches = manager.list_caches()

        assert CacheType.SIGNAL.value in caches
        assert CacheType.DATA.value in caches
        assert CacheType.DATAFRAME.value in caches
        assert CacheType.LRU.value in caches


class TestDataCacheAdapterS3:
    """DataCacheAdapter S3統合テスト"""

    def test_init_with_storage_backend(self, temp_cache_dir, storage_config):
        """storage_backendを渡して初期化"""
        backend = StorageBackend(storage_config)

        adapter = DataCacheAdapter(
            name="test_data_cache",
            cache_dir=temp_cache_dir,
            storage_backend=backend,
        )

        assert adapter._storage_backend is backend

    def test_get_stats_with_storage_backend(self, temp_cache_dir, storage_config):
        """storage_backend有りで統計を取得"""
        backend = StorageBackend(storage_config)

        adapter = DataCacheAdapter(
            name="test_data_cache",
            cache_dir=temp_cache_dir,
            storage_backend=backend,
        )

        stats = adapter.get_stats()

        assert stats.name == "test_data_cache"
        assert stats.cache_type == CacheType.DATA
