"""
SignalPrecomputer StorageBackend対応テスト（task_045_1）

StorageBackend経由でのParquet/JSON読み書きをテスト。
ローカルバックエンドを使用してS3抽象化をテスト。
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.backtest.signal_precompute import SignalPrecomputer
from src.utils.storage_backend import StorageBackend, StorageConfig


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


@pytest.fixture
def temp_cache_dir():
    """一時キャッシュディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestSignalPrecomputerBackendMode:
    """StorageBackend経由での動作テスト"""

    def test_init_with_storage_backend(self, temp_cache_dir):
        """StorageBackendを渡して初期化"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        assert precomputer._use_backend is True
        assert precomputer._backend is backend

    def test_init_legacy_mode(self, temp_cache_dir):
        """cache_dirを渡して初期化（後方互換性）"""
        precomputer = SignalPrecomputer(cache_dir=temp_cache_dir)

        assert precomputer._use_backend is False
        assert precomputer._backend is None
        assert precomputer._cache_dir == Path(temp_cache_dir)

    def test_precompute_all_with_backend(self, temp_cache_dir, sample_prices):
        """StorageBackend経由でシグナルを計算・保存"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [20],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        result = precomputer.precompute_all(sample_prices, config=test_config)

        assert result is True

        # ファイルが作成されたことを確認
        assert backend.exists("momentum_20.parquet")
        assert backend.exists("volatility_20.parquet")
        assert backend.exists("_metadata.json")

    def test_load_signal_with_backend(self, temp_cache_dir, sample_prices):
        """StorageBackend経由でシグナルを読み込み"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        # シグナルを読み込み
        df = precomputer.load_signal("momentum_20")

        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "ticker" in df.columns
        assert "value" in df.columns

    def test_load_signal_with_ticker_filter(self, temp_cache_dir, sample_prices):
        """特定ティッカーでフィルタリング"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        # 特定ティッカーでフィルタリング
        df = precomputer.load_signal("momentum_20", ticker="AAPL")

        assert len(df) > 0
        assert df["ticker"].unique().to_list() == ["AAPL"]

    def test_list_cached_signals_with_backend(self, temp_cache_dir, sample_prices):
        """キャッシュされたシグナル一覧を取得"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20, 60],
            "volatility_periods": [20],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        signals = precomputer.list_cached_signals()

        assert "momentum_20" in signals
        assert "momentum_60" in signals
        assert "volatility_20" in signals

    def test_clear_cache_with_backend(self, temp_cache_dir, sample_prices):
        """キャッシュをクリア"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        # キャッシュが存在することを確認
        assert len(precomputer.list_cached_signals()) > 0

        # クリア
        precomputer.clear_cache()

        # クリアされたことを確認
        assert len(precomputer.list_cached_signals()) == 0

    def test_cache_stats_with_backend(self, temp_cache_dir, sample_prices):
        """キャッシュ統計を取得"""
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)

        precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        stats = precomputer.cache_stats

        assert stats["backend"] == "local"
        assert stats["num_signals"] >= 1
        assert "momentum_20" in stats["signals"]


class TestSignalPrecomputerLegacyMode:
    """後方互換性テスト（cache_dir指定）"""

    def test_precompute_all_legacy(self, temp_cache_dir, sample_prices):
        """cache_dir指定でシグナルを計算・保存"""
        precomputer = SignalPrecomputer(cache_dir=temp_cache_dir)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        result = precomputer.precompute_all(sample_prices, config=test_config)

        assert result is True

        # ファイルが作成されたことを確認
        assert (Path(temp_cache_dir) / "momentum_20.parquet").exists()
        assert (Path(temp_cache_dir) / "_metadata.json").exists()

    def test_load_signal_legacy(self, temp_cache_dir, sample_prices):
        """cache_dir指定でシグナルを読み込み"""
        precomputer = SignalPrecomputer(cache_dir=temp_cache_dir)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        precomputer.precompute_all(sample_prices, config=test_config)

        df = precomputer.load_signal("momentum_20")

        assert len(df) > 0


class TestSignalPrecomputerMixedOperations:
    """BackendモードとLegacyモードの相互運用テスト"""

    def test_backend_reads_legacy_cache(self, temp_cache_dir, sample_prices):
        """Legacy modeで作成したキャッシュをBackend modeで読み込み"""
        # Legacy modeで作成
        legacy_precomputer = SignalPrecomputer(cache_dir=temp_cache_dir)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        legacy_precomputer.precompute_all(sample_prices, config=test_config)

        # Backend modeで読み込み
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)
        backend_precomputer = SignalPrecomputer(storage_backend=backend)

        df = backend_precomputer.load_signal("momentum_20")

        assert len(df) > 0

    def test_legacy_reads_backend_cache(self, temp_cache_dir, sample_prices):
        """Backend modeで作成したキャッシュをLegacy modeで読み込み"""
        # Backend modeで作成
        config = StorageConfig(backend="local", base_path=temp_cache_dir)
        backend = StorageBackend(config)
        backend_precomputer = SignalPrecomputer(storage_backend=backend)

        test_config = {
            "momentum_periods": [20],
            "volatility_periods": [],
            "rsi_periods": [],
            "zscore_periods": [],
            "sharpe_periods": [],
        }

        backend_precomputer.precompute_all(sample_prices, config=test_config)

        # Legacy modeで読み込み
        legacy_precomputer = SignalPrecomputer(cache_dir=temp_cache_dir)

        df = legacy_precomputer.load_signal("momentum_20")

        assert len(df) > 0
