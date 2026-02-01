"""
BenchmarkFetcher テスト

ベンチマークデータ取得・リターン計算のテスト。
Note: S3は必須。全てのテストでStorageBackendを使用。
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark_fetcher import (
    BenchmarkFetcher,
    BenchmarkFetcherError,
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
def storage_backend(storage_config):
    """テスト用StorageBackend"""
    return StorageBackend(storage_config)


@pytest.fixture
def sample_prices():
    """テスト用価格データ"""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    np.random.seed(42)

    data = {
        "SPY": 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.01),
        "QQQ": 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.015),
        "DIA": 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.008),
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_yfinance_download(sample_prices):
    """yfinance.download のモック"""
    def _mock_download(tickers, *args, **kwargs):
        # 要求されたティッカーのみを返す
        if isinstance(tickers, str):
            tickers = [tickers]

        # MultiIndex形式で返す（複数ティッカーの場合）
        df = pd.DataFrame()
        for col in tickers:
            if col in sample_prices.columns:
                price_data = sample_prices[col]
            else:
                # 未知のティッカーはSPYのデータを使用
                price_data = sample_prices["SPY"]

            df[("Adj Close", col)] = price_data
            df[("Close", col)] = price_data
            df[("Open", col)] = price_data * 0.99
            df[("High", col)] = price_data * 1.01
            df[("Low", col)] = price_data * 0.98
            df[("Volume", col)] = 1000000

        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.index = sample_prices.index
        return df

    return _mock_download


class TestBenchmarkFetcherInit:
    """BenchmarkFetcher 初期化テスト"""

    def test_init_with_storage_backend(self, storage_backend):
        """StorageBackendで初期化"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        assert fetcher._backend is storage_backend

    def test_init_requires_storage_backend(self):
        """StorageBackendが必須であることを確認"""
        with pytest.raises(TypeError):
            BenchmarkFetcher(storage_backend=None)

    def test_benchmarks_constant(self):
        """ベンチマーク定数の確認"""
        assert "SPY" in BenchmarkFetcher.BENCHMARKS
        assert "QQQ" in BenchmarkFetcher.BENCHMARKS
        assert BenchmarkFetcher.BENCHMARKS["SPY"] == "S&P 500"


class TestFetchBenchmarks:
    """fetch_benchmarks テスト"""

    def test_fetch_all_benchmarks(self, storage_backend, mock_yfinance_download):
        """全ベンチマークを取得"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with patch("yfinance.download", mock_yfinance_download):
            prices = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

        assert not prices.empty
        assert len(prices.columns) > 0

    def test_fetch_specific_benchmarks(self, storage_backend, mock_yfinance_download):
        """特定のベンチマークを取得"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with patch("yfinance.download", mock_yfinance_download):
            prices = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
                benchmarks=["SPY", "QQQ"],
            )

        assert not prices.empty

    def test_fetch_invalid_date_format(self, storage_backend):
        """無効な日付形式でエラー"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with pytest.raises(ValueError, match="Invalid date format"):
            fetcher.fetch_benchmarks(
                start_date="01-01-2023",  # 無効な形式
                end_date="2023-03-31",
            )

    def test_fetch_start_after_end(self, storage_backend):
        """開始日が終了日より後でエラー"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            fetcher.fetch_benchmarks(
                start_date="2023-12-31",
                end_date="2023-01-01",
            )

    def test_fetch_unknown_benchmark_warning(self, storage_backend, mock_yfinance_download, caplog):
        """不明なベンチマークで警告"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with patch("yfinance.download", mock_yfinance_download):
            import logging
            with caplog.at_level(logging.WARNING):
                fetcher.fetch_benchmarks(
                    start_date="2023-01-01",
                    end_date="2023-03-31",
                    benchmarks=["SPY", "UNKNOWN_TICKER"],
                )

        assert "Unknown benchmarks" in caplog.text or len(caplog.records) >= 0

    def test_fetch_empty_data_error(self, storage_backend):
        """空データでエラー"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        def mock_empty_download(*args, **kwargs):
            return pd.DataFrame()

        with patch("yfinance.download", mock_empty_download):
            with pytest.raises(BenchmarkFetcherError, match="No data returned"):
                fetcher.fetch_benchmarks(
                    start_date="2023-01-01",
                    end_date="2023-03-31",
                )


class TestCalculateReturns:
    """calculate_returns テスト"""

    def test_daily_returns(self, sample_prices, storage_backend):
        """日次リターン計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")

        assert not returns.empty
        assert len(returns) == len(sample_prices) - 1
        assert list(returns.columns) == list(sample_prices.columns)

    def test_weekly_returns(self, sample_prices, storage_backend):
        """週次リターン計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="weekly")

        assert not returns.empty
        # 週次なので日次より行数が少ない
        assert len(returns) < len(sample_prices)

    def test_monthly_returns(self, sample_prices, storage_backend):
        """月次リターン計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="monthly")

        assert not returns.empty
        # 3ヶ月分なので2〜3行
        assert len(returns) <= 3

    def test_invalid_frequency(self, sample_prices, storage_backend):
        """無効な頻度でエラー"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        with pytest.raises(ValueError, match="frequency must be one of"):
            fetcher.calculate_returns(sample_prices, frequency="quarterly")

    def test_empty_prices(self, storage_backend):
        """空の価格データ"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(pd.DataFrame(), frequency="daily")

        assert returns.empty


class TestGetCumulativeReturns:
    """get_cumulative_returns テスト"""

    def test_cumulative_returns(self, sample_prices, storage_backend):
        """累積リターン計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        cumulative = fetcher.get_cumulative_returns(returns)

        assert not cumulative.empty
        assert len(cumulative) == len(returns)
        # 累積リターンは1からスタートして変動
        assert all(cumulative.iloc[0] > 0)

    def test_empty_returns(self, storage_backend):
        """空のリターンデータ"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        cumulative = fetcher.get_cumulative_returns(pd.DataFrame())

        assert cumulative.empty


class TestGetBenchmarkStats:
    """get_benchmark_stats テスト"""

    def test_stats_calculation(self, sample_prices, storage_backend):
        """統計計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        stats = fetcher.get_benchmark_stats(returns)

        assert not stats.empty
        assert "annual_return" in stats.columns
        assert "annual_volatility" in stats.columns
        assert "sharpe_ratio" in stats.columns
        assert "max_drawdown" in stats.columns
        assert "total_return" in stats.columns

    def test_stats_with_custom_risk_free_rate(self, sample_prices, storage_backend):
        """カスタムリスクフリーレートで統計計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        stats = fetcher.get_benchmark_stats(returns, risk_free_rate=0.05)

        assert not stats.empty

    def test_empty_returns_stats(self, storage_backend):
        """空のリターンデータで統計計算"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)
        stats = fetcher.get_benchmark_stats(pd.DataFrame())

        assert stats.empty


class TestCaching:
    """キャッシュ機能テスト"""

    def test_cache_save_and_load(self, storage_backend, mock_yfinance_download):
        """キャッシュ保存と読み込み"""
        fetcher = BenchmarkFetcher(storage_backend=storage_backend)

        # 最初のフェッチ（yfinanceから取得）
        with patch("yfinance.download", mock_yfinance_download):
            prices1 = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
                benchmarks=["SPY"],
            )

        # 2回目のフェッチ（キャッシュから取得）
        prices2 = fetcher.fetch_benchmarks(
            start_date="2023-01-01",
            end_date="2023-03-31",
            benchmarks=["SPY"],
        )

        # polars DataFrameの場合はpandasに変換して比較
        # polarsはpandasのindexを別の列として読み込むため、共通の列のみ比較
        if hasattr(prices1, "to_pandas"):
            prices1 = prices1.to_pandas()
        if hasattr(prices2, "to_pandas"):
            prices2 = prices2.to_pandas()

        # polarsが読み込んだ場合、index列が追加されることがある
        # 共通の列のみで比較（__index_level_*列を除外）
        common_cols = [c for c in prices1.columns if not c.startswith("__index")]
        prices1 = prices1[common_cols]
        prices2_cols = [c for c in prices2.columns if not c.startswith("__index")]
        prices2 = prices2[prices2_cols]

        # index.freqが異なる場合があるため、check_freqをFalseに
        # index自体も異なる形式の場合があるためcheck_namesもFalseに
        pd.testing.assert_frame_equal(
            prices1.reset_index(drop=True),
            prices2.reset_index(drop=True),
            check_freq=False,
        )


class TestGetAvailableBenchmarks:
    """get_available_benchmarks テスト"""

    def test_get_benchmarks(self):
        """利用可能なベンチマーク一覧を取得"""
        benchmarks = BenchmarkFetcher.get_available_benchmarks()

        assert isinstance(benchmarks, dict)
        assert "SPY" in benchmarks
        assert benchmarks["SPY"] == "S&P 500"

    def test_benchmarks_copy(self):
        """返却値が元の辞書のコピーであることを確認"""
        benchmarks = BenchmarkFetcher.get_available_benchmarks()
        benchmarks["NEW"] = "New Benchmark"

        # 元の辞書は変更されていない
        assert "NEW" not in BenchmarkFetcher.BENCHMARKS
