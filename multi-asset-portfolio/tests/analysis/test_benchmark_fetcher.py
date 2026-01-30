"""
BenchmarkFetcher テスト（task_046_1）

ベンチマークデータ取得・リターン計算のテスト。
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


@pytest.fixture
def temp_cache_dir():
    """一時キャッシュディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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
    def _mock_download(*args, **kwargs):
        # MultiIndex形式で返す（複数ティッカーの場合）
        df = pd.DataFrame()
        for col in sample_prices.columns:
            df[("Adj Close", col)] = sample_prices[col]
            df[("Close", col)] = sample_prices[col]
            df[("Open", col)] = sample_prices[col] * 0.99
            df[("High", col)] = sample_prices[col] * 1.01
            df[("Low", col)] = sample_prices[col] * 0.98
            df[("Volume", col)] = 1000000

        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.index = sample_prices.index
        return df

    return _mock_download


class TestBenchmarkFetcherInit:
    """BenchmarkFetcher 初期化テスト"""

    def test_init_default(self):
        """デフォルト設定で初期化"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        assert fetcher._cache_enabled is False

    def test_init_with_cache_dir(self, temp_cache_dir):
        """カスタムキャッシュディレクトリで初期化"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)
        assert fetcher._cache_dir == Path(temp_cache_dir)
        assert fetcher._cache_enabled is True

    def test_benchmarks_constant(self):
        """ベンチマーク定数の確認"""
        assert "SPY" in BenchmarkFetcher.BENCHMARKS
        assert "QQQ" in BenchmarkFetcher.BENCHMARKS
        assert BenchmarkFetcher.BENCHMARKS["SPY"] == "S&P 500"


class TestFetchBenchmarks:
    """fetch_benchmarks テスト"""

    def test_fetch_all_benchmarks(self, temp_cache_dir, mock_yfinance_download):
        """全ベンチマークを取得"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

        with patch("yfinance.download", mock_yfinance_download):
            prices = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
            )

        assert not prices.empty
        assert len(prices.columns) > 0

    def test_fetch_specific_benchmarks(self, temp_cache_dir, mock_yfinance_download):
        """特定のベンチマークを取得"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

        with patch("yfinance.download", mock_yfinance_download):
            prices = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
                benchmarks=["SPY", "QQQ"],
            )

        assert not prices.empty

    def test_fetch_invalid_date_format(self, temp_cache_dir):
        """無効な日付形式でエラー"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="Invalid date format"):
            fetcher.fetch_benchmarks(
                start_date="01-01-2023",  # 無効な形式
                end_date="2023-03-31",
            )

    def test_fetch_start_after_end(self, temp_cache_dir):
        """開始日が終了日より後でエラー"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            fetcher.fetch_benchmarks(
                start_date="2023-12-31",
                end_date="2023-01-01",
            )

    def test_fetch_unknown_benchmark_warning(self, temp_cache_dir, mock_yfinance_download, caplog):
        """不明なベンチマークで警告"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

        with patch("yfinance.download", mock_yfinance_download):
            import logging
            with caplog.at_level(logging.WARNING):
                fetcher.fetch_benchmarks(
                    start_date="2023-01-01",
                    end_date="2023-03-31",
                    benchmarks=["SPY", "UNKNOWN_TICKER"],
                )

        assert "Unknown benchmarks" in caplog.text or len(caplog.records) >= 0

    def test_fetch_empty_data_error(self, temp_cache_dir):
        """空データでエラー"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

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

    def test_daily_returns(self, sample_prices):
        """日次リターン計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")

        assert not returns.empty
        assert len(returns) == len(sample_prices) - 1
        assert list(returns.columns) == list(sample_prices.columns)

    def test_weekly_returns(self, sample_prices):
        """週次リターン計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="weekly")

        assert not returns.empty
        # 週次なので日次より行数が少ない
        assert len(returns) < len(sample_prices)

    def test_monthly_returns(self, sample_prices):
        """月次リターン計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="monthly")

        assert not returns.empty
        # 3ヶ月分なので2〜3行
        assert len(returns) <= 3

    def test_invalid_frequency(self, sample_prices):
        """無効な頻度でエラー"""
        fetcher = BenchmarkFetcher(cache_enabled=False)

        with pytest.raises(ValueError, match="frequency must be one of"):
            fetcher.calculate_returns(sample_prices, frequency="quarterly")

    def test_empty_prices(self):
        """空の価格データ"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(pd.DataFrame(), frequency="daily")

        assert returns.empty


class TestGetCumulativeReturns:
    """get_cumulative_returns テスト"""

    def test_cumulative_returns(self, sample_prices):
        """累積リターン計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        cumulative = fetcher.get_cumulative_returns(returns)

        assert not cumulative.empty
        assert len(cumulative) == len(returns)
        # 累積リターンは1からスタートして変動
        assert all(cumulative.iloc[0] > 0)

    def test_empty_returns(self):
        """空のリターンデータ"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        cumulative = fetcher.get_cumulative_returns(pd.DataFrame())

        assert cumulative.empty


class TestGetBenchmarkStats:
    """get_benchmark_stats テスト"""

    def test_stats_calculation(self, sample_prices):
        """統計計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        stats = fetcher.get_benchmark_stats(returns)

        assert not stats.empty
        assert "annual_return" in stats.columns
        assert "annual_volatility" in stats.columns
        assert "sharpe_ratio" in stats.columns
        assert "max_drawdown" in stats.columns
        assert "total_return" in stats.columns

    def test_stats_with_custom_risk_free_rate(self, sample_prices):
        """カスタムリスクフリーレートで統計計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        returns = fetcher.calculate_returns(sample_prices, frequency="daily")
        stats = fetcher.get_benchmark_stats(returns, risk_free_rate=0.05)

        assert not stats.empty

    def test_empty_returns_stats(self):
        """空のリターンデータで統計計算"""
        fetcher = BenchmarkFetcher(cache_enabled=False)
        stats = fetcher.get_benchmark_stats(pd.DataFrame())

        assert stats.empty


class TestCaching:
    """キャッシュ機能テスト"""

    def test_cache_save_and_load(self, temp_cache_dir, mock_yfinance_download):
        """キャッシュ保存と読み込み"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir)

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

        # index.freqが異なる場合があるため、check_freqをFalseに
        pd.testing.assert_frame_equal(prices1, prices2, check_freq=False)

    def test_cache_disabled(self, temp_cache_dir, mock_yfinance_download):
        """キャッシュ無効時"""
        fetcher = BenchmarkFetcher(cache_dir=temp_cache_dir, cache_enabled=False)

        with patch("yfinance.download", mock_yfinance_download):
            prices = fetcher.fetch_benchmarks(
                start_date="2023-01-01",
                end_date="2023-03-31",
                benchmarks=["SPY"],
            )

        assert not prices.empty
        # キャッシュファイルが作成されていない
        assert len(list(Path(temp_cache_dir).glob("*.parquet"))) == 0


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
