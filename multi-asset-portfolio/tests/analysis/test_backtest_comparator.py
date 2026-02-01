"""
Tests for BacktestComparator

バックテスト結果の比較・再現性検証機能のテスト。
"""

import shutil
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from src.analysis.result_store import BacktestResultStore
from src.analysis.backtest_comparator import BacktestComparator, DiffReport
from src.analysis.backtest_archive import ReproducibilityReport
from src.backtest.base import (
    UnifiedBacktestResult,
    UnifiedBacktestConfig,
    RebalanceRecord,
)


@pytest.fixture
def temp_results_dir():
    """一時ディレクトリを作成し、テスト後に削除"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_result():
    """テスト用のUnifiedBacktestResult"""
    config = UnifiedBacktestConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    n_days = len(dates)
    daily_returns = pd.Series([0.001] * n_days, index=dates)
    portfolio_values = pd.Series(
        [100000.0 * (1.001 ** i) for i in range(n_days)],
        index=dates,
    )

    rebalances = [
        RebalanceRecord(
            date=datetime(2020, 1, 31),
            weights_before={},
            weights_after={"AAPL": 0.5, "MSFT": 0.5},
            turnover=1.0,
            transaction_cost=50.0,
            portfolio_value=100000.0,
        ),
    ]

    return UnifiedBacktestResult(
        total_return=0.5,
        annual_return=0.12,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=-0.1,
        volatility=0.15,
        calmar_ratio=1.2,
        win_rate=0.55,
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        rebalances=rebalances,
        total_turnover=1.0,
        total_transaction_costs=50.0,
        config=config,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        engine_name="test_engine",
    )


@pytest.fixture
def different_result():
    """異なるメトリクスを持つテスト用結果"""
    config = UnifiedBacktestConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        rebalance_frequency="weekly",  # 頻度が違う
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    n_days = len(dates)
    daily_returns = pd.Series([0.0008] * n_days, index=dates)
    portfolio_values = pd.Series(
        [100000.0 * (1.0008 ** i) for i in range(n_days)],
        index=dates,
    )

    rebalances = [
        RebalanceRecord(
            date=datetime(2020, 1, 10),
            weights_before={},
            weights_after={"AAPL": 0.4, "MSFT": 0.6},
            turnover=1.0,
            transaction_cost=40.0,
            portfolio_value=100000.0,
        ),
    ]

    return UnifiedBacktestResult(
        total_return=0.35,
        annual_return=0.08,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=-0.12,
        volatility=0.12,
        calmar_ratio=0.67,
        win_rate=0.52,
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        rebalances=rebalances,
        total_turnover=1.0,
        total_transaction_costs=40.0,
        config=config,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        engine_name="test_engine",
    )


class TestBacktestComparator:
    """BacktestComparatorのテスト"""

    def test_compare_two_archives(self, temp_results_dir, sample_result, different_result):
        """2つのアーカイブを比較できる"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Monthly", universe=["AAPL", "MSFT"])
        id2 = store.save(different_result, name="Weekly", universe=["AAPL", "MSFT"])

        result = comparator.compare([id1, id2])

        assert len(result.archive_ids) == 2
        assert len(result.archives) == 2
        assert not result.metric_comparison.empty

        # メトリクス比較
        assert "total_return" in result.metric_comparison.columns
        assert "sharpe_ratio" in result.metric_comparison.columns

    def test_compare_summary(self, temp_results_dir, sample_result, different_result):
        """比較サマリが生成される"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Monthly", universe=["AAPL", "MSFT"])
        id2 = store.save(different_result, name="Weekly", universe=["AAPL", "MSFT"])

        result = comparator.compare([id1, id2])
        summary = result.summary()

        assert "BACKTEST COMPARISON" in summary
        assert "Monthly" in summary or id1[-12:] in summary
        assert "Total Return" in summary

    def test_verify_reproducibility_same_result(self, temp_results_dir, sample_result):
        """同一結果で再現性が確認できる"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        archive_id = store.save(sample_result, name="Original", universe=["AAPL", "MSFT"])

        # 同一結果で検証
        report = comparator.verify_reproducibility(
            archive_id,
            sample_result,
            universe=["AAPL", "MSFT"],
        )

        assert report.is_reproducible
        assert report.config_hash_match
        assert report.universe_hash_match
        assert report.max_diff < 1e-6

    def test_verify_reproducibility_different_result(
        self, temp_results_dir, sample_result, different_result
    ):
        """異なる結果で再現性が失敗する"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        archive_id = store.save(sample_result, name="Original", universe=["AAPL", "MSFT"])

        # 異なる結果で検証
        report = comparator.verify_reproducibility(
            archive_id,
            different_result,
            universe=["AAPL", "MSFT"],
        )

        # 設定が違うので再現性なし
        assert not report.is_reproducible
        assert not report.config_hash_match
        assert len(report.mismatches) > 0

    def test_diff_two_archives(self, temp_results_dir, sample_result, different_result):
        """2つのアーカイブの差分を取得できる"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Monthly", universe=["AAPL", "MSFT"])
        id2 = store.save(different_result, name="Weekly", universe=["AAPL", "MSFT"])

        diff = comparator.diff(id1, id2)

        assert diff.archive_id_1 == id1
        assert diff.archive_id_2 == id2
        assert len(diff.config_diffs) > 0  # rebalance_frequencyが違う
        assert len(diff.metric_diffs) > 0

    def test_diff_summary(self, temp_results_dir, sample_result, different_result):
        """差分サマリが生成される"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Monthly", universe=["AAPL", "MSFT"])
        id2 = store.save(different_result, name="Weekly", universe=["AAPL", "MSFT"])

        diff = comparator.diff(id1, id2)
        summary = diff.summary()

        assert "DIFF REPORT" in summary
        assert "CONFIG DIFFERENCES" in summary
        assert "METRIC DIFFERENCES" in summary

    def test_find_similar(self, temp_results_dir, sample_result, different_result):
        """類似アーカイブを検索できる"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Original", universe=["AAPL", "MSFT"])
        id2 = store.save(different_result, name="Similar", universe=["AAPL", "MSFT"])

        # id1と類似のアーカイブを検索
        similar = comparator.find_similar(
            id1,
            min_return_diff=0.2,  # 20%以内
            min_sharpe_diff=0.5,  # 0.5以内
        )

        assert len(similar) == 1
        assert similar[0]["archive_id"] == id2

    def test_compare_requires_two_archives(self, temp_results_dir, sample_result):
        """比較には2つ以上のアーカイブが必要"""
        store = BacktestResultStore(temp_results_dir)
        comparator = BacktestComparator(store)

        id1 = store.save(sample_result, name="Only One", universe=["AAPL", "MSFT"])

        with pytest.raises(ValueError, match="At least 2"):
            comparator.compare([id1])


class TestReproducibilityReport:
    """ReproducibilityReportのテスト"""

    def test_summary_reproducible(self):
        """再現性ありのレポートサマリ"""
        report = ReproducibilityReport(
            is_reproducible=True,
            config_hash_match=True,
            universe_hash_match=True,
            metric_diffs={"total_return": 0.0, "sharpe_ratio": 0.0},
            max_diff=0.0,
            mismatches=[],
            tolerance=1e-6,
        )

        summary = report.summary()
        assert "Reproducible: Yes" in summary
        assert "Config Hash Match: Yes" in summary

    def test_summary_not_reproducible(self):
        """再現性なしのレポートサマリ"""
        report = ReproducibilityReport(
            is_reproducible=False,
            config_hash_match=False,
            universe_hash_match=True,
            metric_diffs={"total_return": 0.05},
            max_diff=0.05,
            mismatches=["Config hash mismatch"],
            tolerance=1e-6,
        )

        summary = report.summary()
        assert "Reproducible: No" in summary
        assert "MISMATCHES" in summary
        assert "Config hash mismatch" in summary


class TestDiffReport:
    """DiffReportのテスト"""

    def test_summary(self):
        """差分レポートサマリ"""
        diff = DiffReport(
            archive_id_1="bt_20260101_001",
            archive_id_2="bt_20260101_002",
            config_diffs={"rebalance_frequency": ("monthly", "weekly")},
            metric_diffs={"total_return": 0.15, "sharpe_ratio": 0.3},
            timeseries_correlation=0.95,
            timeseries_rmse=0.02,
            rebalance_count_diff=10,
            weight_correlation=0.85,
        )

        summary = diff.summary()
        assert "DIFF REPORT" in summary
        assert "bt_20260101_001" in summary
        assert "rebalance_frequency" in summary
        assert "total_return" in summary
