"""
RebalanceTracker Tests

リバランス追跡と予測精度分析のテスト。
"""

import pytest
from datetime import datetime
import pandas as pd

from src.backtest.rebalance_tracker import (
    RebalanceTracker,
    RebalanceRecord,
    ForecastMetrics,
)


class TestRebalanceRecord:
    """RebalanceRecord のテスト"""

    def test_forecast_error_property(self):
        """forecast_error プロパティのテスト"""
        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={"AAPL": 0.3},
            weights_after={"AAPL": 0.4},
            expected_returns={"AAPL": 0.01},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
            actual_return=0.015,
        )

        # 予測誤差 = 0.015 - 0.01 = 0.005
        assert record.forecast_error == pytest.approx(0.005)

    def test_forecast_error_none_when_no_actual(self):
        """実績がない場合の forecast_error"""
        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={"AAPL": 0.3},
            weights_after={"AAPL": 0.4},
            expected_returns={"AAPL": 0.01},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
        )

        assert record.forecast_error is None

    def test_total_transaction_cost(self):
        """total_transaction_cost プロパティのテスト"""
        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001, "GOOG": 0.002, "MSFT": 0.001},
            turnover=0.1,
        )

        assert record.total_transaction_cost == pytest.approx(0.004)

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={"AAPL": 0.3},
            weights_after={"AAPL": 0.4, "GOOG": 0.3},
            expected_returns={"AAPL": 0.01, "GOOG": 0.008},
            expected_portfolio_return=0.009,
            transaction_costs={"AAPL": 0.001, "GOOG": 0.001},
            turnover=0.15,
            actual_return=0.012,
        )

        d = record.to_dict()

        assert d["date"] == "2024-01-31T00:00:00"
        assert d["expected_return"] == 0.009
        assert d["actual_return"] == 0.012
        assert d["forecast_error"] == pytest.approx(0.003)
        assert d["turnover"] == 0.15
        assert d["total_cost"] == pytest.approx(0.002)
        assert d["n_assets"] == 2

    def test_to_detailed_dict(self):
        """to_detailed_dict メソッドのテスト"""
        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={"AAPL": 0.3},
            weights_after={"AAPL": 0.4},
            expected_returns={"AAPL": 0.01},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
            metadata={"reason": "test"},
        )

        d = record.to_detailed_dict()

        assert "weights_before" in d
        assert "weights_after" in d
        assert "expected_returns_per_asset" in d
        assert "transaction_costs" in d
        assert d["metadata"]["reason"] == "test"


class TestForecastMetrics:
    """ForecastMetrics のテスト"""

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        metrics = ForecastMetrics(
            mean_expected=0.01,
            mean_actual=0.012,
            mean_error=0.002,
            std_error=0.005,
            correlation=0.75,
            total_cost=0.05,
            total_turnover=0.30,
            n_rebalances=10,
        )

        d = metrics.to_dict()

        assert d["mean_expected"] == 0.01
        assert d["mean_actual"] == 0.012
        assert d["correlation"] == 0.75
        assert d["n_rebalances"] == 10


class TestRebalanceTracker:
    """RebalanceTracker のテスト"""

    def test_record_rebalance(self):
        """リバランス記録のテスト"""
        tracker = RebalanceTracker()

        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={"AAPL": 0.3},
            weights_after={"AAPL": 0.4},
            expected_returns={"AAPL": 0.01},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
        )

        tracker.record_rebalance(record)

        assert len(tracker) == 1

    def test_update_actual_return(self):
        """実績リターン更新のテスト"""
        tracker = RebalanceTracker()
        date = datetime(2024, 1, 31)

        record = RebalanceRecord(
            date=date,
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        )

        tracker.record_rebalance(record)
        success = tracker.update_actual_return(date, 0.015)

        assert success
        assert tracker.get_records()[0].actual_return == 0.015

    def test_update_actual_return_with_per_asset(self):
        """銘柄別実績リターンも更新"""
        tracker = RebalanceTracker()
        date = datetime(2024, 1, 31)

        record = RebalanceRecord(
            date=date,
            weights_before={},
            weights_after={"AAPL": 0.5, "GOOG": 0.5},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        )

        tracker.record_rebalance(record)
        tracker.update_actual_return(
            date,
            actual_return=0.015,
            actual_returns_per_asset={"AAPL": 0.02, "GOOG": 0.01},
        )

        record = tracker.get_records()[0]
        assert record.actual_returns_per_asset["AAPL"] == 0.02
        assert record.actual_returns_per_asset["GOOG"] == 0.01

    def test_update_actual_return_invalid_date(self):
        """存在しない日付への更新"""
        tracker = RebalanceTracker()

        record = RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        )

        tracker.record_rebalance(record)
        success = tracker.update_actual_return(datetime(2024, 2, 28), 0.015)

        assert not success

    def test_update_previous_actual_return(self):
        """前回リバランスの実績更新"""
        tracker = RebalanceTracker()

        # 2回リバランス
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 2, 29),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.012,
            transaction_costs={},
            turnover=0.15,
        ))

        # 最新リバランス時に前回の実績を記録
        success = tracker.update_previous_actual_return(0.015)

        assert success
        records = tracker.get_records()
        assert records[0].actual_return == 0.015
        assert records[1].actual_return is None

    def test_get_completed_records(self):
        """実績記録済みレコードの取得"""
        tracker = RebalanceTracker()

        # 3回リバランス、2回は実績あり
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
            actual_return=0.012,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 2, 29),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.011,
            transaction_costs={},
            turnover=0.1,
            actual_return=0.008,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 3, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.013,
            transaction_costs={},
            turnover=0.1,
            # actual_return なし
        ))

        completed = tracker.get_completed_records()
        assert len(completed) == 2

    def test_get_forecast_metrics(self):
        """予測精度メトリクスの計算"""
        tracker = RebalanceTracker()

        # 複数のリバランス記録
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
            actual_return=0.012,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 2, 29),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.015,
            transaction_costs={"AAPL": 0.002},
            turnover=0.15,
            actual_return=0.010,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 3, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.008,
            transaction_costs={"AAPL": 0.001},
            turnover=0.08,
            actual_return=0.011,
        ))

        metrics = tracker.get_forecast_metrics()

        assert metrics is not None
        assert metrics.n_rebalances == 3
        assert metrics.mean_expected == pytest.approx((0.01 + 0.015 + 0.008) / 3)
        assert metrics.mean_actual == pytest.approx((0.012 + 0.010 + 0.011) / 3)
        assert metrics.total_cost == pytest.approx(0.004)
        assert metrics.total_turnover == pytest.approx(0.33)

    def test_get_forecast_metrics_no_data(self):
        """データがない場合のメトリクス"""
        tracker = RebalanceTracker()

        metrics = tracker.get_forecast_metrics()

        assert metrics is None

    def test_get_forecast_metrics_no_completed(self):
        """実績データがない場合のメトリクス"""
        tracker = RebalanceTracker()

        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
            # actual_return なし
        ))

        metrics = tracker.get_forecast_metrics()

        assert metrics is None

    def test_to_dataframe(self):
        """DataFrame変換のテスト"""
        tracker = RebalanceTracker()

        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={"AAPL": 0.001},
            turnover=0.1,
            actual_return=0.012,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 2, 29),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.015,
            transaction_costs={"AAPL": 0.002},
            turnover=0.15,
        ))

        df = tracker.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "expected_return" in df.columns
        assert "actual_return" in df.columns
        assert "forecast_error" in df.columns

    def test_clear(self):
        """クリアのテスト"""
        tracker = RebalanceTracker()

        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        ))

        tracker.clear()

        assert len(tracker) == 0

    def test_iteration(self):
        """イテレーションのテスト"""
        tracker = RebalanceTracker()

        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
        ))
        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 2, 29),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.015,
            transaction_costs={},
            turnover=0.15,
        ))

        dates = [r.date for r in tracker]

        assert len(dates) == 2
        assert datetime(2024, 1, 31) in dates
        assert datetime(2024, 2, 29) in dates

    def test_correlation_calculation(self):
        """相関係数の計算テスト"""
        tracker = RebalanceTracker()

        # 完全相関のデータ
        for i in range(5):
            expected = 0.01 * (i + 1)
            actual = 0.01 * (i + 1)  # 同じ値
            tracker.record_rebalance(RebalanceRecord(
                date=datetime(2024, 1, 31 - i * 7),  # 適当な日付
                weights_before={},
                weights_after={},
                expected_returns={},
                expected_portfolio_return=expected,
                transaction_costs={},
                turnover=0.1,
                actual_return=actual,
            ))

        metrics = tracker.get_forecast_metrics()

        # 完全相関なので1.0
        assert metrics.correlation == pytest.approx(1.0, abs=1e-9)

    def test_single_record_correlation(self):
        """1件のみの場合の相関"""
        tracker = RebalanceTracker()

        tracker.record_rebalance(RebalanceRecord(
            date=datetime(2024, 1, 31),
            weights_before={},
            weights_after={},
            expected_returns={},
            expected_portfolio_return=0.01,
            transaction_costs={},
            turnover=0.1,
            actual_return=0.012,
        ))

        metrics = tracker.get_forecast_metrics()

        # 1件では相関計算できないので0
        assert metrics.correlation == 0.0
