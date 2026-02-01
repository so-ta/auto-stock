"""
NetProfitAllocator Tests

ネット期待利益ベースのアロケータのテスト。
"""

import pytest
import pandas as pd
import numpy as np

from src.allocation.net_profit_allocator import (
    NetProfitAllocator,
    NetProfitAllocationResult,
    create_net_profit_allocator,
)


class TestNetProfitAllocationResult:
    """NetProfitAllocationResult のテスト"""

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        result = NetProfitAllocationResult(
            weights={"AAPL": 0.5, "GOOG": 0.3, "CASH": 0.2},
            expected_returns={"AAPL": 0.01, "GOOG": 0.008},
            net_profits={"AAPL": 0.009, "GOOG": 0.007},
            transaction_costs={"AAPL": 0.001, "GOOG": 0.001},
            turnover=0.15,
            expected_portfolio_return=0.0074,
            metadata={"reason": "normal"},
        )

        d = result.to_dict()

        assert d["weights"]["AAPL"] == 0.5
        assert d["turnover"] == 0.15
        assert d["metadata"]["reason"] == "normal"


class TestNetProfitAllocator:
    """NetProfitAllocator のテスト"""

    def test_init_default_params(self):
        """デフォルトパラメータでの初期化"""
        allocator = NetProfitAllocator()

        assert allocator.transaction_cost_bps == 20.0
        assert allocator.max_weight == 0.20
        assert allocator.min_net_profit == 0.0
        assert allocator.allocation_method == "proportional"
        assert allocator.periods_per_year == 12

    def test_init_custom_params(self):
        """カスタムパラメータでの初期化"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=30.0,
            max_weight=0.15,
            min_net_profit=0.001,
            allocation_method="equal",
            periods_per_year=52,
        )

        assert allocator.transaction_cost_bps == 30.0
        assert allocator.max_weight == 0.15
        assert allocator.min_net_profit == 0.001
        assert allocator.allocation_method == "equal"
        assert allocator.periods_per_year == 52

    def test_init_invalid_params(self):
        """無効なパラメータでの初期化エラー"""
        with pytest.raises(ValueError):
            NetProfitAllocator(transaction_cost_bps=-1)

        with pytest.raises(ValueError):
            NetProfitAllocator(max_weight=1.5)

        with pytest.raises(ValueError):
            NetProfitAllocator(max_weight=0)

        with pytest.raises(ValueError):
            NetProfitAllocator(allocation_method="invalid")

        with pytest.raises(ValueError):
            NetProfitAllocator(periods_per_year=0)

    def test_allocate_basic(self):
        """基本的な配分テスト"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=20.0,
            max_weight=0.20,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,  # 年率12%
            "GOOG": 0.10,  # 年率10%
            "MSFT": 0.08,  # 年率8%
        })

        result = allocator.allocate(expected_returns)

        # 全銘柄が選択されるはず
        assert "AAPL" in result.weights
        assert "GOOG" in result.weights
        assert "MSFT" in result.weights

        # ウェイトは max_weight を超えない
        for symbol, weight in result.weights.items():
            if symbol != "CASH":
                assert weight <= 0.20 + 1e-9

        # ウェイトの合計は1
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    def test_allocate_with_dict(self):
        """dict形式の期待リターンでの配分"""
        allocator = NetProfitAllocator()

        expected_returns = {
            "AAPL": 0.12,
            "GOOG": 0.10,
        }

        result = allocator.allocate(expected_returns)

        assert result is not None
        assert len(result.weights) > 0

    def test_allocate_with_current_weights(self):
        """現在ウェイトを考慮した配分"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=20.0,
            max_weight=0.30,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "GOOG": 0.10,
        })

        current_weights = {
            "AAPL": 0.5,  # 既に大きなポジション
            "MSFT": 0.3,  # 売却対象
        }

        result = allocator.allocate(expected_returns, current_weights)

        # 結果が返る
        assert result is not None
        # ターンオーバーが計算される
        assert result.turnover >= 0

    def test_allocate_all_negative_returns(self):
        """全銘柄がマイナス期待リターンの場合"""
        allocator = NetProfitAllocator()

        expected_returns = pd.Series({
            "AAPL": -0.05,
            "GOOG": -0.10,
        })

        result = allocator.allocate(expected_returns)

        # 現金100%になるはず
        assert "CASH" in result.weights
        assert result.weights["CASH"] == 1.0
        assert result.metadata.get("reason") == "all_below_threshold"

    def test_allocate_empty_returns(self):
        """空の期待リターンの場合"""
        allocator = NetProfitAllocator()

        expected_returns = pd.Series({})

        result = allocator.allocate(expected_returns)

        # 現金100%になるはず
        assert result.weights == {"CASH": 1.0}
        assert result.metadata.get("reason") == "empty_expected_returns"

    def test_allocate_equal_method(self):
        """均等配分メソッドのテスト"""
        allocator = NetProfitAllocator(
            max_weight=0.25,
            allocation_method="equal",
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "GOOG": 0.10,
            "MSFT": 0.08,
            "AMZN": 0.06,
        })

        result = allocator.allocate(expected_returns)

        # 均等配分（4銘柄で25%ずつ、max_weightで制限）
        non_cash_weights = {k: v for k, v in result.weights.items() if k != "CASH"}
        weights_list = list(non_cash_weights.values())

        # 全て同じウェイトか、max_weightで制限されている
        assert all(w <= 0.25 + 1e-9 for w in weights_list)

    def test_allocate_proportional_method(self):
        """比例配分メソッドのテスト"""
        allocator = NetProfitAllocator(
            max_weight=0.50,  # 大きめに設定
            allocation_method="proportional",
        )

        expected_returns = pd.Series({
            "AAPL": 0.20,  # 高い期待リターン
            "GOOG": 0.05,  # 低い期待リターン
        })

        result = allocator.allocate(expected_returns)

        # AAPLがGOOGより大きなウェイトを持つはず
        assert result.weights.get("AAPL", 0) > result.weights.get("GOOG", 0)

    def test_allocate_max_weight_constraint(self):
        """max_weight 制約のテスト"""
        allocator = NetProfitAllocator(
            max_weight=0.10,  # 厳しい制約
        )

        expected_returns = pd.Series({
            "AAPL": 0.50,  # 非常に高い期待リターン
            "GOOG": 0.01,  # 低い期待リターン
        })

        result = allocator.allocate(expected_returns)

        # max_weight を超えない
        for symbol, weight in result.weights.items():
            if symbol != "CASH":
                assert weight <= 0.10 + 1e-9

    def test_calculate_net_profit(self):
        """単一銘柄のネット期待利益計算"""
        allocator = NetProfitAllocator(transaction_cost_bps=20.0)

        net_profit, cost = allocator.calculate_net_profit(
            expected_return=0.01,  # 1%
            current_weight=0.0,
            target_weight=0.20,  # 20%取得
        )

        # コスト = 0.20 * 0.002 = 0.0004 (4bps)
        assert abs(cost - 0.0004) < 1e-9
        # ネット期待利益 = 0.01 - 0.0004 = 0.0096
        assert abs(net_profit - 0.0096) < 1e-9

    def test_transaction_cost_calculation(self):
        """取引コスト計算の正確性"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=50.0,  # 50bps = 0.5%
            max_weight=0.30,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
        })

        current_weights = {
            "AAPL": 0.10,  # 10%から30%に増加
        }

        result = allocator.allocate(expected_returns, current_weights)

        # 取引コストが計算される
        assert "AAPL" in result.transaction_costs
        aapl_cost = result.transaction_costs["AAPL"]

        # コスト = |target - current| * 0.005
        # 正確なtargetはアルゴリズムによるが、コストは正の値
        assert aapl_cost >= 0

    def test_turnover_calculation(self):
        """ターンオーバー計算の正確性"""
        allocator = NetProfitAllocator(
            max_weight=0.50,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "GOOG": 0.10,
        })

        current_weights = {
            "AAPL": 0.30,
            "MSFT": 0.40,  # 売却対象
        }

        result = allocator.allocate(expected_returns, current_weights)

        # ターンオーバーは0以上1以下
        assert 0 <= result.turnover <= 1

    def test_expected_portfolio_return(self):
        """ポートフォリオ期待リターンの計算"""
        allocator = NetProfitAllocator(
            max_weight=1.0,  # 制約なし
            allocation_method="equal",
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,  # 年率12% → 月次1%
            "GOOG": 0.12,  # 年率12% → 月次1%
        })

        result = allocator.allocate(expected_returns)

        # 期待ポートフォリオリターンが計算される
        assert result.expected_portfolio_return > 0


class TestCreateNetProfitAllocator:
    """ファクトリ関数のテスト"""

    def test_create_with_defaults(self):
        """デフォルトパラメータでの生成"""
        allocator = create_net_profit_allocator()

        assert isinstance(allocator, NetProfitAllocator)
        assert allocator.transaction_cost_bps == 20.0

    def test_create_with_custom_params(self):
        """カスタムパラメータでの生成"""
        allocator = create_net_profit_allocator(
            transaction_cost_bps=30.0,
            max_weight=0.15,
            allocation_method="equal",
        )

        assert allocator.transaction_cost_bps == 30.0
        assert allocator.max_weight == 0.15
        assert allocator.allocation_method == "equal"


class TestNetProfitAllocatorEdgeCases:
    """エッジケースのテスト"""

    def test_single_asset(self):
        """単一銘柄のみの場合"""
        allocator = NetProfitAllocator(max_weight=0.50)

        expected_returns = pd.Series({"AAPL": 0.12})

        result = allocator.allocate(expected_returns)

        # 1銘柄でもmax_weight制約が適用される
        assert result.weights.get("AAPL", 0) <= 0.50 + 1e-9
        # 残りはCASH
        assert "CASH" in result.weights or result.weights.get("AAPL") == 0.50

    def test_very_small_returns(self):
        """非常に小さい期待リターンの場合"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=20.0,
            min_net_profit=0.0001,  # 1bp以上必要
        )

        expected_returns = pd.Series({
            "AAPL": 0.0001,  # 非常に小さい
            "GOOG": 0.0002,  # 非常に小さい
        })

        result = allocator.allocate(expected_returns)

        # 取引コストで相殺されてCASHになる可能性が高い
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_large_number_of_assets(self):
        """多数の銘柄がある場合"""
        allocator = NetProfitAllocator(max_weight=0.05)

        # 50銘柄
        symbols = [f"STOCK_{i}" for i in range(50)]
        expected_returns = pd.Series(
            {s: 0.10 + 0.001 * i for i, s in enumerate(symbols)}
        )

        result = allocator.allocate(expected_returns)

        # 全てmax_weight以下
        for symbol, weight in result.weights.items():
            if symbol != "CASH":
                assert weight <= 0.05 + 1e-9

        # 合計は1
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_zero_transaction_cost(self):
        """取引コストゼロの場合"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=0.0,
            max_weight=0.50,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "GOOG": 0.10,
        })

        result = allocator.allocate(expected_returns)

        # 取引コストはゼロ
        for cost in result.transaction_costs.values():
            assert cost == 0.0

    def test_weekly_rebalance(self):
        """週次リバランスの場合"""
        allocator = NetProfitAllocator(
            periods_per_year=52,  # 週次
            max_weight=0.30,
        )

        expected_returns = pd.Series({
            "AAPL": 0.52,  # 年率52% → 週次1%
        })

        result = allocator.allocate(expected_returns)

        # 期待リターンは週次に換算される
        assert result.expected_returns["AAPL"] == pytest.approx(0.01)

    def test_daily_rebalance(self):
        """日次リバランスの場合"""
        allocator = NetProfitAllocator(
            periods_per_year=252,  # 日次
            max_weight=0.30,
        )

        expected_returns = pd.Series({
            "AAPL": 0.252,  # 年率25.2% → 日次0.1%
        })

        result = allocator.allocate(expected_returns)

        # 期待リターンは日次に換算される
        assert result.expected_returns["AAPL"] == pytest.approx(0.001)


class TestNetProfitAllocatorCategoryCosts:
    """カテゴリ別コストのテスト"""

    def test_with_cost_schedule(self):
        """TransactionCostScheduleを使用した配分"""
        from src.allocation.transaction_cost import (
            TransactionCostConfig,
            TransactionCostSchedule,
        )

        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
                "japan_stocks": TransactionCostConfig(cost_type="percentage", value=0.001),
            },
        )

        allocator = NetProfitAllocator(
            cost_schedule=schedule,
            max_weight=0.30,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "7203.T": 0.10,
        })

        result = allocator.allocate(
            expected_returns,
            portfolio_value=100000,
            categories={"AAPL": "us_stocks", "7203.T": "japan_stocks"},
        )

        # 両銘柄が選択される
        assert "AAPL" in result.weights
        assert "7203.T" in result.weights

        # 取引コストが異なる
        assert result.transaction_costs["AAPL"] != result.transaction_costs["7203.T"]

    def test_with_default_cost_schedule(self):
        """DEFAULT_COST_SCHEDULEを使用した配分"""
        from src.allocation.transaction_cost import DEFAULT_COST_SCHEDULE

        allocator = NetProfitAllocator(
            cost_schedule=DEFAULT_COST_SCHEDULE,
            max_weight=0.20,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
            "SPY": 0.08,
        })

        result = allocator.allocate(
            expected_returns,
            portfolio_value=100000,
            categories={"AAPL": "us_stocks", "SPY": "etfs_equity"},
        )

        assert result is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_backward_compatibility_bps(self):
        """後方互換性: bps指定での動作"""
        allocator = NetProfitAllocator(
            transaction_cost_bps=30.0,  # 従来の指定方法
            max_weight=0.20,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,
        })

        result = allocator.allocate(expected_returns)

        # 正常に動作する
        assert result is not None
        assert "AAPL" in result.weights

    def test_calculate_net_profit_with_category(self):
        """カテゴリ指定でのネット期待利益計算"""
        from src.allocation.transaction_cost import (
            TransactionCostConfig,
            TransactionCostSchedule,
        )

        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        allocator = NetProfitAllocator(cost_schedule=schedule)

        net_profit, cost = allocator.calculate_net_profit(
            expected_return=0.01,
            current_weight=0.0,
            target_weight=0.20,
            symbol="AAPL",
            category="us_stocks",
            portfolio_value=100000,
        )

        # 固定コスト計算
        # trade_value = 100000 * 0.20 = 20000
        # rate = 2.0 / 20000 = 0.0001
        # cost = 0.20 * 0.0001 = 0.00002
        assert cost == pytest.approx(0.00002)
        assert net_profit == pytest.approx(0.01 - 0.00002)

    def test_mixed_categories(self):
        """複数カテゴリの混合ポートフォリオ"""
        from src.allocation.transaction_cost import DEFAULT_COST_SCHEDULE

        allocator = NetProfitAllocator(
            cost_schedule=DEFAULT_COST_SCHEDULE,
            max_weight=0.20,
        )

        expected_returns = pd.Series({
            "AAPL": 0.12,      # 米国株式
            "7203.T": 0.10,   # 日本株式
            "SPY": 0.08,      # ETF
            "UNKNOWN": 0.06,  # カテゴリ不明（デフォルト適用）
        })

        categories = {
            "AAPL": "us_stocks",
            "7203.T": "japan_stocks",
            "SPY": "etfs_equity",
            # UNKNOWNはカテゴリ未指定
        }

        result = allocator.allocate(
            expected_returns,
            portfolio_value=100000,
            categories=categories,
        )

        # 全銘柄が配分される
        assert len(result.weights) >= 4  # 4銘柄 + CASH可能性
        assert sum(result.weights.values()) == pytest.approx(1.0)
