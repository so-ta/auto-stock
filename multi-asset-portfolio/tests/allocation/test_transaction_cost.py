"""
Transaction Cost Tests

カテゴリ別取引コストのテスト。
"""

import pytest
from pathlib import Path
import tempfile

from src.allocation.transaction_cost import (
    TransactionCostConfig,
    TransactionCostSchedule,
    DEFAULT_COST_SCHEDULE,
    load_cost_schedule,
)


class TestTransactionCostConfig:
    """TransactionCostConfig のテスト"""

    def test_init_default(self):
        """デフォルト値での初期化"""
        config = TransactionCostConfig()

        assert config.cost_type == "bps"
        assert config.value == 20.0
        assert config.currency == "USD"

    def test_init_bps(self):
        """bps指定での初期化"""
        config = TransactionCostConfig(cost_type="bps", value=30.0)

        assert config.cost_type == "bps"
        assert config.value == 30.0

    def test_init_percentage(self):
        """percentage指定での初期化"""
        config = TransactionCostConfig(cost_type="percentage", value=0.001)

        assert config.cost_type == "percentage"
        assert config.value == 0.001

    def test_init_fixed(self):
        """fixed指定での初期化"""
        config = TransactionCostConfig(cost_type="fixed", value=2.0, currency="USD")

        assert config.cost_type == "fixed"
        assert config.value == 2.0
        assert config.currency == "USD"

    def test_init_invalid_cost_type(self):
        """無効なcost_type"""
        with pytest.raises(ValueError, match="Invalid cost_type"):
            TransactionCostConfig(cost_type="invalid")

    def test_init_negative_value(self):
        """負のvalue"""
        with pytest.raises(ValueError, match="non-negative"):
            TransactionCostConfig(value=-1.0)

    def test_to_rate_bps(self):
        """bpsからレートへの変換"""
        config = TransactionCostConfig(cost_type="bps", value=20.0)

        rate = config.to_rate()

        assert rate == pytest.approx(0.002)  # 20bps = 0.2%

    def test_to_rate_percentage(self):
        """percentageからレートへの変換"""
        config = TransactionCostConfig(cost_type="percentage", value=0.001)

        rate = config.to_rate()

        assert rate == pytest.approx(0.001)  # 0.1%

    def test_to_rate_fixed_with_trade_value(self):
        """固定額からレートへの変換（取引額指定）"""
        config = TransactionCostConfig(cost_type="fixed", value=2.0)

        rate = config.to_rate(trade_value=5000)  # $5000の取引

        assert rate == pytest.approx(0.0004)  # $2 / $5000 = 0.04%

    def test_to_rate_fixed_without_trade_value(self):
        """固定額からレートへの変換（取引額未指定）"""
        config = TransactionCostConfig(cost_type="fixed", value=2.0)

        rate = config.to_rate()  # 取引額不明

        assert rate == 0.01  # デフォルト1%

    def test_to_rate_fixed_with_zero_trade_value(self):
        """固定額からレートへの変換（取引額ゼロ）"""
        config = TransactionCostConfig(cost_type="fixed", value=2.0)

        rate = config.to_rate(trade_value=0)

        assert rate == 0.01  # デフォルト1%

    def test_from_dict(self):
        """辞書から作成"""
        data = {
            "cost_type": "percentage",
            "value": 0.002,
            "currency": "JPY",
        }

        config = TransactionCostConfig.from_dict(data)

        assert config.cost_type == "percentage"
        assert config.value == 0.002
        assert config.currency == "JPY"

    def test_from_dict_defaults(self):
        """辞書から作成（デフォルト値）"""
        config = TransactionCostConfig.from_dict({})

        assert config.cost_type == "bps"
        assert config.value == 20.0
        assert config.currency == "USD"

    def test_to_dict(self):
        """辞書に変換"""
        config = TransactionCostConfig(
            cost_type="fixed",
            value=2.0,
            currency="USD",
        )

        d = config.to_dict()

        assert d["cost_type"] == "fixed"
        assert d["value"] == 2.0
        assert d["currency"] == "USD"


class TestTransactionCostSchedule:
    """TransactionCostSchedule のテスト"""

    def test_init_default(self):
        """デフォルト値での初期化"""
        schedule = TransactionCostSchedule()

        assert schedule.default.cost_type == "bps"
        assert schedule.default.value == 20.0
        assert len(schedule.category_costs) == 0

    def test_init_with_categories(self):
        """カテゴリ指定での初期化"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
                "japan_stocks": TransactionCostConfig(cost_type="percentage", value=0.001),
            },
        )

        assert len(schedule.category_costs) == 2
        assert schedule.category_costs["us_stocks"].cost_type == "fixed"
        assert schedule.category_costs["japan_stocks"].value == 0.001

    def test_get_cost_config_with_category(self):
        """カテゴリ指定でのコスト設定取得"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        config = schedule.get_cost_config("AAPL", category="us_stocks")

        assert config.cost_type == "fixed"
        assert config.value == 2.0

    def test_get_cost_config_unknown_category(self):
        """未知カテゴリでのコスト設定取得（デフォルト）"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        config = schedule.get_cost_config("UNKNOWN", category="unknown_category")

        assert config.cost_type == "bps"
        assert config.value == 20

    def test_get_cost_config_no_category(self):
        """カテゴリ未指定でのコスト設定取得（デフォルト）"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
        )

        config = schedule.get_cost_config("AAPL")

        assert config.cost_type == "bps"
        assert config.value == 20

    def test_calculate_cost_bps(self):
        """bpsでのコスト計算"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
        )

        cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.05,
        )

        # 0.05 * 0.002 = 0.0001 (0.01%)
        assert cost == pytest.approx(0.0001)

    def test_calculate_cost_percentage(self):
        """percentageでのコスト計算"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "japan_stocks": TransactionCostConfig(cost_type="percentage", value=0.001),
            },
        )

        cost = schedule.calculate_cost(
            symbol="7203.T",
            weight_change=0.05,
            category="japan_stocks",
        )

        # 0.05 * 0.001 = 0.00005 (0.005%)
        assert cost == pytest.approx(0.00005)

    def test_calculate_cost_fixed(self):
        """固定額でのコスト計算"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.05,
            portfolio_value=100000,
            category="us_stocks",
        )

        # trade_value = 100000 * 0.05 = 5000
        # rate = 2.0 / 5000 = 0.0004
        # cost = 0.05 * 0.0004 = 0.00002 (0.002%)
        assert cost == pytest.approx(0.00002)

    def test_calculate_cost_fixed_without_portfolio_value(self):
        """固定額でのコスト計算（ポートフォリオ総額未指定）"""
        schedule = TransactionCostSchedule(
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.05,
            category="us_stocks",
        )

        # portfolio_valueがないのでデフォルト1%を使用
        # cost = 0.05 * 0.01 = 0.0005
        assert cost == pytest.approx(0.0005)

    def test_from_dict(self):
        """辞書から作成"""
        data = {
            "default": {"cost_type": "bps", "value": 20},
            "categories": {
                "us_stocks": {"cost_type": "fixed", "value": 2.0},
                "japan_stocks": {"cost_type": "percentage", "value": 0.001},
            },
        }

        schedule = TransactionCostSchedule.from_dict(data)

        assert schedule.default.value == 20
        assert len(schedule.category_costs) == 2
        assert schedule.category_costs["us_stocks"].cost_type == "fixed"

    def test_from_yaml(self):
        """YAMLファイルから読み込み"""
        yaml_content = """
transaction_costs:
  default:
    cost_type: bps
    value: 25

  categories:
    us_stocks:
      cost_type: fixed
      value: 3.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            schedule = TransactionCostSchedule.from_yaml(f.name)

        assert schedule.default.value == 25
        assert schedule.category_costs["us_stocks"].value == 3.0

    def test_from_yaml_nonexistent(self):
        """存在しないYAMLファイル"""
        schedule = TransactionCostSchedule.from_yaml("/nonexistent/path.yaml")

        # デフォルトスケジュールが返る
        assert schedule.default.value == 20.0

    def test_to_dict(self):
        """辞書に変換"""
        schedule = TransactionCostSchedule(
            default=TransactionCostConfig(cost_type="bps", value=20),
            category_costs={
                "us_stocks": TransactionCostConfig(cost_type="fixed", value=2.0),
            },
        )

        d = schedule.to_dict()

        assert d["default"]["value"] == 20
        assert d["categories"]["us_stocks"]["cost_type"] == "fixed"


class TestDefaultCostSchedule:
    """DEFAULT_COST_SCHEDULE のテスト"""

    def test_has_default(self):
        """デフォルトコストが設定されている"""
        assert DEFAULT_COST_SCHEDULE.default.cost_type == "bps"
        assert DEFAULT_COST_SCHEDULE.default.value == 15  # 0.15% (15bps)

    def test_has_us_stocks(self):
        """米国株式のコストが設定されている"""
        config = DEFAULT_COST_SCHEDULE.category_costs["us_stocks"]

        assert config.cost_type == "fixed"
        assert config.value == 2.0

    def test_has_japan_stocks(self):
        """日本株式のコストが設定されている"""
        config = DEFAULT_COST_SCHEDULE.category_costs["japan_stocks"]

        assert config.cost_type == "percentage"
        assert config.value == 0.001  # 0.1%

    def test_has_etfs(self):
        """ETFのコストが設定されている"""
        for etf_category in ["etfs", "etfs_equity", "etfs_bond", "etfs_commodity"]:
            config = DEFAULT_COST_SCHEDULE.category_costs[etf_category]
            assert config.cost_type == "percentage"
            assert config.value == 0.002  # 0.2%

    def test_us_stocks_cost_calculation(self):
        """米国株式のコスト計算（$2固定）"""
        cost = DEFAULT_COST_SCHEDULE.calculate_cost(
            symbol="AAPL",
            weight_change=0.05,
            portfolio_value=100000,
            category="us_stocks",
        )

        # trade_value = 100000 * 0.05 = 5000
        # rate = 2.0 / 5000 = 0.0004
        # cost = 0.05 * 0.0004 = 0.00002
        assert cost == pytest.approx(0.00002)

    def test_japan_stocks_cost_calculation(self):
        """日本株式のコスト計算（0.1%）"""
        cost = DEFAULT_COST_SCHEDULE.calculate_cost(
            symbol="7203.T",
            weight_change=0.05,
            category="japan_stocks",
        )

        # 0.05 * 0.001 = 0.00005
        assert cost == pytest.approx(0.00005)

    def test_etf_cost_calculation(self):
        """ETFのコスト計算（0.2%）"""
        cost = DEFAULT_COST_SCHEDULE.calculate_cost(
            symbol="SPY",
            weight_change=0.05,
            category="etfs_equity",
        )

        # 0.05 * 0.002 = 0.0001
        assert cost == pytest.approx(0.0001)


class TestLoadCostSchedule:
    """load_cost_schedule のテスト"""

    def test_load_default(self):
        """デフォルト読み込み"""
        schedule = load_cost_schedule()

        # DEFAULT_COST_SCHEDULE または config/cost_schedule.yaml から読み込み
        assert schedule.default is not None

    def test_load_from_path(self):
        """パス指定での読み込み"""
        yaml_content = """
transaction_costs:
  default:
    cost_type: bps
    value: 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            schedule = load_cost_schedule(f.name)

        assert schedule.default.value == 30


class TestIntegration:
    """統合テスト"""

    def test_realistic_portfolio_costs(self):
        """現実的なポートフォリオでのコスト計算"""
        schedule = DEFAULT_COST_SCHEDULE
        portfolio_value = 100000

        # 米国株式 5%購入: $2 / $5000 = 0.04%
        us_cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.05,
            portfolio_value=portfolio_value,
            category="us_stocks",
        )

        # 日本株式 5%購入: 0.1% * 5% = 0.005%
        jp_cost = schedule.calculate_cost(
            symbol="7203.T",
            weight_change=0.05,
            category="japan_stocks",
        )

        # ETF 5%購入: 0.2% * 5% = 0.01%
        etf_cost = schedule.calculate_cost(
            symbol="SPY",
            weight_change=0.05,
            category="etfs_equity",
        )

        # コスト比較
        # 米国株式: 0.00002 (0.002%)
        # 日本株式: 0.00005 (0.005%)
        # ETF: 0.0001 (0.01%)
        assert us_cost < jp_cost < etf_cost
        print(f"US Stock cost: {us_cost * 100:.4f}%")
        print(f"Japan Stock cost: {jp_cost * 100:.4f}%")
        print(f"ETF cost: {etf_cost * 100:.4f}%")

    def test_small_trade_fixed_cost_impact(self):
        """小さな取引での固定コストの影響"""
        schedule = DEFAULT_COST_SCHEDULE
        portfolio_value = 100000

        # 小さな取引（1%）
        small_cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.01,  # 1%
            portfolio_value=portfolio_value,
            category="us_stocks",
        )

        # 大きな取引（10%）
        large_cost = schedule.calculate_cost(
            symbol="AAPL",
            weight_change=0.10,  # 10%
            portfolio_value=portfolio_value,
            category="us_stocks",
        )

        # 固定コストなので、小さな取引の方がレート換算で高コスト
        small_rate = small_cost / 0.01
        large_rate = large_cost / 0.10

        assert small_rate > large_rate  # 小さな取引の方がコスト効率が悪い
