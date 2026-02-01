"""
ロットサイズ調整のテスト

テスト対象:
- LotSizeAdjuster: 重みを購入可能な株数に変換
- OrderGenerator: 発注リスト生成
"""

import pytest

from src.allocation.lot_adjuster import LotSizeAdjuster, LotAdjustmentResult
from src.allocation.order_generator import OrderGenerator, OrderItem, OrderSummary


class TestLotSizeAdjuster:
    """LotSizeAdjusterのテスト"""

    def test_basic_adjustment_us_stocks(self):
        """米国株（1株単位）の基本的な調整"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.5, "GOOGL": 0.5},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
            lot_sizes={"AAPL": 1, "GOOGL": 1},
            fractional_allowed={"AAPL": False, "GOOGL": False},
        )

        # AAPL: 5000 / 150 = 33.33 → 33株
        # GOOGL: 5000 / 100 = 50 → 50株
        assert result.target_shares["AAPL"] == 33
        assert result.target_shares["GOOGL"] == 50

        # 合計投資額
        total = 33 * 150 + 50 * 100
        assert result.total_invested == total

        # 残りキャッシュ
        assert result.cash_remainder == 10000 - total

    def test_japan_stocks_100_lot(self):
        """日本株（100株単位）の調整"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"7203.T": 0.5, "9984.T": 0.5},
            prices={"7203.T": 2500.0, "9984.T": 8000.0},
            portfolio_value=5000000.0,
            lot_sizes={"7203.T": 100, "9984.T": 100},
            fractional_allowed={"7203.T": False, "9984.T": False},
        )

        # 7203.T: 2,500,000 / 2500 = 1000 → 1000株（10ロット）
        # 9984.T: 2,500,000 / 8000 = 312.5 → 300株（3ロット）
        assert result.target_shares["7203.T"] == 1000
        assert result.target_shares["9984.T"] == 300

    def test_optimization_adds_lots_to_reduce_deviation(self):
        """最適化アルゴリズム: 余りキャッシュで乖離最小化"""
        adjuster = LotSizeAdjuster()

        # 意図的に大きな端数が出るケース
        result = adjuster.adjust_to_lot_size(
            target_weights={"A": 0.5, "B": 0.5},
            prices={"A": 100.0, "B": 100.0},
            portfolio_value=1050.0,  # 50の余り
            lot_sizes={"A": 1, "B": 1},
            fractional_allowed={"A": False, "B": False},
        )

        # floor: A=5, B=5 (合計1000)
        # 余り50で A か B に +1
        total_shares = result.target_shares["A"] + result.target_shares["B"]
        assert total_shares == 10  # 余り50で追加なし（100必要）

    def test_etf_fractional_allowed(self):
        """ETF（端数購入可）の処理"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"SPY": 0.5, "AAPL": 0.5},
            prices={"SPY": 450.0, "AAPL": 150.0},
            portfolio_value=10000.0,
            lot_sizes={"SPY": 1, "AAPL": 1},
            fractional_allowed={"SPY": True, "AAPL": False},
        )

        # SPY: 端数購入可 → 5000 / 450 = 11.111...
        assert "SPY" in result.fractional_shares
        assert abs(result.fractional_shares["SPY"] - 11.111111) < 0.001

        # AAPL: 整数のみ → 5000 / 150 = 33.33 → 33
        assert result.target_shares["AAPL"] == 33

    def test_mixed_japan_us_etf(self):
        """日本株・米国株・ETF混在ポートフォリオ"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"7203.T": 0.4, "AAPL": 0.3, "SPY": 0.3},
            prices={"7203.T": 2500.0, "AAPL": 150.0, "SPY": 450.0},
            portfolio_value=1000000.0,
            lot_sizes={"7203.T": 100, "AAPL": 1, "SPY": 1},
            fractional_allowed={"7203.T": False, "AAPL": False, "SPY": True},
        )

        # 7203.T: 400,000 / 2500 = 160 → 100株単位で100株
        assert result.target_shares["7203.T"] == 100

        # AAPL: 300,000 / 150 = 2000 → 最適化で+1追加される可能性あり
        assert result.target_shares["AAPL"] >= 2000

        # SPY: 端数可
        assert "SPY" in result.fractional_shares

    def test_min_position_value(self):
        """最小ポジション金額のフィルタ"""
        adjuster = LotSizeAdjuster(min_position_value=50000.0)
        result = adjuster.adjust_to_lot_size(
            target_weights={"A": 0.01, "B": 0.99},  # Aは1%
            prices={"A": 100000.0, "B": 100.0},  # A: 1株10万円
            portfolio_value=100000.0,  # 総額10万円
            lot_sizes={"A": 1, "B": 1},
            fractional_allowed={"A": False, "B": False},
        )

        # A: 目標1000円 / 100000円 = 0.01株 → floor後0株
        # 最小金額50000円未満なので0株（最適化でも追加されない）
        # ※10万円の1株は追加コストが高すぎる
        assert result.target_shares.get("A", 0) == 0

    def test_weight_deviation_calculation(self):
        """重み乖離（RMSE）の計算"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"A": 0.5, "B": 0.5},
            prices={"A": 100.0, "B": 100.0},
            portfolio_value=1000.0,
            lot_sizes={"A": 1, "B": 1},
            fractional_allowed={"A": False, "B": False},
        )

        # 完全一致なら乖離は0
        assert result.weight_deviation == 0.0

    def test_empty_weights(self):
        """空の重みの処理"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={},
            prices={},
            portfolio_value=100000.0,
        )

        assert result.target_shares == {}
        assert result.cash_remainder == 100000.0

    def test_missing_price(self):
        """価格が欠損している銘柄"""
        adjuster = LotSizeAdjuster()
        result = adjuster.adjust_to_lot_size(
            target_weights={"A": 0.5, "B": 0.5},
            prices={"A": 100.0},  # Bの価格なし
            portfolio_value=10000.0,
            lot_sizes={"A": 1, "B": 1},
            fractional_allowed={"A": False, "B": False},
        )

        # Bはスキップされる
        assert "B" not in result.target_shares
        # 調整ログに「B」に関するメッセージがある
        assert any("B" in adj for adj in result.adjustments_made)


class TestOrderGenerator:
    """OrderGeneratorのテスト"""

    def test_basic_buy_orders(self):
        """基本的な買い注文生成"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.5, "GOOGL": 0.5},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
            lot_sizes={"AAPL": 1, "GOOGL": 1},
            fractional_allowed={"AAPL": False, "GOOGL": False},
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={},  # 新規
            target_result=adjustment,
            prices={"AAPL": 150.0, "GOOGL": 100.0},
        )

        assert summary.buy_count == 2
        assert summary.sell_count == 0
        assert summary.total_buy_amount > 0

    def test_sell_orders(self):
        """売り注文生成"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.5},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
            lot_sizes={"AAPL": 1, "GOOGL": 1},
            fractional_allowed={"AAPL": False, "GOOGL": False},
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={"AAPL": 100, "GOOGL": 50},  # GOOGLを全売り
            target_result=adjustment,
            prices={"AAPL": 150.0, "GOOGL": 100.0},
        )

        # GOOGLは売り
        sell_orders = summary.sell_orders()
        googl_order = next((o for o in sell_orders if o.symbol == "GOOGL"), None)
        assert googl_order is not None
        assert googl_order.action == "SELL"
        assert googl_order.shares == 50

    def test_mixed_buy_sell(self):
        """買いと売りの混在"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.7, "GOOGL": 0.3},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
            lot_sizes={"AAPL": 1, "GOOGL": 1},
            fractional_allowed={"AAPL": False, "GOOGL": False},
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={"AAPL": 30, "GOOGL": 50},  # AAPL不足、GOOGL過剰
            target_result=adjustment,
            prices={"AAPL": 150.0, "GOOGL": 100.0},
        )

        assert summary.buy_count > 0
        assert summary.sell_count > 0

    def test_order_sort_sell_first(self):
        """注文は売り→買いの順"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.7, "GOOGL": 0.3},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={"AAPL": 30, "GOOGL": 50},
            target_result=adjustment,
            prices={"AAPL": 150.0, "GOOGL": 100.0},
        )

        # 最初の売り注文が買い注文より前に来る
        if summary.sell_count > 0 and summary.buy_count > 0:
            first_sell_idx = next(
                i for i, o in enumerate(summary.orders) if o.action == "SELL"
            )
            first_buy_idx = next(
                i for i, o in enumerate(summary.orders) if o.action == "BUY"
            )
            assert first_sell_idx < first_buy_idx

    def test_min_order_amount_filter(self):
        """最小発注金額フィルタ"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"A": 0.5, "B": 0.5},
            prices={"A": 100.0, "B": 100.0},
            portfolio_value=10000.0,
        )

        generator = OrderGenerator(min_order_amount=1000.0)
        summary = generator.generate_orders(
            current_positions={"A": 49, "B": 49},  # 各1株差
            target_result=adjustment,
            prices={"A": 100.0, "B": 100.0},
        )

        # 100円の注文は1000円未満なのでスキップ
        assert summary.buy_count == 0

    def test_fractional_orders(self):
        """端数注文の処理"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"SPY": 1.0},
            prices={"SPY": 450.0},
            portfolio_value=10000.0,
            fractional_allowed={"SPY": True},
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={},
            target_result=adjustment,
            prices={"SPY": 450.0},
        )

        order = summary.orders[0]
        assert order.is_fractional is True
        assert isinstance(order.shares, float)

    def test_format_report(self):
        """レポートフォーマット"""
        adjuster = LotSizeAdjuster()
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={"AAPL": 0.5, "GOOGL": 0.5},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
        )

        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={"AAPL": 10},
            target_result=adjustment,
            prices={"AAPL": 150.0, "GOOGL": 100.0},
        )

        report = summary.format_report()
        assert "発注リスト" in report
        assert "買い" in report or "売り" in report

    def test_generate_from_weights(self):
        """重みから直接発注リスト生成"""
        generator = OrderGenerator()
        summary = generator.generate_from_weights(
            current_positions={"AAPL": 10},
            target_weights={"AAPL": 0.5, "GOOGL": 0.5},
            prices={"AAPL": 150.0, "GOOGL": 100.0},
            portfolio_value=10000.0,
            lot_sizes={"AAPL": 1, "GOOGL": 1},
            fractional_allowed={"AAPL": False, "GOOGL": False},
        )

        assert len(summary.orders) > 0


class TestIntegration:
    """統合テスト"""

    def test_full_workflow_japan_portfolio(self):
        """日本株ポートフォリオの完全ワークフロー"""
        # 1. ロット調整
        adjuster = LotSizeAdjuster(min_position_value=100000.0)
        adjustment = adjuster.adjust_to_lot_size(
            target_weights={
                "7203.T": 0.25,  # トヨタ
                "9984.T": 0.25,  # ソフトバンク
                "6758.T": 0.25,  # ソニー
                "9432.T": 0.25,  # NTT
            },
            prices={
                "7203.T": 2500.0,
                "9984.T": 8000.0,
                "6758.T": 12000.0,
                "9432.T": 170.0,
            },
            portfolio_value=10000000.0,  # 1000万円
            lot_sizes={
                "7203.T": 100,
                "9984.T": 100,
                "6758.T": 100,
                "9432.T": 100,
            },
            fractional_allowed={
                "7203.T": False,
                "9984.T": False,
                "6758.T": False,
                "9432.T": False,
            },
        )

        # 全銘柄が100株単位
        for symbol, shares in adjustment.target_shares.items():
            assert shares % 100 == 0, f"{symbol}が100株単位でない: {shares}"

        # 2. 発注リスト生成
        generator = OrderGenerator()
        summary = generator.generate_orders(
            current_positions={},
            target_result=adjustment,
            prices={
                "7203.T": 2500.0,
                "9984.T": 8000.0,
                "6758.T": 12000.0,
                "9432.T": 170.0,
            },
            lot_sizes={
                "7203.T": 100,
                "9984.T": 100,
                "6758.T": 100,
                "9432.T": 100,
            },
        )

        # 全て買い注文
        assert summary.buy_count == 4
        assert summary.sell_count == 0

        # 総額が予算を超えない
        assert summary.total_buy_amount <= 10000000.0
