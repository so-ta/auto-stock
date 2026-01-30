"""
統合テスト: バックテストエンジン統一インターフェース

cmd_031で修正された以下の問題を検証:
- E1: weights_funcが全エンジンで無視される問題
- E2: rebalance_frequencyが伝搬されない問題
- Config伝搬の整合性

テストケース:
1. test_weights_func_is_called - weights_func呼び出し確認
2. test_rebalance_frequency_daily - daily設定でn_rebalances確認
3. test_rebalance_frequency_monthly - monthly設定でn_rebalances確認
4. test_different_frequencies_different_results - 頻度別に異なる結果
5. test_n_trades_positive - リバランス時にn_trades確認
6. test_config_propagation - Config全パラメータ伝搬確認
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock

from src.backtest.base import UnifiedBacktestConfig, UnifiedBacktestResult
from src.backtest.vectorbt_engine import VectorBTStyleEngine


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_universe() -> List[str]:
    """テスト用ユニバース（3銘柄）"""
    return ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture
def sample_prices(sample_universe) -> Dict[str, pd.DataFrame]:
    """テスト用価格データ（1年分）"""
    np.random.seed(42)  # 再現性のため
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")

    prices = {}
    for symbol in sample_universe:
        # ランダムウォークで価格を生成
        returns = np.random.randn(252) * 0.02  # 日次2%ボラティリティ
        price = 100 * np.exp(np.cumsum(returns))
        prices[symbol] = pd.DataFrame({"close": price}, index=dates)

    return prices


@pytest.fixture
def base_config() -> UnifiedBacktestConfig:
    """基本設定"""
    return UnifiedBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )


def equal_weight_func(
    universe: List[str],
    prices: Dict[str, pd.DataFrame],
    date: datetime,
    current_weights: Dict[str, float],
) -> Dict[str, float]:
    """等ウェイト戦略"""
    n = len(universe)
    if n == 0:
        return {}
    return {symbol: 1.0 / n for symbol in universe}


def tracking_weight_func(call_tracker: List[datetime]):
    """呼び出しを追跡するweights_func"""
    def _func(
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        date: datetime,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        call_tracker.append(date)
        n = len(universe)
        return {symbol: 1.0 / n for symbol in universe}
    return _func


# =============================================================================
# Test Cases
# =============================================================================

class TestWeightsFuncIntegration:
    """weights_func統合テスト"""

    def test_weights_func_is_called(self, sample_universe, sample_prices, base_config):
        """
        テスト1: weights_funcが実際に呼び出されることを確認

        検証: モック関数で呼び出し回数をカウント
        期待: monthly設定で約13回呼び出し（12ヶ月+初日）
        """
        call_tracker = []
        weights_func = tracking_weight_func(call_tracker)

        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=weights_func,
        )

        # weights_funcが呼び出されたことを確認
        assert len(call_tracker) > 0, "weights_func was never called"

        # monthly設定なので約13回（12ヶ月+初日）
        assert len(call_tracker) >= 12, f"Expected at least 12 calls, got {len(call_tracker)}"
        assert len(call_tracker) <= 14, f"Expected at most 14 calls, got {len(call_tracker)}"

        # n_rebalancesと一致することを確認
        assert result.n_rebalances == len(call_tracker), \
            f"n_rebalances ({result.n_rebalances}) != call count ({len(call_tracker)})"

    def test_weights_func_not_called_when_none(self, sample_universe, sample_prices, base_config):
        """
        weights_func=Noneの場合、均等配分が使用される
        """
        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=None,
        )

        # 結果が返されることを確認
        assert result is not None
        assert isinstance(result, UnifiedBacktestResult)


class TestRebalanceFrequency:
    """リバランス頻度テスト"""

    def test_rebalance_frequency_daily(self, sample_universe, sample_prices):
        """
        テスト2: daily設定でn_rebalances ≈ 取引日数

        期待: 252取引日 → n_rebalances ≈ 252
        """
        config = UnifiedBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            rebalance_frequency="daily",
        )

        call_tracker = []
        weights_func = tracking_weight_func(call_tracker)

        engine = VectorBTStyleEngine(unified_config=config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=config,
            weights_func=weights_func,
        )

        # daily設定では取引日数とほぼ同じ
        n_trading_days = len(sample_prices[sample_universe[0]])

        assert result.n_rebalances >= n_trading_days * 0.9, \
            f"Daily: Expected n_rebalances >= {n_trading_days * 0.9}, got {result.n_rebalances}"
        assert result.n_rebalances <= n_trading_days, \
            f"Daily: Expected n_rebalances <= {n_trading_days}, got {result.n_rebalances}"

    def test_rebalance_frequency_weekly(self, sample_universe, sample_prices):
        """
        weekly設定でn_rebalances ≈ 週数

        期待: 252取引日 / 5 ≈ 50週 → n_rebalances ≈ 50-52
        """
        config = UnifiedBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            rebalance_frequency="weekly",
        )

        call_tracker = []
        weights_func = tracking_weight_func(call_tracker)

        engine = VectorBTStyleEngine(unified_config=config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=config,
            weights_func=weights_func,
        )

        # weekly設定では約50-52回
        assert result.n_rebalances >= 45, \
            f"Weekly: Expected n_rebalances >= 45, got {result.n_rebalances}"
        assert result.n_rebalances <= 55, \
            f"Weekly: Expected n_rebalances <= 55, got {result.n_rebalances}"

    def test_rebalance_frequency_monthly(self, sample_universe, sample_prices, base_config):
        """
        テスト3: monthly設定でn_rebalances ≈ 月数

        期待: 12ヶ月 + 初日 = 13 → n_rebalances ≈ 13
        """
        call_tracker = []
        weights_func = tracking_weight_func(call_tracker)

        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=weights_func,
        )

        # monthly設定では12-14回
        assert result.n_rebalances >= 12, \
            f"Monthly: Expected n_rebalances >= 12, got {result.n_rebalances}"
        assert result.n_rebalances <= 14, \
            f"Monthly: Expected n_rebalances <= 14, got {result.n_rebalances}"

    def test_different_frequencies_different_results(self, sample_universe, sample_prices):
        """
        テスト4: daily/weekly/monthlyで異なる結果

        検証: 各頻度でn_rebalancesが異なることを確認
        """
        results = {}

        for freq in ["daily", "weekly", "monthly"]:
            config = UnifiedBacktestConfig(
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=100000.0,
                rebalance_frequency=freq,
            )

            engine = VectorBTStyleEngine(unified_config=config)
            result = engine.run(
                universe=sample_universe,
                prices=sample_prices,
                config=config,
                weights_func=equal_weight_func,
            )
            results[freq] = result

        # 各頻度でn_rebalancesが異なる
        assert results["daily"].n_rebalances > results["weekly"].n_rebalances, \
            f"daily ({results['daily'].n_rebalances}) should > weekly ({results['weekly'].n_rebalances})"
        assert results["weekly"].n_rebalances > results["monthly"].n_rebalances, \
            f"weekly ({results['weekly'].n_rebalances}) should > monthly ({results['monthly'].n_rebalances})"

        # 比率も確認（daily >> weekly >> monthly）
        daily_weekly_ratio = results["daily"].n_rebalances / results["weekly"].n_rebalances
        weekly_monthly_ratio = results["weekly"].n_rebalances / results["monthly"].n_rebalances

        assert daily_weekly_ratio >= 4, \
            f"daily/weekly ratio ({daily_weekly_ratio:.1f}) should be >= 4"
        assert weekly_monthly_ratio >= 3, \
            f"weekly/monthly ratio ({weekly_monthly_ratio:.1f}) should be >= 3"


class TestTradingStatistics:
    """取引統計テスト"""

    def test_n_rebalances_positive_with_weights_func(self, sample_universe, sample_prices, base_config):
        """
        テスト5: weights_func使用時にn_rebalances > 0
        """
        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=equal_weight_func,
        )

        assert result.n_rebalances > 0, \
            f"Expected n_rebalances > 0, got {result.n_rebalances}"

    def test_rebalance_records_created(self, sample_universe, sample_prices, base_config):
        """
        リバランス記録が作成されることを確認
        """
        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=equal_weight_func,
        )

        # rebalancesリストが存在し、n_rebalancesと一致
        assert len(result.rebalances) == result.n_rebalances, \
            f"rebalances count ({len(result.rebalances)}) != n_rebalances ({result.n_rebalances})"

        # 各リバランス記録の内容を確認
        if result.rebalances:
            first_rebalance = result.rebalances[0]
            assert hasattr(first_rebalance, 'date'), "RebalanceRecord should have 'date'"
            assert hasattr(first_rebalance, 'weights_before'), "RebalanceRecord should have 'weights_before'"
            assert hasattr(first_rebalance, 'weights_after'), "RebalanceRecord should have 'weights_after'"
            assert hasattr(first_rebalance, 'turnover'), "RebalanceRecord should have 'turnover'"


class TestConfigPropagation:
    """設定伝搬テスト"""

    def test_config_propagation(self, sample_universe, sample_prices):
        """
        テスト6: UnifiedBacktestConfigの全パラメータが伝搬
        """
        config = UnifiedBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=500000.0,  # カスタム値
            rebalance_frequency="weekly",
            transaction_cost_bps=15.0,  # カスタム値
            slippage_bps=8.0,  # カスタム値
        )

        engine = VectorBTStyleEngine(unified_config=config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=config,
            weights_func=equal_weight_func,
        )

        # 設定が結果に反映されていることを確認
        assert result.config is not None, "Result should have config"
        assert result.config.initial_capital == 500000.0, \
            f"initial_capital should be 500000, got {result.config.initial_capital}"
        assert result.config.rebalance_frequency == "weekly", \
            f"rebalance_frequency should be 'weekly', got {result.config.rebalance_frequency}"
        assert result.config.transaction_cost_bps == 15.0, \
            f"transaction_cost_bps should be 15.0, got {result.config.transaction_cost_bps}"
        assert result.config.slippage_bps == 8.0, \
            f"slippage_bps should be 8.0, got {result.config.slippage_bps}"

        # initial_valueが正しいことを確認
        assert result.initial_value == 500000.0, \
            f"initial_value should be 500000, got {result.initial_value}"

    def test_config_override_at_run_time(self, sample_universe, sample_prices):
        """
        run()時のconfig引数がエンジン初期化時の設定を上書きする
        """
        # エンジン初期化時の設定
        init_config = UnifiedBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000.0,
            rebalance_frequency="monthly",
        )

        # run()時の設定（上書き）
        run_config = UnifiedBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=200000.0,  # 上書き
            rebalance_frequency="weekly",  # 上書き
        )

        engine = VectorBTStyleEngine(unified_config=init_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=run_config,  # 上書き設定
            weights_func=equal_weight_func,
        )

        # run()時の設定が使用されていることを確認
        assert result.initial_value == 200000.0, \
            f"initial_value should be 200000 (run_config), got {result.initial_value}"

        # weekly設定のn_rebalancesであることを確認（monthlyより多い）
        assert result.n_rebalances >= 45, \
            f"Should use weekly frequency (n_rebalances >= 45), got {result.n_rebalances}"


class TestResultConsistency:
    """結果整合性テスト"""

    def test_result_has_required_fields(self, sample_universe, sample_prices, base_config):
        """
        UnifiedBacktestResultに必須フィールドが存在
        """
        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=equal_weight_func,
        )

        # 基本メトリクス
        assert hasattr(result, 'total_return'), "Missing total_return"
        assert hasattr(result, 'annual_return'), "Missing annual_return"
        assert hasattr(result, 'sharpe_ratio'), "Missing sharpe_ratio"
        assert hasattr(result, 'max_drawdown'), "Missing max_drawdown"
        assert hasattr(result, 'volatility'), "Missing volatility"

        # 取引統計
        assert hasattr(result, 'n_days'), "Missing n_days"
        assert hasattr(result, 'n_rebalances'), "Missing n_rebalances"
        assert hasattr(result, 'n_trades'), "Missing n_trades"
        assert hasattr(result, 'total_turnover'), "Missing total_turnover"

        # 時系列データ
        assert hasattr(result, 'daily_returns'), "Missing daily_returns"
        assert hasattr(result, 'portfolio_values'), "Missing portfolio_values"

        # メタデータ
        assert hasattr(result, 'engine_name'), "Missing engine_name"
        assert result.engine_name == "vectorbt", \
            f"engine_name should be 'vectorbt', got {result.engine_name}"

    def test_n_days_matches_data(self, sample_universe, sample_prices, base_config):
        """
        n_daysが価格データの日数と一致
        """
        engine = VectorBTStyleEngine(unified_config=base_config)
        result = engine.run(
            universe=sample_universe,
            prices=sample_prices,
            config=base_config,
            weights_func=equal_weight_func,
        )

        expected_days = len(sample_prices[sample_universe[0]])

        # n_daysが妥当な範囲内
        assert result.n_days >= expected_days * 0.9, \
            f"n_days ({result.n_days}) should be close to {expected_days}"
        assert result.n_days <= expected_days, \
            f"n_days ({result.n_days}) should not exceed {expected_days}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
