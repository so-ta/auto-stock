"""
tests/unit/test_strategy.py - 戦略モジュールの単体テスト

テスト対象:
- GateChecker, GateConfig (gate_checker.py)
- StrategyMetrics, GateResult, GateCheckResult (gate_checker.py)

Task: task_032_10
Author: 足軽5号
"""

import pytest

from src.strategy.gate_checker import (
    GateChecker,
    GateConfig,
    GateType,
    GateResult,
    GateCheckResult,
    StrategyMetrics,
)


# =============================================================================
# GateConfig Tests
# =============================================================================

class TestGateConfig:
    """GateConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定でインスタンス化できる"""
        config = GateConfig()
        assert config.min_trades == 30
        assert config.max_drawdown_pct == 25.0
        assert config.min_expected_value == 0.0
        assert config.min_sharpe_ratio == 0.5
        assert config.min_win_rate_pct == 45.0
        assert config.min_profit_factor == 1.2
        assert config.stability_periods == 6
        assert config.stability_min_positive == 4

    def test_custom_config(self):
        """カスタム設定でインスタンス化できる"""
        config = GateConfig(
            min_trades=50,
            max_drawdown_pct=20.0,
            min_sharpe_ratio=1.0,
        )
        assert config.min_trades == 50
        assert config.max_drawdown_pct == 20.0
        assert config.min_sharpe_ratio == 1.0

    def test_invalid_min_trades(self):
        """min_tradesが0以下の場合エラー"""
        with pytest.raises(ValueError, match="min_trades must be >= 1"):
            GateConfig(min_trades=0)

    def test_invalid_max_drawdown_pct_zero(self):
        """max_drawdown_pctが0の場合エラー"""
        with pytest.raises(ValueError, match="max_drawdown_pct must be in"):
            GateConfig(max_drawdown_pct=0)

    def test_invalid_max_drawdown_pct_over_100(self):
        """max_drawdown_pctが100超の場合エラー"""
        with pytest.raises(ValueError, match="max_drawdown_pct must be in"):
            GateConfig(max_drawdown_pct=101)

    def test_invalid_win_rate_negative(self):
        """min_win_rate_pctが負の場合エラー"""
        with pytest.raises(ValueError, match="min_win_rate_pct must be in"):
            GateConfig(min_win_rate_pct=-1)

    def test_invalid_win_rate_over_100(self):
        """min_win_rate_pctが100超の場合エラー"""
        with pytest.raises(ValueError, match="min_win_rate_pct must be in"):
            GateConfig(min_win_rate_pct=101)

    def test_invalid_profit_factor_negative(self):
        """min_profit_factorが負の場合エラー"""
        with pytest.raises(ValueError, match="min_profit_factor must be >= 0"):
            GateConfig(min_profit_factor=-1)

    def test_invalid_stability_config(self):
        """stability_min_positiveがstability_periodsより大きい場合エラー"""
        with pytest.raises(ValueError, match="stability_min_positive must be <="):
            GateConfig(stability_periods=4, stability_min_positive=5)

    def test_frozen_config(self):
        """GateConfigはfrozen=Trueで変更不可"""
        config = GateConfig()
        with pytest.raises(AttributeError):
            config.min_trades = 100


# =============================================================================
# StrategyMetrics Tests
# =============================================================================

class TestStrategyMetrics:
    """StrategyMetricsのテスト"""

    def test_create_metrics(self):
        """StrategyMetricsを作成できる"""
        metrics = StrategyMetrics(
            strategy_id="momentum",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.001,
            sharpe_ratio=1.5,
            win_rate_pct=55.0,
            profit_factor=1.8,
        )
        assert metrics.strategy_id == "momentum"
        assert metrics.asset_id == "SPY"
        assert metrics.trade_count == 50
        assert metrics.sharpe_ratio == 1.5

    def test_metrics_with_period_returns(self):
        """period_returnsを持つメトリクス"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="TLT",
            trade_count=30,
            max_drawdown_pct=10.0,
            expected_value=0.002,
            sharpe_ratio=0.8,
            win_rate_pct=50.0,
            profit_factor=1.3,
            period_returns=[0.05, 0.02, -0.01, 0.03, 0.04, 0.01],
        )
        assert len(metrics.period_returns) == 6


# =============================================================================
# GateResult Tests
# =============================================================================

class TestGateResult:
    """GateResultのテスト"""

    def test_create_gate_result(self):
        """GateResultを作成できる"""
        result = GateResult(
            gate_type=GateType.MIN_TRADES,
            passed=True,
            actual_value=50,
            threshold=30,
            message="",
        )
        assert result.gate_type == GateType.MIN_TRADES
        assert result.passed is True
        assert result.actual_value == 50


# =============================================================================
# GateCheckResult Tests
# =============================================================================

class TestGateCheckResult:
    """GateCheckResultのテスト"""

    def test_create_check_result(self):
        """GateCheckResultを作成できる"""
        result = GateCheckResult(
            strategy_id="test",
            asset_id="SPY",
            passed=True,
        )
        assert result.strategy_id == "test"
        assert result.passed is True

    def test_to_dict(self):
        """to_dictで辞書に変換できる"""
        gate_result = GateResult(
            gate_type=GateType.MIN_SHARPE,
            passed=False,
            actual_value=0.3,
            threshold=0.5,
            message="Sharpe too low",
        )
        result = GateCheckResult(
            strategy_id="test",
            asset_id="SPY",
            passed=False,
            results=[gate_result],
            rejection_reasons=["Sharpe too low"],
        )
        d = result.to_dict()
        assert d["strategy_id"] == "test"
        assert d["passed"] is False
        assert len(d["results"]) == 1
        assert d["results"][0]["gate_type"] == "min_sharpe"


# =============================================================================
# GateChecker Tests
# =============================================================================

class TestGateChecker:
    """GateCheckerのテスト"""

    @pytest.fixture
    def passing_metrics(self):
        """全ゲート通過するメトリクス"""
        return StrategyMetrics(
            strategy_id="momentum",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
            period_returns=[0.05, 0.02, 0.01, 0.03, 0.04, 0.01],
        )

    @pytest.fixture
    def failing_metrics(self):
        """複数ゲート不通過のメトリクス"""
        return StrategyMetrics(
            strategy_id="bad_strategy",
            asset_id="XYZ",
            trade_count=10,  # min_trades違反
            max_drawdown_pct=30.0,  # max_drawdown違反
            expected_value=-0.001,  # expected_value違反
            sharpe_ratio=0.2,  # sharpe違反
            win_rate_pct=40.0,  # win_rate違反
            profit_factor=0.8,  # profit_factor違反
            period_returns=[-0.01, -0.02, -0.01, 0.01, -0.02, -0.01],  # 安定性違反
        )

    def test_initialization_default(self):
        """デフォルト設定でインスタンス化できる"""
        checker = GateChecker()
        assert checker.config.min_trades == 30

    def test_initialization_custom_config(self):
        """カスタム設定でインスタンス化できる"""
        config = GateConfig(min_trades=50)
        checker = GateChecker(config)
        assert checker.config.min_trades == 50

    def test_check_all_pass(self, passing_metrics):
        """全ゲート通過"""
        checker = GateChecker()
        result = checker.check(passing_metrics)
        assert result.passed is True
        assert len(result.rejection_reasons) == 0

    def test_check_all_fail(self, failing_metrics):
        """複数ゲート不通過"""
        checker = GateChecker()
        result = checker.check(failing_metrics)
        assert result.passed is False
        assert len(result.rejection_reasons) > 0

    def test_check_min_trades_fail(self):
        """min_tradesのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=20,  # 30未満
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_TRADES and not r.passed
                   for r in result.results)

    def test_check_max_drawdown_fail(self):
        """max_drawdownのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=30.0,  # 25%超
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MAX_DRAWDOWN and not r.passed
                   for r in result.results)

    def test_check_min_sharpe_fail(self):
        """min_sharpeのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=0.3,  # 0.5未満
            win_rate_pct=55.0,
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_SHARPE and not r.passed
                   for r in result.results)

    def test_check_min_win_rate_fail(self):
        """min_win_rateのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=40.0,  # 45%未満
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_WIN_RATE and not r.passed
                   for r in result.results)

    def test_check_min_profit_factor_fail(self):
        """min_profit_factorのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.0,  # 1.2未満
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_PROFIT_FACTOR and not r.passed
                   for r in result.results)

    def test_check_stability_fail(self):
        """安定性のみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
            period_returns=[-0.01, 0.01, -0.02, -0.01, 0.02, -0.03],  # 2/6プラス
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.STABILITY and not r.passed
                   for r in result.results)

    def test_check_expected_value_fail(self):
        """min_expected_valueのみ不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=-0.001,  # 0未満
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_EXPECTED_VALUE and not r.passed
                   for r in result.results)

    def test_check_without_period_returns(self):
        """period_returnsなしの場合、安定性チェックをスキップ"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
            # period_returns省略
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is True
        # STABILITYのチェック結果がないことを確認
        assert not any(r.gate_type == GateType.STABILITY for r in result.results)

    def test_filter_strategies(self, passing_metrics, failing_metrics):
        """filter_strategiesで複数戦略をフィルタリング"""
        checker = GateChecker()
        metrics_list = [passing_metrics, failing_metrics]
        passed, results = checker.filter_strategies(metrics_list)
        assert len(passed) == 1
        assert passed[0].strategy_id == "momentum"
        assert len(results) == 2

    def test_filter_strategies_empty_list(self):
        """空リストの場合"""
        checker = GateChecker()
        passed, results = checker.filter_strategies([])
        assert len(passed) == 0
        assert len(results) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestGateCheckerEdgeCases:
    """エッジケースのテスト"""

    def test_boundary_values_pass(self):
        """境界値でちょうど通過する場合"""
        config = GateConfig(
            min_trades=30,
            max_drawdown_pct=25.0,
            min_sharpe_ratio=0.5,
            min_win_rate_pct=45.0,
            min_profit_factor=1.2,
        )
        metrics = StrategyMetrics(
            strategy_id="boundary",
            asset_id="SPY",
            trade_count=30,  # ちょうど30
            max_drawdown_pct=25.0,  # ちょうど25%
            expected_value=0.0001,  # わずかにプラス
            sharpe_ratio=0.5,  # ちょうど0.5
            win_rate_pct=45.0,  # ちょうど45%
            profit_factor=1.2,  # ちょうど1.2
        )
        checker = GateChecker(config)
        result = checker.check(metrics)
        assert result.passed is True

    def test_stability_exact_threshold(self):
        """安定性が閾値ちょうどの場合"""
        config = GateConfig(stability_periods=6, stability_min_positive=4)
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
            period_returns=[0.01, 0.02, 0.03, 0.04, -0.01, -0.02],  # 4/6プラス
        )
        checker = GateChecker(config)
        result = checker.check(metrics)
        # 4/6でちょうど閾値なので通過
        assert result.passed is True

    def test_stability_short_history(self):
        """period_returnsがstability_periodsより短い場合"""
        config = GateConfig(stability_periods=6, stability_min_positive=4)
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.002,
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
            period_returns=[0.01, 0.02, 0.03],  # 3期間のみ
        )
        checker = GateChecker(config)
        result = checker.check(metrics)
        # 3/3プラス < 4必要なので不通過
        assert result.passed is False
        assert any(r.gate_type == GateType.STABILITY and not r.passed
                   for r in result.results)

    def test_zero_expected_value_boundary(self):
        """expected_valueがちょうど0の場合は不通過"""
        metrics = StrategyMetrics(
            strategy_id="test",
            asset_id="SPY",
            trade_count=50,
            max_drawdown_pct=15.0,
            expected_value=0.0,  # ちょうど0
            sharpe_ratio=1.2,
            win_rate_pct=55.0,
            profit_factor=1.5,
        )
        checker = GateChecker()
        result = checker.check(metrics)
        assert result.passed is False
        assert any(r.gate_type == GateType.MIN_EXPECTED_VALUE and not r.passed
                   for r in result.results)
