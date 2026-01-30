"""
tests/unit/test_risk.py - リスクモジュールの単体テスト

テスト対象:
- DrawdownProtector, DrawdownProtectorConfig (drawdown_protection.py)
- RiskBudgetingAllocator, RiskBudgetingConfig (risk_budgeting.py)

Task: task_032_10
Author: 足軽5号
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.drawdown_protection import (
    DrawdownProtector,
    DrawdownProtectorConfig,
    ProtectionStatus,
    RecoveryMode,
    create_drawdown_protector,
    calculate_protection_multiplier,
    quick_adjust_weights,
)
from src.risk.risk_budgeting import (
    RiskBudgetingAllocator,
    RiskBudgetingConfig,
    RiskContribution,
    create_risk_budgeting_allocator,
    quick_risk_budgeting,
    compute_risk_contribution,
)


# =============================================================================
# DrawdownProtectorConfig Tests
# =============================================================================

class TestDrawdownProtectorConfig:
    """DrawdownProtectorConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定でインスタンス化できる"""
        config = DrawdownProtectorConfig()
        assert config.dd_levels == [0.05, 0.10, 0.15, 0.20]
        assert config.risk_reductions == [0.9, 0.7, 0.5, 0.3]
        assert config.recovery_threshold == 0.5
        assert config.recovery_mode == RecoveryMode.THRESHOLD
        assert config.min_protection_days == 3
        assert config.emergency_dd_level == 0.25
        assert config.emergency_cash_ratio == 0.8

    def test_custom_config(self):
        """カスタム設定でインスタンス化できる"""
        config = DrawdownProtectorConfig(
            dd_levels=[0.03, 0.06, 0.10],
            risk_reductions=[0.95, 0.8, 0.6],
            recovery_threshold=0.3,
        )
        assert config.dd_levels == [0.03, 0.06, 0.10]
        assert config.risk_reductions == [0.95, 0.8, 0.6]
        assert config.recovery_threshold == 0.3

    def test_invalid_length_mismatch(self):
        """dd_levelsとrisk_reductionsの長さが異なる場合エラー"""
        with pytest.raises(ValueError, match="must have same length"):
            DrawdownProtectorConfig(
                dd_levels=[0.05, 0.10],
                risk_reductions=[0.9, 0.7, 0.5],
            )

    def test_invalid_dd_levels_range(self):
        """dd_levelsが(0, 1)の範囲外の場合エラー"""
        with pytest.raises(ValueError, match="dd_levels must be in"):
            DrawdownProtectorConfig(
                dd_levels=[0.0, 0.10],  # 0は無効
                risk_reductions=[0.9, 0.7],
            )
        with pytest.raises(ValueError, match="dd_levels must be in"):
            DrawdownProtectorConfig(
                dd_levels=[0.05, 1.0],  # 1は無効
                risk_reductions=[0.9, 0.7],
            )

    def test_invalid_risk_reductions_range(self):
        """risk_reductionsが(0, 1]の範囲外の場合エラー"""
        with pytest.raises(ValueError, match="risk_reductions must be in"):
            DrawdownProtectorConfig(
                dd_levels=[0.05, 0.10],
                risk_reductions=[0.0, 0.7],  # 0は無効
            )

    def test_invalid_dd_levels_order(self):
        """dd_levelsが昇順でない場合エラー"""
        with pytest.raises(ValueError, match="must be sorted ascending"):
            DrawdownProtectorConfig(
                dd_levels=[0.10, 0.05],  # 降順は無効
                risk_reductions=[0.7, 0.9],
            )

    def test_invalid_risk_reductions_order(self):
        """risk_reductionsが降順でない場合エラー"""
        with pytest.raises(ValueError, match="must be sorted descending"):
            DrawdownProtectorConfig(
                dd_levels=[0.05, 0.10],
                risk_reductions=[0.7, 0.9],  # 昇順は無効
            )

    def test_invalid_recovery_threshold(self):
        """recovery_thresholdが(0, 1]の範囲外の場合エラー"""
        with pytest.raises(ValueError, match="recovery_threshold must be in"):
            DrawdownProtectorConfig(recovery_threshold=0.0)
        with pytest.raises(ValueError, match="recovery_threshold must be in"):
            DrawdownProtectorConfig(recovery_threshold=1.5)


# =============================================================================
# DrawdownProtector Tests
# =============================================================================

class TestDrawdownProtector:
    """DrawdownProtectorのテスト"""

    def test_initialization_with_initial_value(self):
        """初期値でインスタンス化できる"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.get_state()
        assert state.hwm == 100000
        assert state.current_value == 100000
        assert state.drawdown == 0.0
        assert state.protection_level == 0
        assert state.status == ProtectionStatus.INACTIVE

    def test_initialization_without_initial_value(self):
        """初期値なしでインスタンス化できる"""
        protector = DrawdownProtector()
        state = protector.get_state()
        assert state.hwm == 0.0
        assert state.current_value == 0.0

    def test_initialize_method(self):
        """initialize()で初期化できる"""
        protector = DrawdownProtector()
        protector.initialize(50000)
        state = protector.get_state()
        assert state.hwm == 50000
        assert state.current_value == 50000

    def test_update_no_drawdown(self):
        """ドローダウンなしのupdate"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(105000)  # 上昇
        assert state.hwm == 105000  # HWM更新
        assert state.current_value == 105000
        assert state.drawdown == 0.0
        assert state.protection_level == 0

    def test_update_with_drawdown_level1(self):
        """5%ドローダウンでレベル1"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(95000)  # 5%下落
        assert state.hwm == 100000
        assert state.current_value == 95000
        assert state.drawdown == 0.05
        assert state.protection_level == 1
        assert state.status == ProtectionStatus.ACTIVE

    def test_update_with_drawdown_level2(self):
        """10%ドローダウンでレベル2"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(90000)
        assert state.drawdown == 0.10
        assert state.protection_level == 2

    def test_update_with_drawdown_level3(self):
        """15%ドローダウンでレベル3"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(85000)
        assert state.drawdown == 0.15
        assert state.protection_level == 3

    def test_update_with_drawdown_level4(self):
        """20%ドローダウンでレベル4"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(80000)
        assert state.drawdown == 0.20
        assert state.protection_level == 4

    def test_update_emergency_level(self):
        """25%以上で緊急レベル"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(74000)  # 26%下落
        assert state.protection_level == 5  # 緊急レベル
        assert state.status == ProtectionStatus.EMERGENCY

    def test_get_risk_multiplier_no_protection(self):
        """プロテクションなしの場合1.0"""
        protector = DrawdownProtector(initial_value=100000)
        assert protector.get_risk_multiplier() == 1.0

    def test_get_risk_multiplier_level1(self):
        """レベル1のリスク乗数"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(95000)
        assert protector.get_risk_multiplier() == 0.9

    def test_get_risk_multiplier_level2(self):
        """レベル2のリスク乗数"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(90000)
        assert protector.get_risk_multiplier() == 0.7

    def test_get_risk_multiplier_emergency(self):
        """緊急レベルのリスク乗数"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(74000)
        # emergency_cash_ratio=0.8 なので 1.0 - 0.8 = 0.2
        assert abs(protector.get_risk_multiplier() - 0.2) < 1e-9

    def test_adjust_weights_no_protection(self):
        """プロテクションなしの場合、重みは変わらない"""
        protector = DrawdownProtector(initial_value=100000)
        weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
        result = protector.adjust_weights(weights)
        assert result.risk_multiplier == 1.0
        pd.testing.assert_series_equal(result.adjusted_weights, weights)

    def test_adjust_weights_with_protection(self):
        """プロテクション時、重みが調整される"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(90000)  # 10% DD -> level 2, multiplier=0.7
        weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
        result = protector.adjust_weights(weights)
        assert result.risk_multiplier == 0.7
        # キャッシュが追加される
        assert "CASH" in result.adjusted_weights.index
        assert result.cash_added > 0
        # 合計は1
        assert abs(result.adjusted_weights.sum() - 1.0) < 1e-6

    def test_is_protection_active(self):
        """is_protection_activeの動作確認"""
        protector = DrawdownProtector(initial_value=100000)
        assert protector.is_protection_active() is False
        protector.update(95000)
        assert protector.is_protection_active() is True

    def test_reset(self):
        """resetで状態がリセットされる"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(90000)
        assert protector.get_state().protection_level == 2
        protector.reset(100000)
        state = protector.get_state()
        assert state.hwm == 100000
        assert state.protection_level == 0

    def test_get_history(self):
        """履歴が記録される"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(98000)
        protector.update(95000)
        protector.update(90000)
        history = protector.get_history()
        assert len(history) == 3

    def test_get_summary(self):
        """サマリーが取得できる"""
        protector = DrawdownProtector(initial_value=100000)
        protector.update(90000)
        summary = protector.get_summary()
        assert "status" in summary
        assert "protection_level" in summary
        assert "risk_multiplier" in summary
        assert "current_drawdown" in summary


# =============================================================================
# DrawdownProtector Helper Functions Tests
# =============================================================================

class TestDrawdownProtectorHelpers:
    """DrawdownProtector関連のヘルパー関数テスト"""

    def test_create_drawdown_protector(self):
        """create_drawdown_protectorファクトリ関数"""
        protector = create_drawdown_protector(
            dd_levels=[0.03, 0.06],
            risk_reductions=[0.95, 0.85],
            initial_value=100000,
        )
        assert protector.get_state().hwm == 100000
        assert protector.config.dd_levels == [0.03, 0.06]

    def test_calculate_protection_multiplier(self):
        """calculate_protection_multiplier便利関数"""
        assert calculate_protection_multiplier(0.0) == 1.0
        assert calculate_protection_multiplier(0.05) == 0.9
        assert calculate_protection_multiplier(0.08) == 0.9  # まだレベル1
        assert calculate_protection_multiplier(0.10) == 0.7
        assert calculate_protection_multiplier(0.20) == 0.3

    def test_quick_adjust_weights(self):
        """quick_adjust_weights便利関数"""
        weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
        adjusted = quick_adjust_weights(
            weights,
            portfolio_value=90000,
            hwm=100000,
        )
        # 10% DD -> multiplier 0.7
        assert "CASH" in adjusted.index or adjusted.sum() < weights.sum()


# =============================================================================
# RiskBudgetingConfig Tests
# =============================================================================

class TestRiskBudgetingConfig:
    """RiskBudgetingConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定でインスタンス化できる"""
        config = RiskBudgetingConfig()
        assert config.risk_budgets is None
        assert config.max_weight == 0.25
        assert config.min_weight == 0.01
        assert config.tolerance == 1e-10

    def test_custom_config(self):
        """カスタム設定でインスタンス化できる"""
        budgets = {"SPY": 0.5, "TLT": 0.5}
        config = RiskBudgetingConfig(
            risk_budgets=budgets,
            max_weight=0.4,
            min_weight=0.05,
        )
        assert config.risk_budgets == budgets
        assert config.max_weight == 0.4
        assert config.min_weight == 0.05


# =============================================================================
# RiskBudgetingAllocator Tests
# =============================================================================

class TestRiskBudgetingAllocator:
    """RiskBudgetingAllocatorのテスト"""

    @pytest.fixture
    def sample_returns(self):
        """サンプルリターンデータ"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # 相関のあるリターンを生成
        spy_returns = np.random.normal(0.0005, 0.01, 252)
        tlt_returns = np.random.normal(0.0002, 0.005, 252) - 0.3 * spy_returns
        gld_returns = np.random.normal(0.0003, 0.008, 252)
        return pd.DataFrame({
            "SPY": spy_returns,
            "TLT": tlt_returns,
            "GLD": gld_returns,
        }, index=dates)

    def test_initialization(self):
        """インスタンス化できる"""
        allocator = RiskBudgetingAllocator()
        assert allocator.max_weight == 0.25
        assert allocator.min_weight == 0.01

    def test_from_config(self):
        """from_configでインスタンス化できる"""
        config = RiskBudgetingConfig(max_weight=0.4)
        allocator = RiskBudgetingAllocator.from_config(config)
        assert allocator.max_weight == 0.4

    def test_compute_risk_contribution(self):
        """リスク貢献度計算"""
        allocator = RiskBudgetingAllocator()
        weights = np.array([0.5, 0.3, 0.2])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.0025, -0.001],
            [0.005, -0.001, 0.0064],
        ])
        rc = allocator.compute_risk_contribution(weights, cov)
        # リスク貢献度の合計は1
        assert abs(np.sum(rc) - 1.0) < 1e-6

    def test_optimize_equal_budgets(self, sample_returns):
        """均等リスクバジェット最適化"""
        allocator = RiskBudgetingAllocator(max_weight=0.5, min_weight=0.1)
        result = allocator.optimize(sample_returns)
        assert result.optimization_success
        # ウェイトの合計は1
        assert abs(result.weights.sum() - 1.0) < 1e-6
        # 均等リスクバジェットの場合、リスク貢献度も概ね均等
        rc_pct = result.risk_contributions.risk_contributions_pct
        assert all(rc_pct > 0.2)  # 各資産20%以上のリスク貢献

    def test_optimize_custom_budgets(self, sample_returns):
        """カスタムリスクバジェット最適化"""
        allocator = RiskBudgetingAllocator(max_weight=0.6, min_weight=0.1)
        target_budgets = {"SPY": 0.6, "TLT": 0.2, "GLD": 0.2}
        result = allocator.optimize(sample_returns, target_budgets=target_budgets)
        assert result.optimization_success
        # 目標バジェットの合計は1に正規化される
        assert abs(result.target_budgets.sum() - 1.0) < 1e-6

    def test_optimize_with_cov(self, sample_returns):
        """共分散行列を直接指定して最適化"""
        allocator = RiskBudgetingAllocator(max_weight=0.5)
        cov = sample_returns.cov()
        result = allocator.optimize(sample_returns, cov=cov)
        assert result.optimization_success

    def test_analyze_risk_contribution(self, sample_returns):
        """リスク貢献度分析"""
        allocator = RiskBudgetingAllocator()
        weights = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}
        analysis = allocator.analyze_risk_contribution(weights, sample_returns)
        assert "weight" in analysis.columns
        assert "risk_contribution" in analysis.columns
        assert "risk_contribution_pct" in analysis.columns
        assert "weight_to_risk_ratio" in analysis.columns

    def test_risk_contribution_to_dict(self, sample_returns):
        """RiskContributionのto_dict"""
        allocator = RiskBudgetingAllocator(max_weight=0.5)
        result = allocator.optimize(sample_returns)
        rc_dict = result.risk_contributions.to_dict()
        assert "weights" in rc_dict
        assert "risk_contributions" in rc_dict
        assert "portfolio_volatility" in rc_dict

    def test_risk_budgeting_result_to_dict(self, sample_returns):
        """RiskBudgetingResultのto_dict"""
        allocator = RiskBudgetingAllocator(max_weight=0.5)
        result = allocator.optimize(sample_returns)
        result_dict = result.to_dict()
        assert "weights" in result_dict
        assert "optimization_success" in result_dict
        assert "objective_value" in result_dict


# =============================================================================
# RiskBudgeting Helper Functions Tests
# =============================================================================

class TestRiskBudgetingHelpers:
    """RiskBudgeting関連のヘルパー関数テスト"""

    @pytest.fixture
    def sample_returns(self):
        """サンプルリターンデータ"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        return pd.DataFrame({
            "SPY": np.random.normal(0.0005, 0.01, 252),
            "TLT": np.random.normal(0.0002, 0.005, 252),
        }, index=dates)

    def test_create_risk_budgeting_allocator(self):
        """create_risk_budgeting_allocatorファクトリ関数"""
        allocator = create_risk_budgeting_allocator(max_weight=0.4)
        assert allocator.max_weight == 0.4

    def test_quick_risk_budgeting(self, sample_returns):
        """quick_risk_budgeting便利関数"""
        weights = quick_risk_budgeting(sample_returns, max_weight=0.6)
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_compute_risk_contribution_function(self, sample_returns):
        """compute_risk_contribution便利関数"""
        weights = {"SPY": 0.6, "TLT": 0.4}
        analysis = compute_risk_contribution(weights, sample_returns)
        assert isinstance(analysis, pd.DataFrame)
        assert len(analysis) == 2


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """エッジケースのテスト"""

    def test_drawdown_protector_zero_hwm(self):
        """HWMがゼロの場合のドローダウン計算"""
        protector = DrawdownProtector()
        state = protector.update(0)
        assert state.drawdown == 0.0

    def test_drawdown_protector_tiny_changes(self):
        """微小な価格変化"""
        protector = DrawdownProtector(initial_value=100000)
        state = protector.update(99999.99)
        assert state.protection_level == 0  # 0.00001%はレベル0

    def test_risk_budgeting_single_asset(self):
        """単一資産のリスクバジェッティング"""
        np.random.seed(42)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.0005, 0.01, 100),
        })
        allocator = RiskBudgetingAllocator(max_weight=1.0, min_weight=0.0)
        result = allocator.optimize(returns)
        # 単一資産は100%ウェイト
        assert abs(result.weights["SPY"] - 1.0) < 0.01

    def test_risk_budgeting_high_correlation(self):
        """高相関資産のリスクバジェッティング"""
        np.random.seed(42)
        base = np.random.normal(0.0005, 0.01, 100)
        returns = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.001, 100),  # 高相関
        })
        allocator = RiskBudgetingAllocator(max_weight=0.7, min_weight=0.1)
        result = allocator.optimize(returns)
        assert result.optimization_success
