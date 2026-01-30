"""Unit tests for strategy evaluation modules (WalkForward, Metrics, GateChecker)."""

import numpy as np
import pandas as pd
import pytest


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_metrics_to_dict(self):
        """Test PerformanceMetrics serialization."""
        from src.strategy.metrics import PerformanceMetrics

        metrics = PerformanceMetrics(
            expected_value=0.001,
            expected_value_gross=0.0015,
            volatility=0.15,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.12,
            var_95=0.025,
            es_95=0.03,
            turnover=0.5,
            n_trades=50,
            n_samples=252,
            win_rate=0.55,
            profit_factor=1.3,
            calmar_ratio=0.8,
            total_return=0.15,
            annualized_return=0.12,
        )

        d = metrics.to_dict()
        assert d["sharpe_ratio"] == 1.2
        assert d["max_drawdown"] == 0.12
        assert d["n_trades"] == 50


class TestGateConfig:
    """Tests for GateConfig."""

    def test_gate_config_defaults(self):
        """Test default gate configuration."""
        from src.strategy.metrics import GateConfig

        config = GateConfig()
        assert config.min_trades == 30
        assert config.max_mdd == 0.25
        assert config.min_expected_value == 0.0


class TestGateResult:
    """Tests for gate checking results."""

    def test_check_gates_all_pass(self):
        """Test gate check when all gates pass."""
        from src.strategy.metrics import GateConfig, GateStatus, PerformanceMetrics

        metrics = PerformanceMetrics(
            expected_value=0.001,
            expected_value_gross=0.0015,
            volatility=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=0.15,
            var_95=0.025,
            es_95=0.03,
            turnover=0.3,
            n_trades=50,
            n_samples=252,
            win_rate=0.55,
            profit_factor=1.3,
            calmar_ratio=1.0,
            total_return=0.15,
            annualized_return=0.12,
        )

        config = GateConfig(
            min_trades=30,
            max_mdd=0.25,
            min_expected_value=0.0,
            min_sharpe=0.5,
        )

        results = metrics.check_gates(config)
        assert all(r.status == GateStatus.PASS for r in results)
        assert metrics.passes_all_gates(config) is True

    def test_check_gates_fail_mdd(self):
        """Test gate check when MDD exceeds threshold."""
        from src.strategy.metrics import GateConfig, GateStatus, PerformanceMetrics

        metrics = PerformanceMetrics(
            expected_value=0.001,
            expected_value_gross=0.0015,
            volatility=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=0.35,  # Exceeds max
            var_95=0.025,
            es_95=0.03,
            turnover=0.3,
            n_trades=50,
            n_samples=252,
            win_rate=0.55,
            profit_factor=1.3,
            calmar_ratio=0.5,
            total_return=0.15,
            annualized_return=0.12,
        )

        config = GateConfig(max_mdd=0.25)
        results = metrics.check_gates(config)

        mdd_result = next(r for r in results if r.gate_name == "max_mdd")
        assert mdd_result.status == GateStatus.FAIL
        assert metrics.passes_all_gates(config) is False

    def test_check_gates_fail_trades(self):
        """Test gate check when trade count is too low."""
        from src.strategy.metrics import GateConfig, GateStatus, PerformanceMetrics

        metrics = PerformanceMetrics(
            expected_value=0.001,
            expected_value_gross=0.0015,
            volatility=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=0.15,
            var_95=0.025,
            es_95=0.03,
            turnover=0.3,
            n_trades=10,  # Below minimum
            n_samples=50,
            win_rate=0.55,
            profit_factor=1.3,
            calmar_ratio=1.0,
            total_return=0.05,
            annualized_return=0.04,
        )

        config = GateConfig(min_trades=30)
        results = metrics.check_gates(config)

        trades_result = next(r for r in results if r.gate_name == "min_trades")
        assert trades_result.status == GateStatus.FAIL


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns

    @pytest.fixture
    def sample_positions(self):
        """Create sample position series."""
        np.random.seed(42)
        return np.random.uniform(-1, 1, 252)

    def test_calculate_metrics(self, sample_returns, sample_positions):
        """Test basic metrics calculation."""
        from src.strategy.metrics import MetricsCalculator

        calc = MetricsCalculator()
        metrics = calc.calculate(sample_returns, sample_positions)

        assert metrics.n_samples == 252
        assert -1 < metrics.sharpe_ratio < 3  # Reasonable range
        assert 0 <= metrics.max_drawdown <= 1
        assert 0 <= metrics.win_rate <= 1

    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        from src.strategy.metrics import MetricsCalculator

        calc = MetricsCalculator(annualization_factor=252)
        metrics = calc.calculate(sample_returns)

        # Sharpe = mean(returns) / std(returns) * sqrt(252)
        expected_sharpe = (
            np.mean(sample_returns) / np.std(sample_returns, ddof=1) * np.sqrt(252)
        )
        assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, rel=0.01)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        from src.strategy.metrics import MetricsCalculator

        # Create returns with known drawdown
        # 100 -> 120 -> 90 -> 100
        # DD = (120-90)/120 = 25%
        returns = np.array([0.2, -0.25, 0.111])

        calc = MetricsCalculator()
        metrics = calc.calculate(returns)

        assert metrics.max_drawdown == pytest.approx(0.25, rel=0.01)

    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        from src.strategy.metrics import MetricsCalculator

        calc = MetricsCalculator()
        metrics = calc.calculate(sample_returns)

        # Sortino should be >= Sharpe for same data
        # (uses only downside deviation)
        assert metrics.sortino_ratio >= metrics.sharpe_ratio * 0.5

    def test_calculate_var_es(self, sample_returns):
        """Test VaR and ES calculation."""
        from src.strategy.metrics import MetricsCalculator

        calc = MetricsCalculator(var_percentile=0.05)
        metrics = calc.calculate(sample_returns)

        assert metrics.var_95 >= 0  # VaR is positive (loss)
        assert metrics.es_95 >= metrics.var_95  # ES >= VaR

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        from src.strategy.metrics import MetricsCalculator

        # 3 wins of 0.1, 2 losses of 0.05
        # PF = 0.3 / 0.1 = 3.0
        returns = np.array([0.1, 0.1, 0.1, -0.05, -0.05])

        calc = MetricsCalculator()
        metrics = calc.calculate(returns)

        assert metrics.profit_factor == pytest.approx(3.0, rel=0.01)

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        from src.strategy.metrics import MetricsCalculator

        returns = np.array([0.1, 0.1, 0.1, -0.05, -0.05])  # 60% win rate

        calc = MetricsCalculator()
        metrics = calc.calculate(returns)

        assert metrics.win_rate == pytest.approx(0.6, rel=0.01)

    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        from src.strategy.metrics import MetricsCalculator

        calc = MetricsCalculator()
        metrics = calc.calculate(np.array([]))

        assert metrics.n_samples == 0
        assert metrics.sharpe_ratio == 0.0


class TestCheckStability:
    """Tests for stability checking function."""

    def test_stable_returns(self):
        """Test stability check with stable returns."""
        from src.strategy.metrics import check_stability

        # 5 periods, 4 positive
        period_returns = [0.05, 0.03, -0.02, 0.04, 0.06]

        is_stable, neg_count = check_stability(
            period_returns, max_negative_periods=2, window=5
        )

        assert is_stable is True
        assert neg_count == 1

    def test_unstable_returns(self):
        """Test stability check with unstable returns."""
        from src.strategy.metrics import check_stability

        # 5 periods, 4 negative
        period_returns = [-0.05, -0.03, -0.02, 0.04, -0.06]

        is_stable, neg_count = check_stability(
            period_returns, max_negative_periods=2, window=5
        )

        assert is_stable is False
        assert neg_count == 4


class TestRollingMetrics:
    """Tests for rolling metrics calculation."""

    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        from src.strategy.metrics import calculate_rolling_metrics

        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=pd.date_range("2024-01-01", periods=100, freq="D"),
        )

        rolling = calculate_rolling_metrics(returns, window=20)

        assert "rolling_sharpe" in rolling.columns
        assert len(rolling) == 100
        # First 10 values should be NaN (min_periods = window // 2 = 10)
        assert rolling["rolling_sharpe"].iloc[:10].isna().sum() >= 5
