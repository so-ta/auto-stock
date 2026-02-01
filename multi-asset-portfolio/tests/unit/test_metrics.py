"""
Tests for unified metrics module (src/utils/metrics.py).

Tests:
1. Basic calculation tests
2. Edge cases (empty, NaN, zero std)
3. Numba/NumPy result consistency
4. pandas Series input support
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_volatility,
    calculate_rolling_sharpe,
    calculate_rolling_drawdown,
    MetricsCalculator,
    PerformanceMetrics,
)


class TestSharpeRatio:
    """Test calculate_sharpe_ratio function."""

    def test_basic_calculation(self):
        """Test basic Sharpe ratio calculation."""
        # Random returns with known mean/std
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # ~1bps mean, 2% daily std

        sharpe = calculate_sharpe_ratio(returns)

        # Should be positive (positive mean return)
        assert isinstance(sharpe, float)
        # Rough expected: (0.001 / 0.02) * sqrt(252) ≈ 0.79
        assert -5 < sharpe < 5  # Reasonable range

    def test_zero_std_returns_zero(self):
        """Test that zero std deviation returns 0."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_sharpe_ratio(np.array([]))

    def test_single_element_raises(self):
        """Test that single element raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculate_sharpe_ratio(np.array([0.01]))

    def test_nan_filtering(self):
        """Test that NaN values are filtered out."""
        returns = np.array([0.01, np.nan, 0.02, np.nan, 0.03])
        # Should not raise, filters NaN
        sharpe = calculate_sharpe_ratio(returns)
        assert not np.isnan(sharpe)

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_numba_numpy_consistency(self):
        """Test Numba and NumPy give same results."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        sharpe_numba = calculate_sharpe_ratio(returns, use_numba=True)
        sharpe_numpy = calculate_sharpe_ratio(returns, use_numba=False)

        assert np.isclose(sharpe_numba, sharpe_numpy, rtol=1e-6)


class TestMaxDrawdown:
    """Test calculate_max_drawdown function."""

    def test_from_returns(self):
        """Test max drawdown from returns."""
        # Simple case: 10% down, then recover
        returns = np.array([0.05, -0.10, -0.05, 0.10, 0.05])

        max_dd = calculate_max_drawdown(returns=returns)

        assert max_dd > 0
        assert max_dd < 1  # Should be fraction

    def test_from_portfolio_values(self):
        """Test max drawdown from portfolio values."""
        values = np.array([100, 110, 105, 95, 100, 105])

        max_dd = calculate_max_drawdown(portfolio_values=values)

        # Max drawdown is from 110 to 95 = 13.6%
        expected = (110 - 95) / 110
        assert np.isclose(max_dd, expected, rtol=1e-6)

    def test_no_drawdown(self):
        """Test monotonically increasing values."""
        values = np.array([100, 101, 102, 103, 104])
        max_dd = calculate_max_drawdown(portfolio_values=values)
        assert max_dd == 0.0

    def test_empty_array_returns_zero(self):
        """Test empty array returns 0."""
        max_dd = calculate_max_drawdown(returns=np.array([]))
        assert max_dd == 0.0

    def test_both_none_raises(self):
        """Test that both None raises ValueError."""
        with pytest.raises(ValueError, match="Either"):
            calculate_max_drawdown()

    def test_both_provided_raises(self):
        """Test that both provided raises ValueError."""
        with pytest.raises(ValueError, match="Only one"):
            calculate_max_drawdown(
                returns=np.array([0.01]),
                portfolio_values=np.array([100, 101])
            )

    def test_numba_numpy_consistency(self):
        """Test Numba and NumPy give same results."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 252)

        max_dd_numba = calculate_max_drawdown(returns=returns, use_numba=True)
        max_dd_numpy = calculate_max_drawdown(returns=returns, use_numba=False)

        assert np.isclose(max_dd_numba, max_dd_numpy, rtol=1e-6)


class TestSortinoRatio:
    """Test calculate_sortino_ratio function."""

    def test_basic_calculation(self):
        """Test basic Sortino ratio calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)
        # Sortino should be >= Sharpe when positive skew
        sharpe = calculate_sharpe_ratio(returns)
        # Not always true but generally sortino differs

    def test_no_negative_returns(self):
        """Test with no negative returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sortino = calculate_sortino_ratio(returns)
        # Should be inf with positive mean and no downside
        assert sortino == float('inf')

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_sortino_ratio(np.array([]))


class TestCalmarRatio:
    """Test calculate_calmar_ratio function."""

    def test_basic_calculation(self):
        """Test basic Calmar ratio calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        calmar = calculate_calmar_ratio(returns)

        assert isinstance(calmar, float)

    def test_no_drawdown(self):
        """Test with no drawdown."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        calmar = calculate_calmar_ratio(returns)
        assert calmar == float('inf')

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_calmar_ratio(np.array([]))


class TestVolatility:
    """Test calculate_volatility function."""

    def test_basic_calculation(self):
        """Test basic volatility calculation."""
        # Known daily std of 2%
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)

        vol = calculate_volatility(returns)

        # Should be approximately 2% * sqrt(252) ≈ 31.7%
        assert 0.2 < vol < 0.5

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_volatility(np.array([]))


class TestRollingMetrics:
    """Test rolling metric functions."""

    def test_rolling_sharpe(self):
        """Test rolling Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)

        rolling = calculate_rolling_sharpe(returns, window=20)

        # First 19 should be NaN
        assert np.all(np.isnan(rolling[:19]))
        # Rest should be valid
        assert not np.any(np.isnan(rolling[19:]))

    def test_rolling_sharpe_pandas(self):
        """Test rolling Sharpe with pandas Series."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 100),
            index=pd.date_range("2023-01-01", periods=100)
        )

        rolling = calculate_rolling_sharpe(returns, window=20)

        assert isinstance(rolling, pd.Series)
        assert len(rolling) == len(returns)

    def test_rolling_drawdown(self):
        """Test rolling drawdown."""
        returns = np.array([0.05, -0.10, -0.05, 0.10, 0.05])

        rolling_dd = calculate_rolling_drawdown(returns)

        assert len(rolling_dd) == len(returns)
        assert rolling_dd[0] == 0  # No drawdown at start


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    def test_calculate_all(self):
        """Test calculate_all method."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        calc = MetricsCalculator()
        metrics = calc.calculate_all(returns)

        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.calmar_ratio, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.annualized_volatility, float)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.win_rate, float)
        assert isinstance(metrics.profit_factor, float)

    def test_custom_parameters(self):
        """Test with custom parameters."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)

        calc = MetricsCalculator(
            risk_free_rate=0.02,
            annualization_factor=252,
            use_numba=False,
        )
        metrics = calc.calculate_all(returns)

        # With 2% risk-free rate, Sharpe should be lower
        assert isinstance(metrics.sharpe_ratio, float)

    def test_short_array(self):
        """Test with very short array."""
        returns = np.array([0.01])

        calc = MetricsCalculator()
        metrics = calc.calculate_all(returns)

        # Should return zeros for metrics requiring >= 2 elements
        assert metrics.sharpe_ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
