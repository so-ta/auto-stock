"""
Tests for PerformanceComparator class.

Verifies:
1. Tracking Error calculation
2. Information Ratio calculation
3. Beta calculation
4. Alpha calculation
5. Up/Down Capture ratios
6. ComparisonResult structure
7. Edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.performance_comparator import (
    ComparisonResult,
    PerformanceComparator,
)


@pytest.fixture
def sample_returns():
    """Create sample return data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")

    # Portfolio: slightly better than benchmark with some tracking error
    portfolio = pd.Series(
        np.random.randn(252) * 0.01 + 0.0003,  # ~7.5% annual return
        index=dates,
        name="portfolio",
    )

    # Benchmark (SPY-like): market returns
    benchmark = pd.Series(
        np.random.randn(252) * 0.01 + 0.0002,  # ~5% annual return
        index=dates,
        name="SPY",
    )

    return portfolio, benchmark


@pytest.fixture
def multiple_benchmarks(sample_returns):
    """Create multiple benchmark DataFrame."""
    portfolio, spy = sample_returns
    np.random.seed(123)

    # Additional benchmarks
    qqq = pd.Series(
        np.random.randn(252) * 0.012 + 0.0003,  # Higher vol, higher return
        index=spy.index,
        name="QQQ",
    )

    agg = pd.Series(
        np.random.randn(252) * 0.003 + 0.0001,  # Low vol bond-like
        index=spy.index,
        name="AGG",
    )

    return pd.DataFrame({"SPY": spy, "QQQ": qqq, "AGG": agg})


@pytest.fixture
def comparator():
    """Create PerformanceComparator instance."""
    return PerformanceComparator(risk_free_rate=0.02)


class TestTrackingError:
    """Test Tracking Error calculation."""

    def test_tracking_error_basic(self, comparator, sample_returns):
        """Test basic tracking error calculation."""
        portfolio, benchmark = sample_returns

        te = comparator.calculate_tracking_error(portfolio, benchmark)

        assert not np.isnan(te)
        assert te > 0  # Tracking error should be positive
        # For random data with ~1% daily vol, TE should be reasonable
        assert 0 < te < 0.5  # Less than 50% annual TE

    def test_tracking_error_identical_returns(self, comparator):
        """Test tracking error when returns are identical."""
        dates = pd.date_range("2024-01-01", periods=100)
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

        te = comparator.calculate_tracking_error(returns, returns)

        assert te == 0 or np.isclose(te, 0, atol=1e-10)

    def test_tracking_error_annualization(self, comparator):
        """Test that tracking error is properly annualized."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        portfolio = pd.Series(np.random.randn(252) * 0.01, index=dates)
        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)

        te = comparator.calculate_tracking_error(portfolio, benchmark)

        # Manual calculation
        excess = portfolio - benchmark
        daily_te = excess.std()
        expected_te = daily_te * np.sqrt(252)

        assert np.isclose(te, expected_te, rtol=1e-10)


class TestInformationRatio:
    """Test Information Ratio calculation."""

    def test_information_ratio_basic(self, comparator, sample_returns):
        """Test basic information ratio calculation."""
        portfolio, benchmark = sample_returns

        ir = comparator.calculate_information_ratio(portfolio, benchmark)

        assert not np.isnan(ir)
        # IR can be positive or negative
        assert -5 < ir < 5  # Reasonable range

    def test_information_ratio_positive_excess(self, comparator):
        """Test IR when portfolio consistently outperforms."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)
        portfolio = benchmark + 0.0005  # Consistent outperformance

        ir = comparator.calculate_information_ratio(portfolio, benchmark)

        assert ir > 0  # Positive excess return should give positive IR

    def test_information_ratio_formula(self, comparator):
        """Test that IR formula is correct: annualized excess / TE."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        portfolio = pd.Series(np.random.randn(252) * 0.01 + 0.001, index=dates)
        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)

        ir = comparator.calculate_information_ratio(portfolio, benchmark)
        te = comparator.calculate_tracking_error(portfolio, benchmark)

        # Manual calculation
        excess = portfolio - benchmark
        annualized_excess = excess.mean() * 252
        expected_ir = annualized_excess / te

        assert np.isclose(ir, expected_ir, rtol=1e-10)


class TestBeta:
    """Test Beta calculation."""

    def test_beta_basic(self, comparator, sample_returns):
        """Test basic beta calculation."""
        portfolio, benchmark = sample_returns

        beta = comparator.calculate_beta(portfolio, benchmark)

        assert not np.isnan(beta)
        # Beta should be in reasonable range for equity portfolio
        assert -2 < beta < 3

    def test_beta_perfect_correlation(self, comparator):
        """Test beta when portfolio = 2x benchmark."""
        dates = pd.date_range("2024-01-01", periods=100)
        benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates)
        portfolio = benchmark * 2  # 2x leverage

        beta = comparator.calculate_beta(portfolio, benchmark)

        assert np.isclose(beta, 2.0, rtol=1e-10)

    def test_beta_formula(self, comparator):
        """Test that beta formula is correct: cov / var."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        portfolio = pd.Series(np.random.randn(252) * 0.01, index=dates)
        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)

        beta = comparator.calculate_beta(portfolio, benchmark)

        # Manual calculation
        expected_beta = portfolio.cov(benchmark) / benchmark.var()

        assert np.isclose(beta, expected_beta, rtol=1e-10)


class TestAlpha:
    """Test Alpha calculation."""

    def test_alpha_basic(self, comparator, sample_returns):
        """Test basic alpha calculation."""
        portfolio, benchmark = sample_returns

        alpha = comparator.calculate_alpha(portfolio, benchmark)

        assert not np.isnan(alpha)

    def test_alpha_outperformance(self, comparator):
        """Test alpha when portfolio consistently outperforms."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)
        # Portfolio with extra return not explained by beta
        portfolio = benchmark + 0.001  # ~25% annual excess

        alpha = comparator.calculate_alpha(portfolio, benchmark, risk_free_rate=0.02)

        # Alpha should be positive for outperformance
        assert alpha > 0

    def test_alpha_formula(self, comparator):
        """Test CAPM alpha formula."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        portfolio = pd.Series(np.random.randn(252) * 0.01 + 0.0003, index=dates)
        benchmark = pd.Series(np.random.randn(252) * 0.01 + 0.0002, index=dates)
        rf = 0.02

        alpha = comparator.calculate_alpha(portfolio, benchmark, risk_free_rate=rf)
        beta = comparator.calculate_beta(portfolio, benchmark)

        # Manual calculation
        port_annual = portfolio.mean() * 252
        bench_annual = benchmark.mean() * 252
        expected_return = rf + beta * (bench_annual - rf)
        expected_alpha = port_annual - expected_return

        assert np.isclose(alpha, expected_alpha, rtol=1e-10)


class TestUpDownCapture:
    """Test Up/Down Capture ratio calculations."""

    def test_up_capture_basic(self, comparator, sample_returns):
        """Test basic up capture calculation."""
        portfolio, benchmark = sample_returns

        up_capture = comparator.calculate_up_capture(portfolio, benchmark)

        assert not np.isnan(up_capture)
        assert up_capture > 0  # Should be positive

    def test_down_capture_basic(self, comparator, sample_returns):
        """Test basic down capture calculation."""
        portfolio, benchmark = sample_returns

        down_capture = comparator.calculate_down_capture(portfolio, benchmark)

        assert not np.isnan(down_capture)

    def test_defensive_portfolio(self, comparator):
        """Test captures for defensive portfolio (low beta)."""
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(42)

        benchmark = pd.Series(np.random.randn(252) * 0.01, index=dates)
        portfolio = benchmark * 0.5  # Low beta

        up_capture = comparator.calculate_up_capture(portfolio, benchmark)
        down_capture = comparator.calculate_down_capture(portfolio, benchmark)

        # Defensive should capture less on both sides
        assert up_capture < 1
        assert down_capture < 1


class TestComparisonResult:
    """Test ComparisonResult structure and methods."""

    def test_compare_single_benchmark(self, comparator, sample_returns):
        """Test compare with single benchmark."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)

        assert isinstance(result, ComparisonResult)
        assert "total_return" in result.portfolio_metrics
        assert "SPY" in result.benchmark_metrics
        assert "SPY" in result.relative_metrics
        assert "tracking_error" in result.relative_metrics["SPY"]

    def test_compare_multiple_benchmarks(
        self, comparator, sample_returns, multiple_benchmarks
    ):
        """Test compare with multiple benchmarks."""
        portfolio, _ = sample_returns

        result = comparator.compare(portfolio, multiple_benchmarks)

        assert len(result.benchmark_metrics) == 3
        assert "SPY" in result.relative_metrics
        assert "QQQ" in result.relative_metrics
        assert "AGG" in result.relative_metrics

    def test_comparison_result_to_dict(self, comparator, sample_returns):
        """Test ComparisonResult.to_dict() method."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)
        result_dict = result.to_dict()

        assert "portfolio" in result_dict
        assert "benchmarks" in result_dict
        assert "relative" in result_dict

    def test_comparison_result_summary(self, comparator, sample_returns):
        """Test ComparisonResult.summary() method."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)
        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Portfolio" in summary.index

    def test_comparison_result_relative_summary(
        self, comparator, sample_returns, multiple_benchmarks
    ):
        """Test ComparisonResult.relative_summary() method."""
        portfolio, _ = sample_returns

        result = comparator.compare(portfolio, multiple_benchmarks)
        rel_summary = result.relative_summary()

        assert isinstance(rel_summary, pd.DataFrame)
        assert "tracking_error" in rel_summary.columns
        assert "information_ratio" in rel_summary.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self, comparator):
        """Test with insufficient overlapping data."""
        dates1 = pd.date_range("2024-01-01", periods=10)
        dates2 = pd.date_range("2024-06-01", periods=10)  # No overlap

        portfolio = pd.Series(np.random.randn(10), index=dates1)
        benchmark = pd.DataFrame({"SPY": np.random.randn(10)}, index=dates2)

        with pytest.raises(ValueError, match="Insufficient"):
            comparator.compare(portfolio, benchmark)

    def test_nan_handling(self, comparator):
        """Test handling of NaN values."""
        dates = pd.date_range("2024-01-01", periods=100)
        portfolio = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates)

        # Add some NaNs
        benchmark.iloc[10:15] = np.nan

        # dropna is handled internally, should still work
        te = comparator.calculate_tracking_error(portfolio, benchmark.dropna())
        assert not np.isnan(te)

    def test_zero_variance_benchmark(self, comparator):
        """Test with zero variance benchmark."""
        dates = pd.date_range("2024-01-01", periods=100)
        portfolio = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark = pd.Series(np.zeros(100), index=dates)  # Zero variance

        beta = comparator.calculate_beta(portfolio, benchmark)
        assert np.isnan(beta)  # Should return NaN, not error

    def test_partial_overlap(self, comparator):
        """Test with partially overlapping dates."""
        dates1 = pd.date_range("2024-01-01", periods=100)
        dates2 = pd.date_range("2024-02-01", periods=100)  # 70 days overlap

        portfolio = pd.Series(np.random.randn(100) * 0.01, index=dates1)
        benchmark = pd.Series(np.random.randn(100) * 0.01, index=dates2)

        te = comparator.calculate_tracking_error(portfolio, benchmark)
        assert not np.isnan(te)  # Should work with overlapping portion


class TestAbsoluteMetrics:
    """Test absolute performance metrics calculation."""

    def test_total_return(self, comparator, sample_returns):
        """Test total return calculation."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)

        # Manual calculation
        expected_total = (1 + portfolio).prod() - 1

        assert np.isclose(
            result.portfolio_metrics["total_return"],
            expected_total,
            rtol=1e-10,
        )

    def test_volatility_annualization(self, comparator, sample_returns):
        """Test volatility is properly annualized."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)

        # Manual calculation
        expected_vol = portfolio.std() * np.sqrt(252)

        assert np.isclose(
            result.portfolio_metrics["volatility"],
            expected_vol,
            rtol=1e-10,
        )

    def test_max_drawdown(self, comparator, sample_returns):
        """Test max drawdown calculation."""
        portfolio, benchmark = sample_returns
        benchmark_df = benchmark.to_frame()

        result = comparator.compare(portfolio, benchmark_df)

        mdd = result.portfolio_metrics["max_drawdown"]

        assert mdd <= 0  # Drawdown should be negative or zero
        assert mdd >= -1  # Can't lose more than 100%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
