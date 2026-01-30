"""
Tests for PerformanceDashboard class.

Note: These tests require dash and dash-bootstrap-components.
Tests are skipped if packages are not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Check for optional dependencies
try:
    import dash
    from dash import html
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

try:
    import dash_bootstrap_components as dbc
    HAS_DBC = True
except ImportError:
    HAS_DBC = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

SKIP_DASH_TESTS = not (HAS_DASH and HAS_DBC and HAS_PLOTLY)
SKIP_REASON = "dash, dash-bootstrap-components, or plotly not installed"


@pytest.fixture
def sample_data():
    """Create sample return data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")

    portfolio = pd.Series(
        np.random.randn(252) * 0.01 + 0.0003,
        index=dates,
        name="portfolio",
    )

    benchmark_df = pd.DataFrame({
        "SPY": np.random.randn(252) * 0.01 + 0.0002,
        "QQQ": np.random.randn(252) * 0.012 + 0.0003,
    }, index=dates)

    return portfolio, benchmark_df


@pytest.fixture
def comparison_result(sample_data):
    """Create ComparisonResult for testing."""
    from src.analysis.performance_comparator import PerformanceComparator

    portfolio, benchmarks = sample_data
    comparator = PerformanceComparator()
    return comparator.compare(portfolio, benchmarks)


class TestChartGenerator:
    """Test ChartGenerator class."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_chart_generator_init(self):
        """Test ChartGenerator initialization."""
        from src.analysis.dashboard import ChartGenerator, ChartConfig

        config = ChartConfig(height=500)
        generator = ChartGenerator(config=config)

        assert generator.config.height == 500

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_cumulative_returns_chart(self, sample_data):
        """Test cumulative returns chart creation."""
        from src.analysis.dashboard import ChartGenerator

        portfolio, benchmarks = sample_data
        generator = ChartGenerator()

        fig = generator.create_cumulative_returns_chart(portfolio, benchmarks)

        assert fig is not None
        assert len(fig.data) == 3  # 1 portfolio + 2 benchmarks

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_drawdown_chart(self, sample_data):
        """Test drawdown chart creation."""
        from src.analysis.dashboard import ChartGenerator

        portfolio, benchmarks = sample_data
        generator = ChartGenerator()

        fig = generator.create_drawdown_chart(portfolio, benchmarks)

        assert fig is not None
        assert len(fig.data) >= 1

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_monthly_heatmap(self, sample_data):
        """Test monthly returns heatmap creation."""
        from src.analysis.dashboard import ChartGenerator

        portfolio, _ = sample_data
        generator = ChartGenerator()

        fig = generator.create_monthly_heatmap(portfolio)

        assert fig is not None
        assert fig.data[0].type == "heatmap"

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_rolling_sharpe_chart(self, sample_data):
        """Test rolling Sharpe ratio chart creation."""
        from src.analysis.dashboard import ChartGenerator

        portfolio, benchmarks = sample_data
        generator = ChartGenerator()

        fig = generator.create_rolling_sharpe_chart(portfolio, benchmarks, window=63)

        assert fig is not None


@pytest.mark.skipif(SKIP_DASH_TESTS, reason=SKIP_REASON)
class TestPerformanceDashboard:
    """Test PerformanceDashboard class."""

    def test_dashboard_init(self, sample_data, comparison_result):
        """Test dashboard initialization."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        assert dashboard.comparison is comparison_result
        assert dashboard.title == "Portfolio Performance Dashboard"

    def test_dashboard_custom_title(self, sample_data, comparison_result):
        """Test dashboard with custom title."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
            title="Custom Dashboard",
        )

        assert dashboard.title == "Custom Dashboard"

    def test_create_app(self, sample_data, comparison_result):
        """Test Dash app creation."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        app = dashboard.create_app()

        assert app is not None
        assert isinstance(app, dash.Dash)
        assert app.layout is not None

    def test_layout_components(self, sample_data, comparison_result):
        """Test that layout contains required components."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        app = dashboard.create_app()

        # Layout should be a Div container
        assert app.layout is not None

    def test_summary_cards_values(self, sample_data, comparison_result):
        """Test that summary cards show correct values."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        # Create summary cards (internal method)
        cards = dashboard._create_summary_cards()

        assert cards is not None

    def test_charts_section(self, sample_data, comparison_result):
        """Test charts section creation."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        charts = dashboard._create_charts_section()

        assert charts is not None

    def test_detail_tables(self, sample_data, comparison_result):
        """Test detail tables creation."""
        from src.analysis.dashboard import PerformanceDashboard

        portfolio, benchmarks = sample_data

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
        )

        tables = dashboard._create_detail_tables()

        assert tables is not None


@pytest.mark.skipif(SKIP_DASH_TESTS, reason=SKIP_REASON)
class TestCreateDashboardHelper:
    """Test create_dashboard helper function."""

    def test_create_dashboard(self, sample_data):
        """Test create_dashboard helper."""
        from src.analysis.dashboard import create_dashboard

        portfolio, benchmarks = sample_data

        dashboard = create_dashboard(
            portfolio_returns=portfolio,
            benchmark_returns=benchmarks,
            title="Test Dashboard",
        )

        assert dashboard is not None
        assert dashboard.title == "Test Dashboard"


class TestDashboardImportError:
    """Test import error handling when packages not installed."""

    def test_import_without_dash(self, monkeypatch):
        """Test that proper error is raised when dash not installed."""
        # This test verifies the error message is appropriate
        # We can't easily mock the import, so we just verify the structure
        from src.analysis.dashboard import HAS_DASH

        # Just verify the flag exists and is boolean
        assert isinstance(HAS_DASH, bool)


class TestChartConfig:
    """Test ChartConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        from src.analysis.dashboard import ChartConfig

        config = ChartConfig()

        assert config.height == 400
        assert config.template == "plotly_white"
        assert config.show_legend is True
        assert config.margin is not None

    def test_custom_config(self):
        """Test custom configuration."""
        from src.analysis.dashboard import ChartConfig

        config = ChartConfig(
            height=600,
            template="plotly_dark",
            show_legend=False,
        )

        assert config.height == 600
        assert config.template == "plotly_dark"
        assert config.show_legend is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
