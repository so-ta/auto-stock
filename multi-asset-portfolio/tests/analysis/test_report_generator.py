"""
Test ReportGenerator (task_046_3).

Tests:
1. PortfolioMetrics dataclass
2. ComparisonResult dataclass
3. ReportGenerator.generate_text_report()
4. ReportGenerator.generate_html_report()
5. Format helpers
"""

import pytest
from pathlib import Path
import tempfile

from src.analysis.report_generator import (
    PortfolioMetrics,
    ComparisonResult,
    ReportGenerator,
)


class TestPortfolioMetrics:
    """Test PortfolioMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = PortfolioMetrics()

        assert metrics.total_return == 0.0
        assert metrics.annual_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    def test_custom_values(self):
        """Test with custom values."""
        metrics = PortfolioMetrics(
            annual_return=0.125,
            sharpe_ratio=0.85,
            max_drawdown=-0.182,
            volatility=0.15,
        )

        assert metrics.annual_return == 0.125
        assert metrics.sharpe_ratio == 0.85
        assert metrics.max_drawdown == -0.182
        assert metrics.volatility == 0.15

    def test_to_dict(self):
        """Test to_dict() method."""
        metrics = PortfolioMetrics(
            annual_return=0.10,
            sharpe_ratio=0.75,
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["annual_return"] == 0.10
        assert d["sharpe_ratio"] == 0.75


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        portfolio = PortfolioMetrics(annual_return=0.12, sharpe_ratio=0.85)
        spy = PortfolioMetrics(annual_return=0.10, sharpe_ratio=0.72)
        qqq = PortfolioMetrics(annual_return=0.15, sharpe_ratio=0.82)

        comparison = ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy, "QQQ": qqq},
        )

        assert comparison.portfolio_metrics.annual_return == 0.12
        assert len(comparison.benchmarks) == 2
        assert "SPY" in comparison.benchmarks
        assert "QQQ" in comparison.benchmarks

    def test_get_benchmark(self):
        """Test get_benchmark() method."""
        portfolio = PortfolioMetrics()
        spy = PortfolioMetrics(annual_return=0.10)

        comparison = ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy},
        )

        assert comparison.get_benchmark("SPY").annual_return == 0.10
        assert comparison.get_benchmark("NONEXISTENT") is None

    def test_to_dict(self):
        """Test to_dict() method."""
        portfolio = PortfolioMetrics(annual_return=0.12)
        spy = PortfolioMetrics(annual_return=0.10)

        comparison = ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy},
            portfolio_name="Test Portfolio",
            start_date="2020-01-01",
            end_date="2025-01-01",
        )

        d = comparison.to_dict()

        assert d["portfolio"]["annual_return"] == 0.12
        assert d["benchmarks"]["SPY"]["annual_return"] == 0.10
        assert d["portfolio_name"] == "Test Portfolio"


class TestReportGeneratorTextReport:
    """Test ReportGenerator text report generation."""

    @pytest.fixture
    def sample_comparison(self):
        """Create sample comparison for testing."""
        portfolio = PortfolioMetrics(
            annual_return=0.125,
            sharpe_ratio=0.85,
            sortino_ratio=1.2,
            max_drawdown=-0.182,
            volatility=0.15,
            calmar_ratio=0.68,
        )
        spy = PortfolioMetrics(
            annual_return=0.102,
            sharpe_ratio=0.72,
            sortino_ratio=0.95,
            max_drawdown=-0.339,
            volatility=0.18,
            calmar_ratio=0.30,
        )
        qqq = PortfolioMetrics(
            annual_return=0.148,
            sharpe_ratio=0.82,
            sortino_ratio=1.1,
            max_drawdown=-0.351,
            volatility=0.22,
            calmar_ratio=0.42,
        )

        return ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy, "QQQ": qqq},
            start_date="2010-01-01",
            end_date="2025-01-01",
        )

    def test_generate_text_report(self, sample_comparison):
        """Test text report generation."""
        generator = ReportGenerator()

        text = generator.generate_text_report(
            sample_comparison,
            portfolio_name="My Portfolio",
        )

        # ヘッダー確認
        assert "My Portfolio" in text
        assert "2010-01-01" in text
        assert "2025-01-01" in text

        # メトリクス確認
        assert "年率リターン" in text
        assert "シャープレシオ" in text
        assert "最大ドローダウン" in text

        # 値確認（パーセント表示）
        assert "12.50%" in text  # portfolio annual_return
        assert "0.85" in text    # portfolio sharpe_ratio

    def test_generate_text_report_single_benchmark(self):
        """Test with single benchmark."""
        portfolio = PortfolioMetrics(annual_return=0.10)
        spy = PortfolioMetrics(annual_return=0.08)

        comparison = ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy},
        )

        generator = ReportGenerator()
        text = generator.generate_text_report(comparison, "Test")

        assert "SPY" in text
        assert "QQQ" not in text


class TestReportGeneratorHTMLReport:
    """Test ReportGenerator HTML report generation."""

    @pytest.fixture
    def sample_comparison(self):
        """Create sample comparison for testing."""
        portfolio = PortfolioMetrics(
            annual_return=0.125,
            sharpe_ratio=0.85,
            sortino_ratio=1.2,
            max_drawdown=-0.182,
            volatility=0.15,
            calmar_ratio=0.68,
        )
        spy = PortfolioMetrics(
            annual_return=0.102,
            sharpe_ratio=0.72,
            sortino_ratio=0.95,
            max_drawdown=-0.339,
            volatility=0.18,
            calmar_ratio=0.30,
        )

        return ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy},
        )

    def test_generate_html_report(self, sample_comparison):
        """Test HTML report generation."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            html = generator.generate_html_report(
                sample_comparison,
                portfolio_name="Test Portfolio",
                start_date="2020-01-01",
                end_date="2025-01-01",
                output_path=str(output_path),
            )

            # HTML内容確認
            assert "<!DOCTYPE html>" in html
            assert "Test Portfolio" in html
            assert "2020-01-01" in html

            # ファイル出力確認
            assert output_path.exists()
            content = output_path.read_text()
            assert "Test Portfolio" in content

    def test_generate_html_report_creates_directory(self, sample_comparison):
        """Test that HTML report creates parent directory."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "report.html"

            html = generator.generate_html_report(
                sample_comparison,
                portfolio_name="Test",
                start_date="2020-01-01",
                end_date="2025-01-01",
                output_path=str(output_path),
            )

            assert output_path.exists()


class TestReportGeneratorFormatters:
    """Test ReportGenerator format helpers."""

    def test_format_percentage(self):
        """Test percentage formatting."""
        generator = ReportGenerator()

        assert generator._format_percentage(0.125) == "12.50%"
        assert generator._format_percentage(-0.05) == "-5.00%"
        assert generator._format_percentage(0.0) == "0.00%"
        assert generator._format_percentage(None) == "N/A"

    def test_format_ratio(self):
        """Test ratio formatting."""
        generator = ReportGenerator()

        assert generator._format_ratio(0.85) == "0.85"
        assert generator._format_ratio(1.234) == "1.23"
        assert generator._format_ratio(-0.5) == "-0.50"
        assert generator._format_ratio(None) == "N/A"


class TestReportGeneratorDataCreation:
    """Test ReportGenerator internal data creation methods."""

    @pytest.fixture
    def generator_and_comparison(self):
        """Create generator and comparison for testing."""
        portfolio = PortfolioMetrics(
            annual_return=0.12,
            total_return=1.5,
            monthly_return=0.01,
            volatility=0.15,
            max_drawdown=-0.20,
            var_95=-0.02,
            sharpe_ratio=0.80,
            sortino_ratio=1.1,
            calmar_ratio=0.60,
        )
        spy = PortfolioMetrics(
            annual_return=0.10,
            total_return=1.2,
            monthly_return=0.008,
            volatility=0.18,
            max_drawdown=-0.30,
            var_95=-0.025,
            sharpe_ratio=0.70,
            sortino_ratio=0.90,
            calmar_ratio=0.33,
        )

        comparison = ComparisonResult(
            portfolio_metrics=portfolio,
            benchmark_metrics={"SPY": spy},
        )

        return ReportGenerator(), comparison

    def test_create_summary_section(self, generator_and_comparison):
        """Test summary section creation."""
        generator, comparison = generator_and_comparison

        summary = generator._create_summary_section(comparison)

        assert "portfolio" in summary
        assert "benchmarks" in summary
        assert summary["portfolio"]["annual_return"] == 0.12
        assert summary["benchmarks"]["SPY"]["annual_return"] == 0.10

    def test_create_returns_table(self, generator_and_comparison):
        """Test returns table creation."""
        generator, comparison = generator_and_comparison

        rows = generator._create_returns_table(comparison)

        assert len(rows) == 3  # 年率, トータル, 月次
        assert rows[0]["metric"] == "年率リターン"
        assert rows[0]["portfolio"] == 0.12
        assert rows[0]["benchmarks"]["SPY"] == 0.10

    def test_create_risk_table(self, generator_and_comparison):
        """Test risk table creation."""
        generator, comparison = generator_and_comparison

        rows = generator._create_risk_table(comparison)

        assert len(rows) == 3  # ボラ, MDD, VaR
        assert rows[0]["metric"] == "ボラティリティ"

    def test_create_ratio_table(self, generator_and_comparison):
        """Test ratio table creation."""
        generator, comparison = generator_and_comparison

        rows = generator._create_ratio_table(comparison)

        assert len(rows) == 3  # シャープ, ソルティノ, カルマー
        assert rows[0]["metric"] == "シャープレシオ"
        assert rows[0]["portfolio"] == 0.80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
