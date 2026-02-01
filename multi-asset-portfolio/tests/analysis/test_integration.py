"""
統合テスト（task_046_9）

BenchmarkFetcher → PerformanceComparator → ReportGenerator/ChartGenerator
の連携動作を確認する統合テスト。
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# テストフィクスチャ
# ═══════════════════════════════════════════════════════════════
@pytest.fixture
def sample_dates():
    """テスト用の日付範囲"""
    return pd.date_range("2020-01-01", "2023-12-31", freq="B")


@pytest.fixture
def sample_portfolio_returns(sample_dates):
    """テスト用ポートフォリオリターン"""
    np.random.seed(42)
    returns = np.random.randn(len(sample_dates)) * 0.01 + 0.0003
    return pd.Series(returns, index=sample_dates, name="portfolio")


@pytest.fixture
def sample_benchmark_prices(sample_dates):
    """テスト用ベンチマーク価格（モック用）"""
    np.random.seed(123)
    n = len(sample_dates)
    data = {
        "SPY": 100 * np.cumprod(1 + np.random.randn(n) * 0.008 + 0.0002),
        "QQQ": 100 * np.cumprod(1 + np.random.randn(n) * 0.012 + 0.0003),
        "DIA": 100 * np.cumprod(1 + np.random.randn(n) * 0.007 + 0.0002),
    }
    return pd.DataFrame(data, index=sample_dates)


@pytest.fixture
def sample_benchmark_returns(sample_benchmark_prices):
    """テスト用ベンチマークリターン"""
    return sample_benchmark_prices.pct_change().dropna()


@pytest.fixture
def temp_output_dir():
    """一時出力ディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ═══════════════════════════════════════════════════════════════
# モジュールインポートテスト
# ═══════════════════════════════════════════════════════════════
class TestModuleImports:
    """全モジュールのインポートテスト"""

    def test_import_benchmark_fetcher(self):
        """BenchmarkFetcherがインポートできる"""
        from src.analysis.benchmark_fetcher import (
            BenchmarkFetcher,
            BenchmarkFetcherError,
        )
        assert BenchmarkFetcher is not None
        assert BenchmarkFetcherError is not None

    def test_import_performance_comparator(self):
        """PerformanceComparatorがインポートできる"""
        from src.analysis.performance_comparator import (
            PerformanceComparator,
            ComparisonResult,
        )
        assert PerformanceComparator is not None
        assert ComparisonResult is not None

    def test_import_report_generator(self):
        """ReportGeneratorがインポートできる"""
        from src.analysis.report_generator import (
            ReportGenerator,
            PortfolioMetrics,
            ComparisonResult as ReportComparisonResult,
        )
        assert ReportGenerator is not None
        assert PortfolioMetrics is not None
        assert ReportComparisonResult is not None

    def test_import_chart_generator(self):
        """ChartGeneratorがインポートできる（plotly必要）"""
        plotly = pytest.importorskip("plotly", reason="plotly required")

        from src.analysis.chart_generator import (
            ChartGenerator,
            ChartGeneratorError,
        )
        assert ChartGenerator is not None
        assert ChartGeneratorError is not None

    def test_import_dashboard(self):
        """ダッシュボードモジュールがインポートできる"""
        from src.analysis.dashboard import (
            PerformanceDashboard,
            create_dashboard,
            HAS_DASH,
        )
        assert PerformanceDashboard is not None
        assert create_dashboard is not None


# ═══════════════════════════════════════════════════════════════
# エンドツーエンドテスト
# ═══════════════════════════════════════════════════════════════
class TestEndToEndFlow:
    """BenchmarkFetcher → PerformanceComparator → ReportGenerator 連携テスト"""

    def test_full_report_generation_flow(
        self,
        sample_portfolio_returns,
        sample_benchmark_prices,
        temp_output_dir,
    ):
        """フルレポート生成フロー"""
        from src.analysis.benchmark_fetcher import BenchmarkFetcher
        from src.analysis.performance_comparator import PerformanceComparator
        from src.analysis.report_generator import (
            ReportGenerator,
            PortfolioMetrics,
            ComparisonResult as ReportComparisonResult,
        )

        # 1. BenchmarkFetcherをモック
        with patch.object(BenchmarkFetcher, "fetch_benchmarks") as mock_fetch:
            mock_fetch.return_value = sample_benchmark_prices

            # Create a mock storage backend with required methods
            mock_backend = MagicMock()
            mock_backend.exists.return_value = False
            mock_backend.read_parquet.return_value = None
            mock_backend.write_parquet.return_value = None

            fetcher = BenchmarkFetcher(storage_backend=mock_backend)
            prices = fetcher.fetch_benchmarks("2020-01-01", "2023-12-31", ["SPY", "QQQ", "DIA"])

        # 2. リターン計算
        benchmark_returns = prices.pct_change().dropna()

        # インデックスを揃える
        common_idx = sample_portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio = sample_portfolio_returns.loc[common_idx]
        benchmarks = benchmark_returns.loc[common_idx]

        # 3. PerformanceComparatorで比較
        comparator = PerformanceComparator()
        comparison = comparator.compare(portfolio, benchmarks)

        assert comparison is not None
        assert "total_return" in comparison.portfolio_metrics
        assert "SPY" in comparison.benchmark_metrics

        # 4. ReportGeneratorでレポート生成
        # PerformanceComparatorの結果をReportGenerator用に変換
        portfolio_metrics = PortfolioMetrics(
            total_return=comparison.portfolio_metrics.get("total_return", 0),
            annual_return=comparison.portfolio_metrics.get("annualized_return", 0),
            volatility=comparison.portfolio_metrics.get("volatility", 0),
            max_drawdown=comparison.portfolio_metrics.get("max_drawdown", 0),
            sharpe_ratio=comparison.portfolio_metrics.get("sharpe_ratio", 0),
        )

        benchmark_metrics_dict = {}
        for name, metrics in comparison.benchmark_metrics.items():
            benchmark_metrics_dict[name] = PortfolioMetrics(
                total_return=metrics.get("total_return", 0),
                annual_return=metrics.get("annualized_return", 0),
                volatility=metrics.get("volatility", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
            )

        report_comparison = ReportComparisonResult(
            portfolio_metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics_dict,
            portfolio_name="Test Portfolio",
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

        generator = ReportGenerator()
        output_path = temp_output_dir / "test_report.html"

        html = generator.generate_html_report(
            comparison=report_comparison,
            portfolio_name="Test Portfolio",
            start_date="2020-01-01",
            end_date="2023-12-31",
            output_path=str(output_path),
        )

        # 5. 検証
        assert html is not None
        assert "Test Portfolio" in html
        assert output_path.exists()

    def test_comparator_to_report_metrics_conversion(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
    ):
        """PerformanceComparatorの結果をReportGenerator用に変換できる"""
        from src.analysis.performance_comparator import PerformanceComparator
        from src.analysis.report_generator import PortfolioMetrics

        comparator = PerformanceComparator()
        comparison = comparator.compare(sample_portfolio_returns, sample_benchmark_returns)

        # 変換
        pm = comparison.portfolio_metrics
        portfolio_metrics = PortfolioMetrics(
            total_return=pm.get("total_return", 0),
            annual_return=pm.get("annualized_return", 0),
            volatility=pm.get("volatility", 0),
            max_drawdown=pm.get("max_drawdown", 0),
            sharpe_ratio=pm.get("sharpe_ratio", 0),
        )

        assert portfolio_metrics.total_return != 0 or portfolio_metrics.sharpe_ratio != 0


# ═══════════════════════════════════════════════════════════════
# ChartGenerator連携テスト
# ═══════════════════════════════════════════════════════════════
class TestChartGeneratorIntegration:
    """ChartGenerator連携テスト"""

    @pytest.fixture(autouse=True)
    def skip_without_plotly(self):
        """plotlyがない場合はスキップ"""
        pytest.importorskip("plotly", reason="plotly required for chart tests")

    def test_chart_generation_and_save(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
        temp_output_dir,
    ):
        """グラフ生成→保存フロー"""
        from src.analysis.chart_generator import ChartGenerator

        generator = ChartGenerator()

        # 資産推移比較グラフ
        fig = generator.plot_equity_comparison(
            portfolio=sample_portfolio_returns,
            benchmarks=sample_benchmark_returns,
            title="統合テスト - 資産推移",
        )

        assert fig is not None

        # HTML保存
        output_path = generator.save_chart(
            fig=fig,
            output_path=temp_output_dir / "equity_comparison",
            format="html",
        )

        assert Path(output_path).exists()
        assert output_path.endswith(".html")

    def test_multiple_chart_types(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
        temp_output_dir,
    ):
        """複数チャートタイプの生成"""
        from src.analysis.chart_generator import ChartGenerator

        generator = ChartGenerator()

        # 各チャートタイプを生成
        charts = []

        # 1. 資産推移
        charts.append(generator.plot_equity_comparison(
            portfolio=sample_portfolio_returns,
            benchmarks=sample_benchmark_returns,
        ))

        # 2. ドローダウン
        portfolio_values = 100 * (1 + sample_portfolio_returns).cumprod()
        portfolio_dd = generator.calculate_drawdown(portfolio_values)
        benchmark_values = 100 * (1 + sample_benchmark_returns).cumprod()
        benchmark_dds = benchmark_values.apply(generator.calculate_drawdown)

        charts.append(generator.plot_drawdown_comparison(
            portfolio_dd=portfolio_dd,
            benchmark_dds=benchmark_dds,
        ))

        # 3. 月次ヒートマップ
        charts.append(generator.plot_monthly_heatmap(returns=sample_portfolio_returns))

        # 4. ローリングシャープ
        charts.append(generator.plot_rolling_sharpe(
            returns=sample_portfolio_returns,
            window=60,  # テスト用に短いウィンドウ
        ))

        # 5. リターン分布
        charts.append(generator.plot_returns_distribution(returns=sample_portfolio_returns))

        # 全て生成できたことを確認
        assert len(charts) == 5
        for chart in charts:
            assert chart is not None

    def test_chart_with_comparator_results(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
        temp_output_dir,
    ):
        """PerformanceComparatorの結果を使ったチャート生成"""
        from src.analysis.performance_comparator import PerformanceComparator
        from src.analysis.chart_generator import ChartGenerator

        # 比較分析
        comparator = PerformanceComparator()
        comparison = comparator.compare(sample_portfolio_returns, sample_benchmark_returns)

        # チャート生成
        generator = ChartGenerator()

        fig = generator.plot_equity_comparison(
            portfolio=sample_portfolio_returns,
            benchmarks=sample_benchmark_returns,
            title=f"Sharpe: {comparison.portfolio_metrics.get('sharpe_ratio', 0):.2f}",
        )

        output_path = generator.save_chart(
            fig=fig,
            output_path=temp_output_dir / "with_metrics.html",
            format="html",
        )

        assert Path(output_path).exists()


# ═══════════════════════════════════════════════════════════════
# ダッシュボード起動テスト
# ═══════════════════════════════════════════════════════════════
class TestDashboardIntegration:
    """ダッシュボード連携テスト（モック使用）"""

    def test_dashboard_creation(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
    ):
        """ダッシュボード作成"""
        dash = pytest.importorskip("dash", reason="dash required for dashboard tests")
        dbc = pytest.importorskip(
            "dash_bootstrap_components",
            reason="dash-bootstrap-components required",
        )

        from src.analysis.dashboard import create_dashboard

        dashboard = create_dashboard(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            title="Integration Test Dashboard",
        )

        assert dashboard is not None
        assert dashboard.title == "Integration Test Dashboard"

    def test_dashboard_app_creation(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
    ):
        """Dashアプリ作成"""
        dash = pytest.importorskip("dash", reason="dash required")
        dbc = pytest.importorskip("dash_bootstrap_components", reason="dbc required")

        from src.analysis.dashboard import create_dashboard

        dashboard = create_dashboard(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
        )

        app = dashboard.create_app()
        assert app is not None
        assert hasattr(app, "layout")

    def test_dashboard_layout_contains_expected_elements(
        self,
        sample_portfolio_returns,
        sample_benchmark_returns,
    ):
        """ダッシュボードレイアウトに期待要素が含まれる"""
        dash = pytest.importorskip("dash", reason="dash required")
        dbc = pytest.importorskip("dash_bootstrap_components", reason="dbc required")

        from src.analysis.dashboard import create_dashboard

        dashboard = create_dashboard(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            title="Test Dashboard",
        )

        app = dashboard.create_app()

        # レイアウトが存在する
        assert app.layout is not None


# ═══════════════════════════════════════════════════════════════
# エッジケーステスト
# ═══════════════════════════════════════════════════════════════
class TestEdgeCases:
    """エッジケーステスト"""

    def test_empty_portfolio_returns(self):
        """空のポートフォリオリターン"""
        from src.analysis.performance_comparator import PerformanceComparator

        comparator = PerformanceComparator()
        empty_returns = pd.Series(dtype=float)
        benchmark_returns = pd.DataFrame({"SPY": [0.01, 0.02, -0.01]})

        with pytest.raises(ValueError):
            comparator.compare(empty_returns, benchmark_returns)

    def test_mismatched_indices(self):
        """インデックスが重ならない場合"""
        from src.analysis.performance_comparator import PerformanceComparator

        comparator = PerformanceComparator()

        portfolio = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=2),
        )
        benchmarks = pd.DataFrame(
            {"SPY": [0.01, 0.02]},
            index=pd.date_range("2025-01-01", periods=2),
        )

        with pytest.raises(ValueError, match="Insufficient"):
            comparator.compare(portfolio, benchmarks)

    def test_single_data_point(self):
        """データ点が1つのみ"""
        from src.analysis.performance_comparator import PerformanceComparator

        comparator = PerformanceComparator()

        idx = pd.date_range("2020-01-01", periods=1)
        portfolio = pd.Series([0.01], index=idx)
        benchmarks = pd.DataFrame({"SPY": [0.01]}, index=idx)

        with pytest.raises(ValueError):
            comparator.compare(portfolio, benchmarks)

    def test_benchmark_fetcher_invalid_dates(self):
        """BenchmarkFetcherに無効な日付"""
        from src.analysis.benchmark_fetcher import BenchmarkFetcher

        # Create a mock storage backend
        mock_backend = MagicMock()
        mock_backend.exists.return_value = False

        fetcher = BenchmarkFetcher(storage_backend=mock_backend)

        # 日付が逆順の場合はValueError
        with pytest.raises(ValueError, match="start_date must be before"):
            fetcher.fetch_benchmarks("2025-01-01", "2020-01-01")  # 逆順

    def test_chart_generator_empty_data(self):
        """ChartGeneratorに空データ"""
        plotly = pytest.importorskip("plotly", reason="plotly required")

        from src.analysis.chart_generator import ChartGenerator, ChartGeneratorError

        generator = ChartGenerator()

        with pytest.raises(ChartGeneratorError, match="Empty"):
            generator.plot_monthly_heatmap(returns=pd.Series(dtype=float))


# ═══════════════════════════════════════════════════════════════
# BenchmarkFetcher モック統合テスト
# ═══════════════════════════════════════════════════════════════
class TestBenchmarkFetcherMocked:
    """BenchmarkFetcherのモック統合テスト"""

    @pytest.fixture
    def yfinance_mock_data(self, sample_dates):
        """yfinance形式のモックデータ（MultiIndex）"""
        np.random.seed(123)
        n = len(sample_dates)

        # MultiIndexのカラム構造（複数ティッカー用）
        columns = pd.MultiIndex.from_product(
            [["Adj Close", "Close", "Open", "High", "Low", "Volume"], ["SPY", "QQQ"]],
            names=["Price", "Ticker"],
        )

        spy_price = 100 * np.cumprod(1 + np.random.randn(n) * 0.008 + 0.0002)
        qqq_price = 100 * np.cumprod(1 + np.random.randn(n) * 0.012 + 0.0003)

        data = np.column_stack([
            spy_price, qqq_price,  # Adj Close
            spy_price, qqq_price,  # Close
            spy_price * 0.99, qqq_price * 0.99,  # Open
            spy_price * 1.01, qqq_price * 1.01,  # High
            spy_price * 0.98, qqq_price * 0.98,  # Low
            np.random.randint(1000000, 10000000, n), np.random.randint(500000, 5000000, n),  # Volume
        ])

        return pd.DataFrame(data, index=sample_dates, columns=columns)

    @pytest.fixture
    def yfinance_mock_data_single(self, sample_dates):
        """yfinance形式のモックデータ（シングルティッカー用）"""
        np.random.seed(123)
        n = len(sample_dates)

        spy_price = 100 * np.cumprod(1 + np.random.randn(n) * 0.008 + 0.0002)

        data = {
            "Adj Close": spy_price,
            "Close": spy_price,
            "Open": spy_price * 0.99,
            "High": spy_price * 1.01,
            "Low": spy_price * 0.98,
            "Volume": np.random.randint(1000000, 10000000, n),
        }

        return pd.DataFrame(data, index=sample_dates)

    def test_fetch_and_calculate_returns(self, yfinance_mock_data):
        """データ取得とリターン計算"""
        from src.analysis.benchmark_fetcher import BenchmarkFetcher

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = yfinance_mock_data

            # Create a mock storage backend
            mock_backend = MagicMock()
            mock_backend.exists.return_value = False

            fetcher = BenchmarkFetcher(storage_backend=mock_backend)
            prices = fetcher.fetch_benchmarks("2020-01-01", "2023-12-31", ["SPY", "QQQ"])
            returns = fetcher.calculate_returns(prices, frequency="daily")

        assert len(returns) > 0
        assert "SPY" in returns.columns
        assert "QQQ" in returns.columns

    def test_fetch_and_get_stats(self, yfinance_mock_data_single):
        """データ取得と統計計算"""
        from src.analysis.benchmark_fetcher import BenchmarkFetcher

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = yfinance_mock_data_single

            # Create a mock storage backend
            mock_backend = MagicMock()
            mock_backend.exists.return_value = False

            fetcher = BenchmarkFetcher(storage_backend=mock_backend)
            prices = fetcher.fetch_benchmarks("2020-01-01", "2023-12-31", ["SPY"])
            returns = fetcher.calculate_returns(prices)
            stats = fetcher.get_benchmark_stats(returns)

        assert "annual_return" in stats.columns
        assert "sharpe_ratio" in stats.columns
        assert "max_drawdown" in stats.columns
