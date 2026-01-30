"""
ChartGenerator テスト（task_046_4）

Plotlyグラフ生成のテスト。
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# plotlyがインストールされていない場合はテストをスキップ
plotly = pytest.importorskip("plotly", reason="plotly is required for chart tests")

from src.analysis.chart_generator import (
    ChartGenerator,
    ChartGeneratorError,
)


@pytest.fixture
def temp_output_dir():
    """一時出力ディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_returns():
    """テスト用リターンデータ"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    np.random.seed(42)
    return pd.Series(
        np.random.randn(len(dates)) * 0.01,
        index=dates,
        name="portfolio",
    )


@pytest.fixture
def sample_portfolio_values():
    """テスト用ポートフォリオ価値"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.01
    values = 100 * np.cumprod(1 + returns)
    return pd.Series(values, index=dates, name="portfolio")


@pytest.fixture
def sample_benchmark_returns():
    """テスト用ベンチマークリターン"""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    np.random.seed(123)

    data = {
        "SPY": np.random.randn(len(dates)) * 0.008,
        "QQQ": np.random.randn(len(dates)) * 0.012,
        "DIA": np.random.randn(len(dates)) * 0.006,
    }

    return pd.DataFrame(data, index=dates)


class TestChartGeneratorInit:
    """ChartGenerator 初期化テスト"""

    def test_init_default(self):
        """デフォルト設定で初期化"""
        generator = ChartGenerator()
        assert generator._template == "plotly_white"

    def test_init_with_template(self):
        """カスタムテンプレートで初期化"""
        generator = ChartGenerator(template="plotly_dark")
        assert generator._template == "plotly_dark"

    def test_init_with_custom_colors(self):
        """カスタムカラーで初期化"""
        custom_colors = {"portfolio": "#000000"}
        generator = ChartGenerator(color_palette=custom_colors)
        assert generator._colors["portfolio"] == "#000000"


class TestPlotEquityComparison:
    """plot_equity_comparison テスト"""

    def test_equity_comparison_basic(self, sample_returns, sample_benchmark_returns):
        """基本的な資産推移比較グラフ"""
        generator = ChartGenerator()
        fig = generator.plot_equity_comparison(
            portfolio=sample_returns,
            benchmarks=sample_benchmark_returns,
        )

        assert fig is not None
        # ポートフォリオ + 3ベンチマーク = 4トレース
        assert len(fig.data) == 4

    def test_equity_comparison_portfolio_only(self, sample_returns):
        """ポートフォリオのみ"""
        generator = ChartGenerator()
        fig = generator.plot_equity_comparison(
            portfolio=sample_returns,
            benchmarks=None,
        )

        assert fig is not None
        assert len(fig.data) == 1

    def test_equity_comparison_custom_title(self, sample_returns, sample_benchmark_returns):
        """カスタムタイトル"""
        generator = ChartGenerator()
        fig = generator.plot_equity_comparison(
            portfolio=sample_returns,
            benchmarks=sample_benchmark_returns,
            title="カスタムタイトル",
        )

        assert fig.layout.title.text == "カスタムタイトル"

    def test_equity_comparison_with_values(self, sample_portfolio_values, sample_benchmark_returns):
        """累積値データで比較"""
        generator = ChartGenerator()
        fig = generator.plot_equity_comparison(
            portfolio=sample_portfolio_values,
            benchmarks=sample_benchmark_returns,
        )

        assert fig is not None


class TestPlotDrawdownComparison:
    """plot_drawdown_comparison テスト"""

    def test_drawdown_comparison_basic(self, sample_portfolio_values, sample_benchmark_returns):
        """基本的なドローダウン比較グラフ"""
        generator = ChartGenerator()

        # ドローダウン計算
        portfolio_dd = generator.calculate_drawdown(sample_portfolio_values)
        benchmark_values = 100 * (1 + sample_benchmark_returns).cumprod()
        benchmark_dds = benchmark_values.apply(generator.calculate_drawdown)

        fig = generator.plot_drawdown_comparison(
            portfolio_dd=portfolio_dd,
            benchmark_dds=benchmark_dds,
        )

        assert fig is not None
        assert len(fig.data) >= 1

    def test_drawdown_comparison_portfolio_only(self, sample_portfolio_values):
        """ポートフォリオのみのドローダウン"""
        generator = ChartGenerator()
        portfolio_dd = generator.calculate_drawdown(sample_portfolio_values)

        fig = generator.plot_drawdown_comparison(
            portfolio_dd=portfolio_dd,
            benchmark_dds=None,
        )

        assert fig is not None
        assert len(fig.data) == 1


class TestPlotMonthlyHeatmap:
    """plot_monthly_heatmap テスト"""

    def test_monthly_heatmap_basic(self, sample_returns):
        """基本的な月次ヒートマップ"""
        generator = ChartGenerator()
        fig = generator.plot_monthly_heatmap(returns=sample_returns)

        assert fig is not None
        assert fig.data[0].type == "heatmap"

    def test_monthly_heatmap_custom_title(self, sample_returns):
        """カスタムタイトル"""
        generator = ChartGenerator()
        fig = generator.plot_monthly_heatmap(
            returns=sample_returns,
            title="カスタム月次リターン",
        )

        assert fig.layout.title.text == "カスタム月次リターン"

    def test_monthly_heatmap_empty_data(self):
        """空データでエラー"""
        generator = ChartGenerator()

        with pytest.raises(ChartGeneratorError, match="Empty returns data"):
            generator.plot_monthly_heatmap(returns=pd.Series(dtype=float))


class TestPlotRollingSharpe:
    """plot_rolling_sharpe テスト"""

    def test_rolling_sharpe_basic(self, sample_returns):
        """基本的なローリングシャープ"""
        generator = ChartGenerator()
        fig = generator.plot_rolling_sharpe(
            returns=sample_returns,
            window=60,  # テスト用に短いウィンドウ
        )

        assert fig is not None
        assert len(fig.data) == 1

    def test_rolling_sharpe_custom_window(self, sample_returns):
        """カスタムウィンドウ"""
        generator = ChartGenerator()
        fig = generator.plot_rolling_sharpe(
            returns=sample_returns,
            window=120,
        )

        assert fig is not None

    def test_rolling_sharpe_insufficient_data(self):
        """データ不足でエラー"""
        generator = ChartGenerator()
        short_returns = pd.Series(np.random.randn(50) * 0.01)

        with pytest.raises(ChartGeneratorError, match="Insufficient data"):
            generator.plot_rolling_sharpe(returns=short_returns, window=252)


class TestPlotReturnsDistribution:
    """plot_returns_distribution テスト"""

    def test_returns_distribution_basic(self, sample_returns):
        """基本的なリターン分布"""
        generator = ChartGenerator()
        fig = generator.plot_returns_distribution(returns=sample_returns)

        assert fig is not None
        assert fig.data[0].type == "histogram"

    def test_returns_distribution_custom_bins(self, sample_returns):
        """カスタムビン数"""
        generator = ChartGenerator()
        fig = generator.plot_returns_distribution(
            returns=sample_returns,
            bins=100,
        )

        assert fig is not None

    def test_returns_distribution_empty_data(self):
        """空データでエラー"""
        generator = ChartGenerator()

        with pytest.raises(ChartGeneratorError, match="Empty returns data"):
            generator.plot_returns_distribution(returns=pd.Series(dtype=float))


class TestPlotCorrelationMatrix:
    """plot_correlation_matrix テスト"""

    def test_correlation_matrix_basic(self, sample_benchmark_returns):
        """基本的な相関行列"""
        generator = ChartGenerator()
        fig = generator.plot_correlation_matrix(returns=sample_benchmark_returns)

        assert fig is not None
        assert fig.data[0].type == "heatmap"

    def test_correlation_matrix_empty_data(self):
        """空データでエラー"""
        generator = ChartGenerator()

        with pytest.raises(ChartGeneratorError, match="Empty returns data"):
            generator.plot_correlation_matrix(returns=pd.DataFrame())


class TestSaveChart:
    """save_chart テスト"""

    def test_save_html(self, temp_output_dir, sample_returns):
        """HTML保存"""
        generator = ChartGenerator()
        fig = generator.plot_returns_distribution(returns=sample_returns)

        output_path = generator.save_chart(
            fig=fig,
            output_path=Path(temp_output_dir) / "test_chart",
            format="html",
        )

        assert Path(output_path).exists()
        assert output_path.endswith(".html")

    def test_save_with_extension(self, temp_output_dir, sample_returns):
        """拡張子付きパス"""
        generator = ChartGenerator()
        fig = generator.plot_returns_distribution(returns=sample_returns)

        output_path = generator.save_chart(
            fig=fig,
            output_path=Path(temp_output_dir) / "test_chart.html",
            format="html",
        )

        assert Path(output_path).exists()

    def test_save_invalid_format(self, temp_output_dir, sample_returns):
        """無効なフォーマットでエラー"""
        generator = ChartGenerator()
        fig = generator.plot_returns_distribution(returns=sample_returns)

        with pytest.raises(ChartGeneratorError, match="Unsupported format"):
            generator.save_chart(
                fig=fig,
                output_path=Path(temp_output_dir) / "test_chart",
                format="invalid",
            )


class TestCalculateDrawdown:
    """calculate_drawdown テスト"""

    def test_calculate_drawdown_basic(self, sample_portfolio_values):
        """基本的なドローダウン計算"""
        dd = ChartGenerator.calculate_drawdown(sample_portfolio_values)

        assert len(dd) == len(sample_portfolio_values)
        assert dd.max() <= 0  # ドローダウンは0以下
        assert dd.iloc[0] == 0  # 最初は0

    def test_calculate_drawdown_monotonic_increase(self):
        """単調増加データ（ドローダウンなし）"""
        values = pd.Series([100, 110, 120, 130, 140])
        dd = ChartGenerator.calculate_drawdown(values)

        assert (dd == 0).all()

    def test_calculate_drawdown_monotonic_decrease(self):
        """単調減少データ（継続的なドローダウン）"""
        values = pd.Series([100, 90, 80, 70, 60])
        dd = ChartGenerator.calculate_drawdown(values)

        assert dd.iloc[0] == 0
        assert dd.iloc[-1] == -0.4  # 60/100 - 1 = -0.4
