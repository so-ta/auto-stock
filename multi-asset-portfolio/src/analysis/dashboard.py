"""
Performance Dashboard - インタラクティブなパフォーマンス分析ダッシュボード

Plotly Dashを使用したインタラクティブダッシュボード。
ポートフォリオとベンチマークの比較分析を視覚的に表示。

依存パッケージ:
    pip install dash>=2.14.0 dash-bootstrap-components>=1.5.0

Usage:
    from src.analysis.dashboard import PerformanceDashboard
    from src.analysis.performance_comparator import PerformanceComparator

    comparator = PerformanceComparator()
    result = comparator.compare(portfolio_returns, benchmark_df)

    dashboard = PerformanceDashboard(
        comparison=result,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_df,
    )
    dashboard.run(port=8050)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Optional imports for Dash
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    dash = None
    dcc = None
    html = None

try:
    import dash_bootstrap_components as dbc
    HAS_DBC = True
except ImportError:
    HAS_DBC = False
    dbc = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

if TYPE_CHECKING:
    from src.analysis.performance_comparator import ComparisonResult

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """チャート設定"""
    height: int = 400
    template: str = "plotly_white"
    show_legend: bool = True
    margin: Dict[str, int] = None

    def __post_init__(self):
        if self.margin is None:
            self.margin = {"l": 50, "r": 50, "t": 50, "b": 50}


class ChartGenerator:
    """チャート生成クラス（内蔵）"""

    def __init__(self, config: ChartConfig = None):
        if not HAS_PLOTLY:
            raise ImportError(
                "plotly is required. Install with: pip install plotly"
            )
        self.config = config or ChartConfig()

    def create_cumulative_returns_chart(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
    ) -> go.Figure:
        """累積リターンチャート"""
        fig = go.Figure()

        # ポートフォリオ
        cumulative_port = (1 + portfolio_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=cumulative_port.index,
            y=cumulative_port.values,
            name="Portfolio",
            line=dict(color="#2E86AB", width=2),
        ))

        # ベンチマーク
        colors = ["#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
        for i, col in enumerate(benchmark_returns.columns):
            cumulative = (1 + benchmark_returns[col]).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                name=col,
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            ))

        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            height=self.config.height,
            template=self.config.template,
            showlegend=self.config.show_legend,
            hovermode="x unified",
        )

        return fig

    def create_drawdown_chart(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
    ) -> go.Figure:
        """ドローダウンチャート"""
        fig = go.Figure()

        def calc_drawdown(returns: pd.Series) -> pd.Series:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.cummax()
            return (cumulative - rolling_max) / rolling_max

        # ポートフォリオ
        dd_port = calc_drawdown(portfolio_returns)
        fig.add_trace(go.Scatter(
            x=dd_port.index,
            y=dd_port.values * 100,
            name="Portfolio",
            fill="tozeroy",
            line=dict(color="#2E86AB", width=1),
            fillcolor="rgba(46, 134, 171, 0.3)",
        ))

        # ベンチマーク
        colors = ["#A23B72", "#F18F01"]
        for i, col in enumerate(benchmark_returns.columns[:2]):  # Max 2
            dd = calc_drawdown(benchmark_returns[col])
            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                name=col,
                line=dict(color=colors[i % len(colors)], width=1, dash="dash"),
            ))

        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=self.config.height,
            template=self.config.template,
            showlegend=self.config.show_legend,
            hovermode="x unified",
        )

        return fig

    def create_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
    ) -> go.Figure:
        """月次リターンヒートマップ"""
        # 月次リターンに変換
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        # ピボットテーブル
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text:.1f}%",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            height=self.config.height,
            template=self.config.template,
        )

        return fig

    def create_rolling_sharpe_chart(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
        window: int = 252,
        risk_free_rate: float = 0.02,
    ) -> go.Figure:
        """ローリングシャープレシオチャート"""
        fig = go.Figure()

        def calc_rolling_sharpe(returns: pd.Series) -> pd.Series:
            daily_rf = risk_free_rate / 252
            excess = returns - daily_rf
            rolling_mean = excess.rolling(window).mean() * 252
            rolling_std = returns.rolling(window).std() * np.sqrt(252)
            return rolling_mean / rolling_std

        # ポートフォリオ
        sharpe_port = calc_rolling_sharpe(portfolio_returns)
        fig.add_trace(go.Scatter(
            x=sharpe_port.index,
            y=sharpe_port.values,
            name="Portfolio",
            line=dict(color="#2E86AB", width=2),
        ))

        # ベンチマーク
        colors = ["#A23B72", "#F18F01"]
        for i, col in enumerate(benchmark_returns.columns[:2]):
            sharpe = calc_rolling_sharpe(benchmark_returns[col])
            fig.add_trace(go.Scatter(
                x=sharpe.index,
                y=sharpe.values,
                name=col,
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            ))

        # 0ライン
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            title=f"Rolling {window}-Day Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=self.config.height,
            template=self.config.template,
            showlegend=self.config.show_legend,
            hovermode="x unified",
        )

        return fig


class PerformanceDashboard:
    """Plotly Dashダッシュボード

    ポートフォリオとベンチマークの比較分析をインタラクティブに表示。

    Example:
        from src.analysis.dashboard import PerformanceDashboard

        dashboard = PerformanceDashboard(
            comparison=comparison_result,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_df,
        )
        dashboard.run(port=8050)
    """

    def __init__(
        self,
        comparison: "ComparisonResult",
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
        chart_generator: ChartGenerator = None,
        title: str = "Portfolio Performance Dashboard",
    ):
        """
        初期化

        Parameters
        ----------
        comparison : ComparisonResult
            PerformanceComparator.compare() の結果
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.DataFrame
            ベンチマークの日次リターン
        chart_generator : ChartGenerator, optional
            カスタムチャートジェネレータ
        title : str
            ダッシュボードタイトル
        """
        if not HAS_DASH:
            raise ImportError(
                "dash is required. Install with: pip install dash dash-bootstrap-components"
            )
        if not HAS_DBC:
            raise ImportError(
                "dash-bootstrap-components is required. Install with: pip install dash-bootstrap-components"
            )

        self.comparison = comparison
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.chart_generator = chart_generator or ChartGenerator()
        self.title = title
        self._app: Optional[dash.Dash] = None

    def create_app(self) -> dash.Dash:
        """Dashアプリ作成"""
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title=self.title,
        )

        app.layout = self._create_layout()
        self._app = app

        return app

    def run(self, port: int = 8050, debug: bool = False, host: str = "127.0.0.1"):
        """ダッシュボード起動"""
        if self._app is None:
            self.create_app()

        logger.info(f"Starting dashboard at http://{host}:{port}")
        self._app.run_server(host=host, port=port, debug=debug)

    def _create_layout(self) -> html.Div:
        """レイアウト作成"""
        return dbc.Container([
            # ヘッダー
            self._create_header(),
            html.Hr(),

            # サマリーカード
            self._create_summary_cards(),
            html.Br(),

            # グラフセクション
            self._create_charts_section(),
            html.Br(),

            # 詳細テーブル
            self._create_detail_tables(),

        ], fluid=True, className="p-4")

    def _create_header(self) -> html.Div:
        """ヘッダー作成"""
        # 期間計算
        start_date = self.portfolio_returns.index.min().strftime("%Y-%m-%d")
        end_date = self.portfolio_returns.index.max().strftime("%Y-%m-%d")

        return dbc.Row([
            dbc.Col([
                html.H1(self.title, className="text-primary"),
                html.P(
                    f"Period: {start_date} to {end_date}",
                    className="text-muted",
                ),
            ])
        ])

    def _create_summary_cards(self) -> html.Div:
        """サマリーカード作成"""
        metrics = self.comparison.portfolio_metrics

        cards = [
            self._metric_card(
                "Total Return",
                f"{metrics.get('total_return', 0) * 100:.1f}%",
                "bi-graph-up-arrow",
                "success" if metrics.get('total_return', 0) > 0 else "danger",
            ),
            self._metric_card(
                "Annualized Return",
                f"{metrics.get('annualized_return', 0) * 100:.1f}%",
                "bi-calendar3",
                "primary",
            ),
            self._metric_card(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                "bi-lightning",
                "info",
            ),
            self._metric_card(
                "Volatility",
                f"{metrics.get('volatility', 0) * 100:.1f}%",
                "bi-activity",
                "warning",
            ),
            self._metric_card(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0) * 100:.1f}%",
                "bi-graph-down-arrow",
                "danger",
            ),
        ]

        return dbc.Row([dbc.Col(card, md=2) for card in cards], className="g-3")

    def _metric_card(
        self,
        title: str,
        value: str,
        icon: str,
        color: str,
    ) -> dbc.Card:
        """メトリクスカード作成"""
        return dbc.Card([
            dbc.CardBody([
                html.I(className=f"bi {icon} fs-4 text-{color}"),
                html.H5(value, className="mt-2 mb-0"),
                html.Small(title, className="text-muted"),
            ])
        ], className="text-center h-100")

    def _create_charts_section(self) -> html.Div:
        """グラフセクション作成"""
        # 共通インデックスに揃える
        common_idx = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )
        port = self.portfolio_returns.loc[common_idx]
        bench = self.benchmark_returns.loc[common_idx]

        return html.Div([
            # Row 1: Cumulative Returns & Drawdown
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=self.chart_generator.create_cumulative_returns_chart(
                            port, bench
                        ),
                        config={"displayModeBar": False},
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=self.chart_generator.create_drawdown_chart(
                            port, bench
                        ),
                        config={"displayModeBar": False},
                    )
                ], md=6),
            ], className="mb-4"),

            # Row 2: Monthly Heatmap & Rolling Sharpe
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=self.chart_generator.create_monthly_heatmap(
                            port, "Portfolio Monthly Returns"
                        ),
                        config={"displayModeBar": False},
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=self.chart_generator.create_rolling_sharpe_chart(
                            port, bench
                        ),
                        config={"displayModeBar": False},
                    )
                ], md=6),
            ]),
        ])

    def _create_detail_tables(self) -> html.Div:
        """詳細テーブル作成"""
        # 絶対パフォーマンステーブル
        abs_df = self.comparison.summary()
        abs_table = self._dataframe_to_table(
            abs_df, "Absolute Performance Metrics"
        )

        # 相対パフォーマンステーブル
        rel_df = self.comparison.relative_summary()
        rel_table = self._dataframe_to_table(
            rel_df, "Relative Performance (vs Benchmarks)"
        )

        return dbc.Row([
            dbc.Col([abs_table], md=6),
            dbc.Col([rel_table], md=6),
        ])

    def _dataframe_to_table(
        self,
        df: pd.DataFrame,
        title: str,
    ) -> dbc.Card:
        """DataFrameをテーブルカードに変換"""
        # 数値フォーマット
        formatted = df.copy()
        for col in formatted.columns:
            if formatted[col].dtype in ["float64", "float32"]:
                # リターン系は%表示、その他は小数2桁
                if any(x in col.lower() for x in ["return", "drawdown", "volatility", "error", "capture"]):
                    formatted[col] = formatted[col].apply(
                        lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A"
                    )
                else:
                    formatted[col] = formatted[col].apply(
                        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                    )

        table = dbc.Table.from_dataframe(
            formatted.reset_index(),
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
        )

        return dbc.Card([
            dbc.CardHeader(html.H5(title, className="mb-0")),
            dbc.CardBody([table]),
        ])


def create_dashboard(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
    title: str = "Portfolio Performance Dashboard",
) -> PerformanceDashboard:
    """ダッシュボードを簡易作成するヘルパー関数

    Parameters
    ----------
    portfolio_returns : pd.Series
        ポートフォリオの日次リターン
    benchmark_returns : pd.DataFrame
        ベンチマークの日次リターン
    risk_free_rate : float
        リスクフリーレート（年率）
    title : str
        ダッシュボードタイトル

    Returns
    -------
    PerformanceDashboard
        設定済みダッシュボードインスタンス
    """
    from src.analysis.performance_comparator import PerformanceComparator

    comparator = PerformanceComparator(risk_free_rate=risk_free_rate)
    comparison = comparator.compare(portfolio_returns, benchmark_returns)

    return PerformanceDashboard(
        comparison=comparison,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        title=title,
    )
