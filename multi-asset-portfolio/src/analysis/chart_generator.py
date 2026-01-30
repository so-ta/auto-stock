"""
Chart Generator - グラフ生成（Plotly）

ポートフォリオ分析用のインタラクティブグラフを生成する。

Usage:
    from src.analysis.chart_generator import ChartGenerator

    generator = ChartGenerator()
    fig = generator.plot_equity_comparison(portfolio, benchmarks)
    generator.save_chart(fig, "output/equity_comparison.html")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

logger = logging.getLogger(__name__)


class ChartGeneratorError(Exception):
    """ChartGenerator関連のエラー"""
    pass


# プロフェッショナルな配色パレット
COLOR_PALETTE = {
    "portfolio": "#1f77b4",  # 青
    "spy": "#ff7f0e",        # オレンジ
    "qqq": "#2ca02c",        # 緑
    "dia": "#d62728",        # 赤
    "iwm": "#9467bd",        # 紫
    "vt": "#8c564b",         # 茶
    "ewj": "#e377c2",        # ピンク
    "default": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ],
    "positive": "#2ca02c",   # 緑（プラス）
    "negative": "#d62728",   # 赤（マイナス）
    "neutral": "#7f7f7f",    # グレー
}

# 日本語フォント設定
FONT_FAMILY = "Hiragino Sans, Yu Gothic, Meiryo, sans-serif"
FONT_CONFIG = {
    "family": FONT_FAMILY,
    "size": 12,
}
TITLE_FONT_CONFIG = {
    "family": FONT_FAMILY,
    "size": 16,
}


class ChartGenerator:
    """
    グラフ生成クラス（Plotly）

    ポートフォリオ分析用のインタラクティブグラフを生成。
    HTML、PNG、PDF形式での出力に対応。

    Attributes:
        color_palette: カラーパレット
        default_template: デフォルトテンプレート
    """

    def __init__(
        self,
        template: str = "plotly_white",
        color_palette: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        初期化

        Parameters
        ----------
        template : str
            Plotlyテンプレート（plotly_white, plotly_dark, ggplot2等）
        color_palette : dict, optional
            カスタムカラーパレット
        """
        if not HAS_PLOTLY:
            raise ChartGeneratorError(
                "plotly is required. Install with: pip install plotly"
            )

        self._template = template
        self._colors = color_palette or COLOR_PALETTE

    def plot_equity_comparison(
        self,
        portfolio: pd.Series,
        benchmarks: pd.DataFrame,
        title: str = "資産推移比較",
        initial_value: float = 100.0,
        show_legend: bool = True,
    ) -> go.Figure:
        """
        累積リターン比較グラフ

        Parameters
        ----------
        portfolio : pd.Series
            ポートフォリオの累積リターン（またはリターン）
        benchmarks : pd.DataFrame
            ベンチマークの累積リターン（列: ティッカー）
        title : str
            グラフタイトル
        initial_value : float
            初期値（デフォルト100）
        show_legend : bool
            凡例を表示するか

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        fig = go.Figure()

        # ポートフォリオをプロット
        if portfolio is not None and len(portfolio) > 0:
            # リターンから累積値に変換（必要な場合）
            if portfolio.iloc[0] < 2:  # リターンデータの場合
                portfolio_values = initial_value * (1 + portfolio).cumprod()
            else:
                portfolio_values = portfolio

            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode="lines",
                name="Portfolio",
                line=dict(color=self._colors["portfolio"], width=2.5),
                hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.2f}<extra></extra>",
            ))

        # ベンチマークをプロット
        if benchmarks is not None and not benchmarks.empty:
            colors = self._colors.get("default", COLOR_PALETTE["default"])
            for i, col in enumerate(benchmarks.columns):
                # リターンから累積値に変換（必要な場合）
                if benchmarks[col].iloc[0] < 2:
                    values = initial_value * (1 + benchmarks[col]).cumprod()
                else:
                    values = benchmarks[col]

                color = self._colors.get(col.lower(), colors[i % len(colors)])
                fig.add_trace(go.Scatter(
                    x=values.index,
                    y=values.values,
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col}: %{{y:.2f}}<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            xaxis_title="日付",
            yaxis_title="資産価値",
            template=self._template,
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="x unified",
            font=FONT_CONFIG,
        )

        return fig

    def plot_drawdown_comparison(
        self,
        portfolio_dd: pd.Series,
        benchmark_dds: pd.DataFrame,
        title: str = "ドローダウン比較",
    ) -> go.Figure:
        """
        ドローダウン比較グラフ

        Parameters
        ----------
        portfolio_dd : pd.Series
            ポートフォリオのドローダウン（負の値）
        benchmark_dds : pd.DataFrame
            ベンチマークのドローダウン（列: ティッカー）
        title : str
            グラフタイトル

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        fig = go.Figure()

        # ポートフォリオのドローダウン
        if portfolio_dd is not None and len(portfolio_dd) > 0:
            fig.add_trace(go.Scatter(
                x=portfolio_dd.index,
                y=portfolio_dd.values * 100,  # パーセント表示
                mode="lines",
                name="Portfolio",
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.3)",
                line=dict(color=self._colors["portfolio"], width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.2f}%<extra></extra>",
            ))

        # ベンチマークのドローダウン
        if benchmark_dds is not None and not benchmark_dds.empty:
            colors = self._colors.get("default", COLOR_PALETTE["default"])
            for i, col in enumerate(benchmark_dds.columns):
                color = self._colors.get(col.lower(), colors[i % len(colors)])
                fig.add_trace(go.Scatter(
                    x=benchmark_dds[col].index,
                    y=benchmark_dds[col].values * 100,
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col}: %{{y:.2f}}%<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            xaxis_title="日付",
            yaxis_title="ドローダウン (%)",
            template=self._template,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="x unified",
            font=FONT_CONFIG,
        )

        # Y軸を反転（ドローダウンは下方向）
        fig.update_yaxes(autorange="reversed")

        return fig

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "月次リターン",
        colorscale: str = "RdYlGn",
    ) -> go.Figure:
        """
        月次リターンヒートマップ（年×月）

        Parameters
        ----------
        returns : pd.Series
            日次または月次リターン
        title : str
            グラフタイトル
        colorscale : str
            カラースケール（RdYlGn, Viridis, Blues等）

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        if returns is None or len(returns) == 0:
            raise ChartGeneratorError("Empty returns data")

        # 月次リターンに変換
        if hasattr(returns.index, 'freq') and returns.index.freq == 'D':
            monthly = returns.resample('ME').sum()
        else:
            # すでに月次または日次データを月次に集約
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # 年と月のピボットテーブル作成
        df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100,  # パーセント
        })
        pivot = df.pivot(index='year', columns='month', values='return')

        # 月名ラベル
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_labels[:pivot.shape[1]],
            y=pivot.index.astype(str),
            colorscale=colorscale,
            zmid=0,  # 0を中心に
            text=np.round(pivot.values, 1),
            texttemplate="%{text:.1f}%",
            textfont=dict(size=10),
            hovertemplate="年: %{y}<br>月: %{x}<br>リターン: %{z:.2f}%<extra></extra>",
            colorbar=dict(
                title=dict(text="リターン (%)", side="right"),
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            xaxis_title="月",
            yaxis_title="年",
            template=self._template,
            font=FONT_CONFIG,
        )

        return fig

    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
        risk_free_rate: float = 0.02,
        title: str = "ローリングシャープレシオ",
    ) -> go.Figure:
        """
        ローリングシャープレシオ

        Parameters
        ----------
        returns : pd.Series
            日次リターン
        window : int
            ローリングウィンドウ（日数）
        risk_free_rate : float
            年率リスクフリーレート
        title : str
            グラフタイトル

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        if returns is None or len(returns) < window:
            raise ChartGeneratorError(f"Insufficient data: need at least {window} points")

        # ローリングシャープ計算
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        rolling_mean = excess_returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        fig = go.Figure()

        # シャープレシオのライン
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name=f"シャープレシオ ({window}日)",
            line=dict(color=self._colors["portfolio"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
        ))

        # ゼロライン
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=self._colors["neutral"],
            annotation_text="0",
            annotation_position="right",
        )

        # 1.0ライン（良好な基準）
        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color=self._colors["positive"],
            annotation_text="1.0",
            annotation_position="right",
        )

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            xaxis_title="日付",
            yaxis_title="シャープレシオ",
            template=self._template,
            hovermode="x unified",
            font=FONT_CONFIG,
        )

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "リターン分布",
        bins: int = 50,
    ) -> go.Figure:
        """
        リターン分布ヒストグラム

        Parameters
        ----------
        returns : pd.Series
            リターンデータ
        title : str
            グラフタイトル
        bins : int
            ビン数

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        if returns is None or len(returns) == 0:
            raise ChartGeneratorError("Empty returns data")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns.values * 100,
            nbinsx=bins,
            name="リターン",
            marker_color=self._colors["portfolio"],
            opacity=0.7,
            hovertemplate="リターン: %{x:.2f}%<br>頻度: %{y}<extra></extra>",
        ))

        # 平均線
        mean_return = returns.mean() * 100
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color=self._colors["negative"],
            annotation_text=f"平均: {mean_return:.2f}%",
            annotation_position="top",
        )

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            xaxis_title="リターン (%)",
            yaxis_title="頻度",
            template=self._template,
            bargap=0.1,
            font=FONT_CONFIG,
        )

        return fig

    def plot_correlation_matrix(
        self,
        returns: pd.DataFrame,
        title: str = "相関行列",
    ) -> go.Figure:
        """
        相関行列ヒートマップ

        Parameters
        ----------
        returns : pd.DataFrame
            リターンデータ（列: 資産）
        title : str
            グラフタイトル

        Returns
        -------
        go.Figure
            Plotly Figure オブジェクト
        """
        if returns is None or returns.empty:
            raise ChartGeneratorError("Empty returns data")

        corr_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            hovertemplate="%{x} vs %{y}<br>相関: %{z:.3f}<extra></extra>",
            colorbar=dict(title="相関"),
        ))

        fig.update_layout(
            title=dict(text=title, font=TITLE_FONT_CONFIG),
            template=self._template,
            font=FONT_CONFIG,
        )

        return fig

    def save_chart(
        self,
        fig: go.Figure,
        output_path: Union[str, Path],
        format: str = "html",
        width: int = 1200,
        height: int = 600,
        scale: float = 2.0,
    ) -> str:
        """
        グラフ保存

        Parameters
        ----------
        fig : go.Figure
            Plotly Figure オブジェクト
        output_path : str or Path
            出力パス
        format : str
            フォーマット: "html", "png", "pdf", "svg"
        width : int
            画像幅（png/pdf/svg用）
        height : int
            画像高さ（png/pdf/svg用）
        scale : float
            画像スケール（png用）

        Returns
        -------
        str
            保存したファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 拡張子を追加（なければ）
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{format}")

        if format == "html":
            fig.write_html(str(output_path), include_plotlyjs="cdn")
        elif format in ["png", "pdf", "svg"]:
            try:
                fig.write_image(
                    str(output_path),
                    width=width,
                    height=height,
                    scale=scale if format == "png" else 1.0,
                )
            except Exception as e:
                if "kaleido" in str(e).lower():
                    raise ChartGeneratorError(
                        f"kaleido is required for {format} export. "
                        "Install with: pip install kaleido"
                    )
                raise
        else:
            raise ChartGeneratorError(f"Unsupported format: {format}")

        logger.info(f"Chart saved: {output_path}")
        return str(output_path)

    @staticmethod
    def calculate_drawdown(values: pd.Series) -> pd.Series:
        """
        ドローダウンを計算

        Parameters
        ----------
        values : pd.Series
            累積リターンまたは価値

        Returns
        -------
        pd.Series
            ドローダウン（負の値）
        """
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        return drawdown
