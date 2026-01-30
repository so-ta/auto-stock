"""
Static Chart Generator - 静的グラフ生成（matplotlib）

PDF/PNG向けの高品質な静的グラフを生成。
matplotlibを使用したプロフェッショナルなチャート出力。

Usage:
    from src.analysis.static_charts import StaticChartGenerator
    import pandas as pd

    # グラフ生成器を作成
    generator = StaticChartGenerator(figsize=(12, 6))

    # 累積リターン比較グラフ
    fig = generator.plot_equity_comparison(
        portfolio=portfolio_returns,
        benchmarks=benchmark_returns,
    )
    fig.savefig("equity_comparison.png", dpi=150)

    # 全グラフを一括保存
    files = generator.save_all_charts(
        portfolio=portfolio_returns,
        benchmarks=benchmark_returns,
        output_dir="charts/",
        format="png",
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib import with backend setting
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# Try to import japanize-matplotlib for Japanese font support
try:
    import japanize_matplotlib
    HAS_JAPANIZE = True
except ImportError:
    HAS_JAPANIZE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class StaticChartGenerator:
    """
    静的グラフ生成（matplotlib）- PDF/PNG向け

    高品質な静的グラフを生成し、レポートやプレゼンテーション用に
    PNG/PDF/SVG形式で出力可能。

    Attributes:
        figsize: グラフサイズ (width, height)
        style: matplotlibスタイル
        dpi: 出力解像度
    """

    # デフォルトカラーパレット
    COLORS = {
        "portfolio": "#4a69bd",    # Blue
        "benchmark1": "#e74c3c",   # Red
        "benchmark2": "#27ae60",   # Green
        "benchmark3": "#f39c12",   # Orange
        "benchmark4": "#9b59b6",   # Purple
        "positive": "#27ae60",
        "negative": "#e74c3c",
        "neutral": "#95a5a6",
    }

    BENCHMARK_COLORS = ["#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 150,
    ):
        """
        初期化

        Args:
            figsize: グラフサイズ (width, height)
            style: matplotlibスタイル
            dpi: 出力解像度
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

        # スタイル設定
        self._setup_style()

    def _setup_style(self) -> None:
        """matplotlibスタイルを設定"""
        try:
            plt.style.use(self.style)
        except OSError:
            # スタイルが見つからない場合はデフォルト使用
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                pass  # Use default style

        # 日本語フォント設定
        if HAS_JAPANIZE:
            pass  # japanize-matplotlib handles this
        else:
            # Fallback: try to use system fonts
            plt.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans", "Arial"]

        # 共通設定
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["grid.color"] = "#e0e0e0"
        plt.rcParams["grid.linestyle"] = "-"
        plt.rcParams["grid.linewidth"] = 0.5

    def plot_equity_comparison(
        self,
        portfolio: pd.Series,
        benchmarks: pd.DataFrame,
        title: str = "資産推移比較",
        normalize: bool = True,
    ) -> Figure:
        """
        累積リターン比較グラフ

        Args:
            portfolio: ポートフォリオの日次リターン（Seriesまたは累積値）
            benchmarks: ベンチマークの日次リターン（DataFrame）
            title: グラフタイトル
            normalize: 100を基準に正規化するか

        Returns:
            Figure: matplotlibのFigureオブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 累積リターンに変換
        if normalize:
            portfolio_cum = self._to_cumulative(portfolio) * 100
            benchmarks_cum = benchmarks.apply(self._to_cumulative) * 100
        else:
            portfolio_cum = portfolio
            benchmarks_cum = benchmarks

        # ポートフォリオをプロット
        ax.plot(
            portfolio_cum.index,
            portfolio_cum.values,
            label="Portfolio",
            color=self.COLORS["portfolio"],
            linewidth=2.5,
            zorder=10,
        )

        # ベンチマークをプロット
        for i, col in enumerate(benchmarks_cum.columns):
            color = self.BENCHMARK_COLORS[i % len(self.BENCHMARK_COLORS)]
            ax.plot(
                benchmarks_cum.index,
                benchmarks_cum[col].values,
                label=col,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

        # 装飾
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)" if normalize else "Value", fontsize=11)

        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        # 凡例
        ax.legend(loc="upper left", framealpha=0.9)

        # グリッド
        ax.grid(True, alpha=0.3)

        # 100ライン（正規化時）
        if normalize:
            ax.axhline(y=100, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_drawdown_comparison(
        self,
        portfolio_dd: pd.Series,
        benchmark_dds: pd.DataFrame,
        title: str = "ドローダウン比較",
    ) -> Figure:
        """
        ドローダウン比較グラフ

        Args:
            portfolio_dd: ポートフォリオのドローダウン（負の値）
            benchmark_dds: ベンチマークのドローダウン（DataFrame）
            title: グラフタイトル

        Returns:
            Figure: matplotlibのFigureオブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # ポートフォリオドローダウンを塗りつぶし
        ax.fill_between(
            portfolio_dd.index,
            0,
            portfolio_dd.values * 100,
            label="Portfolio",
            color=self.COLORS["portfolio"],
            alpha=0.4,
        )
        ax.plot(
            portfolio_dd.index,
            portfolio_dd.values * 100,
            color=self.COLORS["portfolio"],
            linewidth=1.5,
        )

        # ベンチマークドローダウン
        for i, col in enumerate(benchmark_dds.columns):
            color = self.BENCHMARK_COLORS[i % len(self.BENCHMARK_COLORS)]
            ax.plot(
                benchmark_dds.index,
                benchmark_dds[col].values * 100,
                label=col,
                color=color,
                linewidth=1.2,
                alpha=0.8,
            )

        # 装飾
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Drawdown (%)", fontsize=11)

        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        # Y軸を0以下に
        ax.set_ylim(top=0)

        # 凡例
        ax.legend(loc="lower left", framealpha=0.9)

        # グリッド
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "月次リターン",
    ) -> Figure:
        """
        月次リターンヒートマップ

        Args:
            returns: 日次リターン
            title: グラフタイトル

        Returns:
            Figure: matplotlibのFigureオブジェクト
        """
        # 月次リターンに集約
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        # 年×月のピボットテーブル作成
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        # 月名
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))

        # カスタムカラーマップ（赤-白-緑）
        cmap = LinearSegmentedColormap.from_list(
            "rg",
            [(0.9, 0.3, 0.3), (1, 1, 1), (0.3, 0.7, 0.3)],
            N=256
        )

        # ヒートマップデータ
        data = pivot.values * 100  # パーセント表示

        # カラースケールの範囲（対称）
        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        vmin = -vmax

        # ヒートマップ描画
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # 軸ラベル
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index)

        # セルにテキスト追加
        for i in range(len(pivot)):
            for j in range(12):
                val = data[i, j] if not np.isnan(data[i, j]) else None
                if val is not None:
                    text_color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color=text_color, fontsize=9)

        # カラーバー
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Return (%)", fontsize=10)

        # タイトル
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout()
        return fig

    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 252,
        risk_free_rate: float = 0.02,
        title: str = "ローリングシャープレシオ",
    ) -> Figure:
        """
        ローリングシャープレシオ

        Args:
            returns: 日次リターン
            window: ローリングウィンドウ（日数）
            risk_free_rate: 無リスク金利（年率）
            title: グラフタイトル

        Returns:
            Figure: matplotlibのFigureオブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # ローリングシャープ計算
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        rolling_mean = excess_returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        # プロット
        ax.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            color=self.COLORS["portfolio"],
            linewidth=1.5,
        )

        # 塗りつぶし（正負で色分け）
        ax.fill_between(
            rolling_sharpe.index,
            0,
            rolling_sharpe.values,
            where=rolling_sharpe.values >= 0,
            color=self.COLORS["positive"],
            alpha=0.3,
            interpolate=True,
        )
        ax.fill_between(
            rolling_sharpe.index,
            0,
            rolling_sharpe.values,
            where=rolling_sharpe.values < 0,
            color=self.COLORS["negative"],
            alpha=0.3,
            interpolate=True,
        )

        # 0ライン
        ax.axhline(y=0, color="#999999", linestyle="-", linewidth=0.8)

        # 装飾
        ax.set_title(f"{title} ({window}日)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Sharpe Ratio", fontsize=11)

        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        # グリッド
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "リターン分布",
        bins: int = 50,
    ) -> Figure:
        """
        リターン分布ヒストグラム

        Args:
            returns: 日次リターン
            title: グラフタイトル
            bins: ビン数

        Returns:
            Figure: matplotlibのFigureオブジェクト
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # ヒストグラム
        n, bins_edges, patches = ax.hist(
            returns.dropna() * 100,
            bins=bins,
            color=self.COLORS["portfolio"],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        # 正負で色分け
        for i, patch in enumerate(patches):
            if bins_edges[i] >= 0:
                patch.set_facecolor(self.COLORS["positive"])
            else:
                patch.set_facecolor(self.COLORS["negative"])

        # 平均線
        mean_val = returns.mean() * 100
        ax.axvline(x=mean_val, color="#333333", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_val:.2f}%")

        # 0ライン
        ax.axvline(x=0, color="#999999", linestyle="-", linewidth=1)

        # 装飾
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Daily Return (%)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)

        # 凡例
        ax.legend(loc="upper right")

        # グリッド
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def save_all_charts(
        self,
        portfolio: pd.Series,
        benchmarks: pd.DataFrame,
        output_dir: str,
        format: str = "png",
        prefix: str = "",
    ) -> List[str]:
        """
        全グラフを保存

        Args:
            portfolio: ポートフォリオの日次リターン
            benchmarks: ベンチマークの日次リターン
            output_dir: 出力ディレクトリ
            format: 出力形式 (png, pdf, svg)
            prefix: ファイル名プレフィックス

        Returns:
            List[str]: 保存されたファイルパスのリスト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 1. 累積リターン比較
        fig = self.plot_equity_comparison(portfolio, benchmarks)
        filepath = output_path / f"{prefix}equity_comparison.{format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

        # 2. ドローダウン比較
        portfolio_dd = self._calculate_drawdown(portfolio)
        benchmark_dds = benchmarks.apply(self._calculate_drawdown)
        fig = self.plot_drawdown_comparison(portfolio_dd, benchmark_dds)
        filepath = output_path / f"{prefix}drawdown_comparison.{format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

        # 3. 月次リターンヒートマップ
        fig = self.plot_monthly_heatmap(portfolio)
        filepath = output_path / f"{prefix}monthly_heatmap.{format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

        # 4. ローリングシャープ
        fig = self.plot_rolling_sharpe(portfolio)
        filepath = output_path / f"{prefix}rolling_sharpe.{format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

        # 5. リターン分布
        fig = self.plot_returns_distribution(portfolio)
        filepath = output_path / f"{prefix}returns_distribution.{format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

        return saved_files

    def _to_cumulative(self, returns: pd.Series) -> pd.Series:
        """日次リターンを累積リターンに変換"""
        # すでに累積値かチェック（最初の値が1に近いか）
        if len(returns) > 0 and abs(returns.iloc[0]) > 0.5:
            # すでに累積値の場合は正規化
            return returns / returns.iloc[0]
        # 日次リターンから累積リターンを計算
        return (1 + returns).cumprod()

    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """ドローダウンを計算"""
        cumulative = self._to_cumulative(returns)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
