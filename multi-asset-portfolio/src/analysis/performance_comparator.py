"""
Performance Comparator - ポートフォリオとベンチマークの比較分析

ポートフォリオのパフォーマンスを複数のベンチマークと比較し、
相対的なパフォーマンス指標を計算する。

主な機能:
- トラッキングエラー（Tracking Error）
- インフォメーションレシオ（Information Ratio）
- ベータ（Beta）
- アルファ（Alpha / Jensen's Alpha）
- 相対リターン分析

Usage:
    from src.analysis.performance_comparator import PerformanceComparator

    comparator = PerformanceComparator()

    result = comparator.compare(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_df,  # columns = benchmark names
    )

    print(f"Tracking Error vs SPY: {result.relative_metrics['SPY']['tracking_error']:.2%}")
    print(f"Information Ratio: {result.relative_metrics['SPY']['information_ratio']:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 年間取引日数
TRADING_DAYS_PER_YEAR = 252


@dataclass
class ComparisonResult:
    """比較結果

    Attributes:
        portfolio_metrics: ポートフォリオの絶対パフォーマンス指標
        benchmark_metrics: 各ベンチマークの絶対パフォーマンス指標
        relative_metrics: ポートフォリオの相対パフォーマンス指標（vs各ベンチマーク）
    """

    portfolio_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, Dict[str, float]]
    relative_metrics: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "portfolio": self.portfolio_metrics,
            "benchmarks": self.benchmark_metrics,
            "relative": self.relative_metrics,
        }

    def summary(self) -> pd.DataFrame:
        """サマリーをDataFrameで返す"""
        rows = []

        # ポートフォリオ行
        row = {"name": "Portfolio", **self.portfolio_metrics}
        rows.append(row)

        # ベンチマーク行
        for name, metrics in self.benchmark_metrics.items():
            row = {"name": name, **metrics}
            rows.append(row)

        return pd.DataFrame(rows).set_index("name")

    def relative_summary(self) -> pd.DataFrame:
        """相対パフォーマンスのサマリーをDataFrameで返す"""
        rows = []
        for benchmark_name, metrics in self.relative_metrics.items():
            row = {"benchmark": benchmark_name, **metrics}
            rows.append(row)
        return pd.DataFrame(rows).set_index("benchmark")


class PerformanceComparator:
    """ポートフォリオとベンチマークの比較分析

    ポートフォリオのパフォーマンスを複数のベンチマークと比較し、
    相対的なパフォーマンス指標（トラッキングエラー、IR、ベータ、アルファ等）を計算。

    Example:
        comparator = PerformanceComparator()

        # 単一ベンチマーク比較
        te = comparator.calculate_tracking_error(portfolio, benchmark)
        ir = comparator.calculate_information_ratio(portfolio, benchmark)

        # 複数ベンチマーク一括比較
        result = comparator.compare(portfolio, benchmark_df)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        annualization_factor: int = TRADING_DAYS_PER_YEAR,
    ):
        """
        初期化

        Parameters
        ----------
        risk_free_rate : float
            リスクフリーレート（年率）。デフォルトは2%。
        annualization_factor : int
            年率化係数（取引日数）。デフォルトは252。
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def compare(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
    ) -> ComparisonResult:
        """
        ポートフォリオと複数ベンチマークを比較

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.DataFrame
            ベンチマークの日次リターン。カラム名がベンチマーク名。

        Returns
        -------
        ComparisonResult
            比較結果
        """
        # インデックスを揃える
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            raise ValueError("Insufficient overlapping data points")

        portfolio = portfolio_returns.loc[common_index]
        benchmarks = benchmark_returns.loc[common_index]

        # ポートフォリオの絶対パフォーマンス
        portfolio_metrics = self._calculate_absolute_metrics(portfolio)

        # 各ベンチマークの絶対パフォーマンスと相対パフォーマンス
        benchmark_metrics: Dict[str, Dict[str, float]] = {}
        relative_metrics: Dict[str, Dict[str, float]] = {}

        for benchmark_name in benchmarks.columns:
            benchmark = benchmarks[benchmark_name].dropna()

            # 共通インデックスで再度揃える
            common = portfolio.index.intersection(benchmark.index)
            if len(common) < 2:
                logger.warning(f"Skipping {benchmark_name}: insufficient data")
                continue

            port_aligned = portfolio.loc[common]
            bench_aligned = benchmark.loc[common]

            # ベンチマークの絶対パフォーマンス
            benchmark_metrics[benchmark_name] = self._calculate_absolute_metrics(
                bench_aligned
            )

            # 相対パフォーマンス
            relative_metrics[benchmark_name] = self._calculate_relative_metrics(
                port_aligned, bench_aligned
            )

        return ComparisonResult(
            portfolio_metrics=portfolio_metrics,
            benchmark_metrics=benchmark_metrics,
            relative_metrics=relative_metrics,
        )

    def calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        トラッキングエラーを計算（年率化）

        Tracking Error = std(portfolio - benchmark) * sqrt(252)

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン

        Returns
        -------
        float
            年率化トラッキングエラー
        """
        # 共通インデックス
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        excess_returns = (
            portfolio_returns.loc[common] - benchmark_returns.loc[common]
        )

        daily_te = excess_returns.std()
        annualized_te = daily_te * np.sqrt(self.annualization_factor)

        return annualized_te

    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        インフォメーションレシオを計算

        Information Ratio = (mean(portfolio - benchmark) * 252) / Tracking Error

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン

        Returns
        -------
        float
            インフォメーションレシオ
        """
        # 共通インデックス
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        excess_returns = (
            portfolio_returns.loc[common] - benchmark_returns.loc[common]
        )

        # 年率化超過リターン
        annualized_excess = excess_returns.mean() * self.annualization_factor

        # トラッキングエラー
        tracking_error = self.calculate_tracking_error(
            portfolio_returns, benchmark_returns
        )

        if tracking_error == 0 or np.isnan(tracking_error):
            return np.nan

        return annualized_excess / tracking_error

    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        ベータ値を計算

        Beta = cov(portfolio, benchmark) / var(benchmark)

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン

        Returns
        -------
        float
            ベータ値
        """
        # 共通インデックス
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        port = portfolio_returns.loc[common]
        bench = benchmark_returns.loc[common]

        covariance = port.cov(bench)
        variance = bench.var()

        if variance == 0 or np.isnan(variance):
            return np.nan

        return covariance / variance

    def calculate_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """
        アルファを計算（CAPM / Jensen's Alpha）

        Alpha = portfolio_return - (risk_free + beta * (benchmark_return - risk_free))

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン
        risk_free_rate : float, optional
            リスクフリーレート（年率）。省略時はインスタンスのデフォルト値。

        Returns
        -------
        float
            年率化アルファ
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # 共通インデックス
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        port = portfolio_returns.loc[common]
        bench = benchmark_returns.loc[common]

        # 年率化リターン
        portfolio_annual = port.mean() * self.annualization_factor
        benchmark_annual = bench.mean() * self.annualization_factor

        # ベータ
        beta = self.calculate_beta(port, bench)
        if np.isnan(beta):
            return np.nan

        # CAPM期待リターン
        expected_return = risk_free_rate + beta * (benchmark_annual - risk_free_rate)

        # アルファ = 実際のリターン - 期待リターン
        alpha = portfolio_annual - expected_return

        return alpha

    def calculate_up_capture(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        アップキャプチャー比率を計算

        ベンチマークが上昇した期間のポートフォリオリターン / ベンチマークリターン

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン

        Returns
        -------
        float
            アップキャプチャー比率
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        port = portfolio_returns.loc[common]
        bench = benchmark_returns.loc[common]

        up_periods = bench > 0
        if up_periods.sum() == 0:
            return np.nan

        port_up = port[up_periods].mean()
        bench_up = bench[up_periods].mean()

        if bench_up == 0:
            return np.nan

        return port_up / bench_up

    def calculate_down_capture(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        ダウンキャプチャー比率を計算

        ベンチマークが下落した期間のポートフォリオリターン / ベンチマークリターン

        Parameters
        ----------
        portfolio_returns : pd.Series
            ポートフォリオの日次リターン
        benchmark_returns : pd.Series
            ベンチマークの日次リターン

        Returns
        -------
        float
            ダウンキャプチャー比率
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            return np.nan

        port = portfolio_returns.loc[common]
        bench = benchmark_returns.loc[common]

        down_periods = bench < 0
        if down_periods.sum() == 0:
            return np.nan

        port_down = port[down_periods].mean()
        bench_down = bench[down_periods].mean()

        if bench_down == 0:
            return np.nan

        return port_down / bench_down

    def _calculate_absolute_metrics(
        self,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """絶対パフォーマンス指標を計算"""
        if len(returns) < 2:
            return {
                "total_return": np.nan,
                "annualized_return": np.nan,
                "volatility": np.nan,
                "sharpe_ratio": np.nan,
                "max_drawdown": np.nan,
            }

        # 累積リターン
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # 年率化リターン
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (
            self.annualization_factor / n_days
        ) - 1

        # ボラティリティ（年率化）
        volatility = returns.std() * np.sqrt(self.annualization_factor)

        # シャープレシオ
        daily_rf = self.risk_free_rate / self.annualization_factor
        excess_returns = returns - daily_rf
        if volatility > 0:
            sharpe_ratio = (
                excess_returns.mean() * self.annualization_factor / volatility
            )
        else:
            sharpe_ratio = np.nan

        # 最大ドローダウン
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    def _calculate_relative_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """相対パフォーマンス指標を計算"""
        return {
            "tracking_error": self.calculate_tracking_error(
                portfolio_returns, benchmark_returns
            ),
            "information_ratio": self.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            ),
            "beta": self.calculate_beta(portfolio_returns, benchmark_returns),
            "alpha": self.calculate_alpha(portfolio_returns, benchmark_returns),
            "up_capture": self.calculate_up_capture(
                portfolio_returns, benchmark_returns
            ),
            "down_capture": self.calculate_down_capture(
                portfolio_returns, benchmark_returns
            ),
            "excess_return": (
                portfolio_returns.mean() - benchmark_returns.mean()
            )
            * self.annualization_factor,
        }
