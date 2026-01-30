"""
Long-Term Backtest Validation Module - 15年バックテスト検証

task_012_8: パフォーマンスチューニング計画の最終検証

期間: 2010-01-01 ~ 2025-01-01
比較対象:
- SPY Buy-and-Hold
- 60/40 Portfolio (SPY/TLT)
- 改善前システム（ベースライン）

サブ期間分析:
- 2010-2012: GFC回復期
- 2013-2017: 低ボラ上昇相場
- 2018-2019: ボラスパイク
- 2020-2021: COVID
- 2022-2025: 金利上昇

成功基準:
- Sharpe >= 1.0
- MDD < 20%
- SPY超過: 70%以上の期間
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class SubPeriodResult:
    """サブ期間分析結果"""

    name: str
    start_date: str
    end_date: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    spy_excess: float  # SPYに対する超過リターン（年率）


@dataclass
class BenchmarkComparison:
    """ベンチマーク比較結果"""

    benchmark_name: str
    benchmark_return: float
    benchmark_sharpe: float
    benchmark_mdd: float
    strategy_return: float
    strategy_sharpe: float
    strategy_mdd: float
    excess_return: float
    information_ratio: float
    periods_outperforming: float  # SPYを超過した期間の割合


@dataclass
class ValidationResult:
    """15年バックテスト検証結果"""

    # 全体メトリクス
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float

    # サブ期間分析
    sub_period_results: list[SubPeriodResult]

    # ベンチマーク比較
    benchmark_comparisons: list[BenchmarkComparison]

    # 成功基準チェック
    criteria_met: dict[str, bool]

    # 詳細データ
    portfolio_values: pd.Series | None = None
    monthly_returns: pd.Series | None = None
    rolling_sharpe: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_all_criteria(self) -> bool:
        """全ての成功基準を満たすか"""
        return all(self.criteria_met.values())

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "summary": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "calmar_ratio": self.calmar_ratio,
                "sortino_ratio": self.sortino_ratio,
            },
            "criteria_met": self.criteria_met,
            "passes_all_criteria": self.passes_all_criteria,
            "sub_periods": [
                {
                    "name": sp.name,
                    "start_date": sp.start_date,
                    "end_date": sp.end_date,
                    "total_return": sp.total_return,
                    "annualized_return": sp.annualized_return,
                    "sharpe_ratio": sp.sharpe_ratio,
                    "max_drawdown": sp.max_drawdown,
                    "spy_excess": sp.spy_excess,
                }
                for sp in self.sub_period_results
            ],
            "benchmark_comparisons": [
                {
                    "benchmark": bc.benchmark_name,
                    "benchmark_return": bc.benchmark_return,
                    "strategy_return": bc.strategy_return,
                    "excess_return": bc.excess_return,
                    "information_ratio": bc.information_ratio,
                    "periods_outperforming": bc.periods_outperforming,
                }
                for bc in self.benchmark_comparisons
            ],
            "metadata": self.metadata,
        }


# =============================================================================
# Sub-Period Definitions
# =============================================================================
SUB_PERIODS = {
    "gfc_recovery": {
        "name": "GFC回復期",
        "start": datetime(2010, 1, 1),
        "end": datetime(2012, 12, 31),
        "description": "金融危機からの回復局面",
    },
    "low_vol_bull": {
        "name": "低ボラ上昇相場",
        "start": datetime(2013, 1, 1),
        "end": datetime(2017, 12, 31),
        "description": "量的緩和下の安定上昇",
    },
    "vol_spike": {
        "name": "ボラスパイク期",
        "start": datetime(2018, 1, 1),
        "end": datetime(2019, 12, 31),
        "description": "2018年の急落と回復、金利正常化",
    },
    "covid": {
        "name": "COVID期",
        "start": datetime(2020, 1, 1),
        "end": datetime(2021, 12, 31),
        "description": "パンデミックショックと急回復",
    },
    "rate_hike": {
        "name": "金利上昇期",
        "start": datetime(2022, 1, 1),
        "end": datetime(2025, 1, 1),
        "description": "インフレと金融引き締め",
    },
}


# =============================================================================
# Long-Term Validator
# =============================================================================
class LongTermValidator:
    """15年バックテスト検証クラス

    成功基準:
    - Sharpe >= 1.0
    - MDD < 20%
    - SPY超過: 70%以上の期間（月次ベース）

    使用例:
        validator = LongTermValidator()
        result = validator.validate(
            strategy_returns=returns_series,
            spy_returns=spy_series,
            tlt_returns=tlt_series,
        )
        print(result.passes_all_criteria)
    """

    # 成功基準
    SHARPE_TARGET = 1.0
    MDD_TARGET = 0.20
    SPY_OUTPERFORMANCE_TARGET = 0.70

    def __init__(
        self,
        start_date: datetime = datetime(2010, 1, 1),
        end_date: datetime = datetime(2025, 1, 1),
        risk_free_rate: float = 0.02,
    ):
        """初期化

        Args:
            start_date: 検証開始日
            end_date: 検証終了日
            risk_free_rate: 無リスク金利（年率）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate

    def validate(
        self,
        strategy_returns: pd.Series,
        spy_returns: pd.Series | None = None,
        tlt_returns: pd.Series | None = None,
        baseline_returns: pd.Series | None = None,
    ) -> ValidationResult:
        """15年バックテスト検証を実行

        Args:
            strategy_returns: 戦略の日次リターン系列
            spy_returns: SPYの日次リターン（ベンチマーク）
            tlt_returns: TLTの日次リターン（60/40計算用）
            baseline_returns: 改善前システムのリターン

        Returns:
            ValidationResult
        """
        logger.info(
            f"Starting long-term validation: {self.start_date} to {self.end_date}"
        )

        # 期間でフィルタリング
        strategy_returns = self._filter_period(strategy_returns)

        if len(strategy_returns) < 252:  # 最低1年
            logger.error("Insufficient data for validation")
            return self._create_empty_result("Insufficient data")

        # 1. 全体メトリクス計算
        total_return = self._calc_total_return(strategy_returns)
        ann_return = self._calc_annualized_return(strategy_returns)
        volatility = self._calc_annualized_volatility(strategy_returns)
        sharpe = self._calc_sharpe_ratio(strategy_returns)
        mdd = self._calc_max_drawdown(strategy_returns)
        calmar = ann_return / abs(mdd) if mdd < 0 else 0.0
        sortino = self._calc_sortino_ratio(strategy_returns)

        # 2. サブ期間分析
        sub_period_results = self._analyze_sub_periods(strategy_returns, spy_returns)

        # 3. ベンチマーク比較
        benchmark_comparisons = self._compare_benchmarks(
            strategy_returns, spy_returns, tlt_returns, baseline_returns
        )

        # 4. 成功基準チェック
        criteria_met = self._check_criteria(
            sharpe=sharpe,
            mdd=mdd,
            benchmark_comparisons=benchmark_comparisons,
        )

        # 5. 追加メトリクス
        portfolio_values = (1 + strategy_returns).cumprod()
        monthly_returns = strategy_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling_sharpe = self._calc_rolling_sharpe(strategy_returns, window=252)

        result = ValidationResult(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            sub_period_results=sub_period_results,
            benchmark_comparisons=benchmark_comparisons,
            criteria_met=criteria_met,
            portfolio_values=portfolio_values,
            monthly_returns=monthly_returns,
            rolling_sharpe=rolling_sharpe,
            metadata={
                "start_date": str(self.start_date),
                "end_date": str(self.end_date),
                "n_days": len(strategy_returns),
                "risk_free_rate": self.risk_free_rate,
            },
        )

        logger.info(
            f"Validation completed: Sharpe={sharpe:.2f}, MDD={mdd:.1%}, "
            f"Criteria met: {result.passes_all_criteria}"
        )

        return result

    def _filter_period(self, returns: pd.Series) -> pd.Series:
        """期間でフィルタリング"""
        mask = (returns.index >= self.start_date) & (returns.index < self.end_date)
        return returns.loc[mask]

    def _calc_total_return(self, returns: pd.Series) -> float:
        """累積リターンを計算"""
        return float((1 + returns).prod() - 1)

    def _calc_annualized_return(self, returns: pd.Series) -> float:
        """年率リターンを計算"""
        total = self._calc_total_return(returns)
        years = len(returns) / 252
        if years <= 0 or total <= -1:
            return -1.0
        return float((1 + total) ** (1 / years) - 1)

    def _calc_annualized_volatility(self, returns: pd.Series) -> float:
        """年率ボラティリティを計算"""
        return float(returns.std() * np.sqrt(252))

    def _calc_sharpe_ratio(self, returns: pd.Series) -> float:
        """シャープレシオを計算"""
        ann_return = self._calc_annualized_return(returns)
        volatility = self._calc_annualized_volatility(returns)
        if volatility == 0:
            return 0.0
        return float((ann_return - self.risk_free_rate) / volatility)

    def _calc_sortino_ratio(self, returns: pd.Series) -> float:
        """ソルティノレシオを計算"""
        ann_return = self._calc_annualized_return(returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float("inf")
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return 0.0
        return float((ann_return - self.risk_free_rate) / downside_std)

    def _calc_max_drawdown(self, returns: pd.Series) -> float:
        """最大ドローダウンを計算"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return float(drawdown.min())

    def _calc_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """ローリングシャープレシオを計算"""
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        return (rolling_mean - self.risk_free_rate) / rolling_std

    def _analyze_sub_periods(
        self,
        strategy_returns: pd.Series,
        spy_returns: pd.Series | None,
    ) -> list[SubPeriodResult]:
        """サブ期間分析を実行"""
        results = []

        for period_key, period_info in SUB_PERIODS.items():
            start = period_info["start"]
            end = period_info["end"]

            # 期間のリターンを抽出
            mask = (strategy_returns.index >= start) & (strategy_returns.index < end)
            period_returns = strategy_returns.loc[mask]

            if len(period_returns) < 20:
                continue

            # メトリクス計算
            total_ret = self._calc_total_return(period_returns)
            ann_ret = self._calc_annualized_return(period_returns)
            vol = self._calc_annualized_volatility(period_returns)
            sharpe = self._calc_sharpe_ratio(period_returns)
            mdd = self._calc_max_drawdown(period_returns)
            win_rate = (period_returns > 0).mean()

            # SPY超過リターン
            spy_excess = 0.0
            if spy_returns is not None:
                spy_mask = (spy_returns.index >= start) & (spy_returns.index < end)
                spy_period = spy_returns.loc[spy_mask]
                if len(spy_period) > 0:
                    spy_ann_ret = self._calc_annualized_return(spy_period)
                    spy_excess = ann_ret - spy_ann_ret

            results.append(
                SubPeriodResult(
                    name=period_info["name"],
                    start_date=str(start.date()),
                    end_date=str(end.date()),
                    total_return=total_ret,
                    annualized_return=ann_ret,
                    volatility=vol,
                    sharpe_ratio=sharpe,
                    max_drawdown=mdd,
                    win_rate=win_rate,
                    spy_excess=spy_excess,
                )
            )

        return results

    def _compare_benchmarks(
        self,
        strategy_returns: pd.Series,
        spy_returns: pd.Series | None,
        tlt_returns: pd.Series | None,
        baseline_returns: pd.Series | None,
    ) -> list[BenchmarkComparison]:
        """ベンチマーク比較を実行"""
        comparisons = []

        # 戦略のメトリクス
        strategy_ret = self._calc_annualized_return(strategy_returns)
        strategy_sharpe = self._calc_sharpe_ratio(strategy_returns)
        strategy_mdd = self._calc_max_drawdown(strategy_returns)

        # SPY比較
        if spy_returns is not None:
            spy_returns = self._filter_period(spy_returns)
            spy_returns = spy_returns.reindex(strategy_returns.index).dropna()

            if len(spy_returns) > 0:
                spy_ret = self._calc_annualized_return(spy_returns)
                spy_sharpe = self._calc_sharpe_ratio(spy_returns)
                spy_mdd = self._calc_max_drawdown(spy_returns)

                # 超過リターンの計算
                excess_ret = strategy_ret - spy_ret

                # Information Ratio
                tracking_error = (strategy_returns - spy_returns).std() * np.sqrt(252)
                ir = excess_ret / tracking_error if tracking_error > 0 else 0.0

                # 月次ベースでSPYを上回った期間
                strategy_monthly = strategy_returns.resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )
                spy_monthly = spy_returns.resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )
                common_idx = strategy_monthly.index.intersection(spy_monthly.index)
                if len(common_idx) > 0:
                    outperform_pct = (
                        strategy_monthly.loc[common_idx] > spy_monthly.loc[common_idx]
                    ).mean()
                else:
                    outperform_pct = 0.0

                comparisons.append(
                    BenchmarkComparison(
                        benchmark_name="SPY Buy-and-Hold",
                        benchmark_return=spy_ret,
                        benchmark_sharpe=spy_sharpe,
                        benchmark_mdd=spy_mdd,
                        strategy_return=strategy_ret,
                        strategy_sharpe=strategy_sharpe,
                        strategy_mdd=strategy_mdd,
                        excess_return=excess_ret,
                        information_ratio=ir,
                        periods_outperforming=outperform_pct,
                    )
                )

        # 60/40ポートフォリオ比較
        if spy_returns is not None and tlt_returns is not None:
            tlt_returns = self._filter_period(tlt_returns)
            spy_filtered = self._filter_period(spy_returns)

            # 共通インデックス
            common_idx = spy_filtered.index.intersection(tlt_returns.index)
            if len(common_idx) > 0:
                # 60/40ポートフォリオ
                portfolio_60_40 = (
                    0.6 * spy_filtered.loc[common_idx] + 0.4 * tlt_returns.loc[common_idx]
                )

                bench_ret = self._calc_annualized_return(portfolio_60_40)
                bench_sharpe = self._calc_sharpe_ratio(portfolio_60_40)
                bench_mdd = self._calc_max_drawdown(portfolio_60_40)

                excess_ret = strategy_ret - bench_ret
                tracking_error = (
                    strategy_returns.reindex(common_idx) - portfolio_60_40
                ).std() * np.sqrt(252)
                ir = excess_ret / tracking_error if tracking_error > 0 else 0.0

                # 月次超過
                strategy_monthly = strategy_returns.resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )
                bench_monthly = portfolio_60_40.resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )
                common_monthly = strategy_monthly.index.intersection(bench_monthly.index)
                if len(common_monthly) > 0:
                    outperform_pct = (
                        strategy_monthly.loc[common_monthly]
                        > bench_monthly.loc[common_monthly]
                    ).mean()
                else:
                    outperform_pct = 0.0

                comparisons.append(
                    BenchmarkComparison(
                        benchmark_name="60/40 Portfolio",
                        benchmark_return=bench_ret,
                        benchmark_sharpe=bench_sharpe,
                        benchmark_mdd=bench_mdd,
                        strategy_return=strategy_ret,
                        strategy_sharpe=strategy_sharpe,
                        strategy_mdd=strategy_mdd,
                        excess_return=excess_ret,
                        information_ratio=ir,
                        periods_outperforming=outperform_pct,
                    )
                )

        # 改善前システム比較
        if baseline_returns is not None:
            baseline_returns = self._filter_period(baseline_returns)
            baseline_returns = baseline_returns.reindex(strategy_returns.index).dropna()

            if len(baseline_returns) > 0:
                base_ret = self._calc_annualized_return(baseline_returns)
                base_sharpe = self._calc_sharpe_ratio(baseline_returns)
                base_mdd = self._calc_max_drawdown(baseline_returns)

                excess_ret = strategy_ret - base_ret
                tracking_error = (strategy_returns - baseline_returns).std() * np.sqrt(
                    252
                )
                ir = excess_ret / tracking_error if tracking_error > 0 else 0.0

                comparisons.append(
                    BenchmarkComparison(
                        benchmark_name="Baseline System",
                        benchmark_return=base_ret,
                        benchmark_sharpe=base_sharpe,
                        benchmark_mdd=base_mdd,
                        strategy_return=strategy_ret,
                        strategy_sharpe=strategy_sharpe,
                        strategy_mdd=strategy_mdd,
                        excess_return=excess_ret,
                        information_ratio=ir,
                        periods_outperforming=0.0,  # 計算省略
                    )
                )

        return comparisons

    def _check_criteria(
        self,
        sharpe: float,
        mdd: float,
        benchmark_comparisons: list[BenchmarkComparison],
    ) -> dict[str, bool]:
        """成功基準をチェック"""
        criteria = {
            "sharpe_ge_1.0": sharpe >= self.SHARPE_TARGET,
            "mdd_lt_20%": abs(mdd) < self.MDD_TARGET,
        }

        # SPY超過70%
        spy_comparison = next(
            (bc for bc in benchmark_comparisons if bc.benchmark_name == "SPY Buy-and-Hold"),
            None,
        )
        if spy_comparison:
            criteria["spy_outperform_70%"] = (
                spy_comparison.periods_outperforming >= self.SPY_OUTPERFORMANCE_TARGET
            )
        else:
            criteria["spy_outperform_70%"] = False

        return criteria

    def _create_empty_result(self, error_message: str) -> ValidationResult:
        """エラー時の空結果を作成"""
        return ValidationResult(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            sub_period_results=[],
            benchmark_comparisons=[],
            criteria_met={
                "sharpe_ge_1.0": False,
                "mdd_lt_20%": False,
                "spy_outperform_70%": False,
            },
            metadata={"error": error_message},
        )


# =============================================================================
# Convenience Functions
# =============================================================================
def run_long_term_validation(
    strategy_returns: pd.Series,
    spy_returns: pd.Series | None = None,
    tlt_returns: pd.Series | None = None,
    output_path: Path | None = None,
) -> ValidationResult:
    """15年バックテスト検証を実行する便利関数

    Args:
        strategy_returns: 戦略の日次リターン
        spy_returns: SPYのリターン（オプション）
        tlt_returns: TLTのリターン（オプション）
        output_path: 結果の保存先（オプション）

    Returns:
        ValidationResult
    """
    import json

    validator = LongTermValidator()
    result = validator.validate(
        strategy_returns=strategy_returns,
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Validation result saved to {output_path}")

    return result


def generate_validation_report(result: ValidationResult) -> str:
    """検証結果のレポートを生成

    Args:
        result: ValidationResult

    Returns:
        レポート文字列
    """
    lines = [
        "=" * 60,
        "15-Year Backtest Validation Report",
        "=" * 60,
        "",
        "## Overall Metrics",
        f"  Total Return:      {result.total_return:.1%}",
        f"  Annualized Return: {result.annualized_return:.1%}",
        f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}",
        f"  Max Drawdown:      {result.max_drawdown:.1%}",
        f"  Calmar Ratio:      {result.calmar_ratio:.2f}",
        f"  Sortino Ratio:     {result.sortino_ratio:.2f}",
        "",
        "## Success Criteria",
    ]

    for criterion, met in result.criteria_met.items():
        status = "PASS" if met else "FAIL"
        lines.append(f"  [{status}] {criterion}")

    lines.append("")
    lines.append(f"  Overall: {'PASS' if result.passes_all_criteria else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Sub-Period Analysis",
        ]
    )

    for sp in result.sub_period_results:
        lines.extend(
            [
                f"  {sp.name} ({sp.start_date} to {sp.end_date})",
                f"    Return: {sp.annualized_return:.1%}, Sharpe: {sp.sharpe_ratio:.2f}, "
                f"MDD: {sp.max_drawdown:.1%}, SPY Excess: {sp.spy_excess:+.1%}",
            ]
        )

    lines.extend(
        [
            "",
            "## Benchmark Comparison",
        ]
    )

    for bc in result.benchmark_comparisons:
        lines.extend(
            [
                f"  vs {bc.benchmark_name}",
                f"    Strategy: {bc.strategy_return:.1%}, Benchmark: {bc.benchmark_return:.1%}",
                f"    Excess Return: {bc.excess_return:+.1%}, IR: {bc.information_ratio:.2f}",
                f"    Periods Outperforming: {bc.periods_outperforming:.0%}",
            ]
        )

    lines.extend(
        [
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
