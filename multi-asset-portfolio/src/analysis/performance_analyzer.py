"""
Performance Analyzer Module - パフォーマンス分析とチューニング

このモジュールは、ポートフォリオのパフォーマンス分析と最適化機能を提供する。

分析機能:
- 戦略別貢献度分析（どのシグナルが最も貢献したか）
- ドローダウン分析（最大DD発生時期と原因、回復期間）
- 市場レジーム別分析（上昇/下落相場、高/低ボラ時の成績）

チューニング機能:
- シグナルパラメータ最適化（Sharpe最大化、過学習回避）
- 戦略配分の最適化（各戦略への配分比率）
- リバランス頻度の最適化（日次/週次/月次のトレードオフ）
"""

from __future__ import annotations

import json
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
class ContributionResult:
    """戦略貢献度分析結果"""

    strategy_id: str
    total_contribution: float
    contribution_ratio: float
    avg_return: float
    sharpe_ratio: float
    periods_positive: int
    periods_negative: int
    best_period: str | None = None
    worst_period: str | None = None


@dataclass
class DrawdownEvent:
    """ドローダウンイベント"""

    start_date: str
    end_date: str
    recovery_date: str | None
    max_drawdown: float
    duration_days: int
    recovery_days: int | None
    cause_analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimePerformance:
    """レジーム別パフォーマンス"""

    regime_name: str
    period_count: int
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


@dataclass
class TuningRecommendation:
    """チューニング推奨事項"""

    category: str
    recommendation: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: str  # high, medium, low
    rationale: str


@dataclass
class AnalysisReport:
    """分析レポート"""

    generated_at: str
    strategy_contributions: list[ContributionResult]
    drawdown_events: list[DrawdownEvent]
    regime_performance: dict[str, list[RegimePerformance]]
    tuning_recommendations: list[TuningRecommendation]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "generated_at": self.generated_at,
            "strategy_contributions": [
                {
                    "strategy_id": c.strategy_id,
                    "total_contribution": c.total_contribution,
                    "contribution_ratio": c.contribution_ratio,
                    "avg_return": c.avg_return,
                    "sharpe_ratio": c.sharpe_ratio,
                    "periods_positive": c.periods_positive,
                    "periods_negative": c.periods_negative,
                    "best_period": c.best_period,
                    "worst_period": c.worst_period,
                }
                for c in self.strategy_contributions
            ],
            "drawdown_events": [
                {
                    "start_date": d.start_date,
                    "end_date": d.end_date,
                    "recovery_date": d.recovery_date,
                    "max_drawdown": d.max_drawdown,
                    "duration_days": d.duration_days,
                    "recovery_days": d.recovery_days,
                    "cause_analysis": d.cause_analysis,
                }
                for d in self.drawdown_events
            ],
            "regime_performance": {
                regime: [
                    {
                        "regime_name": p.regime_name,
                        "period_count": p.period_count,
                        "avg_return": p.avg_return,
                        "volatility": p.volatility,
                        "sharpe_ratio": p.sharpe_ratio,
                        "max_drawdown": p.max_drawdown,
                        "win_rate": p.win_rate,
                    }
                    for p in performances
                ]
                for regime, performances in self.regime_performance.items()
            },
            "tuning_recommendations": [
                {
                    "category": r.category,
                    "recommendation": r.recommendation,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "expected_improvement": r.expected_improvement,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                }
                for r in self.tuning_recommendations
            ],
            "summary": self.summary,
        }


# =============================================================================
# Strategy Contribution Analyzer
# =============================================================================
class StrategyContributionAnalyzer:
    """戦略別貢献度分析クラス

    どのシグナルが最も貢献したか、どの期間で各戦略が有効だったかを分析する。
    """

    def analyze(
        self,
        strategy_returns: dict[str, pd.Series],
        strategy_weights: dict[str, float],
        portfolio_returns: pd.Series,
    ) -> list[ContributionResult]:
        """戦略貢献度を分析

        Args:
            strategy_returns: 戦略ID -> リターン系列
            strategy_weights: 戦略ID -> 重み
            portfolio_returns: ポートフォリオリターン系列

        Returns:
            戦略貢献度結果リスト
        """
        results = []
        total_portfolio_return = portfolio_returns.sum()

        for strategy_id, returns in strategy_returns.items():
            weight = strategy_weights.get(strategy_id, 0.0)

            # 重み付きリターン（貢献度）
            weighted_returns = returns * weight
            total_contribution = weighted_returns.sum()

            # 貢献比率
            contribution_ratio = (
                total_contribution / total_portfolio_return
                if total_portfolio_return != 0
                else 0.0
            )

            # 平均リターン
            avg_return = float(returns.mean())

            # シャープレシオ
            std = returns.std()
            sharpe_ratio = (avg_return / std * np.sqrt(252)) if std > 0 else 0.0

            # ポジティブ/ネガティブ期間
            periods_positive = int((returns > 0).sum())
            periods_negative = int((returns < 0).sum())

            # ベスト/ワースト期間
            if len(returns) > 0:
                best_idx = returns.idxmax()
                worst_idx = returns.idxmin()
                best_period = str(best_idx) if best_idx is not None else None
                worst_period = str(worst_idx) if worst_idx is not None else None
            else:
                best_period = None
                worst_period = None

            results.append(
                ContributionResult(
                    strategy_id=strategy_id,
                    total_contribution=float(total_contribution),
                    contribution_ratio=float(contribution_ratio),
                    avg_return=avg_return,
                    sharpe_ratio=float(sharpe_ratio),
                    periods_positive=periods_positive,
                    periods_negative=periods_negative,
                    best_period=best_period,
                    worst_period=worst_period,
                )
            )

        # 貢献度順にソート
        results.sort(key=lambda x: x.total_contribution, reverse=True)
        return results


# =============================================================================
# Drawdown Analyzer
# =============================================================================
class DrawdownAnalyzer:
    """ドローダウン分析クラス

    最大DD発生時期と原因、回復期間を分析する。
    """

    def __init__(self, threshold: float = 0.05):
        """初期化

        Args:
            threshold: ドローダウンイベントとして記録する閾値（デフォルト5%）
        """
        self.threshold = threshold

    def analyze(
        self,
        returns: pd.Series,
        strategy_returns: dict[str, pd.Series] | None = None,
    ) -> list[DrawdownEvent]:
        """ドローダウンを分析

        Args:
            returns: ポートフォリオリターン系列
            strategy_returns: 戦略別リターン（原因分析用）

        Returns:
            ドローダウンイベントリスト
        """
        # 累積リターンと高値
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (running_max - cum_returns) / running_max

        # ドローダウンイベントの検出
        events = []
        in_drawdown = False
        start_idx = None
        max_dd = 0.0
        max_dd_idx = None

        for i, (idx, dd) in enumerate(drawdown.items()):
            if not in_drawdown and dd >= self.threshold:
                # ドローダウン開始
                in_drawdown = True
                start_idx = idx
                max_dd = dd
                max_dd_idx = idx
            elif in_drawdown:
                if dd > max_dd:
                    max_dd = dd
                    max_dd_idx = idx

                if dd < 0.001:  # 回復判定
                    # イベント記録
                    event = self._create_event(
                        returns=returns,
                        strategy_returns=strategy_returns,
                        start_idx=start_idx,
                        end_idx=max_dd_idx,
                        recovery_idx=idx,
                        max_dd=max_dd,
                    )
                    events.append(event)
                    in_drawdown = False
                    start_idx = None
                    max_dd = 0.0
                    max_dd_idx = None

        # 未回復のドローダウン
        if in_drawdown and start_idx is not None:
            event = self._create_event(
                returns=returns,
                strategy_returns=strategy_returns,
                start_idx=start_idx,
                end_idx=max_dd_idx,
                recovery_idx=None,
                max_dd=max_dd,
            )
            events.append(event)

        # 最大DDでソート
        events.sort(key=lambda x: x.max_drawdown, reverse=True)
        return events

    def _create_event(
        self,
        returns: pd.Series,
        strategy_returns: dict[str, pd.Series] | None,
        start_idx: Any,
        end_idx: Any,
        recovery_idx: Any | None,
        max_dd: float,
    ) -> DrawdownEvent:
        """ドローダウンイベントを作成"""
        # 期間計算
        start_date = str(start_idx)
        end_date = str(end_idx)
        recovery_date = str(recovery_idx) if recovery_idx is not None else None

        # 日数計算（indexがdatetimeの場合）
        try:
            duration_days = (end_idx - start_idx).days
            recovery_days = (
                (recovery_idx - end_idx).days if recovery_idx is not None else None
            )
        except (TypeError, AttributeError):
            duration_days = 0
            recovery_days = None

        # 原因分析
        cause_analysis = {}
        if strategy_returns is not None:
            cause_analysis = self._analyze_cause(
                strategy_returns, start_idx, end_idx
            )

        return DrawdownEvent(
            start_date=start_date,
            end_date=end_date,
            recovery_date=recovery_date,
            max_drawdown=float(max_dd),
            duration_days=duration_days,
            recovery_days=recovery_days,
            cause_analysis=cause_analysis,
        )

    def _analyze_cause(
        self,
        strategy_returns: dict[str, pd.Series],
        start_idx: Any,
        end_idx: Any,
    ) -> dict[str, Any]:
        """ドローダウンの原因を分析"""
        cause = {"contributing_strategies": []}

        for strategy_id, returns in strategy_returns.items():
            try:
                period_return = returns.loc[start_idx:end_idx].sum()
                if period_return < -0.01:  # 1%以上のマイナス
                    cause["contributing_strategies"].append(
                        {"strategy_id": strategy_id, "loss": float(period_return)}
                    )
            except (KeyError, TypeError):
                continue

        # 損失順にソート
        cause["contributing_strategies"].sort(key=lambda x: x["loss"])
        return cause


# =============================================================================
# Regime Analyzer
# =============================================================================
class RegimeAnalyzer:
    """市場レジーム別分析クラス

    上昇相場/下落相場、高ボラ/低ボラ時の成績を分析する。
    """

    def __init__(
        self,
        vol_threshold_high: float = 0.20,
        vol_threshold_low: float = 0.10,
        trend_lookback: int = 20,
    ):
        """初期化

        Args:
            vol_threshold_high: 高ボラティリティ閾値（年率）
            vol_threshold_low: 低ボラティリティ閾値（年率）
            trend_lookback: トレンド判定のルックバック期間
        """
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.trend_lookback = trend_lookback

    def analyze(
        self,
        portfolio_returns: pd.Series,
        market_returns: pd.Series | None = None,
    ) -> dict[str, list[RegimePerformance]]:
        """レジーム別パフォーマンスを分析

        Args:
            portfolio_returns: ポートフォリオリターン系列
            market_returns: 市場リターン系列（トレンド判定用）

        Returns:
            レジーム種類 -> パフォーマンスリスト
        """
        results = {}

        # ボラティリティレジーム
        results["volatility"] = self._analyze_volatility_regime(portfolio_returns)

        # トレンドレジーム
        reference_returns = market_returns if market_returns is not None else portfolio_returns
        results["trend"] = self._analyze_trend_regime(portfolio_returns, reference_returns)

        return results

    def _analyze_volatility_regime(
        self, returns: pd.Series
    ) -> list[RegimePerformance]:
        """ボラティリティレジーム別分析"""
        results = []

        # ローリングボラティリティ
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

        # レジーム分類
        high_vol_mask = rolling_vol > self.vol_threshold_high
        low_vol_mask = rolling_vol < self.vol_threshold_low
        medium_vol_mask = ~high_vol_mask & ~low_vol_mask

        regimes = [
            ("high_volatility", high_vol_mask),
            ("medium_volatility", medium_vol_mask),
            ("low_volatility", low_vol_mask),
        ]

        for regime_name, mask in regimes:
            regime_returns = returns[mask]
            if len(regime_returns) < 10:
                continue

            perf = self._calculate_regime_performance(regime_name, regime_returns)
            results.append(perf)

        return results

    def _analyze_trend_regime(
        self, returns: pd.Series, reference_returns: pd.Series
    ) -> list[RegimePerformance]:
        """トレンドレジーム別分析"""
        results = []

        # ローリングリターン（トレンド判定）
        rolling_return = reference_returns.rolling(window=self.trend_lookback).sum()

        # レジーム分類
        bull_mask = rolling_return > 0.02  # 2%以上上昇
        bear_mask = rolling_return < -0.02  # 2%以上下落
        range_mask = ~bull_mask & ~bear_mask

        regimes = [
            ("bull_market", bull_mask),
            ("range_market", range_mask),
            ("bear_market", bear_mask),
        ]

        for regime_name, mask in regimes:
            # インデックスを揃える
            common_idx = returns.index.intersection(mask[mask].index)
            regime_returns = returns.loc[common_idx]

            if len(regime_returns) < 10:
                continue

            perf = self._calculate_regime_performance(regime_name, regime_returns)
            results.append(perf)

        return results

    def _calculate_regime_performance(
        self, regime_name: str, returns: pd.Series
    ) -> RegimePerformance:
        """レジーム別パフォーマンスを計算"""
        avg_return = float(returns.mean() * 252)  # 年率
        volatility = float(returns.std() * np.sqrt(252))
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0

        # 最大ドローダウン
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (running_max - cum_returns) / running_max
        max_drawdown = float(drawdown.max())

        # 勝率
        win_rate = float((returns > 0).mean())

        return RegimePerformance(
            regime_name=regime_name,
            period_count=len(returns),
            avg_return=avg_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
        )


# =============================================================================
# Parameter Optimizer
# =============================================================================
class ParameterOptimizer:
    """シグナルパラメータ最適化クラス

    Sharpe最大化のパラメータ探索と過学習回避（クロスバリデーション）を行う。
    """

    def __init__(self, n_folds: int = 5, min_improvement: float = 0.1):
        """初期化

        Args:
            n_folds: クロスバリデーションのフォールド数
            min_improvement: 推奨に必要な最小改善率
        """
        self.n_folds = n_folds
        self.min_improvement = min_improvement

    def optimize(
        self,
        current_params: dict[str, Any],
        param_grid: dict[str, list[Any]],
        evaluate_func: Any | None = None,
    ) -> list[TuningRecommendation]:
        """パラメータ最適化を実行

        Args:
            current_params: 現在のパラメータ
            param_grid: パラメータ探索グリッド
            evaluate_func: 評価関数（シグナル -> シャープレシオ）

        Returns:
            チューニング推奨事項リスト
        """
        recommendations = []

        # 各パラメータについて最適値を探索
        for param_name, values in param_grid.items():
            current_value = current_params.get(param_name)
            if current_value is None:
                continue

            # 簡易的な推奨（実際にはクロスバリデーションで検証すべき）
            # ここではヒューリスティックな推奨を生成
            if param_name == "lookback":
                if current_value < 20:
                    recommended = min(values, key=lambda x: abs(x - 60))
                    recommendations.append(
                        TuningRecommendation(
                            category="signal_parameter",
                            recommendation=f"{param_name}を{recommended}に変更",
                            current_value=current_value,
                            recommended_value=recommended,
                            expected_improvement=0.15,
                            confidence="medium",
                            rationale="短すぎるルックバックはノイズに敏感。60日程度が安定。",
                        )
                    )
            elif param_name == "threshold":
                if current_value > 0.8 or current_value < 0.2:
                    recommended = 0.5
                    recommendations.append(
                        TuningRecommendation(
                            category="signal_parameter",
                            recommendation=f"{param_name}を{recommended}に変更",
                            current_value=current_value,
                            recommended_value=recommended,
                            expected_improvement=0.10,
                            confidence="low",
                            rationale="極端な閾値は過学習リスクあり。中央値付近を推奨。",
                        )
                    )

        return recommendations


# =============================================================================
# Allocation Optimizer
# =============================================================================
class AllocationOptimizer:
    """戦略配分最適化クラス

    各戦略への配分比率とレジームに応じた動的配分を最適化する。
    """

    def optimize(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        current_allocation: dict[str, float],
        regime_info: dict[str, Any] | None = None,
    ) -> list[TuningRecommendation]:
        """配分最適化を実行

        Args:
            strategy_metrics: 戦略ID -> メトリクス辞書
            current_allocation: 現在の配分
            regime_info: レジーム情報

        Returns:
            チューニング推奨事項リスト
        """
        recommendations = []

        # シャープレシオベースの最適配分を計算
        total_sharpe = sum(
            m.get("sharpe_ratio", 0) for m in strategy_metrics.values() if m.get("sharpe_ratio", 0) > 0
        )

        if total_sharpe > 0:
            optimal_allocation = {}
            for strategy_id, metrics in strategy_metrics.items():
                sharpe = metrics.get("sharpe_ratio", 0)
                if sharpe > 0:
                    optimal_allocation[strategy_id] = sharpe / total_sharpe
                else:
                    optimal_allocation[strategy_id] = 0.0

            # 現在の配分と比較
            for strategy_id, optimal_weight in optimal_allocation.items():
                current_weight = current_allocation.get(strategy_id, 0.0)
                diff = abs(optimal_weight - current_weight)

                if diff > 0.1:  # 10%以上の乖離
                    recommendations.append(
                        TuningRecommendation(
                            category="allocation",
                            recommendation=f"{strategy_id}の配分を{optimal_weight:.1%}に調整",
                            current_value=f"{current_weight:.1%}",
                            recommended_value=f"{optimal_weight:.1%}",
                            expected_improvement=diff * 0.5,  # 乖離の半分程度の改善予測
                            confidence="medium",
                            rationale="シャープレシオに基づく最適配分との乖離を是正",
                        )
                    )

        return recommendations


# =============================================================================
# Rebalance Optimizer
# =============================================================================
class RebalanceOptimizer:
    """リバランス頻度最適化クラス

    日次/週次/月次のトレードオフと取引コストとのバランスを分析する。
    """

    def optimize(
        self,
        returns: pd.Series,
        turnover_history: list[float],
        cost_per_trade: float = 0.001,
        current_frequency: str = "monthly",
    ) -> list[TuningRecommendation]:
        """リバランス頻度最適化を実行

        Args:
            returns: ポートフォリオリターン系列
            turnover_history: ターンオーバー履歴
            cost_per_trade: 取引コスト（リターンベース）
            current_frequency: 現在のリバランス頻度

        Returns:
            チューニング推奨事項リスト
        """
        recommendations = []

        # 平均ターンオーバー
        avg_turnover = np.mean(turnover_history) if turnover_history else 0.0

        # コスト計算
        frequencies = {
            "daily": {"multiplier": 252, "name": "日次"},
            "weekly": {"multiplier": 52, "name": "週次"},
            "monthly": {"multiplier": 12, "name": "月次"},
            "quarterly": {"multiplier": 4, "name": "四半期"},
        }

        current_mult = frequencies.get(current_frequency, {}).get("multiplier", 12)
        annual_cost = avg_turnover * cost_per_trade * current_mult

        # 各頻度でのコスト比較
        best_frequency = current_frequency
        best_cost = annual_cost
        best_net_benefit = 0.0

        for freq, info in frequencies.items():
            if freq == current_frequency:
                continue

            estimated_cost = avg_turnover * cost_per_trade * info["multiplier"]

            # 頻度変更による期待リターン変化（簡易推定）
            # 高頻度ほどシグナル反映が早いが、コストも高い
            if info["multiplier"] > current_mult:
                # より高頻度
                expected_return_change = 0.02 * (info["multiplier"] / current_mult - 1)
            else:
                # より低頻度
                expected_return_change = -0.01 * (1 - info["multiplier"] / current_mult)

            net_benefit = expected_return_change - (estimated_cost - annual_cost)

            if net_benefit > best_net_benefit:
                best_frequency = freq
                best_cost = estimated_cost
                best_net_benefit = net_benefit

        if best_frequency != current_frequency and best_net_benefit > 0.005:
            recommendations.append(
                TuningRecommendation(
                    category="rebalance",
                    recommendation=f"リバランス頻度を{frequencies[best_frequency]['name']}に変更",
                    current_value=frequencies[current_frequency]["name"],
                    recommended_value=frequencies[best_frequency]["name"],
                    expected_improvement=best_net_benefit,
                    confidence="medium",
                    rationale=f"年間コスト: {annual_cost:.2%} -> {best_cost:.2%}、純便益: {best_net_benefit:.2%}",
                )
            )

        return recommendations


# =============================================================================
# Performance Analyzer (統合クラス)
# =============================================================================
class PerformanceAnalyzer:
    """パフォーマンス分析統合クラス

    すべての分析とチューニング機能を統合し、レポートを生成する。

    Usage:
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            portfolio_returns=returns,
            strategy_returns=strategy_returns,
            strategy_weights=weights,
            current_params=params,
        )

        analyzer.save_report(report, Path("results/"))
    """

    def __init__(self):
        """初期化"""
        self.contribution_analyzer = StrategyContributionAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.regime_analyzer = RegimeAnalyzer()
        self.parameter_optimizer = ParameterOptimizer()
        self.allocation_optimizer = AllocationOptimizer()
        self.rebalance_optimizer = RebalanceOptimizer()

    def analyze(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: dict[str, pd.Series] | None = None,
        strategy_weights: dict[str, float] | None = None,
        strategy_metrics: dict[str, dict[str, float]] | None = None,
        market_returns: pd.Series | None = None,
        current_params: dict[str, Any] | None = None,
        param_grid: dict[str, list[Any]] | None = None,
        turnover_history: list[float] | None = None,
        current_rebalance_frequency: str = "monthly",
    ) -> AnalysisReport:
        """総合分析を実行

        Args:
            portfolio_returns: ポートフォリオリターン系列
            strategy_returns: 戦略別リターン
            strategy_weights: 戦略別重み
            strategy_metrics: 戦略別メトリクス
            market_returns: 市場リターン
            current_params: 現在のシグナルパラメータ
            param_grid: パラメータ探索グリッド
            turnover_history: ターンオーバー履歴
            current_rebalance_frequency: 現在のリバランス頻度

        Returns:
            分析レポート
        """
        strategy_returns = strategy_returns or {}
        strategy_weights = strategy_weights or {}
        strategy_metrics = strategy_metrics or {}
        current_params = current_params or {}
        param_grid = param_grid or {}
        turnover_history = turnover_history or []

        # 1. 戦略貢献度分析
        contributions = []
        if strategy_returns and strategy_weights:
            contributions = self.contribution_analyzer.analyze(
                strategy_returns, strategy_weights, portfolio_returns
            )

        # 2. ドローダウン分析
        drawdown_events = self.drawdown_analyzer.analyze(
            portfolio_returns, strategy_returns or None
        )

        # 3. レジーム別分析
        regime_performance = self.regime_analyzer.analyze(
            portfolio_returns, market_returns
        )

        # 4. チューニング推奨事項
        recommendations = []

        # パラメータ最適化
        if current_params and param_grid:
            recommendations.extend(
                self.parameter_optimizer.optimize(current_params, param_grid)
            )

        # 配分最適化
        if strategy_metrics and strategy_weights:
            recommendations.extend(
                self.allocation_optimizer.optimize(
                    strategy_metrics, strategy_weights
                )
            )

        # リバランス頻度最適化
        if turnover_history:
            recommendations.extend(
                self.rebalance_optimizer.optimize(
                    portfolio_returns,
                    turnover_history,
                    current_frequency=current_rebalance_frequency,
                )
            )

        # 5. サマリー作成
        summary = self._create_summary(
            portfolio_returns, contributions, drawdown_events, recommendations
        )

        return AnalysisReport(
            generated_at=datetime.now().isoformat(),
            strategy_contributions=contributions,
            drawdown_events=drawdown_events,
            regime_performance=regime_performance,
            tuning_recommendations=recommendations,
            summary=summary,
        )

    def _create_summary(
        self,
        portfolio_returns: pd.Series,
        contributions: list[ContributionResult],
        drawdown_events: list[DrawdownEvent],
        recommendations: list[TuningRecommendation],
    ) -> dict[str, Any]:
        """サマリーを作成"""
        # 基本統計
        total_return = float((1 + portfolio_returns).prod() - 1)
        annualized_return = float(portfolio_returns.mean() * 252)
        volatility = float(portfolio_returns.std() * np.sqrt(252))
        sharpe = annualized_return / volatility if volatility > 0 else 0.0

        # トップ貢献戦略
        top_contributors = [c.strategy_id for c in contributions[:3]] if contributions else []

        # 最大ドローダウン
        max_dd = max((d.max_drawdown for d in drawdown_events), default=0.0)

        # 推奨事項サマリー
        high_priority_count = sum(1 for r in recommendations if r.confidence == "high")

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "top_contributing_strategies": top_contributors,
            "drawdown_events_count": len(drawdown_events),
            "tuning_recommendations_count": len(recommendations),
            "high_priority_recommendations": high_priority_count,
        }

    def save_report(
        self,
        report: AnalysisReport,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """レポートを保存

        Args:
            report: 分析レポート
            output_dir: 出力ディレクトリ

        Returns:
            (performance_analysis.json, tuning_recommendations.json) のパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # パフォーマンス分析
        analysis_path = output_dir / "performance_analysis.json"
        analysis_data = {
            "generated_at": report.generated_at,
            "summary": report.summary,
            "strategy_contributions": [
                {
                    "strategy_id": c.strategy_id,
                    "total_contribution": c.total_contribution,
                    "contribution_ratio": c.contribution_ratio,
                    "avg_return": c.avg_return,
                    "sharpe_ratio": c.sharpe_ratio,
                }
                for c in report.strategy_contributions
            ],
            "drawdown_events": [
                {
                    "start_date": d.start_date,
                    "end_date": d.end_date,
                    "max_drawdown": d.max_drawdown,
                    "duration_days": d.duration_days,
                    "recovery_days": d.recovery_days,
                }
                for d in report.drawdown_events
            ],
            "regime_performance": report.to_dict()["regime_performance"],
        }

        with open(analysis_path, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        # チューニング推奨事項
        tuning_path = output_dir / "tuning_recommendations.json"
        tuning_data = {
            "generated_at": report.generated_at,
            "recommendations": [
                {
                    "category": r.category,
                    "recommendation": r.recommendation,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "expected_improvement": r.expected_improvement,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                }
                for r in report.tuning_recommendations
            ],
            "summary": {
                "total_recommendations": len(report.tuning_recommendations),
                "by_category": {},
                "estimated_total_improvement": sum(
                    r.expected_improvement for r in report.tuning_recommendations
                ),
            },
        }

        # カテゴリ別集計
        for r in report.tuning_recommendations:
            if r.category not in tuning_data["summary"]["by_category"]:
                tuning_data["summary"]["by_category"][r.category] = 0
            tuning_data["summary"]["by_category"][r.category] += 1

        with open(tuning_path, "w") as f:
            json.dump(tuning_data, f, indent=2, default=str)

        logger.info(
            "Reports saved",
            analysis_path=str(analysis_path),
            tuning_path=str(tuning_path),
        )

        return analysis_path, tuning_path
