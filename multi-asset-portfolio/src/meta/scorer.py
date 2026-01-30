"""
Strategy Scorer Module - 戦略スコアリング

戦略の総合スコアを計算する。スコアはSharpe調整値からペナルティを減算して算出。

計算式:
    score = Sharpe_adj - penalty
    penalty = λ1 * turnover + λ2 * MDD + λ3 * instability

設計根拠:
- 要求.md §7.2: 戦略スコアリング
- 過剰適応対策: ペナルティによる調整
- 「直近で偶然勝っただけ」を避ける: instabilityペナルティ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScorerConfig:
    """スコアリング設定

    Attributes:
        penalty_turnover: ターンオーバーペナルティ係数（λ1）
        penalty_mdd: 最大ドローダウンペナルティ係数（λ2）
        penalty_instability: 不安定性ペナルティ係数（λ3）
        mdd_normalization_pct: MDD正規化基準（%）
        min_score: 最小スコア（これ以下は0にクリップ）
        sharpe_adjustment_factor: シャープレシオ調整係数
        return_bonus_scale: リターンボーナス係数
        alpha_bonus_scale: アルファボーナス係数
        benchmark_ticker: ベンチマークティッカー
    """

    penalty_turnover: float = 0.1
    penalty_mdd: float = 0.2
    penalty_instability: float = 0.15
    mdd_normalization_pct: float = 25.0
    min_score: float = 0.0
    sharpe_adjustment_factor: float = 1.0
    return_bonus_scale: float = 0.3
    alpha_bonus_scale: float = 0.5
    benchmark_ticker: str = "SPY"

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.penalty_turnover < 0:
            raise ValueError("penalty_turnover must be >= 0")
        if self.penalty_mdd < 0:
            raise ValueError("penalty_mdd must be >= 0")
        if self.penalty_instability < 0:
            raise ValueError("penalty_instability must be >= 0")
        if self.mdd_normalization_pct <= 0:
            raise ValueError("mdd_normalization_pct must be > 0")
        if self.return_bonus_scale < 0:
            raise ValueError("return_bonus_scale must be >= 0")
        if self.alpha_bonus_scale < 0:
            raise ValueError("alpha_bonus_scale must be >= 0")


@dataclass
class StrategyScoreResult:
    """スコアリング結果

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        raw_sharpe: 元のシャープレシオ
        adjusted_sharpe: 調整後シャープレシオ
        penalty_breakdown: ペナルティ内訳
        total_penalty: 合計ペナルティ
        bonus_breakdown: ボーナス内訳
        total_bonus: 合計ボーナス
        final_score: 最終スコア
        metadata: 追加メタデータ
    """

    strategy_id: str
    asset_id: str
    raw_sharpe: float
    adjusted_sharpe: float
    penalty_breakdown: dict[str, float] = field(default_factory=dict)
    total_penalty: float = 0.0
    bonus_breakdown: dict[str, float] = field(default_factory=dict)
    total_bonus: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "strategy_id": self.strategy_id,
            "asset_id": self.asset_id,
            "raw_sharpe": self.raw_sharpe,
            "adjusted_sharpe": self.adjusted_sharpe,
            "penalty_breakdown": self.penalty_breakdown,
            "total_penalty": self.total_penalty,
            "bonus_breakdown": self.bonus_breakdown,
            "total_bonus": self.total_bonus,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }


@dataclass
class StrategyMetricsInput:
    """スコアリング用の戦略メトリクス入力

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        sharpe_ratio: シャープレシオ
        max_drawdown_pct: 最大ドローダウン（%）
        turnover: ターンオーバー率（0-1）
        period_returns: 各期間のリターン（安定性判定用）
        annualized_return: 年率リターン（リターンボーナス計算用）
        benchmark_return: ベンチマークリターン（アルファボーナス計算用）
    """

    strategy_id: str
    asset_id: str
    sharpe_ratio: float
    max_drawdown_pct: float
    turnover: float = 0.0
    period_returns: list[float] = field(default_factory=list)
    annualized_return: float = 0.0
    benchmark_return: float | None = None


class StrategyScorer:
    """戦略スコアリングクラス

    戦略の総合スコアを計算する。
    スコアは戦略の良さを表す数値で、重み計算の入力となる。

    計算式:
        score = Sharpe_adj - penalty
        penalty = λ1 * turnover + λ2 * MDD_norm + λ3 * instability

    Usage:
        config = ScorerConfig(penalty_mdd=0.2)
        scorer = StrategyScorer(config)

        metrics = StrategyMetricsInput(
            strategy_id="momentum",
            asset_id="AAPL",
            sharpe_ratio=1.5,
            max_drawdown_pct=15.0,
            turnover=0.3,
            period_returns=[0.02, 0.01, -0.01, 0.03, 0.02, 0.01],
        )

        result = scorer.score(metrics)
        print(f"Final score: {result.final_score}")

    Dynamic Parameters:
        use_dynamic=True でデータに基づく動的パラメータを使用可能。
        動的パラメータが計算できない場合はデフォルト値にフォールバック。
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        use_dynamic: bool = False,
        strategy_metrics: Any = None,
        market_conditions: Any = None,
    ) -> None:
        """初期化

        Args:
            config: スコアリング設定。Noneの場合はデフォルト値を使用
            use_dynamic: 動的パラメータを使用するかどうか
            strategy_metrics: 動的パラメータ計算用の戦略メトリクス
            market_conditions: 動的パラメータ計算用の市場状況
        """
        self._use_dynamic = use_dynamic
        self._strategy_metrics = strategy_metrics
        self._market_conditions = market_conditions

        if use_dynamic and config is None:
            self.config = self._compute_dynamic_config()
        else:
            self.config = config or ScorerConfig()

    def _compute_dynamic_config(self) -> ScorerConfig:
        """動的パラメータからConfigを計算

        Returns:
            動的に計算されたScorerConfig
        """
        try:
            from src.meta.dynamic_scorer_params import (
                DynamicScorerParamsCalculator,
            )

            calculator = DynamicScorerParamsCalculator()

            if self._strategy_metrics is not None and self._market_conditions is not None:
                params = calculator.calculate(
                    self._strategy_metrics,
                    self._market_conditions,
                )
                return ScorerConfig(**params.to_scorer_config_kwargs())
            else:
                logger.warning(
                    "Dynamic scorer params requested but no metrics/conditions provided. "
                    "Using default config."
                )
                return ScorerConfig()

        except Exception as e:
            logger.warning(
                "Failed to compute dynamic scorer config: %s. Using defaults.", e
            )
            return ScorerConfig()

    def score(self, metrics: StrategyMetricsInput) -> StrategyScoreResult:
        """戦略のスコアを計算

        計算式:
            score = Sharpe_adj * (1 + return_bonus) - penalty + alpha_bonus

        Args:
            metrics: 戦略メトリクス

        Returns:
            スコアリング結果
        """
        # 調整後シャープレシオ
        adjusted_sharpe = metrics.sharpe_ratio * self.config.sharpe_adjustment_factor

        # ペナルティ計算
        penalty_breakdown = {}

        # 1. ターンオーバーペナルティ
        turnover_penalty = self.config.penalty_turnover * metrics.turnover
        penalty_breakdown["turnover"] = turnover_penalty

        # 2. MDDペナルティ（正規化: 0-25% -> 0-1）
        mdd_normalized = min(
            metrics.max_drawdown_pct / self.config.mdd_normalization_pct, 1.0
        )
        mdd_penalty = self.config.penalty_mdd * mdd_normalized
        penalty_breakdown["mdd"] = mdd_penalty

        # 3. 不安定性ペナルティ
        instability_penalty = self._calculate_instability_penalty(metrics.period_returns)
        penalty_breakdown["instability"] = instability_penalty

        # 合計ペナルティ
        total_penalty = sum(penalty_breakdown.values())

        # ボーナス計算
        bonus_breakdown = {}

        # 1. リターンボーナス（Sharpeに乗算）
        return_bonus = self._calculate_return_bonus(metrics.annualized_return)
        bonus_breakdown["return"] = return_bonus

        # 2. アルファボーナス（加算）
        alpha_bonus = self._calculate_alpha_bonus(
            metrics.annualized_return, metrics.benchmark_return
        )
        bonus_breakdown["alpha"] = alpha_bonus

        # 3. 連勝ボーナス（加算）
        win_streak_bonus = self._calculate_win_streak_bonus(metrics.period_returns)
        bonus_breakdown["win_streak"] = win_streak_bonus

        # 合計ボーナス（加算分のみ）
        total_bonus = alpha_bonus + win_streak_bonus

        # 最終スコア: Sharpe_adj * (1 + return_bonus) - penalty + alpha_bonus + win_streak_bonus
        final_score = adjusted_sharpe * (1 + return_bonus) - total_penalty + total_bonus

        # 最小スコアでクリップ
        final_score = max(final_score, self.config.min_score)

        result = StrategyScoreResult(
            strategy_id=metrics.strategy_id,
            asset_id=metrics.asset_id,
            raw_sharpe=metrics.sharpe_ratio,
            adjusted_sharpe=adjusted_sharpe,
            penalty_breakdown=penalty_breakdown,
            total_penalty=total_penalty,
            bonus_breakdown=bonus_breakdown,
            total_bonus=total_bonus,
            final_score=final_score,
            metadata={
                "mdd_normalized": mdd_normalized,
                "turnover": metrics.turnover,
                "annualized_return": metrics.annualized_return,
                "benchmark_return": metrics.benchmark_return,
            },
        )

        logger.debug(
            "Scored strategy %s for %s: sharpe=%.3f, penalty=%.3f, bonus=%.3f, score=%.3f",
            metrics.strategy_id,
            metrics.asset_id,
            adjusted_sharpe,
            total_penalty,
            total_bonus,
            final_score,
        )

        return result

    def score_batch(
        self, metrics_list: list[StrategyMetricsInput]
    ) -> list[StrategyScoreResult]:
        """複数戦略のスコアを一括計算

        Args:
            metrics_list: 戦略メトリクスのリスト

        Returns:
            スコアリング結果のリスト
        """
        return [self.score(metrics) for metrics in metrics_list]

    def _calculate_instability_penalty(self, period_returns: list[float]) -> float:
        """不安定性ペナルティを計算

        過去N期間のうち、リターンがマイナスだった期間の割合をペナルティとする。
        これにより「直近で偶然勝っただけ」の戦略を抑制する。

        Args:
            period_returns: 各期間のリターン

        Returns:
            不安定性ペナルティ
        """
        if not period_returns:
            return 0.0

        negative_count = sum(1 for r in period_returns if r <= 0)
        negative_ratio = negative_count / len(period_returns)

        return self.config.penalty_instability * negative_ratio

    def _calculate_return_bonus(self, annualized_return: float) -> float:
        """リターンボーナスを計算

        絶対リターンが高い戦略を優遇する。
        20%リターンで飽和するtanh関数を使用。

        Args:
            annualized_return: 年率リターン（例: 0.15 = 15%）

        Returns:
            リターンボーナス（0以上）
        """
        if annualized_return <= 0:
            return 0.0
        return float(np.tanh(annualized_return / 0.20) * self.config.return_bonus_scale)

    def _calculate_alpha_bonus(
        self, strategy_return: float, benchmark_return: float | None
    ) -> float:
        """アルファボーナスを計算

        ベンチマーク超過リターン（アルファ）を評価する。
        正のアルファにのみボーナスを付与。

        Args:
            strategy_return: 戦略リターン（年率）
            benchmark_return: ベンチマークリターン（年率）、Noneの場合は0を返す

        Returns:
            アルファボーナス（0以上）
        """
        if benchmark_return is None:
            return 0.0
        alpha = strategy_return - benchmark_return
        if alpha > 0:
            return alpha * self.config.alpha_bonus_scale
        return 0.0

    def _calculate_win_streak_bonus(self, period_returns: list[float]) -> float:
        """連勝ボーナスを計算

        直近の連続プラスリターンを評価する。
        最大10%ボーナス（5連勝）。

        Args:
            period_returns: 各期間のリターン

        Returns:
            連勝ボーナス（最大0.10）
        """
        if not period_returns:
            return 0.0

        streak = 0
        for r in reversed(period_returns):
            if r > 0:
                streak += 1
            else:
                break

        return min(streak * 0.02, 0.10)

    def rank_strategies(
        self, results: list[StrategyScoreResult]
    ) -> list[StrategyScoreResult]:
        """戦略をスコア順にランキング

        Args:
            results: スコアリング結果のリスト

        Returns:
            スコア降順でソートされた結果
        """
        return sorted(results, key=lambda r: r.final_score, reverse=True)


def create_scorer_from_settings() -> StrategyScorer:
    """グローバル設定からScorerを生成

    Returns:
        設定済みのStrategyScorer
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        config = ScorerConfig(
            penalty_turnover=settings.strategy_weighting.penalty_turnover,
            penalty_mdd=settings.strategy_weighting.penalty_mdd,
            penalty_instability=settings.strategy_weighting.penalty_instability,
        )
        return StrategyScorer(config)
    except ImportError:
        logger.warning("Settings not available, using default ScorerConfig")
        return StrategyScorer()
