"""
Ensemble Combiner Module - アンサンブル統合

複数戦略のスコアを統合し、最終的なシグナルを生成する。

実装手法:
1. StackingEnsemble: メタモデル（Ridge回帰）で統合
2. VotingEnsemble: 閾値ベースの多数決
3. WeightedAverageEnsemble: 過去パフォーマンスに基づく加重平均

計算式（WeightedAverage）:
    final_score = Σ(strategy_weight × strategy_score)
    strategy_weight ∝ exp(β × past_sharpe)

設計根拠:
- アンサンブル学習: 単一戦略より安定したシグナル
- 適応的重み: 過去パフォーマンスで動的調整
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleMethod(str, Enum):
    """アンサンブル手法"""

    STACKING = "stacking"
    VOTING = "voting"
    WEIGHTED_AVG = "weighted_avg"


class VoteType(str, Enum):
    """投票タイプ"""

    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EnsembleCombinerConfig:
    """アンサンブル統合設定

    Attributes:
        method: 統合手法（"stacking", "voting", "weighted_avg"）
        vote_threshold: 投票の閾値（score > threshold → 買い）
        weight_decay: 指数加重の減衰率（最近を重視）
        ridge_alpha: Ridge正則化パラメータ
        beta: 加重平均の温度パラメータ（高いほど勝者重視）
        min_weight: 最小重み（0未満を防止）
        normalize_weights: 重みを正規化するか
    """

    method: Literal["stacking", "voting", "weighted_avg"] = "weighted_avg"
    vote_threshold: float = 0.3
    weight_decay: float = 0.95
    ridge_alpha: float = 1.0
    beta: float = 2.0
    min_weight: float = 0.0
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        valid_methods = {"stacking", "voting", "weighted_avg"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method: {self.method}. Must be one of {valid_methods}"
            )
        if self.vote_threshold < 0:
            raise ValueError("vote_threshold must be >= 0")
        if not 0 < self.weight_decay <= 1:
            raise ValueError("weight_decay must be in (0, 1]")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")


@dataclass
class EnsembleCombineResult:
    """アンサンブル統合結果

    Attributes:
        combined_scores: 統合後のスコア（pd.Series）
        strategy_weights: 各戦略の重み
        method_used: 使用した手法
        vote_counts: 投票カウント（voting時のみ）
        metadata: 追加メタデータ
    """

    combined_scores: pd.Series
    strategy_weights: dict[str, float] = field(default_factory=dict)
    method_used: str = ""
    vote_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な結果かどうか"""
        return not self.combined_scores.empty and not self.combined_scores.isna().all()

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "combined_scores": self.combined_scores.to_dict(),
            "strategy_weights": self.strategy_weights,
            "method_used": self.method_used,
            "vote_counts": self.vote_counts,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


class EnsembleCombiner:
    """アンサンブル統合クラス

    複数戦略のスコアを統合し、最終的なシグナルを生成する。

    Usage:
        config = EnsembleCombinerConfig(method="weighted_avg")
        combiner = EnsembleCombiner(config)

        # strategy_scores: {strategy_name: pd.Series of scores}
        strategy_scores = {
            "momentum": pd.Series([0.5, 0.3, 0.7], index=dates),
            "reversal": pd.Series([0.2, 0.4, 0.1], index=dates),
        }

        # past_performance: {strategy_name: past_sharpe}
        past_performance = {"momentum": 1.5, "reversal": 0.8}

        result = combiner.combine(strategy_scores, past_performance)
        print(result.combined_scores)
    """

    def __init__(self, config: EnsembleCombinerConfig | None = None) -> None:
        """初期化

        Args:
            config: アンサンブル設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or EnsembleCombinerConfig()
        self._meta_model: Any = None  # Stacking用のメタモデル

    def combine(
        self,
        strategy_scores: dict[str, pd.Series],
        past_performance: dict[str, float] | None = None,
    ) -> EnsembleCombineResult:
        """戦略スコアを統合

        Args:
            strategy_scores: 戦略名 -> スコアのSeries
            past_performance: 戦略名 -> 過去パフォーマンス（Sharpe等）

        Returns:
            統合結果
        """
        if not strategy_scores:
            logger.warning("No strategy scores provided")
            return EnsembleCombineResult(
                combined_scores=pd.Series(dtype=float),
                method_used=self.config.method,
                metadata={"error": "No strategy scores"},
            )

        # 手法に応じて統合
        method = self.config.method

        if method == EnsembleMethod.STACKING.value:
            result = self._stacking_combine(strategy_scores)
        elif method == EnsembleMethod.VOTING.value:
            result = self._voting_combine(strategy_scores)
        elif method == EnsembleMethod.WEIGHTED_AVG.value:
            result = self._weighted_avg_combine(strategy_scores, past_performance)
        else:
            raise ValueError(f"Unknown method: {method}")

        result.method_used = method

        logger.info(
            "Ensemble combine completed: method=%s, strategies=%d",
            method,
            len(strategy_scores),
        )

        return result

    def _stacking_combine(
        self,
        strategy_scores: dict[str, pd.Series],
    ) -> EnsembleCombineResult:
        """スタッキングアンサンブル

        各戦略のスコアを特徴量として、Ridge回帰で最終スコアを予測。
        学習データがない場合は単純平均にフォールバック。

        Args:
            strategy_scores: 戦略スコア辞書

        Returns:
            統合結果
        """
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            logger.warning("sklearn not available, falling back to weighted_avg")
            return self._weighted_avg_combine(strategy_scores, None)

        # DataFrameに変換
        scores_df = pd.DataFrame(strategy_scores)
        scores_df = scores_df.dropna()

        if scores_df.empty or len(scores_df) < 2:
            logger.warning("Insufficient data for stacking, using simple average")
            combined = scores_df.mean(axis=1) if not scores_df.empty else pd.Series(dtype=float)
            return EnsembleCombineResult(
                combined_scores=combined,
                strategy_weights={s: 1.0 / len(strategy_scores) for s in strategy_scores},
                metadata={"fallback": "simple_average", "reason": "insufficient_data"},
            )

        # メタモデルが学習済みでない場合
        if self._meta_model is None:
            # 単純な線形結合として、各戦略の寄与を等しくする
            # 実際の学習にはターゲット変数が必要
            # ここでは暫定的に平均をターゲットとして学習
            X = scores_df.values
            y = scores_df.mean(axis=1).values  # 暫定ターゲット

            self._meta_model = Ridge(alpha=self.config.ridge_alpha)
            self._meta_model.fit(X, y)

        # 予測
        X_pred = scores_df.values
        combined_values = self._meta_model.predict(X_pred)
        combined = pd.Series(combined_values, index=scores_df.index)

        # 係数から重みを取得
        coefs = self._meta_model.coef_
        strategy_names = list(strategy_scores.keys())
        weights = {}
        for i, name in enumerate(strategy_names):
            weights[name] = float(coefs[i]) if i < len(coefs) else 0.0

        return EnsembleCombineResult(
            combined_scores=combined,
            strategy_weights=weights,
            metadata={
                "ridge_alpha": self.config.ridge_alpha,
                "intercept": float(self._meta_model.intercept_),
            },
        )

    def _voting_combine(
        self,
        strategy_scores: dict[str, pd.Series],
    ) -> EnsembleCombineResult:
        """投票アンサンブル

        各戦略の「買い/売り/中立」投票を集計し、多数決で最終判断。
        結果は投票スコア（-1〜1）として返す。

        Args:
            strategy_scores: 戦略スコア辞書

        Returns:
            統合結果
        """
        threshold = self.config.vote_threshold

        # DataFrameに変換
        scores_df = pd.DataFrame(strategy_scores)
        scores_df = scores_df.dropna()

        if scores_df.empty:
            return EnsembleCombineResult(
                combined_scores=pd.Series(dtype=float),
                metadata={"error": "No valid data"},
            )

        # 各時点での投票
        combined_scores = []
        vote_counts_by_date: dict[str, dict[str, int]] = {}

        for date, row in scores_df.iterrows():
            buy_votes = 0
            sell_votes = 0
            neutral_votes = 0

            for strategy_name, score in row.items():
                if pd.isna(score):
                    neutral_votes += 1
                elif score > threshold:
                    buy_votes += 1
                elif score < -threshold:
                    sell_votes += 1
                else:
                    neutral_votes += 1

            total_votes = buy_votes + sell_votes + neutral_votes

            # 投票スコア: (買い - 売り) / 総投票数
            if total_votes > 0:
                vote_score = (buy_votes - sell_votes) / total_votes
            else:
                vote_score = 0.0

            combined_scores.append(vote_score)
            vote_counts_by_date[str(date)] = {
                "buy": buy_votes,
                "sell": sell_votes,
                "neutral": neutral_votes,
            }

        combined = pd.Series(combined_scores, index=scores_df.index)

        # 戦略重みは等しい（投票なので）
        n_strategies = len(strategy_scores)
        weights = {s: 1.0 / n_strategies for s in strategy_scores}

        return EnsembleCombineResult(
            combined_scores=combined,
            strategy_weights=weights,
            vote_counts=vote_counts_by_date,
            metadata={
                "vote_threshold": threshold,
                "n_strategies": n_strategies,
            },
        )

    def _weighted_avg_combine(
        self,
        strategy_scores: dict[str, pd.Series],
        past_performance: dict[str, float] | None,
    ) -> EnsembleCombineResult:
        """加重平均アンサンブル

        各戦略の過去パフォーマンスに基づいて重み付け。
        重み ∝ exp(β × past_sharpe)

        Args:
            strategy_scores: 戦略スコア辞書
            past_performance: 過去パフォーマンス辞書

        Returns:
            統合結果
        """
        # DataFrameに変換
        scores_df = pd.DataFrame(strategy_scores)
        scores_df = scores_df.dropna()

        if scores_df.empty:
            return EnsembleCombineResult(
                combined_scores=pd.Series(dtype=float),
                metadata={"error": "No valid data"},
            )

        strategy_names = list(strategy_scores.keys())
        n_strategies = len(strategy_names)

        # 重み計算
        if past_performance is None or not past_performance:
            # 過去パフォーマンスがない場合は等重み
            raw_weights = {s: 1.0 for s in strategy_names}
            logger.debug("No past performance, using equal weights")
        else:
            # 指数加重: weight ∝ exp(β × past_sharpe)
            raw_weights = {}
            for strategy in strategy_names:
                perf = past_performance.get(strategy, 0.0)
                # 数値安定性のため、極端な値をクリップ
                clipped_perf = max(min(perf, 10.0), -10.0)
                raw_weights[strategy] = math.exp(self.config.beta * clipped_perf)

        # 最小重みでクリップ
        clipped_weights = {
            s: max(w, self.config.min_weight) for s, w in raw_weights.items()
        }

        # 正規化
        if self.config.normalize_weights:
            total_weight = sum(clipped_weights.values())
            if total_weight > 0:
                weights = {s: w / total_weight for s, w in clipped_weights.items()}
            else:
                weights = {s: 1.0 / n_strategies for s in strategy_names}
        else:
            weights = clipped_weights

        # 加重平均を計算
        combined_values = np.zeros(len(scores_df))
        for strategy, weight in weights.items():
            if strategy in scores_df.columns:
                combined_values += weight * scores_df[strategy].values

        combined = pd.Series(combined_values, index=scores_df.index)

        return EnsembleCombineResult(
            combined_scores=combined,
            strategy_weights=weights,
            metadata={
                "beta": self.config.beta,
                "weight_decay": self.config.weight_decay,
                "had_past_performance": past_performance is not None,
            },
        )

    def fit_stacking_model(
        self,
        strategy_scores: dict[str, pd.Series],
        target: pd.Series,
    ) -> None:
        """スタッキングモデルを学習

        Args:
            strategy_scores: 戦略スコア辞書（特徴量）
            target: ターゲット変数（実際のリターン等）
        """
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            logger.error("sklearn not available for stacking model training")
            return

        # DataFrameに変換してアライン
        scores_df = pd.DataFrame(strategy_scores)
        aligned_target = target.reindex(scores_df.index).dropna()
        scores_df = scores_df.loc[aligned_target.index].dropna()

        if len(scores_df) < 10:
            logger.warning("Insufficient samples for stacking model training")
            return

        X = scores_df.values
        y = aligned_target.values

        self._meta_model = Ridge(alpha=self.config.ridge_alpha)
        self._meta_model.fit(X, y)

        logger.info(
            "Stacking model trained: %d samples, %d features",
            len(X),
            X.shape[1],
        )

    def calculate_time_weighted_performance(
        self,
        performance_history: dict[str, list[float]],
    ) -> dict[str, float]:
        """時間加重パフォーマンスを計算

        指数加重で最近のパフォーマンスを重視。

        Args:
            performance_history: 戦略名 -> パフォーマンス履歴

        Returns:
            戦略名 -> 時間加重パフォーマンス
        """
        result = {}
        decay = self.config.weight_decay

        for strategy, history in performance_history.items():
            if not history:
                result[strategy] = 0.0
                continue

            # 指数加重平均
            n = len(history)
            weights = [decay ** (n - 1 - i) for i in range(n)]
            total_weight = sum(weights)

            if total_weight > 0:
                weighted_sum = sum(w * p for w, p in zip(weights, history))
                result[strategy] = weighted_sum / total_weight
            else:
                result[strategy] = 0.0

        return result


def create_combiner_from_settings() -> EnsembleCombiner:
    """グローバル設定からCombinerを生成

    Returns:
        設定済みのEnsembleCombiner
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        # ensemble設定があれば使用
        ensemble_config = getattr(settings, "ensemble", None)

        if ensemble_config is not None:
            config = EnsembleCombinerConfig(
                method=getattr(ensemble_config, "method", "weighted_avg"),
                vote_threshold=getattr(ensemble_config, "vote_threshold", 0.3),
                weight_decay=getattr(ensemble_config, "weight_decay", 0.95),
                ridge_alpha=getattr(ensemble_config, "ridge_alpha", 1.0),
                beta=getattr(ensemble_config, "beta", 2.0),
            )
        else:
            config = EnsembleCombinerConfig()

        return EnsembleCombiner(config)
    except ImportError:
        logger.warning("Settings not available, using default EnsembleCombinerConfig")
        return EnsembleCombiner()
