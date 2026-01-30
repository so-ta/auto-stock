"""
Strategy Weighter Module - 戦略重み計算

スコアに基づいて各戦略の重みを計算する。
Softmax関数を使用し、上限制約とスパース化を適用。

計算式:
    w_s ∝ exp(β * score_s)  (softmax)
    制約: w_s <= w_strategy_max
    低スコア戦略は0にクリップ（スパース化）

設計根拠:
- 要求.md §7.3: 戦略重み配分
- 過剰適応対策: β固定または上限付き
- 上限制約で単一戦略への集中を防止
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scorer import StrategyScoreResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeighterConfig:
    """重み計算設定

    Attributes:
        beta: Softmax温度パラメータ（高いほど勝者総取り傾向）
        w_strategy_max: 単一戦略の最大重み
        score_threshold: この値以下のスコアは重み0（スパース化）
        normalize: 重みを正規化するか（合計1.0）
        beta_max: βの上限（過剰適応防止）
    """

    beta: float = 2.0
    w_strategy_max: float = 0.5
    score_threshold: float = 0.0
    normalize: bool = True
    beta_max: float = 10.0

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.beta > self.beta_max:
            raise ValueError(f"beta ({self.beta}) exceeds max ({self.beta_max})")
        if not 0 < self.w_strategy_max <= 1:
            raise ValueError("w_strategy_max must be in (0, 1]")


@dataclass
class StrategyWeightItem:
    """個別戦略の重み情報

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        score: スコア
        raw_weight: 正規化前の重み
        capped_weight: 上限適用後の重み
        final_weight: 最終重み（正規化後）
        is_active: アクティブかどうか（重み > 0）
    """

    strategy_id: str
    asset_id: str
    score: float
    raw_weight: float = 0.0
    capped_weight: float = 0.0
    final_weight: float = 0.0
    is_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "strategy_id": self.strategy_id,
            "asset_id": self.asset_id,
            "score": self.score,
            "raw_weight": self.raw_weight,
            "capped_weight": self.capped_weight,
            "final_weight": self.final_weight,
            "is_active": self.is_active,
        }


@dataclass
class WeightingResult:
    """重み計算結果

    Attributes:
        weights: 戦略ごとの重み情報
        total_raw_weight: 正規化前の合計重み
        total_final_weight: 最終合計重み
        active_count: アクティブ戦略数
        capped_count: 上限適用された戦略数
        sparse_count: スパース化された戦略数
        metadata: 追加メタデータ
    """

    weights: list[StrategyWeightItem] = field(default_factory=list)
    total_raw_weight: float = 0.0
    total_final_weight: float = 0.0
    active_count: int = 0
    capped_count: int = 0
    sparse_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": [w.to_dict() for w in self.weights],
            "total_raw_weight": self.total_raw_weight,
            "total_final_weight": self.total_final_weight,
            "active_count": self.active_count,
            "capped_count": self.capped_count,
            "sparse_count": self.sparse_count,
            "metadata": self.metadata,
        }

    def get_weight_dict(self) -> dict[str, float]:
        """strategy_id -> weight の辞書を取得"""
        return {w.strategy_id: w.final_weight for w in self.weights}

    def get_active_strategies(self) -> list[str]:
        """アクティブな戦略IDリストを取得"""
        return [w.strategy_id for w in self.weights if w.is_active]


class StrategyWeighter:
    """戦略重み計算クラス

    スコアに基づいて各戦略の重みを計算する。
    Softmax関数による確率的重み付けと、上限制約・スパース化を適用。

    Usage:
        config = WeighterConfig(beta=2.0, w_strategy_max=0.5)
        weighter = StrategyWeighter(config)

        score_results = [
            StrategyScoreResult(strategy_id="momentum", asset_id="AAPL", final_score=1.5, ...),
            StrategyScoreResult(strategy_id="reversal", asset_id="AAPL", final_score=0.8, ...),
        ]

        result = weighter.calculate_weights(score_results)
        print(result.get_weight_dict())

    Dynamic Parameters:
        use_dynamic=True でデータに基づく動的パラメータを使用可能。
        動的パラメータが計算できない場合はデフォルト値にフォールバック。
    """

    def __init__(
        self,
        config: WeighterConfig | None = None,
        use_dynamic: bool = False,
        strategy_scores: Any = None,
        num_strategies: int | None = None,
    ) -> None:
        """初期化

        Args:
            config: 重み計算設定。Noneの場合はデフォルト値を使用
            use_dynamic: 動的パラメータを使用するかどうか
            strategy_scores: 動的パラメータ計算用のスコア履歴（DataFrame or Series）
            num_strategies: 戦略数
        """
        self._use_dynamic = use_dynamic
        self._strategy_scores = strategy_scores
        self._num_strategies = num_strategies

        if use_dynamic and config is None:
            self.config = self._compute_dynamic_config()
        else:
            self.config = config or WeighterConfig()

    def _compute_dynamic_config(self) -> WeighterConfig:
        """動的パラメータからConfigを計算

        Returns:
            動的に計算されたWeighterConfig
        """
        try:
            from src.meta.dynamic_weighter_params import (
                DynamicWeighterParamsCalculator,
            )

            calculator = DynamicWeighterParamsCalculator()

            if self._strategy_scores is not None:
                params = calculator.calculate_all(
                    self._strategy_scores,
                    num_strategies=self._num_strategies,
                )
                return WeighterConfig(**params.to_weighter_config_dict())
            else:
                logger.warning(
                    "Dynamic weighter params requested but no score history provided. "
                    "Using default config."
                )
                return WeighterConfig()

        except Exception as e:
            logger.warning(
                "Failed to compute dynamic weighter config: %s. Using defaults.", e
            )
            return WeighterConfig()

    def calculate_weights(
        self, score_results: list[StrategyScoreResult]
    ) -> WeightingResult:
        """戦略重みを計算

        Args:
            score_results: スコアリング結果のリスト

        Returns:
            重み計算結果
        """
        if not score_results:
            return WeightingResult(metadata={"error": "No strategies provided"})

        # スコア抽出とスパース化
        scores = np.array([r.final_score for r in score_results])
        sparse_mask = scores > self.config.score_threshold
        sparse_count = int((~sparse_mask).sum())

        # アクティブなスコアのみでsoftmax計算
        active_scores = np.where(sparse_mask, scores, -np.inf)

        # Softmax計算（数値安定性のためmax減算）
        if np.all(active_scores == -np.inf):
            # 全てスパース化された場合
            raw_weights = np.zeros_like(scores)
        else:
            max_score = np.max(active_scores[active_scores != -np.inf])
            exp_scores = np.exp(self.config.beta * (active_scores - max_score))
            exp_scores = np.where(sparse_mask, exp_scores, 0.0)
            raw_weights = exp_scores / np.sum(exp_scores) if np.sum(exp_scores) > 0 else exp_scores

        total_raw_weight = float(np.sum(raw_weights))

        # 上限制約適用
        capped_weights = np.minimum(raw_weights, self.config.w_strategy_max)
        capped_count = int(np.sum(raw_weights > self.config.w_strategy_max))

        # 正規化（オプション）
        if self.config.normalize and np.sum(capped_weights) > 0:
            final_weights = capped_weights / np.sum(capped_weights)
        else:
            final_weights = capped_weights

        total_final_weight = float(np.sum(final_weights))

        # 結果構築
        weight_items = []
        for i, result in enumerate(score_results):
            is_active = sparse_mask[i] and final_weights[i] > 0
            item = StrategyWeightItem(
                strategy_id=result.strategy_id,
                asset_id=result.asset_id,
                score=result.final_score,
                raw_weight=float(raw_weights[i]),
                capped_weight=float(capped_weights[i]),
                final_weight=float(final_weights[i]),
                is_active=bool(is_active),
            )
            weight_items.append(item)

        active_count = sum(1 for w in weight_items if w.is_active)

        result = WeightingResult(
            weights=weight_items,
            total_raw_weight=total_raw_weight,
            total_final_weight=total_final_weight,
            active_count=active_count,
            capped_count=capped_count,
            sparse_count=sparse_count,
            metadata={
                "beta": self.config.beta,
                "w_strategy_max": self.config.w_strategy_max,
                "score_threshold": self.config.score_threshold,
            },
        )

        logger.info(
            "Calculated weights for %d strategies: %d active, %d capped, %d sparse",
            len(score_results),
            active_count,
            capped_count,
            sparse_count,
        )

        return result

    def calculate_weights_from_scores(
        self,
        strategy_scores: dict[str, float],
        asset_id: str = "default",
    ) -> WeightingResult:
        """スコア辞書から重みを計算（簡易版）

        Args:
            strategy_scores: strategy_id -> score の辞書
            asset_id: アセットID

        Returns:
            重み計算結果
        """
        # ダミーのScoreResultを生成
        score_results = [
            StrategyScoreResult(
                strategy_id=strategy_id,
                asset_id=asset_id,
                raw_sharpe=score,
                adjusted_sharpe=score,
                final_score=score,
            )
            for strategy_id, score in strategy_scores.items()
        ]
        return self.calculate_weights(score_results)


def create_weighter_from_settings() -> StrategyWeighter:
    """グローバル設定からWeighterを生成

    Returns:
        設定済みのStrategyWeighter
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        config = WeighterConfig(
            beta=settings.strategy_weighting.beta,
            w_strategy_max=settings.strategy_weighting.w_strategy_max,
        )
        return StrategyWeighter(config)
    except ImportError:
        logger.warning("Settings not available, using default WeighterConfig")
        return StrategyWeighter()
