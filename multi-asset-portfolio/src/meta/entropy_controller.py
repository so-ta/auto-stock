"""
Entropy Controller Module - エントロピー下限制御

重み分布のエントロピーを監視し、多様性を維持する。
勝ち始めた戦略への「全乗り」を防止するための制御機構。

エントロピー計算:
    H(w) = -Σ w_i * log(w_i)
    正規化エントロピー: H_norm = H(w) / log(n)  (0=集中, 1=均等)

制御:
    H_norm < entropy_min の場合、重みを均等方向に調整

設計根拠:
- 要求.md §7.4: 多様性維持
- 過剰適応対策: 単一戦略への集中を抑制
- リスク分散: 複数戦略の維持
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropyConfig:
    """エントロピー制御設定

    Attributes:
        entropy_min: 最小正規化エントロピー（0=集中OK, 1=完全均等必須）
        adjustment_strength: 調整強度（0-1）
        min_active_strategies: 最小アクティブ戦略数
        max_iterations: 調整の最大イテレーション数
        convergence_threshold: 収束判定閾値
    """

    entropy_min: float = 0.8
    adjustment_strength: float = 0.5
    min_active_strategies: int = 2
    max_iterations: int = 10
    convergence_threshold: float = 0.001

    def __post_init__(self) -> None:
        """バリデーション"""
        if not 0 <= self.entropy_min <= 1:
            raise ValueError("entropy_min must be in [0, 1]")
        if not 0 < self.adjustment_strength <= 1:
            raise ValueError("adjustment_strength must be in (0, 1]")
        if self.min_active_strategies < 1:
            raise ValueError("min_active_strategies must be >= 1")


@dataclass
class EntropyControlResult:
    """エントロピー制御結果

    Attributes:
        original_weights: 調整前の重み
        adjusted_weights: 調整後の重み
        original_entropy: 調整前の正規化エントロピー
        adjusted_entropy: 調整後の正規化エントロピー
        was_adjusted: 調整が行われたか
        iterations_used: 使用したイテレーション数
        metadata: 追加メタデータ
    """

    original_weights: dict[str, float] = field(default_factory=dict)
    adjusted_weights: dict[str, float] = field(default_factory=dict)
    original_entropy: float = 0.0
    adjusted_entropy: float = 0.0
    was_adjusted: bool = False
    iterations_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "original_weights": self.original_weights,
            "adjusted_weights": self.adjusted_weights,
            "original_entropy": self.original_entropy,
            "adjusted_entropy": self.adjusted_entropy,
            "was_adjusted": self.was_adjusted,
            "iterations_used": self.iterations_used,
            "metadata": self.metadata,
        }


class EntropyController:
    """エントロピー制御クラス

    重み分布のエントロピーを監視し、多様性を維持する。
    エントロピーが閾値を下回る場合、重みを均等方向に調整する。

    Usage:
        config = EntropyConfig(entropy_min=0.8)
        controller = EntropyController(config)

        weights = {"momentum": 0.7, "reversal": 0.2, "breakout": 0.1}
        result = controller.control(weights)

        if result.was_adjusted:
            print(f"Adjusted weights: {result.adjusted_weights}")
            print(f"Entropy: {result.original_entropy} -> {result.adjusted_entropy}")
    """

    def __init__(self, config: EntropyConfig | None = None) -> None:
        """初期化

        Args:
            config: エントロピー制御設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or EntropyConfig()

    def control(self, weights: dict[str, float]) -> EntropyControlResult:
        """エントロピー制御を実行

        Args:
            weights: strategy_id -> weight の辞書

        Returns:
            エントロピー制御結果
        """
        if not weights:
            return EntropyControlResult(
                metadata={"error": "No weights provided"}
            )

        # アクティブな重みのみ抽出
        active_weights = {k: v for k, v in weights.items() if v > 0}

        if len(active_weights) < self.config.min_active_strategies:
            # アクティブ戦略が少なすぎる場合は均等配分
            logger.warning(
                "Active strategies (%d) < min (%d), forcing equal weights",
                len(active_weights),
                self.config.min_active_strategies,
            )
            equal_weight = 1.0 / len(weights) if weights else 0.0
            adjusted = {k: equal_weight for k in weights}
            return EntropyControlResult(
                original_weights=weights.copy(),
                adjusted_weights=adjusted,
                original_entropy=self._calculate_normalized_entropy(list(weights.values())),
                adjusted_entropy=1.0,  # 均等配分は最大エントロピー
                was_adjusted=True,
                iterations_used=0,
                metadata={"reason": "min_active_strategies_enforced"},
            )

        # 元のエントロピー計算
        weight_values = list(active_weights.values())
        original_entropy = self._calculate_normalized_entropy(weight_values)

        # エントロピーが閾値以上なら調整不要
        if original_entropy >= self.config.entropy_min:
            return EntropyControlResult(
                original_weights=weights.copy(),
                adjusted_weights=weights.copy(),
                original_entropy=original_entropy,
                adjusted_entropy=original_entropy,
                was_adjusted=False,
                iterations_used=0,
                metadata={"reason": "entropy_sufficient"},
            )

        # エントロピー調整
        adjusted_weights, iterations, final_entropy = self._adjust_weights(
            active_weights, original_entropy
        )

        # 非アクティブな戦略の重みを0で追加
        for k in weights:
            if k not in adjusted_weights:
                adjusted_weights[k] = 0.0

        result = EntropyControlResult(
            original_weights=weights.copy(),
            adjusted_weights=adjusted_weights,
            original_entropy=original_entropy,
            adjusted_entropy=final_entropy,
            was_adjusted=True,
            iterations_used=iterations,
            metadata={
                "reason": "entropy_below_threshold",
                "entropy_min": self.config.entropy_min,
                "adjustment_strength": self.config.adjustment_strength,
            },
        )

        logger.info(
            "Entropy adjusted: %.3f -> %.3f (threshold: %.3f, iterations: %d)",
            original_entropy,
            final_entropy,
            self.config.entropy_min,
            iterations,
        )

        return result

    def _calculate_normalized_entropy(self, weights: list[float]) -> float:
        """正規化エントロピーを計算

        H_norm = H(w) / log(n)
        ここで H(w) = -Σ w_i * log(w_i)

        Args:
            weights: 重みリスト

        Returns:
            正規化エントロピー（0-1）
        """
        weights = np.array(weights)
        weights = weights[weights > 0]  # 0を除外

        if len(weights) <= 1:
            return 0.0

        # 正規化
        weights = weights / np.sum(weights)

        # エントロピー計算
        entropy = -np.sum(weights * np.log(weights))

        # 最大エントロピー（均等配分の場合）
        max_entropy = np.log(len(weights))

        if max_entropy == 0:
            return 0.0

        return float(entropy / max_entropy)

    def _adjust_weights(
        self,
        weights: dict[str, float],
        current_entropy: float,
    ) -> tuple[dict[str, float], int, float]:
        """重みを均等方向に調整

        調整式: w_new = w_old + α * (w_uniform - w_old)
        ここで α = adjustment_strength

        Args:
            weights: 現在の重み
            current_entropy: 現在のエントロピー

        Returns:
            (調整後の重み, イテレーション数, 最終エントロピー)
        """
        n = len(weights)
        uniform_weight = 1.0 / n

        current_weights = np.array(list(weights.values()))
        strategy_ids = list(weights.keys())

        final_entropy = current_entropy
        iterations = 0

        for i in range(self.config.max_iterations):
            iterations = i + 1

            # 均等方向に調整
            adjustment = self.config.adjustment_strength * (uniform_weight - current_weights)
            new_weights = current_weights + adjustment

            # 負の重みを0に、正規化
            new_weights = np.maximum(new_weights, 0)
            new_weights = new_weights / np.sum(new_weights)

            # エントロピー計算
            new_entropy = self._calculate_normalized_entropy(new_weights.tolist())
            final_entropy = new_entropy

            # 収束判定
            if new_entropy >= self.config.entropy_min:
                current_weights = new_weights
                break

            weight_diff = np.max(np.abs(new_weights - current_weights))
            if weight_diff < self.config.convergence_threshold:
                current_weights = new_weights
                break

            current_weights = new_weights

        # 結果を辞書に戻す
        adjusted_weights = {
            strategy_ids[i]: float(current_weights[i])
            for i in range(n)
        }

        return adjusted_weights, iterations, final_entropy

    def check_entropy(self, weights: dict[str, float]) -> tuple[float, bool]:
        """エントロピーをチェック

        Args:
            weights: strategy_id -> weight の辞書

        Returns:
            (正規化エントロピー, 閾値以上かどうか)
        """
        active_weights = [v for v in weights.values() if v > 0]
        entropy = self._calculate_normalized_entropy(active_weights)
        is_sufficient = entropy >= self.config.entropy_min
        return entropy, is_sufficient


def create_entropy_controller_from_settings() -> EntropyController:
    """グローバル設定からEntropyControllerを生成

    Returns:
        設定済みのEntropyController
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        config = EntropyConfig(
            entropy_min=settings.strategy_weighting.entropy_min,
        )
        return EntropyController(config)
    except ImportError:
        logger.warning("Settings not available, using default EntropyConfig")
        return EntropyController()
