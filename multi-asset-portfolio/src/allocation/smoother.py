"""
Weight Smoother Module - 重みスムージング

ポートフォリオ重みの急激な変化を抑制するスムージング処理。

実装手法:
1. 指数移動平均（EMA）: w_final = α*w_new + (1-α)*w_prev
2. ターンオーバー制限: 変更量の上限
3. 最小変更閾値: 小さな変更は無視

設計根拠:
- 要求.md §8.6: 最終出力の安定化（必須）
- w_final = α*w_new + (1-α)*w_prev
- ターンオーバー抑制による取引コスト削減

使用方法:
    smoother = WeightSmoother(alpha=0.3)
    smoothed = smoother.smooth(new_weights, previous_weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmootherConfig:
    """スムージング設定

    Attributes:
        alpha: スムージング係数（0=前回維持, 1=新しい値を使用）
        min_change_threshold: この値以下の変更は無視
        max_single_change: 単一アセットの最大変更量
        preserve_direction: 変更の方向を保持するか
        renormalize: スムージング後に再正規化するか
    """

    alpha: float = 0.3
    min_change_threshold: float = 0.005
    max_single_change: float = 0.1
    preserve_direction: bool = True
    renormalize: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.min_change_threshold < 0:
            raise ValueError("min_change_threshold must be >= 0")
        if self.max_single_change <= 0:
            raise ValueError("max_single_change must be > 0")


@dataclass
class SmoothingResult:
    """スムージング結果

    Attributes:
        weights: スムージング後の重み（Series）
        turnover_before: スムージング前のターンオーバー
        turnover_after: スムージング後のターンオーバー
        turnover_reduction: ターンオーバー削減率
        unchanged_assets: 変更されなかったアセット数
        metadata: 追加メタデータ
    """

    weights: pd.Series
    turnover_before: float = 0.0
    turnover_after: float = 0.0
    turnover_reduction: float = 0.0
    unchanged_assets: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な結果かどうか"""
        if self.weights.empty:
            return False
        return np.isclose(self.weights.sum(), 1.0, atol=1e-4)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights.to_dict(),
            "turnover_before": self.turnover_before,
            "turnover_after": self.turnover_after,
            "turnover_reduction": self.turnover_reduction,
            "unchanged_assets": self.unchanged_assets,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


class WeightSmoother:
    """重みスムージングクラス

    新しい目標重みと前回の重みを組み合わせて、
    急激な変更を抑制したスムーズな重みを生成する。

    Usage:
        config = SmootherConfig(alpha=0.3)
        smoother = WeightSmoother(config)

        # new_weights: 新しい目標重み
        # previous_weights: 前期の重み
        result = smoother.smooth(new_weights, previous_weights)

        print(result.weights)
        print(f"Turnover reduced: {result.turnover_reduction:.2%}")
    """

    def __init__(self, config: SmootherConfig | None = None) -> None:
        """初期化

        Args:
            config: スムージング設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or SmootherConfig()

    def smooth(
        self,
        new_weights: pd.Series,
        previous_weights: pd.Series | None = None,
    ) -> SmoothingResult:
        """重みをスムージング

        Args:
            new_weights: 新しい目標重み
            previous_weights: 前期の重み（Noneの場合はnew_weightsをそのまま返す）

        Returns:
            SmoothingResult: スムージング結果
        """
        if new_weights.empty:
            return SmoothingResult(
                weights=new_weights,
                metadata={"error": "Empty weights"},
            )

        if previous_weights is None:
            # 初回はスムージングなし
            return SmoothingResult(
                weights=new_weights,
                metadata={"note": "No previous weights, skipped smoothing"},
            )

        # インデックスを揃える
        all_assets = new_weights.index.union(previous_weights.index)
        new_aligned = new_weights.reindex(all_assets, fill_value=0.0)
        prev_aligned = previous_weights.reindex(all_assets, fill_value=0.0)

        # スムージング前のターンオーバー
        turnover_before = self._calculate_turnover(new_aligned, prev_aligned)

        # EMAスムージング
        smoothed = self._apply_ema_smoothing(new_aligned, prev_aligned)

        # 最小変更閾値の適用
        smoothed, unchanged_count = self._apply_min_change_threshold(
            smoothed, prev_aligned
        )

        # 単一アセット変更量制限
        smoothed = self._apply_max_single_change(smoothed, prev_aligned)

        # 再正規化
        if self.config.renormalize:
            total = smoothed.sum()
            if total > 0:
                smoothed = smoothed / total

        # スムージング後のターンオーバー
        turnover_after = self._calculate_turnover(smoothed, prev_aligned)

        # ターンオーバー削減率
        if turnover_before > 0:
            reduction = 1 - (turnover_after / turnover_before)
        else:
            reduction = 0.0

        logger.info(
            "Smoothing applied: turnover %.4f -> %.4f (%.1f%% reduction)",
            turnover_before,
            turnover_after,
            reduction * 100,
        )

        return SmoothingResult(
            weights=smoothed,
            turnover_before=turnover_before,
            turnover_after=turnover_after,
            turnover_reduction=reduction,
            unchanged_assets=unchanged_count,
            metadata={
                "alpha": self.config.alpha,
                "min_change_threshold": self.config.min_change_threshold,
            },
        )

    def _apply_ema_smoothing(
        self,
        new_weights: pd.Series,
        prev_weights: pd.Series,
    ) -> pd.Series:
        """EMAスムージングを適用

        w_smoothed = α * w_new + (1 - α) * w_prev

        Args:
            new_weights: 新しい重み
            prev_weights: 前回の重み

        Returns:
            スムージング後の重み
        """
        alpha = self.config.alpha
        smoothed = alpha * new_weights + (1 - alpha) * prev_weights
        return smoothed

    def _apply_min_change_threshold(
        self,
        smoothed: pd.Series,
        prev_weights: pd.Series,
    ) -> tuple[pd.Series, int]:
        """最小変更閾値を適用

        変更量が閾値以下の場合は前回の値を維持。

        Args:
            smoothed: スムージング後の重み
            prev_weights: 前回の重み

        Returns:
            (閾値適用後の重み, 変更されなかったアセット数)
        """
        threshold = self.config.min_change_threshold
        changes = (smoothed - prev_weights).abs()

        # 閾値以下は前回の値を維持
        result = smoothed.copy()
        unchanged_mask = changes <= threshold
        result[unchanged_mask] = prev_weights[unchanged_mask]

        unchanged_count = int(unchanged_mask.sum())

        return result, unchanged_count

    def _apply_max_single_change(
        self,
        smoothed: pd.Series,
        prev_weights: pd.Series,
    ) -> pd.Series:
        """単一アセットの最大変更量を制限

        Args:
            smoothed: スムージング後の重み
            prev_weights: 前回の重み

        Returns:
            制限適用後の重み
        """
        max_change = self.config.max_single_change
        result = smoothed.copy()

        for asset in smoothed.index:
            change = smoothed[asset] - prev_weights[asset]

            if abs(change) > max_change:
                if self.config.preserve_direction:
                    # 方向を保持しつつ制限
                    if change > 0:
                        result[asset] = prev_weights[asset] + max_change
                    else:
                        result[asset] = prev_weights[asset] - max_change
                else:
                    # 単純にクリップ
                    result[asset] = prev_weights[asset] + np.clip(
                        change, -max_change, max_change
                    )

        return result

    def _calculate_turnover(
        self,
        weights1: pd.Series,
        weights2: pd.Series,
    ) -> float:
        """ターンオーバーを計算

        ターンオーバー = Σ|w1 - w2| / 2

        Args:
            weights1: 重み1
            weights2: 重み2

        Returns:
            ターンオーバー
        """
        return float((weights1 - weights2).abs().sum()) / 2


class AdaptiveSmoother:
    """適応的スムージング

    市場状況や変化量に応じてスムージング強度を調整する。
    """

    def __init__(
        self,
        base_alpha: float = 0.3,
        volatility_adjustment: bool = True,
        regime_adjustment: bool = True,
    ) -> None:
        """初期化

        Args:
            base_alpha: 基本のスムージング係数
            volatility_adjustment: ボラティリティに応じた調整を行うか
            regime_adjustment: 市場レジームに応じた調整を行うか
        """
        self.base_alpha = base_alpha
        self.volatility_adjustment = volatility_adjustment
        self.regime_adjustment = regime_adjustment

    def compute_adaptive_alpha(
        self,
        current_volatility: float | None = None,
        average_volatility: float | None = None,
        is_crisis: bool = False,
    ) -> float:
        """適応的なalpha値を計算

        Args:
            current_volatility: 現在のボラティリティ
            average_volatility: 平均ボラティリティ
            is_crisis: 危機モードかどうか

        Returns:
            調整されたalpha値
        """
        alpha = self.base_alpha

        # 危機時は安定性重視（alpha低く）
        if self.regime_adjustment and is_crisis:
            alpha *= 0.5

        # 高ボラティリティ時は慎重に（alpha低く）
        if (
            self.volatility_adjustment
            and current_volatility is not None
            and average_volatility is not None
            and average_volatility > 0
        ):
            vol_ratio = current_volatility / average_volatility
            if vol_ratio > 1.5:
                # ボラが1.5倍以上なら慎重に
                alpha *= max(0.3, 1.0 / vol_ratio)
            elif vol_ratio < 0.7:
                # ボラが低いなら積極的に
                alpha = min(0.8, alpha * 1.3)

        return max(0.1, min(0.9, alpha))

    def smooth(
        self,
        new_weights: pd.Series,
        previous_weights: pd.Series,
        current_volatility: float | None = None,
        average_volatility: float | None = None,
        is_crisis: bool = False,
    ) -> SmoothingResult:
        """適応的スムージングを適用

        Args:
            new_weights: 新しい目標重み
            previous_weights: 前期の重み
            current_volatility: 現在のボラティリティ
            average_volatility: 平均ボラティリティ
            is_crisis: 危機モードか

        Returns:
            SmoothingResult
        """
        adaptive_alpha = self.compute_adaptive_alpha(
            current_volatility, average_volatility, is_crisis
        )

        config = SmootherConfig(alpha=adaptive_alpha)
        smoother = WeightSmoother(config)

        result = smoother.smooth(new_weights, previous_weights)
        result.metadata["adaptive_alpha"] = adaptive_alpha
        result.metadata["base_alpha"] = self.base_alpha

        return result


def create_smoother_from_settings() -> WeightSmoother:
    """グローバル設定からSmootherを生成

    Returns:
        設定済みのWeightSmoother
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        config = SmootherConfig(
            alpha=settings.asset_allocation.smooth_alpha,
            max_single_change=settings.asset_allocation.delta_max,
        )
        return WeightSmoother(config)
    except ImportError:
        logger.warning("Settings not available, using default SmootherConfig")
        return WeightSmoother()
