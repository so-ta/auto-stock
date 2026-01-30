"""
Constraints Module - 制約処理

ポートフォリオ重みに各種制約を適用するモジュール。

制約種別:
1. 総和制約: Σw = 1
2. 上限制約: w_a <= w_asset_max
3. 下限制約: w_a >= 0 (または w_asset_min)
4. 変更量制約: |w(t) - w(t-1)| <= delta_max
5. グループ制約: Σw_group <= group_max

設計根拠:
- 要求.md §8.4: 制約（必須）
- ターンオーバー抑制と集中リスク回避

使用方法:
    processor = ConstraintProcessor(config)
    constrained_weights = processor.apply(raw_weights, previous_weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConstraintViolationType(str, Enum):
    """制約違反の種類"""

    SUM_CONSTRAINT = "sum_constraint"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    TURNOVER = "turnover"
    GROUP_LIMIT = "group_limit"


@dataclass(frozen=True)
class ConstraintConfig:
    """制約設定

    Attributes:
        w_max: 単一アセットの最大重み
        w_min: 単一アセットの最小重み
        delta_max: 1リバランスあたりの最大変更量
        total_sum: 目標総和（通常1.0）
        sum_tolerance: 総和の許容誤差
        allow_short: ショートを許可するか
        max_cash: 最大キャッシュ比率
        group_limits: グループ別上限 {"group_name": max_weight}
    """

    w_max: float = 0.2
    w_min: float = 0.0
    delta_max: float = 0.05
    total_sum: float = 1.0
    sum_tolerance: float = 1e-6
    allow_short: bool = False
    max_cash: float = 0.3
    group_limits: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """バリデーション"""
        if not 0 < self.w_max <= 1:
            raise ValueError("w_max must be in (0, 1]")
        if not 0 <= self.w_min < self.w_max:
            raise ValueError("w_min must be in [0, w_max)")
        if self.delta_max <= 0:
            raise ValueError("delta_max must be > 0")
        if self.total_sum <= 0:
            raise ValueError("total_sum must be > 0")


@dataclass
class ConstraintViolation:
    """制約違反の詳細

    Attributes:
        violation_type: 違反の種類
        asset: 対象アセット（該当する場合）
        original_value: 元の値
        constrained_value: 制約適用後の値
        limit: 制約値
        message: 詳細メッセージ
    """

    violation_type: ConstraintViolationType
    asset: str | None = None
    original_value: float = 0.0
    constrained_value: float = 0.0
    limit: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "violation_type": self.violation_type.value,
            "asset": self.asset,
            "original_value": self.original_value,
            "constrained_value": self.constrained_value,
            "limit": self.limit,
            "message": self.message,
        }


@dataclass
class ConstraintResult:
    """制約適用結果

    Attributes:
        weights: 制約適用後の重み（Series）
        violations: 制約違反のリスト
        turnover: 実際のターンオーバー
        cash_weight: キャッシュ重み
        is_modified: 重みが変更されたか
        metadata: 追加メタデータ
    """

    weights: pd.Series
    violations: list[ConstraintViolation] = field(default_factory=list)
    turnover: float = 0.0
    cash_weight: float = 0.0
    is_modified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な結果かどうか"""
        if self.weights.empty:
            return False
        total = self.weights.sum() + self.cash_weight
        return np.isclose(total, 1.0, atol=1e-4)

    @property
    def violation_count(self) -> int:
        """違反数"""
        return len(self.violations)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "turnover": self.turnover,
            "cash_weight": self.cash_weight,
            "is_modified": self.is_modified,
            "is_valid": self.is_valid,
            "violation_count": self.violation_count,
            "metadata": self.metadata,
        }


class ConstraintProcessor:
    """制約処理クラス

    ポートフォリオ重みに各種制約を適用する。

    Usage:
        config = ConstraintConfig(w_max=0.2, delta_max=0.05)
        processor = ConstraintProcessor(config)

        # raw_weights: 制約前の重み
        # previous_weights: 前期の重み（ターンオーバー制約用）
        result = processor.apply(raw_weights, previous_weights)

        print(result.weights)
        print(result.violations)
    """

    def __init__(
        self,
        config: ConstraintConfig | None = None,
        asset_groups: dict[str, str] | None = None,
    ) -> None:
        """初期化

        Args:
            config: 制約設定。Noneの場合はデフォルト値を使用
            asset_groups: アセット -> グループのマッピング（グループ制約用）
        """
        self.config = config or ConstraintConfig()
        self.asset_groups = asset_groups or {}

    def apply(
        self,
        weights: pd.Series,
        previous_weights: pd.Series | None = None,
    ) -> ConstraintResult:
        """制約を適用

        Args:
            weights: 制約前の重み
            previous_weights: 前期の重み（Noneの場合はターンオーバー制約なし）

        Returns:
            ConstraintResult: 制約適用結果
        """
        if weights.empty:
            return ConstraintResult(
                weights=weights,
                metadata={"error": "Empty weights"},
            )

        original_weights = weights.copy()
        violations: list[ConstraintViolation] = []
        is_modified = False

        # Step 1: 下限制約
        weights, lower_violations = self._apply_lower_bound(weights)
        violations.extend(lower_violations)
        if lower_violations:
            is_modified = True

        # Step 2: 上限制約
        weights, upper_violations = self._apply_upper_bound(weights)
        violations.extend(upper_violations)
        if upper_violations:
            is_modified = True

        # Step 3: グループ制約
        if self.config.group_limits and self.asset_groups:
            weights, group_violations = self._apply_group_limits(weights)
            violations.extend(group_violations)
            if group_violations:
                is_modified = True

        # Step 4: ターンオーバー制約
        if previous_weights is not None:
            weights, turnover_violations = self._apply_turnover_constraint(
                weights, previous_weights
            )
            violations.extend(turnover_violations)
            if turnover_violations:
                is_modified = True

        # Step 5: 総和制約（正規化）
        weights, sum_violation = self._apply_sum_constraint(weights)
        if sum_violation:
            violations.append(sum_violation)
            is_modified = True

        # ターンオーバー計算
        turnover = 0.0
        if previous_weights is not None:
            aligned_prev = previous_weights.reindex(weights.index, fill_value=0.0)
            turnover = float((weights - aligned_prev).abs().sum()) / 2

        # キャッシュ計算
        cash_weight = max(0.0, self.config.total_sum - weights.sum())

        logger.info(
            "Constraints applied: %d violations, turnover=%.4f, modified=%s",
            len(violations),
            turnover,
            is_modified,
        )

        return ConstraintResult(
            weights=weights,
            violations=violations,
            turnover=turnover,
            cash_weight=cash_weight,
            is_modified=is_modified,
            metadata={
                "original_sum": float(original_weights.sum()),
                "final_sum": float(weights.sum()),
            },
        )

    def _apply_lower_bound(
        self, weights: pd.Series
    ) -> tuple[pd.Series, list[ConstraintViolation]]:
        """下限制約を適用

        Args:
            weights: 重み

        Returns:
            (制約後重み, 違反リスト)
        """
        violations = []
        min_val = 0.0 if not self.config.allow_short else -self.config.w_max

        for asset, w in weights.items():
            if w < min_val:
                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.LOWER_BOUND,
                        asset=str(asset),
                        original_value=w,
                        constrained_value=max(w, self.config.w_min),
                        limit=min_val,
                        message=f"{asset}: {w:.4f} < {min_val:.4f}",
                    )
                )

        constrained = weights.clip(lower=max(min_val, self.config.w_min))
        return constrained, violations

    def _apply_upper_bound(
        self, weights: pd.Series
    ) -> tuple[pd.Series, list[ConstraintViolation]]:
        """上限制約を適用

        Args:
            weights: 重み

        Returns:
            (制約後重み, 違反リスト)
        """
        violations = []

        for asset, w in weights.items():
            if w > self.config.w_max:
                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.UPPER_BOUND,
                        asset=str(asset),
                        original_value=w,
                        constrained_value=self.config.w_max,
                        limit=self.config.w_max,
                        message=f"{asset}: {w:.4f} > {self.config.w_max:.4f}",
                    )
                )

        constrained = weights.clip(upper=self.config.w_max)
        return constrained, violations

    def _apply_turnover_constraint(
        self,
        weights: pd.Series,
        previous_weights: pd.Series,
    ) -> tuple[pd.Series, list[ConstraintViolation]]:
        """ターンオーバー制約を適用

        Args:
            weights: 新しい重み
            previous_weights: 前期の重み

        Returns:
            (制約後重み, 違反リスト)
        """
        violations = []
        constrained = weights.copy()

        # インデックスを揃える
        aligned_prev = previous_weights.reindex(weights.index, fill_value=0.0)

        for asset in weights.index:
            w_new = weights[asset]
            w_prev = aligned_prev[asset]
            delta = w_new - w_prev

            if abs(delta) > self.config.delta_max:
                # 変更量を制限
                if delta > 0:
                    constrained_val = w_prev + self.config.delta_max
                else:
                    constrained_val = w_prev - self.config.delta_max

                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.TURNOVER,
                        asset=str(asset),
                        original_value=w_new,
                        constrained_value=constrained_val,
                        limit=self.config.delta_max,
                        message=f"{asset}: delta={delta:.4f} > {self.config.delta_max:.4f}",
                    )
                )
                constrained[asset] = constrained_val

        return constrained, violations

    def _apply_group_limits(
        self, weights: pd.Series
    ) -> tuple[pd.Series, list[ConstraintViolation]]:
        """グループ制約を適用

        Args:
            weights: 重み

        Returns:
            (制約後重み, 違反リスト)
        """
        violations = []
        constrained = weights.copy()

        # グループごとに集計
        group_weights: dict[str, float] = {}
        for asset, w in weights.items():
            group = self.asset_groups.get(str(asset), "default")
            group_weights[group] = group_weights.get(group, 0.0) + w

        # グループ上限をチェック
        for group, total_w in group_weights.items():
            limit = self.config.group_limits.get(group, 1.0)
            if total_w > limit:
                # 比例縮小
                scale = limit / total_w
                for asset, w in constrained.items():
                    if self.asset_groups.get(str(asset), "default") == group:
                        constrained[asset] = w * scale

                violations.append(
                    ConstraintViolation(
                        violation_type=ConstraintViolationType.GROUP_LIMIT,
                        asset=group,
                        original_value=total_w,
                        constrained_value=limit,
                        limit=limit,
                        message=f"Group {group}: {total_w:.4f} > {limit:.4f}",
                    )
                )

        return constrained, violations

    def _apply_sum_constraint(
        self, weights: pd.Series
    ) -> tuple[pd.Series, ConstraintViolation | None]:
        """総和制約を適用（正規化）

        Args:
            weights: 重み

        Returns:
            (正規化後重み, 違反（あれば）)
        """
        current_sum = weights.sum()
        target = self.config.total_sum

        if abs(current_sum - target) <= self.config.sum_tolerance:
            return weights, None

        if current_sum == 0:
            # 全てゼロの場合は均等配分
            n = len(weights)
            return pd.Series(target / n, index=weights.index), ConstraintViolation(
                violation_type=ConstraintViolationType.SUM_CONSTRAINT,
                original_value=0.0,
                constrained_value=target,
                limit=target,
                message="All weights were zero, applied equal weighting",
            )

        # 比例スケーリング
        scale = target / current_sum
        normalized = weights * scale

        violation = ConstraintViolation(
            violation_type=ConstraintViolationType.SUM_CONSTRAINT,
            original_value=current_sum,
            constrained_value=target,
            limit=target,
            message=f"Sum {current_sum:.4f} -> {target:.4f} (scale={scale:.4f})",
        )

        return normalized, violation


def create_processor_from_settings() -> ConstraintProcessor:
    """グローバル設定からConstraintProcessorを生成

    Returns:
        設定済みのConstraintProcessor
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        config = ConstraintConfig(
            w_max=settings.asset_allocation.w_asset_max,
            w_min=settings.asset_allocation.w_asset_min,
            delta_max=settings.asset_allocation.delta_max,
            allow_short=settings.asset_allocation.allow_short,
            max_cash=settings.asset_allocation.max_cash_ratio,
        )
        return ConstraintProcessor(config)
    except ImportError:
        logger.warning("Settings not available, using default ConstraintConfig")
        return ConstraintProcessor()
