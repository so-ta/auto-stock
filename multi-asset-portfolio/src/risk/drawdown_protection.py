"""
Drawdown Protection Module - ドローダウン・プロテクション

段階的なドローダウン・プロテクションを提供する。
ポートフォリオ価値の下落に応じてリスクを自動的に削減し、
大幅な損失を防止する。

主要コンポーネント:
- DrawdownProtector: 段階的ドローダウン・プロテクション
- ProtectionLevel: プロテクションレベルデータクラス
- DrawdownState: ドローダウン状態追跡

設計根拠:
- 段階的削減で過剰反応を防止
- HWM（High Water Mark）追跡で正確なDD計算
- 回復閾値で早期解除を防止

使用例:
    from src.risk.drawdown_protection import (
        DrawdownProtector,
        DrawdownProtectorConfig,
    )

    # プロテクター初期化
    config = DrawdownProtectorConfig(
        dd_levels=[0.05, 0.10, 0.15, 0.20],
        risk_reductions=[0.9, 0.7, 0.5, 0.3],
    )
    protector = DrawdownProtector(config)

    # ポートフォリオ更新
    protector.update(portfolio_value=98000)

    # リスク乗数取得
    multiplier = protector.get_risk_multiplier()

    # 重み調整
    adjusted = protector.adjust_weights(base_weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enum定義
# =============================================================================

class ProtectionStatus(str, Enum):
    """プロテクション状態"""
    INACTIVE = "inactive"        # プロテクション無効
    ACTIVE = "active"            # プロテクション有効
    RECOVERING = "recovering"    # 回復中
    EMERGENCY = "emergency"      # 緊急モード（最大DD超過）


class RecoveryMode(str, Enum):
    """回復モード"""
    GRADUAL = "gradual"          # 段階的回復
    THRESHOLD = "threshold"      # 閾値到達で解除
    IMMEDIATE = "immediate"      # 即座に解除


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class DrawdownProtectorConfig:
    """DrawdownProtector設定

    Attributes:
        dd_levels: ドローダウン閾値リスト（昇順）
        risk_reductions: 各レベルでのリスク乗数
        recovery_threshold: DD回復率（この割合回復で解除）
        recovery_mode: 回復モード
        min_protection_days: 最小プロテクション日数
        emergency_dd_level: 緊急モード閾値
        emergency_cash_ratio: 緊急時のキャッシュ比率
        update_frequency: 更新頻度（daily, intraday）
    """
    dd_levels: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    risk_reductions: list[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.3])
    recovery_threshold: float = 0.5
    recovery_mode: RecoveryMode = RecoveryMode.THRESHOLD
    min_protection_days: int = 3
    emergency_dd_level: float = 0.25
    emergency_cash_ratio: float = 0.8
    update_frequency: str = "daily"

    def __post_init__(self) -> None:
        """バリデーション"""
        if len(self.dd_levels) != len(self.risk_reductions):
            raise ValueError("dd_levels and risk_reductions must have same length")
        if not all(0 < x < 1 for x in self.dd_levels):
            raise ValueError("dd_levels must be in (0, 1)")
        if not all(0 < x <= 1 for x in self.risk_reductions):
            raise ValueError("risk_reductions must be in (0, 1]")
        if not 0 < self.recovery_threshold <= 1:
            raise ValueError("recovery_threshold must be in (0, 1]")
        # 昇順チェック
        if self.dd_levels != sorted(self.dd_levels):
            raise ValueError("dd_levels must be sorted ascending")
        # 降順チェック（リスク乗数）
        if self.risk_reductions != sorted(self.risk_reductions, reverse=True):
            raise ValueError("risk_reductions must be sorted descending")


@dataclass
class ProtectionLevel:
    """プロテクションレベル

    Attributes:
        level: レベル番号（0=なし、1以上=有効）
        dd_threshold: このレベルのDD閾値
        risk_multiplier: リスク乗数
        description: 説明
    """
    level: int
    dd_threshold: float
    risk_multiplier: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "level": self.level,
            "dd_threshold": self.dd_threshold,
            "risk_multiplier": self.risk_multiplier,
            "description": self.description,
        }


@dataclass
class DrawdownState:
    """ドローダウン状態

    Attributes:
        hwm: High Water Mark
        current_value: 現在のポートフォリオ価値
        drawdown: 現在のドローダウン（0-1）
        max_drawdown: 最大ドローダウン
        protection_level: 現在のプロテクションレベル
        status: プロテクション状態
        activation_date: プロテクション発動日
        days_in_protection: プロテクション日数
        recovery_target: 回復目標値
        last_update: 最終更新日時
    """
    hwm: float
    current_value: float
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    protection_level: int = 0
    status: ProtectionStatus = ProtectionStatus.INACTIVE
    activation_date: datetime | None = None
    days_in_protection: int = 0
    recovery_target: float | None = None
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "hwm": self.hwm,
            "current_value": self.current_value,
            "drawdown": self.drawdown,
            "max_drawdown": self.max_drawdown,
            "protection_level": self.protection_level,
            "status": self.status.value,
            "activation_date": self.activation_date.isoformat() if self.activation_date else None,
            "days_in_protection": self.days_in_protection,
            "recovery_target": self.recovery_target,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class ProtectionResult:
    """プロテクション適用結果

    Attributes:
        original_weights: 元の重み
        adjusted_weights: 調整後の重み
        risk_multiplier: 適用されたリスク乗数
        cash_added: 追加されたキャッシュ比率
        state: ドローダウン状態
        level_info: プロテクションレベル情報
        action_taken: 実行されたアクション
    """
    original_weights: pd.Series
    adjusted_weights: pd.Series
    risk_multiplier: float
    cash_added: float = 0.0
    state: DrawdownState | None = None
    level_info: ProtectionLevel | None = None
    action_taken: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "original_weights": self.original_weights.to_dict(),
            "adjusted_weights": self.adjusted_weights.to_dict(),
            "risk_multiplier": self.risk_multiplier,
            "cash_added": self.cash_added,
            "state": self.state.to_dict() if self.state else None,
            "level_info": self.level_info.to_dict() if self.level_info else None,
            "action_taken": self.action_taken,
        }


# =============================================================================
# メインクラス
# =============================================================================

class DrawdownProtector:
    """ドローダウン・プロテクタークラス

    段階的なドローダウン・プロテクションを提供する。

    Usage:
        config = DrawdownProtectorConfig(
            dd_levels=[0.05, 0.10, 0.15, 0.20],
            risk_reductions=[0.9, 0.7, 0.5, 0.3],
        )
        protector = DrawdownProtector(config)

        # 初期化
        protector.initialize(initial_value=100000)

        # 更新
        protector.update(portfolio_value=95000)

        # リスク乗数取得
        multiplier = protector.get_risk_multiplier()
        # DD 5%の場合: multiplier = 0.9

        # 重み調整
        adjusted = protector.adjust_weights(base_weights)
    """

    def __init__(
        self,
        config: DrawdownProtectorConfig | None = None,
        initial_value: float | None = None,
    ) -> None:
        """初期化

        Args:
            config: 設定
            initial_value: 初期ポートフォリオ価値
        """
        self.config = config or DrawdownProtectorConfig()
        self._build_protection_levels()

        # 状態初期化
        if initial_value is not None:
            self._state = DrawdownState(
                hwm=initial_value,
                current_value=initial_value,
            )
        else:
            self._state = DrawdownState(hwm=0.0, current_value=0.0)

        self._history: list[DrawdownState] = []

    def _build_protection_levels(self) -> None:
        """プロテクションレベルを構築"""
        self._levels: list[ProtectionLevel] = []

        # レベル0: プロテクションなし
        self._levels.append(ProtectionLevel(
            level=0,
            dd_threshold=0.0,
            risk_multiplier=1.0,
            description="No protection",
        ))

        # 設定されたレベル
        for i, (dd, risk) in enumerate(
            zip(self.config.dd_levels, self.config.risk_reductions), 1
        ):
            self._levels.append(ProtectionLevel(
                level=i,
                dd_threshold=dd,
                risk_multiplier=risk,
                description=f"Level {i}: DD {dd*100:.0f}%, Risk {risk*100:.0f}%",
            ))

        # 緊急レベル
        self._levels.append(ProtectionLevel(
            level=len(self.config.dd_levels) + 1,
            dd_threshold=self.config.emergency_dd_level,
            risk_multiplier=1.0 - self.config.emergency_cash_ratio,
            description=f"Emergency: DD {self.config.emergency_dd_level*100:.0f}%+",
        ))

    def initialize(self, initial_value: float) -> None:
        """初期化

        Args:
            initial_value: 初期ポートフォリオ価値
        """
        self._state = DrawdownState(
            hwm=initial_value,
            current_value=initial_value,
        )
        self._history = []
        logger.info("DrawdownProtector initialized with HWM: %.2f", initial_value)

    def update(self, portfolio_value: float) -> DrawdownState:
        """ポートフォリオ価値を更新

        Args:
            portfolio_value: 現在のポートフォリオ価値

        Returns:
            更新後の状態
        """
        now = datetime.now()
        prev_state = self._state

        # HWM更新
        new_hwm = max(self._state.hwm, portfolio_value)

        # ドローダウン計算
        if new_hwm > 0:
            drawdown = (new_hwm - portfolio_value) / new_hwm
        else:
            drawdown = 0.0

        # 最大ドローダウン更新
        max_dd = max(self._state.max_drawdown, drawdown)

        # プロテクションレベル判定
        new_level = self._determine_protection_level(drawdown)

        # 状態判定
        status = self._determine_status(drawdown, new_level, prev_state)

        # プロテクション日数
        if status in (ProtectionStatus.ACTIVE, ProtectionStatus.EMERGENCY):
            if prev_state.status == ProtectionStatus.INACTIVE:
                activation_date = now
                days_in_protection = 1
            else:
                activation_date = prev_state.activation_date
                days_in_protection = prev_state.days_in_protection + 1
        else:
            activation_date = None
            days_in_protection = 0

        # 回復目標計算
        if status == ProtectionStatus.ACTIVE:
            recovery_target = new_hwm * (1 - drawdown * self.config.recovery_threshold)
        else:
            recovery_target = None

        # 状態更新
        self._state = DrawdownState(
            hwm=new_hwm,
            current_value=portfolio_value,
            drawdown=drawdown,
            max_drawdown=max_dd,
            protection_level=new_level,
            status=status,
            activation_date=activation_date,
            days_in_protection=days_in_protection,
            recovery_target=recovery_target,
            last_update=now,
        )

        # 履歴追加
        self._history.append(self._state)

        # ログ出力
        if new_level != prev_state.protection_level:
            if new_level > prev_state.protection_level:
                logger.warning(
                    "Protection level increased: %d -> %d (DD: %.2f%%)",
                    prev_state.protection_level, new_level, drawdown * 100,
                )
            else:
                logger.info(
                    "Protection level decreased: %d -> %d (DD: %.2f%%)",
                    prev_state.protection_level, new_level, drawdown * 100,
                )

        return self._state

    def _determine_protection_level(self, drawdown: float) -> int:
        """プロテクションレベルを判定

        Args:
            drawdown: 現在のドローダウン

        Returns:
            プロテクションレベル
        """
        # 緊急レベルチェック
        if drawdown >= self.config.emergency_dd_level:
            return len(self.config.dd_levels) + 1

        # 通常レベルチェック（逆順でチェック）
        for i, dd_threshold in enumerate(reversed(self.config.dd_levels)):
            level = len(self.config.dd_levels) - i
            if drawdown >= dd_threshold:
                return level

        return 0

    def _determine_status(
        self,
        drawdown: float,
        new_level: int,
        prev_state: DrawdownState,
    ) -> ProtectionStatus:
        """プロテクション状態を判定

        Args:
            drawdown: 現在のドローダウン
            new_level: 新しいプロテクションレベル
            prev_state: 前の状態

        Returns:
            プロテクション状態
        """
        # 緊急レベル
        if new_level > len(self.config.dd_levels):
            return ProtectionStatus.EMERGENCY

        # プロテクションなし
        if new_level == 0:
            return ProtectionStatus.INACTIVE

        # 回復中チェック
        if prev_state.status == ProtectionStatus.ACTIVE:
            if prev_state.recovery_target is not None:
                if self._state.current_value >= prev_state.recovery_target:
                    # 回復閾値に到達
                    if prev_state.days_in_protection >= self.config.min_protection_days:
                        return ProtectionStatus.RECOVERING

        return ProtectionStatus.ACTIVE

    def get_risk_multiplier(self) -> float:
        """現在のリスク乗数を取得

        Returns:
            リスク乗数（0-1）
        """
        level = self._state.protection_level

        if level == 0:
            return 1.0

        if level > len(self.config.dd_levels):
            # 緊急レベル
            return 1.0 - self.config.emergency_cash_ratio

        return self.config.risk_reductions[level - 1]

    def get_protection_level(self) -> ProtectionLevel:
        """現在のプロテクションレベル情報を取得

        Returns:
            ProtectionLevel
        """
        level = self._state.protection_level
        if level < len(self._levels):
            return self._levels[level]
        return self._levels[-1]

    def adjust_weights(
        self,
        base_weights: pd.Series,
        cash_ticker: str = "CASH",
    ) -> ProtectionResult:
        """重みを調整

        Args:
            base_weights: 基本重み
            cash_ticker: キャッシュのティッカー

        Returns:
            ProtectionResult
        """
        multiplier = self.get_risk_multiplier()
        level_info = self.get_protection_level()

        if multiplier >= 1.0:
            # プロテクションなし
            return ProtectionResult(
                original_weights=base_weights,
                adjusted_weights=base_weights.copy(),
                risk_multiplier=multiplier,
                state=self._state,
                level_info=level_info,
                action_taken="No adjustment needed",
            )

        # 重み調整
        adjusted = base_weights.copy()

        # キャッシュ以外の重みを削減
        non_cash_mask = adjusted.index != cash_ticker
        cash_reduction = 0.0

        for ticker in adjusted.index:
            if ticker != cash_ticker:
                original = adjusted[ticker]
                adjusted[ticker] = original * multiplier
                cash_reduction += original - adjusted[ticker]

        # キャッシュに追加
        if cash_ticker in adjusted.index:
            adjusted[cash_ticker] += cash_reduction
        else:
            adjusted[cash_ticker] = cash_reduction

        # 正規化
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        action = (
            f"Applied protection level {level_info.level}: "
            f"multiplier={multiplier:.2f}, cash_added={cash_reduction:.2%}"
        )

        logger.info(action)

        return ProtectionResult(
            original_weights=base_weights,
            adjusted_weights=adjusted,
            risk_multiplier=multiplier,
            cash_added=cash_reduction,
            state=self._state,
            level_info=level_info,
            action_taken=action,
        )

    def get_state(self) -> DrawdownState:
        """現在の状態を取得"""
        return self._state

    def get_history(self) -> list[DrawdownState]:
        """履歴を取得"""
        return self._history.copy()

    def reset(self, new_hwm: float | None = None) -> None:
        """リセット

        Args:
            new_hwm: 新しいHWM（Noneの場合は現在値を使用）
        """
        if new_hwm is None:
            new_hwm = self._state.current_value

        self._state = DrawdownState(
            hwm=new_hwm,
            current_value=new_hwm,
        )
        self._history = []
        logger.info("DrawdownProtector reset with new HWM: %.2f", new_hwm)

    def is_protection_active(self) -> bool:
        """プロテクションがアクティブかどうか"""
        return self._state.status in (
            ProtectionStatus.ACTIVE,
            ProtectionStatus.EMERGENCY,
        )

    def get_summary(self) -> dict[str, Any]:
        """サマリーを取得"""
        return {
            "status": self._state.status.value,
            "protection_level": self._state.protection_level,
            "risk_multiplier": self.get_risk_multiplier(),
            "current_drawdown": self._state.drawdown,
            "max_drawdown": self._state.max_drawdown,
            "hwm": self._state.hwm,
            "current_value": self._state.current_value,
            "days_in_protection": self._state.days_in_protection,
        }


# =============================================================================
# 便利関数
# =============================================================================

def create_drawdown_protector(
    dd_levels: list[float] | None = None,
    risk_reductions: list[float] | None = None,
    recovery_threshold: float = 0.5,
    initial_value: float | None = None,
) -> DrawdownProtector:
    """DrawdownProtectorを作成（ファクトリ関数）

    Args:
        dd_levels: ドローダウン閾値
        risk_reductions: リスク乗数
        recovery_threshold: 回復閾値
        initial_value: 初期値

    Returns:
        DrawdownProtector
    """
    config = DrawdownProtectorConfig(
        dd_levels=dd_levels or [0.05, 0.10, 0.15, 0.20],
        risk_reductions=risk_reductions or [0.9, 0.7, 0.5, 0.3],
        recovery_threshold=recovery_threshold,
    )
    return DrawdownProtector(config, initial_value)


def quick_adjust_weights(
    weights: pd.Series,
    portfolio_value: float,
    hwm: float,
    dd_levels: list[float] | None = None,
    risk_reductions: list[float] | None = None,
) -> pd.Series:
    """重みを簡易調整（便利関数）

    Args:
        weights: 基本重み
        portfolio_value: 現在のポートフォリオ価値
        hwm: High Water Mark
        dd_levels: ドローダウン閾値
        risk_reductions: リスク乗数

    Returns:
        調整後の重み
    """
    protector = create_drawdown_protector(
        dd_levels=dd_levels,
        risk_reductions=risk_reductions,
        initial_value=hwm,
    )
    protector.update(portfolio_value)
    result = protector.adjust_weights(weights)
    return result.adjusted_weights


def calculate_protection_multiplier(
    drawdown: float,
    dd_levels: list[float] | None = None,
    risk_reductions: list[float] | None = None,
) -> float:
    """ドローダウンからリスク乗数を計算（便利関数）

    Args:
        drawdown: ドローダウン（0-1）
        dd_levels: ドローダウン閾値
        risk_reductions: リスク乗数

    Returns:
        リスク乗数
    """
    dd_levels = dd_levels or [0.05, 0.10, 0.15, 0.20]
    risk_reductions = risk_reductions or [0.9, 0.7, 0.5, 0.3]

    for dd, risk in zip(reversed(dd_levels), reversed(risk_reductions)):
        if drawdown >= dd:
            return risk

    return 1.0
