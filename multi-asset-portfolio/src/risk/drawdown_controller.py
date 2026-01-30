"""
Drawdown Controller - バックテスト統合向けドローダウン制御

軽量でバックテストエンジンに直接統合可能なドローダウン制御モジュール。
既存のDrawdownProtectorよりシンプルなAPIで、リアルタイム制御に特化。

Key Features:
- トレーリングストップ（15%デフォルト）
- 段階的ポジション削減
- 回復検出と段階的復帰
- バックテストエンジン統合用API

Based on IMP-004: Drawdown control algorithm implementation.

Expected Effect:
- MDD improvement: -5% to -8%
- Sharpe improvement: +0.05 to +0.10

Usage:
    controller = DrawdownController()

    # バックテストループ内
    multiplier = controller.calculate_position_multiplier(current_drawdown=-0.12)
    adjusted_weights = {k: v * multiplier for k, v in weights.items()}

    # トレーリングストップ判定
    if controller.should_exit(current_drawdown=-0.16):
        # ポジション大幅削減または全決済
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DrawdownControllerConfig:
    """ドローダウン制御設定

    Attributes:
        trailing_stop_pct: トレーリングストップ閾値（デフォルト15%）
        position_reduction_thresholds: (DD閾値, 削減率)のリスト
        recovery_threshold: 回復判定閾値（DDがこの割合まで回復で復帰開始）
        recovery_rate: 回復時のポジション復帰率（1日あたり）
        min_position_multiplier: 最小ポジション乗数（完全撤退防止）
        cooldown_days: ストップ発動後のクールダウン期間
    """
    trailing_stop_pct: float = 0.15
    position_reduction_thresholds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.10, 0.20),  # 10%DD時、20%ポジション削減
            (0.15, 0.40),  # 15%DD時、40%ポジション削減
            (0.20, 0.60),  # 20%DD時、60%ポジション削減
            (0.25, 0.80),  # 25%DD時、80%ポジション削減
        ]
    )
    recovery_threshold: float = 0.05  # DDが5%以下で回復開始
    recovery_rate: float = 0.10       # 1日あたり10%復帰
    min_position_multiplier: float = 0.20  # 最小20%は維持
    cooldown_days: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DrawdownControllerConfig":
        """YAMLファイルから設定を読み込み"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        dd_config = data.get("drawdown_control", {})
        trailing = dd_config.get("trailing_stop", {})
        recovery = dd_config.get("recovery", {})

        return cls(
            trailing_stop_pct=trailing.get("threshold", -0.15) * -1 if trailing.get("threshold", -0.15) < 0 else trailing.get("threshold", 0.15),
            recovery_threshold=recovery.get("consecutive_positive_days", 5) * 0.01,  # Approximate
            recovery_rate=recovery.get("gradual_return_rate", 0.10),
        )


@dataclass
class DrawdownState:
    """ドローダウン状態

    Attributes:
        high_water_mark: 過去最高値
        current_value: 現在値
        current_drawdown: 現在のドローダウン（負の値）
        max_drawdown: 最大ドローダウン
        position_multiplier: 現在のポジション乗数
        is_stopped: トレーリングストップ発動中
        stop_triggered_date: ストップ発動日
        days_since_stop: ストップ発動からの日数
        recovery_mode: 回復モード中
        consecutive_positive_days: 連続プラス日数
    """
    high_water_mark: float = 0.0
    current_value: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    position_multiplier: float = 1.0
    is_stopped: bool = False
    stop_triggered_date: datetime | None = None
    days_since_stop: int = 0
    recovery_mode: bool = False
    consecutive_positive_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "high_water_mark": self.high_water_mark,
            "current_value": self.current_value,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "position_multiplier": self.position_multiplier,
            "is_stopped": self.is_stopped,
            "days_since_stop": self.days_since_stop,
            "recovery_mode": self.recovery_mode,
            "consecutive_positive_days": self.consecutive_positive_days,
        }


@dataclass
class ControllerResult:
    """制御結果

    Attributes:
        position_multiplier: ポジション乗数（0-1）
        should_exit: 全ポジション撤退すべきか
        action: 実行されたアクション
        state: 現在の状態
    """
    position_multiplier: float
    should_exit: bool = False
    action: str = ""
    state: DrawdownState | None = None


class DrawdownController:
    """
    バックテスト統合向けドローダウン制御クラス

    トレーリングストップと段階的ポジション削減を組み合わせて、
    大幅なドローダウンを防止する。

    Usage:
        controller = DrawdownController()
        controller.initialize(initial_value=100000)

        # 毎日の更新
        for date, portfolio_value in daily_values:
            result = controller.update(portfolio_value)

            if result.should_exit:
                # 緊急撤退
                weights = {"CASH": 1.0}
            else:
                # ポジション調整
                adjusted_weights = {
                    k: v * result.position_multiplier
                    for k, v in base_weights.items()
                }
    """

    def __init__(
        self,
        config: DrawdownControllerConfig | None = None,
        initial_value: float | None = None,
    ) -> None:
        """初期化

        Args:
            config: 制御設定
            initial_value: 初期ポートフォリオ価値
        """
        self.config = config or DrawdownControllerConfig()
        self._state = DrawdownState()

        if initial_value is not None:
            self.initialize(initial_value)

        self._history: list[DrawdownState] = []
        self._previous_value: float = 0.0

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DrawdownController":
        """設定ファイルからインスタンス作成"""
        config = DrawdownControllerConfig.from_yaml(config_path)
        return cls(config=config)

    def initialize(self, initial_value: float) -> None:
        """初期化

        Args:
            initial_value: 初期ポートフォリオ価値
        """
        self._state = DrawdownState(
            high_water_mark=initial_value,
            current_value=initial_value,
            position_multiplier=1.0,
        )
        self._previous_value = initial_value
        self._history = []
        logger.info(f"DrawdownController initialized with HWM: {initial_value:,.2f}")

    def update(self, portfolio_value: float, date: datetime | None = None) -> ControllerResult:
        """ポートフォリオ価値を更新し、制御結果を返す

        Args:
            portfolio_value: 現在のポートフォリオ価値
            date: 現在日（オプション）

        Returns:
            ControllerResult with position multiplier and actions
        """
        # 日次リターン計算
        daily_return = 0.0
        if self._previous_value > 0:
            daily_return = (portfolio_value - self._previous_value) / self._previous_value

        # HWM更新
        new_hwm = max(self._state.high_water_mark, portfolio_value)

        # ドローダウン計算
        if new_hwm > 0:
            current_dd = (new_hwm - portfolio_value) / new_hwm
        else:
            current_dd = 0.0

        # 最大ドローダウン更新
        max_dd = max(self._state.max_drawdown, current_dd)

        # 連続プラス日数
        if daily_return > 0:
            consecutive_positive = self._state.consecutive_positive_days + 1
        else:
            consecutive_positive = 0

        # ストップ日数
        days_since_stop = self._state.days_since_stop
        if self._state.is_stopped:
            days_since_stop += 1

        # 制御ロジック
        position_multiplier, should_exit, action, is_stopped, recovery_mode = \
            self._calculate_control(current_dd, consecutive_positive, days_since_stop)

        # ストップ発動日更新
        stop_triggered_date = self._state.stop_triggered_date
        if is_stopped and not self._state.is_stopped:
            stop_triggered_date = date or datetime.now()
        elif not is_stopped:
            stop_triggered_date = None
            days_since_stop = 0

        # 状態更新
        self._state = DrawdownState(
            high_water_mark=new_hwm,
            current_value=portfolio_value,
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            position_multiplier=position_multiplier,
            is_stopped=is_stopped,
            stop_triggered_date=stop_triggered_date,
            days_since_stop=days_since_stop,
            recovery_mode=recovery_mode,
            consecutive_positive_days=consecutive_positive,
        )

        # 履歴追加
        self._history.append(self._state)
        self._previous_value = portfolio_value

        return ControllerResult(
            position_multiplier=position_multiplier,
            should_exit=should_exit,
            action=action,
            state=self._state,
        )

    def _calculate_control(
        self,
        current_dd: float,
        consecutive_positive: int,
        days_since_stop: int,
    ) -> tuple[float, bool, str, bool, bool]:
        """制御パラメータを計算

        Returns:
            (position_multiplier, should_exit, action, is_stopped, recovery_mode)
        """
        action = ""
        is_stopped = self._state.is_stopped
        recovery_mode = self._state.recovery_mode

        # トレーリングストップ判定
        if current_dd >= self.config.trailing_stop_pct:
            if not is_stopped:
                is_stopped = True
                action = f"Trailing stop triggered at DD {current_dd:.1%}"
                logger.warning(action)

        # 回復モード判定
        if is_stopped:
            if days_since_stop >= self.config.cooldown_days:
                if current_dd <= self.config.recovery_threshold:
                    recovery_mode = True
                    action = f"Recovery mode entered at DD {current_dd:.1%}"
                    logger.info(action)

        # 完全回復判定
        if recovery_mode and current_dd <= 0.02:  # 2%以下で完全回復
            is_stopped = False
            recovery_mode = False
            action = "Full recovery - normal operation resumed"
            logger.info(action)

        # ポジション乗数計算
        if is_stopped and not recovery_mode:
            # ストップ中：最小ポジション
            position_multiplier = self.config.min_position_multiplier
        elif recovery_mode:
            # 回復中：段階的復帰
            base_multiplier = self._state.position_multiplier
            position_multiplier = min(
                1.0,
                base_multiplier + self.config.recovery_rate
            )
        else:
            # 通常モード：段階的削減
            position_multiplier = self._calculate_position_reduction(current_dd)

        # 完全撤退判定（極端なDD時のみ）
        should_exit = current_dd >= 0.30  # 30%DD超で撤退

        return position_multiplier, should_exit, action, is_stopped, recovery_mode

    def _calculate_position_reduction(self, current_dd: float) -> float:
        """段階的ポジション削減を計算

        Args:
            current_dd: 現在のドローダウン

        Returns:
            ポジション乗数（0-1）
        """
        reduction = 0.0

        for dd_threshold, reduction_pct in self.config.position_reduction_thresholds:
            if current_dd >= dd_threshold:
                reduction = reduction_pct

        multiplier = max(
            self.config.min_position_multiplier,
            1.0 - reduction
        )

        return multiplier

    def calculate_position_multiplier(self, current_drawdown: float) -> float:
        """現在のドローダウンからポジション乗数を計算（便利関数）

        Args:
            current_drawdown: 現在のドローダウン（正の値、例: 0.15 for 15%DD）

        Returns:
            ポジション乗数（0-1）
        """
        # 負の値が渡された場合は絶対値に変換
        dd = abs(current_drawdown)
        return self._calculate_position_reduction(dd)

    def should_exit(self, current_drawdown: float) -> bool:
        """撤退すべきかどうかを判定（便利関数）

        Args:
            current_drawdown: 現在のドローダウン

        Returns:
            True if should exit all positions
        """
        dd = abs(current_drawdown)
        return dd >= self.config.trailing_stop_pct

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
            high_water_mark=new_hwm,
            current_value=new_hwm,
            position_multiplier=1.0,
        )
        self._history = []
        self._previous_value = new_hwm
        logger.info(f"DrawdownController reset with new HWM: {new_hwm:,.2f}")

    def get_summary(self) -> dict[str, Any]:
        """サマリーを取得"""
        return {
            "current_drawdown": f"{self._state.current_drawdown:.2%}",
            "max_drawdown": f"{self._state.max_drawdown:.2%}",
            "position_multiplier": f"{self._state.position_multiplier:.2%}",
            "is_stopped": self._state.is_stopped,
            "recovery_mode": self._state.recovery_mode,
            "high_water_mark": f"{self._state.high_water_mark:,.2f}",
            "current_value": f"{self._state.current_value:,.2f}",
        }


# =============================================================================
# 便利関数
# =============================================================================

def create_drawdown_controller(
    trailing_stop_pct: float = 0.15,
    initial_value: float | None = None,
    config_path: str | Path | None = None,
) -> DrawdownController:
    """DrawdownControllerを作成（ファクトリ関数）

    Args:
        trailing_stop_pct: トレーリングストップ閾値
        initial_value: 初期ポートフォリオ価値
        config_path: 設定ファイルパス

    Returns:
        DrawdownController
    """
    if config_path:
        return DrawdownController.from_config(config_path)

    config = DrawdownControllerConfig(trailing_stop_pct=trailing_stop_pct)
    return DrawdownController(config, initial_value)


def quick_position_multiplier(
    current_drawdown: float,
    thresholds: list[tuple[float, float]] | None = None,
) -> float:
    """ドローダウンからポジション乗数を計算（便利関数）

    Args:
        current_drawdown: 現在のドローダウン（正の値）
        thresholds: (DD閾値, 削減率)のリスト

    Returns:
        ポジション乗数（0-1）
    """
    if thresholds is None:
        thresholds = [
            (0.10, 0.20),
            (0.15, 0.40),
            (0.20, 0.60),
            (0.25, 0.80),
        ]

    dd = abs(current_drawdown)
    reduction = 0.0

    for dd_threshold, reduction_pct in thresholds:
        if dd >= dd_threshold:
            reduction = reduction_pct

    return max(0.20, 1.0 - reduction)


def adjust_weights_for_drawdown(
    weights: dict[str, float],
    current_drawdown: float,
    cash_symbol: str = "CASH",
) -> dict[str, float]:
    """ドローダウンに基づいて重みを調整（便利関数）

    Args:
        weights: 元の重み
        current_drawdown: 現在のドローダウン
        cash_symbol: キャッシュシンボル

    Returns:
        調整後の重み
    """
    multiplier = quick_position_multiplier(current_drawdown)

    if multiplier >= 1.0:
        return weights.copy()

    adjusted = {}
    cash_added = 0.0

    for symbol, weight in weights.items():
        if symbol == cash_symbol:
            adjusted[symbol] = weight
        else:
            adjusted[symbol] = weight * multiplier
            cash_added += weight * (1 - multiplier)

    # キャッシュに追加
    adjusted[cash_symbol] = adjusted.get(cash_symbol, 0.0) + cash_added

    return adjusted
