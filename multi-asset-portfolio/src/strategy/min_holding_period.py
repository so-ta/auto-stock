"""
Minimum Holding Period Filter - 最低保有期間フィルター

過剰取引を防止するための最低保有期間フィルタ。
一度ポジションを取ったら最低N日は保持する。

主要コンポーネント:
- MinHoldingPeriodFilter: 最低保有期間フィルター
- HoldingInfo: 保有情報データクラス
- TradeDecision: 取引判定結果

設計根拠:
- 過剰取引（オーバートレーディング）の防止
- トランザクションコストの削減
- シグナルノイズによるホイップソー回避
- ただし、大きなシグナル反転時は強制エグジット可能

使用例:
    from src.strategy.min_holding_period import (
        MinHoldingPeriodFilter,
        apply_min_holding,
    )

    # フィルター初期化
    filter = MinHoldingPeriodFilter(
        min_periods=5,
        force_exit_on_reversal=True,
        reversal_threshold=-0.5,
    )

    # 取引判定
    decision = filter.should_trade("SPY", signal=0.8, current_weight=0.0)
    # -> {"action": "enter", "signal": 0.8, ...}

    # 期間更新
    filter.update_period("SPY")

    # 一括適用
    adjusted = apply_min_holding(signals, weights, filter)
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

class TradeAction(str, Enum):
    """取引アクション"""
    HOLD = "hold"           # 保持（変更なし）
    ENTER = "enter"         # 新規エントリー
    EXIT = "exit"           # エグジット
    REVERSE = "reverse"     # ポジション反転
    FORCE_EXIT = "force_exit"  # 強制エグジット（シグナル反転）
    BLOCKED = "blocked"     # 最低保有期間によりブロック


class PositionDirection(str, Enum):
    """ポジション方向"""
    LONG = "long"       # ロング
    SHORT = "short"     # ショート
    FLAT = "flat"       # ノーポジション


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class HoldingInfo:
    """保有情報

    Attributes:
        asset: 資産名
        direction: ポジション方向
        periods_held: 保有期間（日数）
        entry_signal: エントリー時のシグナル強度
        entry_date: エントリー日時
        can_trade: 取引可能かどうか（最低期間経過）
    """
    asset: str
    direction: PositionDirection = PositionDirection.FLAT
    periods_held: int = 0
    entry_signal: float = 0.0
    entry_date: datetime | None = None
    can_trade: bool = True

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset": self.asset,
            "direction": self.direction.value,
            "periods_held": self.periods_held,
            "entry_signal": self.entry_signal,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "can_trade": self.can_trade,
        }


@dataclass
class TradeDecision:
    """取引判定結果

    Attributes:
        action: 取引アクション
        signal: シグナル値
        original_signal: 元のシグナル値
        asset: 資産名
        holding_info: 保有情報
        reason: 判定理由
        blocked_periods_remaining: ブロック残期間
    """
    action: TradeAction
    signal: float
    original_signal: float
    asset: str = ""
    holding_info: HoldingInfo | None = None
    reason: str = ""
    blocked_periods_remaining: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "action": self.action.value,
            "signal": self.signal,
            "original_signal": self.original_signal,
            "asset": self.asset,
            "holding_info": self.holding_info.to_dict() if self.holding_info else None,
            "reason": self.reason,
            "blocked_periods_remaining": self.blocked_periods_remaining,
        }


@dataclass
class MinHoldingPeriodConfig:
    """MinHoldingPeriodFilter設定

    Attributes:
        min_periods: 最低保有期間（日数）
        force_exit_on_reversal: シグナル反転時の強制エグジット
        reversal_threshold: 反転閾値（この値以下でシグナル反転とみなす）
        entry_threshold: エントリー閾値
        exit_threshold: エグジット閾値
        use_direction: ポジション方向を考慮するか
    """
    min_periods: int = 5
    force_exit_on_reversal: bool = True
    reversal_threshold: float = -0.5
    entry_threshold: float = 0.3
    exit_threshold: float = 0.1
    use_direction: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        if not -1.0 <= self.reversal_threshold <= 1.0:
            raise ValueError("reversal_threshold must be in [-1, 1]")


# =============================================================================
# メインクラス
# =============================================================================

class MinHoldingPeriodFilter:
    """最低保有期間フィルター

    過剰取引を防止するための最低保有期間フィルタ。
    一度ポジションを取ったら最低N日は保持する。

    Usage:
        filter = MinHoldingPeriodFilter(
            min_periods=5,
            force_exit_on_reversal=True,
            reversal_threshold=-0.5,
        )

        # 取引判定
        decision = filter.should_trade("SPY", signal=0.8, current_weight=0.0)
        if decision.action == TradeAction.ENTER:
            # エントリー処理
            pass

        # 期間更新（毎日呼び出し）
        filter.update_period("SPY")
    """

    def __init__(
        self,
        min_periods: int = 5,
        force_exit_on_reversal: bool = True,
        reversal_threshold: float = -0.5,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.1,
        config: MinHoldingPeriodConfig | None = None,
    ) -> None:
        """初期化

        Args:
            min_periods: 最低保有期間（日数）
            force_exit_on_reversal: シグナル反転時の強制エグジット
            reversal_threshold: 反転閾値
            entry_threshold: エントリー閾値
            exit_threshold: エグジット閾値
            config: 設定オブジェクト（指定時は他の引数は無視）
        """
        if config is not None:
            self.config = config
        else:
            self.config = MinHoldingPeriodConfig(
                min_periods=min_periods,
                force_exit_on_reversal=force_exit_on_reversal,
                reversal_threshold=reversal_threshold,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

        # 保有情報: asset -> HoldingInfo
        self._holdings: dict[str, HoldingInfo] = {}

        # 統計情報
        self._stats = {
            "total_decisions": 0,
            "blocked_trades": 0,
            "force_exits": 0,
            "normal_exits": 0,
            "entries": 0,
        }

    @property
    def min_periods(self) -> int:
        """最低保有期間"""
        return self.config.min_periods

    @property
    def force_exit_on_reversal(self) -> bool:
        """シグナル反転時の強制エグジット"""
        return self.config.force_exit_on_reversal

    @property
    def reversal_threshold(self) -> float:
        """反転閾値"""
        return self.config.reversal_threshold

    @property
    def holdings(self) -> dict[str, HoldingInfo]:
        """保有情報（読み取り専用）"""
        return self._holdings.copy()

    def should_trade(
        self,
        asset: str,
        signal: float,
        current_weight: float = 0.0,
    ) -> TradeDecision:
        """取引すべきか判定

        Args:
            asset: 資産名
            signal: シグナル値（-1から1、正=ロング、負=ショート）
            current_weight: 現在のポートフォリオ重み

        Returns:
            TradeDecision: 取引判定結果
        """
        self._stats["total_decisions"] += 1

        # 保有情報取得（なければ作成）
        holding = self._get_or_create_holding(asset)

        # ポジション方向判定
        current_direction = self._get_direction_from_weight(current_weight)
        signal_direction = self._get_direction_from_signal(signal)

        # ケース1: ノーポジション
        if holding.direction == PositionDirection.FLAT:
            return self._handle_flat_position(
                asset, signal, signal_direction, holding
            )

        # ケース2: ポジション保有中
        return self._handle_active_position(
            asset, signal, signal_direction, current_weight, holding
        )

    def _handle_flat_position(
        self,
        asset: str,
        signal: float,
        signal_direction: PositionDirection,
        holding: HoldingInfo,
    ) -> TradeDecision:
        """ノーポジション時の処理

        Args:
            asset: 資産名
            signal: シグナル値
            signal_direction: シグナル方向
            holding: 保有情報

        Returns:
            TradeDecision
        """
        # エントリー閾値チェック
        if abs(signal) >= self.config.entry_threshold:
            # エントリー
            self._enter_position(asset, signal, signal_direction)
            self._stats["entries"] += 1

            return TradeDecision(
                action=TradeAction.ENTER,
                signal=signal,
                original_signal=signal,
                asset=asset,
                holding_info=self._holdings[asset],
                reason=f"Entry signal {signal:.3f} above threshold {self.config.entry_threshold}",
            )

        # シグナル弱いのでホールド
        return TradeDecision(
            action=TradeAction.HOLD,
            signal=0.0,  # シグナルを無効化
            original_signal=signal,
            asset=asset,
            holding_info=holding,
            reason=f"Signal {signal:.3f} below entry threshold {self.config.entry_threshold}",
        )

    def _handle_active_position(
        self,
        asset: str,
        signal: float,
        signal_direction: PositionDirection,
        current_weight: float,
        holding: HoldingInfo,
    ) -> TradeDecision:
        """ポジション保有時の処理

        Args:
            asset: 資産名
            signal: シグナル値
            signal_direction: シグナル方向
            current_weight: 現在の重み
            holding: 保有情報

        Returns:
            TradeDecision
        """
        periods_remaining = max(0, self.config.min_periods - holding.periods_held)
        can_trade = holding.periods_held >= self.config.min_periods

        # 強制エグジットチェック（シグナル反転）
        if self.config.force_exit_on_reversal:
            reversal = self._check_reversal(holding, signal)
            if reversal:
                # 強制エグジット
                self._exit_position(asset)
                self._stats["force_exits"] += 1

                return TradeDecision(
                    action=TradeAction.FORCE_EXIT,
                    signal=signal,  # 反転シグナルを使用
                    original_signal=signal,
                    asset=asset,
                    holding_info=holding,
                    reason=f"Force exit: signal reversal {signal:.3f} below threshold {self.config.reversal_threshold}",
                    blocked_periods_remaining=0,
                )

        # 最低保有期間チェック
        if not can_trade:
            # ブロック
            self._stats["blocked_trades"] += 1

            # シグナルは現在方向を維持（エグジットしない）
            maintained_signal = self._get_maintained_signal(holding)

            return TradeDecision(
                action=TradeAction.BLOCKED,
                signal=maintained_signal,
                original_signal=signal,
                asset=asset,
                holding_info=holding,
                reason=f"Min holding period: {periods_remaining} periods remaining",
                blocked_periods_remaining=periods_remaining,
            )

        # 最低保有期間経過後
        # エグジット条件チェック
        if self._should_exit(holding, signal, signal_direction):
            self._exit_position(asset)
            self._stats["normal_exits"] += 1

            return TradeDecision(
                action=TradeAction.EXIT,
                signal=signal,
                original_signal=signal,
                asset=asset,
                holding_info=holding,
                reason="Exit condition met after min holding period",
            )

        # ポジション反転チェック
        if self._should_reverse(holding, signal_direction):
            old_direction = holding.direction
            self._exit_position(asset)
            self._enter_position(asset, signal, signal_direction)

            return TradeDecision(
                action=TradeAction.REVERSE,
                signal=signal,
                original_signal=signal,
                asset=asset,
                holding_info=self._holdings[asset],
                reason=f"Position reversal: {old_direction.value} -> {signal_direction.value}",
            )

        # ホールド
        return TradeDecision(
            action=TradeAction.HOLD,
            signal=signal,
            original_signal=signal,
            asset=asset,
            holding_info=holding,
            reason="Continue holding position",
        )

    def _check_reversal(self, holding: HoldingInfo, signal: float) -> bool:
        """シグナル反転をチェック

        Args:
            holding: 保有情報
            signal: 現在のシグナル

        Returns:
            True if reversal detected
        """
        if holding.direction == PositionDirection.LONG:
            # ロング保有中に強いショートシグナル
            return signal <= self.config.reversal_threshold
        elif holding.direction == PositionDirection.SHORT:
            # ショート保有中に強いロングシグナル
            return signal >= -self.config.reversal_threshold

        return False

    def _should_exit(
        self,
        holding: HoldingInfo,
        signal: float,
        signal_direction: PositionDirection,
    ) -> bool:
        """エグジットすべきかチェック

        Args:
            holding: 保有情報
            signal: 現在のシグナル
            signal_direction: シグナル方向

        Returns:
            True if should exit
        """
        # シグナルがエグジット閾値以下
        if abs(signal) < self.config.exit_threshold:
            return True

        # シグナル方向が反転（フラットへ）
        if signal_direction == PositionDirection.FLAT:
            return True

        return False

    def _should_reverse(
        self,
        holding: HoldingInfo,
        signal_direction: PositionDirection,
    ) -> bool:
        """ポジション反転すべきかチェック

        Args:
            holding: 保有情報
            signal_direction: シグナル方向

        Returns:
            True if should reverse
        """
        if signal_direction == PositionDirection.FLAT:
            return False

        # 方向が異なる場合は反転
        return holding.direction != signal_direction

    def _get_maintained_signal(self, holding: HoldingInfo) -> float:
        """保有方向を維持するシグナルを返す

        Args:
            holding: 保有情報

        Returns:
            維持用シグナル
        """
        if holding.direction == PositionDirection.LONG:
            return max(holding.entry_signal, self.config.entry_threshold)
        elif holding.direction == PositionDirection.SHORT:
            return min(holding.entry_signal, -self.config.entry_threshold)
        return 0.0

    def _get_or_create_holding(self, asset: str) -> HoldingInfo:
        """保有情報を取得または作成

        Args:
            asset: 資産名

        Returns:
            HoldingInfo
        """
        if asset not in self._holdings:
            self._holdings[asset] = HoldingInfo(asset=asset)
        return self._holdings[asset]

    def _enter_position(
        self,
        asset: str,
        signal: float,
        direction: PositionDirection,
    ) -> None:
        """ポジションにエントリー

        Args:
            asset: 資産名
            signal: シグナル値
            direction: ポジション方向
        """
        self._holdings[asset] = HoldingInfo(
            asset=asset,
            direction=direction,
            periods_held=0,
            entry_signal=signal,
            entry_date=datetime.now(),
            can_trade=False,
        )
        logger.debug("Entered position: %s, direction=%s, signal=%.3f",
                    asset, direction.value, signal)

    def _exit_position(self, asset: str) -> None:
        """ポジションをエグジット

        Args:
            asset: 資産名
        """
        if asset in self._holdings:
            old = self._holdings[asset]
            logger.debug("Exited position: %s, held=%d periods",
                        asset, old.periods_held)
            self._holdings[asset] = HoldingInfo(asset=asset)

    def _get_direction_from_weight(self, weight: float) -> PositionDirection:
        """重みからポジション方向を取得

        Args:
            weight: ポートフォリオ重み

        Returns:
            PositionDirection
        """
        if weight > 0.01:
            return PositionDirection.LONG
        elif weight < -0.01:
            return PositionDirection.SHORT
        return PositionDirection.FLAT

    def _get_direction_from_signal(self, signal: float) -> PositionDirection:
        """シグナルからポジション方向を取得

        Args:
            signal: シグナル値

        Returns:
            PositionDirection
        """
        if signal > self.config.exit_threshold:
            return PositionDirection.LONG
        elif signal < -self.config.exit_threshold:
            return PositionDirection.SHORT
        return PositionDirection.FLAT

    def update_period(self, asset: str) -> None:
        """保有期間を1日進める

        Args:
            asset: 資産名
        """
        if asset in self._holdings:
            holding = self._holdings[asset]
            if holding.direction != PositionDirection.FLAT:
                holding.periods_held += 1
                holding.can_trade = holding.periods_held >= self.config.min_periods
                logger.debug("Updated period: %s, periods=%d, can_trade=%s",
                            asset, holding.periods_held, holding.can_trade)

    def update_all_periods(self) -> None:
        """全資産の保有期間を1日進める"""
        for asset in self._holdings:
            self.update_period(asset)

    def reset(self, asset: str | None = None) -> None:
        """状態リセット

        Args:
            asset: 資産名（Noneの場合は全リセット）
        """
        if asset is None:
            self._holdings.clear()
            logger.info("Reset all holdings")
        elif asset in self._holdings:
            self._holdings[asset] = HoldingInfo(asset=asset)
            logger.debug("Reset holding: %s", asset)

    def get_holding_info(self, asset: str) -> HoldingInfo | None:
        """保有情報取得

        Args:
            asset: 資産名

        Returns:
            HoldingInfo or None
        """
        return self._holdings.get(asset)

    def get_all_holdings(self) -> dict[str, HoldingInfo]:
        """全保有情報取得

        Returns:
            全保有情報の辞書
        """
        return self._holdings.copy()

    def get_active_holdings(self) -> dict[str, HoldingInfo]:
        """アクティブな保有のみ取得

        Returns:
            アクティブな保有情報の辞書
        """
        return {
            asset: info
            for asset, info in self._holdings.items()
            if info.direction != PositionDirection.FLAT
        }

    def get_blocked_assets(self) -> list[str]:
        """最低保有期間でブロックされている資産を取得

        Returns:
            ブロックされている資産リスト
        """
        return [
            asset
            for asset, info in self._holdings.items()
            if info.direction != PositionDirection.FLAT and not info.can_trade
        ]

    def get_stats(self) -> dict[str, int]:
        """統計情報取得

        Returns:
            統計情報の辞書
        """
        return self._stats.copy()

    def get_summary(self) -> dict[str, Any]:
        """サマリー取得

        Returns:
            サマリー辞書
        """
        active = self.get_active_holdings()
        blocked = self.get_blocked_assets()

        return {
            "config": {
                "min_periods": self.config.min_periods,
                "force_exit_on_reversal": self.config.force_exit_on_reversal,
                "reversal_threshold": self.config.reversal_threshold,
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
            },
            "active_positions": len(active),
            "blocked_positions": len(blocked),
            "blocked_assets": blocked,
            "stats": self._stats,
        }


# =============================================================================
# 便利関数
# =============================================================================

def apply_min_holding(
    signals: dict[str, float] | pd.Series,
    weights: dict[str, float] | pd.Series,
    filter_instance: MinHoldingPeriodFilter,
    update_periods: bool = True,
) -> dict[str, TradeDecision]:
    """最低保有期間フィルターを一括適用

    Args:
        signals: シグナル辞書 {asset: signal}
        weights: 現在の重み辞書 {asset: weight}
        filter_instance: MinHoldingPeriodFilterインスタンス
        update_periods: 期間を更新するか

    Returns:
        取引判定辞書 {asset: TradeDecision}
    """
    # pd.Seriesを辞書に変換
    if isinstance(signals, pd.Series):
        signals = signals.to_dict()
    if isinstance(weights, pd.Series):
        weights = weights.to_dict()

    results: dict[str, TradeDecision] = {}

    for asset, signal in signals.items():
        current_weight = weights.get(asset, 0.0)
        decision = filter_instance.should_trade(asset, signal, current_weight)
        results[asset] = decision

    # 期間更新
    if update_periods:
        for asset in signals:
            filter_instance.update_period(asset)

    return results


def create_min_holding_filter(
    min_periods: int = 5,
    force_exit_on_reversal: bool = True,
    reversal_threshold: float = -0.5,
    entry_threshold: float = 0.3,
    exit_threshold: float = 0.1,
) -> MinHoldingPeriodFilter:
    """MinHoldingPeriodFilterを作成（ファクトリ関数）

    Args:
        min_periods: 最低保有期間
        force_exit_on_reversal: シグナル反転時の強制エグジット
        reversal_threshold: 反転閾値
        entry_threshold: エントリー閾値
        exit_threshold: エグジット閾値

    Returns:
        MinHoldingPeriodFilter
    """
    config = MinHoldingPeriodConfig(
        min_periods=min_periods,
        force_exit_on_reversal=force_exit_on_reversal,
        reversal_threshold=reversal_threshold,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    return MinHoldingPeriodFilter(config=config)


def get_filtered_signals(
    signals: dict[str, float] | pd.Series,
    weights: dict[str, float] | pd.Series,
    filter_instance: MinHoldingPeriodFilter,
) -> pd.Series:
    """フィルタ済みシグナルを取得（便利関数）

    Args:
        signals: シグナル辞書
        weights: 現在の重み辞書
        filter_instance: MinHoldingPeriodFilter

    Returns:
        フィルタ済みシグナル（pd.Series）
    """
    decisions = apply_min_holding(
        signals, weights, filter_instance, update_periods=False
    )

    filtered = {
        asset: decision.signal
        for asset, decision in decisions.items()
    }

    return pd.Series(filtered)
