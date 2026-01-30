"""
Entry/Exit Optimizer Module - エントリー/エグジット最適化

取引のエントリー・エグジットタイミングを最適化するためのツール群。

主要コンポーネント:
- HysteresisFilter: ヒステリシスフィルタ（閾値ベースのエントリー/エグジット）
- GradualEntryExit: 段階的なポジション構築・解消
- StopLossManager: 動的ストップロス管理

設計根拠:
- 頻繁な売買を抑制してトランザクションコストを削減
- シグナルノイズによるホイップソーを回避
- リスク管理の自動化

使用例:
    from src.strategy.entry_exit_optimizer import (
        HysteresisFilter,
        GradualEntryExit,
        StopLossManager,
    )

    # ヒステリシスフィルタ
    filter = HysteresisFilter(entry_threshold=0.3, exit_threshold=0.1)
    filtered_score = filter.filter_signal("AAPL", raw_score=0.35)

    # 段階的エントリー
    gradual = GradualEntryExit(entry_periods=3, exit_periods=2)
    target_weight = gradual.calculate_target_weight("AAPL", final_target=0.1)

    # ストップロス
    stop_loss = StopLossManager(initial_stop_pct=0.05, trailing_pct=0.03)
    should_exit = stop_loss.check_stop_loss("AAPL", current_price=95.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class PositionState(str, Enum):
    """ポジション状態"""
    FLAT = "flat"           # ポジションなし
    ENTERING = "entering"   # エントリー中（段階的）
    HOLDING = "holding"     # 保有中
    EXITING = "exiting"     # エグジット中（段階的）


class StopType(str, Enum):
    """ストップロスのタイプ"""
    INITIAL = "initial"       # 初期ストップ
    TRAILING = "trailing"     # トレーリングストップ
    TIME_BASED = "time_based" # 時間ベースストップ
    SIGNAL = "signal"         # シグナルベースストップ


@dataclass
class PositionInfo:
    """ポジション情報"""
    is_active: bool = False
    periods_held: int = 0
    entry_price: float = 0.0
    entry_date: Optional[datetime] = None
    current_weight: float = 0.0
    target_weight: float = 0.0
    state: PositionState = PositionState.FLAT
    highest_price: float = 0.0  # トレーリングストップ用
    lowest_price: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "periods_held": self.periods_held,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "state": self.state.value,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price if self.lowest_price != float('inf') else None,
        }


@dataclass
class FilterResult:
    """フィルタリング結果"""
    original_score: float
    filtered_score: float
    action: str  # "entry", "exit", "hold", "no_action"
    position_info: PositionInfo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_score": self.original_score,
            "filtered_score": self.filtered_score,
            "action": self.action,
            "position_info": self.position_info.to_dict(),
        }


@dataclass
class StopLossResult:
    """ストップロス判定結果"""
    triggered: bool
    stop_type: Optional[StopType] = None
    stop_price: float = 0.0
    current_price: float = 0.0
    loss_pct: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "stop_type": self.stop_type.value if self.stop_type else None,
            "stop_price": self.stop_price,
            "current_price": self.current_price,
            "loss_pct": self.loss_pct,
            "message": self.message,
        }


# =============================================================================
# HysteresisFilter
# =============================================================================

class HysteresisFilter:
    """
    ヒステリシスフィルタ

    エントリー閾値とエグジット閾値を分けることで、
    シグナルノイズによるホイップソー（頻繁な売買）を抑制する。

    動作:
    - ポジションなし: entry_threshold以上でエントリー
    - ポジションあり: exit_threshold未満でエグジット
    - 最低保有期間: min_holding_periodsを満たすまでエグジットしない

    Usage:
        filter = HysteresisFilter(
            entry_threshold=0.3,
            exit_threshold=0.1,
            min_holding_periods=5
        )

        for date, scores in daily_scores.items():
            for asset, raw_score in scores.items():
                filtered = filter.filter_signal(asset, raw_score)
                print(f"{asset}: {raw_score:.3f} -> {filtered:.3f}")
    """

    def __init__(
        self,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.1,
        min_holding_periods: int = 5,
        use_absolute: bool = True,
    ) -> None:
        """
        初期化

        Args:
            entry_threshold: エントリー閾値（スコアがこれ以上でエントリー）
            exit_threshold: エグジット閾値（スコアがこれ未満でエグジット）
            min_holding_periods: 最低保有期間（この期間はエグジットしない）
            use_absolute: 絶対値を使用するか（Trueなら負のスコアも正として扱う）
        """
        if entry_threshold <= exit_threshold:
            raise ValueError("entry_threshold must be > exit_threshold")

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_holding_periods = min_holding_periods
        self.use_absolute = use_absolute
        self.positions: Dict[str, PositionInfo] = {}

        logger.info(
            f"HysteresisFilter initialized: entry={entry_threshold}, "
            f"exit={exit_threshold}, min_hold={min_holding_periods}"
        )

    def filter_signal(
        self,
        asset: str,
        raw_score: float,
        current_date: Optional[datetime] = None,
    ) -> float:
        """
        シグナルにヒステリシスフィルタを適用

        Args:
            asset: アセットID
            raw_score: 生のシグナルスコア
            current_date: 現在日付（オプション）

        Returns:
            フィルタリング後のスコア（0.0 = ポジションなし）
        """
        # 絶対値を使用する場合
        score_to_check = abs(raw_score) if self.use_absolute else raw_score

        # ポジション情報を取得
        pos_info = self.positions.get(asset, PositionInfo())

        if not pos_info.is_active:
            # ポジションなし → エントリー判定
            if score_to_check >= self.entry_threshold:
                # エントリー
                pos_info.is_active = True
                pos_info.periods_held = 1
                pos_info.state = PositionState.HOLDING
                pos_info.entry_date = current_date
                self.positions[asset] = pos_info
                logger.debug(f"Entry: {asset}, score={raw_score:.3f}")
                return raw_score
            else:
                # エントリーせず
                return 0.0
        else:
            # ポジションあり → エグジット判定
            pos_info.periods_held += 1

            # 最低保有期間チェック
            if pos_info.periods_held < self.min_holding_periods:
                self.positions[asset] = pos_info
                return raw_score

            # エグジット判定
            if score_to_check < self.exit_threshold:
                # エグジット
                pos_info.is_active = False
                pos_info.periods_held = 0
                pos_info.state = PositionState.FLAT
                self.positions[asset] = pos_info
                logger.debug(f"Exit: {asset}, score={raw_score:.3f}")
                return 0.0
            else:
                # 保有継続
                self.positions[asset] = pos_info
                return raw_score

    def filter_signal_detailed(
        self,
        asset: str,
        raw_score: float,
        current_date: Optional[datetime] = None,
    ) -> FilterResult:
        """
        詳細な結果付きでフィルタリング

        Args:
            asset: アセットID
            raw_score: 生のシグナルスコア
            current_date: 現在日付

        Returns:
            FilterResult
        """
        pos_before = self.positions.get(asset, PositionInfo())
        was_active = pos_before.is_active

        filtered_score = self.filter_signal(asset, raw_score, current_date)

        pos_after = self.positions.get(asset, PositionInfo())
        is_active = pos_after.is_active

        # アクションを判定
        if not was_active and is_active:
            action = "entry"
        elif was_active and not is_active:
            action = "exit"
        elif is_active:
            action = "hold"
        else:
            action = "no_action"

        return FilterResult(
            original_score=raw_score,
            filtered_score=filtered_score,
            action=action,
            position_info=pos_after,
        )

    def get_position_info(self, asset: str) -> PositionInfo:
        """ポジション情報を取得"""
        return self.positions.get(asset, PositionInfo())

    def reset(self, asset: Optional[str] = None) -> None:
        """ポジション状態をリセット"""
        if asset:
            self.positions.pop(asset, None)
        else:
            self.positions.clear()


# =============================================================================
# GradualEntryExit
# =============================================================================

class GradualEntryExit:
    """
    段階的エントリー/エグジット

    一度に全ポジションを取らず、複数期間に分けて
    段階的にポジションを構築・解消する。

    利点:
    - 市場インパクトの軽減
    - タイミングリスクの分散
    - 平均取得価格の改善

    Usage:
        gradual = GradualEntryExit(entry_periods=3, exit_periods=2)

        # 目標重み10%に段階的にエントリー
        for period in range(5):
            target = gradual.calculate_target_weight("AAPL", final_target=0.1)
            print(f"Period {period}: target={target:.2%}")
    """

    def __init__(
        self,
        entry_periods: int = 3,
        exit_periods: int = 2,
        entry_schedule: Optional[List[float]] = None,
        exit_schedule: Optional[List[float]] = None,
    ) -> None:
        """
        初期化

        Args:
            entry_periods: エントリーに要する期間数
            exit_periods: エグジットに要する期間数
            entry_schedule: カスタムエントリースケジュール（合計1.0）
            exit_schedule: カスタムエグジットスケジュール（合計1.0）
        """
        self.entry_periods = entry_periods
        self.exit_periods = exit_periods

        # デフォルトは均等分割
        self.entry_schedule = entry_schedule or [1.0 / entry_periods] * entry_periods
        self.exit_schedule = exit_schedule or [1.0 / exit_periods] * exit_periods

        # スケジュールの正規化
        self.entry_schedule = self._normalize_schedule(self.entry_schedule)
        self.exit_schedule = self._normalize_schedule(self.exit_schedule)

        self.positions: Dict[str, PositionInfo] = {}

        logger.info(
            f"GradualEntryExit initialized: entry_periods={entry_periods}, "
            f"exit_periods={exit_periods}"
        )

    def _normalize_schedule(self, schedule: List[float]) -> List[float]:
        """スケジュールを正規化（合計1.0）"""
        total = sum(schedule)
        if total > 0:
            return [x / total for x in schedule]
        return schedule

    def start_entry(
        self,
        asset: str,
        final_target: float,
        current_date: Optional[datetime] = None,
    ) -> None:
        """エントリー開始"""
        pos_info = PositionInfo(
            is_active=True,
            periods_held=0,
            current_weight=0.0,
            target_weight=final_target,
            state=PositionState.ENTERING,
            entry_date=current_date,
        )
        self.positions[asset] = pos_info
        logger.debug(f"Start entry: {asset}, target={final_target:.2%}")

    def start_exit(self, asset: str) -> None:
        """エグジット開始"""
        pos_info = self.positions.get(asset)
        if pos_info and pos_info.is_active:
            pos_info.state = PositionState.EXITING
            pos_info.periods_held = 0  # エグジットカウントをリセット
            self.positions[asset] = pos_info
            logger.debug(f"Start exit: {asset}")

    def calculate_target_weight(
        self,
        asset: str,
        final_target: Optional[float] = None,
    ) -> float:
        """
        現在の目標重みを計算

        Args:
            asset: アセットID
            final_target: 最終目標重み（新規エントリーの場合）

        Returns:
            現在の目標重み
        """
        pos_info = self.positions.get(asset)

        # 新規エントリー
        if pos_info is None or not pos_info.is_active:
            if final_target is not None and final_target > 0:
                self.start_entry(asset, final_target)
                pos_info = self.positions[asset]
            else:
                return 0.0

        if pos_info.state == PositionState.ENTERING:
            # エントリー中
            period = min(pos_info.periods_held, len(self.entry_schedule) - 1)
            cumulative = sum(self.entry_schedule[: period + 1])
            target = pos_info.target_weight * cumulative

            pos_info.periods_held += 1
            pos_info.current_weight = target

            # エントリー完了チェック
            if pos_info.periods_held >= len(self.entry_schedule):
                pos_info.state = PositionState.HOLDING
                pos_info.current_weight = pos_info.target_weight

            self.positions[asset] = pos_info
            return target

        elif pos_info.state == PositionState.EXITING:
            # エグジット中
            period = min(pos_info.periods_held, len(self.exit_schedule) - 1)
            cumulative = sum(self.exit_schedule[: period + 1])
            remaining = 1.0 - cumulative
            target = pos_info.current_weight * remaining

            pos_info.periods_held += 1

            # エグジット完了チェック
            if pos_info.periods_held >= len(self.exit_schedule):
                pos_info.state = PositionState.FLAT
                pos_info.is_active = False
                pos_info.current_weight = 0.0
                target = 0.0

            self.positions[asset] = pos_info
            return target

        elif pos_info.state == PositionState.HOLDING:
            # 保有中
            # 目標重みの変更があれば対応
            if final_target is not None and final_target != pos_info.target_weight:
                if final_target > pos_info.target_weight:
                    # 増加 → 追加エントリー
                    pos_info.target_weight = final_target
                    pos_info.state = PositionState.ENTERING
                    pos_info.periods_held = 0
                elif final_target < pos_info.target_weight:
                    # 減少 → 部分エグジット
                    pos_info.target_weight = final_target
                self.positions[asset] = pos_info

            return pos_info.target_weight

        return 0.0

    def get_position_info(self, asset: str) -> PositionInfo:
        """ポジション情報を取得"""
        return self.positions.get(asset, PositionInfo())

    def reset(self, asset: Optional[str] = None) -> None:
        """状態をリセット"""
        if asset:
            self.positions.pop(asset, None)
        else:
            self.positions.clear()


# =============================================================================
# StopLossManager
# =============================================================================

class StopLossManager:
    """
    動的ストップロス管理

    複数のストップロスメカニズムを提供:
    - 初期ストップ: エントリー価格から固定%
    - トレーリングストップ: 最高値から固定%
    - 時間ベースストップ: 一定期間後に強制エグジット

    Usage:
        stop_loss = StopLossManager(
            initial_stop_pct=0.05,
            trailing_pct=0.03,
            max_holding_periods=60
        )

        # エントリー時
        stop_loss.register_entry("AAPL", entry_price=100.0)

        # 毎日のチェック
        result = stop_loss.check_stop_loss("AAPL", current_price=95.0)
        if result.triggered:
            print(f"Stop loss triggered: {result.stop_type}")
    """

    def __init__(
        self,
        initial_stop_pct: float = 0.05,
        trailing_pct: float = 0.03,
        max_holding_periods: Optional[int] = None,
        use_trailing: bool = True,
        trailing_activation_pct: float = 0.02,
    ) -> None:
        """
        初期化

        Args:
            initial_stop_pct: 初期ストップロス幅（例: 0.05 = 5%）
            trailing_pct: トレーリングストップ幅（例: 0.03 = 3%）
            max_holding_periods: 最大保有期間（None = 無制限）
            use_trailing: トレーリングストップを使用するか
            trailing_activation_pct: トレーリング発動に必要な利益幅
        """
        self.initial_stop_pct = initial_stop_pct
        self.trailing_pct = trailing_pct
        self.max_holding_periods = max_holding_periods
        self.use_trailing = use_trailing
        self.trailing_activation_pct = trailing_activation_pct

        self.positions: Dict[str, PositionInfo] = {}

        logger.info(
            f"StopLossManager initialized: initial={initial_stop_pct:.1%}, "
            f"trailing={trailing_pct:.1%}"
        )

    def register_entry(
        self,
        asset: str,
        entry_price: float,
        entry_date: Optional[datetime] = None,
    ) -> None:
        """
        エントリーを登録

        Args:
            asset: アセットID
            entry_price: エントリー価格
            entry_date: エントリー日付
        """
        pos_info = PositionInfo(
            is_active=True,
            periods_held=0,
            entry_price=entry_price,
            entry_date=entry_date,
            highest_price=entry_price,
            lowest_price=entry_price,
            state=PositionState.HOLDING,
        )
        self.positions[asset] = pos_info
        logger.debug(f"Entry registered: {asset} at {entry_price:.2f}")

    def update_price(self, asset: str, current_price: float) -> None:
        """
        価格を更新（高値/安値を追跡）

        Args:
            asset: アセットID
            current_price: 現在価格
        """
        pos_info = self.positions.get(asset)
        if pos_info and pos_info.is_active:
            pos_info.highest_price = max(pos_info.highest_price, current_price)
            pos_info.lowest_price = min(pos_info.lowest_price, current_price)
            pos_info.periods_held += 1
            self.positions[asset] = pos_info

    def check_stop_loss(
        self,
        asset: str,
        current_price: float,
        signal_score: Optional[float] = None,
    ) -> StopLossResult:
        """
        ストップロスをチェック

        Args:
            asset: アセットID
            current_price: 現在価格
            signal_score: シグナルスコア（シグナルベースストップ用）

        Returns:
            StopLossResult
        """
        pos_info = self.positions.get(asset)

        if not pos_info or not pos_info.is_active:
            return StopLossResult(
                triggered=False,
                message="No active position",
            )

        # 価格を更新
        self.update_price(asset, current_price)
        pos_info = self.positions[asset]

        entry_price = pos_info.entry_price
        highest_price = pos_info.highest_price

        # 1. 初期ストップロスチェック
        initial_stop_price = entry_price * (1 - self.initial_stop_pct)
        if current_price <= initial_stop_price:
            loss_pct = (current_price - entry_price) / entry_price
            self._clear_position(asset)
            return StopLossResult(
                triggered=True,
                stop_type=StopType.INITIAL,
                stop_price=initial_stop_price,
                current_price=current_price,
                loss_pct=loss_pct,
                message=f"Initial stop triggered at {current_price:.2f}",
            )

        # 2. トレーリングストップチェック
        if self.use_trailing:
            # トレーリング発動条件チェック
            profit_pct = (highest_price - entry_price) / entry_price
            if profit_pct >= self.trailing_activation_pct:
                trailing_stop_price = highest_price * (1 - self.trailing_pct)
                if current_price <= trailing_stop_price:
                    loss_from_high = (current_price - highest_price) / highest_price
                    self._clear_position(asset)
                    return StopLossResult(
                        triggered=True,
                        stop_type=StopType.TRAILING,
                        stop_price=trailing_stop_price,
                        current_price=current_price,
                        loss_pct=loss_from_high,
                        message=f"Trailing stop triggered at {current_price:.2f}",
                    )

        # 3. 時間ベースストップチェック
        if self.max_holding_periods is not None:
            if pos_info.periods_held >= self.max_holding_periods:
                pnl_pct = (current_price - entry_price) / entry_price
                self._clear_position(asset)
                return StopLossResult(
                    triggered=True,
                    stop_type=StopType.TIME_BASED,
                    stop_price=current_price,
                    current_price=current_price,
                    loss_pct=pnl_pct,
                    message=f"Time-based stop after {pos_info.periods_held} periods",
                )

        # ストップ発動せず
        return StopLossResult(
            triggered=False,
            current_price=current_price,
            message="No stop triggered",
        )

    def get_stop_prices(self, asset: str, current_price: float) -> Dict[str, float]:
        """
        現在のストップ価格を取得

        Args:
            asset: アセットID
            current_price: 現在価格

        Returns:
            各ストップタイプとその価格
        """
        pos_info = self.positions.get(asset)
        if not pos_info or not pos_info.is_active:
            return {}

        stops = {
            "initial": pos_info.entry_price * (1 - self.initial_stop_pct),
        }

        if self.use_trailing:
            profit_pct = (pos_info.highest_price - pos_info.entry_price) / pos_info.entry_price
            if profit_pct >= self.trailing_activation_pct:
                stops["trailing"] = pos_info.highest_price * (1 - self.trailing_pct)

        return stops

    def _clear_position(self, asset: str) -> None:
        """ポジションをクリア"""
        pos_info = self.positions.get(asset)
        if pos_info:
            pos_info.is_active = False
            pos_info.state = PositionState.FLAT
            self.positions[asset] = pos_info

    def get_position_info(self, asset: str) -> PositionInfo:
        """ポジション情報を取得"""
        return self.positions.get(asset, PositionInfo())

    def reset(self, asset: Optional[str] = None) -> None:
        """状態をリセット"""
        if asset:
            self.positions.pop(asset, None)
        else:
            self.positions.clear()


# =============================================================================
# Integrated Entry/Exit Optimizer
# =============================================================================

@dataclass
class EntryExitConfig:
    """エントリー/エグジット最適化設定"""
    # Hysteresis
    entry_threshold: float = 0.3
    exit_threshold: float = 0.1
    min_holding_periods: int = 5

    # Gradual
    entry_periods: int = 3
    exit_periods: int = 2

    # Stop Loss
    initial_stop_pct: float = 0.05
    trailing_pct: float = 0.03
    use_trailing: bool = True
    max_holding_periods: Optional[int] = None


class EntryExitOptimizer:
    """
    統合エントリー/エグジット最適化

    HysteresisFilter, GradualEntryExit, StopLossManager を統合した
    ポジション管理クラス。

    Usage:
        config = EntryExitConfig(
            entry_threshold=0.3,
            initial_stop_pct=0.05,
        )
        optimizer = EntryExitOptimizer(config)

        # シグナル処理
        result = optimizer.process_signal("AAPL", score=0.35, price=100.0)
        print(f"Target weight: {result['target_weight']:.2%}")
    """

    def __init__(self, config: Optional[EntryExitConfig] = None) -> None:
        self.config = config or EntryExitConfig()

        self.hysteresis = HysteresisFilter(
            entry_threshold=self.config.entry_threshold,
            exit_threshold=self.config.exit_threshold,
            min_holding_periods=self.config.min_holding_periods,
        )

        self.gradual = GradualEntryExit(
            entry_periods=self.config.entry_periods,
            exit_periods=self.config.exit_periods,
        )

        self.stop_loss = StopLossManager(
            initial_stop_pct=self.config.initial_stop_pct,
            trailing_pct=self.config.trailing_pct,
            use_trailing=self.config.use_trailing,
            max_holding_periods=self.config.max_holding_periods,
        )

    def process_signal(
        self,
        asset: str,
        score: float,
        price: float,
        target_weight: float = 0.0,
        current_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        シグナルを処理して最終的なターゲット重みを決定

        Args:
            asset: アセットID
            score: シグナルスコア
            price: 現在価格
            target_weight: シグナルベースの目標重み
            current_date: 現在日付

        Returns:
            処理結果の辞書
        """
        result = {
            "asset": asset,
            "original_score": score,
            "original_target": target_weight,
            "final_target": 0.0,
            "action": "none",
            "stop_triggered": False,
        }

        # 1. ヒステリシスフィルタ
        filter_result = self.hysteresis.filter_signal_detailed(asset, score, current_date)
        result["hysteresis_action"] = filter_result.action

        if filter_result.action == "entry":
            # 新規エントリー
            self.stop_loss.register_entry(asset, price, current_date)
            final_target = self.gradual.calculate_target_weight(asset, target_weight)
            result["final_target"] = final_target
            result["action"] = "entry"

        elif filter_result.action == "exit":
            # エグジット
            self.gradual.start_exit(asset)
            final_target = self.gradual.calculate_target_weight(asset)
            result["final_target"] = final_target
            result["action"] = "exit"

        elif filter_result.action == "hold":
            # 保有中 → ストップロスチェック
            stop_result = self.stop_loss.check_stop_loss(asset, price, score)

            if stop_result.triggered:
                # ストップ発動
                self.gradual.start_exit(asset)
                self.hysteresis.reset(asset)
                result["final_target"] = 0.0
                result["action"] = "stop_exit"
                result["stop_triggered"] = True
                result["stop_type"] = stop_result.stop_type.value if stop_result.stop_type else None
            else:
                # 継続保有
                final_target = self.gradual.calculate_target_weight(asset, target_weight)
                result["final_target"] = final_target
                result["action"] = "hold"

        return result

    def reset(self, asset: Optional[str] = None) -> None:
        """全状態をリセット"""
        self.hysteresis.reset(asset)
        self.gradual.reset(asset)
        self.stop_loss.reset(asset)
