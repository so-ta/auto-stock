"""
Hysteresis Filter Module - ヒステリシス・フィルター

エントリー/エグジット閾値を分離して過剰取引を防止する。
一度ポジションを持ったら、より低い閾値を下回るまで維持する。

設計根拠:
- 要求.md §6: リスク管理
- 過剰取引（チャーン）の防止
- 取引コスト削減

主な機能:
- エントリー閾値: 高め（確信度が必要）
- エグジット閾値: 低め（一度入ったら粘る）
- 最低保有期間の強制

ヒステリシスの例:
    entry_threshold = 0.3, exit_threshold = 0.1 の場合:
    - シグナル 0.35 → エントリー（0.3超）
    - シグナル 0.25 → 維持（0.1超、まだ粘る）
    - シグナル 0.15 → 維持（0.1超、まだ粘る）
    - シグナル 0.05 → エグジット（0.1以下）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """ポジション状態"""
    NO_POSITION = "no_position"     # ポジションなし
    LONG = "long"                   # ロングポジション
    SHORT = "short"                 # ショートポジション


@dataclass
class PositionInfo:
    """ポジション情報

    Attributes:
        state: ポジション状態
        entry_signal: エントリー時のシグナル値
        entry_time: エントリー時刻
        holding_periods: 保有期間数
        peak_signal: 保有中の最大シグナル値
        last_signal: 最新のシグナル値
    """
    state: PositionState = PositionState.NO_POSITION
    entry_signal: float = 0.0
    entry_time: Optional[datetime] = None
    holding_periods: int = 0
    peak_signal: float = 0.0
    last_signal: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "state": self.state.value,
            "entry_signal": self.entry_signal,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "holding_periods": self.holding_periods,
            "peak_signal": self.peak_signal,
            "last_signal": self.last_signal,
        }


@dataclass(frozen=True)
class HysteresisConfig:
    """ヒステリシス設定

    Attributes:
        entry_threshold: エントリー閾値（高め、確信度が必要）
        exit_threshold: エグジット閾値（低め、一度入ったら粘る）
        min_holding_periods: 最低保有期間（これ未満ではエグジットしない）
        symmetric: ショート側も対称的に扱うか
        use_abs_signal: シグナルの絶対値で判定するか
    """
    entry_threshold: float = 0.3
    exit_threshold: float = 0.1
    min_holding_periods: int = 5
    symmetric: bool = True
    use_abs_signal: bool = False

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.entry_threshold <= self.exit_threshold:
            raise ValueError(
                "entry_threshold must be greater than exit_threshold"
            )
        if self.entry_threshold < 0 or self.exit_threshold < 0:
            raise ValueError("Thresholds must be non-negative")
        if self.min_holding_periods < 0:
            raise ValueError("min_holding_periods must be non-negative")


@dataclass
class FilterResult:
    """フィルター結果

    Attributes:
        asset: アセット識別子
        original_signal: 元のシグナル値
        filtered_signal: フィルター後のシグナル値
        position_state: ポジション状態
        action: 取られたアクション（entry/exit/hold/suppress）
        holding_periods: 保有期間
    """
    asset: str
    original_signal: float
    filtered_signal: float
    position_state: PositionState
    action: str
    holding_periods: int

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset": self.asset,
            "original_signal": self.original_signal,
            "filtered_signal": self.filtered_signal,
            "position_state": self.position_state.value,
            "action": self.action,
            "holding_periods": self.holding_periods,
        }


class HysteresisFilter:
    """ヒステリシス・フィルタークラス

    エントリー/エグジット閾値を分離して過剰取引を防止する。
    一度ポジションを持ったら、より低い閾値を下回るまで維持する。

    動作ロジック:
    - ポジションなし → エントリー判定（閾値高め）
    - ポジションあり → エグジット判定（閾値低め）
    - 最低保有期間チェック

    Usage:
        config = HysteresisConfig(
            entry_threshold=0.3,
            exit_threshold=0.1,
            min_holding_periods=5,
        )
        filter = HysteresisFilter(config)

        # シグナルをフィルター
        result = filter.filter("AAPL", 0.35)  # → entry
        result = filter.filter("AAPL", 0.25)  # → hold
        result = filter.filter("AAPL", 0.05)  # → exit

    Attributes:
        config: ヒステリシス設定
        positions: アセット別ポジション情報
    """

    def __init__(self, config: Optional[HysteresisConfig] = None) -> None:
        """初期化

        Args:
            config: ヒステリシス設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or HysteresisConfig()
        self.positions: Dict[str, PositionInfo] = {}

    @property
    def entry_threshold(self) -> float:
        """エントリー閾値"""
        return self.config.entry_threshold

    @property
    def exit_threshold(self) -> float:
        """エグジット閾値"""
        return self.config.exit_threshold

    @property
    def min_holding_periods(self) -> int:
        """最低保有期間"""
        return self.config.min_holding_periods

    def _get_or_create_position(self, asset: str) -> PositionInfo:
        """ポジション情報を取得または作成"""
        if asset not in self.positions:
            self.positions[asset] = PositionInfo()
        return self.positions[asset]

    def filter(
        self,
        asset: str,
        signal_score: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """ヒステリシスフィルタを適用

        Args:
            asset: アセット識別子
            signal_score: シグナル値（-1〜+1）
            timestamp: 現在時刻

        Returns:
            フィルタ後のシグナル（0 or signal_score）
        """
        result = self.filter_with_info(asset, signal_score, timestamp)
        return result.filtered_signal

    def filter_with_info(
        self,
        asset: str,
        signal_score: float,
        timestamp: Optional[datetime] = None,
    ) -> FilterResult:
        """ヒステリシスフィルタを適用（詳細情報付き）

        Args:
            asset: アセット識別子
            signal_score: シグナル値（-1〜+1）
            timestamp: 現在時刻

        Returns:
            FilterResult: フィルター結果と詳細情報
        """
        timestamp = timestamp or datetime.now()
        pos = self._get_or_create_position(asset)

        # シグナルの絶対値を使用するか
        signal_value = abs(signal_score) if self.config.use_abs_signal else signal_score
        signal_direction = 1 if signal_score >= 0 else -1

        # ロング/ショートの閾値判定
        if self.config.symmetric:
            abs_signal = abs(signal_value)
            above_entry = abs_signal > self.entry_threshold
            above_exit = abs_signal > self.exit_threshold
        else:
            # ロングのみ
            above_entry = signal_value > self.entry_threshold
            above_exit = signal_value > self.exit_threshold

        # 現在ポジションなし
        if pos.state == PositionState.NO_POSITION:
            if above_entry:
                # エントリー条件満たす
                new_state = (
                    PositionState.LONG if signal_direction > 0
                    else PositionState.SHORT
                )
                pos.state = new_state
                pos.entry_signal = signal_score
                pos.entry_time = timestamp
                pos.holding_periods = 1
                pos.peak_signal = signal_score
                pos.last_signal = signal_score

                logger.debug(
                    "Entry for %s: signal=%.4f, state=%s",
                    asset, signal_score, new_state.value
                )

                return FilterResult(
                    asset=asset,
                    original_signal=signal_score,
                    filtered_signal=signal_score,
                    position_state=pos.state,
                    action="entry",
                    holding_periods=pos.holding_periods,
                )
            else:
                # エントリー条件満たさない → シグナル抑制
                logger.debug(
                    "Suppress entry for %s: signal=%.4f < threshold=%.4f",
                    asset, abs(signal_score), self.entry_threshold
                )

                return FilterResult(
                    asset=asset,
                    original_signal=signal_score,
                    filtered_signal=0.0,
                    position_state=pos.state,
                    action="suppress",
                    holding_periods=0,
                )

        # 現在ポジションあり
        pos.holding_periods += 1
        pos.last_signal = signal_score
        pos.peak_signal = max(abs(pos.peak_signal), abs(signal_score)) * (
            1 if pos.peak_signal >= 0 else -1
        )

        # 最低保有期間チェック
        if pos.holding_periods < self.min_holding_periods:
            # 最低保有期間内 → 強制ホールド
            logger.debug(
                "Force hold for %s: periods=%d < min=%d",
                asset, pos.holding_periods, self.min_holding_periods
            )

            return FilterResult(
                asset=asset,
                original_signal=signal_score,
                filtered_signal=signal_score,
                position_state=pos.state,
                action="force_hold",
                holding_periods=pos.holding_periods,
            )

        # エグジット判定
        if not above_exit:
            # エグジット条件満たす
            old_state = pos.state
            pos.state = PositionState.NO_POSITION
            holding = pos.holding_periods
            pos.holding_periods = 0

            logger.debug(
                "Exit for %s: signal=%.4f < threshold=%.4f, held=%d periods",
                asset, abs(signal_score), self.exit_threshold, holding
            )

            return FilterResult(
                asset=asset,
                original_signal=signal_score,
                filtered_signal=0.0,
                position_state=PositionState.NO_POSITION,
                action="exit",
                holding_periods=holding,
            )

        # ホールド継続
        # シグナル反転チェック（ロングからショートへの転換など）
        if self.config.symmetric:
            current_is_long = signal_score > 0
            position_is_long = pos.state == PositionState.LONG

            if current_is_long != position_is_long:
                # 方向転換 → 一度エグジットしてからエントリー判定
                if above_entry:
                    # 即座に反転エントリー
                    old_state = pos.state
                    new_state = (
                        PositionState.LONG if current_is_long
                        else PositionState.SHORT
                    )
                    pos.state = new_state
                    pos.entry_signal = signal_score
                    pos.entry_time = timestamp
                    pos.holding_periods = 1
                    pos.peak_signal = signal_score

                    logger.debug(
                        "Flip position for %s: %s -> %s, signal=%.4f",
                        asset, old_state.value, new_state.value, signal_score
                    )

                    return FilterResult(
                        asset=asset,
                        original_signal=signal_score,
                        filtered_signal=signal_score,
                        position_state=pos.state,
                        action="flip",
                        holding_periods=pos.holding_periods,
                    )
                else:
                    # 反転エントリー条件満たさない → エグジット
                    pos.state = PositionState.NO_POSITION
                    holding = pos.holding_periods
                    pos.holding_periods = 0

                    return FilterResult(
                        asset=asset,
                        original_signal=signal_score,
                        filtered_signal=0.0,
                        position_state=PositionState.NO_POSITION,
                        action="exit_on_flip",
                        holding_periods=holding,
                    )

        # 通常ホールド
        logger.debug(
            "Hold for %s: signal=%.4f, periods=%d",
            asset, signal_score, pos.holding_periods
        )

        return FilterResult(
            asset=asset,
            original_signal=signal_score,
            filtered_signal=signal_score,
            position_state=pos.state,
            action="hold",
            holding_periods=pos.holding_periods,
        )

    def reset(self, asset: Optional[str] = None) -> None:
        """状態をリセット

        Args:
            asset: リセット対象のアセット。Noneの場合は全てリセット
        """
        if asset is None:
            self.positions.clear()
            logger.info("Reset all positions")
        elif asset in self.positions:
            del self.positions[asset]
            logger.info("Reset position for %s", asset)

    def get_position_info(self, asset: str) -> Optional[PositionInfo]:
        """ポジション情報を取得

        Args:
            asset: アセット識別子

        Returns:
            ポジション情報。存在しない場合はNone
        """
        return self.positions.get(asset)

    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """全ポジション情報を取得

        Returns:
            {asset: PositionInfo} の辞書
        """
        return self.positions.copy()

    def get_active_positions(self) -> List[str]:
        """アクティブなポジションを持つアセットのリスト

        Returns:
            ポジションを持つアセットのリスト
        """
        return [
            asset for asset, pos in self.positions.items()
            if pos.state != PositionState.NO_POSITION
        ]

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報の辞書
        """
        active = [
            pos for pos in self.positions.values()
            if pos.state != PositionState.NO_POSITION
        ]

        if not active:
            return {
                "n_total": len(self.positions),
                "n_active": 0,
                "n_long": 0,
                "n_short": 0,
                "avg_holding_periods": 0.0,
            }

        n_long = sum(1 for p in active if p.state == PositionState.LONG)
        n_short = sum(1 for p in active if p.state == PositionState.SHORT)
        avg_holding = np.mean([p.holding_periods for p in active])

        return {
            "n_total": len(self.positions),
            "n_active": len(active),
            "n_long": n_long,
            "n_short": n_short,
            "avg_holding_periods": avg_holding,
            "config": {
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
                "min_holding_periods": self.config.min_holding_periods,
            },
        }


class TimeSeriesHysteresis:
    """時系列ヒステリシスフィルター

    DataFrame形式のシグナルにヒステリシスを適用する。

    Usage:
        ts_hysteresis = TimeSeriesHysteresis(
            entry_threshold=0.3,
            exit_threshold=0.1,
        )
        filtered_df = ts_hysteresis.apply(signal_df)
    """

    def __init__(
        self,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.1,
        min_holding_periods: int = 5,
    ) -> None:
        """初期化

        Args:
            entry_threshold: エントリー閾値
            exit_threshold: エグジット閾値
            min_holding_periods: 最低保有期間
        """
        config = HysteresisConfig(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            min_holding_periods=min_holding_periods,
        )
        self.filter = HysteresisFilter(config)

    def apply(self, signals: pd.DataFrame) -> pd.DataFrame:
        """DataFrameにヒステリシスを適用

        Args:
            signals: シグナルのDataFrame（index=date, columns=assets）

        Returns:
            フィルター後のシグナルDataFrame
        """
        result = pd.DataFrame(index=signals.index, columns=signals.columns)

        for asset in signals.columns:
            self.filter.reset(asset)
            for i, (idx, row) in enumerate(signals.iterrows()):
                signal = row[asset]
                if pd.isna(signal):
                    result.loc[idx, asset] = np.nan
                else:
                    result.loc[idx, asset] = self.filter.filter(asset, signal)

        return result.astype(float)


# ============================================================
# 便利関数
# ============================================================

def apply_hysteresis(
    signals_dict: Dict[str, float],
    filter_instance: HysteresisFilter,
    timestamp: Optional[datetime] = None,
) -> Dict[str, float]:
    """シグナル辞書にヒステリシスを適用

    Args:
        signals_dict: {asset: signal_value} の辞書
        filter_instance: HysteresisFilterインスタンス
        timestamp: 現在時刻

    Returns:
        フィルター後のシグナル辞書
    """
    result: Dict[str, float] = {}
    for asset, signal in signals_dict.items():
        result[asset] = filter_instance.filter(asset, signal, timestamp)
    return result


def create_hysteresis_filter(
    entry_threshold: float = 0.3,
    exit_threshold: float = 0.1,
    min_holding_periods: int = 5,
) -> HysteresisFilter:
    """ヒステリシスフィルターを作成

    Args:
        entry_threshold: エントリー閾値
        exit_threshold: エグジット閾値
        min_holding_periods: 最低保有期間

    Returns:
        HysteresisFilter インスタンス
    """
    config = HysteresisConfig(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_holding_periods=min_holding_periods,
    )
    return HysteresisFilter(config)


def filter_signals_with_hysteresis(
    signals: pd.DataFrame,
    entry_threshold: float = 0.3,
    exit_threshold: float = 0.1,
    min_holding_periods: int = 5,
) -> pd.DataFrame:
    """時系列シグナルにヒステリシスを適用（便利関数）

    Args:
        signals: シグナルのDataFrame
        entry_threshold: エントリー閾値
        exit_threshold: エグジット閾値
        min_holding_periods: 最低保有期間

    Returns:
        フィルター後のシグナルDataFrame
    """
    ts_hysteresis = TimeSeriesHysteresis(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_holding_periods=min_holding_periods,
    )
    return ts_hysteresis.apply(signals)
