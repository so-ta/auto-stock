"""
Event-Driven Rebalance Scheduler

月次をベースに、特定イベント発生時に追加リバランスを行う。
週次のコスト（773回）と月次の機会損失（8.22%差）のバランスを取る。

主要トリガー:
1. PositionDeviationTrigger: 目標ウェイトから5%以上乖離
2. VIXSpikeTrigger: VIXが30%以上急騰
3. RegimeChangeTrigger: 市場レジーム変化検出

Usage:
    scheduler = EventDrivenRebalanceScheduler(
        base_frequency='monthly',
        min_interval_days=5,
    )

    for date in trading_days:
        if scheduler.should_rebalance(date, portfolio_state, market_data):
            # リバランス実行
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """トリガータイプの定義."""

    SCHEDULED = "scheduled"           # 定期リバランス
    POSITION_DEVIATION = "position_deviation"  # ウェイト乖離
    VIX_SPIKE = "vix_spike"          # VIX急騰
    REGIME_CHANGE = "regime_change"   # レジーム変化
    DRAWDOWN = "drawdown"            # ドローダウン閾値
    MOMENTUM_REVERSAL = "momentum_reversal"  # モメンタム反転


@dataclass
class PortfolioState:
    """ポートフォリオの現在状態."""

    current_weights: dict[str, float]
    target_weights: dict[str, float]
    portfolio_value: float
    last_rebalance_date: datetime | None = None
    peak_value: float = 0.0

    @property
    def max_deviation(self) -> float:
        """最大ウェイト乖離を計算."""
        all_symbols = set(self.current_weights.keys()) | set(self.target_weights.keys())
        max_dev = 0.0
        for symbol in all_symbols:
            current = self.current_weights.get(symbol, 0.0)
            target = self.target_weights.get(symbol, 0.0)
            max_dev = max(max_dev, abs(current - target))
        return max_dev

    @property
    def current_drawdown(self) -> float:
        """現在のドローダウンを計算."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.portfolio_value) / self.peak_value


@dataclass
class MarketData:
    """市場データ."""

    date: datetime
    vix: float | None = None
    vix_change: float | None = None  # 前日比変化率
    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None

    # レジーム情報
    current_regime: str | None = None  # "bull", "bear", "neutral"
    regime_probability: float | None = None


@dataclass
class TriggerResult:
    """トリガー判定結果."""

    triggered: bool
    trigger_type: TriggerType
    reason: str = ""
    severity: float = 0.0  # 0-1, 高いほど緊急
    metadata: dict[str, Any] = field(default_factory=dict)


class RebalanceTrigger(ABC):
    """リバランストリガーの基底クラス."""

    @property
    @abstractmethod
    def trigger_type(self) -> TriggerType:
        """トリガータイプを返す."""
        pass

    @abstractmethod
    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        """トリガー条件をチェック."""
        pass


class PositionDeviationTrigger(RebalanceTrigger):
    """
    ウェイト乖離トリガー.

    目標ウェイトから一定以上乖離した場合にトリガー。
    """

    def __init__(
        self,
        threshold: float = 0.05,
        critical_threshold: float = 0.10,
    ):
        """
        Args:
            threshold: 乖離閾値（デフォルト5%）
            critical_threshold: 緊急閾値（デフォルト10%）
        """
        self.threshold = threshold
        self.critical_threshold = critical_threshold

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.POSITION_DEVIATION

    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        max_dev = portfolio_state.max_deviation

        if max_dev >= self.critical_threshold:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Critical position deviation: {max_dev:.1%}",
                severity=1.0,
                metadata={"max_deviation": max_dev},
            )
        elif max_dev >= self.threshold:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Position deviation: {max_dev:.1%}",
                severity=0.5 + 0.5 * (max_dev - self.threshold) / (self.critical_threshold - self.threshold),
                metadata={"max_deviation": max_dev},
            )

        return TriggerResult(
            triggered=False,
            trigger_type=self.trigger_type,
            metadata={"max_deviation": max_dev},
        )


class VIXSpikeTrigger(RebalanceTrigger):
    """
    VIX急騰トリガー.

    VIXが急騰（前日比30%以上）した場合にトリガー。
    """

    def __init__(
        self,
        spike_threshold: float = 0.30,
        level_threshold: float = 30.0,
    ):
        """
        Args:
            spike_threshold: VIX変化率閾値（デフォルト30%）
            level_threshold: VIX絶対値閾値（デフォルト30）
        """
        self.spike_threshold = spike_threshold
        self.level_threshold = level_threshold

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.VIX_SPIKE

    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        if market_data.vix is None:
            return TriggerResult(
                triggered=False,
                trigger_type=self.trigger_type,
                reason="VIX data not available",
            )

        # VIX急騰チェック
        spike_triggered = False
        if market_data.vix_change is not None:
            spike_triggered = market_data.vix_change >= self.spike_threshold

        # VIXレベルチェック
        level_triggered = market_data.vix >= self.level_threshold

        if spike_triggered or level_triggered:
            severity = 0.0
            reasons = []

            if spike_triggered:
                severity = max(severity, 0.8)
                reasons.append(f"VIX spike: {market_data.vix_change:.1%}")

            if level_triggered:
                severity = max(severity, 0.6 + 0.4 * min((market_data.vix - self.level_threshold) / 20, 1.0))
                reasons.append(f"VIX level: {market_data.vix:.1f}")

            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=", ".join(reasons),
                severity=severity,
                metadata={
                    "vix": market_data.vix,
                    "vix_change": market_data.vix_change,
                },
            )

        return TriggerResult(
            triggered=False,
            trigger_type=self.trigger_type,
            metadata={
                "vix": market_data.vix,
                "vix_change": market_data.vix_change,
            },
        )


class RegimeChangeTrigger(RebalanceTrigger):
    """
    レジーム変化トリガー.

    市場レジームが変化した場合にトリガー。
    """

    def __init__(
        self,
        probability_threshold: float = 0.7,
    ):
        """
        Args:
            probability_threshold: レジーム確率閾値（デフォルト70%）
        """
        self.probability_threshold = probability_threshold
        self._previous_regime: str | None = None

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.REGIME_CHANGE

    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        if market_data.current_regime is None:
            return TriggerResult(
                triggered=False,
                trigger_type=self.trigger_type,
                reason="Regime data not available",
            )

        # レジーム変化検出
        regime_changed = (
            self._previous_regime is not None
            and self._previous_regime != market_data.current_regime
        )

        # 確率が十分高いかチェック
        high_confidence = (
            market_data.regime_probability is not None
            and market_data.regime_probability >= self.probability_threshold
        )

        # 前回レジームを更新
        old_regime = self._previous_regime
        self._previous_regime = market_data.current_regime

        if regime_changed and high_confidence:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Regime change: {old_regime} -> {market_data.current_regime}",
                severity=0.7,
                metadata={
                    "old_regime": old_regime,
                    "new_regime": market_data.current_regime,
                    "probability": market_data.regime_probability,
                },
            )

        return TriggerResult(
            triggered=False,
            trigger_type=self.trigger_type,
            metadata={
                "current_regime": market_data.current_regime,
                "probability": market_data.regime_probability,
            },
        )


class DrawdownTrigger(RebalanceTrigger):
    """
    ドローダウントリガー.

    ポートフォリオが一定以上のドローダウンに達した場合にトリガー。
    """

    def __init__(
        self,
        threshold: float = 0.10,
        critical_threshold: float = 0.15,
    ):
        """
        Args:
            threshold: ドローダウン閾値（デフォルト10%）
            critical_threshold: 緊急閾値（デフォルト15%）
        """
        self.threshold = threshold
        self.critical_threshold = critical_threshold

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.DRAWDOWN

    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        dd = portfolio_state.current_drawdown

        if dd >= self.critical_threshold:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Critical drawdown: {dd:.1%}",
                severity=1.0,
                metadata={"drawdown": dd},
            )
        elif dd >= self.threshold:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Drawdown alert: {dd:.1%}",
                severity=0.5 + 0.5 * (dd - self.threshold) / (self.critical_threshold - self.threshold),
                metadata={"drawdown": dd},
            )

        return TriggerResult(
            triggered=False,
            trigger_type=self.trigger_type,
            metadata={"drawdown": dd},
        )


class MomentumReversalTrigger(RebalanceTrigger):
    """
    モメンタム反転トリガー.

    短期モメンタムが急反転した場合にトリガー。
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_long: int = 20,
        reversal_threshold: float = 0.02,
    ):
        """
        Args:
            lookback_short: 短期ルックバック（デフォルト5日）
            lookback_long: 長期ルックバック（デフォルト20日）
            reversal_threshold: 反転閾値（デフォルト2%）
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.reversal_threshold = reversal_threshold

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.MOMENTUM_REVERSAL

    def check(
        self,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> TriggerResult:
        if market_data.returns is None or len(market_data.returns) < self.lookback_long:
            return TriggerResult(
                triggered=False,
                trigger_type=self.trigger_type,
                reason="Insufficient return data",
            )

        # ポートフォリオリターンを計算
        portfolio_returns = pd.Series(dtype=float)
        for symbol, weight in portfolio_state.current_weights.items():
            if symbol in market_data.returns.columns and weight > 0:
                portfolio_returns = portfolio_returns.add(
                    market_data.returns[symbol] * weight,
                    fill_value=0,
                )

        if len(portfolio_returns) < self.lookback_long:
            return TriggerResult(
                triggered=False,
                trigger_type=self.trigger_type,
                reason="Insufficient portfolio return data",
            )

        # 短期・長期モメンタム
        short_mom = portfolio_returns.iloc[-self.lookback_short:].mean()
        long_mom = portfolio_returns.iloc[-self.lookback_long:].mean()

        # 反転検出（符号が変わり、かつ大きな変化）
        reversal = np.sign(short_mom) != np.sign(long_mom) and abs(short_mom - long_mom) > self.reversal_threshold

        if reversal:
            return TriggerResult(
                triggered=True,
                trigger_type=self.trigger_type,
                reason=f"Momentum reversal: short={short_mom:.2%}, long={long_mom:.2%}",
                severity=0.6,
                metadata={
                    "short_momentum": short_mom,
                    "long_momentum": long_mom,
                },
            )

        return TriggerResult(
            triggered=False,
            trigger_type=self.trigger_type,
            metadata={
                "short_momentum": short_mom,
                "long_momentum": long_mom,
            },
        )


@dataclass
class RebalanceDecision:
    """リバランス判定結果."""

    should_rebalance: bool
    trigger_results: list[TriggerResult] = field(default_factory=list)
    primary_trigger: TriggerType | None = None
    combined_severity: float = 0.0

    @property
    def reason(self) -> str:
        """トリガー理由のサマリー."""
        triggered = [r for r in self.trigger_results if r.triggered]
        if not triggered:
            return "No rebalance needed"
        return "; ".join(r.reason for r in triggered if r.reason)


class EventDrivenRebalanceScheduler:
    """
    イベントドリブンリバランススケジューラ.

    月次をベースに、特定イベント発生時に追加リバランスを実行。
    これにより週次のコスト（773回）を削減しつつ、
    月次の機会損失（8.22%差）を軽減する。
    """

    def __init__(
        self,
        base_frequency: str = "monthly",
        min_interval_days: int = 5,
        triggers: list[RebalanceTrigger] | None = None,
        max_triggers_per_month: int = 3,
    ):
        """
        Args:
            base_frequency: ベース頻度（monthly/quarterly）
            min_interval_days: 最小リバランス間隔（日）
            triggers: カスタムトリガーリスト
            max_triggers_per_month: 月あたり最大トリガー回数
        """
        self.base_frequency = base_frequency
        self.min_interval_days = min_interval_days
        self.max_triggers_per_month = max_triggers_per_month

        # デフォルトトリガー
        self.triggers = triggers or [
            PositionDeviationTrigger(threshold=0.05),
            VIXSpikeTrigger(spike_threshold=0.30),
            RegimeChangeTrigger(),
            DrawdownTrigger(threshold=0.10),
        ]

        # 状態追跡
        self._last_rebalance_date: datetime | None = None
        self._trigger_count_this_month: int = 0
        self._current_month: tuple[int, int] | None = None

        # 統計
        self._stats = {
            "total_rebalances": 0,
            "scheduled_rebalances": 0,
            "triggered_rebalances": 0,
            "trigger_counts": {t.value: 0 for t in TriggerType},
        }

    def should_rebalance(
        self,
        date: datetime,
        portfolio_state: PortfolioState,
        market_data: MarketData,
    ) -> RebalanceDecision:
        """
        リバランスすべきか判定.

        Args:
            date: 判定日
            portfolio_state: ポートフォリオ状態
            market_data: 市場データ

        Returns:
            RebalanceDecision: リバランス判定結果
        """
        # 月が変わったらカウントリセット
        current_month = (date.year, date.month)
        if self._current_month != current_month:
            self._current_month = current_month
            self._trigger_count_this_month = 0

        trigger_results: list[TriggerResult] = []

        # 1. 定期リバランス（月末/四半期末）チェック
        is_scheduled = self._is_scheduled_date(date)
        if is_scheduled:
            trigger_results.append(TriggerResult(
                triggered=True,
                trigger_type=TriggerType.SCHEDULED,
                reason=f"Scheduled {self.base_frequency} rebalance",
                severity=0.3,
            ))

        # 2. 最小間隔チェック
        if not self._check_min_interval(date):
            return RebalanceDecision(
                should_rebalance=False,
                trigger_results=trigger_results,
            )

        # 3. 各トリガーをチェック
        for trigger in self.triggers:
            result = trigger.check(portfolio_state, market_data)
            trigger_results.append(result)

        # 4. トリガー判定
        triggered_results = [r for r in trigger_results if r.triggered]

        if not triggered_results:
            return RebalanceDecision(
                should_rebalance=False,
                trigger_results=trigger_results,
            )

        # 5. 月間トリガー上限チェック（定期以外）
        non_scheduled = [r for r in triggered_results if r.trigger_type != TriggerType.SCHEDULED]
        if non_scheduled and self._trigger_count_this_month >= self.max_triggers_per_month:
            # 緊急度が高い場合のみ許可
            max_severity = max(r.severity for r in non_scheduled)
            if max_severity < 0.9:
                return RebalanceDecision(
                    should_rebalance=False,
                    trigger_results=trigger_results,
                )

        # 6. 最も緊急度の高いトリガーを特定
        primary = max(triggered_results, key=lambda r: r.severity)
        combined_severity = max(r.severity for r in triggered_results)

        # 7. 状態更新
        self._last_rebalance_date = date
        if primary.trigger_type != TriggerType.SCHEDULED:
            self._trigger_count_this_month += 1

        # 8. 統計更新
        self._stats["total_rebalances"] += 1
        if primary.trigger_type == TriggerType.SCHEDULED:
            self._stats["scheduled_rebalances"] += 1
        else:
            self._stats["triggered_rebalances"] += 1
        self._stats["trigger_counts"][primary.trigger_type.value] += 1

        return RebalanceDecision(
            should_rebalance=True,
            trigger_results=trigger_results,
            primary_trigger=primary.trigger_type,
            combined_severity=combined_severity,
        )

    def _is_scheduled_date(self, date: datetime) -> bool:
        """定期リバランス日かどうか."""
        if self.base_frequency == "monthly":
            return self._is_month_end(date)
        elif self.base_frequency == "quarterly":
            return self._is_quarter_end(date)
        elif self.base_frequency == "weekly":
            return date.weekday() == 4  # 金曜日
        return False

    def _is_month_end(self, date: datetime) -> bool:
        """月末判定（営業日ベースは呼び出し側で）."""
        next_day = date + timedelta(days=1)
        return date.month != next_day.month

    def _is_quarter_end(self, date: datetime) -> bool:
        """四半期末判定."""
        return self._is_month_end(date) and date.month in [3, 6, 9, 12]

    def _check_min_interval(self, date: datetime) -> bool:
        """最小間隔チェック."""
        if self._last_rebalance_date is None:
            return True
        days_since = (date - self._last_rebalance_date).days
        return days_since >= self.min_interval_days

    def get_stats(self) -> dict[str, Any]:
        """統計情報を取得."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """統計情報をリセット."""
        self._stats = {
            "total_rebalances": 0,
            "scheduled_rebalances": 0,
            "triggered_rebalances": 0,
            "trigger_counts": {t.value: 0 for t in TriggerType},
        }
        self._last_rebalance_date = None
        self._trigger_count_this_month = 0
        self._current_month = None


def create_default_scheduler(
    position_threshold: float = 0.05,
    vix_threshold: float = 0.30,
    drawdown_threshold: float = 0.10,
    min_interval_days: int = 5,
) -> EventDrivenRebalanceScheduler:
    """
    デフォルト設定のスケジューラを作成.

    Args:
        position_threshold: ウェイト乖離閾値
        vix_threshold: VIX急騰閾値
        drawdown_threshold: ドローダウン閾値
        min_interval_days: 最小間隔

    Returns:
        EventDrivenRebalanceScheduler
    """
    triggers = [
        PositionDeviationTrigger(threshold=position_threshold),
        VIXSpikeTrigger(spike_threshold=vix_threshold),
        RegimeChangeTrigger(),
        DrawdownTrigger(threshold=drawdown_threshold),
        MomentumReversalTrigger(),
    ]

    return EventDrivenRebalanceScheduler(
        base_frequency="monthly",
        min_interval_days=min_interval_days,
        triggers=triggers,
    )
