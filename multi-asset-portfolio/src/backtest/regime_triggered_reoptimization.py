"""
Regime Triggered Reoptimization Module - レジーム変化での再最適化

市場レジームの変化を検出し、必要に応じてポートフォリオの
再最適化をトリガーする。

設計根拠:
- 要求.md §6: リスク管理
- レジーム変化への適応
- 過剰な再最適化を防ぐクールダウン機構

主なトリガー:
- ボラティリティ急変
- 相関構造の変化
- VIXスパイク
- パフォーマンス劣化（ドローダウン）

推奨アクション:
- full_reoptimization: 全パラメータ再最適化
- covariance_reestimation: 共分散のみ再推定
- risk_reduction: キャッシュ増加
- parameter_adjustment: パラメータ微調整
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Severity(Enum):
    """トリガー深刻度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendedAction(Enum):
    """推奨アクション"""
    NONE = "none"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    COVARIANCE_REESTIMATION = "covariance_reestimation"
    RISK_REDUCTION = "risk_reduction"
    FULL_REOPTIMIZATION = "full_reoptimization"


@dataclass
class ReoptimizationTrigger:
    """再最適化トリガー結果

    Attributes:
        triggered: トリガーされたかどうか
        reason: トリガー理由
        severity: 深刻度（low/medium/high/critical）
        recommended_action: 推奨アクション
        details: 詳細情報
        timestamp: トリガー時刻
    """
    triggered: bool
    reason: str
    severity: str = "low"
    recommended_action: str = "none"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class TriggerCondition:
    """トリガー条件

    Attributes:
        name: 条件名
        triggered: 条件が満たされたか
        current_value: 現在値
        threshold: 閾値
        severity: 深刻度
        action: 推奨アクション
    """
    name: str
    triggered: bool
    current_value: float
    threshold: float
    severity: Severity
    action: RecommendedAction


@dataclass(frozen=True)
class ReoptimizationConfig:
    """再最適化設定

    Attributes:
        vol_change_threshold: ボラティリティ変化閾値（50%増加で発動）
        corr_change_threshold: 相関変化閾値（0.3以上で発動）
        performance_threshold: パフォーマンス閾値（-10%DDで発動）
        vix_spike_threshold: VIXスパイク閾値
        cooldown_days: クールダウン期間（日）
        vol_lookback: ボラティリティ計算ルックバック
        corr_lookback: 相関計算ルックバック
    """
    vol_change_threshold: float = 0.5
    corr_change_threshold: float = 0.3
    performance_threshold: float = -0.10
    vix_spike_threshold: float = 30.0
    cooldown_days: int = 10
    vol_lookback: int = 20
    corr_lookback: int = 60


@dataclass
class ReoptimizationResult:
    """再最適化結果

    Attributes:
        action_taken: 実行されたアクション
        success: 成功したか
        old_params: 旧パラメータ
        new_params: 新パラメータ
        execution_time: 実行時間（秒）
        details: 詳細情報
    """
    action_taken: str
    success: bool
    old_params: Dict[str, Any] = field(default_factory=dict)
    new_params: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "action_taken": self.action_taken,
            "success": self.success,
            "old_params": self.old_params,
            "new_params": self.new_params,
            "execution_time": self.execution_time,
            "details": self.details,
        }


class RegimeTriggeredReoptimizer:
    """レジームトリガー再最適化クラス

    市場レジームの変化を検出し、必要に応じて再最適化をトリガーする。

    Usage:
        reoptimizer = RegimeTriggeredReoptimizer(
            vol_change_threshold=0.5,
            vix_spike_threshold=30,
            cooldown_days=10,
        )

        # トリガーチェック
        trigger = reoptimizer.check_triggers(
            returns=returns_df,
            vix_current=25.5,
            portfolio_drawdown=-0.08,
            current_date=datetime.now(),
        )

        if trigger.triggered:
            result = reoptimizer.execute_reoptimization(
                trigger=trigger,
                optimize_func=my_optimizer,
                portfolio=my_portfolio,
            )

    Attributes:
        config: 再最適化設定
    """

    def __init__(
        self,
        vol_change_threshold: float = 0.5,
        corr_change_threshold: float = 0.3,
        performance_threshold: float = -0.10,
        vix_spike_threshold: float = 30.0,
        cooldown_days: int = 10,
        **kwargs: Any,
    ) -> None:
        """初期化

        Args:
            vol_change_threshold: ボラティリティ変化閾値
            corr_change_threshold: 相関変化閾値
            performance_threshold: パフォーマンス閾値
            vix_spike_threshold: VIXスパイク閾値
            cooldown_days: クールダウン期間
            **kwargs: その他のConfig設定
        """
        self.config = ReoptimizationConfig(
            vol_change_threshold=vol_change_threshold,
            corr_change_threshold=corr_change_threshold,
            performance_threshold=performance_threshold,
            vix_spike_threshold=vix_spike_threshold,
            cooldown_days=cooldown_days,
            **kwargs,
        )

        self._last_reoptimization: Optional[datetime] = None
        self._trigger_history: List[ReoptimizationTrigger] = []
        self._baseline_vol: Optional[float] = None
        self._baseline_corr: Optional[pd.DataFrame] = None

    def _check_cooldown(self, current_date: datetime) -> bool:
        """クールダウン期間中かチェック

        Args:
            current_date: 現在日時

        Returns:
            クールダウン中ならTrue
        """
        if self._last_reoptimization is None:
            return False

        elapsed = current_date - self._last_reoptimization
        return elapsed.days < self.config.cooldown_days

    def _check_volatility_change(
        self,
        returns: pd.DataFrame,
    ) -> TriggerCondition:
        """ボラティリティ変化をチェック

        Args:
            returns: リターンデータ

        Returns:
            トリガー条件結果
        """
        lookback = self.config.vol_lookback

        if len(returns) < lookback * 2:
            return TriggerCondition(
                name="volatility_change",
                triggered=False,
                current_value=0.0,
                threshold=self.config.vol_change_threshold,
                severity=Severity.LOW,
                action=RecommendedAction.NONE,
            )

        # 現在のボラティリティ
        current_vol = returns.iloc[-lookback:].std().mean() * np.sqrt(252)

        # 過去のボラティリティ（基準）
        past_vol = returns.iloc[-lookback * 2:-lookback].std().mean() * np.sqrt(252)

        if past_vol > 0:
            vol_change = (current_vol - past_vol) / past_vol
        else:
            vol_change = 0.0

        # ベースラインを更新
        if self._baseline_vol is None:
            self._baseline_vol = past_vol

        triggered = abs(vol_change) > self.config.vol_change_threshold

        # 深刻度とアクションを決定
        if abs(vol_change) > self.config.vol_change_threshold * 2:
            severity = Severity.HIGH
            action = RecommendedAction.FULL_REOPTIMIZATION
        elif abs(vol_change) > self.config.vol_change_threshold:
            severity = Severity.MEDIUM
            action = RecommendedAction.COVARIANCE_REESTIMATION
        else:
            severity = Severity.LOW
            action = RecommendedAction.NONE

        return TriggerCondition(
            name="volatility_change",
            triggered=triggered,
            current_value=vol_change,
            threshold=self.config.vol_change_threshold,
            severity=severity,
            action=action,
        )

    def _check_correlation_change(
        self,
        returns: pd.DataFrame,
    ) -> TriggerCondition:
        """相関構造変化をチェック

        Args:
            returns: リターンデータ

        Returns:
            トリガー条件結果
        """
        lookback = self.config.corr_lookback

        if len(returns) < lookback * 2 or returns.shape[1] < 2:
            return TriggerCondition(
                name="correlation_change",
                triggered=False,
                current_value=0.0,
                threshold=self.config.corr_change_threshold,
                severity=Severity.LOW,
                action=RecommendedAction.NONE,
            )

        # 現在の相関行列
        current_corr = returns.iloc[-lookback:].corr()

        # 過去の相関行列
        past_corr = returns.iloc[-lookback * 2:-lookback].corr()

        # 相関行列の変化（フロベニウスノルム）
        corr_diff = current_corr - past_corr
        corr_change = np.sqrt((corr_diff ** 2).sum().sum()) / (corr_diff.shape[0] ** 2)

        triggered = corr_change > self.config.corr_change_threshold

        # 深刻度とアクションを決定
        if corr_change > self.config.corr_change_threshold * 2:
            severity = Severity.HIGH
            action = RecommendedAction.FULL_REOPTIMIZATION
        elif corr_change > self.config.corr_change_threshold:
            severity = Severity.MEDIUM
            action = RecommendedAction.COVARIANCE_REESTIMATION
        else:
            severity = Severity.LOW
            action = RecommendedAction.NONE

        return TriggerCondition(
            name="correlation_change",
            triggered=triggered,
            current_value=corr_change,
            threshold=self.config.corr_change_threshold,
            severity=severity,
            action=action,
        )

    def _check_vix_spike(
        self,
        vix_current: float,
    ) -> TriggerCondition:
        """VIXスパイクをチェック

        Args:
            vix_current: 現在のVIX値

        Returns:
            トリガー条件結果
        """
        triggered = vix_current > self.config.vix_spike_threshold

        # 深刻度とアクションを決定
        if vix_current > self.config.vix_spike_threshold * 1.5:
            severity = Severity.CRITICAL
            action = RecommendedAction.RISK_REDUCTION
        elif vix_current > self.config.vix_spike_threshold:
            severity = Severity.HIGH
            action = RecommendedAction.RISK_REDUCTION
        else:
            severity = Severity.LOW
            action = RecommendedAction.NONE

        return TriggerCondition(
            name="vix_spike",
            triggered=triggered,
            current_value=vix_current,
            threshold=self.config.vix_spike_threshold,
            severity=severity,
            action=action,
        )

    def _check_performance_degradation(
        self,
        portfolio_drawdown: float,
    ) -> TriggerCondition:
        """パフォーマンス劣化をチェック

        Args:
            portfolio_drawdown: 現在のドローダウン（負の値）

        Returns:
            トリガー条件結果
        """
        triggered = portfolio_drawdown < self.config.performance_threshold

        # 深刻度とアクションを決定
        if portfolio_drawdown < self.config.performance_threshold * 2:
            severity = Severity.CRITICAL
            action = RecommendedAction.FULL_REOPTIMIZATION
        elif portfolio_drawdown < self.config.performance_threshold:
            severity = Severity.HIGH
            action = RecommendedAction.PARAMETER_ADJUSTMENT
        else:
            severity = Severity.LOW
            action = RecommendedAction.NONE

        return TriggerCondition(
            name="performance_degradation",
            triggered=triggered,
            current_value=portfolio_drawdown,
            threshold=self.config.performance_threshold,
            severity=severity,
            action=action,
        )

    def check_triggers(
        self,
        returns: pd.DataFrame,
        vix_current: float,
        portfolio_drawdown: float,
        current_date: Optional[datetime] = None,
    ) -> ReoptimizationTrigger:
        """全トリガーをチェック

        Args:
            returns: リターンデータ
            vix_current: 現在のVIX値
            portfolio_drawdown: 現在のドローダウン
            current_date: 現在日時

        Returns:
            ReoptimizationTrigger: トリガー結果
        """
        current_date = current_date or datetime.now()

        # クールダウンチェック
        if self._check_cooldown(current_date):
            cooldown_remaining = (
                self._last_reoptimization + timedelta(days=self.config.cooldown_days)
                - current_date
            ).days

            return ReoptimizationTrigger(
                triggered=False,
                reason=f"Cooldown period ({cooldown_remaining} days remaining)",
                severity="low",
                recommended_action="none",
                details={"cooldown_remaining_days": cooldown_remaining},
                timestamp=current_date,
            )

        # 各条件をチェック
        conditions: List[TriggerCondition] = [
            self._check_volatility_change(returns),
            self._check_correlation_change(returns),
            self._check_vix_spike(vix_current),
            self._check_performance_degradation(portfolio_drawdown),
        ]

        # トリガーされた条件を収集
        triggered_conditions = [c for c in conditions if c.triggered]

        if not triggered_conditions:
            return ReoptimizationTrigger(
                triggered=False,
                reason="No triggers activated",
                severity="low",
                recommended_action="none",
                details={
                    "conditions_checked": [c.name for c in conditions],
                },
                timestamp=current_date,
            )

        # 最も深刻な条件を選択
        severity_order = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }
        triggered_conditions.sort(
            key=lambda c: severity_order[c.severity],
            reverse=True,
        )
        primary_condition = triggered_conditions[0]

        # 複数条件がトリガーされた場合のアクション決定
        if len(triggered_conditions) >= 2:
            # 複数条件 → より強いアクションを推奨
            recommended_action = RecommendedAction.FULL_REOPTIMIZATION
        else:
            recommended_action = primary_condition.action

        # 理由の構築
        reasons = [f"{c.name} ({c.current_value:.4f} vs {c.threshold:.4f})"
                   for c in triggered_conditions]

        trigger = ReoptimizationTrigger(
            triggered=True,
            reason="; ".join(reasons),
            severity=primary_condition.severity.value,
            recommended_action=recommended_action.value,
            details={
                "triggered_conditions": [c.name for c in triggered_conditions],
                "all_conditions": {
                    c.name: {
                        "triggered": c.triggered,
                        "value": c.current_value,
                        "threshold": c.threshold,
                    }
                    for c in conditions
                },
            },
            timestamp=current_date,
        )

        # 履歴に追加
        self._trigger_history.append(trigger)

        logger.warning(
            "Reoptimization triggered: %s (severity=%s, action=%s)",
            trigger.reason,
            trigger.severity,
            trigger.recommended_action,
        )

        return trigger

    def execute_reoptimization(
        self,
        trigger: ReoptimizationTrigger,
        optimize_func: Callable[..., Dict[str, Any]],
        **kwargs: Any,
    ) -> ReoptimizationResult:
        """再最適化を実行

        Args:
            trigger: トリガー結果
            optimize_func: 最適化関数
            **kwargs: 最適化関数に渡すパラメータ

        Returns:
            ReoptimizationResult: 再最適化結果
        """
        import time
        start_time = time.time()

        if not trigger.triggered:
            return ReoptimizationResult(
                action_taken="none",
                success=True,
                details={"message": "No reoptimization needed"},
            )

        action = trigger.recommended_action
        old_params = kwargs.get("current_params", {})

        try:
            if action == "full_reoptimization":
                # 全パラメータ再最適化
                new_params = optimize_func(
                    mode="full",
                    **kwargs,
                )
            elif action == "covariance_reestimation":
                # 共分散のみ再推定
                new_params = optimize_func(
                    mode="covariance_only",
                    **kwargs,
                )
            elif action == "risk_reduction":
                # リスク削減（キャッシュ増加）
                new_params = self._apply_risk_reduction(old_params, **kwargs)
            elif action == "parameter_adjustment":
                # パラメータ微調整
                new_params = optimize_func(
                    mode="incremental",
                    **kwargs,
                )
            else:
                new_params = old_params

            execution_time = time.time() - start_time

            # 最終再最適化日時を更新
            self._last_reoptimization = trigger.timestamp

            result = ReoptimizationResult(
                action_taken=action,
                success=True,
                old_params=old_params,
                new_params=new_params,
                execution_time=execution_time,
                details={
                    "trigger_reason": trigger.reason,
                    "trigger_severity": trigger.severity,
                },
            )

            logger.info(
                "Reoptimization completed: action=%s, time=%.2fs",
                action, execution_time
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Reoptimization failed: %s", e)

            return ReoptimizationResult(
                action_taken=action,
                success=False,
                old_params=old_params,
                execution_time=execution_time,
                details={"error": str(e)},
            )

    def _apply_risk_reduction(
        self,
        current_params: Dict[str, Any],
        cash_increase_pct: float = 0.20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """リスク削減を適用

        Args:
            current_params: 現在のパラメータ
            cash_increase_pct: キャッシュ増加率

        Returns:
            更新されたパラメータ
        """
        new_params = current_params.copy()

        # ウェイトをスケールダウン
        if "weights" in new_params:
            weights = new_params["weights"]
            if isinstance(weights, dict):
                scale = 1 - cash_increase_pct
                new_params["weights"] = {
                    k: v * scale for k, v in weights.items()
                }

        # キャッシュ比率を増加
        if "cash_weight" in new_params:
            new_params["cash_weight"] = min(
                1.0,
                new_params["cash_weight"] + cash_increase_pct,
            )

        # ターゲットボラティリティを低下
        if "target_volatility" in new_params:
            new_params["target_volatility"] = (
                new_params["target_volatility"] * 0.8
            )

        return new_params

    def reset_cooldown(self) -> None:
        """クールダウンをリセット"""
        self._last_reoptimization = None
        logger.info("Cooldown reset")

    def get_trigger_history(self) -> List[Dict[str, Any]]:
        """トリガー履歴を取得

        Returns:
            トリガー履歴のリスト
        """
        return [t.to_dict() for t in self._trigger_history]

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報の辞書
        """
        triggered_count = sum(1 for t in self._trigger_history if t.triggered)

        return {
            "total_checks": len(self._trigger_history),
            "triggered_count": triggered_count,
            "trigger_rate": (
                triggered_count / len(self._trigger_history)
                if self._trigger_history else 0.0
            ),
            "last_reoptimization": (
                self._last_reoptimization.isoformat()
                if self._last_reoptimization else None
            ),
            "config": {
                "vol_change_threshold": self.config.vol_change_threshold,
                "corr_change_threshold": self.config.corr_change_threshold,
                "performance_threshold": self.config.performance_threshold,
                "vix_spike_threshold": self.config.vix_spike_threshold,
                "cooldown_days": self.config.cooldown_days,
            },
        }


# ============================================================
# 便利関数
# ============================================================

def create_regime_reoptimizer(
    vol_change_threshold: float = 0.5,
    vix_spike_threshold: float = 30.0,
    cooldown_days: int = 10,
    **kwargs: Any,
) -> RegimeTriggeredReoptimizer:
    """RegimeTriggeredReoptimizerを作成

    Args:
        vol_change_threshold: ボラティリティ変化閾値
        vix_spike_threshold: VIXスパイク閾値
        cooldown_days: クールダウン期間
        **kwargs: その他のパラメータ

    Returns:
        RegimeTriggeredReoptimizer インスタンス
    """
    return RegimeTriggeredReoptimizer(
        vol_change_threshold=vol_change_threshold,
        vix_spike_threshold=vix_spike_threshold,
        cooldown_days=cooldown_days,
        **kwargs,
    )


def check_regime_change(
    returns: pd.DataFrame,
    vix_current: float = 20.0,
    portfolio_drawdown: float = 0.0,
) -> ReoptimizationTrigger:
    """レジーム変化をチェック（ワンライナー）

    Args:
        returns: リターンデータ
        vix_current: 現在のVIX値
        portfolio_drawdown: 現在のドローダウン

    Returns:
        ReoptimizationTrigger
    """
    reoptimizer = create_regime_reoptimizer()
    return reoptimizer.check_triggers(
        returns=returns,
        vix_current=vix_current,
        portfolio_drawdown=portfolio_drawdown,
    )
