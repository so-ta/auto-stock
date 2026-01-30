"""
Parameter Consistency Checker - パラメータ間整合性制約

このモジュールは、ポートフォリオ最適化パラメータ間の整合性を
チェックし、必要に応じて自動調整する機能を提供する。

主要機能:
- ParameterConsistencyChecker: パラメータ整合性チェック・調整
- check_consistency: 整合性チェック（bool, issues）
- ensure_consistency: 整合性を保証する調整

設計根拠:
- パラメータ間には暗黙の制約が存在する
- 不整合なパラメータは予期せぬ動作を引き起こす
- 自動調整により運用の安定性を向上
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ConsistencyLevel(str, Enum):
    """整合性問題の深刻度レベル。"""

    ERROR = "error"  # 動作不能
    WARNING = "warning"  # 動作可能だが非推奨
    INFO = "info"  # 情報提供のみ


@dataclass
class ConsistencyIssue:
    """整合性問題を表すデータクラス。

    Attributes:
        level: 深刻度レベル
        rule_id: ルールの識別子
        message: 問題の説明
        param_names: 関連するパラメータ名
        original_values: 元の値
        adjusted_values: 調整後の値（調整された場合）
    """

    level: ConsistencyLevel
    rule_id: str
    message: str
    param_names: list[str]
    original_values: dict[str, Any] = field(default_factory=dict)
    adjusted_values: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.rule_id}: {self.message}"


@dataclass
class ConsistencyResult:
    """整合性チェック結果。

    Attributes:
        is_consistent: 整合性があるか
        issues: 検出された問題のリスト
        adjusted_params: 調整後のパラメータ（調整された場合）
        original_params: 元のパラメータ
    """

    is_consistent: bool
    issues: list[ConsistencyIssue]
    adjusted_params: dict[str, Any] | None = None
    original_params: dict[str, Any] | None = None

    @property
    def errors(self) -> list[ConsistencyIssue]:
        """ERRORレベルの問題のみを返す。"""
        return [i for i in self.issues if i.level == ConsistencyLevel.ERROR]

    @property
    def warnings(self) -> list[ConsistencyIssue]:
        """WARNINGレベルの問題のみを返す。"""
        return [i for i in self.issues if i.level == ConsistencyLevel.WARNING]

    def has_errors(self) -> bool:
        """ERRORレベルの問題があるか。"""
        return len(self.errors) > 0


class ParameterConsistencyChecker:
    """パラメータ整合性チェッククラス。

    ポートフォリオ最適化パラメータ間の整合性をチェックし、
    必要に応じて自動調整する。

    Usage:
        checker = ParameterConsistencyChecker()

        # バリデーションのみ
        issues = checker.validate(params)

        # チェックと調整
        adjusted = checker.check_and_adjust(params)

        # 詳細な結果を取得
        result = checker.check(params, auto_adjust=True)
    """

    # デフォルトのパラメータ値（整合性チェック用）
    DEFAULT_PARAMS = {
        "top_n": 5,
        "w_asset_max": 0.25,
        "smooth_alpha": 0.3,
        "delta_max": 0.1,
        "beta": 2.0,
        "entropy_min": 0.5,
        "min_active_strategies": 2,
        "rebalance_threshold": 0.05,
    }

    def __init__(
        self,
        strict_mode: bool = False,
        custom_rules: list[Callable[[dict], ConsistencyIssue | None]] | None = None,
    ) -> None:
        """初期化。

        Args:
            strict_mode: Trueの場合、WARNINGもERRORとして扱う
            custom_rules: カスタムルール関数のリスト
        """
        self.strict_mode = strict_mode
        self.custom_rules = custom_rules or []
        self._rules = self._build_rules()

    def _build_rules(self) -> list[Callable[[dict], ConsistencyIssue | None]]:
        """組み込みルールを構築する。"""
        return [
            self._rule_topn_wassetmax,
            self._rule_smooth_delta,
            self._rule_beta_entropy,
            self._rule_min_strategies,
            self._rule_rebalance_threshold,
            self._rule_weight_sum,
        ] + self.custom_rules

    def validate(self, params: dict[str, Any]) -> list[str]:
        """パラメータの整合性を検証し、問題点をリストで返す。

        Args:
            params: 検証するパラメータ

        Returns:
            検出された問題のメッセージリスト
        """
        result = self.check(params, auto_adjust=False)
        return [str(issue) for issue in result.issues]

    def check(
        self,
        params: dict[str, Any],
        auto_adjust: bool = False,
    ) -> ConsistencyResult:
        """パラメータの整合性をチェックする。

        Args:
            params: チェックするパラメータ
            auto_adjust: 自動調整を行うか

        Returns:
            整合性チェック結果
        """
        # デフォルト値でマージ
        full_params = {**self.DEFAULT_PARAMS, **params}
        issues: list[ConsistencyIssue] = []
        adjusted = full_params.copy()

        # 各ルールを適用
        for rule in self._rules:
            issue = rule(adjusted)
            if issue is not None:
                issues.append(issue)

                # 自動調整
                if auto_adjust and issue.adjusted_values:
                    for key, value in issue.adjusted_values.items():
                        adjusted[key] = value
                    logger.info(
                        f"Auto-adjusted params: {issue.adjusted_values} "
                        f"(rule: {issue.rule_id})"
                    )

        # 厳格モードではWARNINGもERRORとして扱う
        if self.strict_mode:
            for issue in issues:
                if issue.level == ConsistencyLevel.WARNING:
                    issue.level = ConsistencyLevel.ERROR

        is_consistent = all(
            issue.level != ConsistencyLevel.ERROR for issue in issues
        )

        return ConsistencyResult(
            is_consistent=is_consistent,
            issues=issues,
            adjusted_params=adjusted if auto_adjust else None,
            original_params=params,
        )

    def check_and_adjust(self, params: dict[str, Any]) -> dict[str, Any]:
        """パラメータをチェックし、整合性を保つよう調整する。

        Args:
            params: 調整するパラメータ

        Returns:
            調整後のパラメータ
        """
        result = self.check(params, auto_adjust=True)
        return result.adjusted_params or params

    # ==================== ルール実装 ====================

    def _rule_topn_wassetmax(self, params: dict) -> ConsistencyIssue | None:
        """ルール1: top_n と w_asset_max の整合性。

        top_n * w_asset_max >= 1.0 が必要。
        そうでないと、全資産を100%配分できない。
        """
        top_n = params.get("top_n", 5)
        w_asset_max = params.get("w_asset_max", 0.25)

        product = top_n * w_asset_max

        if product < 1.0:
            # 調整: w_asset_max を増やす
            adjusted_w_asset_max = 1.0 / top_n + 0.05

            return ConsistencyIssue(
                level=ConsistencyLevel.ERROR,
                rule_id="TOPN_WASSETMAX",
                message=(
                    f"top_n({top_n}) * w_asset_max({w_asset_max:.3f}) = {product:.3f} < 1.0. "
                    f"Cannot allocate 100% of portfolio."
                ),
                param_names=["top_n", "w_asset_max"],
                original_values={"top_n": top_n, "w_asset_max": w_asset_max},
                adjusted_values={"w_asset_max": adjusted_w_asset_max},
            )

        return None

    def _rule_smooth_delta(self, params: dict) -> ConsistencyIssue | None:
        """ルール2: smooth_alpha と delta_max の整合性。

        effective_change = smooth_alpha * delta_max >= 0.01 が望ましい。
        そうでないと、リバランスが事実上無効になる。
        """
        smooth_alpha = params.get("smooth_alpha", 0.3)
        delta_max = params.get("delta_max", 0.1)

        effective_change = smooth_alpha * delta_max

        if effective_change < 0.01:
            # 調整: smooth_alpha を増やす
            adjusted_smooth_alpha = max(smooth_alpha, 0.01 / delta_max)

            return ConsistencyIssue(
                level=ConsistencyLevel.WARNING,
                rule_id="SMOOTH_DELTA",
                message=(
                    f"effective_change = smooth_alpha({smooth_alpha:.3f}) * "
                    f"delta_max({delta_max:.3f}) = {effective_change:.4f} < 0.01. "
                    f"Rebalancing may be ineffective."
                ),
                param_names=["smooth_alpha", "delta_max"],
                original_values={"smooth_alpha": smooth_alpha, "delta_max": delta_max},
                adjusted_values={"smooth_alpha": adjusted_smooth_alpha},
            )

        return None

    def _rule_beta_entropy(self, params: dict) -> ConsistencyIssue | None:
        """ルール3: beta と entropy_min の整合性。

        高beta（集中）と高entropy_min（分散）は矛盾する。
        """
        beta = params.get("beta", 2.0)
        entropy_min = params.get("entropy_min", 0.5)

        if beta > 3.0 and entropy_min > 0.8:
            # 調整: beta を下げる
            adjusted_beta = min(beta, 2.5)

            return ConsistencyIssue(
                level=ConsistencyLevel.WARNING,
                rule_id="BETA_ENTROPY",
                message=(
                    f"High beta({beta:.2f}) conflicts with high entropy_min({entropy_min:.2f}). "
                    f"Beta favors concentration while entropy_min requires diversification."
                ),
                param_names=["beta", "entropy_min"],
                original_values={"beta": beta, "entropy_min": entropy_min},
                adjusted_values={"beta": adjusted_beta},
            )

        return None

    def _rule_min_strategies(self, params: dict) -> ConsistencyIssue | None:
        """ルール4: 戦略数と最小配分の整合性。

        min_active_strategies <= top_n が必要。
        """
        min_active = params.get("min_active_strategies", 2)
        top_n = params.get("top_n", 5)

        if min_active > top_n:
            # 調整: min_active_strategies を下げる
            adjusted_min_active = top_n

            return ConsistencyIssue(
                level=ConsistencyLevel.ERROR,
                rule_id="MIN_STRATEGIES",
                message=(
                    f"min_active_strategies({min_active}) > top_n({top_n}). "
                    f"Cannot activate more strategies than selected."
                ),
                param_names=["min_active_strategies", "top_n"],
                original_values={"min_active_strategies": min_active, "top_n": top_n},
                adjusted_values={"min_active_strategies": adjusted_min_active},
            )

        return None

    def _rule_rebalance_threshold(self, params: dict) -> ConsistencyIssue | None:
        """ルール5: リバランス閾値の妥当性。

        rebalance_threshold が極端に小さいまたは大きい場合に警告。
        """
        threshold = params.get("rebalance_threshold", 0.05)

        if threshold < 0.01:
            return ConsistencyIssue(
                level=ConsistencyLevel.WARNING,
                rule_id="REBALANCE_THRESHOLD_LOW",
                message=(
                    f"rebalance_threshold({threshold:.3f}) is very low. "
                    f"May cause excessive trading."
                ),
                param_names=["rebalance_threshold"],
                original_values={"rebalance_threshold": threshold},
                adjusted_values={"rebalance_threshold": 0.02},
            )

        if threshold > 0.30:
            return ConsistencyIssue(
                level=ConsistencyLevel.INFO,
                rule_id="REBALANCE_THRESHOLD_HIGH",
                message=(
                    f"rebalance_threshold({threshold:.3f}) is high. "
                    f"Portfolio may drift significantly before rebalancing."
                ),
                param_names=["rebalance_threshold"],
                original_values={"rebalance_threshold": threshold},
                adjusted_values=None,  # 情報のみ、調整なし
            )

        return None

    def _rule_weight_sum(self, params: dict) -> ConsistencyIssue | None:
        """ルール6: 戦略重みの合計チェック。

        trend_weight + reversion_weight + macro_weight が存在する場合、
        極端に偏っていないかチェック。
        """
        trend = params.get("trend_weight")
        reversion = params.get("reversion_weight")
        macro = params.get("macro_weight")

        if trend is None and reversion is None and macro is None:
            return None

        # 存在する重みのみをチェック
        weights = {}
        if trend is not None:
            weights["trend_weight"] = trend
        if reversion is not None:
            weights["reversion_weight"] = reversion
        if macro is not None:
            weights["macro_weight"] = macro

        total = sum(weights.values())

        # 合計が極端に小さいまたは大きい場合
        if total < 0.5:
            return ConsistencyIssue(
                level=ConsistencyLevel.WARNING,
                rule_id="WEIGHT_SUM_LOW",
                message=(
                    f"Sum of strategy weights ({total:.3f}) is low. "
                    f"Consider increasing weights for better signal utilization."
                ),
                param_names=list(weights.keys()),
                original_values=weights,
                adjusted_values=None,
            )

        if total > 1.5:
            return ConsistencyIssue(
                level=ConsistencyLevel.INFO,
                rule_id="WEIGHT_SUM_HIGH",
                message=(
                    f"Sum of strategy weights ({total:.3f}) is high. "
                    f"Signals will be amplified, which may increase volatility."
                ),
                param_names=list(weights.keys()),
                original_values=weights,
                adjusted_values=None,
            )

        return None


def check_consistency(params: dict[str, Any]) -> tuple[bool, list[str]]:
    """パラメータの整合性をチェックする便利関数。

    Args:
        params: チェックするパラメータ

    Returns:
        (整合性があるか, 問題メッセージのリスト)
    """
    checker = ParameterConsistencyChecker()
    result = checker.check(params, auto_adjust=False)
    return result.is_consistent, [str(i) for i in result.issues]


def ensure_consistency(params: dict[str, Any]) -> dict[str, Any]:
    """パラメータの整合性を保証する便利関数。

    Args:
        params: 調整するパラメータ

    Returns:
        整合性が保証されたパラメータ
    """
    checker = ParameterConsistencyChecker()
    return checker.check_and_adjust(params)


def create_consistency_checker(
    strict_mode: bool = False,
    custom_rules: list[Callable[[dict], ConsistencyIssue | None]] | None = None,
) -> ParameterConsistencyChecker:
    """ParameterConsistencyChecker のファクトリ関数。

    Args:
        strict_mode: 厳格モード
        custom_rules: カスタムルール

    Returns:
        初期化された ParameterConsistencyChecker
    """
    return ParameterConsistencyChecker(
        strict_mode=strict_mode,
        custom_rules=custom_rules,
    )
