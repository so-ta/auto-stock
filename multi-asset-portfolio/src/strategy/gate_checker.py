"""
Gate Checker Module - ハードゲート判定

戦略がポートフォリオに採用されるための必須条件を検証する。
いずれか1つでも満たさない場合、その戦略は不採用となる。

設計根拠:
- 要求.md §6: 評価指標（ハードゲート）
- docs/design/config_spec.md §4: ハードゲート閾値
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GateType(Enum):
    """ゲートの種類"""

    MIN_TRADES = "min_trades"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_EXPECTED_VALUE = "min_expected_value"
    MIN_SHARPE = "min_sharpe"
    MIN_WIN_RATE = "min_win_rate"
    MIN_PROFIT_FACTOR = "min_profit_factor"
    STABILITY = "stability"


@dataclass(frozen=True)
class GateConfig:
    """ハードゲート閾値の設定

    Attributes:
        min_trades: 最小取引回数（統計的信頼性確保、デフォルト30）
        max_drawdown_pct: 最大ドローダウン許容値（%、デフォルト25.0）
        min_expected_value: 最小期待値（コスト控除後、デフォルト0.0）
        min_sharpe_ratio: 最小シャープレシオ（デフォルト0.5）
        min_win_rate_pct: 最小勝率（%、デフォルト45.0）
        min_profit_factor: 最小プロフィットファクター（デフォルト1.2）
        stability_periods: 安定性判定の期間数（デフォルト6）
        stability_min_positive: 安定性判定の最小プラス期間数（デフォルト4）
    """

    min_trades: int = 30
    max_drawdown_pct: float = 25.0
    min_expected_value: float = 0.0
    min_sharpe_ratio: float = 0.5
    min_win_rate_pct: float = 45.0
    min_profit_factor: float = 1.2
    stability_periods: int = 6
    stability_min_positive: int = 4

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 100:
            raise ValueError("max_drawdown_pct must be in (0, 100]")
        if self.min_win_rate_pct < 0 or self.min_win_rate_pct > 100:
            raise ValueError("min_win_rate_pct must be in [0, 100]")
        if self.min_profit_factor < 0:
            raise ValueError("min_profit_factor must be >= 0")
        if self.stability_min_positive > self.stability_periods:
            raise ValueError(
                "stability_min_positive must be <= stability_periods"
            )


@dataclass
class GateResult:
    """個別ゲートの判定結果

    Attributes:
        gate_type: ゲートの種類
        passed: 通過したかどうか
        actual_value: 実際の値
        threshold: 閾値
        message: 説明メッセージ
    """

    gate_type: GateType
    passed: bool
    actual_value: float | int | None
    threshold: float | int
    message: str


@dataclass
class GateCheckResult:
    """ゲートチェック全体の結果

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        passed: 全ゲートを通過したか
        results: 個別ゲートの結果リスト
        rejection_reasons: 不合格理由のリスト
    """

    strategy_id: str
    asset_id: str
    passed: bool
    results: list[GateResult] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "strategy_id": self.strategy_id,
            "asset_id": self.asset_id,
            "passed": self.passed,
            "results": [
                {
                    "gate_type": r.gate_type.value,
                    "passed": r.passed,
                    "actual_value": r.actual_value,
                    "threshold": r.threshold,
                    "message": r.message,
                }
                for r in self.results
            ],
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class StrategyMetrics:
    """戦略の評価指標

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        trade_count: 取引回数
        max_drawdown_pct: 最大ドローダウン（%）
        expected_value: 期待値（コスト控除後）
        sharpe_ratio: シャープレシオ（年率化）
        win_rate_pct: 勝率（%）
        profit_factor: プロフィットファクター
        period_returns: 各期間のリターン（安定性判定用）
    """

    strategy_id: str
    asset_id: str
    trade_count: int
    max_drawdown_pct: float
    expected_value: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    period_returns: list[float] = field(default_factory=list)


class GateChecker:
    """ハードゲート判定クラス

    戦略がポートフォリオに採用されるための必須条件を検証する。

    Usage:
        config = GateConfig(min_trades=30, max_drawdown_pct=25.0)
        checker = GateChecker(config)
        metrics = StrategyMetrics(...)
        result = checker.check(metrics)
        if result.passed:
            print("Strategy adopted")
        else:
            print(f"Rejected: {result.rejection_reasons}")
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        """初期化

        Args:
            config: ゲート設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or GateConfig()

    def check(self, metrics: StrategyMetrics) -> GateCheckResult:
        """全ゲートを検証

        Args:
            metrics: 戦略の評価指標

        Returns:
            ゲートチェック結果
        """
        results: list[GateResult] = []
        rejection_reasons: list[str] = []

        # 1. 最小取引回数
        result = self._check_min_trades(metrics.trade_count)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 2. 最大ドローダウン
        result = self._check_max_drawdown(metrics.max_drawdown_pct)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 3. 最小期待値
        result = self._check_min_expected_value(metrics.expected_value)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 4. 最小シャープレシオ
        result = self._check_min_sharpe(metrics.sharpe_ratio)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 5. 最小勝率
        result = self._check_min_win_rate(metrics.win_rate_pct)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 6. 最小プロフィットファクター
        result = self._check_min_profit_factor(metrics.profit_factor)
        results.append(result)
        if not result.passed:
            rejection_reasons.append(result.message)

        # 7. 安定性（期間リターンがある場合）
        if metrics.period_returns:
            result = self._check_stability(metrics.period_returns)
            results.append(result)
            if not result.passed:
                rejection_reasons.append(result.message)

        passed = len(rejection_reasons) == 0

        if passed:
            logger.info(
                "Strategy %s for %s passed all gates",
                metrics.strategy_id,
                metrics.asset_id,
            )
        else:
            logger.warning(
                "Strategy %s for %s rejected: %s",
                metrics.strategy_id,
                metrics.asset_id,
                "; ".join(rejection_reasons),
            )

        return GateCheckResult(
            strategy_id=metrics.strategy_id,
            asset_id=metrics.asset_id,
            passed=passed,
            results=results,
            rejection_reasons=rejection_reasons,
        )

    def _check_min_trades(self, trade_count: int) -> GateResult:
        """最小取引回数の検証"""
        passed = trade_count >= self.config.min_trades
        return GateResult(
            gate_type=GateType.MIN_TRADES,
            passed=passed,
            actual_value=trade_count,
            threshold=self.config.min_trades,
            message="" if passed else (
                f"Trade count ({trade_count}) < min ({self.config.min_trades})"
            ),
        )

    def _check_max_drawdown(self, max_drawdown_pct: float) -> GateResult:
        """最大ドローダウンの検証"""
        passed = max_drawdown_pct <= self.config.max_drawdown_pct
        return GateResult(
            gate_type=GateType.MAX_DRAWDOWN,
            passed=passed,
            actual_value=max_drawdown_pct,
            threshold=self.config.max_drawdown_pct,
            message="" if passed else (
                f"Max drawdown ({max_drawdown_pct:.2f}%) > "
                f"limit ({self.config.max_drawdown_pct:.2f}%)"
            ),
        )

    def _check_min_expected_value(self, expected_value: float) -> GateResult:
        """最小期待値の検証"""
        passed = expected_value > self.config.min_expected_value
        return GateResult(
            gate_type=GateType.MIN_EXPECTED_VALUE,
            passed=passed,
            actual_value=expected_value,
            threshold=self.config.min_expected_value,
            message="" if passed else (
                f"Expected value ({expected_value:.6f}) <= "
                f"min ({self.config.min_expected_value})"
            ),
        )

    def _check_min_sharpe(self, sharpe_ratio: float) -> GateResult:
        """最小シャープレシオの検証"""
        passed = sharpe_ratio >= self.config.min_sharpe_ratio
        return GateResult(
            gate_type=GateType.MIN_SHARPE,
            passed=passed,
            actual_value=sharpe_ratio,
            threshold=self.config.min_sharpe_ratio,
            message="" if passed else (
                f"Sharpe ratio ({sharpe_ratio:.3f}) < "
                f"min ({self.config.min_sharpe_ratio})"
            ),
        )

    def _check_min_win_rate(self, win_rate_pct: float) -> GateResult:
        """最小勝率の検証"""
        passed = win_rate_pct >= self.config.min_win_rate_pct
        return GateResult(
            gate_type=GateType.MIN_WIN_RATE,
            passed=passed,
            actual_value=win_rate_pct,
            threshold=self.config.min_win_rate_pct,
            message="" if passed else (
                f"Win rate ({win_rate_pct:.2f}%) < "
                f"min ({self.config.min_win_rate_pct:.2f}%)"
            ),
        )

    def _check_min_profit_factor(self, profit_factor: float) -> GateResult:
        """最小プロフィットファクターの検証"""
        passed = profit_factor >= self.config.min_profit_factor
        return GateResult(
            gate_type=GateType.MIN_PROFIT_FACTOR,
            passed=passed,
            actual_value=profit_factor,
            threshold=self.config.min_profit_factor,
            message="" if passed else (
                f"Profit factor ({profit_factor:.3f}) < "
                f"min ({self.config.min_profit_factor})"
            ),
        )

    def _check_stability(self, period_returns: list[float]) -> GateResult:
        """安定性の検証（過去N期のうちK期で期待値プラス）"""
        recent_returns = period_returns[-self.config.stability_periods:]
        positive_periods = sum(1 for r in recent_returns if r > 0)
        passed = positive_periods >= self.config.stability_min_positive

        return GateResult(
            gate_type=GateType.STABILITY,
            passed=passed,
            actual_value=positive_periods,
            threshold=self.config.stability_min_positive,
            message="" if passed else (
                f"Positive periods ({positive_periods}/{len(recent_returns)}) < "
                f"min ({self.config.stability_min_positive}/{self.config.stability_periods})"
            ),
        )

    def filter_strategies(
        self,
        metrics_list: list[StrategyMetrics],
    ) -> tuple[list[StrategyMetrics], list[GateCheckResult]]:
        """複数戦略をフィルタリング

        Args:
            metrics_list: 戦略メトリクスのリスト

        Returns:
            (合格した戦略のメトリクス, 全戦略のチェック結果)
        """
        passed_metrics: list[StrategyMetrics] = []
        all_results: list[GateCheckResult] = []

        for metrics in metrics_list:
            result = self.check(metrics)
            all_results.append(result)
            if result.passed:
                passed_metrics.append(metrics)

        logger.info(
            "Gate filter: %d/%d strategies passed",
            len(passed_metrics),
            len(metrics_list),
        )

        return passed_metrics, all_results
