"""
Strategy Evaluator Module - 戦略評価統合

WalkForward検証、Metrics算出、GateCheckerを統合し、
Strategy×Assetの評価結果を集約してレポートを生成する。

設計根拠:
- 要求.md §5: 検証方式（データリーク防止）
- 要求.md §6: 評価指標
- 要求.md §7: Strategyの採用と重み

Numba JIT Acceleration (v1.1.0+):
- メトリクス計算にNumba JIT関数を使用
- calculate_max_drawdown, calculate_sharpe_ratio等が高速化
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

import numpy as np

from ..backtest.numba_accelerate import (
    calculate_max_drawdown_pct_numba,
    calculate_sharpe_ratio_numba,
    calculate_volatility_numba,
)
from .gate_checker import GateCheckResult, GateChecker, GateConfig, StrategyMetrics

logger = logging.getLogger(__name__)

# Flag to enable/disable Numba acceleration
USE_NUMBA = True


class EvaluationStatus(Enum):
    """評価ステータス"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EvaluatorConfig:
    """評価器の設定

    Attributes:
        train_period_days: トレーニング期間（営業日）
        test_period_days: テスト期間（営業日）
        step_period_days: スライド幅（営業日）
        min_train_samples: 最小トレーニングサンプル数
        purge_gap_days: train/test間のギャップ（リーク防止）
        embargo_days: テスト後のエンバーゴ期間
        annualization_factor: 年率化係数（日次リターンの場合252）
    """

    train_period_days: int = 252
    test_period_days: int = 63
    step_period_days: int = 21
    min_train_samples: int = 200
    purge_gap_days: int = 5
    embargo_days: int = 5
    annualization_factor: int = 252


class WalkForwardResult(Protocol):
    """WalkForward検証結果のプロトコル

    Phase 3Aで実装されるWalkForwardクラスとの互換性インターフェース
    """

    @property
    def fold_results(self) -> list[dict[str, Any]]:
        """各フォールドの結果"""
        ...

    @property
    def test_returns(self) -> np.ndarray:
        """テスト期間のリターン系列"""
        ...

    @property
    def trades(self) -> list[dict[str, Any]]:
        """取引履歴"""
        ...


class MetricsCalculator(Protocol):
    """メトリクス計算のプロトコル

    Phase 3Aで実装されるMetricsクラスとの互換性インターフェース
    """

    def calculate(
        self,
        returns: np.ndarray,
        trades: list[dict[str, Any]],
    ) -> dict[str, float]:
        """メトリクスを計算"""
        ...


@dataclass
class StrategyEvaluationResult:
    """個別戦略の評価結果

    Attributes:
        strategy_id: 戦略ID
        asset_id: アセットID
        status: 評価ステータス
        metrics: 評価指標（StrategyMetrics形式）
        gate_result: ゲートチェック結果
        walk_forward_summary: WalkForward検証のサマリー
        score: 総合スコア（採用判定用）
        error_message: エラーメッセージ（失敗時）
        evaluated_at: 評価日時
    """

    strategy_id: str
    asset_id: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    metrics: StrategyMetrics | None = None
    gate_result: GateCheckResult | None = None
    walk_forward_summary: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    error_message: str | None = None
    evaluated_at: datetime | None = None

    @property
    def is_adopted(self) -> bool:
        """戦略が採用されたかどうか"""
        return (
            self.status == EvaluationStatus.COMPLETED
            and self.gate_result is not None
            and self.gate_result.passed
        )

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "strategy_id": self.strategy_id,
            "asset_id": self.asset_id,
            "status": self.status.value,
            "is_adopted": self.is_adopted,
            "metrics": {
                "trade_count": self.metrics.trade_count if self.metrics else None,
                "max_drawdown_pct": self.metrics.max_drawdown_pct if self.metrics else None,
                "expected_value": self.metrics.expected_value if self.metrics else None,
                "sharpe_ratio": self.metrics.sharpe_ratio if self.metrics else None,
                "win_rate_pct": self.metrics.win_rate_pct if self.metrics else None,
                "profit_factor": self.metrics.profit_factor if self.metrics else None,
            } if self.metrics else None,
            "gate_result": self.gate_result.to_dict() if self.gate_result else None,
            "walk_forward_summary": self.walk_forward_summary,
            "score": self.score,
            "error_message": self.error_message,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


@dataclass
class AssetEvaluationResult:
    """アセット単位の評価結果

    Attributes:
        asset_id: アセットID
        strategy_results: 各戦略の評価結果
        adopted_strategies: 採用された戦略のID
        expected_return: アセットの期待リターン推定
        volatility: ボラティリティ推定
        data_quality_ok: データ品質チェック結果
    """

    asset_id: str
    strategy_results: list[StrategyEvaluationResult] = field(default_factory=list)
    adopted_strategies: list[str] = field(default_factory=list)
    expected_return: float = 0.0
    volatility: float = 0.0
    data_quality_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset_id": self.asset_id,
            "strategy_results": [r.to_dict() for r in self.strategy_results],
            "adopted_strategies": self.adopted_strategies,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "data_quality_ok": self.data_quality_ok,
        }


@dataclass
class EvaluationReport:
    """評価レポート全体

    Attributes:
        report_id: レポートID
        as_of: 評価基準日時
        asset_results: アセット別の評価結果
        total_strategies_evaluated: 評価した戦略総数
        total_strategies_adopted: 採用された戦略総数
        execution_time_seconds: 実行時間（秒）
        config: 評価に使用した設定
        warnings: 警告メッセージ
        errors: エラーメッセージ
    """

    report_id: str
    as_of: datetime
    asset_results: dict[str, AssetEvaluationResult] = field(default_factory=dict)
    total_strategies_evaluated: int = 0
    total_strategies_adopted: int = 0
    execution_time_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "report_id": self.report_id,
            "as_of": self.as_of.isoformat(),
            "summary": {
                "total_assets": len(self.asset_results),
                "total_strategies_evaluated": self.total_strategies_evaluated,
                "total_strategies_adopted": self.total_strategies_adopted,
                "adoption_rate": (
                    self.total_strategies_adopted / self.total_strategies_evaluated
                    if self.total_strategies_evaluated > 0 else 0.0
                ),
            },
            "asset_results": {
                k: v.to_dict() for k, v in self.asset_results.items()
            },
            "execution_time_seconds": self.execution_time_seconds,
            "config": self.config,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class BaseMetricsCalculator(ABC):
    """メトリクス計算の基底クラス"""

    def __init__(self, annualization_factor: int = 252) -> None:
        self.annualization_factor = annualization_factor

    @abstractmethod
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """シャープレシオを計算"""
        pass

    @abstractmethod
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """最大ドローダウンを計算"""
        pass

    @abstractmethod
    def calculate_win_rate(self, trades: list[dict[str, Any]]) -> float:
        """勝率を計算"""
        pass


class DefaultMetricsCalculator(BaseMetricsCalculator):
    """デフォルトのメトリクス計算実装

    Phase 3Aのmetrics.pyが完成するまでの暫定実装。

    Numba JIT Acceleration (v1.1.0+):
    - calculate_sharpe_ratio, calculate_max_drawdown等がNumba JIT高速化
    """

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """シャープレシオを計算（年率化）

        Uses Numba JIT when USE_NUMBA=True.
        """
        if len(returns) == 0:
            return 0.0

        # Use Numba accelerated version if enabled
        if USE_NUMBA:
            return float(calculate_sharpe_ratio_numba(
                returns.astype(np.float64),
                risk_free_rate=0.0,
                annualization_factor=self.annualization_factor,
            ))

        # Fallback to pure NumPy
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0
        daily_sharpe = mean_return / std_return
        return float(daily_sharpe * np.sqrt(self.annualization_factor))

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """最大ドローダウンを計算（%）

        Uses Numba JIT when USE_NUMBA=True.
        """
        if len(returns) == 0:
            return 0.0

        # Use Numba accelerated version if enabled
        if USE_NUMBA:
            return float(calculate_max_drawdown_pct_numba(returns.astype(np.float64)))

        # Fallback to pure NumPy
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns) * 100)

    def calculate_win_rate(self, trades: list[dict[str, Any]]) -> float:
        """勝率を計算（%）"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return (winning_trades / len(trades)) * 100

    def calculate_profit_factor(self, trades: list[dict[str, Any]]) -> float:
        """プロフィットファクターを計算"""
        if not trades:
            return 0.0
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def calculate_expected_value(self, trades: list[dict[str, Any]]) -> float:
        """期待値を計算（1取引あたり平均損益）"""
        if not trades:
            return 0.0
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        return total_pnl / len(trades)

    def calculate_volatility(self, returns: np.ndarray) -> float:
        """年率ボラティリティを計算

        Uses Numba JIT when USE_NUMBA=True.
        """
        if len(returns) == 0:
            return 0.0

        # Use Numba accelerated version if enabled
        if USE_NUMBA:
            return float(calculate_volatility_numba(
                returns.astype(np.float64),
                annualization_factor=self.annualization_factor,
            ))

        # Fallback to pure NumPy
        return float(np.std(returns, ddof=1) * np.sqrt(self.annualization_factor))

    def calculate_all(
        self,
        returns: np.ndarray,
        trades: list[dict[str, Any]],
    ) -> dict[str, float]:
        """全メトリクスを計算"""
        return {
            "sharpe_ratio": self.calculate_sharpe_ratio(returns),
            "max_drawdown_pct": self.calculate_max_drawdown(returns),
            "win_rate_pct": self.calculate_win_rate(trades),
            "profit_factor": self.calculate_profit_factor(trades),
            "expected_value": self.calculate_expected_value(trades),
            "volatility": self.calculate_volatility(returns),
            "trade_count": len(trades),
        }


class StrategyEvaluator:
    """戦略評価統合クラス

    WalkForward検証、Metrics算出、GateCheckerを統合し、
    Strategy×Assetの評価結果を集約する。

    Usage:
        evaluator_config = EvaluatorConfig()
        gate_config = GateConfig()
        evaluator = StrategyEvaluator(evaluator_config, gate_config)

        # 単一戦略の評価
        result = evaluator.evaluate_strategy(
            strategy_id="momentum",
            asset_id="AAPL",
            returns=np.array([...]),
            trades=[{...}, ...],
        )

        # 複数戦略×複数アセットの評価
        report = evaluator.evaluate_all(
            strategies=["momentum", "reversal"],
            assets=["AAPL", "GOOGL"],
            data_provider=data_provider,
        )
    """

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        gate_config: GateConfig | None = None,
        metrics_calculator: BaseMetricsCalculator | None = None,
    ) -> None:
        """初期化

        Args:
            config: 評価器の設定
            gate_config: ゲートチェックの設定
            metrics_calculator: メトリクス計算器
        """
        self.config = config or EvaluatorConfig()
        self.gate_checker = GateChecker(gate_config)
        self.metrics_calculator = metrics_calculator or DefaultMetricsCalculator(
            annualization_factor=self.config.annualization_factor
        )

    def evaluate_strategy(
        self,
        strategy_id: str,
        asset_id: str,
        returns: np.ndarray,
        trades: list[dict[str, Any]],
        period_returns: list[float] | None = None,
    ) -> StrategyEvaluationResult:
        """単一戦略を評価

        Args:
            strategy_id: 戦略ID
            asset_id: アセットID
            returns: リターン系列
            trades: 取引履歴
            period_returns: 各期間のリターン（安定性判定用）

        Returns:
            評価結果
        """
        result = StrategyEvaluationResult(
            strategy_id=strategy_id,
            asset_id=asset_id,
            status=EvaluationStatus.IN_PROGRESS,
        )

        try:
            # メトリクス計算
            metrics_dict = self.metrics_calculator.calculate_all(returns, trades)

            metrics = StrategyMetrics(
                strategy_id=strategy_id,
                asset_id=asset_id,
                trade_count=metrics_dict["trade_count"],
                max_drawdown_pct=metrics_dict["max_drawdown_pct"],
                expected_value=metrics_dict["expected_value"],
                sharpe_ratio=metrics_dict["sharpe_ratio"],
                win_rate_pct=metrics_dict["win_rate_pct"],
                profit_factor=metrics_dict["profit_factor"],
                period_returns=period_returns or [],
            )

            # ゲートチェック
            gate_result = self.gate_checker.check(metrics)

            # スコア計算
            score = self._calculate_score(metrics)

            result.metrics = metrics
            result.gate_result = gate_result
            result.score = score
            result.status = EvaluationStatus.COMPLETED
            result.evaluated_at = datetime.now()

            logger.info(
                "Evaluated strategy %s for %s: adopted=%s, score=%.3f",
                strategy_id,
                asset_id,
                gate_result.passed,
                score,
            )

        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            logger.exception(
                "Failed to evaluate strategy %s for %s: %s",
                strategy_id,
                asset_id,
                e,
            )

        return result

    def evaluate_asset(
        self,
        asset_id: str,
        strategy_returns: dict[str, np.ndarray],
        strategy_trades: dict[str, list[dict[str, Any]]],
        strategy_period_returns: dict[str, list[float]] | None = None,
    ) -> AssetEvaluationResult:
        """アセット単位で全戦略を評価

        Args:
            asset_id: アセットID
            strategy_returns: 戦略ID -> リターン系列
            strategy_trades: 戦略ID -> 取引履歴
            strategy_period_returns: 戦略ID -> 各期間リターン

        Returns:
            アセット評価結果
        """
        result = AssetEvaluationResult(asset_id=asset_id)
        period_returns = strategy_period_returns or {}

        for strategy_id in strategy_returns:
            returns = strategy_returns.get(strategy_id, np.array([]))
            trades = strategy_trades.get(strategy_id, [])
            periods = period_returns.get(strategy_id, [])

            eval_result = self.evaluate_strategy(
                strategy_id=strategy_id,
                asset_id=asset_id,
                returns=returns,
                trades=trades,
                period_returns=periods,
            )
            result.strategy_results.append(eval_result)

            if eval_result.is_adopted:
                result.adopted_strategies.append(strategy_id)

        # アセット全体の期待リターンとボラティリティを計算
        if result.adopted_strategies:
            adopted_results = [
                r for r in result.strategy_results
                if r.is_adopted and r.metrics is not None
            ]
            if adopted_results:
                result.expected_return = np.mean([
                    r.metrics.expected_value for r in adopted_results
                ])
                # 全リターンを結合してボラティリティ計算
                all_returns = np.concatenate([
                    strategy_returns.get(r.strategy_id, np.array([]))
                    for r in adopted_results
                ])
                if len(all_returns) > 0:
                    result.volatility = self.metrics_calculator.calculate_volatility(
                        all_returns
                    )

        return result

    def generate_report(
        self,
        asset_results: dict[str, AssetEvaluationResult],
        report_id: str | None = None,
        execution_time: float = 0.0,
    ) -> EvaluationReport:
        """評価レポートを生成

        Args:
            asset_results: アセットID -> 評価結果
            report_id: レポートID（省略時は自動生成）
            execution_time: 実行時間（秒）

        Returns:
            評価レポート
        """
        if report_id is None:
            report_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        total_evaluated = 0
        total_adopted = 0
        warnings: list[str] = []

        for asset_id, asset_result in asset_results.items():
            total_evaluated += len(asset_result.strategy_results)
            total_adopted += len(asset_result.adopted_strategies)

            if not asset_result.data_quality_ok:
                warnings.append(f"Data quality issue for {asset_id}")

            if not asset_result.adopted_strategies:
                warnings.append(f"No strategies adopted for {asset_id}")

        report = EvaluationReport(
            report_id=report_id,
            as_of=datetime.now(),
            asset_results=asset_results,
            total_strategies_evaluated=total_evaluated,
            total_strategies_adopted=total_adopted,
            execution_time_seconds=execution_time,
            config={
                "evaluator": {
                    "train_period_days": self.config.train_period_days,
                    "test_period_days": self.config.test_period_days,
                    "step_period_days": self.config.step_period_days,
                },
                "gate": {
                    "min_sharpe_ratio": self.gate_checker.config.min_sharpe_ratio,
                    "max_drawdown_pct": self.gate_checker.config.max_drawdown_pct,
                    "min_trades": self.gate_checker.config.min_trades,
                },
            },
            warnings=warnings,
        )

        logger.info(
            "Generated report %s: %d/%d strategies adopted across %d assets",
            report_id,
            total_adopted,
            total_evaluated,
            len(asset_results),
        )

        return report

    def _calculate_score(self, metrics: StrategyMetrics) -> float:
        """戦略スコアを計算

        要求.md §7.2に基づくスコアリング:
        score = Sharpe_adj - penalty
        penalty = λ1 * turnover + λ2 * MDD + λ3 * instability

        Args:
            metrics: 戦略メトリクス

        Returns:
            総合スコア
        """
        # 基本スコア（シャープレシオ）
        base_score = metrics.sharpe_ratio

        # ペナルティ係数（固定）
        lambda_mdd = 0.5
        lambda_instability = 0.3

        # MDDペナルティ（正規化: 0-25% -> 0-1）
        mdd_penalty = lambda_mdd * (metrics.max_drawdown_pct / 25.0)

        # 安定性ペナルティ
        instability_penalty = 0.0
        if metrics.period_returns:
            negative_ratio = sum(
                1 for r in metrics.period_returns if r <= 0
            ) / len(metrics.period_returns)
            instability_penalty = lambda_instability * negative_ratio

        score = base_score - mdd_penalty - instability_penalty

        return max(0.0, score)  # 負のスコアは0にクリップ
