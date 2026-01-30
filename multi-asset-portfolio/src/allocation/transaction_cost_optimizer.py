"""
Transaction Cost Optimizer - トランザクションコスト最適化

取引コストを考慮した最適配分を計算する。

主要コンポーネント:
1. TransactionCostOptimizer: コスト考慮型の最適化
2. TurnoverConstrainedOptimizer: ターンオーバー制約付き最適化

最適化目的:
    max (期待リターン - リスクペナルティ - コストペナルティ)

使用例:
    from src.allocation.transaction_cost_optimizer import (
        TransactionCostOptimizer,
        TurnoverConstrainedOptimizer,
    )

    optimizer = TransactionCostOptimizer(fixed_cost_bps=10, risk_aversion=2.0)

    # 最適化
    result = optimizer.optimize(
        returns=returns_df,
        current_weights={"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4},
    )
    print(f"Optimal weights: {result.optimal_weights}")
    print(f"Expected transaction cost: {result.transaction_cost:.2f}%")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransactionCostConfig:
    """トランザクションコスト設定

    Attributes:
        fixed_cost_bps: 固定コスト（ベーシスポイント）
        spread_bps: スプレッドコスト（bps）
        slippage_bps: スリッページ（bps）
        risk_aversion: リスク回避度
        cost_aversion: コスト回避度
        max_weight: 最大ウェイト
        min_weight: 最小ウェイト
    """
    fixed_cost_bps: float = 10.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    risk_aversion: float = 2.0
    cost_aversion: float = 1.0
    max_weight: float = 0.20
    min_weight: float = 0.0

    @property
    def total_cost_bps(self) -> float:
        """片道トータルコスト（bps）"""
        return self.fixed_cost_bps + self.spread_bps + self.slippage_bps

    def __post_init__(self) -> None:
        if self.fixed_cost_bps < 0:
            raise ValueError("fixed_cost_bps must be >= 0")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")


@dataclass
class OptimizationResult:
    """最適化結果

    Attributes:
        optimal_weights: 最適ウェイト
        current_weights: 現在のウェイト
        expected_return: 期待リターン
        expected_risk: 期待リスク
        transaction_cost: トランザクションコスト（%）
        turnover: ターンオーバー
        utility: ユーティリティ関数値
        converged: 収束したか
        metadata: 追加情報
    """
    optimal_weights: Dict[str, float]
    current_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    transaction_cost: float
    turnover: float
    utility: float
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_weights": self.optimal_weights,
            "current_weights": self.current_weights,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "transaction_cost": self.transaction_cost,
            "turnover": self.turnover,
            "utility": self.utility,
            "converged": self.converged,
            "metadata": self.metadata,
        }


# =============================================================================
# Transaction Cost Optimizer
# =============================================================================

class TransactionCostOptimizer:
    """トランザクションコスト最適化クラス

    取引コストを考慮した最適配分を計算する。

    目的関数:
        U = μ'w - (λ/2)*w'Σw - κ*TC(w, w_current)

        where:
        - μ: 期待リターン
        - λ: リスク回避度
        - Σ: 共分散行列
        - κ: コスト回避度
        - TC: トランザクションコスト

    Example:
        optimizer = TransactionCostOptimizer(
            fixed_cost_bps=10,
            risk_aversion=2.0,
            cost_aversion=1.0,
        )

        result = optimizer.optimize(
            returns=returns_df,
            current_weights={"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4},
        )
    """

    def __init__(
        self,
        fixed_cost_bps: float = 10.0,
        risk_aversion: float = 2.0,
        cost_aversion: float = 1.0,
        max_weight: float = 0.20,
        config: TransactionCostConfig | None = None,
    ) -> None:
        """初期化

        Args:
            fixed_cost_bps: 固定コスト（bps）
            risk_aversion: リスク回避度
            cost_aversion: コスト回避度
            max_weight: 最大ウェイト
            config: 設定オブジェクト（優先）
        """
        if config is not None:
            self.config = config
        else:
            self.config = TransactionCostConfig(
                fixed_cost_bps=fixed_cost_bps,
                risk_aversion=risk_aversion,
                cost_aversion=cost_aversion,
                max_weight=max_weight,
            )

    def compute_transaction_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """トランザクションコストを計算

        Args:
            current_weights: 現在のウェイト
            target_weights: ターゲットウェイト

        Returns:
            トランザクションコスト（%）
        """
        # 全銘柄を収集
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        # ターンオーバー計算
        turnover = sum(
            abs(target_weights.get(asset, 0) - current_weights.get(asset, 0))
            for asset in all_assets
        )

        # コスト = ターンオーバー * コスト率（片道）
        # ターンオーバーは往復なので2で割る（売り+買い両方含む）
        cost_rate = self.config.total_cost_bps / 10000  # bps to ratio
        transaction_cost = (turnover / 2) * cost_rate * 100  # percentage

        return transaction_cost

    def compute_turnover(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """ターンオーバーを計算

        Args:
            current_weights: 現在のウェイト
            target_weights: ターゲットウェイト

        Returns:
            ターンオーバー（0-2）
        """
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        return sum(
            abs(target_weights.get(asset, 0) - current_weights.get(asset, 0))
            for asset in all_assets
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        expected_returns: pd.Series | None = None,
    ) -> OptimizationResult:
        """最適化を実行

        目的: max (期待リターン - リスクペナルティ - コストペナルティ)

        Args:
            returns: リターンデータ（columns=銘柄）
            current_weights: 現在のウェイト
            expected_returns: 期待リターン（Noneの場合は履歴平均）

        Returns:
            OptimizationResult
        """
        # 銘柄リスト
        assets = list(returns.columns)
        n_assets = len(assets)

        if n_assets == 0:
            raise ValueError("No assets in returns DataFrame")

        # 期待リターン
        if expected_returns is None:
            mu = returns.mean().values * 252  # 年率化
        else:
            mu = expected_returns.reindex(assets).values * 252

        # 共分散行列
        cov = returns.cov().values * 252  # 年率化

        # 現在のウェイトをベクトル化
        w_current = np.array([current_weights.get(asset, 0) for asset in assets])

        # コスト率
        cost_rate = self.config.total_cost_bps / 10000

        # 目的関数（最小化するので符号反転）
        def objective(w: np.ndarray) -> float:
            expected_return = np.dot(mu, w)
            risk = np.dot(w, np.dot(cov, w))
            turnover = np.sum(np.abs(w - w_current))
            transaction_cost = (turnover / 2) * cost_rate

            # U = return - (λ/2)*risk - κ*cost
            utility = (
                expected_return
                - (self.config.risk_aversion / 2) * risk
                - self.config.cost_aversion * transaction_cost
            )
            return -utility  # 最小化

        # 制約条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 合計=1
        ]

        # 境界条件
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

        # 初期値（現在のウェイト、または均等配分）
        w0 = w_current.copy()
        if np.sum(w0) < 0.01:
            w0 = np.ones(n_assets) / n_assets

        # 最適化実行
        result = optimize.minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        # 結果を整形
        w_optimal = result.x
        w_optimal = np.clip(w_optimal, 0, self.config.max_weight)
        w_optimal = w_optimal / np.sum(w_optimal)  # 正規化

        optimal_weights = {assets[i]: float(w_optimal[i]) for i in range(n_assets)}
        turnover = float(np.sum(np.abs(w_optimal - w_current)))
        transaction_cost = (turnover / 2) * cost_rate * 100

        expected_return = float(np.dot(mu, w_optimal))
        expected_risk = float(np.sqrt(np.dot(w_optimal, np.dot(cov, w_optimal))))

        logger.info(
            "Optimization completed: turnover=%.2f, cost=%.3f%%, converged=%s",
            turnover,
            transaction_cost,
            result.success,
        )

        return OptimizationResult(
            optimal_weights=optimal_weights,
            current_weights=current_weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            transaction_cost=transaction_cost,
            turnover=turnover,
            utility=-result.fun,
            converged=result.success,
            metadata={
                "n_iterations": result.nit,
                "message": result.message,
                "risk_aversion": self.config.risk_aversion,
                "cost_aversion": self.config.cost_aversion,
            },
        )

    def compute_optimal_rebalance_threshold(
        self,
        portfolio_vol: float,
        holding_period: int = 20,
    ) -> float:
        """最適リバランス閾値を計算

        Almgren & Chriss (2000) のフレームワークに基づく。
        閾値 = sqrt(2 * コスト * 期待分散改善)

        Args:
            portfolio_vol: ポートフォリオのボラティリティ（年率）
            holding_period: 保有期間（日数）

        Returns:
            リバランス閾値（ターンオーバー）
        """
        # コスト率
        cost_rate = self.config.total_cost_bps / 10000

        # 期待分散改善（簡易推定）
        # 保有期間中のドリフトによる分散増加
        daily_vol = portfolio_vol / np.sqrt(252)
        variance_drift = daily_vol ** 2 * holding_period

        # 最適閾値
        # threshold = sqrt(2 * cost * variance_improvement)
        threshold = np.sqrt(2 * cost_rate * variance_drift)

        logger.debug(
            "Optimal rebalance threshold: vol=%.2f%%, period=%d, threshold=%.3f",
            portfolio_vol * 100,
            holding_period,
            threshold,
        )

        return float(threshold)


# =============================================================================
# Turnover Constrained Optimizer
# =============================================================================

class TurnoverConstrainedOptimizer:
    """ターンオーバー制約付き最適化クラス

    ターンオーバー制約を満たす範囲でターゲットに近づける。

    Example:
        optimizer = TurnoverConstrainedOptimizer(max_turnover=0.20)

        result = optimizer.optimize(
            returns=returns_df,
            current_weights={"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4},
            target_weights={"AAPL": 0.25, "MSFT": 0.35, "SPY": 0.40},
        )
    """

    def __init__(
        self,
        max_turnover: float = 0.20,
        max_weight: float = 0.20,
    ) -> None:
        """初期化

        Args:
            max_turnover: 最大ターンオーバー（0-2）
            max_weight: 最大ウェイト
        """
        if not 0 < max_turnover <= 2:
            raise ValueError("max_turnover must be in (0, 2]")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")

        self.max_turnover = max_turnover
        self.max_weight = max_weight

    def optimize(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> OptimizationResult:
        """ターンオーバー制約付き最適化

        ターゲットに可能な限り近づけるが、
        ターンオーバー制約を満たす範囲で調整。

        Args:
            returns: リターンデータ（リスク計算用）
            current_weights: 現在のウェイト
            target_weights: ターゲットウェイト

        Returns:
            OptimizationResult
        """
        # 全銘柄を収集
        all_assets = sorted(set(current_weights.keys()) | set(target_weights.keys()))
        n_assets = len(all_assets)

        # ベクトル化
        w_current = np.array([current_weights.get(a, 0) for a in all_assets])
        w_target = np.array([target_weights.get(a, 0) for a in all_assets])

        # ターゲットへの方向
        delta = w_target - w_current
        total_turnover = np.sum(np.abs(delta))

        if total_turnover <= self.max_turnover:
            # 制約内ならターゲットをそのまま使用
            w_optimal = w_target.copy()
            actual_turnover = total_turnover
        else:
            # 制約を超える場合は比例縮小
            scale = self.max_turnover / total_turnover
            w_optimal = w_current + scale * delta
            actual_turnover = self.max_turnover

        # 境界条件の適用
        w_optimal = np.clip(w_optimal, 0, self.max_weight)

        # 正規化（合計=1）
        w_sum = np.sum(w_optimal)
        if w_sum > 0:
            w_optimal = w_optimal / w_sum

        # 結果を辞書化
        optimal_weights = {all_assets[i]: float(w_optimal[i]) for i in range(n_assets)}

        # リスク計算（利用可能な銘柄のみ）
        available_assets = [a for a in all_assets if a in returns.columns]
        if len(available_assets) > 0:
            cov = returns[available_assets].cov().values * 252
            w_risk = np.array([optimal_weights.get(a, 0) for a in available_assets])
            expected_risk = float(np.sqrt(np.dot(w_risk, np.dot(cov, w_risk))))
            expected_return = float(returns[available_assets].mean().values @ w_risk * 252)
        else:
            expected_risk = 0.0
            expected_return = 0.0

        logger.info(
            "Turnover-constrained optimization: max=%.2f, actual=%.3f",
            self.max_turnover,
            actual_turnover,
        )

        return OptimizationResult(
            optimal_weights=optimal_weights,
            current_weights=current_weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            transaction_cost=0.0,  # コスト計算は別途
            turnover=actual_turnover,
            utility=0.0,
            converged=True,
            metadata={
                "target_weights": target_weights,
                "max_turnover": self.max_turnover,
                "scale_factor": actual_turnover / total_turnover if total_turnover > 0 else 1.0,
            },
        )

    def adjust_to_turnover_limit(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """ターンオーバー制限に調整（簡易版）

        Args:
            current_weights: 現在のウェイト
            target_weights: ターゲットウェイト

        Returns:
            調整後のウェイト
        """
        all_assets = sorted(set(current_weights.keys()) | set(target_weights.keys()))

        w_current = np.array([current_weights.get(a, 0) for a in all_assets])
        w_target = np.array([target_weights.get(a, 0) for a in all_assets])

        delta = w_target - w_current
        total_turnover = np.sum(np.abs(delta))

        if total_turnover <= self.max_turnover:
            return target_weights.copy()

        scale = self.max_turnover / total_turnover
        w_adjusted = w_current + scale * delta
        w_adjusted = np.clip(w_adjusted, 0, self.max_weight)
        w_adjusted = w_adjusted / np.sum(w_adjusted)

        return {all_assets[i]: float(w_adjusted[i]) for i in range(len(all_assets))}


# =============================================================================
# 便利関数
# =============================================================================

def compute_transaction_cost(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    cost_bps: float = 10.0,
) -> float:
    """トランザクションコストを計算（ショートカット関数）

    Args:
        current_weights: 現在のウェイト
        target_weights: ターゲットウェイト
        cost_bps: コスト（bps）

    Returns:
        トランザクションコスト（%）
    """
    optimizer = TransactionCostOptimizer(fixed_cost_bps=cost_bps)
    return optimizer.compute_transaction_cost(current_weights, target_weights)


def optimize_with_cost(
    returns: pd.DataFrame,
    current_weights: Dict[str, float],
    cost_bps: float = 10.0,
    risk_aversion: float = 2.0,
    max_weight: float = 0.20,
) -> Dict[str, float]:
    """コスト考慮型最適化（ショートカット関数）

    Args:
        returns: リターンデータ
        current_weights: 現在のウェイト
        cost_bps: コスト（bps）
        risk_aversion: リスク回避度
        max_weight: 最大ウェイト

    Returns:
        最適ウェイト
    """
    optimizer = TransactionCostOptimizer(
        fixed_cost_bps=cost_bps,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    result = optimizer.optimize(returns, current_weights)
    return result.optimal_weights
