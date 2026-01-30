"""
Risk Budgeting - リスクバジェッティング

各アセットのリスク貢献度を目標に合わせて配分するモジュール。

主要コンポーネント:
- RiskBudgetingAllocator: リスクバジェット最適化

設計根拠:
- リスクパリティの一般化
- 目標リスク配分に基づく柔軟な配分
- リスク貢献度の可視化

参考文献:
- Bruder, B. and Roncalli, T. (2012) "Managing Risk Exposures
  Using the Risk Budgeting Approach"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass
class RiskBudgetingConfig:
    """リスクバジェッティングの設定。

    Attributes:
        risk_budgets: 資産ごとの目標リスクバジェット
        max_weight: 最大ウェイト
        min_weight: 最小ウェイト
        tolerance: 最適化の許容誤差
    """

    risk_budgets: dict[str, float] | None = None
    max_weight: float = 0.25
    min_weight: float = 0.01
    tolerance: float = 1e-10


@dataclass
class RiskContribution:
    """リスク貢献度の結果。

    Attributes:
        weights: ウェイト
        risk_contributions: リスク貢献度（絶対値）
        risk_contributions_pct: リスク貢献度（%）
        marginal_risks: 限界リスク
        portfolio_volatility: ポートフォリオボラティリティ
    """

    weights: pd.Series
    risk_contributions: pd.Series
    risk_contributions_pct: pd.Series
    marginal_risks: pd.Series
    portfolio_volatility: float

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrameに変換する。"""
        return pd.DataFrame({
            "weight": self.weights,
            "risk_contribution": self.risk_contributions,
            "risk_contribution_pct": self.risk_contributions_pct,
            "marginal_risk": self.marginal_risks,
            "weight_to_risk_ratio": self.weights / self.risk_contributions_pct,
        })

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "weights": self.weights.to_dict(),
            "risk_contributions": self.risk_contributions.to_dict(),
            "risk_contributions_pct": self.risk_contributions_pct.to_dict(),
            "marginal_risks": self.marginal_risks.to_dict(),
            "portfolio_volatility": self.portfolio_volatility,
        }


@dataclass
class RiskBudgetingResult:
    """リスクバジェッティングの最適化結果。

    Attributes:
        weights: 最適ウェイト
        risk_contributions: リスク貢献度
        target_budgets: 目標リスクバジェット
        optimization_success: 最適化成功フラグ
        objective_value: 目的関数の値
        iterations: 反復回数
        timestamp: 計算日時
        metadata: 追加情報
    """

    weights: pd.Series
    risk_contributions: RiskContribution
    target_budgets: pd.Series
    optimization_success: bool
    objective_value: float
    iterations: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "weights": self.weights.to_dict(),
            "risk_contributions": self.risk_contributions.to_dict(),
            "target_budgets": self.target_budgets.to_dict(),
            "optimization_success": self.optimization_success,
            "objective_value": self.objective_value,
            "iterations": self.iterations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class RiskBudgetingAllocator:
    """リスクバジェッティングアロケーター。

    各アセットのリスク貢献度が目標リスクバジェットに一致するよう
    ウェイトを最適化する。

    Usage:
        allocator = RiskBudgetingAllocator(max_weight=0.25)

        # 均等リスクバジェット（リスクパリティと同等）
        result = allocator.optimize(returns)

        # カスタムリスクバジェット
        target_budgets = {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2}
        result = allocator.optimize(returns, target_budgets=target_budgets)

        # リスク貢献度分析
        analysis = allocator.analyze_risk_contribution(weights, returns)
    """

    def __init__(
        self,
        risk_budgets: dict[str, float] | None = None,
        max_weight: float = 0.25,
        min_weight: float = 0.01,
    ) -> None:
        """初期化。

        Args:
            risk_budgets: デフォルトのリスクバジェット
            max_weight: 最大ウェイト
            min_weight: 最小ウェイト
        """
        self.risk_budgets = risk_budgets
        self.max_weight = max_weight
        self.min_weight = min_weight

    @classmethod
    def from_config(cls, config: RiskBudgetingConfig) -> "RiskBudgetingAllocator":
        """設定からアロケーターを作成する。"""
        return cls(
            risk_budgets=config.risk_budgets,
            max_weight=config.max_weight,
            min_weight=config.min_weight,
        )

    def compute_risk_contribution(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """リスク貢献度を計算する。

        RC_i = w_i × (Σw)_i / σ_p

        Args:
            weights: ウェイトベクトル
            cov: 共分散行列

        Returns:
            リスク貢献度ベクトル（合計=1に正規化）
        """
        # ポートフォリオ分散と標準偏差
        portfolio_variance = weights @ cov @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        if portfolio_volatility < 1e-10:
            # ゼロボラティリティの場合は均等配分
            return np.ones(len(weights)) / len(weights)

        # 限界リスク: ∂σ_p/∂w_i = (Σw)_i / σ_p
        marginal_risk = cov @ weights / portfolio_volatility

        # リスク貢献度: RC_i = w_i × MR_i
        risk_contribution = weights * marginal_risk

        # 合計がポートフォリオボラティリティに等しいことを確認
        # （Euler分解: Σ RC_i = σ_p）
        total_rc = np.sum(risk_contribution)

        # リスク貢献度を比率で返す（合計=1）
        if total_rc > 1e-10:
            risk_contribution_pct = risk_contribution / total_rc
        else:
            risk_contribution_pct = np.ones(len(weights)) / len(weights)

        return risk_contribution_pct

    def _objective_function(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        target_budgets: np.ndarray,
    ) -> float:
        """目的関数: リスク貢献度と目標の乖離を最小化。

        Σ (RC_i / RC_target_i - 1)^2

        Args:
            weights: ウェイトベクトル
            cov: 共分散行列
            target_budgets: 目標リスクバジェット

        Returns:
            目的関数の値
        """
        risk_contribution = self.compute_risk_contribution(weights, cov)

        # ゼロ除算を防ぐ
        safe_targets = np.maximum(target_budgets, 1e-10)

        # 相対誤差の二乗和
        relative_error = (risk_contribution / safe_targets - 1) ** 2
        return np.sum(relative_error)

    def optimize(
        self,
        returns: pd.DataFrame,
        target_budgets: dict[str, float] | None = None,
        cov: pd.DataFrame | None = None,
    ) -> RiskBudgetingResult:
        """リスクバジェット最適化を実行する。

        Args:
            returns: リターンデータ（共分散計算用）
            target_budgets: 目標リスクバジェット（Noneの場合は均等）
            cov: 共分散行列（指定された場合はreturnsから計算しない）

        Returns:
            最適化結果
        """
        assets = returns.columns.tolist()
        n = len(assets)

        # 共分散行列
        if cov is not None:
            cov_matrix = cov.values
        else:
            cov_matrix = returns.cov().values

        # 目標リスクバジェット
        if target_budgets is None:
            target_budgets = self.risk_budgets

        if target_budgets is None:
            # 均等リスクバジェット
            target_budgets_arr = np.ones(n) / n
            target_budgets_series = pd.Series(
                target_budgets_arr, index=assets
            )
        else:
            # 指定されたリスクバジェット
            target_budgets_series = pd.Series(target_budgets).reindex(assets)
            # 合計を1に正規化
            target_budgets_arr = target_budgets_series.values
            target_budgets_arr = target_budgets_arr / np.sum(target_budgets_arr)
            target_budgets_series = pd.Series(target_budgets_arr, index=assets)

        # 制約条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 合計=1
        ]

        # 境界条件
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        # 初期値（均等ウェイト）
        w0 = np.ones(n) / n

        # 最適化
        result = optimize.minimize(
            self._objective_function,
            w0,
            args=(cov_matrix, target_budgets_arr),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        optimal_weights = pd.Series(result.x, index=assets)

        # リスク貢献度を再計算
        risk_contribution = self._compute_full_risk_contribution(
            optimal_weights, cov_matrix, assets
        )

        logger.debug(
            f"Risk budgeting optimization: success={result.success}, "
            f"objective={result.fun:.6f}, iterations={result.nit}"
        )

        return RiskBudgetingResult(
            weights=optimal_weights,
            risk_contributions=risk_contribution,
            target_budgets=target_budgets_series,
            optimization_success=result.success,
            objective_value=result.fun,
            iterations=result.nit,
            metadata={
                "max_weight": self.max_weight,
                "min_weight": self.min_weight,
                "n_assets": n,
            },
        )

    def _compute_full_risk_contribution(
        self,
        weights: pd.Series,
        cov: np.ndarray,
        assets: list[str],
    ) -> RiskContribution:
        """完全なリスク貢献度情報を計算する。"""
        w = weights.values

        # ポートフォリオボラティリティ
        portfolio_variance = w @ cov @ w
        portfolio_volatility = np.sqrt(portfolio_variance)

        # 限界リスク
        if portfolio_volatility > 1e-10:
            marginal_risk = cov @ w / portfolio_volatility
        else:
            marginal_risk = np.zeros(len(w))

        # リスク貢献度（絶対値）
        risk_contribution_abs = w * marginal_risk

        # リスク貢献度（%）
        total_rc = np.sum(risk_contribution_abs)
        if total_rc > 1e-10:
            risk_contribution_pct = risk_contribution_abs / total_rc
        else:
            risk_contribution_pct = np.ones(len(w)) / len(w)

        return RiskContribution(
            weights=weights,
            risk_contributions=pd.Series(risk_contribution_abs, index=assets),
            risk_contributions_pct=pd.Series(risk_contribution_pct, index=assets),
            marginal_risks=pd.Series(marginal_risk, index=assets),
            portfolio_volatility=portfolio_volatility,
        )

    def analyze_risk_contribution(
        self,
        weights: dict[str, float] | pd.Series,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """既存ウェイトのリスク貢献度を分析する。

        Args:
            weights: ウェイト
            returns: リターンデータ

        Returns:
            分析結果（weight, risk_contribution, risk_contribution_pct,
            weight_to_risk_ratio）
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)

        assets = returns.columns.tolist()
        weights = weights.reindex(assets).fillna(0.0)

        cov_matrix = returns.cov().values
        risk_contribution = self._compute_full_risk_contribution(
            weights, cov_matrix, assets
        )

        return risk_contribution.to_dataframe()


def create_risk_budgeting_allocator(
    risk_budgets: dict[str, float] | None = None,
    max_weight: float = 0.25,
    min_weight: float = 0.01,
) -> RiskBudgetingAllocator:
    """RiskBudgetingAllocator のファクトリ関数。

    Args:
        risk_budgets: デフォルトリスクバジェット
        max_weight: 最大ウェイト
        min_weight: 最小ウェイト

    Returns:
        初期化された RiskBudgetingAllocator
    """
    return RiskBudgetingAllocator(
        risk_budgets=risk_budgets,
        max_weight=max_weight,
        min_weight=min_weight,
    )


def quick_risk_budgeting(
    returns: pd.DataFrame,
    target_budgets: dict[str, float] | None = None,
    max_weight: float = 0.25,
) -> dict[str, float]:
    """便利関数: リスクバジェッティングを簡易実行する。

    Args:
        returns: リターンデータ
        target_budgets: 目標リスクバジェット（Noneで均等）
        max_weight: 最大ウェイト

    Returns:
        最適ウェイト
    """
    allocator = RiskBudgetingAllocator(max_weight=max_weight)
    result = allocator.optimize(returns, target_budgets=target_budgets)
    return result.weights.to_dict()


def compute_risk_contribution(
    weights: dict[str, float] | pd.Series,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """便利関数: リスク貢献度を計算する。

    Args:
        weights: ウェイト
        returns: リターンデータ

    Returns:
        リスク貢献度分析結果
    """
    allocator = RiskBudgetingAllocator()
    return allocator.analyze_risk_contribution(weights, returns)
