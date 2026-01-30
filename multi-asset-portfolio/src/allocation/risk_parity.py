"""
Risk Parity Module - リスクパリティ

各アセットのリスク寄与を均等化する配分手法。
反復最適化により、全アセットが同じリスク貢献をする重みを求める。

アルゴリズム:
1. 目的関数: Σ(RC_i - RC_target)^2 を最小化
2. 制約: Σw_i = 1, w_i >= 0
3. RC_i = w_i * (Σ * w)_i / σ_p （リスク寄与）

設計根拠:
- 要求.md §8.3: リスクパリティ推奨（推定誤差に比較的強い）
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios

使用方法:
    rp = RiskParity()
    result = rp.allocate(covariance_matrix)
    weights = result.weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskParityConfig:
    """リスクパリティ設定

    Attributes:
        target_risk_contrib: 目標リスク寄与（Noneの場合は均等）
        max_iterations: 最大反復回数
        tolerance: 収束判定閾値
        regularization: 正則化係数（数値安定性）
        solver_method: 最適化手法
    """

    target_risk_contrib: list[float] | None = None
    max_iterations: int = 1000
    tolerance: float = 1e-10
    regularization: float = 1e-8
    solver_method: str = "SLSQP"

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        valid_methods = {"SLSQP", "trust-constr", "COBYLA"}
        if self.solver_method not in valid_methods:
            raise ValueError(
                f"Invalid solver_method: {self.solver_method}. "
                f"Must be one of {valid_methods}"
            )


@dataclass
class RiskParityResult:
    """リスクパリティ配分結果

    Attributes:
        weights: アセット別重み（Series）
        risk_contributions: 各アセットのリスク寄与（Series）
        marginal_risks: 各アセットの限界リスク（Series）
        portfolio_volatility: ポートフォリオ全体のボラティリティ
        risk_contrib_deviation: リスク寄与の標準偏差（均等からの乖離）
        converged: 収束したかどうか
        iterations: 反復回数
        metadata: 追加メタデータ
    """

    weights: pd.Series
    risk_contributions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    marginal_risks: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_volatility: float = 0.0
    risk_contrib_deviation: float = 0.0
    converged: bool = False
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な配分結果かどうか"""
        if self.weights.empty:
            return False
        if self.weights.isna().any():
            return False
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            return False
        if (self.weights < -1e-8).any():
            return False
        return True

    @property
    def is_risk_parity_achieved(self) -> bool:
        """リスクパリティが達成されているか"""
        if self.risk_contributions.empty:
            return False
        # リスク寄与の変動係数が小さいか
        rc = self.risk_contributions
        if rc.mean() == 0:
            return True
        cv = rc.std() / rc.mean()
        return cv < 0.01  # 1%以下なら達成

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights.to_dict(),
            "risk_contributions": self.risk_contributions.to_dict(),
            "marginal_risks": self.marginal_risks.to_dict(),
            "portfolio_volatility": self.portfolio_volatility,
            "risk_contrib_deviation": self.risk_contrib_deviation,
            "converged": self.converged,
            "iterations": self.iterations,
            "is_valid": self.is_valid,
            "is_risk_parity_achieved": self.is_risk_parity_achieved,
            "metadata": self.metadata,
        }


class RiskParity:
    """リスクパリティクラス

    各アセットのリスク寄与を均等化する配分を計算する。

    Usage:
        config = RiskParityConfig()
        rp = RiskParity(config)

        # covariance: (N, N) の共分散行列DataFrame
        result = rp.allocate(covariance)

        print(result.weights)
        print(result.risk_contributions)
    """

    def __init__(self, config: RiskParityConfig | None = None) -> None:
        """初期化

        Args:
            config: リスクパリティ設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or RiskParityConfig()

    def allocate(
        self,
        covariance: pd.DataFrame,
        target_risk_contrib: pd.Series | None = None,
    ) -> RiskParityResult:
        """リスクパリティによる配分を計算

        Args:
            covariance: 共分散行列 (N x N)
            target_risk_contrib: 目標リスク寄与（Noneの場合は均等）

        Returns:
            RiskParityResult: 配分結果
        """
        if covariance.empty:
            logger.warning("Empty covariance matrix provided")
            return self._create_empty_result(covariance.columns)

        assets = covariance.columns.tolist()
        n_assets = len(assets)

        if n_assets == 1:
            return RiskParityResult(
                weights=pd.Series([1.0], index=assets),
                risk_contributions=pd.Series([1.0], index=assets),
                portfolio_volatility=float(np.sqrt(covariance.iloc[0, 0])),
                converged=True,
                iterations=0,
                metadata={"note": "Single asset portfolio"},
            )

        cov_matrix = covariance.values

        # 目標リスク寄与
        if target_risk_contrib is not None:
            target_rc = target_risk_contrib.values
            target_rc = target_rc / target_rc.sum()  # 正規化
        elif self.config.target_risk_contrib is not None:
            target_rc = np.array(self.config.target_risk_contrib)
            target_rc = target_rc / target_rc.sum()
        else:
            # 均等配分
            target_rc = np.ones(n_assets) / n_assets

        # 最適化
        weights, converged, iterations = self._optimize(cov_matrix, target_rc)

        # リスク寄与の計算
        risk_contribs, marginal_risks, port_vol = self._compute_risk_metrics(
            weights, cov_matrix
        )

        # 結果の構築
        weights_series = pd.Series(weights, index=assets)
        rc_series = pd.Series(risk_contribs, index=assets)
        mr_series = pd.Series(marginal_risks, index=assets)

        # リスク寄与の乖離
        rc_deviation = float(np.std(risk_contribs))

        logger.info(
            "Risk parity allocation: %d assets, vol=%.4f, converged=%s, iter=%d",
            n_assets,
            port_vol,
            converged,
            iterations,
        )

        return RiskParityResult(
            weights=weights_series,
            risk_contributions=rc_series,
            marginal_risks=mr_series,
            portfolio_volatility=port_vol,
            risk_contrib_deviation=rc_deviation,
            converged=converged,
            iterations=iterations,
            metadata={
                "target_risk_contrib": target_rc.tolist(),
                "solver_method": self.config.solver_method,
            },
        )

    def _optimize(
        self,
        covariance: NDArray[np.float64],
        target_rc: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], bool, int]:
        """リスクパリティ最適化

        Args:
            covariance: 共分散行列
            target_rc: 目標リスク寄与

        Returns:
            (最適重み, 収束フラグ, 反復回数)
        """
        n = len(target_rc)

        # 初期値: 逆分散配分
        variances = np.diag(covariance)
        inv_var = 1.0 / (variances + self.config.regularization)
        x0 = inv_var / inv_var.sum()

        # 目的関数: リスク寄与の二乗誤差
        def objective(w: NDArray[np.float64]) -> float:
            w = np.maximum(w, 1e-10)  # 数値安定性
            port_var = w @ covariance @ w
            port_vol = np.sqrt(port_var + self.config.regularization)
            marginal_risk = (covariance @ w) / port_vol
            risk_contrib = w * marginal_risk / port_vol
            diff = risk_contrib - target_rc
            return float(np.sum(diff**2))

        # 勾配（数値微分用に提供しないが、SLSQPは自動計算）

        # 制約
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 総和 = 1
        ]

        # 境界条件: 0 <= w <= 1
        bounds = [(0.0, 1.0) for _ in range(n)]

        # 最適化実行
        result = minimize(
            objective,
            x0,
            method=self.config.solver_method,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        weights = result.x
        weights = np.maximum(weights, 0.0)  # 負の値を排除
        weights = weights / weights.sum()  # 正規化

        return weights, result.success, result.nit

    def _compute_risk_metrics(
        self,
        weights: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """リスク指標を計算

        Args:
            weights: 重み配列
            covariance: 共分散行列

        Returns:
            (リスク寄与, 限界リスク, ポートフォリオボラティリティ)
        """
        port_var = weights @ covariance @ weights
        port_vol = np.sqrt(port_var + self.config.regularization)

        # 限界リスク: ∂σ_p / ∂w_i
        marginal_risk = (covariance @ weights) / port_vol

        # リスク寄与: w_i * MR_i / σ_p
        risk_contrib = weights * marginal_risk / port_vol

        return risk_contrib, marginal_risk, port_vol

    def _create_empty_result(self, columns: pd.Index) -> RiskParityResult:
        """空の結果を作成

        Args:
            columns: アセット名

        Returns:
            空のRiskParityResult
        """
        n = len(columns)
        if n > 0:
            weights = pd.Series(np.ones(n) / n, index=columns)
            rc = pd.Series(np.ones(n) / n, index=columns)
        else:
            weights = pd.Series(dtype=float)
            rc = pd.Series(dtype=float)

        return RiskParityResult(
            weights=weights,
            risk_contributions=rc,
            metadata={"error": "Empty or invalid covariance"},
        )


class NaiveRiskParity:
    """ナイーブリスクパリティ

    簡易版: 逆分散配分（各アセットの分散の逆数で配分）
    相関を無視するが、計算が高速で安定。
    """

    def allocate(self, covariance: pd.DataFrame) -> RiskParityResult:
        """逆分散配分を計算

        Args:
            covariance: 共分散行列

        Returns:
            RiskParityResult
        """
        if covariance.empty:
            return RiskParityResult(
                weights=pd.Series(dtype=float),
                metadata={"error": "Empty covariance"},
            )

        assets = covariance.columns
        variances = np.diag(covariance.values)

        # 逆分散配分
        inv_var = 1.0 / (variances + 1e-10)
        weights = inv_var / inv_var.sum()

        weights_series = pd.Series(weights, index=assets)

        # リスク指標計算
        cov_matrix = covariance.values
        port_var = weights @ cov_matrix @ weights
        port_vol = np.sqrt(port_var)
        marginal_risk = (cov_matrix @ weights) / port_vol
        risk_contrib = weights * marginal_risk / port_vol

        return RiskParityResult(
            weights=weights_series,
            risk_contributions=pd.Series(risk_contrib, index=assets),
            marginal_risks=pd.Series(marginal_risk, index=assets),
            portfolio_volatility=port_vol,
            converged=True,
            iterations=0,
            metadata={"method": "naive_inverse_variance"},
        )


def create_risk_parity_from_settings() -> RiskParity:
    """グローバル設定からRiskParityを生成

    Returns:
        設定済みのRiskParity
    """
    config = RiskParityConfig()
    return RiskParity(config)
