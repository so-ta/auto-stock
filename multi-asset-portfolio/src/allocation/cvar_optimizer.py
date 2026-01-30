"""
CVaR (Conditional Value at Risk) Optimizer - テールリスク考慮の配分最適化

CVaR（期待ショートフォール）を最小化する配分最適化を提供する。
通常の分散最適化と異なり、テールリスク（大きな損失）を明示的に考慮。

主要コンポーネント:
- CVaROptimizer: CVaR最小化による配分最適化
- compute_cvar: CVaRの計算
- compute_portfolio_metrics: ポートフォリオリスク指標の計算

設計根拠:
- VaRは閾値のみ、CVaRは閾値を超えた損失の平均を考慮
- テールリスクに敏感なポートフォリオ構築に有効
- 正規分布を仮定しないノンパラメトリック推定

使用例:
    from src.allocation.cvar_optimizer import CVaROptimizer

    optimizer = CVaROptimizer(alpha=0.05, max_weight=0.20)

    # 最適化
    result = optimizer.optimize(returns_df)
    print(f"Optimal weights: {result['weights']}")
    print(f"CVaR 95%: {result['cvar_95']:.4f}")

    # リスク指標計算
    metrics = optimizer.compute_portfolio_metrics(returns_df, weights)
    print(f"VaR 95%: {metrics['var_95']:.4f}")
    print(f"CVaR 99%: {metrics['cvar_99']:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

logger = logging.getLogger(__name__)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class CVaRConfig:
    """CVaR最適化設定

    Attributes:
        alpha: VaR/CVaRの信頼水準（デフォルト0.05 = 95%）
        target_return: 目標リターン制約（Noneで制約なし）
        max_weight: 個別資産の最大ウェイト
        min_weight: 個別資産の最小ウェイト
        allow_short: ショート許可
        risk_free_rate: リスクフリーレート（年率）
    """
    alpha: float = 0.05
    target_return: Optional[float] = None
    max_weight: float = 0.20
    min_weight: float = 0.0
    allow_short: bool = False
    risk_free_rate: float = 0.0


@dataclass
class CVaRResult:
    """CVaR計算結果

    Attributes:
        var: Value at Risk
        cvar: Conditional Value at Risk
        alpha: 信頼水準
        n_tail_observations: テール観測数
    """
    var: float
    cvar: float
    alpha: float
    n_tail_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var": self.var,
            "cvar": self.cvar,
            "alpha": self.alpha,
            "n_tail_observations": self.n_tail_observations,
        }


@dataclass
class OptimizationResult:
    """最適化結果

    Attributes:
        weights: 最適ウェイト
        cvar: 最適化後のCVaR
        var: 最適化後のVaR
        expected_return: 期待リターン（年率）
        volatility: ボラティリティ（年率）
        sharpe_ratio: シャープレシオ
        success: 最適化成功フラグ
        message: 最適化メッセージ
        n_iterations: イテレーション回数
        assets: 資産名リスト
        metrics: 追加のリスク指標
    """
    weights: Dict[str, float]
    cvar: float
    var: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    n_iterations: int
    assets: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "cvar": self.cvar,
            "var": self.var,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "assets": self.assets,
            "metrics": self.metrics,
        }

    def get_weights_array(self) -> np.ndarray:
        """ウェイトを配列として取得"""
        return np.array([self.weights[a] for a in self.assets])


@dataclass
class PortfolioMetrics:
    """ポートフォリオリスク指標

    Attributes:
        var_95: 95% VaR
        cvar_95: 95% CVaR
        var_99: 99% VaR
        cvar_99: 99% CVaR
        expected_return: 期待リターン（年率）
        volatility: ボラティリティ（年率）
        sharpe_ratio: シャープレシオ
        max_drawdown: 最大ドローダウン
        skewness: 歪度
        kurtosis: 尖度
    """
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "var_99": self.var_99,
            "cvar_99": self.cvar_99,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


# =============================================================================
# CVaROptimizer クラス
# =============================================================================

class CVaROptimizer:
    """
    CVaR (Conditional Value at Risk) 最適化

    テールリスクを考慮した配分最適化を行う。
    CVaR（期待ショートフォール）を最小化することで、
    大きな損失に対して頑健なポートフォリオを構築する。

    Usage:
        optimizer = CVaROptimizer(alpha=0.05, max_weight=0.20)

        # 最適化
        result = optimizer.optimize(returns_df)
        print(f"Weights: {result.weights}")
        print(f"CVaR 95%: {result.cvar:.4f}")

        # ターゲットリターン制約付き
        result = optimizer.optimize(
            returns_df,
            target_return=0.10  # 年率10%
        )

        # リスク指標計算
        metrics = optimizer.compute_portfolio_metrics(returns_df, weights)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        target_return: Optional[float] = None,
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        allow_short: bool = False,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        """
        初期化

        Args:
            alpha: VaR/CVaRの信頼水準（デフォルト0.05 = 95% CVaR）
            target_return: 目標リターン制約（年率、Noneで制約なし）
            max_weight: 個別資産の最大ウェイト
            min_weight: 個別資産の最小ウェイト
            allow_short: ショート許可
            risk_free_rate: リスクフリーレート（年率）
            annualization_factor: 年率化係数（デフォルト252営業日）
        """
        self.alpha = alpha
        self.target_return = target_return
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        self.allow_short = allow_short
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        self.config = CVaRConfig(
            alpha=alpha,
            target_return=target_return,
            max_weight=max_weight,
            min_weight=min_weight,
            allow_short=allow_short,
            risk_free_rate=risk_free_rate,
        )

        logger.info(
            f"CVaROptimizer initialized: alpha={alpha}, max_weight={max_weight}"
        )

    def compute_cvar(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        alpha: Optional[float] = None,
    ) -> float:
        """
        CVaR（期待ショートフォール）を計算

        Args:
            returns: リターン行列 (n_observations, n_assets)
            weights: ウェイトベクトル (n_assets,)
            alpha: 信頼水準（Noneでインスタンス設定を使用）

        Returns:
            CVaR（正の値、損失として表現）
        """
        alpha = alpha or self.alpha

        # ポートフォリオリターン
        portfolio_returns = returns @ weights

        # VaR（パーセンタイル）
        var = np.percentile(portfolio_returns, alpha * 100)

        # CVaR（VaR以下のリターンの平均）
        tail_returns = portfolio_returns[portfolio_returns <= var]

        if len(tail_returns) == 0:
            # テール観測がない場合はVaRを使用
            cvar = var
        else:
            cvar = np.mean(tail_returns)

        # 損失として正の値で返す（符号反転）
        return -cvar

    def compute_cvar_detailed(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        alpha: Optional[float] = None,
    ) -> CVaRResult:
        """
        CVaRを詳細に計算

        Args:
            returns: リターン行列
            weights: ウェイトベクトル
            alpha: 信頼水準

        Returns:
            CVaRResult
        """
        alpha = alpha or self.alpha

        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, alpha * 100)
        tail_returns = portfolio_returns[portfolio_returns <= var]

        if len(tail_returns) == 0:
            cvar = var
            n_tail = 0
        else:
            cvar = np.mean(tail_returns)
            n_tail = len(tail_returns)

        return CVaRResult(
            var=-var,  # 損失として正の値
            cvar=-cvar,  # 損失として正の値
            alpha=alpha,
            n_tail_observations=n_tail,
        )

    def _objective(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """最適化の目的関数（CVaR最小化）"""
        return self.compute_cvar(returns, weights)

    def _constraint_sum_to_one(self, weights: np.ndarray) -> float:
        """制約: ウェイトの合計 = 1"""
        return np.sum(weights) - 1.0

    def _constraint_target_return(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        target: float,
    ) -> float:
        """制約: 期待リターン >= ターゲット"""
        daily_target = target / self.annualization_factor
        expected_return = np.mean(returns @ weights)
        return expected_return - daily_target

    def optimize(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        expected_returns: Optional[np.ndarray] = None,
        initial_weights: Optional[np.ndarray] = None,
        target_return: Optional[float] = None,
    ) -> OptimizationResult:
        """
        CVaR最小化による配分最適化

        Args:
            returns: リターンデータ (DataFrame or ndarray)
            expected_returns: 期待リターン（オプション、予測値を使用する場合）
            initial_weights: 初期ウェイト（Noneで均等配分）
            target_return: 目標リターン制約（インスタンス設定をオーバーライド）

        Returns:
            OptimizationResult
        """
        # データ準備
        if isinstance(returns, pd.DataFrame):
            assets = returns.columns.tolist()
            returns_array = returns.values
        else:
            assets = [f"Asset_{i}" for i in range(returns.shape[1])]
            returns_array = returns

        n_assets = returns_array.shape[1]

        # 初期ウェイト
        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # 制約
        constraints = [
            {"type": "eq", "fun": self._constraint_sum_to_one}
        ]

        # ターゲットリターン制約
        target = target_return if target_return is not None else self.target_return
        if target is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: self._constraint_target_return(
                    w, returns_array, target
                ),
            })

        # 境界条件
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # 最適化実行
        result = minimize(
            fun=self._objective,
            x0=initial_weights,
            args=(returns_array,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        # 結果を正規化（数値誤差対応）
        optimal_weights = result.x
        optimal_weights = np.clip(optimal_weights, self.min_weight, self.max_weight)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        # リスク指標計算
        cvar_result = self.compute_cvar_detailed(returns_array, optimal_weights)

        # 期待リターンとボラティリティ
        portfolio_returns = returns_array @ optimal_weights
        daily_return = np.mean(portfolio_returns)
        daily_vol = np.std(portfolio_returns)

        annual_return = daily_return * self.annualization_factor
        annual_vol = daily_vol * np.sqrt(self.annualization_factor)

        # シャープレシオ
        sharpe = (
            (annual_return - self.risk_free_rate) / annual_vol
            if annual_vol > 0 else 0.0
        )

        # ウェイト辞書
        weights_dict = {
            asset: float(w) for asset, w in zip(assets, optimal_weights)
        }

        logger.info(
            f"CVaR optimization {'succeeded' if result.success else 'failed'}: "
            f"CVaR={cvar_result.cvar:.4f}, Sharpe={sharpe:.2f}"
        )

        return OptimizationResult(
            weights=weights_dict,
            cvar=cvar_result.cvar,
            var=cvar_result.var,
            expected_return=annual_return,
            volatility=annual_vol,
            sharpe_ratio=sharpe,
            success=result.success,
            message=result.message,
            n_iterations=result.nit,
            assets=assets,
            metrics={
                "cvar_95": cvar_result.cvar,
                "var_95": cvar_result.var,
                "n_tail_observations": cvar_result.n_tail_observations,
            },
        )

    def compute_portfolio_metrics(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        weights: Union[Dict[str, float], np.ndarray],
    ) -> PortfolioMetrics:
        """
        ポートフォリオのリスク指標を計算

        Args:
            returns: リターンデータ
            weights: ウェイト（辞書または配列）

        Returns:
            PortfolioMetrics
        """
        # データ準備
        if isinstance(returns, pd.DataFrame):
            returns_array = returns.values
            if isinstance(weights, dict):
                weights_array = np.array([weights[col] for col in returns.columns])
            else:
                weights_array = weights
        else:
            returns_array = returns
            if isinstance(weights, dict):
                weights_array = np.array(list(weights.values()))
            else:
                weights_array = weights

        # ポートフォリオリターン
        portfolio_returns = returns_array @ weights_array

        # VaR/CVaR計算（95%と99%）
        cvar_95 = self.compute_cvar_detailed(returns_array, weights_array, alpha=0.05)
        cvar_99 = self.compute_cvar_detailed(returns_array, weights_array, alpha=0.01)

        # 基本統計
        daily_return = np.mean(portfolio_returns)
        daily_vol = np.std(portfolio_returns)

        annual_return = daily_return * self.annualization_factor
        annual_vol = daily_vol * np.sqrt(self.annualization_factor)

        sharpe = (
            (annual_return - self.risk_free_rate) / annual_vol
            if annual_vol > 0 else 0.0
        )

        # 最大ドローダウン
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # 歪度と尖度
        skewness = float(pd.Series(portfolio_returns).skew())
        kurtosis = float(pd.Series(portfolio_returns).kurtosis())

        return PortfolioMetrics(
            var_95=cvar_95.var,
            cvar_95=cvar_95.cvar,
            var_99=cvar_99.var,
            cvar_99=cvar_99.cvar,
            expected_return=annual_return,
            volatility=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def efficient_frontier(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        n_points: int = 20,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
    ) -> List[OptimizationResult]:
        """
        CVaR効率的フロンティアを計算

        Args:
            returns: リターンデータ
            n_points: フロンティア上の点数
            min_return: 最小目標リターン（年率）
            max_return: 最大目標リターン（年率）

        Returns:
            OptimizationResultのリスト
        """
        # データ準備
        if isinstance(returns, pd.DataFrame):
            returns_array = returns.values
        else:
            returns_array = returns

        # リターン範囲を推定
        asset_returns = np.mean(returns_array, axis=0) * self.annualization_factor

        if min_return is None:
            min_return = np.min(asset_returns)
        if max_return is None:
            max_return = np.max(asset_returns)

        target_returns = np.linspace(min_return, max_return, n_points)

        frontier = []
        for target in target_returns:
            try:
                result = self.optimize(returns, target_return=target)
                if result.success:
                    frontier.append(result)
            except Exception as e:
                logger.warning(f"Failed to optimize for target={target:.4f}: {e}")

        logger.info(f"Computed efficient frontier with {len(frontier)} points")
        return frontier


# =============================================================================
# 便利関数
# =============================================================================

def compute_cvar(
    returns: Union[pd.DataFrame, np.ndarray],
    weights: Union[Dict[str, float], np.ndarray],
    alpha: float = 0.05,
) -> float:
    """
    CVaRを計算（便利関数）

    Args:
        returns: リターンデータ
        weights: ウェイト
        alpha: 信頼水準

    Returns:
        CVaR（正の値、損失として表現）
    """
    optimizer = CVaROptimizer(alpha=alpha)

    if isinstance(returns, pd.DataFrame):
        returns_array = returns.values
        if isinstance(weights, dict):
            weights_array = np.array([weights[col] for col in returns.columns])
        else:
            weights_array = weights
    else:
        returns_array = returns
        if isinstance(weights, dict):
            weights_array = np.array(list(weights.values()))
        else:
            weights_array = weights

    return optimizer.compute_cvar(returns_array, weights_array)


def optimize_cvar(
    returns: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05,
    max_weight: float = 0.20,
    target_return: Optional[float] = None,
) -> Dict[str, float]:
    """
    CVaR最適化（便利関数）

    Args:
        returns: リターンデータ
        alpha: 信頼水準
        max_weight: 最大ウェイト
        target_return: 目標リターン

    Returns:
        最適ウェイト辞書
    """
    optimizer = CVaROptimizer(
        alpha=alpha,
        max_weight=max_weight,
        target_return=target_return,
    )
    result = optimizer.optimize(returns)
    return result.weights


def create_cvar_optimizer(
    alpha: float = 0.05,
    target_return: Optional[float] = None,
    max_weight: float = 0.20,
    **kwargs,
) -> CVaROptimizer:
    """
    CVaROptimizerを作成（ファクトリ関数）

    Args:
        alpha: 信頼水準
        target_return: 目標リターン
        max_weight: 最大ウェイト
        **kwargs: 追加引数

    Returns:
        CVaROptimizer
    """
    return CVaROptimizer(
        alpha=alpha,
        target_return=target_return,
        max_weight=max_weight,
        **kwargs,
    )
