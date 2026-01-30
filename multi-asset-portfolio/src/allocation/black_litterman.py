"""
Black-Litterman Model - ブラック・リターマンモデル

このモジュールは、市場均衡リターンと主観的ビューを統合する
Black-Littermanモデルを提供する。

主要コンポーネント:
- BlackLittermanModel: BLモデルのコア実装
- ViewGenerator: シグナルからビューを生成

設計根拠:
- 市場均衡からの出発点により安定した推定
- 主観的ビューの不確実性を明示的にモデル化
- シグナルベースのビュー生成で定量的アプローチを実現

参考文献:
- Black, F. and Litterman, R. (1992) "Global Portfolio Optimization"
- Idzorek, T. (2005) "A Step-by-Step Guide to the Black-Litterman Model"
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
class BlackLittermanConfig:
    """Black-Littermanモデルの設定。

    Attributes:
        risk_aversion: リスク回避係数（δ）
        tau: スケーリング係数（τ）
        default_confidence: デフォルトのビュー信頼度
        max_weight: 最大ウェイト
        signal_threshold: シグナル採用閾値
        return_scale: シグナル→リターン変換スケール
    """

    risk_aversion: float = 2.5
    tau: float = 0.05
    default_confidence: float = 0.5
    max_weight: float = 0.20
    signal_threshold: float = 0.1
    return_scale: float = 0.10


@dataclass
class BlackLittermanResult:
    """Black-Littermanモデルの計算結果。

    Attributes:
        equilibrium_returns: 均衡リターン（π）
        posterior_returns: 事後リターン
        posterior_cov: 事後共分散行列
        optimal_weights: 最適ウェイト
        views_used: 使用されたビュー数
        timestamp: 計算日時
        metadata: 追加情報
    """

    equilibrium_returns: pd.Series
    posterior_returns: pd.Series
    posterior_cov: pd.DataFrame
    optimal_weights: dict[str, float]
    views_used: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weights(self) -> pd.Series:
        """最適ウェイトをSeries形式で返す。"""
        return pd.Series(self.optimal_weights)

    @property
    def n_views(self) -> int:
        """使用されたビュー数を返す。"""
        return self.views_used

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "equilibrium_returns": self.equilibrium_returns.to_dict(),
            "posterior_returns": self.posterior_returns.to_dict(),
            "optimal_weights": self.optimal_weights,
            "views_used": self.views_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ViewSet:
    """ビューセット。

    Attributes:
        P: ピック行列（K x N）- どの資産にビューがあるか
        Q: ビューリターン（K x 1）- 期待リターン
        omega: 不確実性行列（K x K）- ビューの信頼度
        confidences: 各ビューの信頼度
        assets: 資産リスト
    """

    P: np.ndarray
    Q: np.ndarray
    omega: np.ndarray
    confidences: list[float] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        """ビュー数を返す。"""
        return len(self.Q)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "P": self.P.tolist(),
            "Q": self.Q.tolist(),
            "omega_diag": np.diag(self.omega).tolist(),
            "confidences": self.confidences,
            "n_views": self.n_views,
        }


class BlackLittermanModel:
    """Black-Littermanモデル。

    市場均衡リターン（CAPM均衡）と主観的ビューを統合して
    事後期待リターンを計算する。

    Usage:
        model = BlackLittermanModel(risk_aversion=2.5, tau=0.05)

        # 均衡リターン計算
        pi = model.compute_equilibrium_returns(cov, market_weights)

        # ビューとの統合
        posterior = model.combine_views(pi, cov, P, Q, omega)

        # 最適化
        weights = model.optimize(posterior, cov, max_weight=0.20)
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        """初期化。

        Args:
            risk_aversion: リスク回避係数（δ）
            tau: スケーリング係数（τ）- 均衡リターンの不確実性
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

    @classmethod
    def from_config(cls, config: BlackLittermanConfig) -> "BlackLittermanModel":
        """設定からモデルを作成する。

        Args:
            config: BlackLittermanConfig

        Returns:
            初期化された BlackLittermanModel
        """
        return cls(risk_aversion=config.risk_aversion, tau=config.tau)

    def compute_equilibrium_returns(
        self,
        cov: pd.DataFrame,
        market_weights: pd.Series | dict[str, float],
    ) -> pd.Series:
        """均衡リターン（インプライドリターン）を計算する。

        π = δ × Σ × w_mkt

        Args:
            cov: 共分散行列
            market_weights: 市場ウェイト

        Returns:
            均衡リターン
        """
        if isinstance(market_weights, dict):
            market_weights = pd.Series(market_weights)

        # インデックスを揃える
        assets = cov.columns.tolist()
        market_weights = market_weights.reindex(assets).fillna(0.0)

        # π = δΣw
        cov_matrix = cov.values
        weights = market_weights.values
        pi = self.risk_aversion * cov_matrix @ weights

        equilibrium = pd.Series(pi, index=assets)

        logger.debug(
            f"Equilibrium returns: mean={equilibrium.mean():.4f}, "
            f"std={equilibrium.std():.4f}"
        )

        return equilibrium

    def combine_views(
        self,
        pi: pd.Series,
        cov: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray | None = None,
        confidences: list[float] | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """均衡リターンとビューを統合して事後リターンを計算する。

        E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

        Args:
            pi: 均衡リターン
            cov: 共分散行列
            P: ピック行列（K x N）
            Q: ビューリターン（K,）
            omega: 不確実性行列（K x K）、Noneの場合はIdzorek法で計算
            confidences: 各ビューの信頼度（omega計算用）

        Returns:
            (事後リターン, 事後共分散行列)
        """
        assets = cov.columns.tolist()
        n_assets = len(assets)
        n_views = len(Q)

        if n_views == 0:
            # ビューがない場合は均衡リターンをそのまま返す
            return pi, cov * (1 + self.tau)

        # numpy配列に変換
        Sigma = cov.values
        pi_vec = pi.values
        P = np.atleast_2d(P)
        Q = np.atleast_1d(Q)

        # τΣ
        tau_Sigma = self.tau * Sigma

        # Ωの計算（Idzorek法）
        if omega is None:
            omega = self._compute_omega_idzorek(
                P, Sigma, confidences or [0.5] * n_views
            )

        # 逆行列計算
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        omega_inv = np.linalg.inv(omega)

        # 事後精度行列
        # M = (τΣ)^-1 + P'Ω^-1 P
        M = tau_Sigma_inv + P.T @ omega_inv @ P

        # 事後期待リターン
        # E[R] = M^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
        M_inv = np.linalg.inv(M)
        posterior_mean = M_inv @ (tau_Sigma_inv @ pi_vec + P.T @ omega_inv @ Q)

        # 事後共分散
        posterior_cov = M_inv + Sigma

        posterior_returns = pd.Series(posterior_mean, index=assets)
        posterior_cov_df = pd.DataFrame(posterior_cov, index=assets, columns=assets)

        logger.debug(
            f"Posterior returns: mean={posterior_returns.mean():.4f}, "
            f"std={posterior_returns.std():.4f}, views={n_views}"
        )

        return posterior_returns, posterior_cov_df

    def _compute_omega_idzorek(
        self,
        P: np.ndarray,
        Sigma: np.ndarray,
        confidences: list[float],
    ) -> np.ndarray:
        """Idzorek法でΩを計算する。

        各ビューの信頼度に基づいてΩの対角成分を設定。
        confidence=1.0: 完全な確信（Ω→0）
        confidence=0.0: 完全な不確実性（Ω→∞）

        Args:
            P: ピック行列
            Sigma: 共分散行列
            confidences: 各ビューの信頼度（0-1）

        Returns:
            Ω行列
        """
        n_views = P.shape[0]
        omega_diag = np.zeros(n_views)

        for k in range(n_views):
            p_k = P[k, :]
            # ビューの分散 = p' Σ p
            view_variance = p_k @ Sigma @ p_k

            # 信頼度に基づくスケーリング
            # confidence=1 → omega=0, confidence=0 → omega=large
            conf = np.clip(confidences[k], 0.01, 0.99)
            omega_diag[k] = view_variance * (1 - conf) / conf * self.tau

        return np.diag(omega_diag)

    def optimize(
        self,
        posterior_returns: pd.Series,
        cov: pd.DataFrame,
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """事後リターンを使用してSharpe最大化最適化を行う。

        Args:
            posterior_returns: 事後期待リターン
            cov: 共分散行列
            max_weight: 最大ウェイト
            min_weight: 最小ウェイト
            risk_free_rate: リスクフリーレート

        Returns:
            最適ウェイト
        """
        assets = posterior_returns.index.tolist()
        n = len(assets)

        mu = posterior_returns.values
        Sigma = cov.values

        def neg_sharpe(w):
            """負のシャープレシオ（最小化用）"""
            port_return = w @ mu - risk_free_rate
            port_vol = np.sqrt(w @ Sigma @ w)
            if port_vol < 1e-10:
                return 1e10
            return -port_return / port_vol

        # 制約条件
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # 合計=1
        ]

        # 境界条件
        bounds = [(min_weight, max_weight) for _ in range(n)]

        # 初期値（等ウェイト）
        w0 = np.ones(n) / n

        # 最適化
        result = optimize.minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        optimal_weights = dict(zip(assets, result.x))

        logger.debug(
            f"Optimal weights: max={max(result.x):.3f}, "
            f"min={min(result.x):.3f}, sharpe={-result.fun:.4f}"
        )

        return optimal_weights

    def run(
        self,
        cov: pd.DataFrame,
        market_weights: pd.Series | dict[str, float],
        views: ViewSet | None = None,
        max_weight: float = 0.20,
    ) -> BlackLittermanResult:
        """Black-Littermanモデルを実行する。

        Args:
            cov: 共分散行列
            market_weights: 市場ウェイト
            views: ビューセット（オプション）
            max_weight: 最大ウェイト

        Returns:
            計算結果
        """
        # 均衡リターン
        pi = self.compute_equilibrium_returns(cov, market_weights)

        # ビューとの統合
        if views is not None and views.n_views > 0:
            posterior_returns, posterior_cov = self.combine_views(
                pi, cov, views.P, views.Q, views.omega, views.confidences
            )
            views_used = views.n_views
        else:
            posterior_returns = pi
            posterior_cov = cov * (1 + self.tau)
            views_used = 0

        # 最適化
        optimal_weights = self.optimize(
            posterior_returns, posterior_cov, max_weight=max_weight
        )

        return BlackLittermanResult(
            equilibrium_returns=pi,
            posterior_returns=posterior_returns,
            posterior_cov=posterior_cov,
            optimal_weights=optimal_weights,
            views_used=views_used,
            metadata={
                "risk_aversion": self.risk_aversion,
                "tau": self.tau,
                "max_weight": max_weight,
            },
        )


class ViewGenerator:
    """シグナルからビューを生成するクラス。

    トレーディングシグナルをBlack-Littermanモデルの
    ビュー形式に変換する。

    Usage:
        generator = ViewGenerator()
        views = generator.generate_views_from_signals(
            signals={"SPY": 0.5, "TLT": -0.3},
            assets=["SPY", "TLT", "GLD"],
            confidence=0.6,
        )
    """

    DEFAULT_SIGNAL_THRESHOLD = 0.1
    DEFAULT_RETURN_SCALE = 0.10  # シグナル1.0 → 10%リターン

    def __init__(
        self,
        signal_threshold: float | None = None,
        return_scale: float | None = None,
    ) -> None:
        """初期化。

        Args:
            signal_threshold: シグナル採用閾値（絶対値）
            return_scale: シグナル→リターン変換スケール
        """
        self.signal_threshold = signal_threshold or self.DEFAULT_SIGNAL_THRESHOLD
        self.return_scale = return_scale or self.DEFAULT_RETURN_SCALE

    def generate_views_from_signals(
        self,
        signals: dict[str, float] | pd.Series,
        assets: list[str] | None = None,
        confidence: float = 0.5,
        use_relative_views: bool = False,
    ) -> ViewSet:
        """シグナルからビューを生成する。

        Args:
            signals: 資産→シグナルスコア（-1 to +1）
            assets: 全資産リスト（Noneの場合はsignalsのキーを使用）
            confidence: 全ビューの信頼度
            use_relative_views: 相対ビューを使用するか

        Returns:
            ビューセット
        """
        if isinstance(signals, pd.Series):
            signals = signals.to_dict()

        if assets is None:
            assets = list(signals.keys())

        n_assets = len(assets)
        asset_idx = {asset: i for i, asset in enumerate(assets)}

        # 閾値を超えるシグナルのみ採用
        strong_signals = {
            asset: score
            for asset, score in signals.items()
            if abs(score) > self.signal_threshold and asset in asset_idx
        }

        if not strong_signals:
            # ビューなし
            return ViewSet(
                P=np.zeros((0, n_assets)),
                Q=np.array([]),
                omega=np.zeros((0, 0)),
                confidences=[],
                assets=assets,
            )

        if use_relative_views:
            return self._generate_relative_views(
                strong_signals, assets, asset_idx, confidence
            )
        else:
            return self._generate_absolute_views(
                strong_signals, assets, asset_idx, confidence
            )

    def _generate_absolute_views(
        self,
        signals: dict[str, float],
        assets: list[str],
        asset_idx: dict[str, int],
        confidence: float,
    ) -> ViewSet:
        """絶対ビュー（個別資産の期待リターン）を生成する。"""
        n_assets = len(assets)
        n_views = len(signals)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        confidences = []

        for k, (asset, score) in enumerate(signals.items()):
            idx = asset_idx[asset]
            P[k, idx] = 1.0
            Q[k] = score * self.return_scale

            # シグナル強度に応じた信頼度調整
            signal_strength = abs(score)
            adjusted_confidence = confidence * (0.5 + 0.5 * signal_strength)
            confidences.append(adjusted_confidence)

        # Ωは後で計算（combine_viewsで）
        omega = np.eye(n_views) * 0.01  # 仮の値

        return ViewSet(P=P, Q=Q, omega=omega, confidences=confidences, assets=assets)

    def _generate_relative_views(
        self,
        signals: dict[str, float],
        assets: list[str],
        asset_idx: dict[str, int],
        confidence: float,
    ) -> ViewSet:
        """相対ビュー（資産間の相対パフォーマンス）を生成する。"""
        n_assets = len(assets)

        # シグナルを強度順にソート
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_signals) < 2:
            return self._generate_absolute_views(signals, assets, asset_idx, confidence)

        # 上位と下位の比較ビューを生成
        views_list = []
        q_list = []
        confidences = []

        # 上位半分 vs 下位半分
        n = len(sorted_signals)
        top_half = sorted_signals[: n // 2]
        bottom_half = sorted_signals[n // 2 :]

        for top_asset, top_score in top_half:
            for bottom_asset, bottom_score in bottom_half:
                p = np.zeros(n_assets)
                p[asset_idx[top_asset]] = 1.0
                p[asset_idx[bottom_asset]] = -1.0

                # 相対リターン = (top_score - bottom_score) * scale
                q = (top_score - bottom_score) * self.return_scale * 0.5

                views_list.append(p)
                q_list.append(q)

                # 信頼度
                signal_diff = abs(top_score - bottom_score)
                adjusted_confidence = confidence * (0.5 + 0.25 * signal_diff)
                confidences.append(adjusted_confidence)

        P = np.array(views_list)
        Q = np.array(q_list)
        omega = np.eye(len(Q)) * 0.01

        return ViewSet(P=P, Q=Q, omega=omega, confidences=confidences, assets=assets)


def create_black_litterman_model(
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanModel:
    """BlackLittermanModel のファクトリ関数。

    Args:
        risk_aversion: リスク回避係数
        tau: スケーリング係数

    Returns:
        初期化された BlackLittermanModel
    """
    return BlackLittermanModel(risk_aversion=risk_aversion, tau=tau)


def create_view_generator(
    signal_threshold: float = 0.1,
    return_scale: float = 0.10,
) -> ViewGenerator:
    """ViewGenerator のファクトリ関数。

    Args:
        signal_threshold: シグナル採用閾値
        return_scale: リターン変換スケール

    Returns:
        初期化された ViewGenerator
    """
    return ViewGenerator(
        signal_threshold=signal_threshold,
        return_scale=return_scale,
    )


def quick_black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series | dict[str, float],
    signals: dict[str, float] | None = None,
    confidence: float = 0.5,
    max_weight: float = 0.20,
) -> dict[str, float]:
    """便利関数: Black-Littermanモデルを簡易実行する。

    Args:
        cov: 共分散行列
        market_weights: 市場ウェイト
        signals: シグナル（オプション）
        confidence: ビューの信頼度
        max_weight: 最大ウェイト

    Returns:
        最適ウェイト
    """
    model = BlackLittermanModel()
    views = None

    if signals is not None:
        generator = ViewGenerator()
        views = generator.generate_views_from_signals(
            signals=signals,
            assets=cov.columns.tolist(),
            confidence=confidence,
        )

    result = model.run(cov, market_weights, views=views, max_weight=max_weight)
    return result.optimal_weights


def quick_bl_allocation(
    cov: pd.DataFrame,
    market_weights: pd.Series | dict[str, float],
    signals: dict[str, float] | None = None,
    confidence: float = 0.5,
    max_weight: float = 0.20,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """便利関数: Black-Littermanモデルを簡易実行し結果を返す。

    Args:
        cov: 共分散行列
        market_weights: 市場ウェイト
        signals: シグナル（オプション）
        confidence: ビューの信頼度
        max_weight: 最大ウェイト
        risk_aversion: リスク回避係数
        tau: スケーリング係数

    Returns:
        BlackLittermanResult
    """
    model = BlackLittermanModel(risk_aversion=risk_aversion, tau=tau)
    views = None

    if signals is not None:
        generator = ViewGenerator()
        views = generator.generate_views_from_signals(
            signals=signals,
            assets=cov.columns.tolist(),
            confidence=confidence,
        )

    return model.run(cov, market_weights, views=views, max_weight=max_weight)
