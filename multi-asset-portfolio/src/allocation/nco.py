"""
Nested Clustered Optimization (NCO) Module - ネステッドクラスタ最適化

López de Prado (2019) の改良版HRP:
1. 階層的クラスタリングでアセットをグループ化
2. 各クラスタ内で平均分散最適化
3. クラスタ間でリスクパリティ配分

利点:
- HRPの分散化効果を維持
- クラスタ内で最適化することで効率性向上
- 推定誤差に対する頑健性を維持

アルゴリズム:
1. 相関行列から距離行列を計算: dist = sqrt(0.5 * (1 - corr))
2. 階層クラスタリング（Ward法）で n_clusters に分割
3. 各クラスタ内で最小分散最適化
4. クラスタ間で逆ボラティリティ加重（リスクパリティ）
5. 最大ウェイト制約を適用

参考文献:
- López de Prado, M. (2019). A Robust Estimator of the Efficient Frontier
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample

使用方法:
    from src.allocation.nco import NestedClusteredOptimization, NCOConfig

    nco = NestedClusteredOptimization(NCOConfig(
        n_clusters=5,
        intra_cluster_method="min_variance",
        inter_cluster_method="risk_parity",
        max_weight=0.20,
    ))
    result = nco.fit(returns_df)
    weights = result.weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# =============================================================================
# Enum定義
# =============================================================================

class IntraClusterMethod(str, Enum):
    """クラスタ内最適化手法"""
    MIN_VARIANCE = "min_variance"      # 最小分散
    MAX_SHARPE = "max_sharpe"          # 最大シャープ比
    EQUAL_WEIGHT = "equal_weight"      # 等重み
    RISK_PARITY = "risk_parity"        # リスクパリティ


class InterClusterMethod(str, Enum):
    """クラスタ間配分手法"""
    RISK_PARITY = "risk_parity"        # リスクパリティ（逆ボラ加重）
    EQUAL_WEIGHT = "equal_weight"      # 等重み
    MIN_VARIANCE = "min_variance"      # 最小分散


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class NCOConfig:
    """NCO設定

    Attributes:
        n_clusters: クラスタ数
        intra_cluster_method: クラスタ内最適化手法
        inter_cluster_method: クラスタ間配分手法
        max_weight: 最大ウェイト制約
        min_weight: 最小ウェイト制約
        linkage_method: クラスタリング手法
        regularization: 共分散行列の正則化係数
        max_iter: 最適化の最大反復回数
    """
    n_clusters: int = 5
    intra_cluster_method: str = "min_variance"
    inter_cluster_method: str = "risk_parity"
    max_weight: float = 0.20
    min_weight: float = 0.0
    linkage_method: str = "ward"
    regularization: float = 1e-6
    max_iter: int = 1000

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if not 0 <= self.max_weight <= 1:
            raise ValueError("max_weight must be in [0, 1]")
        if not 0 <= self.min_weight <= self.max_weight:
            raise ValueError("min_weight must be in [0, max_weight]")

        valid_linkage = {"ward", "single", "complete", "average"}
        if self.linkage_method not in valid_linkage:
            raise ValueError(f"linkage_method must be one of {valid_linkage}")


@dataclass
class ClusterInfo:
    """クラスタ情報

    Attributes:
        cluster_id: クラスタID
        assets: クラスタ内のアセットリスト
        weights_intra: クラスタ内重み（正規化済み）
        variance: クラスタの分散
        weight_inter: クラスタ間重み
    """
    cluster_id: int
    assets: list[str]
    weights_intra: dict[str, float] = field(default_factory=dict)
    variance: float = 0.0
    weight_inter: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "cluster_id": self.cluster_id,
            "assets": self.assets,
            "weights_intra": self.weights_intra,
            "variance": self.variance,
            "weight_inter": self.weight_inter,
        }


@dataclass
class NCOResult:
    """NCO配分結果

    Attributes:
        weights: アセット別重み（Series）
        clusters: クラスタ情報リスト
        cluster_assignments: アセット→クラスタIDのマッピング
        linkage_matrix: クラスタリングのリンケージ行列
        metadata: 追加メタデータ
    """
    weights: pd.Series
    clusters: list[ClusterInfo] = field(default_factory=list)
    cluster_assignments: dict[str, int] = field(default_factory=dict)
    linkage_matrix: NDArray[np.float64] | None = None
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
        return True

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights.to_dict(),
            "clusters": [c.to_dict() for c in self.clusters],
            "cluster_assignments": self.cluster_assignments,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


# =============================================================================
# メインクラス
# =============================================================================

class NestedClusteredOptimization:
    """Nested Clustered Optimization (NCO)

    López de Prado (2019) の改良版HRP。
    クラスタ内で最適化、クラスタ間でリスクパリティ配分。

    Usage:
        nco = NestedClusteredOptimization(NCOConfig(
            n_clusters=5,
            intra_cluster_method="min_variance",
            inter_cluster_method="risk_parity",
            max_weight=0.20,
        ))

        result = nco.fit(returns_df)
        print(result.weights)
    """

    def __init__(self, config: NCOConfig | None = None) -> None:
        """初期化

        Args:
            config: NCO設定
        """
        self.config = config or NCOConfig()

        # 内部状態
        self._cov: pd.DataFrame | None = None
        self._corr: pd.DataFrame | None = None
        self._assets: list[str] = []
        self._linkage_matrix: NDArray[np.float64] | None = None
        self._cluster_labels: NDArray[np.int32] | None = None

        # クラスタ内最適化キャッシュ
        self._cluster_weights_cache: dict[str, dict[str, float]] = {}
        self._cluster_variance_cache: dict[str, float] = {}
        self._cache_max_size: int = 1000

        # キャッシュ統計
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def fit(self, returns: pd.DataFrame) -> NCOResult:
        """リターンデータから最適配分を計算

        Args:
            returns: リターンデータ（列=アセット、行=日付）

        Returns:
            NCOResult: 配分結果
        """
        if returns.empty:
            logger.warning("Empty returns data")
            return NCOResult(weights=pd.Series(dtype=float))

        self._assets = list(returns.columns)
        n_assets = len(self._assets)

        # アセット数がクラスタ数より少ない場合は調整
        n_clusters = min(self.config.n_clusters, n_assets)
        if n_clusters < 2:
            # クラスタリング不可、等重み
            weights = pd.Series(1.0 / n_assets, index=self._assets)
            return NCOResult(
                weights=weights,
                metadata={"reason": "Too few assets for clustering"},
            )

        # 共分散行列と相関行列を計算
        self._cov = returns.cov()
        self._corr = returns.corr()

        # 正則化
        if self.config.regularization > 0:
            reg_matrix = np.eye(n_assets) * self.config.regularization
            self._cov = self._cov + reg_matrix

        # 距離行列計算
        dist_matrix = self._compute_distance_matrix(self._corr)

        # 階層クラスタリング
        try:
            # 距離行列を縮約形式に変換
            dist_condensed = dist_matrix[np.triu_indices(n_assets, k=1)]
            self._linkage_matrix = linkage(dist_condensed, method=self.config.linkage_method)

            # クラスタ割り当て
            self._cluster_labels = fcluster(
                self._linkage_matrix,
                t=n_clusters,
                criterion="maxclust",
            )
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, falling back to equal weight")
            weights = pd.Series(1.0 / n_assets, index=self._assets)
            return NCOResult(
                weights=weights,
                metadata={"reason": f"Clustering failed: {e}"},
            )

        # クラスタ情報を構築
        clusters = self._build_cluster_info()

        # クラスタ内最適化
        for cluster in clusters:
            self._optimize_intra_cluster(cluster)

        # クラスタ間配分
        self._allocate_inter_cluster(clusters)

        # 最終重みを計算
        final_weights = self._compute_final_weights(clusters)

        # 制約適用
        final_weights = self._apply_constraints(final_weights)

        # 結果作成
        result = NCOResult(
            weights=final_weights,
            clusters=clusters,
            cluster_assignments={
                asset: int(self._cluster_labels[i])
                for i, asset in enumerate(self._assets)
            },
            linkage_matrix=self._linkage_matrix,
            metadata={
                "n_clusters": n_clusters,
                "n_assets": n_assets,
                "intra_method": self.config.intra_cluster_method,
                "inter_method": self.config.inter_cluster_method,
            },
        )

        logger.info(
            "NCO allocation completed",
            extra={
                "n_clusters": n_clusters,
                "n_assets": n_assets,
            },
        )

        return result

    def _compute_distance_matrix(self, corr: pd.DataFrame) -> NDArray[np.float64]:
        """相関行列から距離行列を計算

        距離 = sqrt(0.5 * (1 - corr))

        Args:
            corr: 相関行列

        Returns:
            距離行列
        """
        # 相関を[-1, 1]にクリップ
        corr_clipped = np.clip(corr.values, -1.0, 1.0)

        # 距離計算: dist = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(0.5 * (1 - corr_clipped))

        # 対角成分は0
        np.fill_diagonal(dist, 0.0)

        return dist

    def _build_cluster_info(self) -> list[ClusterInfo]:
        """クラスタ情報を構築

        Returns:
            ClusterInfoのリスト
        """
        clusters: list[ClusterInfo] = []
        unique_labels = np.unique(self._cluster_labels)

        for label in unique_labels:
            mask = self._cluster_labels == label
            assets = [self._assets[i] for i in np.where(mask)[0]]

            cluster = ClusterInfo(
                cluster_id=int(label),
                assets=assets,
            )
            clusters.append(cluster)

        return clusters

    def _hash_cluster_cov(
        self,
        assets: list[str],
        cov_cluster: NDArray[np.float64],
    ) -> str:
        """クラスタ共分散行列をハッシュ化

        固有値ベースのハッシュで類似した共分散行列を
        同一キーにマップし、キャッシュヒット率を向上。

        Args:
            assets: アセットリスト
            cov_cluster: クラスタ内共分散行列

        Returns:
            ハッシュ文字列
        """
        import hashlib

        n = len(assets)

        # 固有値を計算
        try:
            eigenvalues = np.linalg.eigvalsh(cov_cluster)
            eigenvalue_sum = np.round(eigenvalues.sum(), 4)
            eigenvalue_var = np.round(eigenvalues.var(), 6)
        except np.linalg.LinAlgError:
            eigenvalue_sum = 0.0
            eigenvalue_var = 0.0

        # 共分散行列の統計量
        cov_mean = np.round(cov_cluster.mean(), 6)
        cov_trace = np.round(np.trace(cov_cluster), 4)

        # アセット名のソートハッシュ（順序の影響を排除）
        assets_hash = hashlib.md5("_".join(sorted(assets)).encode()).hexdigest()[:8]

        # 複合キー
        key = (
            f"{n}_"
            f"ev{eigenvalue_sum}_"
            f"var{eigenvalue_var}_"
            f"mean{cov_mean}_"
            f"tr{cov_trace}_"
            f"a{assets_hash}_"
            f"{self.config.intra_cluster_method}"
        )

        return hashlib.md5(key.encode()).hexdigest()

    def _manage_cache_size(self) -> None:
        """キャッシュサイズを管理（LRU風に古いエントリを削除）"""
        if len(self._cluster_weights_cache) > self._cache_max_size:
            # 最初の10%を削除
            n_remove = self._cache_max_size // 10
            keys_to_remove = list(self._cluster_weights_cache.keys())[:n_remove]
            for key in keys_to_remove:
                self._cluster_weights_cache.pop(key, None)
                self._cluster_variance_cache.pop(key, None)

    def _optimize_intra_cluster(self, cluster: ClusterInfo) -> None:
        """クラスタ内最適化（キャッシュ付き）

        Args:
            cluster: クラスタ情報（更新される）

        Note:
            キャッシュを使用してクラスタ内最適化結果を再利用。
            類似した共分散行列は同一の最適化結果を返す。
        """
        assets = cluster.assets
        n = len(assets)

        if n == 1:
            # 単一アセット
            cluster.weights_intra = {assets[0]: 1.0}
            cluster.variance = float(self._cov.loc[assets[0], assets[0]])
            return

        # クラスタ内共分散行列を抽出
        cov_cluster = self._cov.loc[assets, assets].values

        # キャッシュキーを生成
        cache_key = self._hash_cluster_cov(assets, cov_cluster)

        # キャッシュヒット確認
        if cache_key in self._cluster_weights_cache:
            cached_weights = self._cluster_weights_cache[cache_key]
            cached_variance = self._cluster_variance_cache[cache_key]

            # アセット名をマッピング（キャッシュはソート順で保存）
            sorted_assets = sorted(assets)
            cluster.weights_intra = {
                asset: cached_weights.get(asset, 0.0) for asset in assets
            }
            cluster.variance = cached_variance
            self._cache_hits += 1
            return

        self._cache_misses += 1

        # 最適化手法に応じて処理
        if self.config.intra_cluster_method == "min_variance":
            weights = self._min_variance_weights(cov_cluster)
        elif self.config.intra_cluster_method == "equal_weight":
            weights = np.ones(n) / n
        elif self.config.intra_cluster_method == "risk_parity":
            weights = self._risk_parity_weights(cov_cluster)
        else:
            weights = np.ones(n) / n

        # クラスタの分散を計算
        cluster_var = float(weights @ cov_cluster @ weights)

        # 結果を格納
        cluster.weights_intra = {
            asset: float(w) for asset, w in zip(assets, weights)
        }
        cluster.variance = cluster_var

        # キャッシュに保存
        self._cluster_weights_cache[cache_key] = cluster.weights_intra.copy()
        self._cluster_variance_cache[cache_key] = cluster_var

        # キャッシュサイズ管理
        self._manage_cache_size()

    def _min_variance_weights(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """最小分散ポートフォリオの重みを計算

        Args:
            cov: 共分散行列

        Returns:
            重み配列
        """
        n = cov.shape[0]

        # 目的関数: ポートフォリオ分散
        def objective(w: NDArray[np.float64]) -> float:
            return float(w @ cov @ w)

        # 制約: 重みの合計 = 1
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        # 境界: 0 <= w <= 1
        bounds = [(0.0, 1.0) for _ in range(n)]

        # 初期値: 等重み
        w0 = np.ones(n) / n

        # 最適化
        try:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.config.max_iter, "ftol": 1e-10},
            )

            if result.success:
                weights = result.x
                # 負の重みを0にクリップして正規化
                weights = np.clip(weights, 0.0, None)
                weights = weights / weights.sum()
                return weights
        except Exception as e:
            logger.warning(f"Min variance optimization failed: {e}")

        # フォールバック: 等重み
        return np.ones(n) / n

    def _risk_parity_weights(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """リスクパリティ重みを計算

        Args:
            cov: 共分散行列

        Returns:
            重み配列
        """
        n = cov.shape[0]

        # 各アセットのボラティリティ
        vols = np.sqrt(np.diag(cov))
        vols = np.where(vols > 1e-10, vols, 1e-10)

        # 逆ボラティリティ加重
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        return weights

    def _allocate_inter_cluster(self, clusters: list[ClusterInfo]) -> None:
        """クラスタ間配分

        Args:
            clusters: クラスタ情報リスト（更新される）
        """
        if not clusters:
            return

        # クラスタの分散を取得
        cluster_vars = np.array([c.variance for c in clusters])
        cluster_vars = np.where(cluster_vars > 1e-10, cluster_vars, 1e-10)

        if self.config.inter_cluster_method == "risk_parity":
            # 逆ボラティリティ加重
            cluster_vols = np.sqrt(cluster_vars)
            inv_vols = 1.0 / cluster_vols
            inter_weights = inv_vols / inv_vols.sum()

        elif self.config.inter_cluster_method == "equal_weight":
            # 等重み
            inter_weights = np.ones(len(clusters)) / len(clusters)

        elif self.config.inter_cluster_method == "min_variance":
            # 最小分散（クラスタを独立と仮定）
            inv_vars = 1.0 / cluster_vars
            inter_weights = inv_vars / inv_vars.sum()

        else:
            inter_weights = np.ones(len(clusters)) / len(clusters)

        # クラスタに割り当て
        for cluster, weight in zip(clusters, inter_weights):
            cluster.weight_inter = float(weight)

    def _compute_final_weights(self, clusters: list[ClusterInfo]) -> pd.Series:
        """最終重みを計算

        Args:
            clusters: クラスタ情報リスト

        Returns:
            アセット別重み
        """
        weights_dict: dict[str, float] = {}

        for cluster in clusters:
            inter_weight = cluster.weight_inter
            for asset, intra_weight in cluster.weights_intra.items():
                weights_dict[asset] = inter_weight * intra_weight

        weights = pd.Series(weights_dict)

        # 正規化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """制約を適用

        Args:
            weights: 重み

        Returns:
            制約適用後の重み
        """
        # 最大ウェイト制約
        if self.config.max_weight < 1.0:
            # 最大値を超えている重みをクリップ
            excess_mask = weights > self.config.max_weight
            if excess_mask.any():
                excess = (weights[excess_mask] - self.config.max_weight).sum()
                weights = weights.clip(upper=self.config.max_weight)

                # 超過分を他のアセットに再分配
                non_excess_mask = ~excess_mask
                if non_excess_mask.any():
                    remaining = weights[non_excess_mask]
                    if remaining.sum() > 0:
                        redistribution = excess * remaining / remaining.sum()
                        weights.loc[non_excess_mask] += redistribution

        # 最小ウェイト制約
        if self.config.min_weight > 0:
            below_min_mask = weights < self.config.min_weight
            if below_min_mask.any():
                weights.loc[below_min_mask] = self.config.min_weight

        # 正規化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def get_cluster_summary(self) -> dict[str, Any]:
        """クラスタサマリーを取得

        Returns:
            クラスタサマリー辞書
        """
        if self._cluster_labels is None:
            return {}

        summary: dict[str, Any] = {
            "n_clusters": len(np.unique(self._cluster_labels)),
            "n_assets": len(self._assets),
            "cluster_sizes": {},
        }

        for label in np.unique(self._cluster_labels):
            mask = self._cluster_labels == label
            assets = [self._assets[i] for i in np.where(mask)[0]]
            summary["cluster_sizes"][int(label)] = len(assets)

        return summary


# =============================================================================
# 便利関数
# =============================================================================

def create_nco_from_settings(settings: Any = None) -> NestedClusteredOptimization:
    """設定からNCOを作成

    Args:
        settings: 設定オブジェクト

    Returns:
        NestedClusteredOptimization
    """
    if settings is None:
        return NestedClusteredOptimization()

    # 設定から NCO パラメータを取得
    nco_config = getattr(settings, "nco", None)
    if nco_config is None:
        return NestedClusteredOptimization()

    config = NCOConfig(
        n_clusters=getattr(nco_config, "n_clusters", 5),
        intra_cluster_method=getattr(nco_config, "intra_cluster_method", "min_variance"),
        inter_cluster_method=getattr(nco_config, "inter_cluster_method", "risk_parity"),
        max_weight=getattr(nco_config, "max_weight", 0.20),
        min_weight=getattr(nco_config, "min_weight", 0.0),
        linkage_method=getattr(nco_config, "linkage_method", "ward"),
    )

    return NestedClusteredOptimization(config)


def quick_nco_allocation(
    returns: pd.DataFrame,
    n_clusters: int = 5,
    max_weight: float = 0.20,
) -> pd.Series:
    """簡易NCO配分（便利関数）

    Args:
        returns: リターンデータ
        n_clusters: クラスタ数
        max_weight: 最大ウェイト

    Returns:
        アセット別重み
    """
    config = NCOConfig(
        n_clusters=n_clusters,
        max_weight=max_weight,
    )
    nco = NestedClusteredOptimization(config)
    result = nco.fit(returns)
    return result.weights
