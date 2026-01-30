"""
Streaming Covariance Estimation - ストリーミング共分散計算

大規模ユニバース（800+銘柄）でのメモリ効率的な共分散行列計算。
Welfordのオンラインアルゴリズムを使用し、チャンク単位で更新可能。

Key Features:
- O(n²)メモリを維持しつつ、全データを一度に保持しない
- チャンク単位での逐次更新
- Ledoit-Wolf風shrinkage対応
- 指数加重移動共分散（EWMA）オプション

Based on HI-005: Streaming covariance for large universe support.

Expected Effect:
- Memory usage: 50% reduction
- Support for 800+ symbols

Usage:
    cov_estimator = StreamingCovariance(n_assets=100)

    # バッチ更新
    for chunk in data_chunks:
        cov_estimator.update(chunk)

    # 共分散行列取得
    cov_matrix = cov_estimator.get_covariance()

    # shrinkage適用
    shrunk_cov = cov_estimator.shrink(shrinkage=0.1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class StreamingCovarianceConfig:
    """ストリーミング共分散設定

    Attributes:
        chunk_size: 推奨チャンクサイズ（参考値）
        shrinkage_target: shrinkage先（"identity", "diagonal", "constant_corr"）
        auto_shrinkage: 自動shrinkage強度計算
        ewma_halflife: EWMA半減期（None=単純平均）
        min_samples: 最小サンプル数
    """
    chunk_size: int = 100
    shrinkage_target: str = "identity"
    auto_shrinkage: bool = True
    ewma_halflife: int | None = None
    min_samples: int = 30

    def __post_init__(self) -> None:
        valid_targets = {"identity", "diagonal", "constant_corr"}
        if self.shrinkage_target not in valid_targets:
            raise ValueError(
                f"Invalid shrinkage_target: {self.shrinkage_target}. "
                f"Must be one of {valid_targets}"
            )


@dataclass
class StreamingCovarianceResult:
    """共分散計算結果

    Attributes:
        covariance: 共分散行列
        correlation: 相関行列
        n_samples: サンプル数
        shrinkage_intensity: 適用されたshrinkage強度
        is_valid: 有効な結果かどうか
    """
    covariance: NDArray[np.float64]
    correlation: NDArray[np.float64]
    n_samples: int
    shrinkage_intensity: float = 0.0
    is_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(
        self,
        columns: list[str] | pd.Index | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """DataFrameに変換

        Args:
            columns: カラム名

        Returns:
            (covariance_df, correlation_df)
        """
        if columns is None:
            columns = [f"asset_{i}" for i in range(self.covariance.shape[0])]

        cov_df = pd.DataFrame(self.covariance, index=columns, columns=columns)
        corr_df = pd.DataFrame(self.correlation, index=columns, columns=columns)
        return cov_df, corr_df


class StreamingCovariance:
    """
    ストリーミング共分散推定クラス

    Welfordのオンラインアルゴリズムを使用して、
    チャンク単位で共分散行列を更新する。

    Memory Efficiency:
    - 全リターンデータを保持しない
    - 必要なのはrunning_mean (n,)とrunning_cov (n,n)のみ
    - n=800でも約5MBのメモリで済む

    Algorithm:
    - Welford's online algorithm for mean
    - Parallel algorithm for covariance (Chan et al.)

    Usage:
        estimator = StreamingCovariance(n_assets=800)

        # リターンデータをチャンクで処理
        for i in range(0, len(returns), chunk_size):
            chunk = returns[i:i+chunk_size]
            estimator.update(chunk)

        # 結果取得
        cov = estimator.get_covariance()
        corr = estimator.get_correlation()
    """

    def __init__(
        self,
        n_assets: int | None = None,
        config: StreamingCovarianceConfig | None = None,
        asset_names: list[str] | pd.Index | None = None,
    ) -> None:
        """初期化

        Args:
            n_assets: アセット数（初回update時に自動検出も可能）
            config: 設定
            asset_names: アセット名
        """
        self.config = config or StreamingCovarianceConfig()
        self._n_assets = n_assets
        self._asset_names = asset_names

        # ランニング統計量
        self._running_mean: NDArray[np.float64] | None = None
        self._running_cov: NDArray[np.float64] | None = None
        self._n_samples: int = 0

        # EWMA用
        self._ewma_weights: NDArray[np.float64] | None = None
        self._ewma_sum: float = 0.0

    @property
    def n_assets(self) -> int:
        """アセット数"""
        return self._n_assets or 0

    @property
    def n_samples(self) -> int:
        """サンプル数"""
        return self._n_samples

    @property
    def is_initialized(self) -> bool:
        """初期化済みかどうか"""
        return self._running_mean is not None

    def _initialize(self, n_assets: int) -> None:
        """内部状態を初期化

        Args:
            n_assets: アセット数
        """
        self._n_assets = n_assets
        self._running_mean = np.zeros(n_assets, dtype=np.float64)
        self._running_cov = np.zeros((n_assets, n_assets), dtype=np.float64)
        self._n_samples = 0

        logger.debug(f"StreamingCovariance initialized for {n_assets} assets")

    def reset(self) -> None:
        """状態をリセット"""
        if self._n_assets is not None:
            self._initialize(self._n_assets)
        else:
            self._running_mean = None
            self._running_cov = None
            self._n_samples = 0

    def update(self, returns_chunk: NDArray[np.float64] | pd.DataFrame) -> None:
        """リターンチャンクで統計量を更新

        Welfordのオンラインアルゴリズムのバッチ版を使用。

        Args:
            returns_chunk: リターンデータ (n_samples, n_assets)
        """
        # DataFrameの場合は変換
        if isinstance(returns_chunk, pd.DataFrame):
            if self._asset_names is None:
                self._asset_names = returns_chunk.columns
            returns_chunk = returns_chunk.values

        # NaN処理
        if np.any(np.isnan(returns_chunk)):
            # 行ごとにNaN含む行を除外
            valid_mask = ~np.any(np.isnan(returns_chunk), axis=1)
            returns_chunk = returns_chunk[valid_mask]
            if len(returns_chunk) == 0:
                logger.warning("All rows contain NaN, skipping update")
                return

        n_new, n_assets = returns_chunk.shape

        # 初回初期化
        if not self.is_initialized:
            self._initialize(n_assets)

        # アセット数チェック
        if n_assets != self._n_assets:
            raise ValueError(
                f"Asset count mismatch: expected {self._n_assets}, got {n_assets}"
            )

        # バッチ平均
        chunk_mean = np.mean(returns_chunk, axis=0)

        if self.config.ewma_halflife is not None:
            # EWMA更新
            self._update_ewma(returns_chunk, chunk_mean, n_new)
        else:
            # 標準的なWelford更新
            self._update_welford(returns_chunk, chunk_mean, n_new)

    def _update_welford(
        self,
        returns_chunk: NDArray[np.float64],
        chunk_mean: NDArray[np.float64],
        n_new: int,
    ) -> None:
        """Welfordのオンラインアルゴリズムで更新

        並列アルゴリズム（Chan et al.）を使用して、
        バッチ全体を一度に処理。

        Reference:
        Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983).
        Algorithms for computing the sample variance.
        """
        n_old = self._n_samples
        n_total = n_old + n_new

        # デルタ
        delta = chunk_mean - self._running_mean

        # 平均更新
        self._running_mean = (
            n_old * self._running_mean + n_new * chunk_mean
        ) / n_total

        # チャンク内共分散
        chunk_centered = returns_chunk - chunk_mean
        chunk_cov = chunk_centered.T @ chunk_centered

        # 共分散更新（並列アルゴリズム）
        # C_total = C_old + C_new + (n_old * n_new / n_total) * outer(delta, delta)
        self._running_cov = (
            self._running_cov
            + chunk_cov
            + (n_old * n_new / n_total) * np.outer(delta, delta)
        )

        self._n_samples = n_total

    def _update_ewma(
        self,
        returns_chunk: NDArray[np.float64],
        chunk_mean: NDArray[np.float64],
        n_new: int,
    ) -> None:
        """EWMA（指数加重移動平均）で更新

        Args:
            returns_chunk: リターンチャンク
            chunk_mean: チャンク平均
            n_new: 新サンプル数
        """
        halflife = self.config.ewma_halflife
        alpha = 1 - np.exp(-np.log(2) / halflife)

        for i in range(n_new):
            x = returns_chunk[i]

            # 重み更新
            self._ewma_sum = alpha + (1 - alpha) * self._ewma_sum

            # 平均更新
            old_mean = self._running_mean.copy()
            self._running_mean = (
                alpha * x + (1 - alpha) * self._running_mean
            )

            # 共分散更新
            delta_old = x - old_mean
            delta_new = x - self._running_mean
            self._running_cov = (
                (1 - alpha) * self._running_cov
                + alpha * np.outer(delta_old, delta_new)
            )

            self._n_samples += 1

    def get_covariance(self, shrinkage: float | None = None) -> NDArray[np.float64]:
        """共分散行列を取得

        Args:
            shrinkage: shrinkage強度（None=設定に従う）

        Returns:
            共分散行列 (n_assets, n_assets)
        """
        if not self.is_initialized or self._n_samples < 2:
            raise ValueError(
                f"Insufficient samples: {self._n_samples}. Need at least 2."
            )

        # Bessel補正
        cov = self._running_cov / (self._n_samples - 1)

        # Shrinkage適用
        if shrinkage is not None:
            cov = self._apply_shrinkage(cov, shrinkage)
        elif self.config.auto_shrinkage:
            optimal_shrinkage = self._compute_optimal_shrinkage(cov)
            cov = self._apply_shrinkage(cov, optimal_shrinkage)

        return cov

    def get_correlation(self) -> NDArray[np.float64]:
        """相関行列を取得

        Returns:
            相関行列 (n_assets, n_assets)
        """
        cov = self.get_covariance(shrinkage=0.0)  # shrinkageなし

        # 標準偏差
        std = np.sqrt(np.diag(cov))
        std = np.where(std > 0, std, 1.0)  # ゼロ除算防止

        # 相関行列
        corr = cov / np.outer(std, std)

        # 数値誤差をクリップ
        corr = np.clip(corr, -1.0, 1.0)

        # 対角を1に
        np.fill_diagonal(corr, 1.0)

        return corr

    def get_result(
        self,
        shrinkage: float | None = None,
    ) -> StreamingCovarianceResult:
        """完全な結果を取得

        Args:
            shrinkage: shrinkage強度

        Returns:
            StreamingCovarianceResult
        """
        cov = self.get_covariance(shrinkage)
        corr = self.get_correlation()

        # 実際に適用されたshrinkage
        if shrinkage is not None:
            intensity = shrinkage
        elif self.config.auto_shrinkage:
            intensity = self._compute_optimal_shrinkage(
                self._running_cov / (self._n_samples - 1)
            )
        else:
            intensity = 0.0

        return StreamingCovarianceResult(
            covariance=cov,
            correlation=corr,
            n_samples=self._n_samples,
            shrinkage_intensity=intensity,
            is_valid=self._n_samples >= self.config.min_samples,
            metadata={
                "n_assets": self._n_assets,
                "shrinkage_target": self.config.shrinkage_target,
                "ewma_halflife": self.config.ewma_halflife,
            },
        )

    def get_covariance_df(
        self,
        shrinkage: float | None = None,
    ) -> pd.DataFrame:
        """共分散行列をDataFrameで取得

        Args:
            shrinkage: shrinkage強度

        Returns:
            共分散行列DataFrame
        """
        cov = self.get_covariance(shrinkage)

        if self._asset_names is not None:
            return pd.DataFrame(cov, index=self._asset_names, columns=self._asset_names)
        else:
            cols = [f"asset_{i}" for i in range(self._n_assets)]
            return pd.DataFrame(cov, index=cols, columns=cols)

    def get_correlation_df(self) -> pd.DataFrame:
        """相関行列をDataFrameで取得

        Returns:
            相関行列DataFrame
        """
        corr = self.get_correlation()

        if self._asset_names is not None:
            return pd.DataFrame(corr, index=self._asset_names, columns=self._asset_names)
        else:
            cols = [f"asset_{i}" for i in range(self._n_assets)]
            return pd.DataFrame(corr, index=cols, columns=cols)

    def shrink(
        self,
        shrinkage: float = 0.1,
        target: str | None = None,
    ) -> NDArray[np.float64]:
        """Shrinkage適用済み共分散を取得

        Args:
            shrinkage: shrinkage強度（0-1）
            target: shrinkage先（None=設定値）

        Returns:
            shrinkage適用済み共分散行列
        """
        cov = self._running_cov / (self._n_samples - 1)
        return self._apply_shrinkage(cov, shrinkage, target)

    def _apply_shrinkage(
        self,
        cov: NDArray[np.float64],
        shrinkage: float,
        target: str | None = None,
    ) -> NDArray[np.float64]:
        """Shrinkageを適用

        Args:
            cov: 共分散行列
            shrinkage: 強度（0-1）
            target: shrinkage先

        Returns:
            shrinkage適用済み共分散
        """
        if shrinkage <= 0:
            return cov

        target = target or self.config.shrinkage_target
        n = cov.shape[0]

        if target == "identity":
            # 単位行列へのshrinkage
            avg_var = np.trace(cov) / n
            target_matrix = avg_var * np.eye(n)
        elif target == "diagonal":
            # 対角行列へのshrinkage
            target_matrix = np.diag(np.diag(cov))
        elif target == "constant_corr":
            # 一定相関行列へのshrinkage
            std = np.sqrt(np.diag(cov))
            std = np.where(std > 0, std, 1.0)
            corr = cov / np.outer(std, std)
            # 平均相関（対角除く）
            mask = ~np.eye(n, dtype=bool)
            avg_corr = corr[mask].mean()
            target_corr = avg_corr * np.ones((n, n))
            np.fill_diagonal(target_corr, 1.0)
            target_matrix = target_corr * np.outer(std, std)
        else:
            raise ValueError(f"Unknown shrinkage target: {target}")

        return (1 - shrinkage) * cov + shrinkage * target_matrix

    def _compute_optimal_shrinkage(
        self,
        cov: NDArray[np.float64],
    ) -> float:
        """Ledoit-Wolf最適shrinkage強度を計算

        Reference:
        Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator
        for large-dimensional covariance matrices.

        Args:
            cov: 共分散行列

        Returns:
            最適shrinkage強度
        """
        n = cov.shape[0]
        n_samples = self._n_samples

        if n_samples < 2:
            return 0.5  # デフォルト

        # ターゲット（単位行列 * 平均分散）
        avg_var = np.trace(cov) / n

        # Frobenius normの2乗
        delta = cov - avg_var * np.eye(n)
        delta_sq = np.sum(delta ** 2)

        # サンプル共分散の分散推定（簡略版）
        # 完全なLedoit-Wolf推定には元データが必要
        # ここでは近似値を使用
        approx_variance = delta_sq / (n_samples - 1)

        # 最適shrinkage
        if delta_sq > 0:
            shrinkage = min(1.0, approx_variance / delta_sq)
        else:
            shrinkage = 0.0

        return shrinkage

    def get_memory_usage(self) -> dict[str, int]:
        """メモリ使用量を取得（バイト）

        Returns:
            メモリ使用量辞書
        """
        mean_size = self._running_mean.nbytes if self._running_mean is not None else 0
        cov_size = self._running_cov.nbytes if self._running_cov is not None else 0

        return {
            "running_mean": mean_size,
            "running_cov": cov_size,
            "total": mean_size + cov_size,
            "total_mb": (mean_size + cov_size) / (1024 * 1024),
        }

    def get_summary(self) -> dict[str, Any]:
        """サマリーを取得"""
        mem = self.get_memory_usage()
        return {
            "n_assets": self._n_assets,
            "n_samples": self._n_samples,
            "is_initialized": self.is_initialized,
            "memory_mb": f"{mem['total_mb']:.2f}",
            "config": {
                "chunk_size": self.config.chunk_size,
                "shrinkage_target": self.config.shrinkage_target,
                "auto_shrinkage": self.config.auto_shrinkage,
                "ewma_halflife": self.config.ewma_halflife,
            },
        }


# =============================================================================
# 便利関数
# =============================================================================

def create_streaming_covariance(
    n_assets: int | None = None,
    chunk_size: int = 100,
    shrinkage_target: str = "identity",
    auto_shrinkage: bool = True,
    ewma_halflife: int | None = None,
    asset_names: list[str] | pd.Index | None = None,
) -> StreamingCovariance:
    """StreamingCovarianceを作成（ファクトリ関数）

    Args:
        n_assets: アセット数
        chunk_size: 推奨チャンクサイズ
        shrinkage_target: shrinkage先
        auto_shrinkage: 自動shrinkage
        ewma_halflife: EWMA半減期
        asset_names: アセット名

    Returns:
        StreamingCovariance
    """
    config = StreamingCovarianceConfig(
        chunk_size=chunk_size,
        shrinkage_target=shrinkage_target,
        auto_shrinkage=auto_shrinkage,
        ewma_halflife=ewma_halflife,
    )
    return StreamingCovariance(
        n_assets=n_assets,
        config=config,
        asset_names=asset_names,
    )


def streaming_covariance_from_returns(
    returns: pd.DataFrame | NDArray[np.float64],
    chunk_size: int = 100,
    shrinkage: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """リターンデータから共分散・相関行列を計算（便利関数）

    チャンク処理でメモリ効率的に計算。

    Args:
        returns: リターンデータ (n_samples, n_assets)
        chunk_size: チャンクサイズ
        shrinkage: shrinkage強度

    Returns:
        (covariance_df, correlation_df)
    """
    if isinstance(returns, pd.DataFrame):
        asset_names = returns.columns
        returns_array = returns.values
    else:
        asset_names = None
        returns_array = returns

    n_samples, n_assets = returns_array.shape

    # StreamingCovariance作成
    estimator = StreamingCovariance(
        n_assets=n_assets,
        asset_names=asset_names,
    )

    # チャンク処理
    for i in range(0, n_samples, chunk_size):
        chunk = returns_array[i:i + chunk_size]
        estimator.update(chunk)

    # 結果取得
    result = estimator.get_result(shrinkage)
    return result.to_dataframe(asset_names)


def compare_memory_usage(
    n_assets: int,
    n_samples: int,
) -> dict[str, Any]:
    """通常計算とストリーミング計算のメモリ使用量を比較

    Args:
        n_assets: アセット数
        n_samples: サンプル数

    Returns:
        比較結果
    """
    # 通常計算: 全データ + 共分散
    normal_data = n_samples * n_assets * 8  # float64
    normal_cov = n_assets * n_assets * 8
    normal_total = normal_data + normal_cov

    # ストリーミング: mean + cov のみ
    streaming_mean = n_assets * 8
    streaming_cov = n_assets * n_assets * 8
    streaming_total = streaming_mean + streaming_cov

    reduction = (1 - streaming_total / normal_total) * 100

    return {
        "normal": {
            "data_mb": normal_data / (1024 ** 2),
            "cov_mb": normal_cov / (1024 ** 2),
            "total_mb": normal_total / (1024 ** 2),
        },
        "streaming": {
            "mean_mb": streaming_mean / (1024 ** 2),
            "cov_mb": streaming_cov / (1024 ** 2),
            "total_mb": streaming_total / (1024 ** 2),
        },
        "reduction_pct": reduction,
        "params": {
            "n_assets": n_assets,
            "n_samples": n_samples,
        },
    }
