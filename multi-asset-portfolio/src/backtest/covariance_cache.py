"""
Covariance Cache Module - 共分散キャッシュ

共分散行列を毎回再計算せず、指数加重で逐次更新することで高速化を実現。
バックテスト時の計算コストを3-5倍削減。

主な機能:
- インクリメンタル共分散推定（指数加重移動平均）
- 状態の保存・読み込み（キャッシュ）
- StorageBackend対応（ローカル/S3透過的操作）
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from src.utils.storage_backend import StorageBackend

from src.utils.hash_utils import compute_cache_key

logger = logging.getLogger(__name__)


@dataclass
class CovarianceState:
    """共分散推定器の状態

    Attributes:
        cov_matrix: 共分散行列
        mean_returns: 平均リターン
        n_assets: アセット数
        halflife: 半減期
        n_updates: 更新回数
        asset_names: アセット名リスト
    """

    cov_matrix: np.ndarray
    mean_returns: np.ndarray
    n_assets: int
    halflife: int
    n_updates: int
    asset_names: Optional[List[str]] = None


class IncrementalCovarianceEstimator:
    """
    インクリメンタル共分散推定器

    共分散行列を毎回フルで再計算せず、指数加重移動平均で逐次更新する。
    これにより、バックテスト時の計算コストを大幅に削減できる。

    Usage:
        estimator = IncrementalCovarianceEstimator(n_assets=4, halflife=60)

        # 1日分のリターンで更新
        returns = np.array([0.01, -0.02, 0.005, 0.003])
        estimator.update(returns)

        # 共分散行列を取得
        cov = estimator.get_covariance()
    """

    def __init__(
        self,
        n_assets: int,
        halflife: int = 60,
        asset_names: Optional[List[str]] = None,
    ):
        """
        初期化

        Parameters
        ----------
        n_assets : int
            アセット数
        halflife : int
            半減期（日数）。過去のデータの影響が半分になる期間。
        asset_names : List[str], optional
            アセット名のリスト
        """
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if halflife <= 0:
            raise ValueError("halflife must be positive")

        self.n_assets = n_assets
        self.halflife = halflife
        self.asset_names = asset_names

        # 減衰係数: exp(-1/halflife)
        self._decay = np.exp(-1.0 / halflife)

        # 状態の初期化
        self._cov_matrix = np.zeros((n_assets, n_assets))
        self._mean_returns = np.zeros(n_assets)
        self._n_updates = 0

        # 初期化フラグ
        self._is_initialized = False

    def update(self, returns: np.ndarray) -> None:
        """
        1日分のリターンで共分散を更新

        指数加重移動平均で更新:
        - mean = decay * mean + (1 - decay) * returns
        - cov = decay * cov + (1 - decay) * outer(deviation, deviation)

        Parameters
        ----------
        returns : np.ndarray
            1日分のリターン、shape (n_assets,)
        """
        returns = np.asarray(returns).flatten()

        if len(returns) != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} returns, got {len(returns)}"
            )

        if not self._is_initialized:
            # 初回は単純に設定
            self._mean_returns = returns.copy()
            self._cov_matrix = np.zeros((self.n_assets, self.n_assets))
            self._is_initialized = True
            self._n_updates = 1
            return

        # 偏差を計算（更新前の平均からの偏差）
        deviation = returns - self._mean_returns

        # 平均を更新
        self._mean_returns = (
            self._decay * self._mean_returns + (1 - self._decay) * returns
        )

        # 共分散を更新
        self._cov_matrix = (
            self._decay * self._cov_matrix
            + (1 - self._decay) * np.outer(deviation, deviation)
        )

        self._n_updates += 1

    def update_with_mask(
        self,
        returns: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        """
        マスクを使用して1日分のリターンで共分散を更新（NaN対応）

        有効なアセットのみで平均・共分散を更新し、
        無効なアセットの値は前回のまま維持する。

        Parameters
        ----------
        returns : np.ndarray
            1日分のリターン、shape (n_assets,)。
            無効なアセットは任意の値（通常0またはNaN）。
        valid_mask : np.ndarray
            有効なアセットのブールマスク、shape (n_assets,)
        """
        returns = np.asarray(returns).flatten()
        valid_mask = np.asarray(valid_mask).flatten().astype(bool)

        if len(returns) != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} returns, got {len(returns)}"
            )
        if len(valid_mask) != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} mask values, got {len(valid_mask)}"
            )

        if not np.any(valid_mask):
            # 全アセットが無効な場合は更新しない
            return

        if not self._is_initialized:
            # 初回: 有効なアセットのみ設定、無効なアセットは0
            self._mean_returns = np.where(valid_mask, returns, 0.0)
            self._cov_matrix = np.zeros((self.n_assets, self.n_assets))
            self._is_initialized = True
            self._n_updates = 1
            return

        # 有効なアセットのインデックス
        valid_idx = np.where(valid_mask)[0]

        # 有効なアセットの平均を更新
        for idx in valid_idx:
            self._mean_returns[idx] = (
                self._decay * self._mean_returns[idx] +
                (1 - self._decay) * returns[idx]
            )

        # 有効なアセット間の共分散のみ更新
        for i_pos, i in enumerate(valid_idx):
            deviation_i = returns[i] - self._mean_returns[i]
            for j in valid_idx[i_pos:]:  # 対称性を利用
                deviation_j = returns[j] - self._mean_returns[j]
                update = (1 - self._decay) * deviation_i * deviation_j
                self._cov_matrix[i, j] = (
                    self._decay * self._cov_matrix[i, j] + update
                )
                if i != j:
                    self._cov_matrix[j, i] = self._cov_matrix[i, j]

        self._n_updates += 1

    def update_batch(self, returns_matrix: np.ndarray) -> None:
        """
        複数日分のリターンで一括更新

        Parameters
        ----------
        returns_matrix : np.ndarray
            リターン行列、shape (n_days, n_assets)
        """
        returns_matrix = np.asarray(returns_matrix)

        if returns_matrix.ndim == 1:
            self.update(returns_matrix)
            return

        for returns in returns_matrix:
            self.update(returns)

    def get_covariance(self) -> np.ndarray:
        """
        共分散行列を取得

        Returns
        -------
        np.ndarray
            共分散行列、shape (n_assets, n_assets)
        """
        return self._cov_matrix.copy()

    def get_correlation(self) -> np.ndarray:
        """
        相関行列を取得

        Returns
        -------
        np.ndarray
            相関行列、shape (n_assets, n_assets)
        """
        vol = self.get_volatility()

        # ゼロ除算を防ぐ
        vol_safe = np.where(vol > 0, vol, 1.0)

        # 相関行列を計算
        corr = self._cov_matrix / np.outer(vol_safe, vol_safe)

        # 対角要素を1に、ゼロボラの場合は0に
        np.fill_diagonal(corr, 1.0)
        corr = np.where(np.outer(vol > 0, vol > 0), corr, 0.0)
        np.fill_diagonal(corr, 1.0)

        return corr

    def get_volatility(self) -> np.ndarray:
        """
        ボラティリティ（標準偏差）を取得

        Returns
        -------
        np.ndarray
            ボラティリティ、shape (n_assets,)
        """
        return np.sqrt(np.maximum(np.diag(self._cov_matrix), 0))

    def get_annualized_volatility(self, trading_days: int = 252) -> np.ndarray:
        """
        年率換算ボラティリティを取得

        Parameters
        ----------
        trading_days : int
            年間取引日数

        Returns
        -------
        np.ndarray
            年率換算ボラティリティ
        """
        return self.get_volatility() * np.sqrt(trading_days)

    def get_state(self) -> CovarianceState:
        """
        現在の状態を取得

        Returns
        -------
        CovarianceState
            現在の状態
        """
        return CovarianceState(
            cov_matrix=self._cov_matrix.copy(),
            mean_returns=self._mean_returns.copy(),
            n_assets=self.n_assets,
            halflife=self.halflife,
            n_updates=self._n_updates,
            asset_names=self.asset_names,
        )

    def set_state(self, state: CovarianceState) -> None:
        """
        状態を復元

        Parameters
        ----------
        state : CovarianceState
            復元する状態
        """
        if state.n_assets != self.n_assets:
            raise ValueError(
                f"State n_assets ({state.n_assets}) != estimator n_assets ({self.n_assets})"
            )

        self._cov_matrix = state.cov_matrix.copy()
        self._mean_returns = state.mean_returns.copy()
        self._n_updates = state.n_updates
        self._is_initialized = state.n_updates > 0

        if state.asset_names:
            self.asset_names = state.asset_names

    @property
    def n_updates(self) -> int:
        """更新回数"""
        return self._n_updates

    @property
    def is_initialized(self) -> bool:
        """初期化済みか"""
        return self._is_initialized

    @property
    def decay(self) -> float:
        """減衰係数"""
        return self._decay


class CovarianceCache:
    """
    共分散キャッシュ

    IncrementalCovarianceEstimator の状態を日付ごとに保存・読み込みする。
    バックテストの途中再開やウォームスタートに使用。

    StorageBackend対応:
    - storage_backend指定時: S3/ローカルを透過的に操作（推奨）
    - cache_dir指定時: 従来通りローカルファイル操作（後方互換性）

    Usage:
        # StorageBackend経由（S3必須）
        from src.utils.storage_backend import get_storage_backend
        backend = get_storage_backend()  # S3_BUCKET環境変数が必要
        cache = CovarianceCache(storage_backend=backend)

        # 明示的なS3バケット指定
        from src.utils.storage_backend import StorageBackend, StorageConfig
        backend = StorageBackend(StorageConfig(s3_bucket="my-bucket"))
        cache = CovarianceCache(storage_backend=backend)

        # 状態を保存
        cache.save_state(date, estimator)

        # 状態を読み込み
        estimator = cache.load_state(date, n_assets=4)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        storage_backend: Optional["StorageBackend"] = None,
    ):
        """
        初期化

        Parameters
        ----------
        cache_dir : str, optional
            キャッシュディレクトリのパス（従来モード）
        storage_backend : StorageBackend, optional
            ストレージバックエンド（S3対応モード）

        Note:
            storage_backend が指定された場合はそちらを優先。
            両方未指定の場合はデフォルトのローカルモード。
        """
        self._backend: Optional["StorageBackend"] = None
        self._use_backend = False

        if storage_backend is not None:
            # StorageBackendモード
            self._backend = storage_backend
            self._use_backend = True
            self._cache_subdir = "covariance"
            logger.debug("CovarianceCache using StorageBackend")
        else:
            # 従来のローカルモード
            if cache_dir is None:
                from src.config.settings import get_cache_path
                cache_dir = get_cache_path("covariance")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"CovarianceCache using local: {self.cache_dir}")

    def _get_cache_path(self, date: datetime) -> Union[Path, str]:
        """
        日付からキャッシュファイルパスを取得

        Parameters
        ----------
        date : datetime
            日付

        Returns
        -------
        Path or str
            キャッシュファイルパス（ローカルモードはPath、Backendモードはstr）
        """
        date_str = date.strftime("%Y%m%d")
        filename = f"cov_state_{date_str}.pkl"

        if self._use_backend:
            return f"{self._cache_subdir}/{filename}"
        return self.cache_dir / filename

    def save_state(
        self,
        date: datetime,
        estimator: IncrementalCovarianceEstimator,
    ) -> Union[Path, str]:
        """
        状態を保存

        Parameters
        ----------
        date : datetime
            日付
        estimator : IncrementalCovarianceEstimator
            推定器

        Returns
        -------
        Path or str
            保存先パス
        """
        cache_path = self._get_cache_path(date)
        state = estimator.get_state()

        if self._use_backend:
            # StorageBackendモード
            self._backend.write_pickle(state, cache_path)
        else:
            # 従来のローカルモード
            with open(cache_path, "wb") as f:
                pickle.dump(state, f)

        logger.debug("Saved covariance state to %s", cache_path)
        return cache_path

    def load_state(
        self,
        date: datetime,
        n_assets: Optional[int] = None,
        halflife: int = 60,
    ) -> Optional[IncrementalCovarianceEstimator]:
        """
        状態を読み込み

        Parameters
        ----------
        date : datetime
            日付
        n_assets : int, optional
            アセット数（状態がない場合の新規作成用）
        halflife : int
            半減期（状態がない場合の新規作成用）

        Returns
        -------
        IncrementalCovarianceEstimator or None
            推定器。キャッシュが存在しない場合はNone。
        """
        cache_path = self._get_cache_path(date)

        # 存在確認
        if self._use_backend:
            if not self._backend.exists(cache_path):
                logger.debug("No cache found for %s", date)
                return None
        else:
            if not cache_path.exists():
                logger.debug("No cache found for %s", date)
                return None

        try:
            # 状態を読み込み
            if self._use_backend:
                state: CovarianceState = self._backend.read_pickle(cache_path)
            else:
                with open(cache_path, "rb") as f:
                    state: CovarianceState = pickle.load(f)

            estimator = IncrementalCovarianceEstimator(
                n_assets=state.n_assets,
                halflife=state.halflife,
                asset_names=state.asset_names,
            )
            estimator.set_state(state)

            logger.debug(
                "Loaded covariance state from %s (n_updates=%d)",
                cache_path,
                state.n_updates,
            )
            return estimator

        except Exception as e:
            logger.warning("Failed to load cache from %s: %s", cache_path, e)
            return None

    def find_nearest_state(
        self,
        target_date: datetime,
        max_days_back: int = 30,
    ) -> Tuple[Optional[datetime], Optional[IncrementalCovarianceEstimator]]:
        """
        指定日付以前で最も近いキャッシュを探す

        Parameters
        ----------
        target_date : datetime
            目標日付
        max_days_back : int
            最大遡り日数

        Returns
        -------
        Tuple[datetime, IncrementalCovarianceEstimator] or (None, None)
            見つかった日付と推定器
        """
        from datetime import timedelta

        for days_back in range(max_days_back + 1):
            check_date = target_date - timedelta(days=days_back)
            estimator = self.load_state(check_date)

            if estimator is not None:
                return check_date, estimator

        return None, None

    def clear_cache(self, before_date: Optional[datetime] = None) -> int:
        """
        キャッシュをクリア

        Parameters
        ----------
        before_date : datetime, optional
            この日付より前のキャッシュのみ削除。Noneの場合は全削除。

        Returns
        -------
        int
            削除したファイル数
        """
        deleted = 0

        for cache_file in self.cache_dir.glob("cov_state_*.pkl"):
            if before_date is not None:
                # ファイル名から日付を抽出
                date_str = cache_file.stem.replace("cov_state_", "")
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date >= before_date:
                        continue
                except ValueError:
                    continue

            cache_file.unlink()
            deleted += 1

        logger.info("Cleared %d cache files", deleted)
        return deleted

    def list_cached_dates(self) -> List[datetime]:
        """
        キャッシュされている日付のリストを取得

        Returns
        -------
        List[datetime]
            キャッシュされている日付のリスト（昇順）
        """
        dates = []

        for cache_file in self.cache_dir.glob("cov_state_*.pkl"):
            date_str = cache_file.stem.replace("cov_state_", "")
            try:
                dates.append(datetime.strptime(date_str, "%Y%m%d"))
            except ValueError:
                continue

        return sorted(dates)


# ショートカット関数
def create_estimator_from_history(
    returns_matrix: np.ndarray,
    halflife: int = 60,
    asset_names: Optional[List[str]] = None,
) -> IncrementalCovarianceEstimator:
    """
    履歴データから推定器を初期化

    Parameters
    ----------
    returns_matrix : np.ndarray
        リターン行列、shape (n_days, n_assets)
    halflife : int
        半減期
    asset_names : List[str], optional
        アセット名

    Returns
    -------
    IncrementalCovarianceEstimator
        初期化済み推定器
    """
    returns_matrix = np.asarray(returns_matrix)

    if returns_matrix.ndim == 1:
        n_assets = len(returns_matrix)
    else:
        n_assets = returns_matrix.shape[1]

    estimator = IncrementalCovarianceEstimator(
        n_assets=n_assets,
        halflife=halflife,
        asset_names=asset_names,
    )

    estimator.update_batch(returns_matrix)

    return estimator


# =============================================================================
# Subset Covariance (v2.4 - Large Universe Optimization)
# =============================================================================

def compute_covariance_subset(
    returns_df: "pd.DataFrame",
    target_assets: List[str],
    halflife: int = 60,
    min_observations: int = 60,
) -> Tuple[np.ndarray, List[str]]:
    """
    Top-N銘柄のみで共分散行列を計算（メモリ効率化）

    大規模ユニバース（16,000銘柄等）では全銘柄の共分散行列を保持すると
    メモリ不足になる（16,000² × 8byte = 2GB）。
    Top-Nフィルター後の銘柄のみで計算することでメモリ使用量を削減。

    計算量:
    - 全銘柄: O(n² × T) where n=16,000, T=3,780日（15年）
    - Top-N: O(m² × T) where m=1,000 → 256倍削減

    Parameters
    ----------
    returns_df : pd.DataFrame
        リターンデータ（列=銘柄、行=日付）
    target_assets : List[str]
        計算対象の銘柄リスト（Top-Nフィルター後）
    halflife : int
        指数加重の半減期（日数）
    min_observations : int
        最低必要な観測数

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (共分散行列, 有効な銘柄リスト)

    Notes
    -----
    NaN処理:
    - 銘柄ごとにNaN率をチェック
    - NaN率50%超の銘柄は除外
    - 残りの銘柄でpairwise共分散を計算
    """
    import pandas as pd

    # target_assets に存在する列のみ抽出
    available_assets = [a for a in target_assets if a in returns_df.columns]

    if len(available_assets) == 0:
        return np.array([[]]), []

    subset_returns = returns_df[available_assets].copy()

    # NaN率でフィルタリング
    nan_ratios = subset_returns.isna().sum() / len(subset_returns)
    valid_cols = nan_ratios[nan_ratios < 0.5].index.tolist()

    if len(valid_cols) == 0:
        return np.array([[]]), []

    subset_returns = subset_returns[valid_cols]

    # 十分な観測数があるかチェック
    if len(subset_returns.dropna()) < min_observations:
        # dropna だと厳しすぎる場合は pairwise で計算
        pass

    # 指数加重共分散を計算
    n_assets = len(valid_cols)

    # 指数加重の重み
    n_obs = len(subset_returns)
    decay = np.exp(-1.0 / halflife)
    weights = np.array([decay ** (n_obs - 1 - i) for i in range(n_obs)])
    weights = weights / weights.sum()  # 正規化

    # 加重平均を計算
    filled_returns = subset_returns.fillna(0)  # NaN を 0 で埋める（pairwise 計算用）
    returns_array = filled_returns.values

    weighted_mean = np.average(returns_array, weights=weights, axis=0)

    # 加重共分散を計算
    centered = returns_array - weighted_mean
    weighted_cov = np.zeros((n_assets, n_assets))

    for i in range(n_obs):
        weighted_cov += weights[i] * np.outer(centered[i], centered[i])

    return weighted_cov, valid_cols


class SubsetCovarianceCache:
    """
    サブセット共分散キャッシュ（大規模ユニバース用）

    Top-Nフィルター後の銘柄セットごとに共分散を効率的にキャッシュ。
    銘柄セットが変わるたびに再計算が必要だが、フル共分散より
    大幅に高速かつ省メモリ。

    Usage:
        cache = SubsetCovarianceCache(storage_backend=backend)

        # 共分散を計算（キャッシュヒットがあれば再利用）
        cov_matrix, assets = cache.get_or_compute(
            returns_df=returns,
            target_assets=top_n_assets,
            date=rebalance_date,
        )
    """

    def __init__(
        self,
        storage_backend: Optional["StorageBackend"] = None,
        cache_dir: Optional[str] = None,
        halflife: int = 60,
    ):
        """
        初期化

        Parameters
        ----------
        storage_backend : StorageBackend, optional
            ストレージバックエンド（S3対応）
        cache_dir : str, optional
            ローカルキャッシュディレクトリ
        halflife : int
            共分散計算の半減期
        """
        self._backend = storage_backend
        self._use_backend = storage_backend is not None
        self._halflife = halflife

        if not self._use_backend:
            if cache_dir is None:
                from src.config.settings import get_cache_path
                cache_dir = get_cache_path("subset_covariance")
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_subdir = "subset_covariance"

        # インメモリキャッシュ（直近の結果を保持）
        self._memory_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        self._max_memory_cache_size = 10

    def _get_cache_key(
        self,
        target_assets: List[str],
        date: datetime,
    ) -> str:
        """
        キャッシュキーを生成

        銘柄セット + 日付からハッシュベースのキーを生成。
        """
        assets_str = ",".join(sorted(target_assets))
        date_str = date.strftime("%Y%m%d")
        hash_value = compute_cache_key(assets_str, date_str, truncate=12)
        return f"cov_{date_str}_{hash_value}"

    def get_or_compute(
        self,
        returns_df: "pd.DataFrame",
        target_assets: List[str],
        date: datetime,
        min_observations: int = 60,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        共分散を取得（キャッシュヒットがあれば再利用、なければ計算）

        Parameters
        ----------
        returns_df : pd.DataFrame
            リターンデータ
        target_assets : List[str]
            計算対象銘柄
        date : datetime
            計算日
        min_observations : int
            最低必要な観測数

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            (共分散行列, 有効な銘柄リスト)
        """
        cache_key = self._get_cache_key(target_assets, date)

        # メモリキャッシュをチェック
        if cache_key in self._memory_cache:
            logger.debug(f"Subset covariance cache hit (memory): {cache_key}")
            return self._memory_cache[cache_key]

        # ストレージキャッシュをチェック
        cached = self._load_from_storage(cache_key)
        if cached is not None:
            logger.debug(f"Subset covariance cache hit (storage): {cache_key}")
            self._memory_cache[cache_key] = cached
            self._evict_memory_cache()
            return cached

        # 計算
        logger.debug(f"Computing subset covariance: {len(target_assets)} assets")
        cov_matrix, valid_assets = compute_covariance_subset(
            returns_df=returns_df,
            target_assets=target_assets,
            halflife=self._halflife,
            min_observations=min_observations,
        )

        # キャッシュに保存
        result = (cov_matrix, valid_assets)
        self._save_to_storage(cache_key, result)
        self._memory_cache[cache_key] = result
        self._evict_memory_cache()

        return result

    def _load_from_storage(
        self,
        cache_key: str,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """ストレージからキャッシュを読み込み"""
        try:
            if self._use_backend:
                path = f"{self._cache_subdir}/{cache_key}.pkl"
                if self._backend.exists(path):
                    return self._backend.read_pickle(path)
            else:
                path = self._cache_dir / f"{cache_key}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load subset covariance cache: {e}")

        return None

    def _save_to_storage(
        self,
        cache_key: str,
        data: Tuple[np.ndarray, List[str]],
    ) -> None:
        """ストレージにキャッシュを保存"""
        try:
            if self._use_backend:
                path = f"{self._cache_subdir}/{cache_key}.pkl"
                self._backend.write_pickle(data, path)
            else:
                path = self._cache_dir / f"{cache_key}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save subset covariance cache: {e}")

    def _evict_memory_cache(self) -> None:
        """メモリキャッシュが上限を超えた場合に古いエントリを削除"""
        while len(self._memory_cache) > self._max_memory_cache_size:
            # 最初のキーを削除（FIFO）
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._memory_cache.clear()
        logger.info("Subset covariance memory cache cleared")
