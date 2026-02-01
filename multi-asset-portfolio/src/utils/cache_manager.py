"""
Unified Cache Manager - 統一キャッシュ管理システム

全キャッシュ実装を統一インターフェースで管理。
- 共通CacheInterface定義
- メモリ使用量の一元管理
- 統一されたキャッシュポリシー
- 各用途向けラッパー

既存実装との互換性を維持しつつ、統一APIを提供。

使用例:
    from src.utils.cache_manager import unified_cache_manager, CacheType

    # シグナルキャッシュ取得
    signal_cache = unified_cache_manager.get_cache(CacheType.SIGNAL)

    # データキャッシュ取得
    data_cache = unified_cache_manager.get_cache(CacheType.DATA)

    # 全キャッシュ統計
    stats = unified_cache_manager.get_all_stats()

    # 全キャッシュクリア
    unified_cache_manager.clear_all()
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend, StorageConfig

from src.config.resource_config import get_current_resource_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheType(Enum):
    """キャッシュタイプ"""
    SIGNAL = "signal"           # シグナル計算キャッシュ
    DATAFRAME = "dataframe"     # DataFrame汎用キャッシュ
    DATA = "data"               # OHLCVデータキャッシュ
    LRU = "lru"                 # 汎用LRUキャッシュ
    INCREMENTAL = "incremental" # インクリメンタル計算キャッシュ
    COVARIANCE = "covariance"   # 共分散行列キャッシュ
    SUBSET_COVARIANCE = "subset_covariance"  # サブセット共分散キャッシュ


@dataclass
class UnifiedCacheStats:
    """統一キャッシュ統計情報"""
    name: str
    cache_type: CacheType
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    memory_bytes: int = 0
    disk_size_bytes: int = 0
    last_cleanup: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
        """ヒット率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        """総サイズ（MB）"""
        return (self.memory_bytes + self.disk_size_bytes) / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "name": self.name,
            "type": self.cache_type.value,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "size": self.size,
            "max_size": self.max_size,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "disk_mb": self.disk_size_bytes / (1024 * 1024),
            "total_mb": self.total_size_mb,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
        }


@dataclass
class CachePolicy:
    """キャッシュポリシー設定"""
    max_memory_mb: int = 256          # 最大メモリ使用量（MB）
    max_disk_mb: int = 500            # 最大ディスク使用量（MB）
    max_entries: int = 1000           # 最大エントリ数
    max_age_days: int = 30            # 最大保持日数
    cleanup_interval_seconds: int = 3600  # クリーンアップ間隔（秒）
    enable_disk_cache: bool = True    # ディスクキャッシュ有効化

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_disk_mb": self.max_disk_mb,
            "max_entries": self.max_entries,
            "max_age_days": self.max_age_days,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "enable_disk_cache": self.enable_disk_cache,
        }


def get_default_cache_policy() -> CachePolicy:
    """
    ResourceConfigに基づくデフォルトCachePolicyを取得

    Returns:
        CachePolicy: システムリソースに基づいた最適なキャッシュポリシー
    """
    rc = get_current_resource_config()
    return CachePolicy(
        max_memory_mb=rc.cache_max_memory_mb,
        max_disk_mb=rc.cache_max_disk_mb if rc.cache_max_disk_mb else 500,
        max_entries=rc.cache_max_entries,
        max_age_days=30,
        cleanup_interval_seconds=3600,
        enable_disk_cache=True,
    )


class CacheInterface(ABC, Generic[T]):
    """
    統一キャッシュインターフェース

    全キャッシュ実装はこのインターフェースを実装するか、
    アダプターを通じて準拠する。
    """

    @property
    @abstractmethod
    def cache_type(self) -> CacheType:
        """キャッシュタイプ"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """キャッシュ名"""
        ...

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """キャッシュから取得"""
        ...

    @abstractmethod
    def put(self, key: str, value: T) -> None:
        """キャッシュに格納"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """キャッシュをクリア"""
        ...

    @abstractmethod
    def get_stats(self) -> UnifiedCacheStats:
        """統計情報を取得"""
        ...

    def cleanup(self) -> int:
        """期限切れエントリを削除（オプション）"""
        return 0

    def contains(self, key: str) -> bool:
        """キーが存在するか確認"""
        return self.get(key) is not None


class LRUCacheAdapter(CacheInterface[Any]):
    """
    LRUCacheのアダプター

    src.backtest.cache.LRUCache を統一インターフェースに適合させる。
    """

    def __init__(
        self,
        name: str = "lru_cache",
        max_size: int = 1000,
        max_memory_mb: int = 256,
    ):
        self._name = name
        self._max_size = max_size
        self._max_memory_mb = max_memory_mb

        # 遅延インポートで循環参照を回避
        from src.backtest.cache import LRUCache
        self._cache: LRUCache = LRUCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb,
        )

    @property
    def cache_type(self) -> CacheType:
        return CacheType.LRU

    @property
    def name(self) -> str:
        return self._name

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def put(self, key: str, value: Any, size_bytes: int = 0) -> None:
        self._cache.put(key, value, size_bytes)

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> UnifiedCacheStats:
        stats = self._cache.stats
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.LRU,
            hits=stats.get("hits", 0),
            misses=stats.get("misses", 0),
            size=stats.get("size", 0),
            max_size=self._max_size,
            memory_bytes=int(stats.get("memory_mb", 0) * 1024 * 1024),
        )


class SignalCacheAdapter(CacheInterface[Any]):
    """
    SignalCacheのアダプター

    src.backtest.cache.SignalCache を統一インターフェースに適合させる。
    """

    def __init__(
        self,
        name: str = "signal_cache",
        cache_dir: Optional[Union[str, Path]] = None,
        policy: Optional[CachePolicy] = None,
    ):
        self._name = name
        policy = policy or get_default_cache_policy()

        # 遅延インポート
        from src.backtest.cache import SignalCache
        self._cache = SignalCache(
            cache_dir=cache_dir,
            max_memory_entries=policy.max_entries,
            max_memory_mb=policy.max_memory_mb,
            enable_disk_cache=policy.enable_disk_cache,
            max_disk_cache_mb=policy.max_disk_mb,
            cleanup_interval_seconds=policy.cleanup_interval_seconds,
            max_age_days=policy.max_age_days,
        )

    @property
    def cache_type(self) -> CacheType:
        return CacheType.SIGNAL

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self):
        """元のSignalCacheインスタンスを取得（高度な操作用）"""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        # SignalCacheは複合キーを使用するため、単純なget不可
        # 直接underlying.get()を使用
        return None

    def put(self, key: str, value: Any) -> None:
        # SignalCacheは複合キーを使用するため、単純なput不可
        pass

    def clear(self) -> None:
        self._cache.clear_all()

    def get_stats(self) -> UnifiedCacheStats:
        stats = self._cache.stats
        memory_stats = stats.get("memory", {})
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.SIGNAL,
            hits=memory_stats.get("hits", 0),
            misses=memory_stats.get("misses", 0),
            size=memory_stats.get("size", 0),
            memory_bytes=int(memory_stats.get("memory_mb", 0) * 1024 * 1024),
            disk_size_bytes=int(stats.get("disk_size_mb", 0) * 1024 * 1024),
        )

    def cleanup(self) -> int:
        result = self._cache.force_cleanup()
        return result.get("total_removed", 0)


class DataFrameCacheAdapter(CacheInterface[Any]):
    """
    DataFrameCacheのアダプター

    src.backtest.cache.DataFrameCache を統一インターフェースに適合させる。
    """

    def __init__(
        self,
        name: str = "dataframe_cache",
        cache_dir: Optional[Union[str, Path]] = None,
        policy: Optional[CachePolicy] = None,
    ):
        self._name = name
        policy = policy or get_default_cache_policy()

        from src.backtest.cache import DataFrameCache
        self._cache = DataFrameCache(
            cache_dir=cache_dir,
            max_memory_entries=min(policy.max_entries, 100),  # DataFrameは大きいので制限
            max_memory_mb=policy.max_memory_mb,
            max_disk_cache_mb=policy.max_disk_mb,
            cleanup_interval_seconds=policy.cleanup_interval_seconds,
            max_age_days=policy.max_age_days,
        )

    @property
    def cache_type(self) -> CacheType:
        return CacheType.DATAFRAME

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self):
        """元のDataFrameCacheインスタンスを取得"""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def put(self, key: str, value: Any, persist: bool = True) -> None:
        self._cache.put(key, value, persist=persist)

    def clear(self) -> None:
        self._cache._memory_cache.clear()

    def get_stats(self) -> UnifiedCacheStats:
        memory_stats = self._cache._memory_cache.stats
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.DATAFRAME,
            hits=memory_stats.get("hits", 0),
            misses=memory_stats.get("misses", 0),
            size=memory_stats.get("size", 0),
            memory_bytes=int(memory_stats.get("memory_mb", 0) * 1024 * 1024),
        )

    def cleanup(self) -> int:
        result = self._cache.force_cleanup()
        return result.get("total_removed", 0)


class DataCacheAdapter(CacheInterface[Any]):
    """
    DataCacheのアダプター

    src.data.cache.DataCache を統一インターフェースに適合させる。
    StorageBackend対応（task_045_4）。
    """

    def __init__(
        self,
        name: str = "data_cache",
        cache_dir: Optional[Union[str, Path]] = None,
        policy: Optional[CachePolicy] = None,
        backend: str = "parquet",
        storage_backend: Optional["StorageBackend"] = None,
    ):
        self._name = name
        policy = policy or get_default_cache_policy()
        self._storage_backend = storage_backend

        from src.data.cache import CacheBackend, DataCache
        backend_enum = CacheBackend.PARQUET if backend == "parquet" else CacheBackend.DUCKDB
        if cache_dir is None:
            from src.config.settings import get_cache_path
            cache_dir = get_cache_path("data")
        self._cache = DataCache(
            cache_dir=cache_dir,
            backend=backend_enum,
            max_age_days=policy.max_age_days,
            storage_backend=storage_backend,
        )

    @property
    def cache_type(self) -> CacheType:
        return CacheType.DATA

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self):
        """元のDataCacheインスタンスを取得"""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        # DataCacheは複合キーを使用
        return None

    def put(self, key: str, value: Any) -> None:
        # DataCacheは複合キーを使用
        pass

    def clear(self) -> None:
        self._cache.invalidate()

    def get_stats(self) -> UnifiedCacheStats:
        stats = self._cache.get_stats()
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.DATA,
            size=stats.get("entry_count", 0),
            disk_size_bytes=int(stats.get("total_size_mb", 0) * 1024 * 1024),
        )

    def cleanup(self) -> int:
        return self._cache.cleanup_expired()


class CovarianceCacheAdapter(CacheInterface[Any]):
    """
    CovarianceCacheのアダプター

    src.backtest.covariance_cache.CovarianceCache を統一インターフェースに適合させる。

    Key format: "date:YYYYMMDD"
    get() returns None on cache miss.

    Note:
        This adapter wraps the date-based covariance cache which stores
        IncrementalCovarianceEstimator states.
    """

    def __init__(
        self,
        name: str = "covariance_cache",
        storage_backend: Optional["StorageBackend"] = None,
    ):
        """
        Initialize covariance cache adapter.

        Args:
            name: Cache name for identification
            storage_backend: StorageBackend for S3/local operations (required)
        """
        self._name = name
        self._storage_backend = storage_backend
        self._hits = 0
        self._misses = 0

        # Lazy import to avoid circular dependencies
        from src.backtest.covariance_cache import CovarianceCache
        self._cache = CovarianceCache(storage_backend=storage_backend)

    @property
    def cache_type(self) -> CacheType:
        return CacheType.COVARIANCE

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self) -> Any:
        """Get the underlying CovarianceCache instance."""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        """
        Get estimator state by date key.

        Args:
            key: Date key in format "date:YYYYMMDD"

        Returns:
            IncrementalCovarianceEstimator or None if not found

        Raises:
            ValueError: If key format is invalid
        """
        if not key.startswith("date:"):
            raise ValueError(f"Invalid key format: {key}. Expected 'date:YYYYMMDD'")

        date_str = key.split(":")[1]
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format in key: {key}") from e

        # CovarianceCache.load_state needs n_assets, which we don't have here
        # Return None to indicate lookup should use underlying cache directly
        result = self._cache.load_state(date, n_assets=0)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, key: str, value: Any) -> None:
        """
        Save estimator state by date key.

        Args:
            key: Date key in format "date:YYYYMMDD"
            value: IncrementalCovarianceEstimator instance

        Raises:
            ValueError: If key format is invalid or value is wrong type
        """
        if not key.startswith("date:"):
            raise ValueError(f"Invalid key format: {key}. Expected 'date:YYYYMMDD'")

        date_str = key.split(":")[1]
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format in key: {key}") from e

        from src.backtest.covariance_cache import IncrementalCovarianceEstimator
        if not isinstance(value, IncrementalCovarianceEstimator):
            raise ValueError(
                f"Value must be IncrementalCovarianceEstimator, got {type(value)}"
            )

        self._cache.save_state(date, value)

    def clear(self) -> None:
        """Clear statistics (cache files are not deleted)."""
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> UnifiedCacheStats:
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.COVARIANCE,
            hits=self._hits,
            misses=self._misses,
        )


class SubsetCovarianceCacheAdapter(CacheInterface[Any]):
    """
    SubsetCovarianceCacheのアダプター

    src.backtest.covariance_cache.SubsetCovarianceCache を統一インターフェースに適合させる。

    Note:
        get() is not directly supported - use underlying.get_or_compute() instead.
        This adapter primarily provides cache statistics and cleanup interface.
    """

    def __init__(
        self,
        name: str = "subset_covariance_cache",
        storage_backend: Optional["StorageBackend"] = None,
        halflife: int = 60,
    ):
        """
        Initialize subset covariance cache adapter.

        Args:
            name: Cache name for identification
            storage_backend: StorageBackend for S3/local operations
            halflife: Halflife for covariance calculation
        """
        self._name = name
        self._storage_backend = storage_backend

        # Lazy import
        from src.backtest.covariance_cache import SubsetCovarianceCache
        self._cache = SubsetCovarianceCache(
            storage_backend=storage_backend,
            halflife=halflife,
        )

    @property
    def cache_type(self) -> CacheType:
        return CacheType.SUBSET_COVARIANCE

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self) -> Any:
        """Get the underlying SubsetCovarianceCache instance."""
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        """
        Not directly supported - use underlying.get_or_compute().

        Returns:
            None (always)
        """
        return None

    def put(self, key: str, value: Any) -> None:
        """
        Not directly supported - cache is populated via get_or_compute().
        """
        pass

    def clear(self) -> None:
        """Clear the subset cache."""
        self._cache.clear_cache()

    def get_stats(self) -> UnifiedCacheStats:
        # SubsetCovarianceCache doesn't expose stats, use memory cache size
        memory_cache_size = len(self._cache._memory_cache)
        return UnifiedCacheStats(
            name=self._name,
            cache_type=CacheType.SUBSET_COVARIANCE,
            size=memory_cache_size,
            max_size=self._cache._max_memory_cache_size,
        )


class UnifiedCacheManager:
    """
    統一キャッシュマネージャー

    全キャッシュを一元管理し、統一されたAPIを提供。

    機能:
    - 複数キャッシュタイプの管理
    - グローバルメモリ制限
    - 統一されたキャッシュポリシー
    - 一括クリーンアップ
    - 統合統計情報

    使用例:
        manager = UnifiedCacheManager()

        # デフォルトキャッシュ取得
        signal_cache = manager.get_cache(CacheType.SIGNAL)

        # カスタムキャッシュ登録
        manager.register_cache("my_cache", my_cache_adapter)

        # 全統計取得
        stats = manager.get_all_stats()

        # 全クリア
        manager.clear_all()
    """

    def __init__(
        self,
        global_memory_limit_mb: int = 1024,
        default_policy: Optional[CachePolicy] = None,
        cache_base_dir: Optional[Union[str, Path]] = None,
        storage_config: Optional["StorageConfig"] = None,
    ):
        """
        初期化

        Args:
            global_memory_limit_mb: グローバルメモリ制限（MB）
            default_policy: デフォルトキャッシュポリシー
            cache_base_dir: キャッシュベースディレクトリ
            storage_config: StorageConfig for S3/local backend (task_045_4).
                           If provided, creates a StorageBackend and passes it
                           to supported cache adapters.
        """
        self._caches: Dict[str, CacheInterface] = {}
        self._lock = threading.RLock()
        self._global_memory_limit = global_memory_limit_mb * 1024 * 1024
        self._default_policy = default_policy or get_default_cache_policy()
        self._cache_base_dir = Path(cache_base_dir) if cache_base_dir else Path("./cache")
        self._created_at = datetime.now()
        self._last_global_cleanup = time.time()
        self._cleanup_interval = 3600  # 1時間

        # StorageBackend for S3 integration (task_045_4)
        self._storage_backend: Optional["StorageBackend"] = None
        if storage_config is not None:
            from src.utils.storage_backend import get_storage_backend
            self._storage_backend = get_storage_backend(storage_config)
            logger.info(f"UnifiedCacheManager: StorageBackend initialized (s3_bucket={storage_config.s3_bucket})")

        # 初期化時にデフォルトキャッシュを遅延作成するためのフラグ
        self._default_caches_initialized = False

    def _ensure_default_caches(self) -> None:
        """デフォルトキャッシュを初期化（遅延初期化）"""
        if self._default_caches_initialized:
            return

        with self._lock:
            if self._default_caches_initialized:
                return

            # メモリを分配
            signal_mem = self._default_policy.max_memory_mb // 2
            df_mem = self._default_policy.max_memory_mb // 4
            lru_mem = self._default_policy.max_memory_mb // 4

            # デフォルトキャッシュを作成
            default_caches = [
                (
                    CacheType.SIGNAL.value,
                    SignalCacheAdapter(
                        name=CacheType.SIGNAL.value,
                        cache_dir=self._cache_base_dir / "signals",
                        policy=CachePolicy(
                            max_memory_mb=signal_mem,
                            max_disk_mb=self._default_policy.max_disk_mb,
                            max_age_days=self._default_policy.max_age_days,
                        ),
                    ),
                ),
                (
                    CacheType.DATAFRAME.value,
                    DataFrameCacheAdapter(
                        name=CacheType.DATAFRAME.value,
                        cache_dir=self._cache_base_dir / "dataframes",
                        policy=CachePolicy(
                            max_memory_mb=df_mem,
                            max_disk_mb=self._default_policy.max_disk_mb,
                            max_age_days=self._default_policy.max_age_days,
                        ),
                    ),
                ),
                (
                    CacheType.DATA.value,
                    DataCacheAdapter(
                        name=CacheType.DATA.value,
                        cache_dir=self._cache_base_dir / "data",
                        policy=self._default_policy,
                        storage_backend=self._storage_backend,  # S3 integration (task_045_4)
                    ),
                ),
                (
                    CacheType.LRU.value,
                    LRUCacheAdapter(
                        name=CacheType.LRU.value,
                        max_size=self._default_policy.max_entries,
                        max_memory_mb=lru_mem,
                    ),
                ),
            ]

            for name, cache in default_caches:
                if name not in self._caches:
                    self._caches[name] = cache

            self._default_caches_initialized = True

    def register_cache(
        self,
        name: str,
        cache: CacheInterface,
        replace: bool = False,
    ) -> bool:
        """
        キャッシュを登録

        Args:
            name: キャッシュ名
            cache: キャッシュアダプター
            replace: 既存を置き換えるか

        Returns:
            登録成功時True
        """
        with self._lock:
            if name in self._caches and not replace:
                logger.warning(f"Cache '{name}' already exists. Use replace=True to override.")
                return False

            self._caches[name] = cache
            logger.info(f"Registered cache: {name} (type={cache.cache_type.value})")
            return True

    def unregister_cache(self, name: str) -> bool:
        """
        キャッシュを登録解除

        Args:
            name: キャッシュ名

        Returns:
            解除成功時True
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                logger.info(f"Unregistered cache: {name}")
                return True
            return False

    def get_cache(
        self,
        cache_type: CacheType,
        name: Optional[str] = None,
    ) -> Optional[CacheInterface]:
        """
        キャッシュを取得

        Args:
            cache_type: キャッシュタイプ
            name: キャッシュ名（省略時はタイプのデフォルト）

        Returns:
            キャッシュアダプター
        """
        self._ensure_default_caches()

        cache_name = name or cache_type.value
        with self._lock:
            return self._caches.get(cache_name)

    def get_cache_by_name(self, name: str) -> Optional[CacheInterface]:
        """
        名前でキャッシュを取得

        Args:
            name: キャッシュ名

        Returns:
            キャッシュアダプター
        """
        self._ensure_default_caches()

        with self._lock:
            return self._caches.get(name)

    def clear(self, name: str) -> bool:
        """
        指定キャッシュをクリア

        Args:
            name: キャッシュ名

        Returns:
            クリア成功時True
        """
        with self._lock:
            cache = self._caches.get(name)
            if cache:
                cache.clear()
                logger.info(f"Cleared cache: {name}")
                return True
            return False

    def clear_by_type(self, cache_type: CacheType) -> int:
        """
        タイプ別にキャッシュをクリア

        Args:
            cache_type: キャッシュタイプ

        Returns:
            クリアしたキャッシュ数
        """
        self._ensure_default_caches()

        cleared = 0
        with self._lock:
            for name, cache in self._caches.items():
                if cache.cache_type == cache_type:
                    cache.clear()
                    cleared += 1
        logger.info(f"Cleared {cleared} caches of type {cache_type.value}")
        return cleared

    def clear_all(self) -> int:
        """
        全キャッシュをクリア

        Returns:
            クリアしたキャッシュ数
        """
        self._ensure_default_caches()

        cleared = 0
        with self._lock:
            for name, cache in self._caches.items():
                try:
                    cache.clear()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to clear cache {name}: {e}")
        logger.info(f"Cleared {cleared} caches")
        return cleared

    def cleanup_all(self) -> Dict[str, int]:
        """
        全キャッシュのクリーンアップ（期限切れ削除）

        Returns:
            キャッシュ名 → 削除数の辞書
        """
        self._ensure_default_caches()

        results = {}
        with self._lock:
            for name, cache in self._caches.items():
                try:
                    removed = cache.cleanup()
                    results[name] = removed
                except Exception as e:
                    logger.warning(f"Failed to cleanup cache {name}: {e}")
                    results[name] = 0

        total = sum(results.values())
        logger.info(f"Cleanup completed: {total} entries removed")
        self._last_global_cleanup = time.time()
        return results

    def get_stats(self, name: str) -> Optional[UnifiedCacheStats]:
        """
        指定キャッシュの統計を取得

        Args:
            name: キャッシュ名

        Returns:
            統計情報
        """
        self._ensure_default_caches()

        with self._lock:
            cache = self._caches.get(name)
            if cache:
                return cache.get_stats()
            return None

    def get_all_stats(self) -> Dict[str, UnifiedCacheStats]:
        """
        全キャッシュの統計を取得

        Returns:
            キャッシュ名 → 統計情報の辞書
        """
        self._ensure_default_caches()

        results = {}
        with self._lock:
            for name, cache in self._caches.items():
                try:
                    results[name] = cache.get_stats()
                except Exception as e:
                    logger.warning(f"Failed to get stats for {name}: {e}")
        return results

    def get_total_memory_usage(self) -> int:
        """
        全キャッシュの合計メモリ使用量（バイト）

        Returns:
            メモリ使用量（バイト）
        """
        stats = self.get_all_stats()
        return sum(s.memory_bytes for s in stats.values())

    def get_total_disk_usage(self) -> int:
        """
        全キャッシュの合計ディスク使用量（バイト）

        Returns:
            ディスク使用量（バイト）
        """
        stats = self.get_all_stats()
        return sum(s.disk_size_bytes for s in stats.values())

    def is_memory_limit_exceeded(self) -> bool:
        """
        グローバルメモリ制限を超過しているか

        Returns:
            超過時True
        """
        return self.get_total_memory_usage() > self._global_memory_limit

    def enforce_memory_limit(self) -> int:
        """
        メモリ制限を強制（超過時にクリーンアップ）

        Returns:
            削除したエントリ数
        """
        if not self.is_memory_limit_exceeded():
            return 0

        logger.warning("Global memory limit exceeded, performing cleanup")
        results = self.cleanup_all()
        return sum(results.values())

    def list_caches(self) -> List[str]:
        """
        登録済みキャッシュ名一覧

        Returns:
            キャッシュ名のリスト
        """
        self._ensure_default_caches()

        with self._lock:
            return list(self._caches.keys())

    def summary(self) -> str:
        """
        統計サマリ文字列を生成

        Returns:
            人間可読なサマリ
        """
        stats = self.get_all_stats()

        lines = [
            "=" * 70,
            "  UNIFIED CACHE MANAGER STATISTICS",
            "=" * 70,
            "",
        ]

        total_hits = 0
        total_misses = 0
        total_memory = 0
        total_disk = 0

        for name, stat in stats.items():
            total_hits += stat.hits
            total_misses += stat.misses
            total_memory += stat.memory_bytes
            total_disk += stat.disk_size_bytes

            lines.append(
                f"  [{stat.cache_type.value.upper():12}] {name}:"
            )
            lines.append(
                f"    hits={stat.hits:,}, misses={stat.misses:,}, "
                f"hit_rate={stat.hit_rate:.1%}"
            )
            lines.append(
                f"    size={stat.size:,}, "
                f"memory={stat.memory_bytes / (1024*1024):.1f}MB, "
                f"disk={stat.disk_size_bytes / (1024*1024):.1f}MB"
            )
            lines.append("")

        lines.append("-" * 70)
        total = total_hits + total_misses
        overall_rate = total_hits / total if total > 0 else 0.0
        lines.append(f"  TOTAL:")
        lines.append(
            f"    hits={total_hits:,}, misses={total_misses:,}, "
            f"hit_rate={overall_rate:.1%}"
        )
        lines.append(
            f"    memory={total_memory / (1024*1024):.1f}MB / "
            f"{self._global_memory_limit / (1024*1024):.0f}MB, "
            f"disk={total_disk / (1024*1024):.1f}MB"
        )
        lines.append("=" * 70)

        return "\n".join(lines)


# グローバルシングルトンインスタンス
unified_cache_manager = UnifiedCacheManager()


# ショートカット関数
def get_signal_cache() -> SignalCacheAdapter:
    """シグナルキャッシュを取得"""
    cache = unified_cache_manager.get_cache(CacheType.SIGNAL)
    if cache is None:
        raise RuntimeError("Signal cache not initialized")
    return cache  # type: ignore


def get_data_cache() -> DataCacheAdapter:
    """データキャッシュを取得"""
    cache = unified_cache_manager.get_cache(CacheType.DATA)
    if cache is None:
        raise RuntimeError("Data cache not initialized")
    return cache  # type: ignore


def get_dataframe_cache() -> DataFrameCacheAdapter:
    """DataFrameキャッシュを取得"""
    cache = unified_cache_manager.get_cache(CacheType.DATAFRAME)
    if cache is None:
        raise RuntimeError("DataFrame cache not initialized")
    return cache  # type: ignore


def clear_all_caches() -> int:
    """全キャッシュをクリア"""
    return unified_cache_manager.clear_all()


def get_cache_summary() -> str:
    """キャッシュサマリを取得"""
    return unified_cache_manager.summary()
