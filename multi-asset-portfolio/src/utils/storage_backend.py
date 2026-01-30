"""
Storage Backend - ローカル/S3を透過的に扱うストレージ抽象化レイヤー

fsspecを使用してローカルファイルシステムとS3を統一インターフェースで操作。
ローカルキャッシュ付きS3モードでは、S3をバックエンドとしつつローカルにキャッシュを保持。

Usage:
    from src.utils.storage_backend import get_storage_backend, StorageConfig

    # ローカルモード
    backend = get_storage_backend(StorageConfig(backend="local", base_path=".cache"))

    # S3モード（ローカルキャッシュ付き）
    backend = get_storage_backend(StorageConfig(
        backend="s3",
        s3_bucket="my-bucket",
        s3_prefix=".cache",
        local_cache_path="/tmp/.cache",
        local_cache_ttl_hours=24,
    ))

    # 透過的に操作
    backend.write_parquet(df, "signals/momentum.parquet")
    df = backend.read_parquet("signals/momentum.parquet")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import fsspec
    from fsspec.implementations.local import LocalFileSystem
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False
    fsspec = None
    LocalFileSystem = None

try:
    import s3fs
    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False
    s3fs = None


@dataclass
class StorageConfig:
    """ストレージ設定"""

    backend: str = "local"  # "local" or "s3"
    base_path: str = ".cache"  # ローカルモード時のベースパス

    # S3設定
    s3_bucket: str = ""
    s3_prefix: str = ".cache"
    s3_region: str = "ap-northeast-1"

    # 認証情報（環境変数からも取得可能）
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # ローカルキャッシュ設定（S3モード時）
    local_cache_enabled: bool = True
    local_cache_path: str = "/tmp/.backtest_cache"
    local_cache_ttl_hours: int = 24

    # パフォーマンス設定
    use_parquet_optimization: bool = True  # fsspec.parquet最適化

    def __post_init__(self):
        """環境変数から認証情報を取得"""
        if self.aws_access_key_id is None:
            self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        if self.aws_secret_access_key is None:
            self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")


class StorageBackend:
    """
    ストレージバックエンド抽象化クラス

    ローカルファイルシステムとS3を統一インターフェースで操作。
    S3モードではローカルキャッシュを併用して高速化。
    """

    def __init__(self, config: StorageConfig):
        """
        初期化

        Args:
            config: ストレージ設定
        """
        if not HAS_FSSPEC:
            raise ImportError("fsspec is required. Install with: pip install fsspec")

        self.config = config
        self._fs: Any = None
        self._local_cache_fs: Any = None
        self._cache_metadata: Dict[str, Dict[str, Any]] = {}

        self._init_filesystem()

    def _init_filesystem(self) -> None:
        """ファイルシステムを初期化"""
        if self.config.backend == "local":
            self._fs = fsspec.filesystem("file")
            self._base_path = self.config.base_path
            Path(self._base_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage backend: local ({self._base_path})")

        elif self.config.backend == "s3":
            if not HAS_S3FS:
                raise ImportError("s3fs is required for S3 backend. Install with: pip install s3fs")

            # S3ファイルシステム
            s3_options = {
                "anon": False,
            }
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                s3_options["key"] = self.config.aws_access_key_id
                s3_options["secret"] = self.config.aws_secret_access_key

            self._fs = fsspec.filesystem("s3", **s3_options)
            self._base_path = f"{self.config.s3_bucket}/{self.config.s3_prefix}"

            # ローカルキャッシュ用ファイルシステム
            if self.config.local_cache_enabled:
                self._local_cache_fs = fsspec.filesystem("file")
                Path(self.config.local_cache_path).mkdir(parents=True, exist_ok=True)
                self._load_cache_metadata()

            logger.info(f"Storage backend: s3 (s3://{self._base_path})")
            if self.config.local_cache_enabled:
                logger.info(f"Local cache: {self.config.local_cache_path}")
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def _get_full_path(self, path: str) -> str:
        """フルパスを取得"""
        if self.config.backend == "s3":
            return f"{self._base_path}/{path}"
        return str(Path(self._base_path) / path)

    def _get_local_cache_path(self, path: str) -> Path:
        """ローカルキャッシュパスを取得"""
        return Path(self.config.local_cache_path) / path

    def _is_cache_valid(self, path: str) -> bool:
        """ローカルキャッシュが有効か確認"""
        if not self.config.local_cache_enabled:
            return False

        local_path = self._get_local_cache_path(path)
        if not local_path.exists():
            return False

        # メタデータでTTLチェック
        meta = self._cache_metadata.get(path)
        if meta:
            cached_at = datetime.fromisoformat(meta.get("cached_at", "2000-01-01"))
            ttl = timedelta(hours=self.config.local_cache_ttl_hours)
            if datetime.now() - cached_at > ttl:
                logger.debug(f"Cache expired: {path}")
                return False

        return True

    def _update_cache_metadata(self, path: str, s3_etag: Optional[str] = None) -> None:
        """キャッシュメタデータを更新"""
        self._cache_metadata[path] = {
            "cached_at": datetime.now().isoformat(),
            "s3_etag": s3_etag,
        }
        self._save_cache_metadata()

    def _load_cache_metadata(self) -> None:
        """キャッシュメタデータを読み込み"""
        meta_path = Path(self.config.local_cache_path) / ".cache_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    self._cache_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._cache_metadata = {}

    def _save_cache_metadata(self) -> None:
        """キャッシュメタデータを保存"""
        meta_path = Path(self.config.local_cache_path) / ".cache_metadata.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(self._cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    # =========================================================================
    # 基本操作
    # =========================================================================

    def exists(self, path: str) -> bool:
        """ファイル/ディレクトリの存在確認"""
        # ローカルキャッシュをまず確認
        if self._is_cache_valid(path):
            return True

        full_path = self._get_full_path(path)
        try:
            return self._fs.exists(full_path)
        except Exception as e:
            logger.warning(f"Failed to check existence: {path}, {e}")
            return False

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """ディレクトリ作成"""
        full_path = self._get_full_path(path)
        try:
            self._fs.makedirs(full_path, exist_ok=exist_ok)
        except Exception:
            pass  # S3ではディレクトリは自動作成される

        # ローカルキャッシュにも作成
        if self.config.backend == "s3" and self.config.local_cache_enabled:
            local_path = self._get_local_cache_path(path)
            local_path.mkdir(parents=True, exist_ok=True)

    def list_files(self, path: str = "", pattern: str = "*") -> List[str]:
        """ファイル一覧を取得"""
        full_path = self._get_full_path(path)
        try:
            if self.config.backend == "s3":
                files = self._fs.glob(f"{full_path}/{pattern}")
                # プレフィックスを除去
                prefix = f"{self._base_path}/"
                return [f.replace(prefix, "") for f in files]
            else:
                files = self._fs.glob(f"{full_path}/{pattern}")
                prefix = f"{self._base_path}/"
                return [f.replace(prefix, "") for f in files]
        except Exception as e:
            logger.warning(f"Failed to list files: {path}, {e}")
            return []

    def delete(self, path: str) -> bool:
        """ファイル削除"""
        full_path = self._get_full_path(path)
        try:
            self._fs.rm(full_path)

            # ローカルキャッシュも削除
            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                if local_path.exists():
                    local_path.unlink()
                if path in self._cache_metadata:
                    del self._cache_metadata[path]
                    self._save_cache_metadata()

            return True
        except Exception as e:
            logger.warning(f"Failed to delete: {path}, {e}")
            return False

    # =========================================================================
    # Parquet操作
    # =========================================================================

    def read_parquet(self, path: str) -> Any:
        """
        Parquetファイルを読み込み

        Returns:
            polars.DataFrame (polars available) or pandas.DataFrame
        """
        # ローカルキャッシュから読み込み
        if self._is_cache_valid(path):
            local_path = self._get_local_cache_path(path)
            logger.debug(f"Reading from local cache: {path}")
            if HAS_POLARS:
                return pl.read_parquet(local_path)
            elif HAS_PANDAS:
                return pd.read_parquet(local_path)

        # S3/ローカルから読み込み
        full_path = self._get_full_path(path)
        try:
            if self.config.backend == "s3":
                with self._fs.open(full_path, "rb") as f:
                    if HAS_POLARS:
                        df = pl.read_parquet(f)
                    elif HAS_PANDAS:
                        df = pd.read_parquet(f)
                    else:
                        raise ImportError("polars or pandas required")

                # ローカルキャッシュに保存
                if self.config.local_cache_enabled:
                    local_path = self._get_local_cache_path(path)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if HAS_POLARS:
                        df.write_parquet(local_path)
                    elif HAS_PANDAS:
                        df.to_parquet(local_path)
                    self._update_cache_metadata(path)

                return df
            else:
                if HAS_POLARS:
                    return pl.read_parquet(full_path)
                elif HAS_PANDAS:
                    return pd.read_parquet(full_path)
        except Exception as e:
            logger.error(f"Failed to read parquet: {path}, {e}")
            raise

    def write_parquet(self, df: Any, path: str) -> None:
        """
        Parquetファイルを書き込み

        Args:
            df: polars.DataFrame or pandas.DataFrame
            path: 保存パス
        """
        full_path = self._get_full_path(path)

        try:
            if self.config.backend == "s3":
                # S3に書き込み
                with self._fs.open(full_path, "wb") as f:
                    if HAS_POLARS and hasattr(df, "write_parquet"):
                        df.write_parquet(f)
                    elif HAS_PANDAS and hasattr(df, "to_parquet"):
                        df.to_parquet(f)
                    else:
                        raise ValueError("Unknown dataframe type")

                # ローカルキャッシュにも保存
                if self.config.local_cache_enabled:
                    local_path = self._get_local_cache_path(path)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if HAS_POLARS and hasattr(df, "write_parquet"):
                        df.write_parquet(local_path)
                    elif HAS_PANDAS and hasattr(df, "to_parquet"):
                        df.to_parquet(local_path)
                    self._update_cache_metadata(path)
            else:
                # ローカルに書き込み
                Path(full_path).parent.mkdir(parents=True, exist_ok=True)
                if HAS_POLARS and hasattr(df, "write_parquet"):
                    df.write_parquet(full_path)
                elif HAS_PANDAS and hasattr(df, "to_parquet"):
                    df.to_parquet(full_path)
                else:
                    raise ValueError("Unknown dataframe type")

            logger.debug(f"Wrote parquet: {path}")
        except Exception as e:
            logger.error(f"Failed to write parquet: {path}, {e}")
            raise

    # =========================================================================
    # Pickle操作（共分散キャッシュ等）
    # =========================================================================

    def read_pickle(self, path: str) -> Any:
        """Pickleファイルを読み込み"""
        # ローカルキャッシュから読み込み
        if self._is_cache_valid(path):
            local_path = self._get_local_cache_path(path)
            logger.debug(f"Reading pickle from local cache: {path}")
            with open(local_path, "rb") as f:
                return pickle.load(f)

        full_path = self._get_full_path(path)
        try:
            with self._fs.open(full_path, "rb") as f:
                data = pickle.load(f)

            # ローカルキャッシュに保存
            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    pickle.dump(data, f)
                self._update_cache_metadata(path)

            return data
        except Exception as e:
            logger.error(f"Failed to read pickle: {path}, {e}")
            raise

    def write_pickle(self, data: Any, path: str) -> None:
        """Pickleファイルを書き込み"""
        full_path = self._get_full_path(path)

        try:
            with self._fs.open(full_path, "wb") as f:
                pickle.dump(data, f)

            # ローカルキャッシュにも保存
            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    pickle.dump(data, f)
                self._update_cache_metadata(path)

            logger.debug(f"Wrote pickle: {path}")
        except Exception as e:
            logger.error(f"Failed to write pickle: {path}, {e}")
            raise

    # =========================================================================
    # JSON操作（メタデータ等）
    # =========================================================================

    def read_json(self, path: str) -> Dict[str, Any]:
        """JSONファイルを読み込み"""
        if self._is_cache_valid(path):
            local_path = self._get_local_cache_path(path)
            with open(local_path, "r") as f:
                return json.load(f)

        full_path = self._get_full_path(path)
        try:
            with self._fs.open(full_path, "r") as f:
                data = json.load(f)

            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "w") as f:
                    json.dump(data, f)
                self._update_cache_metadata(path)

            return data
        except Exception as e:
            logger.error(f"Failed to read json: {path}, {e}")
            raise

    def write_json(self, data: Dict[str, Any], path: str) -> None:
        """JSONファイルを書き込み"""
        full_path = self._get_full_path(path)

        try:
            with self._fs.open(full_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                self._update_cache_metadata(path)

            logger.debug(f"Wrote json: {path}")
        except Exception as e:
            logger.error(f"Failed to write json: {path}, {e}")
            raise

    # =========================================================================
    # NumPy操作
    # =========================================================================

    def read_numpy(self, path: str) -> np.ndarray:
        """NumPy配列を読み込み"""
        if self._is_cache_valid(path):
            local_path = self._get_local_cache_path(path)
            return np.load(local_path)

        full_path = self._get_full_path(path)
        try:
            with self._fs.open(full_path, "rb") as f:
                data = np.load(f)

            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(local_path, data)
                self._update_cache_metadata(path)

            return data
        except Exception as e:
            logger.error(f"Failed to read numpy: {path}, {e}")
            raise

    def write_numpy(self, data: np.ndarray, path: str) -> None:
        """NumPy配列を書き込み"""
        full_path = self._get_full_path(path)

        try:
            with self._fs.open(full_path, "wb") as f:
                np.save(f, data)

            if self.config.backend == "s3" and self.config.local_cache_enabled:
                local_path = self._get_local_cache_path(path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(local_path, data)
                self._update_cache_metadata(path)

            logger.debug(f"Wrote numpy: {path}")
        except Exception as e:
            logger.error(f"Failed to write numpy: {path}, {e}")
            raise

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def clear_local_cache(self) -> int:
        """ローカルキャッシュをクリア"""
        if not self.config.local_cache_enabled:
            return 0

        cache_path = Path(self.config.local_cache_path)
        if not cache_path.exists():
            return 0

        count = 0
        for item in cache_path.rglob("*"):
            if item.is_file():
                item.unlink()
                count += 1

        self._cache_metadata = {}
        self._save_cache_metadata()

        logger.info(f"Cleared {count} files from local cache")
        return count

    def sync_to_remote(self, local_path: str) -> int:
        """
        ローカルキャッシュをリモートに同期

        Args:
            local_path: 同期するローカルパス

        Returns:
            同期したファイル数
        """
        if self.config.backend != "s3":
            logger.warning("sync_to_remote is only for S3 backend")
            return 0

        local_base = Path(local_path)
        if not local_base.exists():
            return 0

        count = 0
        for item in local_base.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(local_base))
                full_path = self._get_full_path(rel_path)

                try:
                    with open(item, "rb") as src:
                        with self._fs.open(full_path, "wb") as dst:
                            dst.write(src.read())
                    count += 1
                    logger.debug(f"Synced: {rel_path}")
                except Exception as e:
                    logger.warning(f"Failed to sync {rel_path}: {e}")

        logger.info(f"Synced {count} files to S3")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """ストレージ統計を取得"""
        stats = {
            "backend": self.config.backend,
            "base_path": self._base_path,
            "local_cache_enabled": self.config.local_cache_enabled,
        }

        if self.config.local_cache_enabled:
            cache_path = Path(self.config.local_cache_path)
            if cache_path.exists():
                files = list(cache_path.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                stats["local_cache_files"] = file_count
                stats["local_cache_size_mb"] = round(total_size / (1024 * 1024), 2)
                stats["cached_items"] = len(self._cache_metadata)

        return stats


# =============================================================================
# グローバルインスタンス管理
# =============================================================================

_storage_backend: Optional[StorageBackend] = None


def get_storage_backend(config: Optional[StorageConfig] = None) -> StorageBackend:
    """
    ストレージバックエンドを取得（シングルトン）

    Args:
        config: 初回呼び出し時の設定（省略時はローカルモード）

    Returns:
        StorageBackend インスタンス
    """
    global _storage_backend

    if _storage_backend is None:
        if config is None:
            config = StorageConfig()
        _storage_backend = StorageBackend(config)

    return _storage_backend


def reset_storage_backend() -> None:
    """ストレージバックエンドをリセット（テスト用）"""
    global _storage_backend
    _storage_backend = None


def init_s3_backend(
    bucket: str,
    prefix: str = ".cache",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    local_cache_path: str = "/tmp/.backtest_cache",
    local_cache_ttl_hours: int = 24,
) -> StorageBackend:
    """
    S3バックエンドを初期化するヘルパー関数

    Args:
        bucket: S3バケット名
        prefix: S3プレフィックス
        aws_access_key_id: AWSアクセスキーID
        aws_secret_access_key: AWSシークレットアクセスキー
        local_cache_path: ローカルキャッシュパス
        local_cache_ttl_hours: ローカルキャッシュTTL（時間）

    Returns:
        StorageBackend インスタンス
    """
    config = StorageConfig(
        backend="s3",
        s3_bucket=bucket,
        s3_prefix=prefix,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        local_cache_enabled=True,
        local_cache_path=local_cache_path,
        local_cache_ttl_hours=local_cache_ttl_hours,
    )

    global _storage_backend
    _storage_backend = StorageBackend(config)
    return _storage_backend
