"""
Storage Backend - S3必須のストレージ抽象化レイヤー

fsspecを使用してローカルファイルシステムとS3を統一インターフェースで操作。
常にhybridモードで動作: ローカルをプライマリ、S3をセカンダリ（write-through）として使用。

IMPORTANT: S3は必須。ローカルのみモードは廃止済み。

Usage:
    from src.utils.storage_backend import get_storage_backend, StorageConfig

    # S3バケット指定が必須
    backend = get_storage_backend(StorageConfig(
        s3_bucket="my-bucket",        # 必須
        s3_prefix=".cache",
        base_path=".cache",           # ローカルキャッシュパス
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
    """ストレージ設定

    S3 Required Mode:
        S3バケットの設定は必須。常にhybridモードで動作:
        - READ: Local first → S3 fallback → FileNotFoundError
        - WRITE: Local save → S3 sync (write-through)

        This ensures fast local access while maintaining S3 as durable storage.

    Raises:
        ValueError: s3_bucket が空の場合
    """

    # S3設定（必須）
    s3_bucket: str  # 必須（デフォルトなし）
    s3_prefix: str = ".cache"
    s3_region: str = "ap-northeast-1"

    # ローカルキャッシュパス（常に使用）
    base_path: str = ".cache"

    # 認証情報（環境変数からも取得可能）
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # ローカルキャッシュTTL（S3からフェッチしたファイルの有効期限）
    local_cache_ttl_hours: int = 24

    # パフォーマンス設定
    use_parquet_optimization: bool = True  # fsspec.parquet最適化

    def __post_init__(self):
        """環境変数から認証情報を取得し、S3バケットの必須チェック"""
        # S3バケット必須チェック
        if not self.s3_bucket:
            raise ValueError(
                "s3_bucket is required. Local-only mode is not supported. "
                "Set S3_BUCKET environment variable or provide s3_bucket parameter."
            )

        if self.aws_access_key_id is None:
            self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        if self.aws_secret_access_key is None:
            self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        logger.info(f"Storage config: s3_bucket={self.s3_bucket}, base_path={self.base_path}")


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
        """ファイルシステムを初期化（常にhybridモード）

        Hybrid mode: local primary + S3 write-through
        - READ: Local first → S3 fallback
        - WRITE: Local save → S3 sync
        """
        if not HAS_S3FS:
            raise ImportError("s3fs is required. Install with: pip install s3fs")

        # Local filesystem (primary)
        self._fs = fsspec.filesystem("file")
        self._base_path = self.config.base_path
        Path(self._base_path).mkdir(parents=True, exist_ok=True)

        # S3 filesystem (secondary, required)
        s3_options = {"anon": False}
        if self.config.aws_access_key_id and self.config.aws_secret_access_key:
            s3_options["key"] = self.config.aws_access_key_id
            s3_options["secret"] = self.config.aws_secret_access_key

        # Set region for proper signature (default us-east-1 causes Access Denied on other regions)
        if self.config.s3_region:
            s3_options["client_kwargs"] = {"region_name": self.config.s3_region}

        self._s3_fs = fsspec.filesystem("s3", **s3_options)
        self._s3_base_path = f"{self.config.s3_bucket}/{self.config.s3_prefix}"

        logger.info(f"Storage backend: hybrid (local={self._base_path}, s3=s3://{self._s3_base_path})")

    def _get_full_path(self, path: str) -> str:
        """ローカルフルパスを取得"""
        return str(Path(self._base_path) / path)

    def _get_s3_full_path(self, path: str) -> str:
        """S3フルパスを取得（hybridモード用）"""
        return f"{self._s3_base_path}/{path}"

    def _local_exists(self, path: str) -> bool:
        """ローカルにファイルが存在するか確認"""
        local_path = Path(self._base_path) / path
        return local_path.exists()

    def _s3_exists(self, path: str) -> bool:
        """S3にファイルが存在するか確認"""
        if self._s3_fs is None:
            return False
        s3_path = self._get_s3_full_path(path)
        try:
            return self._s3_fs.exists(s3_path)
        except Exception as e:
            logger.debug(f"S3 exists check failed for {path}: {e}")
            return False

    def _sync_to_s3(self, path: str) -> bool:
        """ローカルファイルをS3に同期（hybridモード用）"""
        if self._s3_fs is None:
            return False
        local_path = Path(self._base_path) / path
        if not local_path.exists():
            return False
        s3_path = self._get_s3_full_path(path)
        try:
            with open(local_path, "rb") as src:
                with self._s3_fs.open(s3_path, "wb") as dst:
                    dst.write(src.read())
            logger.debug(f"Synced to S3: {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to sync to S3 {path}: {e}")
            return False

    def _fetch_from_s3_to_local(self, path: str) -> bool:
        """S3からローカルにファイルをダウンロード（hybridモード用）"""
        if self._s3_fs is None:
            return False
        local_path = Path(self._base_path) / path
        s3_path = self._get_s3_full_path(path)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with self._s3_fs.open(s3_path, "rb") as src:
                with open(local_path, "wb") as dst:
                    dst.write(src.read())
            logger.debug(f"Fetched from S3 to local: {path}")
            return True
        except Exception as e:
            logger.debug(f"Failed to fetch from S3 {path}: {e}")
            return False


    # =========================================================================
    # 基本操作
    # =========================================================================

    def exists(self, path: str) -> bool:
        """ファイル/ディレクトリの存在確認

        Check order: Local first → S3 fallback
        """
        if self._local_exists(path):
            return True
        return self._s3_exists(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """ディレクトリ作成（ローカルとS3両方）"""
        # Create in local
        full_path = self._get_full_path(path)
        try:
            self._fs.makedirs(full_path, exist_ok=exist_ok)
        except Exception as e:
            logger.debug(f"Local makedirs failed for {path}: {e}")

        # Create in S3 (virtual directories)
        if self._s3_fs is not None:
            try:
                s3_path = self._get_s3_full_path(path)
                self._s3_fs.makedirs(s3_path, exist_ok=exist_ok)
            except Exception as e:
                logger.debug(f"S3 makedirs failed for {path}: {e}")  # S3 directories are virtual

    def list_files(self, path: str = "", pattern: str = "*") -> List[str]:
        """ファイル一覧を取得（ローカルとS3をマージ）"""
        full_path = self._get_full_path(path)
        try:
            local_files = set()
            s3_files = set()

            # Local files
            local_glob = self._fs.glob(f"{full_path}/{pattern}")
            prefix = f"{self._base_path}/"
            local_files = {f.replace(prefix, "") for f in local_glob}

            # S3 files
            if self._s3_fs is not None:
                s3_full_path = self._get_s3_full_path(path)
                try:
                    s3_glob = self._s3_fs.glob(f"{s3_full_path}/{pattern}")
                    s3_prefix = f"{self._s3_base_path}/"
                    s3_files = {f.replace(s3_prefix, "") for f in s3_glob}
                except Exception as e:
                    logger.debug(f"S3 glob failed for {path}/{pattern}: {e}")

            return sorted(local_files | s3_files)
        except Exception as e:
            logger.warning(f"Failed to list files: {path}, {e}")
            return []

    def delete(self, path: str) -> bool:
        """ファイル削除（ローカルとS3両方から）"""
        success = False

        # Delete from local
        local_path = Path(self._base_path) / path
        if local_path.exists():
            try:
                local_path.unlink()
                success = True
            except Exception as e:
                logger.warning(f"Failed to delete local {path}: {e}")

        # Delete from S3
        if self._s3_fs is not None:
            s3_path = self._get_s3_full_path(path)
            try:
                if self._s3_fs.exists(s3_path):
                    self._s3_fs.rm(s3_path)
                    success = True
            except Exception as e:
                logger.warning(f"Failed to delete S3 {path}: {e}")

        return success

    # =========================================================================
    # Parquet操作
    # =========================================================================

    def read_parquet(self, path: str) -> Any:
        """
        Parquetファイルを読み込み

        Workflow:
            1. Check local first
            2. If not local, fetch from S3 → save to local
            3. If not in S3, raise FileNotFoundError

        Returns:
            polars.DataFrame (polars available) or pandas.DataFrame
        """
        local_path = Path(self._base_path) / path

        # 1. Try local
        if local_path.exists():
            logger.debug(f"Reading from local: {path}")
            if HAS_POLARS:
                return pl.read_parquet(local_path)
            elif HAS_PANDAS:
                return pd.read_parquet(local_path)

        # 2. Try S3 → cache to local
        if self._s3_exists(path):
            logger.debug(f"Fetching from S3: {path}")
            if self._fetch_from_s3_to_local(path):
                if HAS_POLARS:
                    return pl.read_parquet(local_path)
                elif HAS_PANDAS:
                    return pd.read_parquet(local_path)

        # 3. Not found
        raise FileNotFoundError(f"File not found in local or S3: {path}")

    def write_parquet(self, df: Any, path: str) -> None:
        """
        Parquetファイルを書き込み

        Workflow:
            1. Write to local
            2. Sync to S3 (write-through)

        Args:
            df: polars.DataFrame or pandas.DataFrame
            path: 保存パス
        """
        local_path = Path(self._base_path) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Write to local
        if HAS_POLARS and hasattr(df, "write_parquet"):
            df.write_parquet(local_path)
        elif HAS_PANDAS and hasattr(df, "to_parquet"):
            df.to_parquet(local_path)
        else:
            raise ValueError("Unknown dataframe type")

        # 2. Sync to S3 (write-through)
        self._sync_to_s3(path)

        logger.debug(f"Wrote parquet: {path}")

    # =========================================================================
    # Pickle操作（共分散キャッシュ等）
    # =========================================================================

    def read_pickle(self, path: str) -> Any:
        """Pickleファイルを読み込み"""
        local_path = Path(self._base_path) / path
        if local_path.exists():
            with open(local_path, "rb") as f:
                return pickle.load(f)
        if self._s3_exists(path) and self._fetch_from_s3_to_local(path):
            with open(local_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Pickle not found: {path}")

    def write_pickle(self, data: Any, path: str) -> None:
        """Pickleファイルを書き込み"""
        local_path = Path(self._base_path) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            pickle.dump(data, f)
        self._sync_to_s3(path)
        logger.debug(f"Wrote pickle: {path}")

    # =========================================================================
    # JSON操作（メタデータ等）
    # =========================================================================

    def read_json(self, path: str) -> Dict[str, Any]:
        """JSONファイルを読み込み"""
        local_path = Path(self._base_path) / path
        if local_path.exists():
            with open(local_path, "r") as f:
                return json.load(f)
        if self._s3_exists(path) and self._fetch_from_s3_to_local(path):
            with open(local_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"JSON not found: {path}")

    def write_json(self, data: Dict[str, Any], path: str) -> None:
        """JSONファイルを書き込み"""
        local_path = Path(self._base_path) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._sync_to_s3(path)
        logger.debug(f"Wrote json: {path}")

    # =========================================================================
    # NumPy操作
    # =========================================================================

    def read_numpy(self, path: str) -> np.ndarray:
        """NumPy配列を読み込み"""
        local_path = Path(self._base_path) / path
        if local_path.exists():
            return np.load(local_path)
        if self._s3_exists(path) and self._fetch_from_s3_to_local(path):
            return np.load(local_path)
        raise FileNotFoundError(f"NumPy not found: {path}")

    def write_numpy(self, data: np.ndarray, path: str) -> None:
        """NumPy配列を書き込み"""
        local_path = Path(self._base_path) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(local_path, data)
        self._sync_to_s3(path)
        logger.debug(f"Wrote numpy: {path}")

    # =========================================================================
    # テキスト・YAML操作
    # =========================================================================

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """テキストファイル読み込み

        Args:
            path: 読み込むパス
            encoding: エンコーディング（デフォルト: utf-8）

        Returns:
            ファイル内容の文字列
        """
        local_path = Path(self._base_path) / path
        if local_path.exists():
            return local_path.read_text(encoding=encoding)
        if self._s3_exists(path) and self._fetch_from_s3_to_local(path):
            return local_path.read_text(encoding=encoding)
        raise FileNotFoundError(f"Text file not found: {path}")

    def write_text(self, content: str, path: str, encoding: str = "utf-8") -> None:
        """テキストファイル書き込み

        Args:
            content: 書き込む内容
            path: 書き込むパス
            encoding: エンコーディング（デフォルト: utf-8）
        """
        local_path = Path(self._base_path) / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding=encoding)
        self._sync_to_s3(path)
        logger.debug(f"Wrote text: {path}")

    def read_yaml(self, path: str) -> Dict[str, Any]:
        """YAML読み込み

        Args:
            path: 読み込むパス

        Returns:
            YAML内容の辞書
        """
        import yaml
        content = self.read_text(path)
        return yaml.safe_load(content) or {}

    def write_yaml(self, data: Dict[str, Any], path: str) -> None:
        """YAML書き込み

        Args:
            data: 書き込むデータ
            path: 書き込むパス
        """
        import yaml
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        self.write_text(content, path)

    def delete_directory(self, path: str, recursive: bool = True) -> bool:
        """ディレクトリ削除（ローカルとS3両方から）

        Args:
            path: 削除するパス
            recursive: 再帰的に削除するか

        Returns:
            成功したかどうか
        """
        local_path = Path(self._base_path) / path
        if local_path.exists():
            if recursive:
                shutil.rmtree(local_path)
            else:
                local_path.rmdir()
        if self._s3_fs:
            try:
                self._s3_fs.rm(self._get_s3_full_path(path), recursive=recursive)
            except Exception as e:
                logger.debug(f"S3 delete failed (may not exist): {e}")
        return True

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def clear_local_cache(self) -> int:
        """ローカルキャッシュをクリア"""
        cache_path = Path(self._base_path)
        if not cache_path.exists():
            return 0

        count = 0
        for item in cache_path.rglob("*"):
            if item.is_file():
                item.unlink()
                count += 1

        logger.info(f"Cleared {count} files from local cache")
        return count

    def sync_to_remote(self, local_path: str | None = None) -> int:
        """
        ローカルキャッシュをS3に同期

        Args:
            local_path: 同期するローカルパス（省略時はbase_path）

        Returns:
            同期したファイル数
        """
        local_base = Path(local_path) if local_path else Path(self._base_path)
        if not local_base.exists():
            return 0

        count = 0
        for item in local_base.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(local_base))
                if self._sync_to_s3(rel_path):
                    count += 1

        logger.info(f"Synced {count} files to S3")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """ストレージ統計を取得"""
        stats = {
            "backend": "hybrid",
            "base_path": self._base_path,
            "s3_base_path": self._s3_base_path,
        }

        # Local stats
        local_path = Path(self._base_path)
        if local_path.exists():
            files = list(local_path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            stats["local_files"] = file_count
            stats["local_size_mb"] = round(total_size / (1024 * 1024), 2)

        # S3 stats
        if self._s3_fs is not None:
            try:
                s3_files = self._s3_fs.glob(f"{self._s3_base_path}/**/*")
                stats["s3_files"] = len([f for f in s3_files if not f.endswith("/")])
            except Exception:
                stats["s3_files"] = "unknown"

        return stats


# =============================================================================
# グローバルインスタンス管理
# =============================================================================

_storage_backend: Optional[StorageBackend] = None


def get_storage_backend(config: Optional[StorageConfig] = None) -> StorageBackend:
    """
    ストレージバックエンドを取得（シングルトン）

    Args:
        config: 初回呼び出し時の設定（s3_bucket必須）

    Returns:
        StorageBackend インスタンス

    Raises:
        ValueError: config が None で s3_bucket 環境変数もない場合
    """
    global _storage_backend

    if _storage_backend is None:
        if config is None:
            # 環境変数からS3バケットを取得
            s3_bucket = os.environ.get("S3_BUCKET")
            if not s3_bucket:
                raise ValueError(
                    "S3_BUCKET environment variable is required. "
                    "Local-only mode is not supported."
                )
            config = StorageConfig(s3_bucket=s3_bucket)
        _storage_backend = StorageBackend(config)

    return _storage_backend


def reset_storage_backend() -> None:
    """ストレージバックエンドをリセット（テスト用）"""
    global _storage_backend
    _storage_backend = None


def init_storage_backend(
    s3_bucket: str,
    s3_prefix: str = ".cache",
    base_path: str = ".cache",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    local_cache_ttl_hours: int = 24,
) -> StorageBackend:
    """
    ストレージバックエンドを初期化するヘルパー関数

    Args:
        s3_bucket: S3バケット名（必須）
        s3_prefix: S3プレフィックス
        base_path: ローカルキャッシュパス
        aws_access_key_id: AWSアクセスキーID
        aws_secret_access_key: AWSシークレットアクセスキー
        local_cache_ttl_hours: ローカルキャッシュTTL（時間）

    Returns:
        StorageBackend インスタンス
    """
    config = StorageConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        base_path=base_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        local_cache_ttl_hours=local_cache_ttl_hours,
    )

    global _storage_backend
    _storage_backend = StorageBackend(config)
    return _storage_backend
