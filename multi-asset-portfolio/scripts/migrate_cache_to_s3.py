#!/usr/bin/env python3
"""
キャッシュS3マイグレーションスクリプト

既存のローカルキャッシュをS3にアップロードする。

Usage:
    # 環境変数で認証情報を設定
    export AWS_ACCESS_KEY_ID=xxx
    export AWS_SECRET_ACCESS_KEY=xxx

    # 実行
    python scripts/migrate_cache_to_s3.py

    # ドライラン（実際にはアップロードしない）
    python scripts/migrate_cache_to_s3.py --dry-run

    # 特定のディレクトリのみ
    python scripts/migrate_cache_to_s3.py --source .cache/signals
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_file_list(source_dir: Path) -> List[Tuple[Path, int]]:
    """
    アップロード対象のファイル一覧を取得

    Returns:
        List of (path, size_bytes)
    """
    files = []
    for item in source_dir.rglob("*"):
        if item.is_file():
            # メタデータファイルはスキップ
            if item.name.startswith("."):
                continue
            files.append((item, item.stat().st_size))
    return files


def migrate_to_s3(
    source_dir: str,
    bucket: str,
    prefix: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    ローカルキャッシュをS3にアップロード

    Returns:
        Tuple of (uploaded_count, skipped_count, failed_count)
    """
    try:
        import s3fs
    except ImportError:
        logger.error("s3fs is required. Install with: pip install s3fs")
        sys.exit(1)

    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 0, 0, 0

    # S3ファイルシステム初期化
    fs = s3fs.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
    )

    # ファイル一覧取得
    files = get_file_list(source_path)
    total_size = sum(size for _, size in files)

    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: s3://{bucket}/{prefix}")
    logger.info(f"Files: {len(files)}")
    logger.info(f"Total size: {total_size / (1024*1024):.2f} MB")

    if dry_run:
        logger.info("DRY RUN - No files will be uploaded")
        for file_path, size in files:
            rel_path = file_path.relative_to(source_path)
            logger.info(f"  Would upload: {rel_path} ({size / 1024:.1f} KB)")
        return len(files), 0, 0

    uploaded = 0
    skipped = 0
    failed = 0

    for file_path, size in files:
        rel_path = file_path.relative_to(source_path)
        s3_key = f"{bucket}/{prefix}/{rel_path}"

        try:
            # 既に存在するかチェック
            if fs.exists(s3_key):
                # サイズが同じならスキップ
                s3_info = fs.info(s3_key)
                if s3_info.get("size") == size:
                    logger.debug(f"Skipped (exists): {rel_path}")
                    skipped += 1
                    continue

            # アップロード
            with open(file_path, "rb") as src:
                with fs.open(s3_key, "wb") as dst:
                    dst.write(src.read())

            logger.info(f"Uploaded: {rel_path} ({size / 1024:.1f} KB)")
            uploaded += 1

        except Exception as e:
            logger.error(f"Failed: {rel_path} - {e}")
            failed += 1

    return uploaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Migrate local cache to S3")
    parser.add_argument(
        "--source",
        type=str,
        default=".cache",
        help="Source directory (default: .cache)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("S3_CACHE_BUCKET", "stock-local-dev-014498665038"),
        help="S3 bucket name",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=".cache",
        help="S3 prefix (default: .cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (don't actually upload)",
    )
    args = parser.parse_args()

    # 認証情報取得
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)

    # マイグレーション実行
    uploaded, skipped, failed = migrate_to_s3(
        source_dir=args.source,
        bucket=args.bucket,
        prefix=args.prefix,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        dry_run=args.dry_run,
    )

    # 結果表示
    logger.info("=" * 60)
    logger.info("Migration completed")
    logger.info(f"  Uploaded: {uploaded}")
    logger.info(f"  Skipped:  {skipped}")
    logger.info(f"  Failed:   {failed}")
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
