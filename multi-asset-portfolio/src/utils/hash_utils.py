"""統一ハッシュユーティリティ

複数のモジュールで重複していたハッシュ計算ロジックを統合。
キャッシュキー生成、設定ハッシュ、銘柄リストハッシュなどを提供。
"""
import hashlib
import json
from typing import Any


def compute_hash(
    data: str | bytes, algorithm: str = "md5", truncate: int = 16
) -> str:
    """汎用ハッシュ計算

    Args:
        data: ハッシュ対象のデータ（文字列またはバイト列）
        algorithm: ハッシュアルゴリズム ("md5" or "sha256")
        truncate: ハッシュを切り詰める文字数（0以下で切り詰めなし）

    Returns:
        16進数ハッシュ文字列
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = hashlib.md5(data) if algorithm == "md5" else hashlib.sha256(data)
    return h.hexdigest()[:truncate] if truncate > 0 else h.hexdigest()


def compute_universe_hash(symbols: list[str], truncate: int = 16) -> str:
    """銘柄リストのハッシュ

    Args:
        symbols: 銘柄シンボルのリスト
        truncate: ハッシュを切り詰める文字数

    Returns:
        銘柄リストのMD5ハッシュ（ソート済み）
    """
    return compute_hash(",".join(sorted(symbols)), "md5", truncate)


def compute_config_hash(
    config: dict[str, Any], truncate: int = 16, algorithm: str = "sha256"
) -> str:
    """設定辞書のハッシュ

    Args:
        config: 設定辞書
        truncate: ハッシュを切り詰める文字数
        algorithm: ハッシュアルゴリズム

    Returns:
        設定のハッシュ
    """
    return compute_hash(
        json.dumps(config, sort_keys=True, default=str), algorithm, truncate
    )


def compute_cache_key(*parts: Any, truncate: int = 16) -> str:
    """キャッシュキー生成

    可変引数を"|"で結合してハッシュ化。

    Args:
        *parts: キーの構成要素
        truncate: ハッシュを切り詰める文字数

    Returns:
        キャッシュキー文字列
    """
    return compute_hash("|".join(str(p) for p in parts), "md5", truncate)
