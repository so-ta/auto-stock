"""
共通定数モジュール (QA-009: Hardcoded values migration)

ハードコード値を設定ファイルから読み込むユーティリティを提供。
デフォルト値はフォールバックとして使用される。

使用例:
    from src.config.constants import get_constant, TRADING_DAYS_PER_YEAR

    # 定数として使用
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    # 設定から取得（フォールバック付き）
    lookback = get_constant("backtest.constants.default_lookback_medium", 60)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =============================================================================
# Default Constants (Fallback values)
# =============================================================================

# Trading calendar
TRADING_DAYS_PER_YEAR: int = 252
TRADING_DAYS_PER_MONTH: int = 21
TRADING_DAYS_PER_WEEK: int = 5

# Default lookback periods
DEFAULT_LOOKBACK_SHORT: int = 20
DEFAULT_LOOKBACK_MEDIUM: int = 60
DEFAULT_LOOKBACK_LONG: int = 252

# Covariance estimation
COV_HALFLIFE: int = 60

# Chunk sizes
DEFAULT_CHUNK_SIZE: int = 1000
STREAMING_CHUNK_SIZE: int = 100

# Cache sizes
DEFAULT_CACHE_SIZE: int = 1000
SIGNAL_CACHE_SIZE: int = 500
COVARIANCE_CACHE_SIZE: int = 100

# Signal parameters
MOMENTUM_SHORT_WINDOW: int = 20
MOMENTUM_MEDIUM_WINDOW: int = 60
MOMENTUM_LONG_WINDOW: int = 120

MEAN_REVERSION_WINDOW: int = 20
MEAN_REVERSION_Z_THRESHOLD: float = 2.0
MEAN_REVERSION_HALFLIFE: int = 21

VOLATILITY_WINDOW: int = 20
TARGET_VOLATILITY: float = 0.15

RSI_PERIOD: int = 14
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30

BOLLINGER_WINDOW: int = 20
BOLLINGER_NUM_STD: float = 2.0

REGIME_LOOKBACK: int = 60

# Meta layer
SCORER_MIN_WEIGHT: float = 0.05
SCORER_MAX_WEIGHT: float = 0.30

# Thresholds
LARGE_UNIVERSE_THRESHOLD: int = 500


# =============================================================================
# Configuration Cache
# =============================================================================

_config_cache: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def _load_config() -> Dict[str, Any]:
    """設定ファイルを読み込む（キャッシュ付き）"""
    global _config_cache, _config_path

    if _config_cache is not None:
        return _config_cache

    # 設定ファイルパスを探索
    search_paths = [
        Path("config/default.yaml"),
        Path("../config/default.yaml"),
        Path(__file__).parent.parent.parent / "config" / "default.yaml",
    ]

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _config_cache = yaml.safe_load(f) or {}
                    _config_path = path
                    logger.debug(f"Loaded config from {path}")
                    return _config_cache
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

    logger.warning("Config file not found, using defaults")
    _config_cache = {}
    return _config_cache


def get_constant(
    path: str,
    default: T,
    config: Optional[Dict[str, Any]] = None,
) -> T:
    """
    設定パスから定数を取得

    Parameters
    ----------
    path : str
        ドット区切りの設定パス（例: "backtest.constants.trading_days_per_year"）
    default : T
        設定が見つからない場合のデフォルト値
    config : Dict, optional
        カスタム設定辞書（Noneの場合はdefault.yamlから読み込み）

    Returns
    -------
    T
        設定値またはデフォルト値

    Examples
    --------
    >>> lookback = get_constant("signals.momentum.short_window", 20)
    >>> chunk_size = get_constant("backtest.constants.default_chunk_size", 1000)
    """
    cfg = config if config is not None else _load_config()

    try:
        keys = path.split(".")
        value = cfg

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        # 型チェック
        if type(value) == type(default):
            return value
        elif isinstance(default, int) and isinstance(value, (int, float)):
            return int(value)  # type: ignore
        elif isinstance(default, float) and isinstance(value, (int, float)):
            return float(value)  # type: ignore
        else:
            return value  # type: ignore

    except Exception as e:
        logger.debug(f"Failed to get config value for {path}: {e}")
        return default


def get_nested_config(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    ネストされた設定セクションを取得

    Parameters
    ----------
    path : str
        ドット区切りの設定パス
    config : Dict, optional
        カスタム設定辞書

    Returns
    -------
    Optional[Dict[str, Any]]
        設定セクション（見つからない場合はNone）
    """
    cfg = config if config is not None else _load_config()

    try:
        keys = path.split(".")
        value = cfg

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        if isinstance(value, dict):
            return value
        return None

    except Exception:
        return None


def reload_config() -> None:
    """設定キャッシュをクリアして再読み込み"""
    global _config_cache
    _config_cache = None
    _load_config()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_trading_days(period: str = "year") -> int:
    """取引日数を取得"""
    mapping = {
        "year": ("backtest.constants.trading_days_per_year", TRADING_DAYS_PER_YEAR),
        "month": ("backtest.constants.trading_days_per_month", TRADING_DAYS_PER_MONTH),
        "week": ("backtest.constants.trading_days_per_week", TRADING_DAYS_PER_WEEK),
    }
    path, default = mapping.get(period, mapping["year"])
    return get_constant(path, default)


def get_lookback(period: str = "medium") -> int:
    """ルックバック期間を取得"""
    mapping = {
        "short": ("backtest.constants.default_lookback_short", DEFAULT_LOOKBACK_SHORT),
        "medium": ("backtest.constants.default_lookback_medium", DEFAULT_LOOKBACK_MEDIUM),
        "long": ("backtest.constants.default_lookback_long", DEFAULT_LOOKBACK_LONG),
    }
    path, default = mapping.get(period, mapping["medium"])
    return get_constant(path, default)


def get_signal_param(signal: str, param: str, default: T) -> T:
    """シグナルパラメータを取得"""
    path = f"signals.{signal}.{param}"
    return get_constant(path, default)


def get_meta_param(section: str, param: str, default: T) -> T:
    """メタレイヤーパラメータを取得"""
    path = f"meta.{section}.{param}"
    return get_constant(path, default)


# =============================================================================
# Module initialization
# =============================================================================

# プリロード（オプション）
# _load_config()
