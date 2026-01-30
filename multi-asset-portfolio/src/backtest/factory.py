"""
Backtest Engine Factory - シンプル化されたエンジン生成

全てのバックテストはNumba JIT高速化エンジン（BacktestEngine）で実行。
1137x高速化により、他のエンジンは不要となった。

使用例:
    from src.backtest.factory import create_engine

    # 基本的な使用法
    engine = create_engine()

    # 設定を指定
    engine = create_engine(config=unified_config)

    # キーワード引数で直接設定
    engine = create_engine(
        initial_capital=1_000_000,
        commission_rate=0.001,
    )
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.base import UnifiedBacktestConfig
    from src.backtest.fast_engine import BacktestEngine


def create_engine(
    config: "UnifiedBacktestConfig | None" = None,
    **kwargs,
) -> "BacktestEngine":
    """
    唯一のバックテストエンジンを生成

    Numba JIT高速化エンジン（1137x高速）を返す。
    従来の複数エンジン選択は不要となった。

    Args:
        config: 統一バックテスト設定またはBacktestConfig（省略可）
        **kwargs: エンジン直接設定
            - start_date: 開始日（config未指定時は必須）
            - end_date: 終了日（config未指定時は必須）
            - initial_capital: 初期資金（デフォルト: 100,000）
            - その他BacktestConfigの引数

    Returns:
        BacktestEngine: Numba JIT高速化エンジン

    Examples:
        # UnifiedBacktestConfigから生成
        config = UnifiedBacktestConfig(...)
        engine = create_engine(config=config)

        # BacktestConfigから生成
        from src.backtest.fast_engine import BacktestConfig
        config = BacktestConfig(start_date=..., end_date=...)
        engine = create_engine(config=config)

        # 直接パラメータ指定
        from datetime import datetime
        engine = create_engine(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=10_000_000,
        )
    """
    from src.backtest.fast_engine import BacktestEngine, BacktestConfig
    from src.backtest.base import UnifiedBacktestConfig

    # configが指定されている場合
    if config is not None:
        # UnifiedBacktestConfig から BacktestConfig への変換
        if isinstance(config, UnifiedBacktestConfig):
            fast_config = BacktestConfig(
                start_date=config.start_date,
                end_date=config.end_date,
                rebalance_frequency=config.rebalance_frequency,
                initial_capital=config.initial_capital,
                transaction_cost_bps=config.transaction_cost_bps,
                slippage_bps=config.slippage_bps,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
            )
            return BacktestEngine(fast_config, **kwargs)
        return BacktestEngine(config, **kwargs)

    # kwargsにconfigが含まれている場合
    if "config" in kwargs:
        cfg = kwargs.pop("config")
        return BacktestEngine(cfg, **kwargs)

    # kwargsからBacktestConfigを構築
    if "start_date" in kwargs and "end_date" in kwargs:
        cfg = BacktestConfig(**kwargs)
        return BacktestEngine(cfg)

    raise ValueError(
        "config または start_date/end_date が必要です。"
        "例: create_engine(config=config) または "
        "create_engine(start_date=datetime(2020,1,1), end_date=datetime(2024,12,31))"
    )


def create(
    mode: str | None = None,
    config: "UnifiedBacktestConfig | None" = None,
    **kwargs,
) -> "BacktestEngine":
    """
    後方互換用create関数（非推奨）

    以前のAPIとの互換性のために残されている。
    新規コードでは create_engine() を使用すること。

    Args:
        mode: エンジンモード（無視される、非推奨）
        config: 統一バックテスト設定
        **kwargs: エンジン設定

    Returns:
        BacktestEngine: Numba JIT高速化エンジン

    .. deprecated::
        mode引数は無視されます。全てBacktestEngineに統一されました。
        create_engine() を使用してください。
    """
    if mode is not None:
        warnings.warn(
            f"mode引数 '{mode}' は非推奨です。"
            "全てBacktestEngineに統一されました。"
            "create_engine() を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
    return create_engine(config=config, **kwargs)


# 後方互換エイリアス
BacktestEngineFactory = None  # 削除済み、使用するとエラー


def _raise_removed_error(*args, **kwargs):
    """削除されたAPIへのアクセス時にエラーを発生"""
    raise RuntimeError(
        "BacktestEngineFactory は削除されました。"
        "create_engine() を使用してください。"
    )


# 削除されたAPIへのアクセスをエラーにする
list_engines = _raise_removed_error
recommend_engine = _raise_removed_error
init_ray = _raise_removed_error
create_ray_engine = _raise_removed_error
