"""
Regime Adaptive Signal Parameters - レジーム適応シグナルパラメータ

市場レジームに応じてシグナルパラメータを自動調整する。
各レジームに最適化されたパラメータセットを提供し、
シグナル生成の効率を向上させる。

主要コンポーネント:
- REGIME_SIGNAL_PARAMS: レジーム別パラメータ定義
- RegimeAdaptiveParams: パラメータ調整クラス
- SignalParamSet: シグナルパラメータセットデータクラス

設計根拠:
- bull_trend: 長期トレンドを追うため、長めのルックバック
- bear_market: 素早い反応のため、短いルックバック
- range_bound: 短期振動を捉えるため、最短ルックバック
- high_vol: ノイズに強い設定、低いスケール

使用例:
    from src.signals.regime_adaptive_params import (
        RegimeAdaptiveParams,
        get_regime_params,
        adjust_signal_params,
    )

    # レジームに応じたパラメータ取得
    params = get_regime_params("bull_trend", "momentum")
    print(f"Momentum params for bull trend: {params}")

    # 基本パラメータをレジームで調整
    base_params = {"type": "momentum", "lookback": 20}
    adjusted = adjust_signal_params(base_params, "bear_market")
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# レジーム定義
# =============================================================================

class MarketRegime(str, Enum):
    """市場レジーム"""
    BULL_TREND = "bull_trend"
    BEAR_MARKET = "bear_market"
    RANGE_BOUND = "range_bound"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    DEFAULT = "default"


# =============================================================================
# レジーム別シグナルパラメータ定義
# =============================================================================

REGIME_SIGNAL_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    # ========================================
    # Bull Trend: 長期トレンドを追う
    # ========================================
    "bull_trend": {
        "momentum": {
            "lookback": [40, 60, 90],      # 長期トレンドを捉える
            "scale": 3.0,                  # 中程度のスケール
            "description": "Long-term trend following",
        },
        "bollinger": {
            "period": [30, 40],            # 長期移動平均
            "num_std": [2.5, 3.0],         # 広いバンド（偽シグナル抑制）
            "description": "Wide bands for trend continuation",
        },
        "rsi": {
            "period": [21],                # 長期RSI
            "oversold": 25,                # 深い売られ過ぎ
            "overbought": 75,              # 浅い買われ過ぎ（トレンド継続を許容）
            "description": "Asymmetric levels favoring trend",
        },
        "zscore": {
            "period": [30, 40],
            "entry_threshold": 2.5,
            "description": "Wide threshold for trend",
        },
        "trend": {
            "fast_period": [10, 20],
            "slow_period": [50, 100],
            "description": "Standard trend following",
        },
    },

    # ========================================
    # Bear Market: 素早い反応
    # ========================================
    "bear_market": {
        "momentum": {
            "lookback": [10, 20, 40],      # 短〜中期ルックバック
            "scale": 8.0,                  # 高いスケール（敏感に反応）
            "description": "Quick reaction to momentum shifts",
        },
        "bollinger": {
            "period": [14, 20],            # 短期移動平均
            "num_std": [1.5, 2.0],         # 狭いバンド（早期シグナル）
            "description": "Tight bands for quick signals",
        },
        "rsi": {
            "period": [10, 14],            # 短期RSI
            "oversold": 20,                # 極端な売られ過ぎ
            "overbought": 80,              # 極端な買われ過ぎ
            "description": "Extreme levels for oversold bounces",
        },
        "zscore": {
            "period": [14, 20],
            "entry_threshold": 1.5,        # 早めのエントリー
            "description": "Quick mean reversion entries",
        },
        "trend": {
            "fast_period": [5, 10],
            "slow_period": [20, 40],
            "description": "Fast trend detection",
        },
    },

    # ========================================
    # Range Bound: 短期振動を捉える
    # ========================================
    "range_bound": {
        "momentum": {
            "lookback": [5, 10, 20],       # 最短ルックバック
            "scale": 10.0,                 # 最高スケール（小さな動きを捉える）
            "description": "Capture short-term oscillations",
        },
        "bollinger": {
            "period": [14, 20],            # 中程度の期間
            "num_std": [2.0, 2.5],         # 標準的なバンド幅
            "description": "Standard bands for range trading",
        },
        "rsi": {
            "period": [9, 14],             # 短期RSI
            "oversold": 30,                # 標準レベル
            "overbought": 70,              # 標準レベル
            "description": "Standard levels for mean reversion",
        },
        "zscore": {
            "period": [10, 20],
            "entry_threshold": 2.0,
            "description": "Standard Z-score for range",
        },
        "trend": {
            "fast_period": [3, 5],
            "slow_period": [10, 20],
            "description": "Ultra-short trend detection",
        },
    },

    # ========================================
    # High Volatility: ノイズに強い設定
    # ========================================
    "high_vol": {
        "momentum": {
            "lookback": [20, 40, 60],      # 中〜長期ルックバック
            "scale": 2.0,                  # 低いスケール（ノイズ耐性）
            "description": "Noise-resistant momentum",
        },
        "bollinger": {
            "period": [20, 25],            # 中期移動平均
            "num_std": [2.5, 3.0],         # 広いバンド（ボラティリティ対応）
            "description": "Wide bands for high volatility",
        },
        "rsi": {
            "period": [14, 21],            # 中〜長期RSI
            "oversold": 20,                # 極端レベル
            "overbought": 80,              # 極端レベル
            "description": "Extreme levels to filter noise",
        },
        "zscore": {
            "period": [20, 30],
            "entry_threshold": 3.0,        # 高い閾値（偽シグナル抑制）
            "description": "High threshold for noise resistance",
        },
        "trend": {
            "fast_period": [10, 20],
            "slow_period": [40, 60],
            "description": "Smoothed trend for volatility",
        },
    },

    # ========================================
    # Low Volatility: 追従重視
    # ========================================
    "low_vol": {
        "momentum": {
            "lookback": [10, 20, 30],      # 短〜中期ルックバック
            "scale": 8.0,                  # 高いスケール（小さな動きを捉える）
            "description": "Sensitive momentum for low vol",
        },
        "bollinger": {
            "period": [15, 20],            # 短〜中期移動平均
            "num_std": [1.5, 2.0],         # 狭いバンド
            "description": "Tight bands for small moves",
        },
        "rsi": {
            "period": [10, 14],            # 短期RSI
            "oversold": 35,                # やや浅いレベル
            "overbought": 65,              # やや浅いレベル
            "description": "Shallow levels for quick signals",
        },
        "zscore": {
            "period": [15, 20],
            "entry_threshold": 1.5,
            "description": "Low threshold for small deviations",
        },
        "trend": {
            "fast_period": [5, 10],
            "slow_period": [20, 30],
            "description": "Responsive trend for low vol",
        },
    },

    # ========================================
    # Default: バランス設定
    # ========================================
    "default": {
        "momentum": {
            "lookback": [20, 40, 60],
            "scale": 5.0,
            "description": "Balanced momentum",
        },
        "bollinger": {
            "period": [20],
            "num_std": [2.0],
            "description": "Standard Bollinger settings",
        },
        "rsi": {
            "period": [14],
            "oversold": 30,
            "overbought": 70,
            "description": "Standard RSI levels",
        },
        "zscore": {
            "period": [20],
            "entry_threshold": 2.0,
            "description": "Standard Z-score settings",
        },
        "trend": {
            "fast_period": [10],
            "slow_period": [40],
            "description": "Standard trend settings",
        },
    },
}


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class SignalParamSet:
    """シグナルパラメータセット

    Attributes:
        signal_type: シグナルタイプ（momentum, bollinger, rsi等）
        regime: 適用レジーム
        params: パラメータ辞書
        description: 説明
        created_at: 作成日時
    """
    signal_type: str
    regime: str
    params: dict[str, Any]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "signal_type": self.signal_type,
            "regime": self.regime,
            "params": self.params,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }

    def get_primary_values(self) -> dict[str, Any]:
        """主要パラメータ値を取得（リストの場合は最初の値）"""
        result = {}
        for key, value in self.params.items():
            if key == "description":
                continue
            if isinstance(value, list):
                result[key] = value[0] if value else None
            else:
                result[key] = value
        return result


@dataclass
class RegimeParamAdjustment:
    """レジームパラメータ調整結果

    Attributes:
        original_params: 元のパラメータ
        adjusted_params: 調整後のパラメータ
        regime: 適用されたレジーム
        adjustments_made: 適用された調整のリスト
    """
    original_params: dict[str, Any]
    adjusted_params: dict[str, Any]
    regime: str
    adjustments_made: list[str] = field(default_factory=list)


# =============================================================================
# メインクラス
# =============================================================================

class RegimeAdaptiveParams:
    """レジーム適応パラメータクラス

    市場レジームに応じてシグナルパラメータを自動調整する。

    Usage:
        adapter = RegimeAdaptiveParams()

        # レジーム別パラメータ取得
        params = adapter.get_params("bull_trend", "momentum")

        # 基本パラメータをレジームで調整
        base = {"type": "momentum", "lookback": 20, "scale": 5.0}
        adjusted = adapter.adjust_for_regime(base, "bear_market")
    """

    def __init__(
        self,
        regime_params: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        """初期化

        Args:
            regime_params: カスタムレジームパラメータ（Noneの場合はデフォルト使用）
        """
        self.regime_params = regime_params or REGIME_SIGNAL_PARAMS

    def get_params(
        self,
        regime: str,
        signal_type: str,
    ) -> dict[str, Any]:
        """レジーム別シグナルパラメータを取得

        Args:
            regime: 市場レジーム（bull_trend, bear_market等）
            signal_type: シグナルタイプ（momentum, bollinger, rsi等）

        Returns:
            パラメータ辞書（見つからない場合は空辞書）
        """
        regime_data = self.regime_params.get(regime, {})
        params = regime_data.get(signal_type, {})

        if not params and regime != "default":
            # レジームが見つからない場合はデフォルトを使用
            default_data = self.regime_params.get("default", {})
            params = default_data.get(signal_type, {})
            if params:
                logger.debug(
                    "Using default params for %s/%s (regime %s not found)",
                    signal_type, regime, regime,
                )

        return deepcopy(params)

    def get_param_set(
        self,
        regime: str,
        signal_type: str,
    ) -> SignalParamSet:
        """パラメータセットを取得

        Args:
            regime: 市場レジーム
            signal_type: シグナルタイプ

        Returns:
            SignalParamSet
        """
        params = self.get_params(regime, signal_type)
        description = params.pop("description", "")

        return SignalParamSet(
            signal_type=signal_type,
            regime=regime,
            params=params,
            description=description,
        )

    def adjust_for_regime(
        self,
        base_params: dict[str, Any],
        regime: str,
    ) -> dict[str, Any]:
        """基本パラメータをレジームに応じて調整

        Args:
            base_params: 基本パラメータ（"type"キーでシグナルタイプを指定）
            regime: 適用するレジーム

        Returns:
            調整後のパラメータ辞書
        """
        signal_type = base_params.get("type", "")
        if not signal_type:
            logger.warning("No 'type' key in base_params, returning unchanged")
            return deepcopy(base_params)

        regime_overrides = self.get_params(regime, signal_type)
        if not regime_overrides:
            return deepcopy(base_params)

        # 基本パラメータをコピーしてオーバーライド適用
        adjusted = deepcopy(base_params)

        for key, value in regime_overrides.items():
            if key == "description":
                continue  # descriptionは適用しない

            if key in adjusted:
                # 既存キーのオーバーライド
                if isinstance(value, list):
                    # リストの場合は最初の値を使用
                    adjusted[key] = value[0]
                else:
                    adjusted[key] = value
            else:
                # 新規キーの追加
                if isinstance(value, list):
                    adjusted[key] = value[0]
                else:
                    adjusted[key] = value

        return adjusted

    def get_adjustment_result(
        self,
        base_params: dict[str, Any],
        regime: str,
    ) -> RegimeParamAdjustment:
        """調整結果を詳細に取得

        Args:
            base_params: 基本パラメータ
            regime: 適用するレジーム

        Returns:
            RegimeParamAdjustment
        """
        adjusted = self.adjust_for_regime(base_params, regime)

        adjustments_made = []
        for key in adjusted:
            if key not in base_params:
                adjustments_made.append(f"Added: {key}={adjusted[key]}")
            elif adjusted[key] != base_params[key]:
                adjustments_made.append(
                    f"Changed: {key}: {base_params[key]} -> {adjusted[key]}"
                )

        return RegimeParamAdjustment(
            original_params=base_params,
            adjusted_params=adjusted,
            regime=regime,
            adjustments_made=adjustments_made,
        )

    def get_all_regimes(self) -> list[str]:
        """利用可能なレジーム一覧を取得"""
        return list(self.regime_params.keys())

    def get_signal_types_for_regime(self, regime: str) -> list[str]:
        """指定レジームで定義されているシグナルタイプ一覧を取得"""
        regime_data = self.regime_params.get(regime, {})
        return list(regime_data.keys())

    def get_lookback_range(
        self,
        regime: str,
        signal_type: str,
    ) -> tuple[int, int] | None:
        """ルックバック期間の範囲を取得

        Args:
            regime: 市場レジーム
            signal_type: シグナルタイプ

        Returns:
            (min_lookback, max_lookback) または None
        """
        params = self.get_params(regime, signal_type)
        lookback = params.get("lookback", params.get("period"))

        if lookback is None:
            return None

        if isinstance(lookback, list):
            return (min(lookback), max(lookback))
        else:
            return (lookback, lookback)


# =============================================================================
# 便利関数
# =============================================================================

# デフォルトインスタンス
_default_adapter = RegimeAdaptiveParams()


def get_regime_params(
    regime: str,
    signal_type: str,
) -> dict[str, Any]:
    """レジーム別シグナルパラメータを取得（便利関数）

    Args:
        regime: 市場レジーム（bull_trend, bear_market, range_bound, high_vol等）
        signal_type: シグナルタイプ（momentum, bollinger, rsi, zscore, trend等）

    Returns:
        パラメータ辞書

    Example:
        params = get_regime_params("bull_trend", "momentum")
        # {'lookback': [40, 60, 90], 'scale': 3.0, 'description': '...'}
    """
    return _default_adapter.get_params(regime, signal_type)


def adjust_signal_params(
    params: dict[str, Any],
    regime: str,
) -> dict[str, Any]:
    """シグナルパラメータをレジームに応じて調整（便利関数）

    Args:
        params: 基本パラメータ（"type"キー必須）
        regime: 適用するレジーム

    Returns:
        調整後のパラメータ辞書

    Example:
        base = {"type": "momentum", "lookback": 20, "scale": 5.0}
        adjusted = adjust_signal_params(base, "bear_market")
        # {'type': 'momentum', 'lookback': 10, 'scale': 8.0}
    """
    return _default_adapter.adjust_for_regime(params, regime)


def get_param_set(
    regime: str,
    signal_type: str,
) -> SignalParamSet:
    """パラメータセットを取得（便利関数）

    Args:
        regime: 市場レジーム
        signal_type: シグナルタイプ

    Returns:
        SignalParamSet
    """
    return _default_adapter.get_param_set(regime, signal_type)


def list_available_regimes() -> list[str]:
    """利用可能なレジーム一覧を取得"""
    return _default_adapter.get_all_regimes()


def list_signal_types(regime: str = "default") -> list[str]:
    """利用可能なシグナルタイプ一覧を取得"""
    return _default_adapter.get_signal_types_for_regime(regime)


def create_regime_adaptive_params(
    custom_params: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> RegimeAdaptiveParams:
    """RegimeAdaptiveParamsインスタンスを作成（ファクトリ関数）

    Args:
        custom_params: カスタムレジームパラメータ

    Returns:
        RegimeAdaptiveParams
    """
    return RegimeAdaptiveParams(regime_params=custom_params)
