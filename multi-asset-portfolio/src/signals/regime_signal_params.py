"""
Regime Signal Parameters - レジーム別シグナルパラメータ

レジームに応じてシグナルパラメータを動的調整する。
フラットな構造でパラメータを定義し、遷移期のパラメータ補間もサポート。

主要コンポーネント:
- REGIME_SIGNAL_PARAMS: レジーム別パラメータ定義（フラット構造）
- RegimeSignalParamSelector: パラメータ選択・補間クラス
- apply_regime_params: 既存設定へのレジーム適用

設計根拠:
- bull_trend: 長期ルックバック、低スケール（ノイズ抑制）、広いボリンジャーバンド
- bear_market: 短期ルックバック、高スケール（敏感反応）、狭いバンド
- high_vol: 中期ルックバック、低スケール、広いバンド（ノイズ耐性）
- low_vol: 中期ルックバック、高スケール（小さな動きを捉える）
- neutral: バランス設定

使用例:
    from src.signals.regime_signal_params import (
        RegimeSignalParamSelector,
        REGIME_SIGNAL_PARAMS,
        apply_regime_params,
    )

    # パラメータ取得
    selector = RegimeSignalParamSelector()
    params = selector.get_params("bull_trend")
    print(f"Bull trend params: {params}")

    # 遷移期のパラメータ補間
    transition_params = selector.interpolate_params("bull_trend", "bear_market", weight=0.7)

    # 既存設定へのレジーム適用
    base_config = {"momentum_lookbacks": [20, 40], "rsi_period": 14}
    adjusted = apply_regime_params(base_config, "high_vol")
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# レジーム定義
# =============================================================================

class SignalRegime(str, Enum):
    """シグナルパラメータ用レジーム"""
    BULL_TREND = "bull_trend"
    BEAR_MARKET = "bear_market"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


# =============================================================================
# レジーム別シグナルパラメータ定義（フラット構造）
# =============================================================================

REGIME_SIGNAL_PARAMS: Dict[str, Dict[str, Any]] = {
    # ========================================
    # Bull Trend: 長期トレンドを追う
    # 長いルックバック、低いスケール（ノイズ抑制）
    # 広いバンド幅で偽シグナルを抑制
    # ========================================
    "bull_trend": {
        "momentum_lookbacks": [40, 60, 90],
        "momentum_scale": 3.0,
        "bollinger_period": 30,
        "bollinger_std": 2.5,
        "rsi_period": 21,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
    },

    # ========================================
    # Bear Market: 素早い反応
    # 短いルックバック、高いスケール（敏感）
    # 狭いバンドで早期シグナル
    # ========================================
    "bear_market": {
        "momentum_lookbacks": [10, 20, 40],
        "momentum_scale": 8.0,
        "bollinger_period": 14,
        "bollinger_std": 2.0,
        "rsi_period": 10,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
    },

    # ========================================
    # High Volatility: ノイズ耐性重視
    # 中期ルックバック、低いスケール（ノイズ抑制）
    # 広いバンドでボラティリティ対応
    # ========================================
    "high_vol": {
        "momentum_lookbacks": [20, 40],
        "momentum_scale": 2.0,
        "bollinger_period": 20,
        "bollinger_std": 3.0,
        "rsi_period": 14,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
    },

    # ========================================
    # Low Volatility: 追従重視
    # 中期ルックバック、高いスケール（小さな動きを捉える）
    # 狭めのバンドで敏感に反応
    # ========================================
    "low_vol": {
        "momentum_lookbacks": [20, 40, 60],
        "momentum_scale": 6.0,
        "bollinger_period": 25,
        "bollinger_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    },

    # ========================================
    # Neutral: バランス設定
    # デフォルトとして使用
    # ========================================
    "neutral": {
        "momentum_lookbacks": [20, 40, 60],
        "momentum_scale": 4.0,
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    },
}


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class RegimeParams:
    """レジームパラメータ結果

    Attributes:
        regime: レジーム名
        params: パラメータ辞書
        is_interpolated: 補間されたものか
        source_regimes: 補間元のレジーム（補間時のみ）
        interpolation_weight: 補間重み（補間時のみ）
    """
    regime: str
    params: Dict[str, Any]
    is_interpolated: bool = False
    source_regimes: Optional[List[str]] = None
    interpolation_weight: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "regime": self.regime,
            "params": self.params,
            "is_interpolated": self.is_interpolated,
        }
        if self.is_interpolated:
            result["source_regimes"] = self.source_regimes
            result["interpolation_weight"] = self.interpolation_weight
        return result

    def get_momentum_lookbacks(self) -> List[int]:
        """モメンタムルックバック期間を取得"""
        return self.params.get("momentum_lookbacks", [20, 40, 60])

    def get_bollinger_params(self) -> Dict[str, Any]:
        """ボリンジャーパラメータを取得"""
        return {
            "period": self.params.get("bollinger_period", 20),
            "std": self.params.get("bollinger_std", 2.0),
        }

    def get_rsi_params(self) -> Dict[str, Any]:
        """RSIパラメータを取得"""
        return {
            "period": self.params.get("rsi_period", 14),
            "oversold": self.params.get("rsi_oversold", 30),
            "overbought": self.params.get("rsi_overbought", 70),
        }


@dataclass
class AppliedParams:
    """レジーム適用後のパラメータ

    Attributes:
        original: 元のパラメータ
        applied: 適用後のパラメータ
        regime: 適用されたレジーム
        changes: 変更されたキーのリスト
    """
    original: Dict[str, Any]
    applied: Dict[str, Any]
    regime: str
    changes: List[str] = field(default_factory=list)


# =============================================================================
# RegimeSignalParamSelector クラス
# =============================================================================

class RegimeSignalParamSelector:
    """
    レジーム別シグナルパラメータセレクター

    レジームに応じたパラメータを取得し、遷移期のパラメータ補間もサポート。

    Usage:
        selector = RegimeSignalParamSelector()

        # パラメータ取得
        params = selector.get_params("bull_trend")

        # 補間（bull_trend 70% + bear_market 30%）
        transition = selector.interpolate_params("bull_trend", "bear_market", weight=0.7)

        # 全レジーム一覧
        regimes = selector.list_regimes()
    """

    def __init__(
        self,
        regime_params: Optional[Dict[str, Dict[str, Any]]] = None,
        default_regime: str = "neutral",
    ) -> None:
        """
        初期化

        Args:
            regime_params: カスタムレジームパラメータ（Noneでデフォルト使用）
            default_regime: デフォルトレジーム
        """
        self.regime_params = regime_params or REGIME_SIGNAL_PARAMS.copy()
        self.default_regime = default_regime

        logger.info(
            f"RegimeSignalParamSelector initialized with {len(self.regime_params)} regimes"
        )

    def get_params(self, regime: str) -> Dict[str, Any]:
        """
        レジーム別パラメータを取得

        Args:
            regime: レジーム名（bull_trend, bear_market, high_vol, low_vol, neutral）

        Returns:
            パラメータ辞書（見つからない場合はデフォルトレジームのパラメータ）
        """
        params = self.regime_params.get(regime)

        if params is None:
            logger.warning(
                f"Regime '{regime}' not found, using default '{self.default_regime}'"
            )
            params = self.regime_params.get(self.default_regime, {})

        return deepcopy(params)

    def get_regime_params(self, regime: str) -> RegimeParams:
        """
        RegimeParamsオブジェクトとして取得

        Args:
            regime: レジーム名

        Returns:
            RegimeParams
        """
        params = self.get_params(regime)
        return RegimeParams(
            regime=regime,
            params=params,
            is_interpolated=False,
        )

    def interpolate_params(
        self,
        regime1: str,
        regime2: str,
        weight: float,
    ) -> Dict[str, Any]:
        """
        2つのレジーム間でパラメータを補間

        遷移期において、徐々にパラメータを変化させるために使用。

        Args:
            regime1: 第1レジーム
            regime2: 第2レジーム
            weight: regime1の重み（0.0〜1.0）。regime2の重みは1-weight。

        Returns:
            補間されたパラメータ辞書

        Example:
            # bull_trend 70% + bear_market 30%
            params = selector.interpolate_params("bull_trend", "bear_market", weight=0.7)
        """
        weight = max(0.0, min(1.0, weight))  # clamp to [0, 1]

        params1 = self.get_params(regime1)
        params2 = self.get_params(regime2)

        interpolated = {}

        # 両方に存在するキーを補間
        all_keys = set(params1.keys()) | set(params2.keys())

        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)

            if val1 is None:
                interpolated[key] = val2
            elif val2 is None:
                interpolated[key] = val1
            elif isinstance(val1, list) and isinstance(val2, list):
                # リストの補間: 各要素を補間
                interpolated[key] = self._interpolate_lists(val1, val2, weight)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数値の補間
                interpolated[key] = self._interpolate_numeric(val1, val2, weight)
            else:
                # 補間不可能な場合は重みの大きい方を採用
                interpolated[key] = val1 if weight >= 0.5 else val2

        return interpolated

    def interpolate_regime_params(
        self,
        regime1: str,
        regime2: str,
        weight: float,
    ) -> RegimeParams:
        """
        RegimeParamsオブジェクトとして補間結果を取得

        Args:
            regime1: 第1レジーム
            regime2: 第2レジーム
            weight: regime1の重み

        Returns:
            RegimeParams（is_interpolated=True）
        """
        params = self.interpolate_params(regime1, regime2, weight)
        return RegimeParams(
            regime=f"{regime1}_{regime2}_interpolated",
            params=params,
            is_interpolated=True,
            source_regimes=[regime1, regime2],
            interpolation_weight=weight,
        )

    def _interpolate_numeric(
        self,
        val1: Union[int, float],
        val2: Union[int, float],
        weight: float,
    ) -> Union[int, float]:
        """数値を線形補間"""
        result = val1 * weight + val2 * (1 - weight)
        # 元が両方整数なら整数で返す
        if isinstance(val1, int) and isinstance(val2, int):
            return int(round(result))
        return result

    def _interpolate_lists(
        self,
        list1: List[Any],
        list2: List[Any],
        weight: float,
    ) -> List[Any]:
        """リストを要素ごとに補間"""
        max_len = max(len(list1), len(list2))
        result = []

        for i in range(max_len):
            val1 = list1[i] if i < len(list1) else list1[-1] if list1 else 0
            val2 = list2[i] if i < len(list2) else list2[-1] if list2 else 0

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                result.append(self._interpolate_numeric(val1, val2, weight))
            else:
                result.append(val1 if weight >= 0.5 else val2)

        return result

    def list_regimes(self) -> List[str]:
        """利用可能なレジーム一覧を取得"""
        return list(self.regime_params.keys())

    def get_param_keys(self, regime: Optional[str] = None) -> List[str]:
        """
        パラメータキー一覧を取得

        Args:
            regime: レジーム名（Noneの場合は全レジームの共通キー）

        Returns:
            パラメータキーのリスト
        """
        if regime:
            params = self.get_params(regime)
            return list(params.keys())
        else:
            # 全レジームの共通キーを取得
            all_keys = None
            for params in self.regime_params.values():
                if all_keys is None:
                    all_keys = set(params.keys())
                else:
                    all_keys &= set(params.keys())
            return list(all_keys or [])

    def compare_regimes(
        self,
        regime1: str,
        regime2: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        2つのレジームのパラメータを比較

        Args:
            regime1: 第1レジーム
            regime2: 第2レジーム

        Returns:
            {key: {"regime1": val1, "regime2": val2, "diff": diff}, ...}
        """
        params1 = self.get_params(regime1)
        params2 = self.get_params(regime2)

        comparison = {}
        all_keys = set(params1.keys()) | set(params2.keys())

        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)

            entry = {
                regime1: val1,
                regime2: val2,
            }

            # 差分計算（数値の場合）
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                entry["diff"] = val2 - val1
                entry["diff_pct"] = (
                    (val2 - val1) / val1 * 100 if val1 != 0 else float('inf')
                )

            comparison[key] = entry

        return comparison


# =============================================================================
# 便利関数
# =============================================================================

# デフォルトセレクター
_default_selector = RegimeSignalParamSelector()


def get_regime_signal_params(regime: str) -> Dict[str, Any]:
    """
    レジーム別シグナルパラメータを取得（便利関数）

    Args:
        regime: レジーム名

    Returns:
        パラメータ辞書
    """
    return _default_selector.get_params(regime)


def interpolate_regime_params(
    regime1: str,
    regime2: str,
    weight: float,
) -> Dict[str, Any]:
    """
    2つのレジーム間でパラメータを補間（便利関数）

    Args:
        regime1: 第1レジーム
        regime2: 第2レジーム
        weight: regime1の重み

    Returns:
        補間されたパラメータ辞書
    """
    return _default_selector.interpolate_params(regime1, regime2, weight)


def apply_regime_params(
    signals_config: Dict[str, Any],
    regime: str,
) -> Dict[str, Any]:
    """
    既存のシグナル設定をレジームに応じて調整

    指定されたレジームのパラメータで既存設定をオーバーライドする。
    既存設定に存在するキーのみ更新する。

    Args:
        signals_config: 既存のシグナル設定
        regime: 適用するレジーム

    Returns:
        調整後のシグナル設定

    Example:
        base_config = {
            "momentum_lookbacks": [20, 40],
            "rsi_period": 14,
            "custom_param": "keep_this",
        }
        adjusted = apply_regime_params(base_config, "high_vol")
        # momentum_lookbacks と rsi_period は high_vol の値で上書き
        # custom_param はそのまま維持
    """
    regime_params = _default_selector.get_params(regime)
    adjusted = deepcopy(signals_config)

    for key in signals_config:
        if key in regime_params:
            adjusted[key] = regime_params[key]
            logger.debug(f"Applied regime param: {key} = {regime_params[key]}")

    return adjusted


def apply_regime_params_full(
    signals_config: Dict[str, Any],
    regime: str,
) -> AppliedParams:
    """
    既存設定へのレジーム適用（詳細結果付き）

    Args:
        signals_config: 既存のシグナル設定
        regime: 適用するレジーム

    Returns:
        AppliedParams（変更内容の詳細付き）
    """
    regime_params = _default_selector.get_params(regime)
    adjusted = deepcopy(signals_config)
    changes = []

    for key in signals_config:
        if key in regime_params:
            old_val = signals_config[key]
            new_val = regime_params[key]
            if old_val != new_val:
                adjusted[key] = new_val
                changes.append(key)

    return AppliedParams(
        original=signals_config,
        applied=adjusted,
        regime=regime,
        changes=changes,
    )


def list_signal_regimes() -> List[str]:
    """利用可能なレジーム一覧を取得"""
    return _default_selector.list_regimes()


def create_regime_signal_param_selector(
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> RegimeSignalParamSelector:
    """
    RegimeSignalParamSelectorインスタンスを作成（ファクトリ関数）

    Args:
        custom_params: カスタムレジームパラメータ

    Returns:
        RegimeSignalParamSelector
    """
    return RegimeSignalParamSelector(regime_params=custom_params)
