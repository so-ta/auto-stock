"""
Dynamic Covariance Parameters Module - 共分散動的パラメータ計算

CovarianceConfigのパラメータを市場状況に応じて動的に計算する。

機能:
1. calculate_ewma_halflife: レジームに応じた半減期計算
2. calculate_correlation_adjustment: 過去データから相関調整係数を計算
3. calculate_regime_thresholds: ボラティリティ分布から閾値を計算
4. calculate_covariance_params: 統合関数
5. detect_market_regime: レジーム検出

設計根拠:
- 要求.md §8: 共分散推定の動的パラメータ化
- クライシス時は直近を重視（短い半減期）
- 通常時は安定重視（長い半減期）
- 過去データから相関調整係数を自動計算
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """マーケットレジーム"""

    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"


@dataclass
class DynamicCovarianceParams:
    """動的共分散パラメータ

    Attributes:
        ewma_halflife: EWMA半減期（日数）
        crisis_corr_adjustment: クライシス時の相関上方調整率
        low_vol_corr_adjustment: 低ボラ時の相関下方調整率
        crisis_vol_threshold: クライシスレジームのボラティリティ閾値（年率）
        low_vol_threshold: 低ボラレジームのボラティリティ閾値（年率）
        lookback_days: 計算に使用したルックバック日数
        calculated_at: 計算日時
    """

    ewma_halflife: int
    crisis_corr_adjustment: float
    low_vol_corr_adjustment: float
    crisis_vol_threshold: float
    low_vol_threshold: float
    lookback_days: int
    calculated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ewma_halflife": self.ewma_halflife,
            "crisis_corr_adjustment": self.crisis_corr_adjustment,
            "low_vol_corr_adjustment": self.low_vol_corr_adjustment,
            "crisis_vol_threshold": self.crisis_vol_threshold,
            "low_vol_threshold": self.low_vol_threshold,
            "lookback_days": self.lookback_days,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class RegimeThresholds:
    """レジーム閾値

    Attributes:
        crisis_vol_threshold: クライシス閾値（年率ボラティリティ）
        low_vol_threshold: 低ボラ閾値（年率ボラティリティ）
        high_vol_percentile: クライシス判定パーセンタイル
        low_vol_percentile: 低ボラ判定パーセンタイル
    """

    crisis_vol_threshold: float
    low_vol_threshold: float
    high_vol_percentile: float = 0.90
    low_vol_percentile: float = 0.25


@dataclass
class CorrelationAdjustments:
    """相関調整係数

    Attributes:
        crisis_adjustment: クライシス時の調整（正=相関上昇）
        low_vol_adjustment: 低ボラ時の調整（負=相関低下）
        crisis_avg_corr: クライシス時の平均相関
        normal_avg_corr: 通常時の平均相関
        low_vol_avg_corr: 低ボラ時の平均相関
    """

    crisis_adjustment: float
    low_vol_adjustment: float
    crisis_avg_corr: float
    normal_avg_corr: float
    low_vol_avg_corr: float


# =============================================================================
# EWMA Half-life Calculation
# =============================================================================
def calculate_ewma_halflife(
    market_regime: MarketRegime | str,
    crisis_halflife: int = 20,
    normal_halflife: int = 60,
    low_vol_halflife: int = 40,
    high_vol_halflife: int = 30,
) -> int:
    """レジームに応じたEWMA半減期を計算

    Args:
        market_regime: 現在の市場レジーム
        crisis_halflife: クライシス時の半減期（速い適応、直近重視）
        normal_halflife: 通常時の半減期（安定重視）
        low_vol_halflife: 低ボラ時の半減期（中間的）
        high_vol_halflife: 高ボラ時の半減期

    Returns:
        半減期（日数）

    設計根拠:
        - クライシス時: 20日（急激な変化に対応）
        - 通常時: 60日（ノイズを平滑化）
        - 低ボラ時: 40日（中間的）
        - 高ボラ時: 30日（やや速い適応）
    """
    if isinstance(market_regime, str):
        market_regime = MarketRegime(market_regime)

    halflife_map = {
        MarketRegime.CRISIS: crisis_halflife,
        MarketRegime.HIGH_VOL: high_vol_halflife,
        MarketRegime.NORMAL: normal_halflife,
        MarketRegime.LOW_VOL: low_vol_halflife,
    }

    halflife = halflife_map.get(market_regime, normal_halflife)

    logger.debug(
        "EWMA halflife for regime %s: %d days",
        market_regime.value,
        halflife,
    )

    return halflife


# =============================================================================
# Correlation Adjustment Calculation
# =============================================================================
def calculate_correlation_adjustment(
    returns: pd.DataFrame,
    volatility_history: pd.Series | None = None,
    lookback_days: int = 504,
    crisis_percentile: float = 0.90,
    low_vol_percentile: float = 0.25,
    rolling_window: int = 20,
) -> CorrelationAdjustments:
    """過去データから相関調整係数を計算

    Args:
        returns: 日次リターンのDataFrame (T x N)
        volatility_history: ボラティリティ履歴（省略時は計算）
        lookback_days: ルックバック日数
        crisis_percentile: クライシス判定パーセンタイル
        low_vol_percentile: 低ボラ判定パーセンタイル
        rolling_window: ボラティリティ計算のローリング窓

    Returns:
        CorrelationAdjustments

    計算方法:
        - crisis_corr_adjustment = クライシス時平均相関 - 通常時平均相関
        - low_vol_corr_adjustment = 低ボラ時平均相関 - 通常時平均相関
    """
    if len(returns) < lookback_days:
        lookback_days = len(returns)

    returns = returns.iloc[-lookback_days:]

    # ボラティリティ履歴の計算（ポートフォリオ等重み）
    if volatility_history is None:
        portfolio_returns = returns.mean(axis=1)
        volatility_history = (
            portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
        ).dropna()

    if len(volatility_history) < 50:
        logger.warning("Insufficient volatility history, using defaults")
        return CorrelationAdjustments(
            crisis_adjustment=0.30,
            low_vol_adjustment=-0.15,
            crisis_avg_corr=0.70,
            normal_avg_corr=0.40,
            low_vol_avg_corr=0.25,
        )

    # レジーム閾値
    crisis_threshold = volatility_history.quantile(crisis_percentile)
    low_vol_threshold = volatility_history.quantile(low_vol_percentile)

    # 各レジームの期間を特定
    crisis_mask = volatility_history >= crisis_threshold
    low_vol_mask = volatility_history <= low_vol_threshold
    normal_mask = ~crisis_mask & ~low_vol_mask

    # 各レジームでの平均相関を計算
    def calc_avg_correlation_for_mask(mask: pd.Series) -> float:
        """マスクされた期間の平均相関を計算"""
        mask_aligned = mask.reindex(returns.index).fillna(False)
        regime_returns = returns[mask_aligned]

        if len(regime_returns) < 20:
            return np.nan

        corr_matrix = regime_returns.corr()
        # 対角成分を除いた平均
        n = corr_matrix.shape[0]
        if n < 2:
            return np.nan

        upper_triangle = corr_matrix.values[np.triu_indices(n, k=1)]
        return float(np.nanmean(upper_triangle))

    crisis_avg_corr = calc_avg_correlation_for_mask(crisis_mask)
    normal_avg_corr = calc_avg_correlation_for_mask(normal_mask)
    low_vol_avg_corr = calc_avg_correlation_for_mask(low_vol_mask)

    # NaN処理
    if np.isnan(crisis_avg_corr):
        crisis_avg_corr = 0.70
    if np.isnan(normal_avg_corr):
        normal_avg_corr = 0.40
    if np.isnan(low_vol_avg_corr):
        low_vol_avg_corr = 0.25

    # 調整係数を計算
    # crisis_adjustment: クライシス時に相関がどれだけ上昇するか
    if normal_avg_corr > 0:
        crisis_adjustment = (crisis_avg_corr - normal_avg_corr) / normal_avg_corr
    else:
        crisis_adjustment = 0.30

    # low_vol_adjustment: 低ボラ時に相関がどれだけ低下するか
    if normal_avg_corr > 0:
        low_vol_adjustment = (low_vol_avg_corr - normal_avg_corr) / normal_avg_corr
    else:
        low_vol_adjustment = -0.15

    # クリップ
    crisis_adjustment = np.clip(crisis_adjustment, 0.0, 1.0)
    low_vol_adjustment = np.clip(low_vol_adjustment, -1.0, 0.0)

    result = CorrelationAdjustments(
        crisis_adjustment=float(crisis_adjustment),
        low_vol_adjustment=float(low_vol_adjustment),
        crisis_avg_corr=float(crisis_avg_corr),
        normal_avg_corr=float(normal_avg_corr),
        low_vol_avg_corr=float(low_vol_avg_corr),
    )

    logger.info(
        "Correlation adjustments: crisis=%.2f%%, low_vol=%.2f%% "
        "(crisis_corr=%.2f, normal_corr=%.2f, low_vol_corr=%.2f)",
        crisis_adjustment * 100,
        low_vol_adjustment * 100,
        crisis_avg_corr,
        normal_avg_corr,
        low_vol_avg_corr,
    )

    return result


# =============================================================================
# Regime Thresholds Calculation
# =============================================================================
def calculate_regime_thresholds(
    volatility_history: pd.Series,
    crisis_percentile: float = 0.90,
    low_vol_percentile: float = 0.25,
) -> RegimeThresholds:
    """過去ボラティリティ分布からレジーム閾値を計算

    Args:
        volatility_history: 年率ボラティリティの履歴
        crisis_percentile: クライシス判定パーセンタイル（デフォルト90%）
        low_vol_percentile: 低ボラ判定パーセンタイル（デフォルト25%）

    Returns:
        RegimeThresholds

    計算方法:
        - crisis_vol_threshold: 過去ボラ分布の90パーセンタイル
        - low_vol_threshold: 過去ボラ分布の25パーセンタイル
    """
    if len(volatility_history) < 20:
        logger.warning("Insufficient volatility history, using defaults")
        return RegimeThresholds(
            crisis_vol_threshold=0.25,
            low_vol_threshold=0.10,
            high_vol_percentile=crisis_percentile,
            low_vol_percentile=low_vol_percentile,
        )

    # 欠損値除去
    vol_clean = volatility_history.dropna()

    crisis_threshold = float(vol_clean.quantile(crisis_percentile))
    low_vol_threshold = float(vol_clean.quantile(low_vol_percentile))

    result = RegimeThresholds(
        crisis_vol_threshold=crisis_threshold,
        low_vol_threshold=low_vol_threshold,
        high_vol_percentile=crisis_percentile,
        low_vol_percentile=low_vol_percentile,
    )

    logger.info(
        "Regime thresholds: crisis=%.2f%% (p%.0f), low_vol=%.2f%% (p%.0f)",
        crisis_threshold * 100,
        crisis_percentile * 100,
        low_vol_threshold * 100,
        low_vol_percentile * 100,
    )

    return result


# =============================================================================
# Market Regime Detection
# =============================================================================
def detect_market_regime(
    current_vol: float,
    thresholds: RegimeThresholds,
) -> MarketRegime:
    """現在のボラティリティからレジームを検出

    Args:
        current_vol: 現在の年率ボラティリティ
        thresholds: レジーム閾値

    Returns:
        MarketRegime
    """
    if current_vol >= thresholds.crisis_vol_threshold:
        regime = MarketRegime.CRISIS
    elif current_vol >= thresholds.crisis_vol_threshold * 0.8:
        # 90%閾値の80%以上 = 高ボラ
        regime = MarketRegime.HIGH_VOL
    elif current_vol <= thresholds.low_vol_threshold:
        regime = MarketRegime.LOW_VOL
    else:
        regime = MarketRegime.NORMAL

    logger.debug(
        "Detected regime: %s (vol=%.2f%%, crisis_th=%.2f%%, low_th=%.2f%%)",
        regime.value,
        current_vol * 100,
        thresholds.crisis_vol_threshold * 100,
        thresholds.low_vol_threshold * 100,
    )

    return regime


def detect_market_regime_from_returns(
    returns: pd.DataFrame,
    lookback_days: int = 252,
    rolling_window: int = 20,
) -> tuple[MarketRegime, float, RegimeThresholds]:
    """リターンデータからレジームを検出

    Args:
        returns: 日次リターンのDataFrame
        lookback_days: ルックバック日数
        rolling_window: ボラティリティ計算のローリング窓

    Returns:
        (MarketRegime, current_vol, RegimeThresholds)
    """
    if len(returns) < rolling_window:
        return MarketRegime.NORMAL, 0.15, RegimeThresholds(0.25, 0.10)

    # ポートフォリオリターン（等重み）
    portfolio_returns = returns.mean(axis=1)

    # ボラティリティ履歴
    vol_history = (
        portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
    ).dropna()

    # 現在のボラティリティ
    current_vol = float(vol_history.iloc[-1]) if len(vol_history) > 0 else 0.15

    # 閾値計算
    lookback_vol = vol_history.iloc[-lookback_days:] if len(vol_history) > lookback_days else vol_history
    thresholds = calculate_regime_thresholds(lookback_vol)

    # レジーム検出
    regime = detect_market_regime(current_vol, thresholds)

    return regime, current_vol, thresholds


# =============================================================================
# Integrated Parameter Calculation
# =============================================================================
def calculate_covariance_params(
    returns: pd.DataFrame,
    lookback_days: int = 504,
    rolling_window: int = 20,
    crisis_percentile: float = 0.90,
    low_vol_percentile: float = 0.25,
) -> DynamicCovarianceParams:
    """統合関数: 動的共分散パラメータを計算

    Args:
        returns: 日次リターンのDataFrame (T x N)
        lookback_days: ルックバック日数（デフォルト2年）
        rolling_window: ボラティリティ計算のローリング窓
        crisis_percentile: クライシス判定パーセンタイル
        low_vol_percentile: 低ボラ判定パーセンタイル

    Returns:
        DynamicCovarianceParams

    使用例:
        params = calculate_covariance_params(returns, lookback_days=504)
        config = CovarianceConfig(
            ewma_halflife=params.ewma_halflife,
            crisis_corr_adjustment=params.crisis_corr_adjustment,
            low_vol_corr_adjustment=params.low_vol_corr_adjustment,
            ...
        )
    """
    effective_lookback = min(lookback_days, len(returns))

    if effective_lookback < 60:
        logger.warning(
            "Insufficient data for dynamic params: %d < 60. Using defaults.",
            effective_lookback,
        )
        return DynamicCovarianceParams(
            ewma_halflife=60,
            crisis_corr_adjustment=0.30,
            low_vol_corr_adjustment=-0.15,
            crisis_vol_threshold=0.25,
            low_vol_threshold=0.10,
            lookback_days=effective_lookback,
            calculated_at=datetime.now(),
        )

    # ポートフォリオリターン
    portfolio_returns = returns.mean(axis=1)

    # ボラティリティ履歴
    vol_history = (
        portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
    ).dropna()

    # 1. レジーム閾値を計算
    thresholds = calculate_regime_thresholds(
        vol_history.iloc[-lookback_days:] if len(vol_history) > lookback_days else vol_history,
        crisis_percentile=crisis_percentile,
        low_vol_percentile=low_vol_percentile,
    )

    # 2. 現在のレジームを検出
    current_vol = float(vol_history.iloc[-1]) if len(vol_history) > 0 else 0.15
    current_regime = detect_market_regime(current_vol, thresholds)

    # 3. EWMA半減期を計算
    ewma_halflife = calculate_ewma_halflife(current_regime)

    # 4. 相関調整係数を計算
    corr_adjustments = calculate_correlation_adjustment(
        returns.iloc[-lookback_days:] if len(returns) > lookback_days else returns,
        volatility_history=vol_history,
        lookback_days=lookback_days,
        crisis_percentile=crisis_percentile,
        low_vol_percentile=low_vol_percentile,
        rolling_window=rolling_window,
    )

    result = DynamicCovarianceParams(
        ewma_halflife=ewma_halflife,
        crisis_corr_adjustment=corr_adjustments.crisis_adjustment,
        low_vol_corr_adjustment=corr_adjustments.low_vol_adjustment,
        crisis_vol_threshold=thresholds.crisis_vol_threshold,
        low_vol_threshold=thresholds.low_vol_threshold,
        lookback_days=effective_lookback,
        calculated_at=datetime.now(),
    )

    logger.info(
        "Dynamic covariance params calculated: regime=%s, halflife=%d, "
        "crisis_adj=%.2f%%, low_vol_adj=%.2f%%",
        current_regime.value,
        ewma_halflife,
        corr_adjustments.crisis_adjustment * 100,
        corr_adjustments.low_vol_adjustment * 100,
    )

    return result


# =============================================================================
# Utility Functions
# =============================================================================
def create_covariance_config_from_params(
    params: DynamicCovarianceParams,
    method: str = "regime_conditional",
) -> dict[str, Any]:
    """DynamicCovarianceParamsからCovarianceConfig用の辞書を生成

    Args:
        params: 動的パラメータ
        method: 共分散推定手法

    Returns:
        CovarianceConfig初期化用の辞書
    """
    return {
        "method": method,
        "ewma_halflife": params.ewma_halflife,
        "crisis_corr_adjustment": params.crisis_corr_adjustment,
        "low_vol_corr_adjustment": params.low_vol_corr_adjustment,
        "crisis_vol_threshold": params.crisis_vol_threshold,
        "low_vol_threshold": params.low_vol_threshold,
    }


def update_covariance_config(
    returns: pd.DataFrame,
    lookback_days: int = 504,
) -> dict[str, Any]:
    """リターンデータから共分散設定を更新

    便利関数: calculate_covariance_paramsとcreate_covariance_config_from_paramsを統合

    Args:
        returns: 日次リターンのDataFrame
        lookback_days: ルックバック日数

    Returns:
        CovarianceConfig初期化用の辞書
    """
    params = calculate_covariance_params(returns, lookback_days=lookback_days)
    return create_covariance_config_from_params(params)
