"""
Signal Dynamic Parameters - シグナルパラメータの動的計算

各シグナルのパラメータを過去データに基づいて動的に計算する。
静的なパラメータではなく、市場環境に適応したパラメータを提供。

主要コンポーネント:
- MomentumDynamicParams: モメンタムシグナル用パラメータ
- BollingerDynamicParams: ボリンジャーバンド用パラメータ
- RSIDynamicParams: RSI用パラメータ
- ZScoreDynamicParams: Zスコア用パラメータ
- SignalParamsBundle: 全シグナルパラメータのバンドル

使用例:
    from src.signals.dynamic_params import (
        calculate_signal_params,
        SignalParamsBundle,
    )

    # 過去データからパラメータ計算
    params = calculate_signal_params(prices, lookback_days=252)
    print(f"Momentum tanh_scale: {params.momentum.tanh_scale:.4f}")
    print(f"Bollinger num_std: {params.bollinger.num_std:.2f}")
    print(f"RSI oversold: {params.rsi.oversold_level:.1f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ボラティリティレジーム
# =============================================================================

class VolatilityRegime(str, Enum):
    """ボラティリティレジーム"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


def detect_volatility_regime(
    returns: pd.Series,
    lookback_days: int = 60,
) -> VolatilityRegime:
    """ボラティリティレジームを検出

    Args:
        returns: リターン系列
        lookback_days: ルックバック期間

    Returns:
        VolatilityRegime
    """
    if len(returns) < lookback_days:
        return VolatilityRegime.NORMAL

    recent_vol = returns.tail(20).std() * np.sqrt(252)
    long_vol = returns.tail(lookback_days).std() * np.sqrt(252)

    if long_vol == 0:
        return VolatilityRegime.NORMAL

    vol_ratio = recent_vol / long_vol

    if vol_ratio < 0.7:
        return VolatilityRegime.LOW
    elif vol_ratio < 1.3:
        return VolatilityRegime.NORMAL
    elif vol_ratio < 2.0:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


# =============================================================================
# データクラス定義
# =============================================================================

@dataclass
class MomentumDynamicParams:
    """モメンタムシグナル用動的パラメータ

    Attributes:
        tanh_scale: tanh正規化のスケール係数
        lookback_period: ルックバック期間
        returns_std: リターンの標準偏差
        regime: 現在のボラティリティレジーム
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    tanh_scale: float
    lookback_period: int
    returns_std: float
    regime: str
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "tanh_scale": self.tanh_scale,
            "lookback_period": self.lookback_period,
            "returns_std": self.returns_std,
            "regime": self.regime,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BollingerDynamicParams:
    """ボリンジャーバンド用動的パラメータ

    Attributes:
        num_std: 標準偏差の倍数
        period: 計算期間
        band_hit_rate: バンドヒット率
        optimal_std: 最適な標準偏差倍数
        regime: 現在のボラティリティレジーム
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    num_std: float
    period: int
    band_hit_rate: float
    optimal_std: float
    regime: str
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "num_std": self.num_std,
            "period": self.period,
            "band_hit_rate": self.band_hit_rate,
            "optimal_std": self.optimal_std,
            "regime": self.regime,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RSIDynamicParams:
    """RSI用動的パラメータ

    Attributes:
        oversold_level: 売られ過ぎ水準
        overbought_level: 買われ過ぎ水準
        period: RSI計算期間
        rsi_mean: RSI平均値
        rsi_std: RSI標準偏差
        regime: 現在のボラティリティレジーム
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    oversold_level: float
    overbought_level: float
    period: int
    rsi_mean: float
    rsi_std: float
    regime: str
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "oversold_level": self.oversold_level,
            "overbought_level": self.overbought_level,
            "period": self.period,
            "rsi_mean": self.rsi_mean,
            "rsi_std": self.rsi_std,
            "regime": self.regime,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ZScoreDynamicParams:
    """Zスコア用動的パラメータ

    Attributes:
        entry_threshold: エントリー閾値（95%信頼区間）
        exit_threshold: エグジット閾値
        mean_reversion_period: 平均回帰期間
        half_life: 半減期
        zscore_std: Zスコアの標準偏差
        regime: 現在のボラティリティレジーム
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    entry_threshold: float
    exit_threshold: float
    mean_reversion_period: int
    half_life: float
    zscore_std: float
    regime: str
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "mean_reversion_period": self.mean_reversion_period,
            "half_life": self.half_life,
            "zscore_std": self.zscore_std,
            "regime": self.regime,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SignalParamsBundle:
    """全シグナルパラメータのバンドル

    Attributes:
        momentum: モメンタムパラメータ
        bollinger: ボリンジャーパラメータ
        rsi: RSIパラメータ
        zscore: Zスコアパラメータ
        regime: 現在のボラティリティレジーム
        computed_at: 計算日時
        metadata: 追加メタデータ
    """

    momentum: MomentumDynamicParams
    bollinger: BollingerDynamicParams
    rsi: RSIDynamicParams
    zscore: ZScoreDynamicParams
    regime: str
    computed_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "momentum": self.momentum.to_dict(),
            "bollinger": self.bollinger.to_dict(),
            "rsi": self.rsi.to_dict(),
            "zscore": self.zscore.to_dict(),
            "regime": self.regime,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# パラメータ計算クラス
# =============================================================================

class SignalDynamicParamsCalculator:
    """シグナル動的パラメータ計算クラス

    各シグナルのパラメータを過去データに基づいて動的に計算する。

    Usage:
        calculator = SignalDynamicParamsCalculator()
        params = calculator.calculate_all(prices, lookback_days=252)
        print(f"Momentum tanh_scale: {params.momentum.tanh_scale}")
    """

    def __init__(
        self,
        min_observations: int = 60,
    ) -> None:
        """初期化

        Args:
            min_observations: 最小観測数
        """
        self.min_observations = min_observations

    def calculate_all(
        self,
        prices: pd.DataFrame,
        lookback_days: int = 252,
    ) -> SignalParamsBundle:
        """全シグナルパラメータを計算

        Args:
            prices: 価格DataFrame（'close'列必須）
            lookback_days: ルックバック期間

        Returns:
            SignalParamsBundle
        """
        # リターン計算
        if "close" not in prices.columns:
            raise ValueError("prices must contain 'close' column")

        returns = prices["close"].pct_change().dropna()
        if len(returns) > lookback_days:
            returns = returns.tail(lookback_days)

        # ボラティリティレジーム検出
        regime = detect_volatility_regime(returns, lookback_days)

        # 各パラメータ計算
        momentum_params = self.calculate_momentum_params(returns, regime)
        bollinger_params = self.calculate_bollinger_params(prices, returns, regime)
        rsi_params = self.calculate_rsi_params(prices, returns, regime)
        zscore_params = self.calculate_zscore_params(prices, returns, regime)

        return SignalParamsBundle(
            momentum=momentum_params,
            bollinger=bollinger_params,
            rsi=rsi_params,
            zscore=zscore_params,
            regime=regime.value,
            metadata={
                "lookback_days": lookback_days,
                "observations": len(returns),
            },
        )

    def calculate_momentum_params(
        self,
        returns: pd.Series,
        regime: VolatilityRegime,
    ) -> MomentumDynamicParams:
        """モメンタムパラメータを計算

        Args:
            returns: リターン系列
            regime: ボラティリティレジーム

        Returns:
            MomentumDynamicParams
        """
        returns_std = float(returns.std())

        # tanh_scale: 3×標準偏差で±1に達するように設定
        if returns_std > 0:
            tanh_scale = 1.0 / (3.0 * returns_std)
        else:
            tanh_scale = 5.0  # デフォルト

        # ルックバック期間: レジームに応じて選択
        lookback_map = {
            VolatilityRegime.LOW: 40,
            VolatilityRegime.NORMAL: 20,
            VolatilityRegime.HIGH: 10,
            VolatilityRegime.EXTREME: 5,
        }
        lookback_period = lookback_map.get(regime, 20)

        return MomentumDynamicParams(
            tanh_scale=tanh_scale,
            lookback_period=lookback_period,
            returns_std=returns_std,
            regime=regime.value,
            metadata={
                "annualized_vol": returns_std * np.sqrt(252),
            },
        )

    def calculate_bollinger_params(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        regime: VolatilityRegime,
    ) -> BollingerDynamicParams:
        """ボリンジャーパラメータを計算

        Args:
            prices: 価格DataFrame
            returns: リターン系列
            regime: ボラティリティレジーム

        Returns:
            BollingerDynamicParams
        """
        close = prices["close"]

        # 期間: レジームに応じて選択
        period_map = {
            VolatilityRegime.LOW: 25,
            VolatilityRegime.NORMAL: 20,
            VolatilityRegime.HIGH: 12,
            VolatilityRegime.EXTREME: 10,
        }
        period = period_map.get(regime, 20)

        # バンドヒット率から最適な標準偏差倍数を計算
        optimal_std, band_hit_rate = self._optimize_bollinger_std(close, period)

        # 最終的な標準偏差倍数
        num_std = optimal_std

        return BollingerDynamicParams(
            num_std=num_std,
            period=period,
            band_hit_rate=band_hit_rate,
            optimal_std=optimal_std,
            regime=regime.value,
            metadata={
                "period_options": list(period_map.values()),
            },
        )

    def _optimize_bollinger_std(
        self,
        close: pd.Series,
        period: int,
        target_hit_rate: float = 0.05,
    ) -> tuple[float, float]:
        """ボリンジャーバンドの最適な標準偏差倍数を計算

        Args:
            close: 終値系列
            period: 計算期間
            target_hit_rate: 目標バンドヒット率

        Returns:
            (最適な標準偏差倍数, 実際のヒット率)
        """
        if len(close) < period + 10:
            return 2.0, 0.05

        # 移動平均と標準偏差
        sma = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        best_std = 2.0
        best_diff = float("inf")
        actual_hit_rate = 0.05

        # 1.5～3.0の範囲で最適な標準偏差倍数を探索
        for std_mult in np.arange(1.5, 3.1, 0.1):
            upper = sma + std_mult * rolling_std
            lower = sma - std_mult * rolling_std

            # バンドタッチ回数
            touches_upper = (close >= upper).sum()
            touches_lower = (close <= lower).sum()
            total_touches = touches_upper + touches_lower

            hit_rate = total_touches / len(close.dropna())
            diff = abs(hit_rate - target_hit_rate)

            if diff < best_diff:
                best_diff = diff
                best_std = std_mult
                actual_hit_rate = hit_rate

        return best_std, actual_hit_rate

    def calculate_rsi_params(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        regime: VolatilityRegime,
    ) -> RSIDynamicParams:
        """RSIパラメータを計算

        Args:
            prices: 価格DataFrame
            returns: リターン系列
            regime: ボラティリティレジーム

        Returns:
            RSIDynamicParams
        """
        close = prices["close"]

        # 期間: レジームに応じて選択
        period_map = {
            VolatilityRegime.LOW: 21,
            VolatilityRegime.NORMAL: 14,
            VolatilityRegime.HIGH: 9,
            VolatilityRegime.EXTREME: 7,
        }
        period = period_map.get(regime, 14)

        # RSI計算
        rsi = self._calculate_rsi(close, period)

        if len(rsi.dropna()) < self.min_observations:
            # データ不足時のデフォルト
            return RSIDynamicParams(
                oversold_level=30.0,
                overbought_level=70.0,
                period=period,
                rsi_mean=50.0,
                rsi_std=15.0,
                regime=regime.value,
                metadata={"warning": "insufficient_data"},
            )

        # RSI分布から閾値を計算
        rsi_clean = rsi.dropna()
        oversold_level = float(np.percentile(rsi_clean, 10))
        overbought_level = float(np.percentile(rsi_clean, 90))
        rsi_mean = float(rsi_clean.mean())
        rsi_std = float(rsi_clean.std())

        return RSIDynamicParams(
            oversold_level=oversold_level,
            overbought_level=overbought_level,
            period=period,
            rsi_mean=rsi_mean,
            rsi_std=rsi_std,
            regime=regime.value,
            metadata={
                "rsi_min": float(rsi_clean.min()),
                "rsi_max": float(rsi_clean.max()),
                "rsi_median": float(rsi_clean.median()),
            },
        )

    def _calculate_rsi(
        self,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """RSIを計算

        Args:
            close: 終値系列
            period: 計算期間

        Returns:
            RSI系列
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_zscore_params(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        regime: VolatilityRegime,
    ) -> ZScoreDynamicParams:
        """Zスコアパラメータを計算

        Args:
            prices: 価格DataFrame
            returns: リターン系列
            regime: ボラティリティレジーム

        Returns:
            ZScoreDynamicParams
        """
        close = prices["close"]

        # 半減期を計算（自己相関から推定）
        half_life = self._estimate_half_life(close)

        # 平均回帰期間
        mean_reversion_period = max(10, int(half_life * 2))

        # Zスコア計算
        zscore = (close - close.rolling(mean_reversion_period).mean()) / \
                 close.rolling(mean_reversion_period).std()

        zscore_clean = zscore.dropna()

        if len(zscore_clean) < self.min_observations:
            return ZScoreDynamicParams(
                entry_threshold=2.0,
                exit_threshold=0.5,
                mean_reversion_period=20,
                half_life=10.0,
                zscore_std=1.0,
                regime=regime.value,
                metadata={"warning": "insufficient_data"},
            )

        # 95%信頼区間からエントリー閾値を計算
        zscore_std = float(zscore_clean.std())
        entry_threshold = float(np.percentile(np.abs(zscore_clean), 95))

        # エグジット閾値
        exit_threshold = entry_threshold * 0.25

        return ZScoreDynamicParams(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            mean_reversion_period=mean_reversion_period,
            half_life=half_life,
            zscore_std=zscore_std,
            regime=regime.value,
            metadata={
                "zscore_min": float(zscore_clean.min()),
                "zscore_max": float(zscore_clean.max()),
            },
        )

    def _estimate_half_life(
        self,
        series: pd.Series,
        max_lag: int = 60,
    ) -> float:
        """半減期を推定

        Ornstein-Uhlenbeck過程を仮定し、
        自己相関から半減期を推定する。

        Args:
            series: 時系列データ
            max_lag: 最大ラグ

        Returns:
            半減期（日数）
        """
        if len(series) < max_lag + 10:
            return 20.0  # デフォルト

        # 対数価格の差分
        log_price = np.log(series.dropna())
        lag_price = log_price.shift(1)

        # 回帰: Δlog(P) = α + β * log(P_{t-1})
        y = (log_price - lag_price).dropna()
        x = lag_price.iloc[1:len(y) + 1]

        # 共通インデックス
        common_idx = y.index.intersection(x.index)
        y = y.loc[common_idx]
        x = x.loc[common_idx]

        if len(y) < 30:
            return 20.0

        # 簡易OLS
        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

        # 半減期 = -ln(2) / β
        if beta < 0:
            half_life = -np.log(2) / beta
        else:
            half_life = 20.0  # 平均回帰しない場合のデフォルト

        # 合理的な範囲にクリップ
        half_life = max(5.0, min(60.0, half_life))

        return float(half_life)


# =============================================================================
# ユーティリティ関数
# =============================================================================

def calculate_signal_params(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> SignalParamsBundle:
    """シグナルパラメータを計算（便利関数）

    Args:
        prices: 価格DataFrame（'close'列必須）
        lookback_days: ルックバック期間

    Returns:
        SignalParamsBundle
    """
    calculator = SignalDynamicParamsCalculator()
    return calculator.calculate_all(prices, lookback_days)


def get_momentum_params(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> MomentumDynamicParams:
    """モメンタムパラメータのみを取得

    Args:
        prices: 価格DataFrame
        lookback_days: ルックバック期間

    Returns:
        MomentumDynamicParams
    """
    params = calculate_signal_params(prices, lookback_days)
    return params.momentum


def get_bollinger_params(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> BollingerDynamicParams:
    """ボリンジャーパラメータのみを取得

    Args:
        prices: 価格DataFrame
        lookback_days: ルックバック期間

    Returns:
        BollingerDynamicParams
    """
    params = calculate_signal_params(prices, lookback_days)
    return params.bollinger


def get_rsi_params(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> RSIDynamicParams:
    """RSIパラメータのみを取得

    Args:
        prices: 価格DataFrame
        lookback_days: ルックバック期間

    Returns:
        RSIDynamicParams
    """
    params = calculate_signal_params(prices, lookback_days)
    return params.rsi


def get_zscore_params(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> ZScoreDynamicParams:
    """Zスコアパラメータのみを取得

    Args:
        prices: 価格DataFrame
        lookback_days: ルックバック期間

    Returns:
        ZScoreDynamicParams
    """
    params = calculate_signal_params(prices, lookback_days)
    return params.zscore
