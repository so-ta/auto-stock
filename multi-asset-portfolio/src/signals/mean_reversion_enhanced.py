"""
Enhanced Mean Reversion Strategy - 強化版ミーンリバージョン戦略

Z-Scoreベースの逆張りシグナル生成。レジームフィルターで
トレンド相場では抑制、レンジ相場で強化。

Usage:
    from src.signals.mean_reversion_enhanced import EnhancedMeanReversion

    mr = EnhancedMeanReversion(config={
        'z_score_threshold': 2.0,
        'lookback_short': 5,
        'lookback_long': 20,
    })

    signals = mr.generate_signal(price_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MeanReversionConfig:
    """ミーンリバージョン設定"""

    # Z-Score閾値（この値以上で逆張りシグナル）
    z_score_threshold: float = 2.0

    # ルックバック期間
    lookback_short: int = 5   # 短期（確認用）
    lookback_long: int = 20   # 長期（平均計算用）

    # 確認日数（連続でシグナル出る必要あり）
    confirmation_days: int = 2

    # レジームフィルター
    regime_filter: bool = True

    # レジーム判定パラメータ
    trend_threshold: float = 0.02  # 2%以上動いたらトレンド
    trend_lookback: int = 10       # トレンド判定期間

    # シグナル強度調整
    signal_decay: float = 0.9      # シグナル減衰率
    max_signal: float = 1.0        # 最大シグナル値

    # ボリンジャーバンド連携
    use_bollinger: bool = True
    bollinger_period: int = 20
    bollinger_std: float = 2.0


class EnhancedMeanReversion:
    """
    強化版ミーンリバージョン戦略

    Z-Scoreベースの逆張りシグナルを生成。
    レジームフィルターでトレンド相場では抑制し、
    レンジ相場で強化する。

    Attributes:
        config: MeanReversionConfig設定
    """

    def __init__(self, config: dict | MeanReversionConfig | None = None):
        """
        初期化

        Args:
            config: 設定辞書またはMeanReversionConfig
        """
        if config is None:
            self.config = MeanReversionConfig()
        elif isinstance(config, dict):
            self.config = MeanReversionConfig(**config)
        else:
            self.config = config

    def calculate_z_score(self, prices: pd.Series) -> pd.Series:
        """
        Z-Scoreを計算

        Args:
            prices: 価格シリーズ

        Returns:
            Z-Scoreシリーズ
        """
        rolling_mean = prices.rolling(window=self.config.lookback_long).mean()
        rolling_std = prices.rolling(window=self.config.lookback_long).std()

        # ゼロ除算防止
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (prices - rolling_mean) / rolling_std
        return z_score

    def calculate_bollinger_position(self, prices: pd.Series) -> pd.Series:
        """
        ボリンジャーバンド位置を計算（0-1の範囲）

        Args:
            prices: 価格シリーズ

        Returns:
            バンド内位置（0=下限、1=上限）
        """
        rolling_mean = prices.rolling(window=self.config.bollinger_period).mean()
        rolling_std = prices.rolling(window=self.config.bollinger_period).std()

        upper_band = rolling_mean + self.config.bollinger_std * rolling_std
        lower_band = rolling_mean - self.config.bollinger_std * rolling_std

        band_width = upper_band - lower_band
        band_width = band_width.replace(0, np.nan)

        position = (prices - lower_band) / band_width
        return position.clip(0, 1)

    def detect_regime(self, prices: pd.Series) -> pd.Series:
        """
        レジーム（トレンド/レンジ）を検出

        Args:
            prices: 価格シリーズ

        Returns:
            レジームシリーズ（1=トレンド、0=レンジ）
        """
        # 期間リターンでトレンド判定
        returns = prices.pct_change(self.config.trend_lookback)

        # 絶対リターンが閾値以上ならトレンド
        is_trending = returns.abs() >= self.config.trend_threshold

        return is_trending.astype(int)

    def detect_regime_advanced(self, prices: pd.Series) -> pd.Series:
        """
        高度なレジーム検出（ADX風）

        Args:
            prices: 価格シリーズ

        Returns:
            レジーム強度（0-1、高いほどトレンド）
        """
        # 方向性指標
        lookback = self.config.trend_lookback

        # 上昇・下降の強さ
        high_change = prices.diff()
        low_change = -prices.diff()

        plus_dm = high_change.where(high_change > low_change, 0).where(high_change > 0, 0)
        minus_dm = low_change.where(low_change > high_change, 0).where(low_change > 0, 0)

        # 平滑化
        plus_di = plus_dm.rolling(lookback).mean()
        minus_di = minus_dm.rolling(lookback).mean()

        # ADX風指標
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / di_sum

        adx = dx.rolling(lookback).mean()

        # 0-1に正規化
        return adx.clip(0, 1)

    def generate_raw_signal(self, prices: pd.Series) -> pd.Series:
        """
        生のミーンリバージョンシグナルを生成

        Args:
            prices: 価格シリーズ

        Returns:
            シグナルシリーズ（-1=売り、0=中立、+1=買い）
        """
        z_score = self.calculate_z_score(prices)
        threshold = self.config.z_score_threshold

        # Z-Scoreが閾値を超えたら逆張り
        # Z > threshold: 売りシグナル（上がりすぎ）
        # Z < -threshold: 買いシグナル（下がりすぎ）
        signal = pd.Series(0.0, index=prices.index)

        signal = signal.where(z_score <= threshold, -1.0)   # 売り
        signal = signal.where(z_score >= -threshold, 1.0)   # 買い

        return signal

    def apply_confirmation(self, signal: pd.Series) -> pd.Series:
        """
        確認日数フィルターを適用

        Args:
            signal: 生のシグナル

        Returns:
            確認済みシグナル
        """
        if self.config.confirmation_days <= 1:
            return signal

        # 連続で同じシグナルが出た場合のみ有効
        confirmed = signal.copy()
        days = self.config.confirmation_days

        for i in range(days - 1):
            shifted = signal.shift(i + 1)
            # 符号が同じかチェック
            same_sign = (signal * shifted) > 0
            confirmed = confirmed.where(same_sign, 0)

        return confirmed

    def apply_regime_filter(
        self,
        signal: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        """
        レジームフィルターを適用

        Args:
            signal: シグナル
            prices: 価格シリーズ

        Returns:
            フィルター適用後シグナル
        """
        if not self.config.regime_filter:
            return signal

        # レジーム検出
        regime_strength = self.detect_regime_advanced(prices)

        # トレンド相場ではシグナルを弱める
        # レンジ相場（regime_strength低い）ではシグナルを強める
        adjustment = 1 - regime_strength  # 0-1、レンジで高い

        return signal * adjustment

    def generate_signal(
        self,
        prices: pd.DataFrame | pd.Series,
        ticker: str | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        ミーンリバージョンシグナルを生成

        Args:
            prices: 価格データ（DataFrameの場合は各列に対して計算）
            ticker: 特定銘柄のみ計算する場合

        Returns:
            シグナル（-1～+1の範囲）
        """
        if isinstance(prices, pd.DataFrame):
            if ticker is not None:
                return self._generate_single_signal(prices[ticker])

            # 全銘柄に対して計算
            signals = pd.DataFrame(index=prices.index)
            for col in prices.columns:
                signals[col] = self._generate_single_signal(prices[col])
            return signals
        else:
            return self._generate_single_signal(prices)

    def _generate_single_signal(self, prices: pd.Series) -> pd.Series:
        """単一銘柄のシグナル生成"""
        # 生シグナル
        signal = self.generate_raw_signal(prices)

        # 確認フィルター
        signal = self.apply_confirmation(signal)

        # レジームフィルター
        signal = self.apply_regime_filter(signal, prices)

        # ボリンジャー連携（オプション）
        if self.config.use_bollinger:
            bb_pos = self.calculate_bollinger_position(prices)
            # バンド外にいるほどシグナル強化
            bb_extremity = (bb_pos - 0.5).abs() * 2  # 0-1
            signal = signal * (1 + bb_extremity * 0.5)

        # クリッピング
        return signal.clip(-self.config.max_signal, self.config.max_signal)

    def get_signal_strength(self, prices: pd.Series) -> pd.Series:
        """
        シグナル強度を取得（デバッグ用）

        Args:
            prices: 価格シリーズ

        Returns:
            強度情報のDataFrame
        """
        z_score = self.calculate_z_score(prices)
        regime = self.detect_regime_advanced(prices)
        bb_pos = self.calculate_bollinger_position(prices)
        signal = self.generate_signal(prices)

        return pd.DataFrame({
            'z_score': z_score,
            'regime_strength': regime,
            'bb_position': bb_pos,
            'signal': signal,
        })


class MomentumMeanReversionBlender:
    """
    モメンタムとミーンリバージョンのブレンダー

    レジームに応じて両戦略の重みを動的に調整。
    """

    def __init__(
        self,
        momentum_weight: float = 0.6,
        mean_reversion_weight: float = 0.4,
        regime_adaptive: bool = True,
    ):
        """
        初期化

        Args:
            momentum_weight: モメンタム基本重み
            mean_reversion_weight: ミーンリバージョン基本重み
            regime_adaptive: レジーム適応的な重み調整
        """
        self.base_momentum_weight = momentum_weight
        self.base_mr_weight = mean_reversion_weight
        self.regime_adaptive = regime_adaptive

        self.mr_strategy = EnhancedMeanReversion()

    def get_regime_weights(
        self,
        prices: pd.Series,
    ) -> tuple[float, float]:
        """
        レジームに応じた重みを取得

        Args:
            prices: 価格シリーズ

        Returns:
            (momentum_weight, mean_reversion_weight)のタプル
        """
        if not self.regime_adaptive:
            return self.base_momentum_weight, self.base_mr_weight

        # レジーム強度（トレンド度合い）
        regime_strength = self.mr_strategy.detect_regime_advanced(prices)
        current_regime = regime_strength.iloc[-1] if len(regime_strength) > 0 else 0.5

        # トレンド相場: モメンタム重視
        # レンジ相場: ミーンリバージョン重視
        momentum_weight = self.base_momentum_weight + (current_regime - 0.5) * 0.4
        mr_weight = self.base_mr_weight - (current_regime - 0.5) * 0.4

        # 正規化
        total = momentum_weight + mr_weight
        return momentum_weight / total, mr_weight / total

    def blend_signals(
        self,
        momentum_signal: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        """
        モメンタムとミーンリバージョンシグナルをブレンド

        Args:
            momentum_signal: モメンタムシグナル
            prices: 価格シリーズ（ミーンリバージョン計算用）

        Returns:
            ブレンドされたシグナル
        """
        # ミーンリバージョンシグナル生成
        mr_signal = self.mr_strategy.generate_signal(prices)

        # 重み取得
        mom_w, mr_w = self.get_regime_weights(prices)

        # ブレンド
        blended = momentum_signal * mom_w + mr_signal * mr_w

        return blended.clip(-1, 1)


# 便利関数
def calculate_mean_reversion_score(
    prices: pd.Series,
    lookback: int = 20,
    threshold: float = 2.0,
) -> float:
    """
    簡易ミーンリバージョンスコアを計算

    Args:
        prices: 価格シリーズ
        lookback: ルックバック期間
        threshold: Z-Score閾値

    Returns:
        スコア（-1～+1）
    """
    if len(prices) < lookback:
        return 0.0

    mean = prices.iloc[-lookback:].mean()
    std = prices.iloc[-lookback:].std()

    if std == 0:
        return 0.0

    z_score = (prices.iloc[-1] - mean) / std

    # 閾値超えで逆張りシグナル
    if z_score > threshold:
        return -min(1.0, (z_score - threshold) / threshold)
    elif z_score < -threshold:
        return min(1.0, (-z_score - threshold) / threshold)
    else:
        return 0.0
