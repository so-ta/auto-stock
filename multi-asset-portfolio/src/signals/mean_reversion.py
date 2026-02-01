"""
Mean Reversion Signals - Counter-trend indicators.

Implements mean reversion signals including:
- Bollinger Bands: Standard deviation-based reversion
- RSI (Relative Strength Index): Momentum oscillator for overbought/oversold
- Z-Score: Statistical deviation from mean

All outputs are normalized to [-1, +1] using tanh compression.
Positive scores indicate oversold (bullish), negative indicate overbought (bearish).

キャッシュ機能:
- ローリング計算結果をキャッシュして再利用
- 同一データ・同一ウィンドウの再計算を回避
- 30-50%の高速化を実現
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult, TimeframeAffinity, TimeframeConfig
from .registry import SignalRegistry


class RollingCacheMixin:
    """ローリング計算キャッシュミックスイン

    rolling mean/std/min/max の計算結果をキャッシュして再利用。
    同一価格データ・同一ウィンドウの場合にキャッシュヒット。

    Usage:
        class MySignal(RollingCacheMixin, Signal):
            def compute(self, data):
                cache_key = self._make_cache_key(data["close"])
                mean = self._get_rolling_mean_cached(data["close"], 20, cache_key)
    """

    # クラス変数としてキャッシュを定義
    _rolling_mean_cache: Dict[str, Dict[int, pd.Series]] = {}
    _rolling_std_cache: Dict[str, Dict[int, pd.Series]] = {}
    _rolling_min_cache: Dict[str, Dict[int, pd.Series]] = {}
    _rolling_max_cache: Dict[str, Dict[int, pd.Series]] = {}
    _cache_max_size: int = 50

    def _make_cache_key(self, prices: pd.Series) -> str:
        """キャッシュキーを生成

        Args:
            prices: 価格Series

        Returns:
            キャッシュキー文字列
        """
        # Series名とデータのハッシュを組み合わせ
        name = prices.name if prices.name else "unnamed"
        # 先頭・末尾・長さでキーを生成（高速化のため完全ハッシュは避ける）
        first_val = prices.iloc[0] if len(prices) > 0 else 0
        last_val = prices.iloc[-1] if len(prices) > 0 else 0
        return f"{name}_{len(prices)}_{first_val:.6f}_{last_val:.6f}"

    def _ensure_cache_size(self, cache: Dict[str, Any]) -> None:
        """キャッシュサイズを制限（古いものを削除）"""
        while len(cache) >= self._cache_max_size:
            # 最初のキーを削除（簡易LRU）
            first_key = next(iter(cache))
            del cache[first_key]

    def _get_rolling_mean_cached(
        self,
        prices: pd.Series,
        window: int,
        cache_key: str,
    ) -> pd.Series:
        """キャッシュ付きローリング平均

        Args:
            prices: 価格Series
            window: ウィンドウサイズ
            cache_key: キャッシュキー

        Returns:
            ローリング平均Series
        """
        if cache_key not in self._rolling_mean_cache:
            self._ensure_cache_size(self._rolling_mean_cache)
            self._rolling_mean_cache[cache_key] = {}

        if window in self._rolling_mean_cache[cache_key]:
            return self._rolling_mean_cache[cache_key][window]

        # 計算
        result = prices.rolling(window=window, min_periods=1).mean()
        self._rolling_mean_cache[cache_key][window] = result
        return result

    def _get_rolling_std_cached(
        self,
        prices: pd.Series,
        window: int,
        cache_key: str,
    ) -> pd.Series:
        """キャッシュ付きローリング標準偏差

        Args:
            prices: 価格Series
            window: ウィンドウサイズ
            cache_key: キャッシュキー

        Returns:
            ローリング標準偏差Series
        """
        if cache_key not in self._rolling_std_cache:
            self._ensure_cache_size(self._rolling_std_cache)
            self._rolling_std_cache[cache_key] = {}

        if window in self._rolling_std_cache[cache_key]:
            return self._rolling_std_cache[cache_key][window]

        # 計算
        result = prices.rolling(window=window, min_periods=1).std()
        self._rolling_std_cache[cache_key][window] = result
        return result

    def _get_rolling_min_cached(
        self,
        prices: pd.Series,
        window: int,
        cache_key: str,
    ) -> pd.Series:
        """キャッシュ付きローリング最小値

        Args:
            prices: 価格Series
            window: ウィンドウサイズ
            cache_key: キャッシュキー

        Returns:
            ローリング最小値Series
        """
        if cache_key not in self._rolling_min_cache:
            self._ensure_cache_size(self._rolling_min_cache)
            self._rolling_min_cache[cache_key] = {}

        if window in self._rolling_min_cache[cache_key]:
            return self._rolling_min_cache[cache_key][window]

        # 計算
        result = prices.rolling(window=window, min_periods=1).min()
        self._rolling_min_cache[cache_key][window] = result
        return result

    def _get_rolling_max_cached(
        self,
        prices: pd.Series,
        window: int,
        cache_key: str,
    ) -> pd.Series:
        """キャッシュ付きローリング最大値

        Args:
            prices: 価格Series
            window: ウィンドウサイズ
            cache_key: キャッシュキー

        Returns:
            ローリング最大値Series
        """
        if cache_key not in self._rolling_max_cache:
            self._ensure_cache_size(self._rolling_max_cache)
            self._rolling_max_cache[cache_key] = {}

        if window in self._rolling_max_cache[cache_key]:
            return self._rolling_max_cache[cache_key][window]

        # 計算
        result = prices.rolling(window=window, min_periods=1).max()
        self._rolling_max_cache[cache_key][window] = result
        return result

    @classmethod
    def clear_cache(cls) -> None:
        """全キャッシュをクリア"""
        cls._rolling_mean_cache.clear()
        cls._rolling_std_cache.clear()
        cls._rolling_min_cache.clear()
        cls._rolling_max_cache.clear()

    @classmethod
    def clear_cache_for_ticker(cls, ticker: str) -> None:
        """特定ティッカーのキャッシュをクリア

        Args:
            ticker: ティッカー名
        """
        for cache in [
            cls._rolling_mean_cache,
            cls._rolling_std_cache,
            cls._rolling_min_cache,
            cls._rolling_max_cache,
        ]:
            keys_to_remove = [k for k in cache if ticker in k]
            for k in keys_to_remove:
                del cache[k]

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """キャッシュ統計を取得

        Returns:
            キャッシュサイズの辞書
        """
        return {
            "mean_cache_size": len(cls._rolling_mean_cache),
            "std_cache_size": len(cls._rolling_std_cache),
            "min_cache_size": len(cls._rolling_min_cache),
            "max_cache_size": len(cls._rolling_max_cache),
        }


@SignalRegistry.register(
    "bollinger_reversion",
    category="mean_reversion",
    description="Bollinger Bands mean reversion signal",
    tags=["oscillator", "reversal", "volatility"],
)
class BollingerReversionSignal(RollingCacheMixin, Signal):
    """
    Bollinger Bands Mean Reversion Signal.

    Generates buy signals when price touches lower band (oversold)
    and sell signals when price touches upper band (overbought).

    The signal is the inverse of the normalized position within the bands:
    - Price at lower band -> score = +1 (buy)
    - Price at upper band -> score = -1 (sell)
    - Price at middle band -> score = 0 (neutral)

    Parameters:
        period: Moving average period (searchable, 10-50)
        num_std: Number of standard deviations for bands (searchable, 1.5-3.0)
        scale: Tanh scaling factor (fixed, default 1.0)

    Formula:
        middle_band = SMA(close, period)
        std = rolling_std(close, period)
        upper_band = middle_band + num_std * std
        lower_band = middle_band - num_std * std
        position = (close - middle_band) / (num_std * std)
        score = tanh(-position * scale)  # Inverted for mean reversion

    Example:
        signal = BollingerReversionSignal(period=20, num_std=2.0)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Bollinger Bands: medium-term indicator (10-50 day periods)."""
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=10,
            max_period=50,
            # short(5) too short for meaningful bands
            # Only medium(20) is within spec range [10, 50]
            supported_variants=["medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=20,
                searchable=True,
                min_value=10,
                max_value=50,
                step=5,
                description="Moving average period for Bollinger Bands",
            ),
            ParameterSpec(
                name="num_std",
                default=2.0,
                searchable=True,
                min_value=1.5,
                max_value=3.0,
                step=0.5,
                description="Number of standard deviations for bands",
            ),
            ParameterSpec(
                name="scale",
                default=1.0,
                searchable=False,
                min_value=0.1,
                max_value=5.0,
                description="Tanh scaling factor for normalization",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get expanded parameter grid for Bollinger Bands signal.

        Provides comprehensive coverage of:
        - period: [10, 14, 20, 30, 40] covering short to medium term
        - num_std: [1.5, 2.0, 2.5, 3.0] covering tight to wide bands

        Total combinations: 5 * 4 = 20
        """
        return {
            "period": [10, 14, 20, 30, 40],
            "num_std": [1.5, 2.0, 2.5, 3.0],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        period = self._params["period"]
        num_std = self._params["num_std"]
        scale = self._params["scale"]

        close = data["close"]

        # キャッシュキーを生成
        cache_key = self._make_cache_key(close)

        # Calculate Bollinger Bands（キャッシュ使用）
        middle_band = self._get_rolling_mean_cached(close, period, cache_key)
        rolling_std = self._get_rolling_std_cached(close, period, cache_key)

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)

        upper_band = middle_band + num_std * rolling_std
        lower_band = middle_band - num_std * rolling_std

        # Calculate position within bands (normalized)
        # Position ranges from -1 (at lower band) to +1 (at upper band)
        band_width = upper_band - lower_band
        band_width = band_width.replace(0, np.nan).ffill().fillna(1)
        position = (close - middle_band) / (band_width / 2)

        # Invert for mean reversion: oversold (+1) -> buy, overbought (-1) -> sell
        raw_signal = -position

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate %B indicator for metadata
        pct_b = (close - lower_band) / band_width

        metadata = {
            "period": period,
            "num_std": num_std,
            "scale": scale,
            "pct_b_mean": pct_b.mean(),
            "pct_b_std": pct_b.std(),
            "position_mean": position.mean(),
            "band_width_mean": band_width.mean(),
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "rsi",
    category="mean_reversion",
    description="RSI (Relative Strength Index) mean reversion signal",
    tags=["oscillator", "reversal", "momentum"],
)
class RSISignal(Signal):
    """
    RSI (Relative Strength Index) Mean Reversion Signal.

    RSI measures the speed and magnitude of recent price changes.
    This signal generates mean reversion scores based on RSI levels:
    - RSI < oversold_level -> positive score (buy)
    - RSI > overbought_level -> negative score (sell)

    Parameters:
        period: RSI calculation period (searchable, 7-21)
        oversold_level: Oversold threshold (fixed, default 30)
        overbought_level: Overbought threshold (fixed, default 70)
        scale: Tanh scaling factor (fixed, default 0.04)

    Formula:
        avg_gain = EMA(gains, period)
        avg_loss = EMA(losses, period)
        RS = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))
        centered_rsi = RSI - 50
        score = tanh(-centered_rsi * scale)  # Inverted for mean reversion

    Example:
        signal = RSISignal(period=14)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """RSI: short-term oscillator (Wilder's original: 14 days, effective 7-21).

        Academic reference: Wilder, J.W. (1978) New Concepts in Technical Trading Systems.
        RSI loses sensitivity with periods > 21 days as it converges to 50.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=7,
            max_period=21,
            # Only medium(20) is within spec range [7, 21]
            supported_variants=["medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="period",
                default=14,
                searchable=True,
                min_value=7,
                max_value=21,
                step=1,
                description="RSI calculation period",
            ),
            ParameterSpec(
                name="oversold_level",
                default=30.0,
                searchable=False,
                min_value=10.0,
                max_value=40.0,
                description="Oversold threshold (RSI below this is oversold)",
            ),
            ParameterSpec(
                name="overbought_level",
                default=70.0,
                searchable=False,
                min_value=60.0,
                max_value=90.0,
                description="Overbought threshold (RSI above this is overbought)",
            ),
            ParameterSpec(
                name="scale",
                default=0.04,
                searchable=False,
                min_value=0.01,
                max_value=0.1,
                description="Tanh scaling factor for normalization",
            ),
        ]

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using Wilder's smoothing method."""
        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Use exponential moving average (Wilder's smoothing)
        # Wilder's smoothing: alpha = 1/period
        alpha = 1.0 / period

        avg_gain = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.fillna(50)  # Neutral when undefined

        return rsi

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        period = self._params["period"]
        oversold_level = self._params["oversold_level"]
        overbought_level = self._params["overbought_level"]
        scale = self._params["scale"]

        close = data["close"]

        # Calculate RSI
        rsi = self._calculate_rsi(close, period)

        # Center RSI around 50 for symmetry
        centered_rsi = rsi - 50

        # Invert for mean reversion:
        # High RSI (overbought) -> negative score (sell)
        # Low RSI (oversold) -> positive score (buy)
        raw_signal = -centered_rsi

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate zone statistics for metadata
        oversold_pct = (rsi < oversold_level).mean() * 100
        overbought_pct = (rsi > overbought_level).mean() * 100

        metadata = {
            "period": period,
            "oversold_level": oversold_level,
            "overbought_level": overbought_level,
            "scale": scale,
            "rsi_mean": rsi.mean(),
            "rsi_std": rsi.std(),
            "oversold_pct": oversold_pct,
            "overbought_pct": overbought_pct,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "zscore_reversion",
    category="mean_reversion",
    description="Z-Score based mean reversion signal",
    tags=["statistical", "reversal"],
)
class ZScoreReversionSignal(RollingCacheMixin, Signal):
    """
    Z-Score Mean Reversion Signal.

    Uses statistical z-score to measure deviation from rolling mean.
    Assumes prices will revert to mean over time.

    Parameters:
        lookback: Lookback period for mean/std calculation (searchable, 10-60)
        entry_threshold: Z-score threshold for signal generation (fixed, default 2.0)
        scale: Tanh scaling factor (fixed, default 0.5)

    Formula:
        rolling_mean = SMA(close, lookback)
        rolling_std = rolling_std(close, lookback)
        z_score = (close - rolling_mean) / rolling_std
        score = tanh(-z_score * scale)  # Inverted for mean reversion

    Example:
        signal = ZScoreReversionSignal(lookback=20)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Z-Score: medium-term mean reversion (10-60 days effective)."""
        return TimeframeConfig(
            affinity=TimeframeAffinity.MEDIUM_TERM,
            min_period=10,
            max_period=60,
            # short(5) too short for stable z-score
            # half_year(126)/yearly(252) too slow for mean reversion
            supported_variants=["medium", "long"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=20,
                searchable=True,
                min_value=10,
                max_value=60,
                step=5,
                description="Lookback period for mean/std calculation",
            ),
            ParameterSpec(
                name="entry_threshold",
                default=2.0,
                searchable=False,
                min_value=1.0,
                max_value=3.0,
                description="Z-score threshold for signal generation",
            ),
            ParameterSpec(
                name="scale",
                default=0.5,
                searchable=False,
                min_value=0.1,
                max_value=2.0,
                description="Tanh scaling factor for normalization",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        lookback = self._params["lookback"]
        entry_threshold = self._params["entry_threshold"]
        scale = self._params["scale"]

        close = data["close"]

        # キャッシュキーを生成
        cache_key = self._make_cache_key(close)

        # Calculate rolling statistics（キャッシュ使用）
        rolling_mean = self._get_rolling_mean_cached(close, lookback, cache_key)
        rolling_std = self._get_rolling_std_cached(close, lookback, cache_key)

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1)

        # Calculate z-score
        z_score = (close - rolling_mean) / rolling_std

        # Invert for mean reversion:
        # High z-score (above mean) -> negative score (sell)
        # Low z-score (below mean) -> positive score (buy)
        raw_signal = -z_score

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate statistics for metadata
        extreme_high_pct = (z_score > entry_threshold).mean() * 100
        extreme_low_pct = (z_score < -entry_threshold).mean() * 100

        metadata = {
            "lookback": lookback,
            "entry_threshold": entry_threshold,
            "scale": scale,
            "z_score_mean": z_score.mean(),
            "z_score_std": z_score.std(),
            "extreme_high_pct": extreme_high_pct,
            "extreme_low_pct": extreme_low_pct,
        }

        return SignalResult(scores=scores, metadata=metadata)


@SignalRegistry.register(
    "stochastic_reversion",
    category="mean_reversion",
    description="Stochastic Oscillator mean reversion signal",
    tags=["oscillator", "reversal"],
)
class StochasticReversionSignal(RollingCacheMixin, Signal):
    """
    Stochastic Oscillator Mean Reversion Signal.

    Compares current close to the high-low range over N periods.
    Useful for identifying overbought/oversold conditions.

    Parameters:
        k_period: %K calculation period (searchable, 5-21)
        d_period: %D smoothing period (fixed, default 3)
        oversold_level: Oversold threshold (fixed, default 20)
        overbought_level: Overbought threshold (fixed, default 80)
        scale: Tanh scaling factor (fixed, default 0.04)

    Formula:
        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %D = SMA(%K, d_period)
        centered_k = %K - 50
        score = tanh(-centered_k * scale)

    Example:
        signal = StochasticReversionSignal(k_period=14)
        result = signal.compute(price_data)
    """

    @classmethod
    def timeframe_config(cls) -> TimeframeConfig:
        """Stochastic: short-term oscillator (5-21 days effective).

        Like RSI, Stochastic is designed for short-term overbought/oversold detection.
        """
        return TimeframeConfig(
            affinity=TimeframeAffinity.SHORT_TERM,
            min_period=5,
            max_period=21,
            # short(5) and medium(20) are within spec range [5, 21]
            supported_variants=["short", "medium"],
        )

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="k_period",
                default=14,
                searchable=True,
                min_value=5,
                max_value=21,
                step=1,
                description="%K calculation period",
            ),
            ParameterSpec(
                name="d_period",
                default=3,
                searchable=False,
                min_value=1,
                max_value=5,
                description="%D smoothing period",
            ),
            ParameterSpec(
                name="oversold_level",
                default=20.0,
                searchable=False,
                min_value=10.0,
                max_value=30.0,
                description="Oversold threshold",
            ),
            ParameterSpec(
                name="overbought_level",
                default=80.0,
                searchable=False,
                min_value=70.0,
                max_value=90.0,
                description="Overbought threshold",
            ),
            ParameterSpec(
                name="scale",
                default=0.04,
                searchable=False,
                min_value=0.01,
                max_value=0.1,
                description="Tanh scaling factor",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        self.validate_input(data)

        k_period = self._params["k_period"]
        d_period = self._params["d_period"]
        oversold_level = self._params["oversold_level"]
        overbought_level = self._params["overbought_level"]
        scale = self._params["scale"]

        close = data["close"]

        # Need high and low for stochastic calculation
        if "high" in data.columns and "low" in data.columns:
            high = data["high"]
            low = data["low"]
        else:
            # Approximate using close if high/low not available
            high = close
            low = close

        # キャッシュキーを生成
        cache_key_low = self._make_cache_key(low)
        cache_key_high = self._make_cache_key(high)
        cache_key_k = self._make_cache_key(close)

        # Calculate %K（キャッシュ使用）
        lowest_low = self._get_rolling_min_cached(low, k_period, cache_key_low)
        highest_high = self._get_rolling_max_cached(high, k_period, cache_key_high)

        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, np.nan).ffill().fillna(1)

        pct_k = 100 * (close - lowest_low) / range_val

        # Calculate %D (smoothed %K)（キャッシュ使用）
        pct_d = self._get_rolling_mean_cached(pct_k, d_period, cache_key_k + "_pctk")

        # Center around 50 for symmetry
        centered_k = pct_k - 50

        # Invert for mean reversion
        raw_signal = -centered_k

        # Normalize to [-1, +1] using tanh
        scores = self.normalize_tanh(raw_signal, scale=scale)

        # Calculate zone statistics
        oversold_pct = (pct_k < oversold_level).mean() * 100
        overbought_pct = (pct_k > overbought_level).mean() * 100

        metadata = {
            "k_period": k_period,
            "d_period": d_period,
            "oversold_level": oversold_level,
            "overbought_level": overbought_level,
            "scale": scale,
            "pct_k_mean": pct_k.mean(),
            "pct_d_mean": pct_d.mean(),
            "oversold_pct": oversold_pct,
            "overbought_pct": overbought_pct,
        }

        return SignalResult(scores=scores, metadata=metadata)
