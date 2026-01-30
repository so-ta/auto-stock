"""
Signal Decay Module - シグナル減衰フィルター

古いシグナルの重みを指数減衰させることで、
新しい情報により高い重要性を与える。

設計根拠:
- 要求.md §5: シグナル生成
- シグナルの鮮度に基づく重み付け
- 情報の陳腐化を反映

主な機能:
- 指数減衰によるシグナル重み調整
- 最小重みによるシグナル完全消失の防止
- アセット別のシグナル履歴管理

減衰式:
    weight = max(min_weight, 0.5 ^ (age / halflife))
    decayed_signal = original_signal * weight

例:
    halflife=5 の場合:
    - 0日経過: weight = 1.0
    - 5日経過: weight = 0.5
    - 10日経過: weight = 0.25
    - 15日経過: weight = 0.125
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecayConfig:
    """シグナル減衰設定

    Attributes:
        halflife: 半減期（日数）。この日数でシグナル重みが半分になる
        min_weight: 最小重み。これ以下には減衰しない（0-1）
        max_age: 最大経過日数。これを超えるとシグナルを破棄
        use_trading_days: 営業日ベースで計算するか
    """
    halflife: float = 5.0
    min_weight: float = 0.1
    max_age: Optional[int] = None
    use_trading_days: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.halflife <= 0:
            raise ValueError("halflife must be positive")
        if not 0.0 <= self.min_weight <= 1.0:
            raise ValueError("min_weight must be in [0, 1]")
        if self.max_age is not None and self.max_age <= 0:
            raise ValueError("max_age must be positive if specified")


@dataclass
class SignalEntry:
    """シグナルエントリ

    Attributes:
        signal_value: シグナル値（-1〜+1）
        timestamp: シグナル発生時刻
        original_weight: 初期重み（通常1.0）
        source: シグナルソース名
        metadata: 追加メタデータ
    """
    signal_value: float
    timestamp: datetime
    original_weight: float = 1.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_in_days(self, current_time: Optional[datetime] = None) -> float:
        """シグナルの経過日数を計算

        Args:
            current_time: 現在時刻。Noneの場合は現在時刻を使用

        Returns:
            経過日数（小数点以下あり）
        """
        if current_time is None:
            current_time = datetime.now()
        delta = current_time - self.timestamp
        return delta.total_seconds() / (24 * 3600)


@dataclass
class DecayedSignal:
    """減衰後シグナル

    Attributes:
        asset: アセット識別子
        original_signal: 元のシグナル値
        decayed_signal: 減衰後のシグナル値
        current_weight: 現在の重み
        age_days: 経過日数
        is_expired: 期限切れかどうか
    """
    asset: str
    original_signal: float
    decayed_signal: float
    current_weight: float
    age_days: float
    is_expired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset": self.asset,
            "original_signal": self.original_signal,
            "decayed_signal": self.decayed_signal,
            "current_weight": self.current_weight,
            "age_days": self.age_days,
            "is_expired": self.is_expired,
        }


class SignalDecayFilter:
    """シグナル減衰フィルタークラス

    古いシグナルの重みを指数減衰させる。
    各アセットごとにシグナル履歴を管理し、
    新しいシグナルがあれば更新、なければ既存シグナルを減衰させる。

    減衰式:
        weight = max(min_weight, 0.5 ^ (age / halflife))
        decayed_signal = original_signal * weight

    Usage:
        config = DecayConfig(halflife=5, min_weight=0.1)
        decay_filter = SignalDecayFilter(config)

        # シグナル更新
        decay_filter.update_and_decay("AAPL", 0.8)
        decay_filter.update_and_decay("MSFT", 0.6)

        # 数日後に減衰シグナルを取得
        signal = decay_filter.get_decayed_signal("AAPL")
        print(f"Decayed signal: {signal.decayed_signal}")

    Attributes:
        config: 減衰設定
        _signal_history: アセット別シグナル履歴
    """

    def __init__(self, config: Optional[DecayConfig] = None) -> None:
        """初期化

        Args:
            config: 減衰設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or DecayConfig()
        self._signal_history: Dict[str, SignalEntry] = {}

    def _calculate_weight(self, age_days: float) -> float:
        """減衰重みを計算

        Args:
            age_days: 経過日数

        Returns:
            減衰後の重み（min_weight以上）
        """
        # 指数減衰: weight = 0.5 ^ (age / halflife)
        raw_weight = 0.5 ** (age_days / self.config.halflife)
        return max(self.config.min_weight, raw_weight)

    def update_and_decay(
        self,
        asset: str,
        new_signal: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DecayedSignal:
        """シグナルを更新・減衰

        新しいシグナルがあれば更新し、なければ既存シグナルを減衰させる。

        Args:
            asset: アセット識別子
            new_signal: 新しいシグナル値（-1〜+1）。Noneの場合は減衰のみ
            timestamp: シグナル発生時刻。Noneの場合は現在時刻
            source: シグナルソース名
            metadata: 追加メタデータ

        Returns:
            DecayedSignal: 減衰後シグナル情報
        """
        current_time = timestamp or datetime.now()

        if new_signal is not None:
            # 新しいシグナルで更新
            entry = SignalEntry(
                signal_value=new_signal,
                timestamp=current_time,
                original_weight=1.0,
                source=source,
                metadata=metadata or {},
            )
            self._signal_history[asset] = entry

            logger.debug(
                "Updated signal for %s: value=%.4f, source=%s",
                asset, new_signal, source
            )

            return DecayedSignal(
                asset=asset,
                original_signal=new_signal,
                decayed_signal=new_signal,
                current_weight=1.0,
                age_days=0.0,
                is_expired=False,
            )

        # 既存シグナルを減衰
        if asset not in self._signal_history:
            # 履歴がない場合
            return DecayedSignal(
                asset=asset,
                original_signal=0.0,
                decayed_signal=0.0,
                current_weight=0.0,
                age_days=0.0,
                is_expired=True,
            )

        entry = self._signal_history[asset]
        age_days = entry.age_in_days(current_time)

        # 最大経過日数チェック
        is_expired = False
        if self.config.max_age is not None and age_days > self.config.max_age:
            is_expired = True

        # 減衰重み計算
        weight = self._calculate_weight(age_days)
        decayed_signal = entry.signal_value * weight

        logger.debug(
            "Decayed signal for %s: original=%.4f, decayed=%.4f, "
            "weight=%.4f, age=%.1f days",
            asset, entry.signal_value, decayed_signal, weight, age_days
        )

        return DecayedSignal(
            asset=asset,
            original_signal=entry.signal_value,
            decayed_signal=decayed_signal,
            current_weight=weight,
            age_days=age_days,
            is_expired=is_expired,
        )

    def get_decayed_signal(
        self,
        asset: str,
        current_time: Optional[datetime] = None,
    ) -> DecayedSignal:
        """減衰後シグナルを取得

        Args:
            asset: アセット識別子
            current_time: 現在時刻。Noneの場合は現在時刻

        Returns:
            DecayedSignal: 減衰後シグナル情報
        """
        return self.update_and_decay(asset, None, current_time)

    def get_all_decayed_signals(
        self,
        current_time: Optional[datetime] = None,
        include_expired: bool = False,
    ) -> Dict[str, DecayedSignal]:
        """全アセットの減衰後シグナルを取得

        Args:
            current_time: 現在時刻
            include_expired: 期限切れシグナルを含めるか

        Returns:
            {asset: DecayedSignal} の辞書
        """
        result: Dict[str, DecayedSignal] = {}

        for asset in list(self._signal_history.keys()):
            decayed = self.get_decayed_signal(asset, current_time)
            if include_expired or not decayed.is_expired:
                result[asset] = decayed

        return result

    def reset(self, asset: Optional[str] = None) -> None:
        """履歴をリセット

        Args:
            asset: リセット対象のアセット。Noneの場合は全てリセット
        """
        if asset is None:
            self._signal_history.clear()
            logger.info("Reset all signal history")
        elif asset in self._signal_history:
            del self._signal_history[asset]
            logger.info("Reset signal history for %s", asset)

    def cleanup_expired(
        self,
        current_time: Optional[datetime] = None,
    ) -> List[str]:
        """期限切れシグナルを削除

        Args:
            current_time: 現在時刻

        Returns:
            削除されたアセットのリスト
        """
        if self.config.max_age is None:
            return []

        current_time = current_time or datetime.now()
        expired: List[str] = []

        for asset in list(self._signal_history.keys()):
            entry = self._signal_history[asset]
            if entry.age_in_days(current_time) > self.config.max_age:
                expired.append(asset)
                del self._signal_history[asset]

        if expired:
            logger.info("Cleaned up %d expired signals: %s", len(expired), expired)

        return expired

    @property
    def active_assets(self) -> List[str]:
        """アクティブなシグナルを持つアセットのリスト"""
        return list(self._signal_history.keys())

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報の辞書
        """
        if not self._signal_history:
            return {
                "n_assets": 0,
                "avg_age_days": 0.0,
                "avg_weight": 0.0,
            }

        current_time = datetime.now()
        ages = [
            entry.age_in_days(current_time)
            for entry in self._signal_history.values()
        ]
        weights = [self._calculate_weight(age) for age in ages]

        return {
            "n_assets": len(self._signal_history),
            "avg_age_days": np.mean(ages),
            "max_age_days": np.max(ages),
            "min_age_days": np.min(ages),
            "avg_weight": np.mean(weights),
            "min_weight": np.min(weights),
            "config": {
                "halflife": self.config.halflife,
                "min_weight": self.config.min_weight,
                "max_age": self.config.max_age,
            },
        }


class TimeSeriesSignalDecay:
    """時系列シグナル減衰

    時系列データ（DataFrame）に対して減衰を適用する。
    過去のシグナル値を減衰させた加重平均を計算。

    Usage:
        ts_decay = TimeSeriesSignalDecay(halflife=5, min_weight=0.1)
        decayed_signals = ts_decay.apply(signal_df)
    """

    def __init__(
        self,
        halflife: float = 5.0,
        min_weight: float = 0.1,
        lookback: int = 20,
    ) -> None:
        """初期化

        Args:
            halflife: 半減期（日数）
            min_weight: 最小重み
            lookback: 過去何日分を考慮するか
        """
        self.halflife = halflife
        self.min_weight = min_weight
        self.lookback = lookback

    def _decay_weights(self, n: int) -> np.ndarray:
        """減衰重みベクトルを生成

        Args:
            n: 重みの数

        Returns:
            減衰重みの配列（最新が先頭）
        """
        ages = np.arange(n)
        weights = 0.5 ** (ages / self.halflife)
        weights = np.maximum(weights, self.min_weight)
        return weights / weights.sum()  # 正規化

    def apply(
        self,
        signals: Union[pd.DataFrame, pd.Series],
    ) -> Union[pd.DataFrame, pd.Series]:
        """減衰を適用

        Args:
            signals: シグナルの時系列（DataFrame or Series）

        Returns:
            減衰後のシグナル
        """
        if isinstance(signals, pd.Series):
            return self._apply_to_series(signals)
        else:
            return signals.apply(self._apply_to_series)

    def _apply_to_series(self, series: pd.Series) -> pd.Series:
        """Seriesに減衰を適用"""
        result = pd.Series(index=series.index, dtype=float)

        for i in range(len(series)):
            start = max(0, i - self.lookback + 1)
            window = series.iloc[start:i + 1].values[::-1]  # 最新が先頭
            weights = self._decay_weights(len(window))
            result.iloc[i] = np.sum(window * weights)

        return result


# ============================================================
# 便利関数
# ============================================================

def apply_signal_decay(
    signals_dict: Dict[str, float],
    halflife: float = 5.0,
    min_weight: float = 0.1,
    ages: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """シグナル辞書に減衰を適用（簡易版）

    Args:
        signals_dict: {asset: signal_value} の辞書
        halflife: 半減期（日数）
        min_weight: 最小重み
        ages: {asset: age_in_days} の辞書。Noneの場合は全て0日

    Returns:
        減衰後のシグナル辞書
    """
    result: Dict[str, float] = {}
    ages = ages or {}

    for asset, signal in signals_dict.items():
        age = ages.get(asset, 0.0)
        weight = max(min_weight, 0.5 ** (age / halflife))
        result[asset] = signal * weight

    return result


def create_decay_filter(
    halflife: float = 5.0,
    min_weight: float = 0.1,
    max_age: Optional[int] = None,
) -> SignalDecayFilter:
    """減衰フィルターを作成

    Args:
        halflife: 半減期（日数）
        min_weight: 最小重み
        max_age: 最大経過日数

    Returns:
        SignalDecayFilter インスタンス
    """
    config = DecayConfig(
        halflife=halflife,
        min_weight=min_weight,
        max_age=max_age,
    )
    return SignalDecayFilter(config)


def calculate_decay_weight(
    age_days: float,
    halflife: float = 5.0,
    min_weight: float = 0.1,
) -> float:
    """減衰重みを計算（スタンドアロン関数）

    Args:
        age_days: 経過日数
        halflife: 半減期
        min_weight: 最小重み

    Returns:
        減衰重み
    """
    raw_weight = 0.5 ** (age_days / halflife)
    return max(min_weight, raw_weight)


def decay_time_series(
    signals: pd.DataFrame,
    halflife: float = 5.0,
    min_weight: float = 0.1,
    lookback: int = 20,
) -> pd.DataFrame:
    """時系列シグナルに減衰を適用（便利関数）

    Args:
        signals: シグナルのDataFrame（index=date, columns=assets）
        halflife: 半減期
        min_weight: 最小重み
        lookback: 過去何日分を考慮するか

    Returns:
        減衰後のシグナルDataFrame
    """
    ts_decay = TimeSeriesSignalDecay(
        halflife=halflife,
        min_weight=min_weight,
        lookback=lookback,
    )
    return ts_decay.apply(signals)
