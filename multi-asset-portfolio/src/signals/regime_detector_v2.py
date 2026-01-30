"""
Enhanced Regime Detector V2 - 改善版レジーム検出

HMM、ボラティリティ、トレンドの3手法をアンサンブルして
より精度の高いレジーム検出を実現。

5レジーム:
- crisis: 危機相場（高ボラ+下落トレンド）
- high_vol: 高ボラティリティ（方向性不明瞭）
- normal: 通常相場
- low_vol: 低ボラティリティ（安定相場）
- trending: 強トレンド（明確な方向性）

Usage:
    from src.signals.regime_detector_v2 import EnhancedRegimeDetector

    detector = EnhancedRegimeDetector()
    regime = detector.detect_regime(price_df)
    probabilities = detector.get_regime_probability()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class RegimeType(Enum):
    """レジーム種別"""
    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"
    TRENDING = "trending"


@dataclass
class RegimeDetectorConfig:
    """レジーム検出設定"""

    # アンサンブル重み
    hmm_weight: float = 0.4
    volatility_weight: float = 0.3
    trend_weight: float = 0.3

    # ルックバック期間
    lookback: int = 60
    short_lookback: int = 20

    # 遷移スムージング（急激な変化を抑制）
    transition_smoothing: int = 5

    # ボラティリティ閾値
    vol_crisis_threshold: float = 0.4    # 年率40%以上で危機
    vol_high_threshold: float = 0.25     # 年率25%以上で高ボラ
    vol_low_threshold: float = 0.10      # 年率10%以下で低ボラ

    # トレンド閾値
    trend_strong_threshold: float = 0.15  # 15%以上で強トレンド
    trend_weak_threshold: float = 0.05    # 5%以下で方向性なし

    # HMM設定（簡易版）
    hmm_n_states: int = 3
    hmm_use_returns: bool = True


@dataclass
class RegimeResult:
    """レジーム検出結果"""
    regime: RegimeType
    confidence: float
    probabilities: Dict[str, float]
    method_votes: Dict[str, str]
    raw_metrics: Dict[str, float]


class EnhancedRegimeDetector:
    """
    改善版レジーム検出器

    3つの手法（HMM、ボラティリティ、トレンド）をアンサンブルして
    より精度の高いレジーム検出を実現。

    Attributes:
        config: RegimeDetectorConfig設定
    """

    def __init__(self, config: dict | RegimeDetectorConfig | None = None):
        """
        初期化

        Args:
            config: 設定辞書またはRegimeDetectorConfig
        """
        if config is None:
            self.config = RegimeDetectorConfig()
        elif isinstance(config, dict):
            self.config = RegimeDetectorConfig(**config)
        else:
            self.config = config

        self._last_regime: Optional[RegimeType] = None
        self._regime_history: List[RegimeType] = []
        self._probabilities: Dict[str, float] = {}

    def _calculate_volatility(self, prices: pd.Series) -> float:
        """
        年率ボラティリティを計算

        Args:
            prices: 価格シリーズ

        Returns:
            年率ボラティリティ
        """
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def _calculate_trend_strength(self, prices: pd.Series) -> Tuple[float, float]:
        """
        トレンド強度と方向を計算

        Args:
            prices: 価格シリーズ

        Returns:
            (トレンド強度, 方向) - 方向は正=上昇、負=下落
        """
        if len(prices) < 2:
            return 0.0, 0.0

        # 期間リターン
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

        # 線形回帰の傾き（正規化）
        x = np.arange(len(prices))
        y = prices.values
        if np.std(y) == 0:
            return 0.0, 0.0

        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope * len(prices) / prices.mean()

        # トレンド強度（絶対値）
        strength = abs(normalized_slope)

        # 方向
        direction = 1.0 if total_return > 0 else -1.0

        return float(strength), float(direction)

    def _calculate_drawdown(self, prices: pd.Series) -> float:
        """
        現在のドローダウンを計算

        Args:
            prices: 価格シリーズ

        Returns:
            ドローダウン（負の値）
        """
        if len(prices) < 2:
            return 0.0

        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return float(drawdown.iloc[-1])

    def _hmm_detect(self, prices: pd.Series) -> Tuple[str, float]:
        """
        簡易HMM風レジーム検出

        実際のHMMは依存関係が重いので、
        リターンと変動性の組み合わせで近似。

        Args:
            prices: 価格シリーズ

        Returns:
            (レジーム名, 確信度)
        """
        if len(prices) < self.config.lookback:
            return "normal", 0.5

        returns = prices.pct_change().dropna()
        recent_returns = returns.iloc[-self.config.short_lookback:]

        # 平均リターンと標準偏差
        mean_ret = recent_returns.mean()
        std_ret = recent_returns.std()

        # 状態推定（簡易版）
        # State 0: 低リターン・高ボラ（危機）
        # State 1: 中リターン・中ボラ（通常）
        # State 2: 高リターン・低ボラ（好調）

        if std_ret == 0:
            return "normal", 0.5

        # シャープレシオ風指標
        sharpe_like = mean_ret / std_ret * np.sqrt(252)

        if sharpe_like < -1.0 and std_ret * np.sqrt(252) > 0.3:
            return "crisis", 0.8
        elif sharpe_like < -0.5:
            return "high_vol", 0.6
        elif sharpe_like > 1.5 and std_ret * np.sqrt(252) < 0.15:
            return "low_vol", 0.7
        elif sharpe_like > 1.0:
            return "trending", 0.7
        else:
            return "normal", 0.6

    def _volatility_detect(self, prices: pd.Series) -> Tuple[str, float]:
        """
        ボラティリティベースレジーム検出

        Args:
            prices: 価格シリーズ

        Returns:
            (レジーム名, 確信度)
        """
        vol = self._calculate_volatility(prices.iloc[-self.config.lookback:])
        drawdown = self._calculate_drawdown(prices.iloc[-self.config.lookback:])

        # 危機判定：高ボラ + 大きなドローダウン
        if vol >= self.config.vol_crisis_threshold and drawdown < -0.15:
            return "crisis", 0.9

        if vol >= self.config.vol_crisis_threshold:
            return "high_vol", 0.8

        if vol >= self.config.vol_high_threshold:
            return "high_vol", 0.7

        if vol <= self.config.vol_low_threshold:
            return "low_vol", 0.8

        return "normal", 0.6

    def _trend_detect(self, prices: pd.Series) -> Tuple[str, float]:
        """
        トレンドベースレジーム検出

        Args:
            prices: 価格シリーズ

        Returns:
            (レジーム名, 確信度)
        """
        strength, direction = self._calculate_trend_strength(
            prices.iloc[-self.config.lookback:]
        )

        # 強トレンド
        if strength >= self.config.trend_strong_threshold:
            if direction < 0:
                # 強い下落トレンド = 危機の可能性
                return "crisis", 0.7
            else:
                return "trending", 0.8

        # 弱トレンド（方向性なし）
        if strength <= self.config.trend_weak_threshold:
            return "normal", 0.5

        return "normal", 0.6

    def _ensemble_vote(
        self,
        votes: List[Tuple[str, float]],
        weights: List[float],
    ) -> Tuple[str, Dict[str, float]]:
        """
        アンサンブル投票

        Args:
            votes: (レジーム名, 確信度)のリスト
            weights: 各手法の重み

        Returns:
            (最終レジーム, 各レジームの確率)
        """
        # 確率を集計
        regime_scores: Dict[str, float] = {
            "crisis": 0.0,
            "high_vol": 0.0,
            "normal": 0.0,
            "low_vol": 0.0,
            "trending": 0.0,
        }

        total_weight = sum(weights)
        for (regime, confidence), weight in zip(votes, weights):
            regime_scores[regime] += weight * confidence / total_weight

        # 正規化
        total = sum(regime_scores.values())
        if total > 0:
            regime_scores = {k: v / total for k, v in regime_scores.items()}

        # 最も高いスコアのレジームを選択
        best_regime = max(regime_scores.items(), key=lambda x: x[1])

        return best_regime[0], regime_scores

    def _apply_smoothing(self, regime: str) -> str:
        """
        遷移スムージングを適用

        急激なレジーム変化を抑制。

        Args:
            regime: 新しいレジーム

        Returns:
            スムージング後のレジーム
        """
        if len(self._regime_history) < self.config.transition_smoothing:
            return regime

        # 直近の履歴をチェック
        recent = self._regime_history[-self.config.transition_smoothing:]

        # 過半数が同じレジームなら維持
        from collections import Counter
        counts = Counter(recent)
        most_common = counts.most_common(1)[0]

        if most_common[1] >= self.config.transition_smoothing // 2 + 1:
            if regime != most_common[0]:
                # 新レジームへの遷移には連続性が必要
                recent_new = [r for r in recent if r == regime]
                if len(recent_new) < 2:
                    return most_common[0]

        return regime

    def detect_regime(
        self,
        prices: pd.DataFrame | pd.Series,
        ticker: str | None = None,
    ) -> RegimeResult:
        """
        レジームを検出

        Args:
            prices: 価格データ
            ticker: 特定銘柄（DataFrameの場合）

        Returns:
            RegimeResult
        """
        if isinstance(prices, pd.DataFrame):
            if ticker is not None:
                price_series = prices[ticker]
            else:
                # 最初の列または平均を使用
                price_series = prices.iloc[:, 0] if prices.shape[1] > 0 else pd.Series()
        else:
            price_series = prices

        if len(price_series) < self.config.lookback:
            return RegimeResult(
                regime=RegimeType.NORMAL,
                confidence=0.3,
                probabilities={"normal": 1.0},
                method_votes={},
                raw_metrics={},
            )

        # 各手法で検出
        hmm_result = self._hmm_detect(price_series)
        vol_result = self._volatility_detect(price_series)
        trend_result = self._trend_detect(price_series)

        # アンサンブル
        weights = [
            self.config.hmm_weight,
            self.config.volatility_weight,
            self.config.trend_weight,
        ]
        final_regime, probabilities = self._ensemble_vote(
            [hmm_result, vol_result, trend_result],
            weights,
        )

        # スムージング適用
        smoothed_regime = self._apply_smoothing(final_regime)

        # 履歴更新
        self._regime_history.append(smoothed_regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

        self._last_regime = RegimeType(smoothed_regime)
        self._probabilities = probabilities

        # 生メトリクス
        raw_metrics = {
            "volatility": self._calculate_volatility(price_series.iloc[-self.config.lookback:]),
            "drawdown": self._calculate_drawdown(price_series.iloc[-self.config.lookback:]),
            "trend_strength": self._calculate_trend_strength(price_series.iloc[-self.config.lookback:])[0],
        }

        return RegimeResult(
            regime=RegimeType(smoothed_regime),
            confidence=probabilities.get(smoothed_regime, 0.5),
            probabilities=probabilities,
            method_votes={
                "hmm": hmm_result[0],
                "volatility": vol_result[0],
                "trend": trend_result[0],
            },
            raw_metrics=raw_metrics,
        )

    def get_regime_probability(self) -> Dict[str, float]:
        """
        各レジームの確率を取得

        Returns:
            レジーム名→確率の辞書
        """
        return self._probabilities.copy()

    def get_current_regime(self) -> Optional[RegimeType]:
        """
        現在のレジームを取得

        Returns:
            現在のRegimeType（未検出ならNone）
        """
        return self._last_regime

    def get_regime_history(self) -> List[str]:
        """
        レジーム履歴を取得

        Returns:
            レジーム名のリスト
        """
        return self._regime_history.copy()

    def reset(self) -> None:
        """検出状態をリセット"""
        self._last_regime = None
        self._regime_history = []
        self._probabilities = {}


class RegimeAdaptiveStrategy:
    """
    レジーム適応戦略ヘルパー

    検出されたレジームに応じてパラメータを調整。
    """

    # レジーム別推奨パラメータ
    REGIME_PARAMS = {
        "crisis": {
            "risk_budget": 0.5,      # リスク予算を半減
            "cash_allocation": 0.3,  # 現金30%
            "momentum_weight": 0.2,  # モメンタム低下
            "reversion_weight": 0.1, # リバージョンも低下
            "rebalance_freq": "daily",
        },
        "high_vol": {
            "risk_budget": 0.7,
            "cash_allocation": 0.15,
            "momentum_weight": 0.3,
            "reversion_weight": 0.2,
            "rebalance_freq": "weekly",
        },
        "normal": {
            "risk_budget": 1.0,
            "cash_allocation": 0.05,
            "momentum_weight": 0.5,
            "reversion_weight": 0.3,
            "rebalance_freq": "weekly",
        },
        "low_vol": {
            "risk_budget": 1.2,      # リスク予算増加
            "cash_allocation": 0.0,
            "momentum_weight": 0.4,
            "reversion_weight": 0.4, # リバージョン強化
            "rebalance_freq": "monthly",
        },
        "trending": {
            "risk_budget": 1.0,
            "cash_allocation": 0.0,
            "momentum_weight": 0.7,  # モメンタム強化
            "reversion_weight": 0.1,
            "rebalance_freq": "weekly",
        },
    }

    @classmethod
    def get_params(cls, regime: str | RegimeType) -> Dict:
        """
        レジームに応じたパラメータを取得

        Args:
            regime: レジーム名またはRegimeType

        Returns:
            パラメータ辞書
        """
        if isinstance(regime, RegimeType):
            regime = regime.value

        return cls.REGIME_PARAMS.get(regime, cls.REGIME_PARAMS["normal"]).copy()

    @classmethod
    def interpolate_params(
        cls,
        probabilities: Dict[str, float],
    ) -> Dict:
        """
        確率に基づいてパラメータを補間

        Args:
            probabilities: レジーム→確率の辞書

        Returns:
            補間されたパラメータ辞書
        """
        result = {
            "risk_budget": 0.0,
            "cash_allocation": 0.0,
            "momentum_weight": 0.0,
            "reversion_weight": 0.0,
        }

        for regime, prob in probabilities.items():
            params = cls.REGIME_PARAMS.get(regime, cls.REGIME_PARAMS["normal"])
            for key in result:
                result[key] += prob * params[key]

        # リバランス頻度は最も確率の高いレジームから
        top_regime = max(probabilities.items(), key=lambda x: x[1])[0]
        result["rebalance_freq"] = cls.REGIME_PARAMS[top_regime]["rebalance_freq"]

        return result


# 便利関数
def detect_market_regime(
    prices: pd.Series,
    lookback: int = 60,
) -> str:
    """
    簡易レジーム検出

    Args:
        prices: 価格シリーズ
        lookback: ルックバック期間

    Returns:
        レジーム名
    """
    detector = EnhancedRegimeDetector({"lookback": lookback})
    result = detector.detect_regime(prices)
    return result.regime.value


def get_regime_adjusted_weights(
    prices: pd.Series,
    base_momentum_weight: float = 0.6,
    base_reversion_weight: float = 0.4,
) -> Tuple[float, float]:
    """
    レジーム調整済み重みを取得

    Args:
        prices: 価格シリーズ
        base_momentum_weight: 基本モメンタム重み
        base_reversion_weight: 基本リバージョン重み

    Returns:
        (モメンタム重み, リバージョン重み)
    """
    detector = EnhancedRegimeDetector()
    result = detector.detect_regime(prices)

    params = RegimeAdaptiveStrategy.interpolate_params(result.probabilities)

    # 基本重みをレジームパラメータで調整
    mom_adjustment = params["momentum_weight"] / 0.5  # normalが0.5
    rev_adjustment = params["reversion_weight"] / 0.3  # normalが0.3

    adjusted_mom = base_momentum_weight * mom_adjustment
    adjusted_rev = base_reversion_weight * rev_adjustment

    # 正規化
    total = adjusted_mom + adjusted_rev
    return adjusted_mom / total, adjusted_rev / total
