"""
Dynamic Scorer Parameters - スコアラーパラメータの動的計算

このモジュールは、ScorerConfig のパラメータを市場状況や
過去のパフォーマンスに基づいて動的に計算する機能を提供する。

主要機能:
- calculate_penalty_coefficients: ペナルティ係数の動的計算
- calculate_mdd_normalization: MDD正規化基準の動的計算
- calculate_sharpe_adjustment: シャープレシオ調整係数の動的計算

設計根拠:
- 静的なパラメータでは市場環境の変化に対応できない
- 過去のデータから適切なペナルティ水準を学習
- ボラティリティ環境に応じたシャープレシオの価値調整
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicScorerParams:
    """動的に計算されたスコアラーパラメータ。

    ScorerConfig に渡すパラメータを保持する。
    これらの値は市場状況や過去のパフォーマンスに基づいて計算される。

    Attributes:
        penalty_turnover: ターンオーバーペナルティ係数（λ1）
        penalty_mdd: 最大ドローダウンペナルティ係数（λ2）
        penalty_instability: 不安定性ペナルティ係数（λ3）
        mdd_normalization_pct: MDD正規化基準（%）
        sharpe_adjustment: シャープレシオ調整係数
        lookback_days: 計算に使用したルックバック日数
        calculated_at: 計算日時
    """

    penalty_turnover: float
    penalty_mdd: float
    penalty_instability: float
    mdd_normalization_pct: float
    sharpe_adjustment: float
    lookback_days: int
    calculated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        result = asdict(self)
        result["calculated_at"] = self.calculated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicScorerParams:
        """辞書から生成する。"""
        data = data.copy()
        if isinstance(data["calculated_at"], str):
            data["calculated_at"] = datetime.fromisoformat(data["calculated_at"])
        return cls(**data)

    def to_scorer_config_kwargs(self) -> dict[str, float]:
        """ScorerConfig に渡すキーワード引数を生成する。"""
        return {
            "penalty_turnover": self.penalty_turnover,
            "penalty_mdd": self.penalty_mdd,
            "penalty_instability": self.penalty_instability,
            "mdd_normalization_pct": self.mdd_normalization_pct,
            "sharpe_adjustment_factor": self.sharpe_adjustment,
        }


@dataclass
class StrategyMetrics:
    """戦略メトリクスの時系列データ。

    Attributes:
        returns: 日次リターン系列
        turnover: 日次ターンオーバー系列
        mdd: 日次最大ドローダウン系列（累積）
        win_rate: ローリング勝率系列
    """

    returns: pd.Series
    turnover: pd.Series
    mdd: pd.Series
    win_rate: pd.Series | None = None


@dataclass
class MarketConditions:
    """市場状況。

    Attributes:
        current_volatility: 現在のボラティリティ（年率）
        historical_volatility_mean: 過去のボラティリティ平均（年率）
        historical_volatility_std: 過去のボラティリティ標準偏差
        vix_level: VIX水準（利用可能な場合）
    """

    current_volatility: float
    historical_volatility_mean: float
    historical_volatility_std: float
    vix_level: float | None = None


class DynamicScorerParamsCalculator:
    """スコアラーパラメータの動的計算クラス。

    過去の戦略パフォーマンスと市場状況に基づいて、
    スコアリングに最適なパラメータを計算する。

    Usage:
        calculator = DynamicScorerParamsCalculator()

        metrics = StrategyMetrics(
            returns=returns_series,
            turnover=turnover_series,
            mdd=mdd_series,
        )
        market = MarketConditions(
            current_volatility=0.15,
            historical_volatility_mean=0.18,
            historical_volatility_std=0.05,
        )

        params = calculator.calculate(metrics, market)
    """

    # デフォルトのペナルティ範囲
    PENALTY_TURNOVER_RANGE = (0.05, 0.30)
    PENALTY_MDD_RANGE = (0.10, 0.40)
    PENALTY_INSTABILITY_RANGE = (0.05, 0.30)

    # シャープ調整の範囲
    SHARPE_ADJUSTMENT_RANGE = (0.7, 1.5)

    def __init__(
        self,
        lookback_days: int = 252,
        penalty_turnover_range: tuple[float, float] | None = None,
        penalty_mdd_range: tuple[float, float] | None = None,
        penalty_instability_range: tuple[float, float] | None = None,
    ) -> None:
        """初期化。

        Args:
            lookback_days: 計算に使用するルックバック日数
            penalty_turnover_range: ターンオーバーペナルティの範囲 (min, max)
            penalty_mdd_range: MDDペナルティの範囲 (min, max)
            penalty_instability_range: 不安定性ペナルティの範囲 (min, max)
        """
        self.lookback_days = lookback_days
        self.penalty_turnover_range = penalty_turnover_range or self.PENALTY_TURNOVER_RANGE
        self.penalty_mdd_range = penalty_mdd_range or self.PENALTY_MDD_RANGE
        self.penalty_instability_range = penalty_instability_range or self.PENALTY_INSTABILITY_RANGE

    def calculate(
        self,
        strategy_metrics: StrategyMetrics,
        market_conditions: MarketConditions,
        historical_mdd: pd.Series | None = None,
    ) -> DynamicScorerParams:
        """全パラメータを計算する。

        Args:
            strategy_metrics: 戦略メトリクス
            market_conditions: 市場状況
            historical_mdd: 過去のMDD分布（MDD正規化計算用）

        Returns:
            計算されたスコアラーパラメータ
        """
        penalties = self.calculate_penalty_coefficients(strategy_metrics)
        mdd_norm = self.calculate_mdd_normalization(
            historical_mdd if historical_mdd is not None else strategy_metrics.mdd
        )
        sharpe_adj = self.calculate_sharpe_adjustment(market_conditions)

        return DynamicScorerParams(
            penalty_turnover=penalties["turnover"],
            penalty_mdd=penalties["mdd"],
            penalty_instability=penalties["instability"],
            mdd_normalization_pct=mdd_norm,
            sharpe_adjustment=sharpe_adj,
            lookback_days=self.lookback_days,
            calculated_at=datetime.now(),
        )

    def calculate_penalty_coefficients(
        self,
        strategy_metrics: StrategyMetrics,
    ) -> dict[str, float]:
        """ペナルティ係数を計算する。

        過去のパフォーマンスデータから、各ペナルティの適切な係数を算出。

        Args:
            strategy_metrics: 戦略メトリクス

        Returns:
            各ペナルティの係数 {"turnover", "mdd", "instability"}
        """
        returns = strategy_metrics.returns.iloc[-self.lookback_days:]
        turnover = strategy_metrics.turnover.iloc[-self.lookback_days:]
        mdd = strategy_metrics.mdd.iloc[-self.lookback_days:]

        # 1. ターンオーバーペナルティ
        penalty_turnover = self._calculate_turnover_penalty(returns, turnover)

        # 2. MDDペナルティ
        penalty_mdd = self._calculate_mdd_penalty(returns, mdd)

        # 3. 不安定性ペナルティ
        penalty_instability = self._calculate_instability_penalty(strategy_metrics)

        return {
            "turnover": penalty_turnover,
            "mdd": penalty_mdd,
            "instability": penalty_instability,
        }

    def _calculate_turnover_penalty(
        self,
        returns: pd.Series,
        turnover: pd.Series,
    ) -> float:
        """ターンオーバーペナルティを計算する。

        ターンオーバーとリターンの相関から決定。
        相関が負（高ターンオーバー→低リターン）なら高ペナルティ。

        Args:
            returns: リターン系列
            turnover: ターンオーバー系列

        Returns:
            ターンオーバーペナルティ係数
        """
        if len(returns) < 20 or len(turnover) < 20:
            return np.mean(self.penalty_turnover_range)

        # データを揃える
        aligned = pd.concat([returns, turnover], axis=1, keys=["returns", "turnover"])
        aligned = aligned.dropna()

        if len(aligned) < 20:
            return np.mean(self.penalty_turnover_range)

        correlation = aligned["returns"].corr(aligned["turnover"])

        if np.isnan(correlation):
            return np.mean(self.penalty_turnover_range)

        # 相関が -1 に近いほど高ペナルティ、+1 に近いほど低ペナルティ
        # correlation: [-1, 1] -> penalty: [max, min]
        min_p, max_p = self.penalty_turnover_range
        penalty = max_p - (correlation + 1) / 2 * (max_p - min_p)

        logger.debug(f"Turnover-return correlation: {correlation:.3f} -> penalty: {penalty:.3f}")
        return float(np.clip(penalty, min_p, max_p))

    def _calculate_mdd_penalty(
        self,
        returns: pd.Series,
        mdd: pd.Series,
    ) -> float:
        """MDDペナルティを計算する。

        DDからの回復速度に基づいて決定。
        回復が遅いほど高ペナルティ。

        Args:
            returns: リターン系列
            mdd: MDD系列（0以下の値、例: -0.15 = 15%ドローダウン）

        Returns:
            MDDペナルティ係数
        """
        if len(returns) < 20 or len(mdd) < 20:
            return np.mean(self.penalty_mdd_range)

        # ドローダウンからの回復日数を計算
        recovery_days = self._estimate_recovery_days(mdd)

        if recovery_days is None or recovery_days == 0:
            return np.mean(self.penalty_mdd_range)

        # 回復日数の正規化（長いほど高ペナルティ）
        # 基準: 20日で回復 = 低ペナルティ、60日以上 = 高ペナルティ
        normalized = np.clip((recovery_days - 20) / 40, 0, 1)

        min_p, max_p = self.penalty_mdd_range
        penalty = min_p + normalized * (max_p - min_p)

        logger.debug(f"Average recovery days: {recovery_days:.1f} -> penalty: {penalty:.3f}")
        return float(penalty)

    def _estimate_recovery_days(self, mdd: pd.Series) -> float | None:
        """平均回復日数を推定する。

        Args:
            mdd: MDD系列

        Returns:
            平均回復日数、推定不能な場合None
        """
        # MDDが-1%より深くなった時点から回復までの日数をカウント
        threshold = -0.01
        in_drawdown = False
        drawdown_start = 0
        recovery_days_list = []

        for i, dd in enumerate(mdd):
            if not in_drawdown and dd < threshold:
                in_drawdown = True
                drawdown_start = i
            elif in_drawdown and dd >= threshold * 0.1:  # 90%回復
                recovery_days_list.append(i - drawdown_start)
                in_drawdown = False

        if not recovery_days_list:
            return None

        return float(np.mean(recovery_days_list))

    def _calculate_instability_penalty(
        self,
        strategy_metrics: StrategyMetrics,
    ) -> float:
        """不安定性ペナルティを計算する。

        勝率のばらつきに基づいて決定。
        勝率の標準偏差が大きいほど高ペナルティ。

        Args:
            strategy_metrics: 戦略メトリクス

        Returns:
            不安定性ペナルティ係数
        """
        returns = strategy_metrics.returns.iloc[-self.lookback_days:]

        if len(returns) < 40:
            return np.mean(self.penalty_instability_range)

        # ローリング勝率を計算（20日窓）
        win_indicator = (returns > 0).astype(float)
        rolling_win_rate = win_indicator.rolling(window=20).mean().dropna()

        if len(rolling_win_rate) < 10:
            return np.mean(self.penalty_instability_range)

        # 勝率の標準偏差
        win_rate_std = rolling_win_rate.std()

        # 標準偏差の正規化（高いほど高ペナルティ）
        # 基準: std=0.05 = 低ペナルティ、std=0.20 = 高ペナルティ
        normalized = np.clip((win_rate_std - 0.05) / 0.15, 0, 1)

        min_p, max_p = self.penalty_instability_range
        penalty = min_p + normalized * (max_p - min_p)

        logger.debug(f"Win rate std: {win_rate_std:.3f} -> penalty: {penalty:.3f}")
        return float(penalty)

    def calculate_mdd_normalization(
        self,
        historical_mdd: pd.Series,
    ) -> float:
        """MDD正規化基準を計算する。

        過去のMDD分布の75パーセンタイルを基準とする。

        Args:
            historical_mdd: 過去のMDD系列

        Returns:
            MDD正規化基準（%、正の値）
        """
        if len(historical_mdd) < 20:
            return 25.0  # デフォルト値

        # MDDは負の値なので絶対値を取る
        mdd_abs = historical_mdd.abs() * 100  # %に変換

        # 75パーセンタイルを基準とする
        p75 = float(mdd_abs.quantile(0.75))

        # 最小5%、最大50%に制限
        result = np.clip(p75, 5.0, 50.0)

        logger.debug(f"MDD 75th percentile: {p75:.1f}% -> normalization: {result:.1f}%")
        return result

    def calculate_sharpe_adjustment(
        self,
        market_conditions: MarketConditions,
    ) -> float:
        """シャープレシオ調整係数を計算する。

        市場のボラティリティ環境に応じて調整。
        - 低ボラ環境: sharpe_adjustment > 1.0（シャープの価値が高い）
        - 高ボラ環境: sharpe_adjustment < 1.0（シャープ達成が容易）

        Args:
            market_conditions: 市場状況

        Returns:
            シャープレシオ調整係数
        """
        current_vol = market_conditions.current_volatility
        hist_mean = market_conditions.historical_volatility_mean
        hist_std = market_conditions.historical_volatility_std

        if hist_std == 0 or hist_mean == 0:
            return 1.0

        # 現在のボラティリティのzスコア
        z_score = (current_vol - hist_mean) / hist_std

        # zスコアを調整係数に変換
        # z < 0（低ボラ）-> adjustment > 1
        # z > 0（高ボラ）-> adjustment < 1
        min_adj, max_adj = self.SHARPE_ADJUSTMENT_RANGE
        mid_adj = (min_adj + max_adj) / 2

        # zスコア ±2 を範囲の端にマッピング
        adjustment = mid_adj - z_score * (max_adj - min_adj) / 4

        result = float(np.clip(adjustment, min_adj, max_adj))

        logger.debug(
            f"Volatility z-score: {z_score:.2f} -> sharpe adjustment: {result:.3f}"
        )
        return result


def create_dynamic_scorer_params(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    mdd: pd.Series | None = None,
    market_volatility: float | None = None,
    historical_volatility_mean: float | None = None,
    lookback_days: int = 252,
) -> DynamicScorerParams:
    """便利関数: 動的スコアラーパラメータを生成する。

    Args:
        returns: リターン系列
        turnover: ターンオーバー系列（オプション）
        mdd: MDD系列（オプション）
        market_volatility: 現在の市場ボラティリティ（オプション）
        historical_volatility_mean: 過去の平均ボラティリティ（オプション）
        lookback_days: ルックバック日数

    Returns:
        計算されたスコアラーパラメータ
    """
    # デフォルト値の設定
    if turnover is None:
        turnover = pd.Series(0.0, index=returns.index)

    if mdd is None:
        # リターンからMDDを計算
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        mdd = (cumulative - rolling_max) / rolling_max

    # ボラティリティの計算/設定
    vol = returns.std() * np.sqrt(252)
    if market_volatility is None:
        market_volatility = vol
    if historical_volatility_mean is None:
        historical_volatility_mean = vol

    historical_volatility_std = returns.rolling(63).std().std() * np.sqrt(252)
    if np.isnan(historical_volatility_std):
        historical_volatility_std = vol * 0.3

    # メトリクスと市場状況の構築
    strategy_metrics = StrategyMetrics(
        returns=returns,
        turnover=turnover,
        mdd=mdd,
    )

    market_conditions = MarketConditions(
        current_volatility=market_volatility,
        historical_volatility_mean=historical_volatility_mean,
        historical_volatility_std=historical_volatility_std,
    )

    # 計算
    calculator = DynamicScorerParamsCalculator(lookback_days=lookback_days)
    return calculator.calculate(strategy_metrics, market_conditions)


def update_scorer_config_dynamically(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    mdd: pd.Series | None = None,
) -> dict[str, float]:
    """便利関数: ScorerConfig 用のパラメータを動的に生成する。

    Args:
        returns: リターン系列
        turnover: ターンオーバー系列（オプション）
        mdd: MDD系列（オプション）

    Returns:
        ScorerConfig のキーワード引数として使える辞書

    Usage:
        from src.meta import ScorerConfig, StrategyScorer

        kwargs = update_scorer_config_dynamically(returns)
        config = ScorerConfig(**kwargs)
        scorer = StrategyScorer(config)
    """
    params = create_dynamic_scorer_params(returns, turnover, mdd)
    return params.to_scorer_config_kwargs()
