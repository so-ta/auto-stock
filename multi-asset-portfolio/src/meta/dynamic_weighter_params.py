"""
Dynamic Weighter Parameters - 動的パラメータ計算

WeighterConfig のパラメータを市場状況に応じて動的に計算する。

機能:
1. calculate_optimal_beta: スコア分布から最適なβを計算
2. calculate_w_strategy_max: 戦略数と目標分散度から上限を計算
3. calculate_score_threshold: スコア分布から閾値を計算

設計根拠:
- スコアの分散が大きい → beta低め（分散重視）
- スコアの分散が小さい → beta高め（集中許容）
- 動的調整で市場環境に適応
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DynamicWeighterParams:
    """動的計算されたウェイターパラメータ

    Attributes:
        beta: Softmax温度パラメータ
        w_strategy_max: 単一戦略の最大重み
        score_threshold: 採用スコア閾値
        lookback_days: 計算に使用したルックバック日数
        calculated_at: 計算日時
        metadata: 計算に関する追加情報
    """

    beta: float
    w_strategy_max: float
    score_threshold: float
    lookback_days: int
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "beta": self.beta,
            "w_strategy_max": self.w_strategy_max,
            "score_threshold": self.score_threshold,
            "lookback_days": self.lookback_days,
            "calculated_at": self.calculated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_weighter_config_dict(self) -> dict[str, Any]:
        """WeighterConfig 用の辞書形式に変換"""
        return {
            "beta": self.beta,
            "w_strategy_max": self.w_strategy_max,
            "score_threshold": self.score_threshold,
        }


@dataclass(frozen=True)
class DynamicParamsConfig:
    """動的パラメータ計算の設定

    Attributes:
        lookback_days: スコア履歴のルックバック日数
        beta_base: βの基準値
        beta_min: βの下限
        beta_max: βの上限
        beta_sensitivity: スコア分散に対するβの感度
        diversification_target: 目標有効戦略数
        score_percentile: スコア閾値のパーセンタイル（下位）
        min_w_strategy_max: 戦略最大重みの下限
        max_w_strategy_max: 戦略最大重みの上限
    """

    lookback_days: int = 252
    beta_base: float = 2.0
    beta_min: float = 0.5
    beta_max: float = 5.0
    beta_sensitivity: float = 1.0
    diversification_target: float = 4.0
    score_percentile: float = 10.0
    min_w_strategy_max: float = 0.15
    max_w_strategy_max: float = 0.50

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")
        if self.beta_min >= self.beta_max:
            raise ValueError("beta_min must be < beta_max")
        if not 0 < self.diversification_target:
            raise ValueError("diversification_target must be > 0")
        if not 0 < self.score_percentile < 100:
            raise ValueError("score_percentile must be in (0, 100)")


class DynamicWeighterParamsCalculator:
    """動的ウェイターパラメータ計算クラス

    過去のスコア履歴と戦略情報から、WeighterConfig のパラメータを動的に計算する。

    使用例:
        calculator = DynamicWeighterParamsCalculator()

        # スコア履歴（DataFrame: 行=日付, 列=戦略）
        score_history = pd.DataFrame(...)

        # パラメータ計算
        params = calculator.calculate_all(score_history, num_strategies=10)

        # WeighterConfig に適用
        from src.meta.weighter import WeighterConfig
        config = WeighterConfig(**params.to_weighter_config_dict())
    """

    def __init__(self, config: DynamicParamsConfig | None = None) -> None:
        """初期化

        Args:
            config: 計算設定（Noneの場合デフォルト）
        """
        self.config = config or DynamicParamsConfig()

    def calculate_all(
        self,
        strategy_scores: pd.DataFrame | pd.Series,
        num_strategies: int | None = None,
    ) -> DynamicWeighterParams:
        """全パラメータを計算

        Args:
            strategy_scores: 戦略スコア履歴
                DataFrame: 行=日付, 列=戦略
                Series: 単一戦略のスコア履歴
            num_strategies: 戦略数（Noneの場合は strategy_scores から推定）

        Returns:
            DynamicWeighterParams
        """
        # DataFrame に統一
        if isinstance(strategy_scores, pd.Series):
            strategy_scores = strategy_scores.to_frame()

        if num_strategies is None:
            num_strategies = len(strategy_scores.columns)

        # ルックバック期間でフィルタ
        lookback = min(self.config.lookback_days, len(strategy_scores))
        scores_lookback = strategy_scores.tail(lookback)

        # 各パラメータを計算
        beta = self.calculate_optimal_beta(scores_lookback)
        w_strategy_max = self.calculate_w_strategy_max(
            num_strategies, self.config.diversification_target
        )
        score_threshold = self.calculate_score_threshold(scores_lookback)

        # メタデータ
        metadata = {
            "score_mean": float(scores_lookback.values.mean()),
            "score_std": float(scores_lookback.values.std()),
            "num_strategies": num_strategies,
            "diversification_target": self.config.diversification_target,
            "score_percentile_used": self.config.score_percentile,
        }

        return DynamicWeighterParams(
            beta=beta,
            w_strategy_max=w_strategy_max,
            score_threshold=score_threshold,
            lookback_days=lookback,
            metadata=metadata,
        )

    def calculate_optimal_beta(
        self,
        strategy_scores: pd.DataFrame | pd.Series,
        lookback_days: int | None = None,
    ) -> float:
        """最適なβ（温度パラメータ）を計算

        スコアの分散が大きい → beta低め（分散重視、多様化）
        スコアの分散が小さい → beta高め（集中許容、勝者重視）

        計算式:
            beta = beta_base / (1 + sensitivity * normalized_std)
            normalized_std = score_std / mean(|score|)

        Args:
            strategy_scores: 戦略スコア履歴
            lookback_days: ルックバック日数（Noneでconfig値使用）

        Returns:
            最適なβ値（beta_min〜beta_max でクリップ）
        """
        lookback = lookback_days or self.config.lookback_days

        # DataFrame に統一
        if isinstance(strategy_scores, pd.Series):
            scores = strategy_scores.tail(lookback).values.flatten()
        else:
            scores = strategy_scores.tail(lookback).values.flatten()

        # NaN除去
        scores = scores[~np.isnan(scores)]

        if len(scores) < 10:
            logger.warning(
                f"Too few score samples ({len(scores)}), using default beta"
            )
            return self.config.beta_base

        # スコアの標準偏差
        score_std = np.std(scores)
        score_abs_mean = np.mean(np.abs(scores))

        if score_abs_mean < 1e-6:
            # スコアがほぼゼロの場合
            return self.config.beta_base

        # 正規化された標準偏差
        normalized_std = score_std / score_abs_mean

        # β計算: 分散が大きいほどβは小さく
        # beta = beta_base / (1 + sensitivity * normalized_std)
        beta = self.config.beta_base / (
            1 + self.config.beta_sensitivity * normalized_std
        )

        # クリップ
        beta = np.clip(beta, self.config.beta_min, self.config.beta_max)

        logger.debug(
            f"calculate_optimal_beta: std={score_std:.4f}, "
            f"norm_std={normalized_std:.4f}, beta={beta:.4f}"
        )

        return float(beta)

    def calculate_w_strategy_max(
        self,
        num_strategies: int,
        diversification_target: float | None = None,
    ) -> float:
        """最大戦略重みを計算

        目標有効戦略数から最大重みを逆算。
        有効戦略数 N_eff ≈ 1 / sum(w_i^2) の考え方を使用。

        簡易計算:
            w_max = 1 / min_effective_n
            例: 目標4戦略なら w_max = 0.25

        Args:
            num_strategies: 利用可能な戦略数
            diversification_target: 目標有効戦略数（Noneでconfig値使用）

        Returns:
            最大戦略重み（min_w_strategy_max〜max_w_strategy_max でクリップ）
        """
        target = diversification_target or self.config.diversification_target

        if num_strategies <= 0:
            return self.config.max_w_strategy_max

        # 目標有効戦略数は、実際の戦略数を超えない
        effective_target = min(target, num_strategies)

        if effective_target < 1:
            effective_target = 1

        # w_max = 1 / N_eff
        w_max = 1.0 / effective_target

        # クリップ
        w_max = np.clip(
            w_max, self.config.min_w_strategy_max, self.config.max_w_strategy_max
        )

        logger.debug(
            f"calculate_w_strategy_max: n={num_strategies}, "
            f"target={target}, w_max={w_max:.4f}"
        )

        return float(w_max)

    def calculate_score_threshold(
        self,
        strategy_scores: pd.DataFrame | pd.Series,
        percentile: float | None = None,
    ) -> float:
        """スコア閾値を計算

        過去スコア分布の下位 N パーセンタイルを閾値とする。
        これ以下のスコアの戦略は採用しない。

        Args:
            strategy_scores: 戦略スコア履歴
            percentile: 閾値パーセンタイル（Noneでconfig値使用）

        Returns:
            スコア閾値
        """
        pct = percentile or self.config.score_percentile

        # DataFrame に統一
        if isinstance(strategy_scores, pd.Series):
            scores = strategy_scores.values.flatten()
        else:
            scores = strategy_scores.values.flatten()

        # NaN除去
        scores = scores[~np.isnan(scores)]

        if len(scores) < 10:
            logger.warning(
                f"Too few score samples ({len(scores)}), using 0 as threshold"
            )
            return 0.0

        # パーセンタイル計算
        threshold = float(np.percentile(scores, pct))

        logger.debug(
            f"calculate_score_threshold: p{pct}={threshold:.4f}, "
            f"n_samples={len(scores)}"
        )

        return threshold

    def calculate_adaptive_params(
        self,
        strategy_scores: pd.DataFrame,
        current_volatility: float,
        vol_regime: str = "medium",
    ) -> DynamicWeighterParams:
        """ボラティリティレジームを考慮した適応的パラメータ計算

        Args:
            strategy_scores: 戦略スコア履歴
            current_volatility: 現在の年率ボラティリティ
            vol_regime: ボラティリティレジーム（low/medium/high）

        Returns:
            DynamicWeighterParams
        """
        # 基本パラメータを計算
        base_params = self.calculate_all(strategy_scores)

        # レジームに応じた調整
        regime_adjustments = {
            "low": {"beta_mult": 1.2, "w_max_mult": 1.1, "threshold_mult": 0.8},
            "medium": {"beta_mult": 1.0, "w_max_mult": 1.0, "threshold_mult": 1.0},
            "high": {"beta_mult": 0.7, "w_max_mult": 0.8, "threshold_mult": 1.2},
        }

        adj = regime_adjustments.get(vol_regime, regime_adjustments["medium"])

        # 調整適用
        adjusted_beta = np.clip(
            base_params.beta * adj["beta_mult"],
            self.config.beta_min,
            self.config.beta_max,
        )
        adjusted_w_max = np.clip(
            base_params.w_strategy_max * adj["w_max_mult"],
            self.config.min_w_strategy_max,
            self.config.max_w_strategy_max,
        )
        adjusted_threshold = base_params.score_threshold * adj["threshold_mult"]

        # メタデータ更新
        metadata = base_params.metadata.copy()
        metadata.update({
            "vol_regime": vol_regime,
            "current_volatility": current_volatility,
            "regime_adjustment": adj,
            "base_beta": base_params.beta,
            "base_w_max": base_params.w_strategy_max,
            "base_threshold": base_params.score_threshold,
        })

        return DynamicWeighterParams(
            beta=float(adjusted_beta),
            w_strategy_max=float(adjusted_w_max),
            score_threshold=float(adjusted_threshold),
            lookback_days=base_params.lookback_days,
            metadata=metadata,
        )


def calculate_optimal_beta(
    strategy_scores: pd.DataFrame | pd.Series,
    lookback_days: int = 252,
) -> float:
    """最適なβを計算（ショートカット関数）

    Args:
        strategy_scores: 戦略スコア履歴
        lookback_days: ルックバック日数

    Returns:
        最適なβ値
    """
    config = DynamicParamsConfig(lookback_days=lookback_days)
    calculator = DynamicWeighterParamsCalculator(config)
    return calculator.calculate_optimal_beta(strategy_scores)


def calculate_w_strategy_max(
    num_strategies: int,
    diversification_target: float = 4.0,
) -> float:
    """最大戦略重みを計算（ショートカット関数）

    Args:
        num_strategies: 戦略数
        diversification_target: 目標有効戦略数

    Returns:
        最大戦略重み
    """
    config = DynamicParamsConfig(diversification_target=diversification_target)
    calculator = DynamicWeighterParamsCalculator(config)
    return calculator.calculate_w_strategy_max(num_strategies, diversification_target)


def calculate_score_threshold(
    strategy_scores: pd.DataFrame | pd.Series,
    percentile: float = 10.0,
) -> float:
    """スコア閾値を計算（ショートカット関数）

    Args:
        strategy_scores: 戦略スコア履歴
        percentile: 閾値パーセンタイル

    Returns:
        スコア閾値
    """
    config = DynamicParamsConfig(score_percentile=percentile)
    calculator = DynamicWeighterParamsCalculator(config)
    return calculator.calculate_score_threshold(strategy_scores, percentile)
