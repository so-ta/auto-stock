"""
Alpha Scorer Module - アルファスコア計算

期待値（アルファ）に基づいて銘柄をスコアリングし、選定・配分に活用する。

アルファスコア計算式:
    α = w₁×モメンタム + w₂×シグナル品質 + w₃×期待リターン

設計目的:
- シャープレシオを維持しつつリターンを最大化
- 期待値の高い銘柄を優先的に選択
- リスク調整後のリターン向上

使用方法:
    from src.meta.alpha_scorer import AlphaScorer, AlphaScoringConfig

    scorer = AlphaScorer(AlphaScoringConfig(
        momentum_weight=0.3,
        quality_weight=0.3,
        expected_return_weight=0.4,
    ))
    alpha_scores = scorer.compute(returns_df, signal_scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaScoringConfig:
    """アルファスコアリング設定

    Attributes:
        enabled: アルファスコアリングを有効化
        lookback_days: モメンタム計算のルックバック日数
        momentum_weight: モメンタムの重み (w₁)
        quality_weight: シグナル品質の重み (w₂)
        expected_return_weight: 期待リターンの重み (w₃)
        annual_scale: 年率スケール（期待リターン変換用）
        min_observations: 最小観測数
        benchmark_ticker: ベンチマークティッカー（相対アルファ計算用）
    """

    enabled: bool = True
    lookback_days: int = 60
    momentum_weight: float = 0.3
    quality_weight: float = 0.3
    expected_return_weight: float = 0.4
    annual_scale: float = 0.15  # 年率15%スケール
    min_observations: int = 20
    benchmark_ticker: str = "SPY"

    def __post_init__(self) -> None:
        """バリデーション"""
        total_weight = self.momentum_weight + self.quality_weight + self.expected_return_weight
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        if self.lookback_days < 5:
            raise ValueError("lookback_days must be >= 5")
        if self.min_observations < 5:
            raise ValueError("min_observations must be >= 5")


@dataclass(frozen=True)
class AlphaFilterConfig:
    """アルファフィルタリング設定

    Attributes:
        method: フィルタ方法 ("percentile" or "threshold")
        alpha_percentile: 上位パーセンタイル (0-1、例: 0.5 = 上位50%)
        alpha_threshold: 固定閾値（thresholdメソッド用）
        min_assets: 最小アセット数（これ以下にはならない）
    """

    method: str = "percentile"  # "percentile" or "threshold"
    alpha_percentile: float = 0.5  # 上位50%を選択
    alpha_threshold: float = 0.0  # threshold方式の閾値
    min_assets: int = 5

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.method not in ("percentile", "threshold"):
            raise ValueError("method must be 'percentile' or 'threshold'")
        if not 0 < self.alpha_percentile <= 1.0:
            raise ValueError("alpha_percentile must be in (0, 1]")
        if self.min_assets < 1:
            raise ValueError("min_assets must be >= 1")


@dataclass
class AlphaScore:
    """個別アセットのアルファスコア

    Attributes:
        symbol: アセットシンボル
        alpha_score: 総合アルファスコア
        momentum_score: モメンタムスコア（クロスセクションランク）
        quality_score: シグナル品質スコア
        expected_return: 期待リターン
        rank: クロスセクションランク（1が最高）
        metadata: 追加メタデータ
    """

    symbol: str
    alpha_score: float
    momentum_score: float = 0.0
    quality_score: float = 0.0
    expected_return: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "alpha_score": self.alpha_score,
            "momentum_score": self.momentum_score,
            "quality_score": self.quality_score,
            "expected_return": self.expected_return,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class AlphaRankingResult:
    """アルファランキング結果

    Attributes:
        scores: シンボル -> AlphaScore の辞書
        rankings: ランク順の銘柄リスト
        filtered_symbols: フィルタ後の銘柄リスト
        metadata: 追加メタデータ
    """

    scores: dict[str, AlphaScore] = field(default_factory=dict)
    rankings: list[str] = field(default_factory=list)
    filtered_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_alpha_dict(self) -> dict[str, float]:
        """シンボル -> アルファスコア の辞書を返す"""
        return {symbol: score.alpha_score for symbol, score in self.scores.items()}

    def get_filtered_alpha_dict(self) -> dict[str, float]:
        """フィルタ後のシンボル -> アルファスコア の辞書を返す"""
        return {
            symbol: self.scores[symbol].alpha_score
            for symbol in self.filtered_symbols
            if symbol in self.scores
        }

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "rankings": self.rankings,
            "filtered_symbols": self.filtered_symbols,
            "metadata": self.metadata,
        }


class AlphaScorer:
    """アルファスコアラー

    銘柄のアルファスコアを計算し、期待値に基づく選択をサポート。

    Usage:
        scorer = AlphaScorer(AlphaScoringConfig())
        result = scorer.compute(
            returns_df=returns_df,
            signal_scores=signal_scores,
        )
        alpha_scores = result.get_alpha_dict()
    """

    def __init__(
        self,
        config: AlphaScoringConfig | None = None,
        filter_config: AlphaFilterConfig | None = None,
    ) -> None:
        """初期化

        Args:
            config: スコアリング設定
            filter_config: フィルタリング設定
        """
        self.config = config or AlphaScoringConfig()
        self.filter_config = filter_config or AlphaFilterConfig()

    def compute(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float] | None = None,
        expected_returns: dict[str, float] | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> AlphaRankingResult:
        """アルファスコアを計算

        Args:
            returns_df: リターンデータ（列=アセット、行=日付）
            signal_scores: シグナルスコア辞書（シンボル -> スコア）
            expected_returns: 期待リターン辞書（シンボル -> 期待リターン）
            benchmark_returns: ベンチマークリターン（オプション）

        Returns:
            AlphaRankingResult: アルファランキング結果
        """
        if not self.config.enabled:
            logger.info("Alpha scoring disabled, returning empty result")
            return AlphaRankingResult(metadata={"enabled": False})

        if returns_df.empty:
            logger.warning("Empty returns data for alpha scoring")
            return AlphaRankingResult(metadata={"error": "empty_returns"})

        symbols = list(returns_df.columns)

        # Step 1: モメンタムスコア計算（クロスセクションランク）
        momentum_scores = self._compute_momentum_scores(returns_df)

        # Step 2: シグナル品質スコア計算
        quality_scores = self._compute_quality_scores(
            returns_df, signal_scores
        )

        # Step 3: 期待リターンの取得/計算
        exp_returns = self._get_expected_returns(
            returns_df, expected_returns, benchmark_returns
        )

        # Step 4: 総合アルファスコア計算
        alpha_scores: dict[str, AlphaScore] = {}
        raw_alphas: list[tuple[str, float]] = []

        for symbol in symbols:
            mom = momentum_scores.get(symbol, 0.0)
            qual = quality_scores.get(symbol, 0.0)
            exp_ret = exp_returns.get(symbol, 0.0)

            # 正規化（クロスセクション）
            alpha = (
                self.config.momentum_weight * mom +
                self.config.quality_weight * qual +
                self.config.expected_return_weight * exp_ret
            )

            alpha_scores[symbol] = AlphaScore(
                symbol=symbol,
                alpha_score=alpha,
                momentum_score=mom,
                quality_score=qual,
                expected_return=exp_ret,
            )
            raw_alphas.append((symbol, alpha))

        # Step 5: ランキング
        sorted_alphas = sorted(raw_alphas, key=lambda x: x[1], reverse=True)
        rankings = [s for s, _ in sorted_alphas]

        for rank, symbol in enumerate(rankings, 1):
            alpha_scores[symbol].rank = rank

        # Step 6: フィルタリング
        filtered_symbols = self._apply_filter(alpha_scores, rankings)

        logger.info(
            "Alpha scoring completed",
            extra={
                "n_symbols": len(symbols),
                "n_filtered": len(filtered_symbols),
                "top_alpha": sorted_alphas[0] if sorted_alphas else None,
            },
        )

        return AlphaRankingResult(
            scores=alpha_scores,
            rankings=rankings,
            filtered_symbols=filtered_symbols,
            metadata={
                "config": {
                    "momentum_weight": self.config.momentum_weight,
                    "quality_weight": self.config.quality_weight,
                    "expected_return_weight": self.config.expected_return_weight,
                    "lookback_days": self.config.lookback_days,
                },
                "filter_config": {
                    "method": self.filter_config.method,
                    "percentile": self.filter_config.alpha_percentile,
                },
            },
        )

    def _compute_momentum_scores(
        self,
        returns_df: pd.DataFrame,
    ) -> dict[str, float]:
        """モメンタムスコアを計算（クロスセクションランク）

        Args:
            returns_df: リターンデータ

        Returns:
            シンボル -> モメンタムスコア の辞書（-1 to +1 正規化）
        """
        lookback = min(self.config.lookback_days, len(returns_df))
        if lookback < self.config.min_observations:
            return {}

        # ルックバック期間のリターン
        recent_returns = returns_df.tail(lookback)
        cumulative_returns = (1 + recent_returns).prod() - 1

        # クロスセクションランク（0-1にスケール後、-1 to +1に変換）
        n = len(cumulative_returns)
        if n < 2:
            return {s: 0.0 for s in returns_df.columns}

        ranks = cumulative_returns.rank(method="average")
        # rank/n -> [1/n, 1] -> 正規化して [-1, +1]
        normalized = 2 * (ranks / n) - 1

        return normalized.to_dict()

    def _compute_quality_scores(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float] | None,
    ) -> dict[str, float]:
        """シグナル品質スコアを計算

        シグナルスコアと将来リターンの相関を品質指標として使用。
        シグナルスコアが提供されない場合は、リターンの安定性を使用。

        Args:
            returns_df: リターンデータ
            signal_scores: シグナルスコア辞書

        Returns:
            シンボル -> 品質スコア の辞書
        """
        symbols = list(returns_df.columns)

        if signal_scores:
            # シグナルスコアが提供された場合、クロスセクション正規化
            signal_values = np.array([signal_scores.get(s, 0.0) for s in symbols])
            if signal_values.std() > 1e-10:
                z_scores = (signal_values - signal_values.mean()) / signal_values.std()
                normalized = np.clip(z_scores / 3, -1, 1)  # Z-scoreを-1,+1にクリップ
                return dict(zip(symbols, normalized))
            return {s: 0.0 for s in symbols}

        # シグナルスコアがない場合、リターンの安定性（低ボラ=高品質）を使用
        lookback = min(self.config.lookback_days, len(returns_df))
        if lookback < self.config.min_observations:
            return {s: 0.0 for s in symbols}

        recent_returns = returns_df.tail(lookback)
        volatilities = recent_returns.std()

        # 低ボラ = 高品質（逆転してランク）
        if volatilities.std() > 1e-10:
            inv_vol = 1 / (volatilities + 1e-10)
            z_scores = (inv_vol - inv_vol.mean()) / inv_vol.std()
            normalized = np.clip(z_scores / 3, -1, 1)
            return dict(zip(symbols, normalized))

        return {s: 0.0 for s in symbols}

    def _get_expected_returns(
        self,
        returns_df: pd.DataFrame,
        expected_returns: dict[str, float] | None,
        benchmark_returns: pd.Series | None,
    ) -> dict[str, float]:
        """期待リターンを取得/計算

        期待リターンが提供されない場合は、過去リターンから推定。

        Args:
            returns_df: リターンデータ
            expected_returns: 期待リターン辞書
            benchmark_returns: ベンチマークリターン

        Returns:
            シンボル -> 期待リターン の辞書（正規化済み）
        """
        symbols = list(returns_df.columns)

        if expected_returns:
            # 提供された期待リターンをクロスセクション正規化
            values = np.array([expected_returns.get(s, 0.0) for s in symbols])
            if values.std() > 1e-10:
                z_scores = (values - values.mean()) / values.std()
                normalized = np.clip(z_scores / 3, -1, 1)
                return dict(zip(symbols, normalized))
            return {s: 0.0 for s in symbols}

        # 過去リターンから期待リターンを推定
        lookback = min(self.config.lookback_days, len(returns_df))
        if lookback < self.config.min_observations:
            return {s: 0.0 for s in symbols}

        recent_returns = returns_df.tail(lookback)
        mean_returns = recent_returns.mean()

        # ベンチマーク対比（アルファ）を計算
        if benchmark_returns is not None:
            benchmark_recent = benchmark_returns.tail(lookback)
            if len(benchmark_recent) == len(recent_returns):
                benchmark_mean = benchmark_recent.mean()
                alphas = mean_returns - benchmark_mean
            else:
                alphas = mean_returns
        else:
            alphas = mean_returns

        # クロスセクション正規化
        if alphas.std() > 1e-10:
            z_scores = (alphas - alphas.mean()) / alphas.std()
            normalized = np.clip(z_scores / 3, -1, 1)
            return dict(zip(symbols, normalized))

        return {s: 0.0 for s in symbols}

    def _apply_filter(
        self,
        alpha_scores: dict[str, AlphaScore],
        rankings: list[str],
    ) -> list[str]:
        """アルファフィルタを適用

        Args:
            alpha_scores: アルファスコア辞書
            rankings: ランク順の銘柄リスト

        Returns:
            フィルタ後の銘柄リスト
        """
        if not rankings:
            return []

        n_total = len(rankings)

        if self.filter_config.method == "percentile":
            # 上位パーセンタイルを選択
            n_select = max(
                self.filter_config.min_assets,
                int(n_total * self.filter_config.alpha_percentile),
            )
            return rankings[:n_select]

        elif self.filter_config.method == "threshold":
            # 閾値以上を選択
            filtered = [
                symbol for symbol in rankings
                if alpha_scores[symbol].alpha_score >= self.filter_config.alpha_threshold
            ]
            # 最小数を保証
            if len(filtered) < self.filter_config.min_assets:
                return rankings[:self.filter_config.min_assets]
            return filtered

        return rankings


def create_alpha_scorer_from_settings(settings: Any = None) -> AlphaScorer:
    """設定からAlphaScorerを作成

    Args:
        settings: 設定オブジェクト

    Returns:
        AlphaScorer
    """
    if settings is None:
        return AlphaScorer()

    # alpha_ranking設定を取得
    alpha_config = getattr(settings, "alpha_ranking", None)
    if alpha_config is None:
        return AlphaScorer()

    scoring = getattr(alpha_config, "scoring", None)
    filtering = getattr(alpha_config, "filtering", None)

    scoring_config = AlphaScoringConfig(
        enabled=getattr(alpha_config, "enabled", True),
        lookback_days=getattr(scoring, "lookback_days", 60) if scoring else 60,
        momentum_weight=getattr(scoring, "momentum_weight", 0.3) if scoring else 0.3,
        quality_weight=getattr(scoring, "quality_weight", 0.3) if scoring else 0.3,
        expected_return_weight=getattr(scoring, "expected_return_weight", 0.4) if scoring else 0.4,
    )

    filter_config = AlphaFilterConfig(
        method=getattr(filtering, "method", "percentile") if filtering else "percentile",
        alpha_percentile=getattr(filtering, "alpha_percentile", 0.5) if filtering else 0.5,
    )

    return AlphaScorer(config=scoring_config, filter_config=filter_config)


def quick_alpha_ranking(
    returns_df: pd.DataFrame,
    top_percentile: float = 0.5,
    lookback_days: int = 60,
) -> list[str]:
    """簡易アルファランキング（便利関数）

    Args:
        returns_df: リターンデータ
        top_percentile: 選択する上位パーセンタイル
        lookback_days: ルックバック日数

    Returns:
        フィルタ後の銘柄リスト
    """
    scoring_config = AlphaScoringConfig(lookback_days=lookback_days)
    filter_config = AlphaFilterConfig(alpha_percentile=top_percentile)
    scorer = AlphaScorer(config=scoring_config, filter_config=filter_config)
    result = scorer.compute(returns_df)
    return result.filtered_symbols
