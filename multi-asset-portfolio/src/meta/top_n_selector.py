"""
Top-N Strategy Selector Module - 上位N戦略選択

評価結果から上位N個の戦略を選択し、重み付けを行う。
CASHを特別なアセットとして扱い、全戦略のスコアがマイナスでも
CASHが上位に来れば採用される設計。

設計根拠:
- 要求.md §7: Strategyの採用と重み
- 戦略数の上限管理: 過剰な分散を防止
- CASH位置: リスクオフ時の安全弁

計算式:
    1. 全評価をスコア順にソート
    2. CASHを追加（スコア = cash_score）
    3. min_score以下を除外
    4. 上位N個を返す
    5. Softmaxで重み計算（temperature調整可能）
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopNSelectorConfig:
    """Top-N選択の設定

    Attributes:
        n: 選択する戦略数の上限
        cash_score: CASHの固定スコア
        min_score: 最小スコア閾値（これ以下は除外）
        softmax_temperature: Softmax温度パラメータ（低いほど勝者総取り）
        include_cash: CASHを候補に含めるか
        cash_symbol: CASHのシンボル名
        alpha_filter_enabled: アルファスコアフィルタを有効化
        alpha_percentile: アルファフィルタのパーセンタイル (0-1)
    """

    n: int = 10
    cash_score: float = 0.0
    min_score: float = -999.0
    softmax_temperature: float = 1.0
    include_cash: bool = True
    cash_symbol: str = "CASH"
    alpha_filter_enabled: bool = False
    alpha_percentile: float = 0.5

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0")
        if not 0 < self.alpha_percentile <= 1.0:
            raise ValueError("alpha_percentile must be in (0, 1]")


@dataclass
class StrategyScore:
    """戦略スコア情報

    Attributes:
        asset: アセット名（AAPL, CASH等）
        signal_name: シグナル名（momentum_return等）
        score: スコア（Sharpe等）
        metrics: 詳細指標
        is_cash: CASHかどうか
    """

    asset: str
    signal_name: str
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    is_cash: bool = False

    @property
    def strategy_key(self) -> str:
        """戦略の一意キー"""
        return f"{self.asset}:{self.signal_name}"

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset": self.asset,
            "signal_name": self.signal_name,
            "score": self.score,
            "metrics": self.metrics,
            "is_cash": self.is_cash,
            "strategy_key": self.strategy_key,
        }


@dataclass
class TopNSelectionResult:
    """Top-N選択結果

    Attributes:
        selected: 選択された戦略リスト
        weights: アセット別の最終重み
        strategy_weights: 戦略別の重み（asset:signal -> weight）
        excluded_count: 除外された戦略数
        cash_selected: CASHが選択されたか
        cash_weight: CASHの重み
        metadata: 追加メタデータ
    """

    selected: list[StrategyScore] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    excluded_count: int = 0
    cash_selected: bool = False
    cash_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "selected": [s.to_dict() for s in self.selected],
            "weights": self.weights,
            "strategy_weights": self.strategy_weights,
            "excluded_count": self.excluded_count,
            "cash_selected": self.cash_selected,
            "cash_weight": self.cash_weight,
            "metadata": self.metadata,
        }


class TopNSelector:
    """Top-N戦略選択クラス

    評価結果から上位N個の戦略を選択し、Softmaxで重み付けを行う。
    CASHを特別なアセットとして扱い、リスクオフ時の安全弁とする。

    Usage:
        config = TopNSelectorConfig(n=10, cash_score=0.0)
        selector = TopNSelector(config)

        evaluations = {
            "AAPL": {"momentum": {"score": 1.5, "sharpe": 1.5, ...}},
            "GOOGL": {"reversal": {"score": 0.8, "sharpe": 0.8, ...}},
        }

        result = selector.select(evaluations)
        print(result.weights)  # {"AAPL": 0.6, "GOOGL": 0.3, "CASH": 0.1}
    """

    def __init__(
        self,
        n: int = 10,
        cash_score: float = 0.0,
        min_score: float = -999.0,
        softmax_temperature: float = 1.0,
        config: TopNSelectorConfig | None = None,
    ) -> None:
        """初期化

        Args:
            n: 選択する戦略数の上限
            cash_score: CASHの固定スコア
            min_score: 最小スコア閾値
            softmax_temperature: Softmax温度パラメータ
            config: 設定（指定時は個別パラメータより優先）
        """
        if config is not None:
            self.config = config
        else:
            self.config = TopNSelectorConfig(
                n=n,
                cash_score=cash_score,
                min_score=min_score,
                softmax_temperature=softmax_temperature,
            )

    @property
    def n(self) -> int:
        """選択数上限"""
        return self.config.n

    @property
    def cash_score(self) -> float:
        """CASHスコア"""
        return self.config.cash_score

    @property
    def min_score(self) -> float:
        """最小スコア閾値"""
        return self.config.min_score

    @property
    def temperature(self) -> float:
        """Softmax温度"""
        return self.config.softmax_temperature

    def select(
        self,
        evaluations: dict[str, dict[str, dict[str, Any]]],
        alpha_scores: dict[str, float] | None = None,
    ) -> TopNSelectionResult:
        """上位N個の戦略を選択

        Args:
            evaluations: アセット -> シグナル名 -> 評価結果 の辞書
                         評価結果には "score" キーが必須
            alpha_scores: アルファスコア辞書（アセット -> スコア）
                         アルファフィルタが有効な場合に使用

        Returns:
            選択結果
        """
        # 0. アルファスコアでプレフィルタ (NEW)
        if alpha_scores and self.config.alpha_filter_enabled:
            evaluations = self._apply_alpha_filter(evaluations, alpha_scores)

        # 1. 全評価をStrategyScoreに変換
        all_scores: list[StrategyScore] = []

        for asset, signals in evaluations.items():
            for signal_name, eval_data in signals.items():
                score = eval_data.get("score", 0.0)
                metrics = {k: v for k, v in eval_data.items() if k != "score"}

                strategy_score = StrategyScore(
                    asset=asset,
                    signal_name=signal_name,
                    score=score,
                    metrics=metrics,
                    is_cash=False,
                )
                all_scores.append(strategy_score)

        # 2. CASHを追加
        if self.config.include_cash:
            cash_strategy = StrategyScore(
                asset=self.config.cash_symbol,
                signal_name="hold",
                score=self.cash_score,
                metrics={"type": "cash", "risk_free": True},
                is_cash=True,
            )
            all_scores.append(cash_strategy)

        # 3. スコア順にソート（降順）
        all_scores.sort(key=lambda x: x.score, reverse=True)

        # 4. min_score以下を除外
        filtered_scores = [s for s in all_scores if s.score > self.min_score]
        excluded_count = len(all_scores) - len(filtered_scores)

        if excluded_count > 0:
            logger.debug(
                "Excluded %d strategies below min_score (%.3f)",
                excluded_count,
                self.min_score,
            )

        # 5. 上位N個を選択
        selected = filtered_scores[: self.n]

        if not selected:
            logger.warning("No strategies selected after filtering")
            return TopNSelectionResult(
                excluded_count=excluded_count,
                metadata={
                    "n": self.n,
                    "min_score": self.min_score,
                    "total_candidates": len(all_scores),
                },
            )

        # 6. 重み計算
        weights = self.calculate_weights(selected)

        # 7. アセット別に重み合算
        asset_weights = self._aggregate_weights_by_asset(selected, weights)

        # CASH情報
        cash_selected = any(s.is_cash for s in selected)
        cash_weight = asset_weights.get(self.config.cash_symbol, 0.0)

        result = TopNSelectionResult(
            selected=selected,
            weights=asset_weights,
            strategy_weights=weights,
            excluded_count=excluded_count,
            cash_selected=cash_selected,
            cash_weight=cash_weight,
            metadata={
                "n": self.n,
                "cash_score": self.cash_score,
                "min_score": self.min_score,
                "temperature": self.temperature,
                "total_candidates": len(all_scores),
                "selected_count": len(selected),
            },
        )

        logger.info(
            "Selected %d/%d strategies, CASH=%s (%.2f%%)",
            len(selected),
            len(all_scores),
            cash_selected,
            cash_weight * 100,
        )

        return result

    def select_from_evaluation_results(
        self,
        evaluation_results: list[Any],
    ) -> TopNSelectionResult:
        """StrategyEvaluationResultのリストから選択

        Args:
            evaluation_results: StrategyEvaluationResultのリスト
                               （src.strategy.evaluator から）

        Returns:
            選択結果
        """
        evaluations: dict[str, dict[str, dict[str, Any]]] = {}

        for result in evaluation_results:
            asset_id = getattr(result, "asset_id", "unknown")
            strategy_id = getattr(result, "strategy_id", "unknown")

            if asset_id not in evaluations:
                evaluations[asset_id] = {}

            # スコアとメトリクスを抽出
            score = getattr(result, "score", 0.0)
            metrics_obj = getattr(result, "metrics", None)

            metrics: dict[str, Any] = {}
            if metrics_obj is not None:
                metrics = {
                    "sharpe_ratio": getattr(metrics_obj, "sharpe_ratio", 0.0),
                    "max_drawdown_pct": getattr(metrics_obj, "max_drawdown_pct", 0.0),
                    "win_rate_pct": getattr(metrics_obj, "win_rate_pct", 0.0),
                    "profit_factor": getattr(metrics_obj, "profit_factor", 0.0),
                    "trade_count": getattr(metrics_obj, "trade_count", 0),
                    "expected_value": getattr(metrics_obj, "expected_value", 0.0),
                }

            evaluations[asset_id][strategy_id] = {
                "score": score,
                **metrics,
            }

        return self.select(evaluations)

    def calculate_weights(
        self,
        selected: list[StrategyScore],
    ) -> dict[str, float]:
        """選択された戦略の重みを計算（Softmax）

        Args:
            selected: 選択された戦略リスト

        Returns:
            strategy_key -> weight の辞書
        """
        if not selected:
            return {}

        # スコアを抽出
        scores = [s.score for s in selected]

        # Softmax計算（数値安定性のためmax減算）
        max_score = max(scores)
        exp_scores = [
            math.exp((score - max_score) / self.temperature)
            for score in scores
        ]
        sum_exp = sum(exp_scores)

        if sum_exp == 0:
            # 全てのスコアが-infの場合、等重み
            n = len(selected)
            return {s.strategy_key: 1.0 / n for s in selected}

        weights = {
            selected[i].strategy_key: exp_scores[i] / sum_exp
            for i in range(len(selected))
        }

        return weights

    def _aggregate_weights_by_asset(
        self,
        selected: list[StrategyScore],
        strategy_weights: dict[str, float],
    ) -> dict[str, float]:
        """戦略重みをアセット別に合算

        同一アセットの複数戦略がある場合、重みを合算する。

        Args:
            selected: 選択された戦略リスト
            strategy_weights: 戦略キー -> 重み

        Returns:
            アセット -> 重み の辞書
        """
        asset_weights: dict[str, float] = {}

        for strategy in selected:
            weight = strategy_weights.get(strategy.strategy_key, 0.0)
            if strategy.asset in asset_weights:
                asset_weights[strategy.asset] += weight
            else:
                asset_weights[strategy.asset] = weight

        return asset_weights

    def _apply_alpha_filter(
        self,
        evaluations: dict[str, dict[str, dict[str, Any]]],
        alpha_scores: dict[str, float],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """アルファスコアでプレフィルタ

        アルファスコアが閾値以上のアセットのみを残す。

        Args:
            evaluations: アセット -> シグナル名 -> 評価結果 の辞書
            alpha_scores: アルファスコア辞書（アセット -> スコア）

        Returns:
            フィルタ後の評価辞書
        """
        if not alpha_scores:
            return evaluations

        # アルファスコアの閾値を計算（パーセンタイル）
        alpha_values = list(alpha_scores.values())
        if not alpha_values:
            return evaluations

        # 上位 alpha_percentile を選択するための閾値
        # percentile=0.5 なら上位50% = 下位50%パーセンタイル以上
        threshold = np.percentile(
            alpha_values,
            (1 - self.config.alpha_percentile) * 100,
        )

        # フィルタ適用
        filtered_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
        for asset, signals in evaluations.items():
            asset_alpha = alpha_scores.get(asset, float("-inf"))
            if asset_alpha >= threshold:
                filtered_evaluations[asset] = signals

        n_before = len(evaluations)
        n_after = len(filtered_evaluations)

        logger.info(
            "Alpha filter applied: %d -> %d assets (threshold=%.4f, percentile=%.2f)",
            n_before,
            n_after,
            threshold,
            self.config.alpha_percentile,
        )

        return filtered_evaluations

    def get_top_strategies_for_asset(
        self,
        evaluations: dict[str, dict[str, dict[str, Any]]],
        asset: str,
        top_k: int = 3,
    ) -> list[StrategyScore]:
        """特定アセットの上位K戦略を取得

        Args:
            evaluations: 評価辞書
            asset: アセット名
            top_k: 取得数

        Returns:
            上位K戦略のリスト
        """
        if asset not in evaluations:
            return []

        asset_scores: list[StrategyScore] = []
        for signal_name, eval_data in evaluations[asset].items():
            score = eval_data.get("score", 0.0)
            if score > self.min_score:
                strategy_score = StrategyScore(
                    asset=asset,
                    signal_name=signal_name,
                    score=score,
                    metrics={k: v for k, v in eval_data.items() if k != "score"},
                )
                asset_scores.append(strategy_score)

        asset_scores.sort(key=lambda x: x.score, reverse=True)
        return asset_scores[:top_k]


def create_selector_from_settings() -> TopNSelector:
    """グローバル設定からSelectorを生成

    Returns:
        設定済みのTopNSelector
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()
        strategy_weighting = settings.strategy_weighting

        # アルファランキング設定を取得
        alpha_ranking = getattr(settings, "alpha_ranking", None)
        alpha_filter_enabled = False
        alpha_percentile = 0.5
        if alpha_ranking is not None:
            alpha_filter_enabled = getattr(alpha_ranking, "enabled", False)
            filtering = getattr(alpha_ranking, "filtering", None)
            if filtering is not None:
                alpha_percentile = getattr(filtering, "alpha_percentile", 0.5)

        config = TopNSelectorConfig(
            n=getattr(strategy_weighting, "max_strategies", 10),
            cash_score=getattr(settings, "cash_score", 0.0),
            min_score=getattr(strategy_weighting, "score_threshold", -999.0),
            softmax_temperature=1.0 / getattr(strategy_weighting, "beta", 2.0),
            alpha_filter_enabled=alpha_filter_enabled,
            alpha_percentile=alpha_percentile,
        )
        return TopNSelector(config=config)
    except (ImportError, AttributeError) as e:
        logger.warning("Settings not available, using defaults: %s", e)
        return TopNSelector()
