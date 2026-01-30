"""
Enhanced Sector Rotation - 景気サイクル適応型セクターローテーション

景気サイクルに応じた動的セクター配分を実現。
静的配分と比較して Sharpe +0.05~0.1、超過リターン +2~3% を期待。

主要コンポーネント:
1. EconomicCycleDetector: 景気サイクル判定
2. SectorMomentumCalculator: セクターモメンタム計算
3. RelativeStrengthCalculator: 相対強度スコア計算
4. EnhancedSectorRotation: 統合クラス

景気サイクルとセクター:
- early_expansion (初期拡大): 金融、資本財、素材
- mid_expansion (中期拡大): テック、一般消費財、通信
- late_expansion (後期拡大): エネルギー、素材、資本財
- contraction (縮小): 公益、生活必需品、ヘルスケア

Usage:
    from src.signals.sector_rotation_v2 import EnhancedSectorRotation

    rotation = EnhancedSectorRotation()
    weights = rotation.get_sector_weights(
        sector_prices=sector_price_df,
        market_data={'spy': spy_prices, 'vix': vix_series}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry
from .sector import SECTOR_ETFS, SECTOR_ETF_TICKERS

logger = logging.getLogger(__name__)


class EconomicCycle(str, Enum):
    """景気サイクルの定義."""

    EARLY_EXPANSION = "early_expansion"  # 初期拡大
    MID_EXPANSION = "mid_expansion"      # 中期拡大
    LATE_EXPANSION = "late_expansion"    # 後期拡大
    CONTRACTION = "contraction"          # 縮小/リセッション


# 景気サイクルとセクターETFのマッピング
ECONOMIC_CYCLE_SECTORS: dict[EconomicCycle, list[str]] = {
    EconomicCycle.EARLY_EXPANSION: ["XLF", "XLI", "XLB"],      # 金融、資本財、素材
    EconomicCycle.MID_EXPANSION: ["XLK", "XLY", "XLC"],        # テック、一般消費財、通信
    EconomicCycle.LATE_EXPANSION: ["XLE", "XLB", "XLI"],       # エネルギー、素材、資本財
    EconomicCycle.CONTRACTION: ["XLU", "XLP", "XLV"],          # 公益、生活必需品、ヘルスケア
}

# セクターの景気感応度（ベータ係数的な概念）
SECTOR_CYCLICALITY: dict[str, float] = {
    "XLK": 1.2,   # テクノロジー: 高感応
    "XLF": 1.1,   # 金融: 高感応
    "XLY": 1.3,   # 一般消費財: 高感応
    "XLC": 1.0,   # 通信: 中程度
    "XLI": 1.1,   # 資本財: 高感応
    "XLB": 1.2,   # 素材: 高感応
    "XLE": 1.4,   # エネルギー: 非常に高感応
    "XLRE": 0.9,  # 不動産: 中程度
    "XLV": 0.7,   # ヘルスケア: 低感応（ディフェンシブ）
    "XLP": 0.5,   # 生活必需品: 低感応（ディフェンシブ）
    "XLU": 0.4,   # 公益: 低感応（ディフェンシブ）
}


@dataclass
class CycleDetectionResult:
    """景気サイクル判定結果."""

    cycle: EconomicCycle
    confidence: float  # 0-1
    indicators: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorScore:
    """セクタースコア."""

    ticker: str
    momentum_score: float  # -1 to 1
    relative_strength_score: float  # -1 to 1
    cycle_alignment_score: float  # 0 to 1
    combined_score: float  # -1 to 1
    weight: float  # 0 to 1 (最終ウェイト)


@dataclass
class RotationResult:
    """セクターローテーション結果."""

    cycle: EconomicCycle
    cycle_confidence: float
    sector_scores: list[SectorScore]
    weights: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class EconomicCycleDetector:
    """
    景気サイクル検出器.

    複数の指標を組み合わせて景気サイクルを判定:
    1. 株式市場モメンタム（SPY）
    2. VIXレベル・変化
    3. セクター間相対パフォーマンス
    4. イールドカーブ（オプション）
    """

    def __init__(
        self,
        momentum_short: int = 21,
        momentum_long: int = 126,
        vix_threshold_low: float = 15.0,
        vix_threshold_high: float = 25.0,
        cyclical_vs_defensive_threshold: float = 0.02,
    ):
        """
        Args:
            momentum_short: 短期モメンタム期間（日）
            momentum_long: 長期モメンタム期間（日）
            vix_threshold_low: VIX低水準閾値
            vix_threshold_high: VIX高水準閾値
            cyclical_vs_defensive_threshold: 景気敏感/ディフェンシブ相対閾値
        """
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long
        self.vix_threshold_low = vix_threshold_low
        self.vix_threshold_high = vix_threshold_high
        self.cyclical_vs_defensive_threshold = cyclical_vs_defensive_threshold

    def detect(
        self,
        spy_prices: pd.Series,
        vix_series: pd.Series | None = None,
        sector_prices: pd.DataFrame | None = None,
    ) -> CycleDetectionResult:
        """
        景気サイクルを検出.

        Args:
            spy_prices: S&P500価格系列
            vix_series: VIX系列（オプション）
            sector_prices: セクターETF価格（オプション）

        Returns:
            CycleDetectionResult
        """
        indicators: dict[str, float] = {}
        scores: dict[EconomicCycle, float] = {c: 0.0 for c in EconomicCycle}

        # 1. 市場モメンタム分析
        mom_short = self._calculate_momentum(spy_prices, self.momentum_short)
        mom_long = self._calculate_momentum(spy_prices, self.momentum_long)

        indicators["momentum_short"] = mom_short
        indicators["momentum_long"] = mom_long

        # モメンタムベースのサイクル判定
        if mom_long > 0 and mom_short > 0:
            if mom_short > mom_long:
                scores[EconomicCycle.EARLY_EXPANSION] += 0.3
            else:
                scores[EconomicCycle.MID_EXPANSION] += 0.3
        elif mom_long > 0 and mom_short <= 0:
            scores[EconomicCycle.LATE_EXPANSION] += 0.3
        else:
            scores[EconomicCycle.CONTRACTION] += 0.3

        # 2. VIX分析
        if vix_series is not None and len(vix_series) > 0:
            current_vix = float(vix_series.iloc[-1])
            vix_sma = float(vix_series.rolling(21).mean().iloc[-1])

            indicators["vix_current"] = current_vix
            indicators["vix_sma"] = vix_sma

            if current_vix < self.vix_threshold_low:
                scores[EconomicCycle.MID_EXPANSION] += 0.2
            elif current_vix > self.vix_threshold_high:
                scores[EconomicCycle.CONTRACTION] += 0.2
            elif current_vix < vix_sma:
                scores[EconomicCycle.EARLY_EXPANSION] += 0.2
            else:
                scores[EconomicCycle.LATE_EXPANSION] += 0.2

        # 3. セクター相対パフォーマンス分析
        if sector_prices is not None and len(sector_prices) > self.momentum_short:
            cyclical_perf = self._calculate_cyclical_performance(sector_prices)
            indicators["cyclical_vs_defensive"] = cyclical_perf

            if cyclical_perf > self.cyclical_vs_defensive_threshold:
                scores[EconomicCycle.EARLY_EXPANSION] += 0.25
                scores[EconomicCycle.MID_EXPANSION] += 0.25
            elif cyclical_perf < -self.cyclical_vs_defensive_threshold:
                scores[EconomicCycle.CONTRACTION] += 0.25
                scores[EconomicCycle.LATE_EXPANSION] += 0.25

        # 最大スコアのサイクルを選択
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {c: s / total_score for c, s in scores.items()}

        detected_cycle = max(scores, key=scores.get)
        confidence = scores[detected_cycle]

        return CycleDetectionResult(
            cycle=detected_cycle,
            confidence=confidence,
            indicators=indicators,
            metadata={
                "all_scores": {c.value: s for c, s in scores.items()},
            },
        )

    def _calculate_momentum(self, prices: pd.Series, lookback: int) -> float:
        """モメンタムを計算."""
        if len(prices) < lookback + 1:
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[-lookback - 1]

        if past == 0:
            return 0.0

        return float((current - past) / past)

    def _calculate_cyclical_performance(self, sector_prices: pd.DataFrame) -> float:
        """景気敏感セクター vs ディフェンシブセクターの相対パフォーマンス."""
        cyclical_etfs = ["XLK", "XLF", "XLY", "XLI", "XLB"]
        defensive_etfs = ["XLU", "XLP", "XLV"]

        # 利用可能なETFをフィルタ
        cyclical_available = [e for e in cyclical_etfs if e in sector_prices.columns]
        defensive_available = [e for e in defensive_etfs if e in sector_prices.columns]

        if not cyclical_available or not defensive_available:
            return 0.0

        lookback = min(self.momentum_short, len(sector_prices) - 1)

        # 各グループの平均リターンを計算
        def avg_return(etfs: list[str]) -> float:
            returns = []
            for etf in etfs:
                prices = sector_prices[etf]
                if len(prices) > lookback:
                    ret = (prices.iloc[-1] - prices.iloc[-lookback - 1]) / prices.iloc[-lookback - 1]
                    returns.append(ret)
            return float(np.mean(returns)) if returns else 0.0

        cyclical_return = avg_return(cyclical_available)
        defensive_return = avg_return(defensive_available)

        return cyclical_return - defensive_return


class SectorMomentumCalculator:
    """
    セクターモメンタム計算器.

    複数期間のモメンタムを組み合わせてスコア化。
    """

    def __init__(
        self,
        lookback_short: int = 21,
        lookback_medium: int = 63,
        lookback_long: int = 126,
        weight_short: float = 0.3,
        weight_medium: float = 0.4,
        weight_long: float = 0.3,
    ):
        """
        Args:
            lookback_short: 短期ルックバック（21日 = 1ヶ月）
            lookback_medium: 中期ルックバック（63日 = 3ヶ月）
            lookback_long: 長期ルックバック（126日 = 6ヶ月）
            weight_*: 各期間の重み
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.weight_short = weight_short
        self.weight_medium = weight_medium
        self.weight_long = weight_long

    def calculate(self, sector_prices: pd.DataFrame) -> dict[str, float]:
        """
        全セクターのモメンタムスコアを計算.

        Args:
            sector_prices: セクターETF価格DataFrame

        Returns:
            {ETF: momentum_score} 辞書（-1 to 1）
        """
        scores: dict[str, float] = {}

        for etf in sector_prices.columns:
            prices = sector_prices[etf].dropna()

            if len(prices) < self.lookback_long + 1:
                scores[etf] = 0.0
                continue

            # 各期間のモメンタム計算
            mom_short = self._momentum(prices, self.lookback_short)
            mom_medium = self._momentum(prices, self.lookback_medium)
            mom_long = self._momentum(prices, self.lookback_long)

            # 加重平均
            combined = (
                self.weight_short * mom_short
                + self.weight_medium * mom_medium
                + self.weight_long * mom_long
            )

            # -1 to 1 にスケーリング（tanh）
            scores[etf] = float(np.tanh(combined * 5))

        return scores

    def _momentum(self, prices: pd.Series, lookback: int) -> float:
        """単一期間のモメンタム計算."""
        if len(prices) < lookback + 1:
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[-lookback - 1]

        if past == 0:
            return 0.0

        return float((current - past) / past)


class RelativeStrengthCalculator:
    """
    相対強度（RS）計算器.

    各セクターのS&P500に対する相対パフォーマンスを評価。
    """

    def __init__(
        self,
        lookback: int = 63,
        threshold: float = 0.6,
    ):
        """
        Args:
            lookback: 相対強度計算期間
            threshold: 上位/下位の閾値（0-1）
        """
        self.lookback = lookback
        self.threshold = threshold

    def calculate(
        self,
        sector_prices: pd.DataFrame,
        benchmark_prices: pd.Series,
    ) -> dict[str, float]:
        """
        全セクターの相対強度スコアを計算.

        Args:
            sector_prices: セクターETF価格DataFrame
            benchmark_prices: ベンチマーク（SPY等）価格Series

        Returns:
            {ETF: rs_score} 辞書（-1 to 1）
        """
        if len(benchmark_prices) < self.lookback + 1:
            return {etf: 0.0 for etf in sector_prices.columns}

        # ベンチマークリターン
        bench_return = self._return(benchmark_prices, self.lookback)

        # 各セクターの相対強度
        relative_returns: dict[str, float] = {}
        for etf in sector_prices.columns:
            prices = sector_prices[etf].dropna()
            if len(prices) < self.lookback + 1:
                relative_returns[etf] = 0.0
                continue

            sector_return = self._return(prices, self.lookback)
            relative_returns[etf] = sector_return - bench_return

        # ランク付け（パーセンタイル）
        if not relative_returns:
            return {}

        values = list(relative_returns.values())
        ranks = pd.Series(values).rank(pct=True)

        scores: dict[str, float] = {}
        for i, etf in enumerate(relative_returns.keys()):
            rank = ranks.iloc[i]
            # 0-1 のランクを -1 to 1 にスケーリング
            scores[etf] = float((rank - 0.5) * 2)

        return scores

    def _return(self, prices: pd.Series, lookback: int) -> float:
        """リターン計算."""
        current = prices.iloc[-1]
        past = prices.iloc[-lookback - 1]
        if past == 0:
            return 0.0
        return float((current - past) / past)


@SignalRegistry.register(
    "enhanced_sector_rotation",
    category="sector",
    description="Enhanced sector rotation with economic cycle detection",
    tags=["sector", "rotation", "cycle", "macro"],
)
class EnhancedSectorRotation(Signal):
    """
    景気サイクル適応型セクターローテーション.

    景気サイクルを判定し、各サイクルに最適なセクターを選択。
    モメンタムと相対強度を組み合わせてウェイトを決定。
    """

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="momentum_lookback",
                default=63,
                searchable=True,
                min_value=21,
                max_value=126,
                step=21,
                description="Momentum lookback period (days)",
            ),
            ParameterSpec(
                name="relative_strength_threshold",
                default=0.6,
                searchable=True,
                min_value=0.4,
                max_value=0.8,
                step=0.1,
                description="Relative strength threshold for selection",
            ),
            ParameterSpec(
                name="cycle_weight",
                default=0.4,
                searchable=True,
                min_value=0.2,
                max_value=0.6,
                step=0.1,
                description="Weight for cycle alignment score",
            ),
            ParameterSpec(
                name="momentum_weight",
                default=0.35,
                searchable=False,
                min_value=0.1,
                max_value=0.5,
                description="Weight for momentum score",
            ),
            ParameterSpec(
                name="rs_weight",
                default=0.25,
                searchable=False,
                min_value=0.1,
                max_value=0.4,
                description="Weight for relative strength score",
            ),
            ParameterSpec(
                name="min_weight",
                default=0.02,
                searchable=False,
                min_value=0.0,
                max_value=0.1,
                description="Minimum weight per sector",
            ),
            ParameterSpec(
                name="max_weight",
                default=0.25,
                searchable=False,
                min_value=0.1,
                max_value=0.5,
                description="Maximum weight per sector",
            ),
        ]

    def __init__(self, **params: Any):
        super().__init__(**params)

        self.cycle_detector = EconomicCycleDetector()
        self.momentum_calculator = SectorMomentumCalculator(
            lookback_medium=self._params.get("momentum_lookback", 63),
        )
        self.rs_calculator = RelativeStrengthCalculator(
            lookback=self._params.get("momentum_lookback", 63),
            threshold=self._params.get("relative_strength_threshold", 0.6),
        )

    def compute(
        self,
        data: pd.DataFrame,
        sector_prices: pd.DataFrame | None = None,
        benchmark_prices: pd.Series | None = None,
        vix_series: pd.Series | None = None,
    ) -> SignalResult:
        """
        セクターローテーションシグナルを計算.

        Args:
            data: メイン資産のOHLCVデータ（またはベンチマーク）
            sector_prices: 全セクターETFの価格DataFrame
            benchmark_prices: ベンチマーク（SPY等）の価格
            vix_series: VIX系列

        Returns:
            SignalResult with sector allocation recommendations
        """
        self.validate_input(data)

        # ベンチマークはデータから取得（close列使用）
        if benchmark_prices is None:
            benchmark_prices = data["close"]

        # セクター価格がない場合はデータから構築
        if sector_prices is None:
            # 単一資産の場合、スコアは0
            scores = pd.Series(0.0, index=data.index)
            return SignalResult(
                scores=scores,
                metadata={"error": "No sector prices provided"},
            )

        # ローテーション結果を取得
        result = self.get_sector_weights(
            sector_prices=sector_prices,
            benchmark_prices=benchmark_prices,
            vix_series=vix_series,
        )

        # 最後の日のスコア（代表値として）
        # 各セクターのcombined_scoreを使用
        final_scores = {
            s.ticker: s.combined_score for s in result.sector_scores
        }

        # 時系列として返す（最後の値を全期間に拡張）
        avg_score = np.mean(list(final_scores.values())) if final_scores else 0.0
        scores = pd.Series(avg_score, index=data.index)

        metadata = {
            "cycle": result.cycle.value,
            "cycle_confidence": result.cycle_confidence,
            "weights": result.weights,
            "sector_scores": {s.ticker: s.combined_score for s in result.sector_scores},
            **result.metadata,
        }

        return SignalResult(scores=scores, metadata=metadata)

    def get_sector_weights(
        self,
        sector_prices: pd.DataFrame,
        benchmark_prices: pd.Series | None = None,
        vix_series: pd.Series | None = None,
    ) -> RotationResult:
        """
        セクターウェイトを取得.

        Args:
            sector_prices: セクターETF価格DataFrame
            benchmark_prices: ベンチマーク価格（SPY等）
            vix_series: VIX系列（オプション）

        Returns:
            RotationResult with weights and analysis
        """
        # 1. 景気サイクル判定
        spy_prices = benchmark_prices if benchmark_prices is not None else sector_prices.get("SPY")
        if spy_prices is None and "SPY" not in sector_prices.columns:
            # SPYがない場合は全セクターの平均を使用
            spy_prices = sector_prices.mean(axis=1)

        cycle_result = self.cycle_detector.detect(
            spy_prices=spy_prices,
            vix_series=vix_series,
            sector_prices=sector_prices,
        )

        # 2. モメンタムスコア計算
        momentum_scores = self.momentum_calculator.calculate(sector_prices)

        # 3. 相対強度スコア計算
        if benchmark_prices is not None:
            rs_scores = self.rs_calculator.calculate(sector_prices, benchmark_prices)
        else:
            rs_scores = {etf: 0.0 for etf in sector_prices.columns}

        # 4. サイクル整合スコア計算
        cycle_alignment = self._calculate_cycle_alignment(
            cycle_result.cycle,
            list(sector_prices.columns),
        )

        # 5. スコアを組み合わせ
        sector_scores: list[SectorScore] = []

        cycle_weight = self._params.get("cycle_weight", 0.4)
        momentum_weight = self._params.get("momentum_weight", 0.35)
        rs_weight = self._params.get("rs_weight", 0.25)

        for etf in sector_prices.columns:
            mom = momentum_scores.get(etf, 0.0)
            rs = rs_scores.get(etf, 0.0)
            alignment = cycle_alignment.get(etf, 0.5)

            # 加重平均（alignmentは0-1なので-1~1に変換）
            alignment_score = (alignment - 0.5) * 2

            combined = (
                cycle_weight * alignment_score
                + momentum_weight * mom
                + rs_weight * rs
            )

            sector_scores.append(SectorScore(
                ticker=etf,
                momentum_score=mom,
                relative_strength_score=rs,
                cycle_alignment_score=alignment,
                combined_score=combined,
                weight=0.0,  # 後で設定
            ))

        # 6. ウェイト計算
        weights = self._scores_to_weights(sector_scores)

        for score in sector_scores:
            score.weight = weights.get(score.ticker, 0.0)

        return RotationResult(
            cycle=cycle_result.cycle,
            cycle_confidence=cycle_result.confidence,
            sector_scores=sector_scores,
            weights=weights,
            metadata={
                "cycle_indicators": cycle_result.indicators,
                "cycle_all_scores": cycle_result.metadata.get("all_scores", {}),
            },
        )

    def _calculate_cycle_alignment(
        self,
        cycle: EconomicCycle,
        sectors: list[str],
    ) -> dict[str, float]:
        """
        各セクターのサイクル整合度を計算.

        Args:
            cycle: 現在の景気サイクル
            sectors: セクターETFリスト

        Returns:
            {ETF: alignment_score} 辞書（0-1）
        """
        preferred_sectors = ECONOMIC_CYCLE_SECTORS.get(cycle, [])

        alignment: dict[str, float] = {}
        for etf in sectors:
            if etf in preferred_sectors:
                # 優先セクター: 高スコア
                alignment[etf] = 1.0
            else:
                # 非優先セクター: 景気感応度で調整
                cyclicality = SECTOR_CYCLICALITY.get(etf, 1.0)

                if cycle == EconomicCycle.CONTRACTION:
                    # 縮小局面: ディフェンシブ（低感応度）を優先
                    alignment[etf] = max(0.2, 1.0 - cyclicality / 2)
                else:
                    # 拡大局面: 感応度に応じたスコア
                    alignment[etf] = min(0.8, 0.3 + cyclicality / 4)

        return alignment

    def _scores_to_weights(
        self,
        sector_scores: list[SectorScore],
    ) -> dict[str, float]:
        """
        スコアをウェイトに変換.

        Softmax + 制約適用。

        Args:
            sector_scores: セクタースコアリスト

        Returns:
            {ETF: weight} 辞書
        """
        min_weight = self._params.get("min_weight", 0.02)
        max_weight = self._params.get("max_weight", 0.25)

        # combined_scoreを取得
        scores = {s.ticker: s.combined_score for s in sector_scores}

        if not scores:
            return {}

        # Softmax変換
        score_values = np.array(list(scores.values()))

        # 温度パラメータで調整（高いほど均等に近づく）
        temperature = 0.5

        # 数値安定化
        score_values = score_values - score_values.max()
        exp_scores = np.exp(score_values / temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        # 制約適用
        weights = np.clip(softmax_weights, min_weight, max_weight)

        # 再正規化
        weights = weights / weights.sum()

        return dict(zip(scores.keys(), weights))


def create_sector_rotation_strategy(
    momentum_lookback: int = 63,
    cycle_weight: float = 0.4,
) -> EnhancedSectorRotation:
    """
    セクターローテーション戦略を作成.

    Args:
        momentum_lookback: モメンタム計算期間
        cycle_weight: サイクル整合の重み

    Returns:
        EnhancedSectorRotation インスタンス
    """
    return EnhancedSectorRotation(
        momentum_lookback=momentum_lookback,
        cycle_weight=cycle_weight,
    )


def get_current_cycle_sectors(
    spy_prices: pd.Series,
    vix_series: pd.Series | None = None,
) -> tuple[EconomicCycle, list[str]]:
    """
    現在の景気サイクルと推奨セクターを取得.

    Args:
        spy_prices: S&P500価格系列
        vix_series: VIX系列（オプション）

    Returns:
        (EconomicCycle, [推奨セクターETF])
    """
    detector = EconomicCycleDetector()
    result = detector.detect(spy_prices, vix_series)

    recommended = ECONOMIC_CYCLE_SECTORS.get(result.cycle, [])

    return result.cycle, recommended
