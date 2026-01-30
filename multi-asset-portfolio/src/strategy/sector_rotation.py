"""
Sector Rotation Strategy Module - セクターローテーション戦略

経済サイクルに基づくセクターローテーション戦略を提供する。

経済サイクルと推奨セクター:
- Early Recovery: 景気底打ち、金融緩和 → Consumer Discretionary, Financials, Industrials, Materials
- Mid Expansion: 景気拡大、金利上昇 → Technology, Industrials, Materials
- Late Expansion: 景気ピーク、インフレ → Energy, Materials, Healthcare
- Recession: 景気後退、金融緩和 → Utilities, Consumer Staples, Healthcare

セクターETFマッピング:
- XLK: Technology
- XLY: Consumer Discretionary
- XLF: Financials
- XLI: Industrials
- XLB: Materials
- XLE: Energy
- XLU: Utilities
- XLP: Consumer Staples
- XLV: Healthcare
- XLRE: Real Estate
- XLC: Communication Services

設計根拠:
- 要求.md: マルチアセット対応
- 経済サイクルに応じたセクター配分の最適化
- マクロ指標を活用したフェーズ検出

使用例:
    from src.strategy.sector_rotation import EconomicCycleSectorRotator

    rotator = EconomicCycleSectorRotator()

    # 経済フェーズ検出
    phase = rotator.detect_economic_phase(
        yield_curve_slope=0.5,
        ism_pmi=55.0,
        unemployment_rate_change=-0.2,
        credit_spread=1.5,
    )
    print(f"Economic Phase: {phase}")

    # セクター調整係数
    adjustments = rotator.get_sector_adjustments(phase)
    print(f"Sector Adjustments: {adjustments}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================

class EconomicPhase(str, Enum):
    """経済サイクルフェーズ"""
    EARLY_RECOVERY = "early_recovery"   # 景気底打ち・回復初期
    MID_EXPANSION = "mid_expansion"     # 景気拡大中盤
    LATE_EXPANSION = "late_expansion"   # 景気拡大後期・ピーク
    RECESSION = "recession"             # 景気後退


class SectorCategory(str, Enum):
    """セクターカテゴリ"""
    CYCLICAL = "cyclical"           # 景気敏感
    DEFENSIVE = "defensive"         # ディフェンシブ
    GROWTH = "growth"               # グロース
    INTEREST_SENSITIVE = "interest_sensitive"  # 金利敏感


# S&P 500 GICS セクターETFマッピング
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Consumer Discretionary": "XLY",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Consumer Staples": "XLP",
    "Healthcare": "XLV",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# 逆引きマッピング
ETF_TO_SECTOR: Dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# セクターカテゴリ分類
SECTOR_CATEGORIES: Dict[str, SectorCategory] = {
    "XLK": SectorCategory.GROWTH,
    "XLY": SectorCategory.CYCLICAL,
    "XLF": SectorCategory.INTEREST_SENSITIVE,
    "XLI": SectorCategory.CYCLICAL,
    "XLB": SectorCategory.CYCLICAL,
    "XLE": SectorCategory.CYCLICAL,
    "XLU": SectorCategory.DEFENSIVE,
    "XLP": SectorCategory.DEFENSIVE,
    "XLV": SectorCategory.DEFENSIVE,
    "XLRE": SectorCategory.INTEREST_SENSITIVE,
    "XLC": SectorCategory.GROWTH,
}

# 経済サイクル別セクター推奨
CYCLE_SECTOR_RECOMMENDATIONS: Dict[EconomicPhase, Dict[str, List[str]]] = {
    EconomicPhase.EARLY_RECOVERY: {
        "overweight": ["XLY", "XLF", "XLI", "XLB"],
        "neutral": ["XLK", "XLE", "XLC", "XLRE"],
        "underweight": ["XLU", "XLP", "XLV"],
    },
    EconomicPhase.MID_EXPANSION: {
        "overweight": ["XLK", "XLI", "XLB", "XLC"],
        "neutral": ["XLY", "XLF", "XLE", "XLV"],
        "underweight": ["XLU", "XLP", "XLRE"],
    },
    EconomicPhase.LATE_EXPANSION: {
        "overweight": ["XLE", "XLB", "XLV"],
        "neutral": ["XLK", "XLI", "XLP", "XLU", "XLC"],
        "underweight": ["XLY", "XLF", "XLRE"],
    },
    EconomicPhase.RECESSION: {
        "overweight": ["XLU", "XLP", "XLV"],
        "neutral": ["XLK", "XLC", "XLRE"],
        "underweight": ["XLY", "XLF", "XLI", "XLB", "XLE"],
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MacroIndicators:
    """マクロ経済指標"""
    yield_curve_slope: float = 0.0      # 10Y-2Y スプレッド
    ism_pmi: float = 50.0               # ISM製造業PMI
    unemployment_rate_change: float = 0.0  # 失業率の変化（前月比）
    credit_spread: float = 1.0          # IG-Treasury スプレッド
    vix: float = 20.0                   # VIX指数
    inflation_rate: float = 2.0         # インフレ率
    fed_funds_rate: float = 2.0         # FF金利

    def to_dict(self) -> Dict[str, float]:
        return {
            "yield_curve_slope": self.yield_curve_slope,
            "ism_pmi": self.ism_pmi,
            "unemployment_rate_change": self.unemployment_rate_change,
            "credit_spread": self.credit_spread,
            "vix": self.vix,
            "inflation_rate": self.inflation_rate,
            "fed_funds_rate": self.fed_funds_rate,
        }


@dataclass
class PhaseDetectionResult:
    """フェーズ検出結果"""
    phase: EconomicPhase
    confidence: float  # 0-1
    scores: Dict[EconomicPhase, float] = field(default_factory=dict)
    indicators_used: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "confidence": self.confidence,
            "scores": {k.value: v for k, v in self.scores.items()},
            "indicators_used": self.indicators_used,
            "reasoning": self.reasoning,
        }


@dataclass
class SectorAdjustment:
    """セクター調整結果"""
    sector_etf: str
    sector_name: str
    category: SectorCategory
    adjustment: float  # -1.0 to +1.0
    recommendation: str  # "overweight", "neutral", "underweight"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_etf": self.sector_etf,
            "sector_name": self.sector_name,
            "category": self.category.value,
            "adjustment": self.adjustment,
            "recommendation": self.recommendation,
        }


@dataclass
class RotationResult:
    """ローテーション結果"""
    phase: EconomicPhase
    adjustments: Dict[str, SectorAdjustment]
    target_weights: Dict[str, float]
    phase_detection: PhaseDetectionResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "adjustments": {k: v.to_dict() for k, v in self.adjustments.items()},
            "target_weights": self.target_weights,
            "phase_detection": self.phase_detection.to_dict(),
            "metadata": self.metadata,
        }


# =============================================================================
# Economic Cycle Sector Rotator
# =============================================================================

class EconomicCycleSectorRotator:
    """
    経済サイクルベースのセクターローテーター

    マクロ経済指標から経済サイクルのフェーズを検出し、
    各フェーズに適したセクター配分調整を提供する。

    Usage:
        rotator = EconomicCycleSectorRotator()

        # フェーズ検出
        phase_result = rotator.detect_economic_phase(
            yield_curve_slope=0.5,
            ism_pmi=55.0,
            unemployment_rate_change=-0.2,
            credit_spread=1.5,
        )

        # セクター調整
        adjustments = rotator.get_sector_adjustments(
            phase_result.phase,
            adjustment_pct=0.20,
        )

        # ローテーション実行
        result = rotator.rotate(
            current_weights={"XLK": 0.1, "XLF": 0.1, ...},
            macro_indicators=MacroIndicators(...)
        )
    """

    # フェーズ検出の閾値
    THRESHOLDS = {
        "yield_curve_inversion": 0.0,      # イールドカーブ逆転
        "pmi_expansion": 50.0,             # PMI拡大・縮小境界
        "pmi_strong": 55.0,                # 強い拡大
        "pmi_weak": 45.0,                  # 弱い縮小
        "unemployment_improving": -0.1,    # 失業率改善
        "unemployment_worsening": 0.1,     # 失業率悪化
        "credit_spread_tight": 1.0,        # タイトなスプレッド
        "credit_spread_wide": 2.0,         # ワイドなスプレッド
        "vix_low": 15.0,                   # 低VIX
        "vix_high": 25.0,                  # 高VIX
    }

    def __init__(
        self,
        sector_etfs: Optional[Dict[str, str]] = None,
        cycle_recommendations: Optional[Dict[EconomicPhase, Dict[str, List[str]]]] = None,
    ) -> None:
        """
        初期化

        Args:
            sector_etfs: セクター名→ETFのマッピング（カスタム）
            cycle_recommendations: サイクル別推奨セクター（カスタム）
        """
        self.sector_etfs = sector_etfs or SECTOR_ETFS.copy()
        self.etf_to_sector = {v: k for k, v in self.sector_etfs.items()}
        self.cycle_recommendations = cycle_recommendations or CYCLE_SECTOR_RECOMMENDATIONS.copy()
        self.all_sector_etfs = list(self.etf_to_sector.keys())

        logger.info(
            f"EconomicCycleSectorRotator initialized with {len(self.all_sector_etfs)} sectors"
        )

    def detect_economic_phase(
        self,
        yield_curve_slope: float,
        ism_pmi: float,
        unemployment_rate_change: float,
        credit_spread: float,
        vix: Optional[float] = None,
    ) -> PhaseDetectionResult:
        """
        経済フェーズを検出

        Args:
            yield_curve_slope: 10Y-2Y スプレッド（正=順イールド、負=逆イールド）
            ism_pmi: ISM製造業PMI（50以上=拡大）
            unemployment_rate_change: 失業率の変化（負=改善）
            credit_spread: クレジットスプレッド
            vix: VIX指数（オプション）

        Returns:
            PhaseDetectionResult
        """
        # 各フェーズのスコアを計算
        scores: Dict[EconomicPhase, float] = {
            EconomicPhase.EARLY_RECOVERY: 0.0,
            EconomicPhase.MID_EXPANSION: 0.0,
            EconomicPhase.LATE_EXPANSION: 0.0,
            EconomicPhase.RECESSION: 0.0,
        }

        reasoning_parts = []

        # 1. イールドカーブ分析
        if yield_curve_slope < self.THRESHOLDS["yield_curve_inversion"]:
            # 逆イールド → Late Expansion or approaching Recession
            scores[EconomicPhase.LATE_EXPANSION] += 0.3
            scores[EconomicPhase.RECESSION] += 0.2
            reasoning_parts.append("逆イールド検出")
        elif yield_curve_slope > 1.0:
            # 急勾配 → Early Recovery
            scores[EconomicPhase.EARLY_RECOVERY] += 0.3
            reasoning_parts.append("急勾配イールドカーブ")
        else:
            # 正常な順イールド → Mid Expansion
            scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasoning_parts.append("正常なイールドカーブ")

        # 2. PMI分析
        if ism_pmi >= self.THRESHOLDS["pmi_strong"]:
            scores[EconomicPhase.MID_EXPANSION] += 0.3
            scores[EconomicPhase.LATE_EXPANSION] += 0.1
            reasoning_parts.append(f"強いPMI ({ism_pmi:.1f})")
        elif ism_pmi >= self.THRESHOLDS["pmi_expansion"]:
            scores[EconomicPhase.MID_EXPANSION] += 0.2
            scores[EconomicPhase.EARLY_RECOVERY] += 0.1
            reasoning_parts.append(f"拡大PMI ({ism_pmi:.1f})")
        elif ism_pmi >= self.THRESHOLDS["pmi_weak"]:
            scores[EconomicPhase.LATE_EXPANSION] += 0.2
            scores[EconomicPhase.RECESSION] += 0.1
            reasoning_parts.append(f"縮小PMI ({ism_pmi:.1f})")
        else:
            scores[EconomicPhase.RECESSION] += 0.3
            reasoning_parts.append(f"深い縮小PMI ({ism_pmi:.1f})")

        # 3. 失業率分析
        if unemployment_rate_change < self.THRESHOLDS["unemployment_improving"]:
            scores[EconomicPhase.EARLY_RECOVERY] += 0.2
            scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasoning_parts.append("失業率改善中")
        elif unemployment_rate_change > self.THRESHOLDS["unemployment_worsening"]:
            scores[EconomicPhase.LATE_EXPANSION] += 0.1
            scores[EconomicPhase.RECESSION] += 0.3
            reasoning_parts.append("失業率悪化中")
        else:
            scores[EconomicPhase.MID_EXPANSION] += 0.1
            reasoning_parts.append("失業率安定")

        # 4. クレジットスプレッド分析
        if credit_spread < self.THRESHOLDS["credit_spread_tight"]:
            scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasoning_parts.append("タイトなクレジットスプレッド")
        elif credit_spread > self.THRESHOLDS["credit_spread_wide"]:
            scores[EconomicPhase.RECESSION] += 0.2
            scores[EconomicPhase.EARLY_RECOVERY] += 0.1
            reasoning_parts.append("ワイドなクレジットスプレッド")
        else:
            scores[EconomicPhase.LATE_EXPANSION] += 0.1
            reasoning_parts.append("正常なクレジットスプレッド")

        # 5. VIX分析（オプション）
        if vix is not None:
            if vix < self.THRESHOLDS["vix_low"]:
                scores[EconomicPhase.MID_EXPANSION] += 0.1
                reasoning_parts.append(f"低VIX ({vix:.1f})")
            elif vix > self.THRESHOLDS["vix_high"]:
                scores[EconomicPhase.RECESSION] += 0.1
                scores[EconomicPhase.LATE_EXPANSION] += 0.05
                reasoning_parts.append(f"高VIX ({vix:.1f})")

        # スコアを正規化
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        # 最高スコアのフェーズを選択
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase]

        return PhaseDetectionResult(
            phase=best_phase,
            confidence=confidence,
            scores=scores,
            indicators_used={
                "yield_curve_slope": yield_curve_slope,
                "ism_pmi": ism_pmi,
                "unemployment_rate_change": unemployment_rate_change,
                "credit_spread": credit_spread,
                "vix": vix if vix is not None else 0.0,
            },
            reasoning="; ".join(reasoning_parts),
        )

    def get_sector_adjustments(
        self,
        phase: EconomicPhase,
        adjustment_pct: float = 0.20,
    ) -> Dict[str, SectorAdjustment]:
        """
        経済フェーズに基づくセクター調整係数を取得

        Args:
            phase: 経済フェーズ
            adjustment_pct: 調整幅（例: 0.20 = ±20%）

        Returns:
            セクターETF → SectorAdjustment のマッピング
        """
        recommendations = self.cycle_recommendations.get(phase, {})
        overweight = recommendations.get("overweight", [])
        neutral = recommendations.get("neutral", [])
        underweight = recommendations.get("underweight", [])

        adjustments = {}

        for etf in self.all_sector_etfs:
            sector_name = self.etf_to_sector.get(etf, etf)
            category = SECTOR_CATEGORIES.get(etf, SectorCategory.CYCLICAL)

            if etf in overweight:
                adjustment = adjustment_pct
                recommendation = "overweight"
            elif etf in underweight:
                adjustment = -adjustment_pct
                recommendation = "underweight"
            else:
                adjustment = 0.0
                recommendation = "neutral"

            adjustments[etf] = SectorAdjustment(
                sector_etf=etf,
                sector_name=sector_name,
                category=category,
                adjustment=adjustment,
                recommendation=recommendation,
            )

        return adjustments

    def apply_adjustments(
        self,
        base_weights: Dict[str, float],
        adjustments: Dict[str, SectorAdjustment],
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        ベース配分に調整を適用

        Args:
            base_weights: ベース配分 {ETF: weight}
            adjustments: セクター調整
            normalize: 正規化するか

        Returns:
            調整後の配分
        """
        adjusted = {}

        for etf, base_weight in base_weights.items():
            adj = adjustments.get(etf)
            if adj:
                # 調整を適用（乗算方式）
                multiplier = 1.0 + adj.adjustment
                adjusted[etf] = base_weight * multiplier
            else:
                adjusted[etf] = base_weight

        # 負の重みを0に
        adjusted = {k: max(0.0, v) for k, v in adjusted.items()}

        # 正規化
        if normalize:
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def rotate(
        self,
        current_weights: Dict[str, float],
        macro_indicators: Optional[MacroIndicators] = None,
        yield_curve_slope: Optional[float] = None,
        ism_pmi: Optional[float] = None,
        unemployment_rate_change: Optional[float] = None,
        credit_spread: Optional[float] = None,
        vix: Optional[float] = None,
        adjustment_pct: float = 0.20,
    ) -> RotationResult:
        """
        セクターローテーションを実行

        Args:
            current_weights: 現在のセクター配分
            macro_indicators: マクロ指標（MacroIndicatorsオブジェクト）
            yield_curve_slope: イールドカーブ（macro_indicatorsがない場合）
            ism_pmi: PMI（macro_indicatorsがない場合）
            unemployment_rate_change: 失業率変化（macro_indicatorsがない場合）
            credit_spread: クレジットスプレッド（macro_indicatorsがない場合）
            vix: VIX（オプション）
            adjustment_pct: 調整幅

        Returns:
            RotationResult
        """
        # マクロ指標を取得
        if macro_indicators:
            yc = macro_indicators.yield_curve_slope
            pmi = macro_indicators.ism_pmi
            ue = macro_indicators.unemployment_rate_change
            cs = macro_indicators.credit_spread
            v = macro_indicators.vix
        else:
            yc = yield_curve_slope if yield_curve_slope is not None else 0.5
            pmi = ism_pmi if ism_pmi is not None else 50.0
            ue = unemployment_rate_change if unemployment_rate_change is not None else 0.0
            cs = credit_spread if credit_spread is not None else 1.5
            v = vix

        # フェーズ検出
        phase_result = self.detect_economic_phase(
            yield_curve_slope=yc,
            ism_pmi=pmi,
            unemployment_rate_change=ue,
            credit_spread=cs,
            vix=v,
        )

        # セクター調整
        adjustments = self.get_sector_adjustments(phase_result.phase, adjustment_pct)

        # 配分を調整
        target_weights = self.apply_adjustments(current_weights, adjustments)

        logger.info(
            f"Sector rotation: phase={phase_result.phase.value}, "
            f"confidence={phase_result.confidence:.2f}"
        )

        return RotationResult(
            phase=phase_result.phase,
            adjustments=adjustments,
            target_weights=target_weights,
            phase_detection=phase_result,
            metadata={
                "adjustment_pct": adjustment_pct,
                "n_overweight": sum(1 for a in adjustments.values() if a.recommendation == "overweight"),
                "n_underweight": sum(1 for a in adjustments.values() if a.recommendation == "underweight"),
            },
        )

    def get_phase_recommendations(
        self,
        phase: EconomicPhase,
    ) -> Dict[str, List[str]]:
        """
        フェーズの推奨セクターを取得

        Args:
            phase: 経済フェーズ

        Returns:
            {"overweight": [...], "neutral": [...], "underweight": [...]}
        """
        return self.cycle_recommendations.get(phase, {
            "overweight": [],
            "neutral": self.all_sector_etfs,
            "underweight": [],
        })


# =============================================================================
# Momentum-Based Sector Rotator
# =============================================================================

class MomentumSectorRotator:
    """
    モメンタムベースのセクターローテーター

    相対モメンタムに基づいてセクター配分を決定する。

    Usage:
        rotator = MomentumSectorRotator(lookback_periods=[21, 63, 126])

        # モメンタムスコア計算
        scores = rotator.calculate_momentum_scores(sector_prices)

        # 配分決定
        weights = rotator.allocate(scores, top_n=4)
    """

    def __init__(
        self,
        lookback_periods: Optional[List[int]] = None,
        weights_by_period: Optional[List[float]] = None,
    ) -> None:
        """
        初期化

        Args:
            lookback_periods: モメンタム計算期間のリスト
            weights_by_period: 各期間の重み
        """
        self.lookback_periods = lookback_periods or [21, 63, 126]
        self.weights_by_period = weights_by_period or [0.5, 0.3, 0.2]

        # 重みを正規化
        total = sum(self.weights_by_period)
        self.weights_by_period = [w / total for w in self.weights_by_period]

    def calculate_momentum_scores(
        self,
        prices: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """
        各セクターのモメンタムスコアを計算

        Args:
            prices: セクターETF → 価格系列のマッピング

        Returns:
            セクターETF → モメンタムスコア
        """
        scores = {}

        for etf, price_series in prices.items():
            if len(price_series) < max(self.lookback_periods):
                scores[etf] = 0.0
                continue

            combined_momentum = 0.0

            for period, weight in zip(self.lookback_periods, self.weights_by_period):
                if len(price_series) >= period:
                    momentum = (
                        price_series.iloc[-1] / price_series.iloc[-period] - 1
                    )
                    combined_momentum += momentum * weight

            scores[etf] = combined_momentum

        return scores

    def rank_sectors(
        self,
        scores: Dict[str, float],
    ) -> List[Tuple[str, float, int]]:
        """
        セクターをスコアでランキング

        Args:
            scores: セクターETF → スコア

        Returns:
            [(ETF, score, rank), ...] のリスト
        """
        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(etf, score, i + 1) for i, (etf, score) in enumerate(sorted_sectors)]

    def allocate(
        self,
        scores: Dict[str, float],
        top_n: int = 4,
        equal_weight: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, float]:
        """
        モメンタムスコアに基づいて配分

        Args:
            scores: セクターETF → スコア
            top_n: 上位N銘柄に配分
            equal_weight: 均等配分するか
            min_score: 最低スコア閾値

        Returns:
            セクターETF → 配分重み
        """
        # スコアでフィルタ
        filtered = {k: v for k, v in scores.items() if v >= min_score}

        if not filtered:
            return {}

        # ランキング
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        selected = ranked[:top_n]

        # 配分計算
        if equal_weight:
            weight = 1.0 / len(selected)
            return {etf: weight for etf, _ in selected}
        else:
            # スコア比例
            total_score = sum(score for _, score in selected)
            if total_score > 0:
                return {etf: score / total_score for etf, score in selected}
            else:
                weight = 1.0 / len(selected)
                return {etf: weight for etf, _ in selected}


# =============================================================================
# Factory Functions
# =============================================================================

def create_economic_cycle_rotator() -> EconomicCycleSectorRotator:
    """デフォルト設定で経済サイクルローテーターを作成"""
    return EconomicCycleSectorRotator()


def create_momentum_rotator(
    lookback_periods: Optional[List[int]] = None,
) -> MomentumSectorRotator:
    """モメンタムローテーターを作成"""
    return MomentumSectorRotator(lookback_periods=lookback_periods)


def get_all_sector_etfs() -> List[str]:
    """全セクターETFのリストを取得"""
    return list(SECTOR_ETFS.values())


def get_sector_category(etf: str) -> Optional[SectorCategory]:
    """セクターETFのカテゴリを取得"""
    return SECTOR_CATEGORIES.get(etf)
