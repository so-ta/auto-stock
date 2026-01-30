"""
Macro Timing Module - 経済サイクルに基づくセクターアロケーション

経済指標（ISM PMI、失業率、イールドカーブ）を基に
経済サイクルのフェーズを判定し、適切なセクター配分を提供する。

機能:
1. EconomicCycleAllocator: 経済サイクルに基づくセクターアロケーション
2. MacroDataFetcher: 経済指標データの取得
3. 便利関数: get_current_cycle_phase(), get_cycle_adjusted_weights()

経済サイクルフェーズ:
- Early Expansion: 景気回復初期（Technology, Consumer Discretionary）
- Mid Expansion: 景気拡大中期（Industrials, Materials）
- Late Expansion: 景気拡大後期（Energy, Commodities）
- Recession: 景気後退（Utilities, Consumer Staples, Healthcare）

設計根拠:
- 要求.md: マクロ経済連動
- セクターローテーション理論に基づく
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================
class EconomicPhase(str, Enum):
    """経済サイクルフェーズ"""

    EARLY_EXPANSION = "early_expansion"
    MID_EXPANSION = "mid_expansion"
    LATE_EXPANSION = "late_expansion"
    RECESSION = "recession"
    UNKNOWN = "unknown"


# セクターETF定義
SECTOR_ETFS = {
    "technology": "XLK",
    "consumer_discretionary": "XLY",
    "financials": "XLF",
    "industrials": "XLI",
    "materials": "XLB",
    "energy": "XLE",
    "utilities": "XLU",
    "consumer_staples": "XLP",
    "healthcare": "XLV",
    "real_estate": "XLRE",
    "communications": "XLC",
    "gold": "GLD",
    "long_treasury": "TLT",
}


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class MacroIndicators:
    """マクロ経済指標

    Attributes:
        ism_pmi: ISM製造業PMI（50以上=拡張、50未満=収縮）
        unemployment_rate: 失業率（%）
        unemployment_change: 失業率変化（前月比）
        yield_curve_slope: イールドカーブ傾斜（10Y-2Y, bp）
        cpi_yoy: CPI前年比（%）
        fed_funds_rate: FF金利（%）
        as_of_date: データ基準日
    """

    ism_pmi: float
    unemployment_rate: float
    unemployment_change: float
    yield_curve_slope: float
    cpi_yoy: float = 0.0
    fed_funds_rate: float = 0.0
    as_of_date: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ism_pmi": self.ism_pmi,
            "unemployment_rate": self.unemployment_rate,
            "unemployment_change": self.unemployment_change,
            "yield_curve_slope": self.yield_curve_slope,
            "cpi_yoy": self.cpi_yoy,
            "fed_funds_rate": self.fed_funds_rate,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
        }


@dataclass
class CycleAllocation:
    """サイクルアロケーション結果

    Attributes:
        phase: 経済サイクルフェーズ
        allocations: セクター配分（ETFシンボル -> 重み）
        confidence: 判定の確信度（0-1）
        indicators_used: 使用した指標
        reasoning: 判定理由
    """

    phase: EconomicPhase
    allocations: dict[str, float]
    confidence: float = 0.5
    indicators_used: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "phase": self.phase.value,
            "allocations": self.allocations,
            "confidence": self.confidence,
            "indicators_used": self.indicators_used,
            "reasoning": self.reasoning,
        }


# =============================================================================
# Economic Cycle Allocator
# =============================================================================
class EconomicCycleAllocator:
    """経済サイクルに基づくセクターアロケーション

    経済指標からサイクルフェーズを判定し、適切なセクター配分を返す。

    フェーズ別推奨セクター:
    - Early Expansion: Technology, Consumer Discretionary（景気敏感）
    - Mid Expansion: Industrials, Materials（設備投資関連）
    - Late Expansion: Energy, Commodities（インフレヘッジ）
    - Recession: Utilities, Consumer Staples, Healthcare（ディフェンシブ）

    Usage:
        allocator = EconomicCycleAllocator()

        indicators = MacroIndicators(
            ism_pmi=55.0,
            unemployment_rate=4.0,
            unemployment_change=-0.1,
            yield_curve_slope=100,
        )

        result = allocator.allocate(indicators)
        print(f"Phase: {result.phase}")
        print(f"Allocations: {result.allocations}")
    """

    # 経済サイクルフェーズ別配分
    CYCLE_ALLOCATIONS: dict[str, dict[str, float]] = {
        "early_expansion": {
            "XLK": 0.25,  # Technology
            "XLY": 0.20,  # Consumer Discretionary
            "XLF": 0.15,  # Financials
            "XLI": 0.10,  # Industrials
            "others": 0.30,
        },
        "mid_expansion": {
            "XLI": 0.20,  # Industrials
            "XLB": 0.15,  # Materials
            "XLK": 0.15,  # Technology
            "XLF": 0.15,  # Financials
            "others": 0.35,
        },
        "late_expansion": {
            "XLE": 0.20,  # Energy
            "XLB": 0.15,  # Materials
            "GLD": 0.15,  # Gold
            "XLI": 0.10,  # Industrials
            "others": 0.40,
        },
        "recession": {
            "XLU": 0.20,  # Utilities
            "XLP": 0.20,  # Consumer Staples
            "XLV": 0.15,  # Healthcare
            "TLT": 0.20,  # Long Treasury
            "others": 0.25,
        },
    }

    # フェーズ判定閾値
    ISM_EXPANSION_THRESHOLD = 50.0
    ISM_STRONG_EXPANSION = 55.0
    ISM_STRONG_CONTRACTION = 45.0
    YIELD_CURVE_INVERSION_THRESHOLD = 0.0  # bp
    UNEMPLOYMENT_RISING_THRESHOLD = 0.1  # %

    def __init__(
        self,
        custom_allocations: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """初期化

        Args:
            custom_allocations: カスタムアロケーション設定
        """
        self.allocations = custom_allocations or self.CYCLE_ALLOCATIONS

    def detect_cycle_phase(
        self,
        ism_pmi: float,
        unemployment_change: float,
        yield_curve_slope: float,
    ) -> tuple[EconomicPhase, float, str]:
        """経済指標からサイクルフェーズを判定

        Args:
            ism_pmi: ISM製造業PMI
            unemployment_change: 失業率変化（前月比）
            yield_curve_slope: イールドカーブ傾斜（10Y-2Y, bp）

        Returns:
            (EconomicPhase, confidence, reasoning)

        判定ロジック:
        - ISM PMI > 55, 失業率低下 → Early/Mid Expansion
        - ISM PMI > 50, 横ばい → Late Expansion
        - ISM PMI < 50, 失業率上昇 → Recession
        - イールドカーブ逆転 → Recession警戒
        """
        reasons = []
        phase_scores = {
            EconomicPhase.EARLY_EXPANSION: 0.0,
            EconomicPhase.MID_EXPANSION: 0.0,
            EconomicPhase.LATE_EXPANSION: 0.0,
            EconomicPhase.RECESSION: 0.0,
        }

        # 1. ISM PMIに基づく判定
        if ism_pmi >= self.ISM_STRONG_EXPANSION:
            phase_scores[EconomicPhase.EARLY_EXPANSION] += 0.3
            phase_scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasons.append(f"ISM PMI strong ({ism_pmi:.1f} > 55)")
        elif ism_pmi >= self.ISM_EXPANSION_THRESHOLD:
            phase_scores[EconomicPhase.MID_EXPANSION] += 0.2
            phase_scores[EconomicPhase.LATE_EXPANSION] += 0.3
            reasons.append(f"ISM PMI expansionary ({ism_pmi:.1f} > 50)")
        elif ism_pmi >= self.ISM_STRONG_CONTRACTION:
            phase_scores[EconomicPhase.LATE_EXPANSION] += 0.2
            phase_scores[EconomicPhase.RECESSION] += 0.3
            reasons.append(f"ISM PMI contractionary ({ism_pmi:.1f} < 50)")
        else:
            phase_scores[EconomicPhase.RECESSION] += 0.4
            reasons.append(f"ISM PMI deep contraction ({ism_pmi:.1f} < 45)")

        # 2. 失業率変化に基づく判定
        if unemployment_change < -self.UNEMPLOYMENT_RISING_THRESHOLD:
            # 失業率低下 → 景気拡大初期〜中期
            phase_scores[EconomicPhase.EARLY_EXPANSION] += 0.3
            phase_scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasons.append(f"Unemployment falling ({unemployment_change:.2f}%)")
        elif unemployment_change > self.UNEMPLOYMENT_RISING_THRESHOLD:
            # 失業率上昇 → 景気後退
            phase_scores[EconomicPhase.RECESSION] += 0.4
            reasons.append(f"Unemployment rising ({unemployment_change:.2f}%)")
        else:
            # 横ばい → 景気拡大後期
            phase_scores[EconomicPhase.MID_EXPANSION] += 0.1
            phase_scores[EconomicPhase.LATE_EXPANSION] += 0.2
            reasons.append(f"Unemployment stable ({unemployment_change:.2f}%)")

        # 3. イールドカーブに基づく判定
        if yield_curve_slope < self.YIELD_CURVE_INVERSION_THRESHOLD:
            # 逆イールド → Recession警戒
            phase_scores[EconomicPhase.RECESSION] += 0.3
            phase_scores[EconomicPhase.LATE_EXPANSION] += 0.1
            reasons.append(f"Yield curve inverted ({yield_curve_slope:.0f}bp)")
        elif yield_curve_slope < 50:
            # フラットニング → Late Expansion
            phase_scores[EconomicPhase.LATE_EXPANSION] += 0.2
            reasons.append(f"Yield curve flattening ({yield_curve_slope:.0f}bp)")
        elif yield_curve_slope < 150:
            # 正常 → Mid Expansion
            phase_scores[EconomicPhase.MID_EXPANSION] += 0.2
            reasons.append(f"Yield curve normal ({yield_curve_slope:.0f}bp)")
        else:
            # スティープ → Early Expansion
            phase_scores[EconomicPhase.EARLY_EXPANSION] += 0.3
            reasons.append(f"Yield curve steep ({yield_curve_slope:.0f}bp)")

        # 最高スコアのフェーズを選択
        max_phase = max(phase_scores, key=phase_scores.get)  # type: ignore
        max_score = phase_scores[max_phase]

        # 確信度を計算（0-1）
        total_score = sum(phase_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.25

        reasoning = "; ".join(reasons)

        logger.info(
            "Cycle phase detected: %s (confidence: %.2f) - %s",
            max_phase.value,
            confidence,
            reasoning,
        )

        return max_phase, confidence, reasoning

    def allocate(self, indicators: MacroIndicators) -> CycleAllocation:
        """経済指標からセクター配分を決定

        Args:
            indicators: マクロ経済指標

        Returns:
            CycleAllocation
        """
        # フェーズ判定
        phase, confidence, reasoning = self.detect_cycle_phase(
            ism_pmi=indicators.ism_pmi,
            unemployment_change=indicators.unemployment_change,
            yield_curve_slope=indicators.yield_curve_slope,
        )

        # 配分取得
        allocations = self.allocations.get(
            phase.value, self.allocations.get("mid_expansion", {})
        )

        return CycleAllocation(
            phase=phase,
            allocations=allocations.copy(),
            confidence=confidence,
            indicators_used={
                "ism_pmi": indicators.ism_pmi,
                "unemployment_change": indicators.unemployment_change,
                "yield_curve_slope": indicators.yield_curve_slope,
            },
            reasoning=reasoning,
        )

    def get_phase_allocations(self, phase: EconomicPhase) -> dict[str, float]:
        """指定フェーズの配分を取得

        Args:
            phase: 経済サイクルフェーズ

        Returns:
            セクター配分
        """
        return self.allocations.get(
            phase.value, self.allocations.get("mid_expansion", {})
        ).copy()


# =============================================================================
# Macro Data Fetcher
# =============================================================================
class MacroDataFetcher:
    """経済指標データの取得

    FRED API等から経済指標を取得する。
    実際のAPIキーがない場合はモックデータを返す。

    対応指標:
    - ISM PMI
    - 失業率
    - イールドカーブ（10Y-2Y）
    - CPI
    - FF金利

    Usage:
        fetcher = MacroDataFetcher()
        indicators = fetcher.fetch_latest()
        # または
        indicators = fetcher.fetch_as_of(date(2024, 1, 1))
    """

    # FRED Series IDs
    FRED_SERIES = {
        "ism_pmi": "NAPM",  # ISM Manufacturing PMI
        "unemployment_rate": "UNRATE",  # Unemployment Rate
        "yield_10y": "DGS10",  # 10-Year Treasury
        "yield_2y": "DGS2",  # 2-Year Treasury
        "cpi": "CPIAUCSL",  # CPI
        "fed_funds": "FEDFUNDS",  # Federal Funds Rate
    }

    def __init__(
        self,
        api_key: str | None = None,
        use_cache: bool = True,
    ) -> None:
        """初期化

        Args:
            api_key: FRED API Key（Noneの場合はモックデータ）
            use_cache: キャッシュを使用するか
        """
        self.api_key = api_key
        self.use_cache = use_cache
        self._cache: dict[str, Any] = {}

    def fetch_latest(self) -> MacroIndicators:
        """最新の経済指標を取得

        Returns:
            MacroIndicators
        """
        # APIキーがない場合はモックデータ
        if self.api_key is None:
            return self._get_mock_indicators()

        # 実際のFRED API呼び出し（実装例）
        try:
            return self._fetch_from_fred()
        except Exception as e:
            logger.warning(f"Failed to fetch from FRED: {e}. Using mock data.")
            return self._get_mock_indicators()

    def fetch_as_of(self, as_of_date: datetime) -> MacroIndicators:
        """指定日時点の経済指標を取得

        Args:
            as_of_date: 基準日

        Returns:
            MacroIndicators
        """
        # 簡易実装：最新データを返す
        indicators = self.fetch_latest()
        indicators.as_of_date = as_of_date
        return indicators

    def fetch_history(
        self,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """経済指標の履歴を取得

        Args:
            start_date: 開始日
            end_date: 終了日（省略時は現在）

        Returns:
            経済指標の履歴DataFrame
        """
        # モックデータで履歴を生成
        if end_date is None:
            end_date = datetime.now()

        date_range = pd.date_range(start=start_date, end=end_date, freq="MS")

        data = []
        for date in date_range:
            # 季節性を持つモックデータ
            month = date.month
            cycle_position = (month - 1) / 12  # 0-1

            ism_pmi = 50 + 5 * np.sin(2 * np.pi * cycle_position) + np.random.normal(0, 2)
            unemployment = 4.5 + 0.5 * np.cos(2 * np.pi * cycle_position) + np.random.normal(0, 0.2)
            yield_slope = 100 + 50 * np.sin(2 * np.pi * cycle_position) + np.random.normal(0, 20)

            data.append({
                "date": date,
                "ism_pmi": max(30, min(70, ism_pmi)),
                "unemployment_rate": max(3, min(10, unemployment)),
                "yield_curve_slope": yield_slope,
            })

        return pd.DataFrame(data).set_index("date")

    def _fetch_from_fred(self) -> MacroIndicators:
        """FRED APIから取得（実装例）

        Returns:
            MacroIndicators

        Note:
            実際のFRED API呼び出しにはfreddaモジュール等を使用
        """
        # 実装例（実際にはfreddaやpandasdataread等を使用）
        raise NotImplementedError("FRED API implementation required")

    def _get_mock_indicators(self) -> MacroIndicators:
        """モック経済指標を生成

        Returns:
            MacroIndicators
        """
        # 現実的なモックデータ
        return MacroIndicators(
            ism_pmi=52.5,  # 緩やかな拡張
            unemployment_rate=4.2,
            unemployment_change=-0.1,  # 緩やかな低下
            yield_curve_slope=80,  # やや正常
            cpi_yoy=3.2,
            fed_funds_rate=5.25,
            as_of_date=datetime.now(),
        )


# =============================================================================
# Convenience Functions
# =============================================================================
def get_current_cycle_phase(
    ism_pmi: float | None = None,
    unemployment_change: float | None = None,
    yield_curve_slope: float | None = None,
    fetcher: MacroDataFetcher | None = None,
) -> tuple[EconomicPhase, float]:
    """現在の経済サイクルフェーズを取得

    Args:
        ism_pmi: ISM PMI（省略時はfetcherから取得）
        unemployment_change: 失業率変化（省略時はfetcherから取得）
        yield_curve_slope: イールドカーブ傾斜（省略時はfetcherから取得）
        fetcher: MacroDataFetcher（省略時は新規作成）

    Returns:
        (EconomicPhase, confidence)

    Usage:
        phase, confidence = get_current_cycle_phase()
        print(f"Current phase: {phase.value} ({confidence:.0%})")

        # または指標を直接指定
        phase, confidence = get_current_cycle_phase(
            ism_pmi=55.0,
            unemployment_change=-0.1,
            yield_curve_slope=100,
        )
    """
    if fetcher is None:
        fetcher = MacroDataFetcher()

    # 指標が指定されていない場合は取得
    if ism_pmi is None or unemployment_change is None or yield_curve_slope is None:
        indicators = fetcher.fetch_latest()
        ism_pmi = ism_pmi or indicators.ism_pmi
        unemployment_change = unemployment_change or indicators.unemployment_change
        yield_curve_slope = yield_curve_slope or indicators.yield_curve_slope

    allocator = EconomicCycleAllocator()
    phase, confidence, _ = allocator.detect_cycle_phase(
        ism_pmi=ism_pmi,
        unemployment_change=unemployment_change,
        yield_curve_slope=yield_curve_slope,
    )

    return phase, confidence


def get_cycle_adjusted_weights(
    base_weights: dict[str, float],
    cycle_phase: EconomicPhase | str | None = None,
    adjustment_strength: float = 0.3,
    fetcher: MacroDataFetcher | None = None,
) -> dict[str, float]:
    """サイクル調整後の重みを取得

    Args:
        base_weights: 基本重み（アセット -> 重み）
        cycle_phase: 経済サイクルフェーズ（省略時は自動検出）
        adjustment_strength: 調整強度（0-1、0=調整なし、1=完全置換）
        fetcher: MacroDataFetcher

    Returns:
        調整後の重み

    Usage:
        base = {"AAPL": 0.3, "XLK": 0.2, "XLU": 0.1, "TLT": 0.2, "GLD": 0.2}
        adjusted = get_cycle_adjusted_weights(base, adjustment_strength=0.3)
    """
    # フェーズ取得
    if cycle_phase is None:
        phase, _ = get_current_cycle_phase(fetcher=fetcher)
    elif isinstance(cycle_phase, str):
        phase = EconomicPhase(cycle_phase)
    else:
        phase = cycle_phase

    allocator = EconomicCycleAllocator()
    cycle_allocations = allocator.get_phase_allocations(phase)

    # othersを除去
    cycle_allocations.pop("others", None)

    # 調整後の重みを計算
    adjusted_weights = {}
    total_base = sum(base_weights.values())

    for asset, base_weight in base_weights.items():
        # サイクル配分に含まれるアセットは調整
        if asset in cycle_allocations:
            cycle_weight = cycle_allocations[asset]
            adjusted_weight = (1 - adjustment_strength) * base_weight + adjustment_strength * cycle_weight
        else:
            # サイクル配分に含まれないアセットは基本重みを維持
            adjusted_weight = base_weight

        adjusted_weights[asset] = adjusted_weight

    # 正規化
    total_adjusted = sum(adjusted_weights.values())
    if total_adjusted > 0:
        adjusted_weights = {
            asset: weight / total_adjusted * total_base
            for asset, weight in adjusted_weights.items()
        }

    logger.info(
        "Cycle-adjusted weights: phase=%s, adjustment=%.0f%%",
        phase.value,
        adjustment_strength * 100,
    )

    return adjusted_weights


def get_sector_rotation_signal(
    returns: pd.DataFrame,
    lookback: int = 60,
) -> dict[str, float]:
    """セクターローテーションシグナルを計算

    Args:
        returns: セクターETFリターン（列=ETFシンボル）
        lookback: ルックバック期間

    Returns:
        セクター -> シグナル（-1〜+1）
    """
    signals = {}

    for sector in returns.columns:
        if sector not in returns.columns:
            continue

        sector_returns = returns[sector].iloc[-lookback:]

        # モメンタム
        momentum = sector_returns.sum()

        # ボラティリティ調整
        vol = sector_returns.std()
        if vol > 0:
            risk_adjusted = momentum / vol
        else:
            risk_adjusted = 0.0

        # 正規化（tanh）
        signal = np.tanh(risk_adjusted)
        signals[sector] = float(signal)

    return signals


# =============================================================================
# Export for Signal Registry Integration
# =============================================================================
def create_macro_timing_signal():
    """マクロタイミングシグナルを作成（SignalRegistry用）

    Returns:
        MacroTimingSignal instance
    """
    from .base import Signal, SignalResult

    class MacroTimingSignal(Signal):
        """マクロタイミングシグナル

        経済サイクルに基づくシグナルを生成。
        """

        @classmethod
        def parameter_specs(cls):
            return []

        @classmethod
        def get_param_grid(cls):
            return {}

        def compute(self, data: pd.DataFrame) -> SignalResult:
            # 経済指標が利用可能な場合
            fetcher = MacroDataFetcher()
            indicators = fetcher.fetch_latest()

            allocator = EconomicCycleAllocator()
            result = allocator.allocate(indicators)

            # フェーズをスコアに変換
            phase_scores = {
                EconomicPhase.EARLY_EXPANSION: 1.0,
                EconomicPhase.MID_EXPANSION: 0.5,
                EconomicPhase.LATE_EXPANSION: 0.0,
                EconomicPhase.RECESSION: -1.0,
                EconomicPhase.UNKNOWN: 0.0,
            }

            score = phase_scores.get(result.phase, 0.0)

            # 全期間に同じスコアを適用
            scores = pd.Series(score, index=data.index)

            return SignalResult(
                scores=scores,
                metadata={
                    "phase": result.phase.value,
                    "confidence": result.confidence,
                    "allocations": result.allocations,
                    "reasoning": result.reasoning,
                },
            )

    return MacroTimingSignal()
