"""
Enhanced Yield Curve Signal - イールドカーブ形状分析

FREDの金利データを使用して、イールドカーブの詳細な形状分析を行う。

形状の種類:
- normal: 正常（短期 < 長期）- 経済拡大期
- flat: フラット - 景気転換点の可能性
- inverted: 逆イールド - 景気後退リスク
- humped: 逆U字型 - 中期金利が最高

シグナルの解釈:
- slope_2_10 > 0: 正常な傾斜
- slope_2_10 < 0: 逆イールド
- curvature > 0: 凸型（humped）
- curvature < 0: 凹型

使用例:
    from src.signals.yield_curve_signal import EnhancedYieldCurveSignal

    signal = EnhancedYieldCurveSignal()

    # 金利データを取得
    yields = signal.fetch_yields()

    # 形状分析
    shape_result = signal.compute_curve_shape(yields)
    print(f"Shape: {shape_result['shape']}")
    print(f"Slope (2-10): {shape_result['slope_2_10']:.2f}%")

    # 配分調整
    adjustments = signal.get_allocation_adjustment(shape_result['shape'])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.signals.base import ParameterSpec, Signal, SignalResult
from src.signals.registry import SignalRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================

class CurveShape(str, Enum):
    """イールドカーブ形状"""
    NORMAL = "normal"       # 正常（短期 < 長期）
    FLAT = "flat"           # フラット
    INVERTED = "inverted"   # 逆イールド
    HUMPED = "humped"       # 逆U字型（中期が最高）


# FRED系列コード
FRED_YIELD_SERIES = {
    "3M": "DGS3MO",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class YieldCurveData:
    """イールドカーブデータ

    Attributes:
        yields: 金利データ（テナー別）
        timestamp: データ取得日時
        source: データソース
    """
    yields: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "FRED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yields": self.yields,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


@dataclass
class CurveShapeResult:
    """イールドカーブ形状分析結果

    Attributes:
        slope_2_10: 10Y - 2Y スプレッド（%）
        slope_3m_10: 10Y - 3M スプレッド（%）
        curvature: 曲率 2*(5Y) - (2Y + 10Y)
        shape: カーブ形状
        yields_used: 使用した金利データ
        metadata: 追加情報
    """
    slope_2_10: float
    slope_3m_10: float
    curvature: float
    shape: CurveShape
    yields_used: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope_2_10": self.slope_2_10,
            "slope_3m_10": self.slope_3m_10,
            "curvature": self.curvature,
            "shape": self.shape.value,
            "yields_used": self.yields_used,
            "metadata": self.metadata,
        }


@dataclass
class AllocationAdjustment:
    """配分調整結果

    Attributes:
        equity_adjustment: 株式調整（正=オーバーウェイト）
        bond_adjustment: 債券調整
        cash_adjustment: 現金調整
        mid_term_bond_adjustment: 中期債調整
        reason: 調整理由
    """
    equity_adjustment: float
    bond_adjustment: float
    cash_adjustment: float
    mid_term_bond_adjustment: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, float]:
        return {
            "equity": self.equity_adjustment,
            "bond": self.bond_adjustment,
            "cash": self.cash_adjustment,
            "mid_term_bond": self.mid_term_bond_adjustment,
        }


# =============================================================================
# Enhanced Yield Curve Signal
# =============================================================================

class EnhancedYieldCurveSignal:
    """拡張イールドカーブシグナル

    FREDの金利データを使用して、詳細なイールドカーブ形状分析を行う。

    機能:
    1. compute_curve_shape: カーブ形状を分析
    2. get_allocation_adjustment: 形状に応じた配分調整を提案
    3. compute_slope_change_signal: スロープ変化率をシグナル化

    Example:
        signal = EnhancedYieldCurveSignal()

        # 形状分析
        yields = {"3M": 5.25, "2Y": 4.50, "5Y": 4.20, "10Y": 4.30, "30Y": 4.50}
        shape = signal.compute_curve_shape(yields)
        print(f"Shape: {shape.shape}")

        # 配分調整
        adjustments = signal.get_allocation_adjustment(shape.shape)
        print(f"Equity adj: {adjustments.equity_adjustment:+.0%}")
    """

    # 閾値設定
    INVERSION_THRESHOLD = -0.10  # -10bps以下で逆イールド
    FLAT_THRESHOLD = 0.25       # 25bps以下でフラット
    HUMPED_THRESHOLD = 0.10     # 曲率10bps以上でハンプ

    def __init__(
        self,
        inversion_threshold: float = -0.10,
        flat_threshold: float = 0.25,
        humped_threshold: float = 0.10,
    ) -> None:
        """初期化

        Args:
            inversion_threshold: 逆イールド判定閾値（%）
            flat_threshold: フラット判定閾値（%）
            humped_threshold: ハンプ判定閾値（%）
        """
        self.inversion_threshold = inversion_threshold
        self.flat_threshold = flat_threshold
        self.humped_threshold = humped_threshold
        self._yield_history: Optional[pd.DataFrame] = None

    def fetch_yields(
        self,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
    ) -> Dict[str, float]:
        """FREDから金利データを取得

        Args:
            start_date: 開始日（履歴取得用）
            end_date: 終了日

        Returns:
            テナー別の最新金利（%）
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not available, returning mock data")
            return self._get_mock_yields()

        # 日付設定
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        # Treasury yield tickers (Yahoo Finance)
        # Note: Yahoo doesn't have direct FRED access, using Treasury ETFs as proxy
        ticker_map = {
            "3M": "^IRX",    # 13 Week Treasury Bill
            "2Y": "2YY=F",   # 2 Year Treasury Yield (Futures proxy)
            "5Y": "^FVX",    # 5 Year Treasury Yield
            "10Y": "^TNX",   # 10 Year Treasury Yield
            "30Y": "^TYX",   # 30 Year Treasury Yield
        }

        yields = {}
        for tenor, ticker in ticker_map.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(start=start_date, end=end_date)
                if not hist.empty:
                    # ^TNX等は実際の金利の10倍で表示される
                    value = hist["Close"].iloc[-1]
                    if ticker in ["^TNX", "^TYX", "^FVX", "^IRX"]:
                        value = value / 10.0  # Convert to percentage
                    yields[tenor] = float(value)
            except Exception as e:
                logger.warning(f"Failed to fetch {tenor} yield: {e}")

        if not yields:
            logger.warning("No yields fetched, using mock data")
            return self._get_mock_yields()

        logger.info(f"Fetched yields: {yields}")
        return yields

    def _get_mock_yields(self) -> Dict[str, float]:
        """モックデータを返す（テスト用）"""
        return {
            "3M": 5.25,
            "2Y": 4.50,
            "5Y": 4.30,
            "10Y": 4.35,
            "30Y": 4.50,
        }

    def compute_curve_shape(
        self,
        yields: Dict[str, float],
    ) -> CurveShapeResult:
        """イールドカーブ形状を計算

        Args:
            yields: テナー別金利（%）

        Returns:
            CurveShapeResult
        """
        # 必要な金利を取得（デフォルト値付き）
        y_3m = yields.get("3M", yields.get("2Y", 4.0))
        y_2y = yields.get("2Y", 4.0)
        y_5y = yields.get("5Y", 4.0)
        y_10y = yields.get("10Y", 4.0)

        # スロープ計算
        slope_2_10 = y_10y - y_2y
        slope_3m_10 = y_10y - y_3m

        # 曲率計算: 2*(5Y) - (2Y + 10Y)
        # 正の値 = 中期が高い（humped）
        # 負の値 = 中期が低い（凹型）
        curvature = 2 * y_5y - (y_2y + y_10y)

        # 形状判定
        if slope_2_10 < self.inversion_threshold:
            shape = CurveShape.INVERTED
        elif abs(slope_2_10) < self.flat_threshold:
            if curvature > self.humped_threshold:
                shape = CurveShape.HUMPED
            else:
                shape = CurveShape.FLAT
        elif curvature > self.humped_threshold:
            shape = CurveShape.HUMPED
        else:
            shape = CurveShape.NORMAL

        logger.debug(
            "Curve shape: slope_2_10=%.2f%%, curvature=%.2f%%, shape=%s",
            slope_2_10,
            curvature,
            shape.value,
        )

        return CurveShapeResult(
            slope_2_10=slope_2_10,
            slope_3m_10=slope_3m_10,
            curvature=curvature,
            shape=shape,
            yields_used=yields.copy(),
            metadata={
                "inversion_threshold": self.inversion_threshold,
                "flat_threshold": self.flat_threshold,
                "humped_threshold": self.humped_threshold,
            },
        )

    def get_allocation_adjustment(
        self,
        shape: CurveShape | str,
    ) -> AllocationAdjustment:
        """形状に応じた配分調整を取得

        Args:
            shape: カーブ形状

        Returns:
            AllocationAdjustment
        """
        if isinstance(shape, str):
            shape = CurveShape(shape)

        if shape == CurveShape.NORMAL:
            # 正常: 株式オーバーウェイト
            return AllocationAdjustment(
                equity_adjustment=0.10,    # +10%
                bond_adjustment=-0.05,     # -5%
                cash_adjustment=-0.05,     # -5%
                mid_term_bond_adjustment=0.0,
                reason="Normal yield curve indicates economic expansion",
            )
        elif shape == CurveShape.FLAT:
            # フラット: ニュートラル
            return AllocationAdjustment(
                equity_adjustment=0.0,
                bond_adjustment=0.0,
                cash_adjustment=0.0,
                mid_term_bond_adjustment=0.0,
                reason="Flat yield curve - stay neutral",
            )
        elif shape == CurveShape.INVERTED:
            # 逆イールド: 債券・現金オーバーウェイト
            return AllocationAdjustment(
                equity_adjustment=-0.15,   # -15%
                bond_adjustment=0.05,      # +5%
                cash_adjustment=0.10,      # +10%
                mid_term_bond_adjustment=0.0,
                reason="Inverted yield curve signals recession risk",
            )
        elif shape == CurveShape.HUMPED:
            # ハンプ: 中期債オーバーウェイト
            return AllocationAdjustment(
                equity_adjustment=-0.05,   # -5%
                bond_adjustment=0.0,
                cash_adjustment=0.0,
                mid_term_bond_adjustment=0.05,  # +5%
                reason="Humped curve favors intermediate-term bonds",
            )
        else:
            # デフォルト: ニュートラル
            return AllocationAdjustment(
                equity_adjustment=0.0,
                bond_adjustment=0.0,
                cash_adjustment=0.0,
                reason="Unknown shape - stay neutral",
            )

    def compute_slope_change_signal(
        self,
        slope_history: pd.Series,
        lookback: int = 20,
    ) -> float:
        """スロープ変化率をシグナル化

        急激なフラット化はリスクオフシグナル。

        Args:
            slope_history: スロープの履歴（2-10スプレッド等）
            lookback: ルックバック期間

        Returns:
            シグナル値（-1 to +1）
            - 負: フラット化（リスクオフ）
            - 正: スティープ化（リスクオン）
        """
        if len(slope_history) < lookback + 1:
            logger.warning("Insufficient slope history for change signal")
            return 0.0

        # スロープの変化量
        current_slope = slope_history.iloc[-1]
        past_slope = slope_history.iloc[-lookback]
        slope_change = current_slope - past_slope

        # 標準偏差で正規化
        slope_std = slope_history.tail(lookback * 2).std()
        if slope_std < 0.01:
            slope_std = 0.01  # ゼロ除算防止

        # z-scoreをtanhで[-1, 1]に圧縮
        z_score = slope_change / slope_std
        signal = float(np.tanh(z_score / 2))

        logger.debug(
            "Slope change signal: change=%.2f%%, z=%.2f, signal=%.3f",
            slope_change,
            z_score,
            signal,
        )

        return signal

    def generate(
        self,
        yields: Dict[str, float] | None = None,
    ) -> SignalResult:
        """SignalRegistry互換のシグナル生成

        Args:
            yields: 金利データ（Noneの場合は取得）

        Returns:
            SignalResult
        """
        if yields is None:
            yields = self.fetch_yields()

        shape_result = self.compute_curve_shape(yields)

        # シグナル値を計算
        # 正常 → +1, 逆イールド → -1
        signal_map = {
            CurveShape.NORMAL: 1.0,
            CurveShape.FLAT: 0.0,
            CurveShape.INVERTED: -1.0,
            CurveShape.HUMPED: -0.3,
        }
        signal_value = signal_map.get(shape_result.shape, 0.0)

        # スロープに基づく連続シグナル
        # slope_2_10を[-1, 1]にマッピング（-1%〜+2%を想定）
        continuous_signal = np.clip(shape_result.slope_2_10 / 1.5, -1, 1)

        # 加重平均
        final_signal = 0.6 * continuous_signal + 0.4 * signal_value

        return SignalResult(
            scores=pd.Series([final_signal]),
            metadata={
                "shape": shape_result.shape.value,
                "slope_2_10": shape_result.slope_2_10,
                "slope_3m_10": shape_result.slope_3m_10,
                "curvature": shape_result.curvature,
                "yields": yields,
                "discrete_signal": signal_value,
                "continuous_signal": continuous_signal,
            },
        )


# =============================================================================
# SignalRegistry互換クラス
# =============================================================================

@SignalRegistry.register(
    "enhanced_yield_curve",
    category="macro",
    description="Enhanced yield curve shape analysis using treasury rates",
    tags=["rates", "recession", "regime", "treasury", "fred"],
)
class EnhancedYieldCurveRegisteredSignal(Signal):
    """SignalRegistry登録用の拡張イールドカーブシグナル

    Output interpretation:
    - +1.0: 正常なカーブ（経済拡大期）
    -  0.0: フラットなカーブ
    - -1.0: 逆イールド（景気後退リスク）
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="inversion_threshold",
                default=-0.10,
                searchable=False,
                min_value=-0.5,
                max_value=0.0,
                description="Threshold for inversion detection (%)",
            ),
            ParameterSpec(
                name="flat_threshold",
                default=0.25,
                searchable=True,
                min_value=0.1,
                max_value=0.5,
                step=0.05,
                description="Threshold for flat curve detection (%)",
            ),
            ParameterSpec(
                name="slope_scale",
                default=1.5,
                searchable=True,
                min_value=0.5,
                max_value=3.0,
                step=0.5,
                description="Scale factor for slope to signal conversion",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """シグナルを計算

        Args:
            data: 価格データ（直接は使用しない、金利を別途取得）

        Returns:
            SignalResult
        """
        self.validate_input(data)

        inversion_threshold = self._params["inversion_threshold"]
        flat_threshold = self._params["flat_threshold"]
        slope_scale = self._params["slope_scale"]

        # EnhancedYieldCurveSignalを使用
        enhanced_signal = EnhancedYieldCurveSignal(
            inversion_threshold=inversion_threshold,
            flat_threshold=flat_threshold,
        )

        # 金利データ取得
        try:
            yields = enhanced_signal.fetch_yields()
        except Exception as e:
            logger.warning(f"Failed to fetch yields: {e}")
            return SignalResult(
                scores=pd.Series(0.0, index=data.index),
                metadata={"error": str(e)},
            )

        # 形状分析
        shape_result = enhanced_signal.compute_curve_shape(yields)

        # シグナル計算
        slope_signal = np.clip(shape_result.slope_2_10 / slope_scale, -1, 1)

        # 全期間同じシグナルを出力（最新の形状に基づく）
        scores = pd.Series(float(slope_signal), index=data.index)

        return SignalResult(
            scores=scores,
            metadata={
                "shape": shape_result.shape.value,
                "slope_2_10": shape_result.slope_2_10,
                "curvature": shape_result.curvature,
                "yields": yields,
            },
        )


# =============================================================================
# 便利関数
# =============================================================================

def get_current_yield_curve_shape() -> CurveShapeResult:
    """現在のイールドカーブ形状を取得

    Returns:
        CurveShapeResult
    """
    signal = EnhancedYieldCurveSignal()
    yields = signal.fetch_yields()
    return signal.compute_curve_shape(yields)


def get_yield_curve_allocation_adjustment() -> AllocationAdjustment:
    """現在のイールドカーブに基づく配分調整を取得

    Returns:
        AllocationAdjustment
    """
    shape_result = get_current_yield_curve_shape()
    signal = EnhancedYieldCurveSignal()
    return signal.get_allocation_adjustment(shape_result.shape)
