"""
VIX-Based Cash Allocation - VIXベースのキャッシュ配分

VIX（恐怖指数）の水準に応じて動的にキャッシュ比率を調整する。

ロジック:
- VIX低（<15）: 市場は楽観的 → キャッシュ少（5%）
- VIX高（>35）: 市場は恐怖状態 → キャッシュ多（50%）
- 中間は段階的に調整

設計根拠:
- VIXは市場の不確実性を反映
- 高VIX時はリスク資産を減らしキャッシュを増やす
- 低VIX時はフルインベストに近づける

使用例:
    from src.risk.vix_cash_allocation import VIXCashAllocator

    allocator = VIXCashAllocator()

    # 現在のVIXからキャッシュ比率を計算
    cash_ratio = allocator.compute_cash_ratio(vix_current=25.0)
    print(f"Cash ratio: {cash_ratio:.1%}")

    # 重みを調整
    base_weights = {"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4}
    adjusted = allocator.adjust_weights(base_weights, cash_ratio)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VIXCashConfig:
    """VIXキャッシュ配分の設定

    Attributes:
        vix_low: 低VIX閾値（これ以下は最小キャッシュ）
        vix_mid: 中VIX閾値
        vix_high: 高VIX閾値
        vix_extreme: 極端VIX閾値（これ以上は最大キャッシュ）
        base_cash: 基本キャッシュ比率（最低限保持）
        max_cash: 最大キャッシュ比率
        use_dynamic_thresholds: 動的閾値を使用するか
        lookback_days: 動的閾値計算のルックバック日数
        vix_ticker: VIXティッカーシンボル
    """

    vix_low: float = 15.0
    vix_mid: float = 20.0
    vix_high: float = 25.0
    vix_extreme: float = 35.0
    base_cash: float = 0.05
    max_cash: float = 0.50
    use_dynamic_thresholds: bool = False
    lookback_days: int = 252
    vix_ticker: str = "^VIX"

    def __post_init__(self) -> None:
        """バリデーション"""
        if not (self.vix_low < self.vix_mid < self.vix_high < self.vix_extreme):
            raise ValueError("VIX thresholds must be in ascending order")
        if not 0 <= self.base_cash < self.max_cash <= 1:
            raise ValueError("Cash ratios must be 0 <= base_cash < max_cash <= 1")


@dataclass
class VIXThresholds:
    """動的VIX閾値

    Attributes:
        low: 低閾値（25パーセンタイル）
        mid: 中閾値（50パーセンタイル）
        high: 高閾値（75パーセンタイル）
        extreme: 極端閾値（95パーセンタイル）
        calculated_from: 計算元データ期間
        n_observations: 観測数
    """

    low: float
    mid: float
    high: float
    extreme: float
    calculated_from: Optional[Tuple[datetime, datetime]] = None
    n_observations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "low": self.low,
            "mid": self.mid,
            "high": self.high,
            "extreme": self.extreme,
            "calculated_from": (
                (self.calculated_from[0].isoformat(), self.calculated_from[1].isoformat())
                if self.calculated_from else None
            ),
            "n_observations": self.n_observations,
        }


@dataclass
class VIXCashResult:
    """VIXキャッシュ配分結果

    Attributes:
        cash_ratio: 計算されたキャッシュ比率
        vix_current: 現在のVIX値
        vix_level: VIXレベル（low/mid/high/extreme）
        thresholds_used: 使用された閾値
        original_weights: 元の重み
        adjusted_weights: 調整後の重み
        metadata: 追加情報
    """

    cash_ratio: float
    vix_current: float
    vix_level: str
    thresholds_used: Dict[str, float]
    original_weights: Dict[str, float] = field(default_factory=dict)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "cash_ratio": self.cash_ratio,
            "vix_current": self.vix_current,
            "vix_level": self.vix_level,
            "thresholds_used": self.thresholds_used,
            "original_weights": self.original_weights,
            "adjusted_weights": self.adjusted_weights,
            "metadata": self.metadata,
        }


class VIXCashAllocator:
    """VIXベースのキャッシュ配分クラス

    VIX水準に応じて動的にキャッシュ比率を調整する。

    VIXレベルとキャッシュ比率:
    - VIX < 15:     5% (base_cash)
    - VIX 15-20:   10%
    - VIX 20-25:   20%
    - VIX 25-35:   35%
    - VIX > 35:    50% (max_cash)

    Example:
        allocator = VIXCashAllocator()

        # キャッシュ比率計算
        cash_ratio = allocator.compute_cash_ratio(vix_current=22.5)
        # -> 0.20 (20%)

        # 重み調整
        base_weights = {"AAPL": 0.5, "MSFT": 0.5}
        result = allocator.adjust_weights(base_weights, cash_ratio=0.20)
        # -> {"AAPL": 0.4, "MSFT": 0.4, "CASH": 0.2}
    """

    def __init__(self, config: VIXCashConfig | None = None) -> None:
        """初期化

        Args:
            config: 設定オブジェクト（Noneでデフォルト）
        """
        self.config = config or VIXCashConfig()
        self._dynamic_thresholds: Optional[VIXThresholds] = None
        self._vix_history: Optional[pd.Series] = None

    def compute_cash_ratio(
        self,
        vix_current: float,
        vix_history: pd.Series | None = None,
    ) -> float:
        """VIXからキャッシュ比率を計算

        Args:
            vix_current: 現在のVIX値
            vix_history: VIX履歴データ（動的閾値使用時）

        Returns:
            キャッシュ比率（0-1）
        """
        # 動的閾値の計算
        if self.config.use_dynamic_thresholds and vix_history is not None:
            thresholds = self.compute_dynamic_thresholds(vix_history)
            vix_low = thresholds.low
            vix_mid = thresholds.mid
            vix_high = thresholds.high
            vix_extreme = thresholds.extreme
        else:
            vix_low = self.config.vix_low
            vix_mid = self.config.vix_mid
            vix_high = self.config.vix_high
            vix_extreme = self.config.vix_extreme

        # キャッシュ比率の段階的計算
        base_cash = self.config.base_cash
        max_cash = self.config.max_cash

        if vix_current < vix_low:
            # 低VIX: 最小キャッシュ
            cash_ratio = base_cash
        elif vix_current < vix_mid:
            # 低〜中: 5% → 10%
            progress = (vix_current - vix_low) / (vix_mid - vix_low)
            cash_ratio = base_cash + progress * (0.10 - base_cash)
        elif vix_current < vix_high:
            # 中〜高: 10% → 20%
            progress = (vix_current - vix_mid) / (vix_high - vix_mid)
            cash_ratio = 0.10 + progress * (0.20 - 0.10)
        elif vix_current < vix_extreme:
            # 高〜極端: 20% → 35%
            progress = (vix_current - vix_high) / (vix_extreme - vix_high)
            cash_ratio = 0.20 + progress * (0.35 - 0.20)
        else:
            # 極端VIX: 最大キャッシュ
            cash_ratio = max_cash

        logger.debug(
            "VIX cash ratio: vix=%.1f, ratio=%.2f",
            vix_current,
            cash_ratio,
        )

        return cash_ratio

    def compute_dynamic_thresholds(
        self,
        vix_history: pd.Series,
        lookback_days: int | None = None,
    ) -> VIXThresholds:
        """過去のVIX分布から動的閾値を計算

        パーセンタイルベース:
        - 25%: 低閾値
        - 50%: 中閾値
        - 75%: 高閾値
        - 95%: 極端閾値

        Args:
            vix_history: VIX履歴データ（Seriesまたは'close'列を持つDataFrame）
            lookback_days: ルックバック日数

        Returns:
            VIXThresholds
        """
        lookback = lookback_days or self.config.lookback_days

        # ルックバック期間でフィルタ
        if len(vix_history) > lookback:
            vix_data = vix_history.tail(lookback)
        else:
            vix_data = vix_history

        # NaN除去
        vix_data = vix_data.dropna()

        if len(vix_data) < 20:
            logger.warning(
                "Insufficient VIX history (%d), using default thresholds",
                len(vix_data),
            )
            return VIXThresholds(
                low=self.config.vix_low,
                mid=self.config.vix_mid,
                high=self.config.vix_high,
                extreme=self.config.vix_extreme,
                n_observations=len(vix_data),
            )

        # パーセンタイル計算
        thresholds = VIXThresholds(
            low=float(np.percentile(vix_data, 25)),
            mid=float(np.percentile(vix_data, 50)),
            high=float(np.percentile(vix_data, 75)),
            extreme=float(np.percentile(vix_data, 95)),
            calculated_from=(vix_data.index.min(), vix_data.index.max()),
            n_observations=len(vix_data),
        )

        self._dynamic_thresholds = thresholds

        logger.info(
            "Dynamic VIX thresholds: low=%.1f, mid=%.1f, high=%.1f, extreme=%.1f (n=%d)",
            thresholds.low,
            thresholds.mid,
            thresholds.high,
            thresholds.extreme,
            thresholds.n_observations,
        )

        return thresholds

    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        cash_ratio: float,
        cash_key: str = "CASH",
    ) -> Dict[str, float]:
        """キャッシュ比率に基づいて配分調整

        リスク資産を比例縮小し、キャッシュを追加。

        Args:
            base_weights: 元の重み（合計1.0を想定）
            cash_ratio: 目標キャッシュ比率
            cash_key: キャッシュのキー名

        Returns:
            調整後の重み（キャッシュ込み、合計1.0）
        """
        # 既存のキャッシュを除外して計算
        risk_weights = {k: v for k, v in base_weights.items() if k != cash_key}
        existing_cash = base_weights.get(cash_key, 0.0)

        # リスク資産の合計
        total_risk = sum(risk_weights.values())

        if total_risk <= 0:
            # リスク資産がない場合は全額キャッシュ
            return {cash_key: 1.0}

        # リスク資産のスケールファクター
        # 新しいリスク資産比率 = 1 - cash_ratio
        risk_scale = (1 - cash_ratio) / total_risk

        # 調整後の重み
        adjusted = {k: v * risk_scale for k, v in risk_weights.items()}
        adjusted[cash_key] = cash_ratio

        logger.debug(
            "Weights adjusted: cash_ratio=%.2f, risk_scale=%.3f",
            cash_ratio,
            risk_scale,
        )

        return adjusted

    def get_vix_level(self, vix_current: float) -> str:
        """VIX値からレベルを判定

        Args:
            vix_current: 現在のVIX値

        Returns:
            レベル文字列（low/mid/high/extreme）
        """
        thresholds = self._dynamic_thresholds or VIXThresholds(
            low=self.config.vix_low,
            mid=self.config.vix_mid,
            high=self.config.vix_high,
            extreme=self.config.vix_extreme,
        )

        if vix_current < thresholds.low:
            return "very_low"
        elif vix_current < thresholds.mid:
            return "low"
        elif vix_current < thresholds.high:
            return "mid"
        elif vix_current < thresholds.extreme:
            return "high"
        else:
            return "extreme"

    def process(
        self,
        vix_current: float,
        base_weights: Dict[str, float],
        vix_history: pd.Series | None = None,
    ) -> VIXCashResult:
        """一括処理: キャッシュ比率計算 → 重み調整

        Args:
            vix_current: 現在のVIX値
            base_weights: 元の重み
            vix_history: VIX履歴（動的閾値用）

        Returns:
            VIXCashResult
        """
        # キャッシュ比率計算
        cash_ratio = self.compute_cash_ratio(vix_current, vix_history)

        # 重み調整
        adjusted_weights = self.adjust_weights(base_weights, cash_ratio)

        # 閾値情報
        thresholds = self._dynamic_thresholds or VIXThresholds(
            low=self.config.vix_low,
            mid=self.config.vix_mid,
            high=self.config.vix_high,
            extreme=self.config.vix_extreme,
        )

        return VIXCashResult(
            cash_ratio=cash_ratio,
            vix_current=vix_current,
            vix_level=self.get_vix_level(vix_current),
            thresholds_used=thresholds.to_dict(),
            original_weights=base_weights,
            adjusted_weights=adjusted_weights,
            metadata={
                "use_dynamic_thresholds": self.config.use_dynamic_thresholds,
                "risk_asset_scale": 1 - cash_ratio,
            },
        )


# =============================================================================
# VIXデータ取得
# =============================================================================


def fetch_vix_data(
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
    lookback_days: int = 252,
) -> pd.Series:
    """yfinanceからVIXデータを取得

    Args:
        start_date: 開始日（Noneの場合はlookback_days前）
        end_date: 終了日（Noneの場合は今日）
        lookback_days: ルックバック日数（start_dateがNoneの場合使用）

    Returns:
        VIX終値のSeries
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        raise ImportError("yfinance is required for VIX data fetching")

    # 日付設定
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    if start_date is None:
        start_date = end_date - timedelta(days=lookback_days)
    elif isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    # VIXデータ取得
    ticker = yf.Ticker("^VIX")
    hist = ticker.history(start=start_date, end=end_date)

    if hist.empty:
        logger.warning("No VIX data returned from yfinance")
        return pd.Series(dtype=float)

    vix_series = hist["Close"]
    vix_series.name = "VIX"

    logger.info(
        "VIX data fetched: %d observations from %s to %s",
        len(vix_series),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    return vix_series


def get_current_vix() -> float | None:
    """現在のVIX値を取得

    Returns:
        現在のVIX値（取得失敗時はNone）
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker("^VIX")
        # 直近1日のデータを取得
        hist = ticker.history(period="1d")

        if hist.empty:
            logger.warning("Could not fetch current VIX")
            return None

        return float(hist["Close"].iloc[-1])

    except Exception as e:
        logger.error(f"Failed to fetch current VIX: {e}")
        return None


# =============================================================================
# 便利関数
# =============================================================================


def compute_vix_cash_ratio(
    vix_current: float,
    vix_low: float = 15.0,
    vix_mid: float = 20.0,
    vix_high: float = 25.0,
    vix_extreme: float = 35.0,
    base_cash: float = 0.05,
    max_cash: float = 0.50,
) -> float:
    """VIXからキャッシュ比率を計算（ショートカット関数）

    Args:
        vix_current: 現在のVIX値
        vix_low: 低閾値
        vix_mid: 中閾値
        vix_high: 高閾値
        vix_extreme: 極端閾値
        base_cash: 基本キャッシュ
        max_cash: 最大キャッシュ

    Returns:
        キャッシュ比率
    """
    config = VIXCashConfig(
        vix_low=vix_low,
        vix_mid=vix_mid,
        vix_high=vix_high,
        vix_extreme=vix_extreme,
        base_cash=base_cash,
        max_cash=max_cash,
    )
    allocator = VIXCashAllocator(config)
    return allocator.compute_cash_ratio(vix_current)


def apply_vix_cash_allocation(
    base_weights: Dict[str, float],
    vix_current: float,
    vix_history: pd.Series | None = None,
    use_dynamic: bool = False,
) -> Dict[str, float]:
    """VIXベースのキャッシュ配分を適用（ショートカット関数）

    Args:
        base_weights: 元の重み
        vix_current: 現在のVIX値
        vix_history: VIX履歴（動的閾値用）
        use_dynamic: 動的閾値を使用するか

    Returns:
        調整後の重み
    """
    config = VIXCashConfig(use_dynamic_thresholds=use_dynamic)
    allocator = VIXCashAllocator(config)
    result = allocator.process(vix_current, base_weights, vix_history)
    return result.adjusted_weights
