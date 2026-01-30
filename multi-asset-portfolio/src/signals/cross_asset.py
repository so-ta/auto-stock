"""
Cross Asset Momentum - クロスアセットモメンタム

このモジュールは、複数のアセットクラス間のモメンタムを計測し、
相対的なランキングに基づいて配分調整を行う機能を提供する。

主要コンポーネント:
- CrossAssetMomentumRanker: アセットクラス間のモメンタムランキング
- AssetClass: アセットクラスの定義と管理
- AllocationAdjustment: 配分調整結果

設計根拠:
- アセットクラス間のモメンタムは個別銘柄より安定
- クロスセクショナルランキングでタイミングリスクを軽減
- 配分調整により動的なリスク管理を実現
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


class AssetClassType(str, Enum):
    """アセットクラスの種類。"""

    US_EQUITY = "us_equity"
    INTL_EQUITY = "intl_equity"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    REAL_ESTATE = "real_estate"
    CASH = "cash"


@dataclass
class AssetClass:
    """アセットクラスの定義。

    Attributes:
        name: アセットクラス名
        asset_type: アセットクラスの種類
        tickers: 構成銘柄のティッカーリスト
        description: 説明
    """

    name: str
    asset_type: AssetClassType
    tickers: list[str]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(f"Asset class '{self.name}' must have at least one ticker")


# デフォルトのアセットクラス定義
DEFAULT_ASSET_CLASSES: dict[AssetClassType, AssetClass] = {
    AssetClassType.US_EQUITY: AssetClass(
        name="US Equity",
        asset_type=AssetClassType.US_EQUITY,
        tickers=["SPY", "QQQ", "IWM"],
        description="米国株式（大型、テック、小型）",
    ),
    AssetClassType.INTL_EQUITY: AssetClass(
        name="International Equity",
        asset_type=AssetClassType.INTL_EQUITY,
        tickers=["EFA", "EEM", "VWO"],
        description="国際株式（先進国、新興国）",
    ),
    AssetClassType.BONDS: AssetClass(
        name="Bonds",
        asset_type=AssetClassType.BONDS,
        tickers=["TLT", "IEF", "LQD", "HYG"],
        description="債券（長期国債、中期国債、投資適格社債、ハイイールド）",
    ),
    AssetClassType.COMMODITIES: AssetClass(
        name="Commodities",
        asset_type=AssetClassType.COMMODITIES,
        tickers=["GLD", "SLV", "USO", "DBA"],
        description="コモディティ（金、銀、原油、農産物）",
    ),
    AssetClassType.REAL_ESTATE: AssetClass(
        name="Real Estate",
        asset_type=AssetClassType.REAL_ESTATE,
        tickers=["VNQ", "IYR"],
        description="不動産（REIT）",
    ),
    AssetClassType.CASH: AssetClass(
        name="Cash",
        asset_type=AssetClassType.CASH,
        tickers=["SHY", "BIL"],
        description="現金同等物（短期国債）",
    ),
}


@dataclass
class AssetClassScore:
    """アセットクラスのスコア。

    Attributes:
        asset_type: アセットクラスの種類
        raw_return: 生のリターン（期間リターン）
        rank: クロスセクショナルランク（0-1、1が最高）
        normalized_score: 正規化スコア（-1 to +1）
        constituents_returns: 構成銘柄の個別リターン
    """

    asset_type: AssetClassType
    raw_return: float
    rank: float
    normalized_score: float
    constituents_returns: dict[str, float] = field(default_factory=dict)


@dataclass
class RankingResult:
    """ランキング結果。

    Attributes:
        scores: アセットクラス別スコア
        lookback: 使用したルックバック期間
        timestamp: 計算日時
        top_classes: 上位アセットクラス（スコア順）
        bottom_classes: 下位アセットクラス（スコア順）
    """

    scores: dict[AssetClassType, AssetClassScore]
    lookback: int
    timestamp: datetime
    top_classes: list[AssetClassType] = field(default_factory=list)
    bottom_classes: list[AssetClassType] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "scores": {
                k.value: {
                    "raw_return": v.raw_return,
                    "rank": v.rank,
                    "normalized_score": v.normalized_score,
                }
                for k, v in self.scores.items()
            },
            "lookback": self.lookback,
            "timestamp": self.timestamp.isoformat(),
            "top_classes": [c.value for c in self.top_classes],
            "bottom_classes": [c.value for c in self.bottom_classes],
        }

    def get_score_series(self) -> pd.Series:
        """スコアをpd.Seriesで返す。"""
        return pd.Series(
            {k.value: v.normalized_score for k, v in self.scores.items()}
        )


@dataclass
class AllocationAdjustment:
    """配分調整結果。

    Attributes:
        original_weights: 元のウェイト
        adjusted_weights: 調整後のウェイト
        adjustments: 調整量
        class_scores: 使用されたクラススコア
        adjustment_strength: 使用された調整強度
    """

    original_weights: dict[str, float]
    adjusted_weights: dict[str, float]
    adjustments: dict[str, float]
    class_scores: dict[AssetClassType, float]
    adjustment_strength: float

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "original_weights": self.original_weights,
            "adjusted_weights": self.adjusted_weights,
            "adjustments": self.adjustments,
            "class_scores": {k.value: v for k, v in self.class_scores.items()},
            "adjustment_strength": self.adjustment_strength,
        }


class CrossAssetMomentumRanker:
    """クロスアセットモメンタムランカー。

    アセットクラス間のモメンタムを計測し、
    クロスセクショナルランキングを生成する。

    Usage:
        ranker = CrossAssetMomentumRanker()

        # ランキング計算
        result = ranker.rank_asset_classes(prices, lookback=60)
        print(result.top_classes)

        # 配分調整
        adjustment = ranker.generate_allocation_adjustment(
            base_weights={"SPY": 0.3, "TLT": 0.3, "GLD": 0.2, "VNQ": 0.2},
            class_scores=result.scores,
            adjustment_strength=0.3,
        )
        print(adjustment.adjusted_weights)
    """

    def __init__(
        self,
        asset_classes: dict[AssetClassType, AssetClass] | None = None,
        ticker_to_class: dict[str, AssetClassType] | None = None,
    ) -> None:
        """初期化。

        Args:
            asset_classes: カスタムアセットクラス定義
            ticker_to_class: ティッカーからアセットクラスへのマッピング
        """
        self.asset_classes = asset_classes or DEFAULT_ASSET_CLASSES
        self._ticker_to_class = ticker_to_class or self._build_ticker_mapping()

    def _build_ticker_mapping(self) -> dict[str, AssetClassType]:
        """ティッカーからアセットクラスへのマッピングを構築する。"""
        mapping = {}
        for asset_type, asset_class in self.asset_classes.items():
            for ticker in asset_class.tickers:
                mapping[ticker] = asset_type
        return mapping

    def get_asset_class(self, ticker: str) -> AssetClassType | None:
        """ティッカーのアセットクラスを取得する。"""
        return self._ticker_to_class.get(ticker)

    def rank_asset_classes(
        self,
        prices: pd.DataFrame,
        lookback: int = 60,
    ) -> RankingResult:
        """アセットクラスをモメンタムでランキングする。

        Args:
            prices: 価格データ（columns=ティッカー）
            lookback: ルックバック期間（日数）

        Returns:
            ランキング結果
        """
        if len(prices) < lookback:
            logger.warning(
                f"Insufficient data for lookback={lookback}, using {len(prices)-1} days"
            )
            lookback = max(1, len(prices) - 1)

        # アセットクラス別のリターンを計算
        class_returns: dict[AssetClassType, list[float]] = {
            at: [] for at in self.asset_classes.keys()
        }
        constituents_returns: dict[AssetClassType, dict[str, float]] = {
            at: {} for at in self.asset_classes.keys()
        }

        for ticker in prices.columns:
            asset_type = self.get_asset_class(ticker)
            if asset_type is None:
                continue

            # ルックバック期間のリターン
            try:
                ret = prices[ticker].iloc[-1] / prices[ticker].iloc[-lookback] - 1
                if not np.isnan(ret):
                    class_returns[asset_type].append(ret)
                    constituents_returns[asset_type][ticker] = ret
            except Exception as e:
                logger.warning(f"Failed to calculate return for {ticker}: {e}")

        # クラス平均リターン
        avg_returns: dict[AssetClassType, float] = {}
        for asset_type, returns in class_returns.items():
            if returns:
                avg_returns[asset_type] = np.mean(returns)
            else:
                avg_returns[asset_type] = 0.0

        # クロスセクショナルランク
        return_series = pd.Series(avg_returns)
        ranks = return_series.rank(pct=True)

        # スコア計算（-1 to +1に正規化）
        scores: dict[AssetClassType, AssetClassScore] = {}
        for asset_type in self.asset_classes.keys():
            raw_return = avg_returns.get(asset_type, 0.0)
            rank = ranks.get(asset_type, 0.5)
            normalized_score = (rank - 0.5) * 2  # 0-1 -> -1 to +1

            scores[asset_type] = AssetClassScore(
                asset_type=asset_type,
                raw_return=raw_return,
                rank=rank,
                normalized_score=normalized_score,
                constituents_returns=constituents_returns.get(asset_type, {}),
            )

        # 上位・下位クラスの決定
        sorted_classes = sorted(
            scores.items(),
            key=lambda x: x[1].normalized_score,
            reverse=True,
        )
        top_classes = [c[0] for c in sorted_classes[:2]]
        bottom_classes = [c[0] for c in sorted_classes[-2:]]

        logger.debug(
            f"Asset class ranking: top={[c.value for c in top_classes]}, "
            f"bottom={[c.value for c in bottom_classes]}"
        )

        return RankingResult(
            scores=scores,
            lookback=lookback,
            timestamp=datetime.now(),
            top_classes=top_classes,
            bottom_classes=bottom_classes,
        )

    def generate_allocation_adjustment(
        self,
        base_weights: dict[str, float],
        class_scores: dict[AssetClassType, AssetClassScore] | RankingResult,
        adjustment_strength: float = 0.3,
    ) -> AllocationAdjustment:
        """モメンタムスコアに基づいて配分を調整する。

        Args:
            base_weights: ベースとなるウェイト（ticker -> weight）
            class_scores: アセットクラススコア（RankingResultも可）
            adjustment_strength: 調整強度（0-1）

        Returns:
            配分調整結果
        """
        # RankingResultからスコアを取り出す
        if isinstance(class_scores, RankingResult):
            class_scores = class_scores.scores

        # 調整強度のクリップ
        adjustment_strength = np.clip(adjustment_strength, 0.0, 1.0)

        # 各ティッカーの調整量を計算
        adjustments: dict[str, float] = {}
        class_scores_used: dict[AssetClassType, float] = {}

        for ticker, weight in base_weights.items():
            asset_type = self.get_asset_class(ticker)

            if asset_type is None:
                # アセットクラスが不明な場合は調整なし
                adjustments[ticker] = 0.0
                continue

            if asset_type not in class_scores:
                adjustments[ticker] = 0.0
                continue

            score = class_scores[asset_type]
            normalized_score = score.normalized_score

            # 調整量 = スコア × 調整強度 × ベースウェイト
            # 上位クラス（スコア > 0）: ウェイト増加
            # 下位クラス（スコア < 0）: ウェイト減少
            adjustment = normalized_score * adjustment_strength * weight
            adjustments[ticker] = adjustment
            class_scores_used[asset_type] = normalized_score

        # 調整後ウェイトの計算
        adjusted_weights: dict[str, float] = {}
        for ticker, weight in base_weights.items():
            adj = adjustments.get(ticker, 0.0)
            adjusted_weights[ticker] = max(0.0, weight + adj)

        # 正規化（合計=1.0）
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        else:
            # フォールバック: 元のウェイトを使用
            adjusted_weights = base_weights.copy()

        logger.debug(
            f"Allocation adjustment: strength={adjustment_strength:.2f}, "
            f"total_adjustment={sum(abs(a) for a in adjustments.values()):.4f}"
        )

        return AllocationAdjustment(
            original_weights=base_weights,
            adjusted_weights=adjusted_weights,
            adjustments=adjustments,
            class_scores=class_scores_used,
            adjustment_strength=adjustment_strength,
        )

    def compute_class_momentum_series(
        self,
        prices: pd.DataFrame,
        lookback: int = 60,
        window: int = 20,
    ) -> pd.DataFrame:
        """アセットクラスのモメンタム時系列を計算する。

        Args:
            prices: 価格データ
            lookback: モメンタム計算のルックバック期間
            window: ローリングウィンドウ

        Returns:
            アセットクラス別モメンタム時系列
        """
        # 各アセットクラスの代表値（等ウェイト平均）を計算
        class_prices: dict[str, pd.Series] = {}

        for asset_type, asset_class in self.asset_classes.items():
            available_tickers = [t for t in asset_class.tickers if t in prices.columns]
            if available_tickers:
                # 正規化して等ウェイト平均
                normalized = prices[available_tickers].div(
                    prices[available_tickers].iloc[0]
                )
                class_prices[asset_type.value] = normalized.mean(axis=1)

        if not class_prices:
            return pd.DataFrame()

        class_df = pd.DataFrame(class_prices)

        # ローリングモメンタム
        momentum = class_df.pct_change(lookback).rolling(window=window).mean()

        return momentum


def create_cross_asset_ranker(
    custom_classes: dict[AssetClassType, list[str]] | None = None,
) -> CrossAssetMomentumRanker:
    """CrossAssetMomentumRanker のファクトリ関数。

    Args:
        custom_classes: カスタムアセットクラス定義
            例: {AssetClassType.US_EQUITY: ["SPY", "IVV", "VOO"]}

    Returns:
        初期化された CrossAssetMomentumRanker
    """
    if custom_classes is None:
        return CrossAssetMomentumRanker()

    # カスタムクラスの構築
    asset_classes = {}
    for asset_type, tickers in custom_classes.items():
        default = DEFAULT_ASSET_CLASSES.get(asset_type)
        if default:
            asset_classes[asset_type] = AssetClass(
                name=default.name,
                asset_type=asset_type,
                tickers=tickers,
                description=default.description,
            )
        else:
            asset_classes[asset_type] = AssetClass(
                name=asset_type.value,
                asset_type=asset_type,
                tickers=tickers,
            )

    return CrossAssetMomentumRanker(asset_classes=asset_classes)


def quick_rank_asset_classes(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.Series:
    """便利関数: アセットクラスのスコアを簡易取得する。

    Args:
        prices: 価格データ
        lookback: ルックバック期間

    Returns:
        アセットクラス別スコア（-1 to +1）
    """
    ranker = CrossAssetMomentumRanker()
    result = ranker.rank_asset_classes(prices, lookback=lookback)
    return result.get_score_series()
