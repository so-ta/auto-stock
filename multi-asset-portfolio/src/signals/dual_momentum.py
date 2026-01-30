"""
Enhanced Dual Momentum Module - デュアルモメンタム強化版

Gary Antonacci のデュアルモメンタム戦略を拡張し、
複数期間の絶対モメンタムと相対モメンタムを組み合わせた
高度なモメンタム戦略を実装する。

主な機能:
- 複数期間の絶対モメンタム（短期に高い重み）
- ユニバース内での相対モメンタムランキング
- 安全資産への逃避メカニズム

設計根拠:
- 要求.md §5: シグナル生成
- Antonacci (2014) "Dual Momentum Investing"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MomentumSignal(Enum):
    """モメンタムシグナルの種類"""
    BULLISH = "bullish"       # 強気（リスクオン）
    BEARISH = "bearish"       # 弱気（リスクオフ）
    NEUTRAL = "neutral"       # 中立


@dataclass(frozen=True)
class DualMomentumConfig:
    """デュアルモメンタム設定

    Attributes:
        abs_lookbacks: 絶対モメンタム計算用のルックバック期間リスト
        abs_weights: 各ルックバック期間の重み（短期ほど高い重み）
        rel_lookback: 相対モメンタム計算用のルックバック期間
        safe_asset: 安全資産のティッカー（絶対モメンタム < 0 時に逃避）
        top_n_selection: 相対モメンタム上位から選択する銘柄数
        momentum_threshold: モメンタム閾値（これ以下は弱気と判定）
        use_excess_return: 安全資産利回りを控除するか
    """
    abs_lookbacks: tuple[int, ...] = (60, 120, 252)
    abs_weights: tuple[float, ...] = (0.5, 0.3, 0.2)  # 短期に高い重み
    rel_lookback: int = 60
    safe_asset: str = "BIL"
    top_n_selection: int = 3
    momentum_threshold: float = 0.0
    use_excess_return: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        if len(self.abs_lookbacks) != len(self.abs_weights):
            raise ValueError(
                "abs_lookbacks and abs_weights must have the same length"
            )
        if abs(sum(self.abs_weights) - 1.0) > 1e-6:
            raise ValueError("abs_weights must sum to 1.0")
        if self.rel_lookback <= 0:
            raise ValueError("rel_lookback must be positive")
        if self.top_n_selection <= 0:
            raise ValueError("top_n_selection must be positive")


@dataclass
class MomentumResult:
    """モメンタム計算結果

    Attributes:
        ticker: ティッカーシンボル
        absolute_momentum: 加重平均された絶対モメンタム
        absolute_momentums: 各期間の絶対モメンタム
        relative_momentum: 相対モメンタム（ユニバース内ランク）
        relative_rank: ユニバース内順位（1が最高）
        signal: モメンタムシグナル（bullish/bearish/neutral）
        is_selected: 最終選択されたか
    """
    ticker: str
    absolute_momentum: float
    absolute_momentums: Dict[int, float] = field(default_factory=dict)
    relative_momentum: float = 0.0
    relative_rank: int = 0
    signal: MomentumSignal = MomentumSignal.NEUTRAL
    is_selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ticker": self.ticker,
            "absolute_momentum": self.absolute_momentum,
            "absolute_momentums": self.absolute_momentums,
            "relative_momentum": self.relative_momentum,
            "relative_rank": self.relative_rank,
            "signal": self.signal.value,
            "is_selected": self.is_selected,
        }


@dataclass
class DualMomentumSignals:
    """デュアルモメンタムシグナル結果

    Attributes:
        date: シグナル計算日
        results: 各銘柄のモメンタム結果
        selected_tickers: 選択された銘柄リスト
        safe_asset_selected: 安全資産が選択されたか
        allocation: 推奨アロケーション（ticker -> weight）
        metadata: 追加メタデータ
    """
    date: pd.Timestamp
    results: List[MomentumResult] = field(default_factory=list)
    selected_tickers: List[str] = field(default_factory=list)
    safe_asset_selected: bool = False
    allocation: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "date": self.date.isoformat() if self.date else None,
            "results": [r.to_dict() for r in self.results],
            "selected_tickers": self.selected_tickers,
            "safe_asset_selected": self.safe_asset_selected,
            "allocation": self.allocation,
            "metadata": self.metadata,
        }


class EnhancedDualMomentum:
    """デュアルモメンタム強化版クラス

    Gary Antonacci のデュアルモメンタム戦略を拡張し、
    複数期間の絶対モメンタムと相対モメンタムを組み合わせる。

    戦略ロジック:
    1. 絶対モメンタム < 0 → 安全資産（BIL等）へ100%逃避
    2. 絶対モメンタム > 0 → 相対モメンタム上位N銘柄に配分

    Usage:
        config = DualMomentumConfig(
            abs_lookbacks=(60, 120, 252),
            rel_lookback=60,
            safe_asset="BIL",
            top_n_selection=3,
        )
        dm = EnhancedDualMomentum(config)

        # 価格データ（DataFrame: index=date, columns=tickers）
        prices = pd.DataFrame(...)
        universe = ["SPY", "QQQ", "EFA", "EEM", "VNQ"]

        # シグナル生成
        signals = dm.generate_signals(prices, universe)
        print(signals.allocation)

    Attributes:
        config: デュアルモメンタム設定
    """

    def __init__(self, config: Optional[DualMomentumConfig] = None) -> None:
        """初期化

        Args:
            config: デュアルモメンタム設定。Noneの場合はデフォルト値を使用
        """
        self.config = config or DualMomentumConfig()

    def calculate_absolute_momentum(
        self,
        prices: pd.DataFrame,
        ticker: str,
        lookbacks: Optional[tuple[int, ...]] = None,
        weights: Optional[tuple[float, ...]] = None,
    ) -> tuple[float, Dict[int, float]]:
        """絶対モメンタムを計算

        複数期間のモメンタムを加重平均して計算する。
        短期のモメンタムに高い重みを付けることで、
        トレンド転換への反応を早める。

        Args:
            prices: 価格データ（index=date, columns=tickers）
            ticker: 対象ティッカー
            lookbacks: ルックバック期間リスト。Noneの場合は設定値を使用
            weights: 各期間の重み。Noneの場合は設定値を使用

        Returns:
            (加重平均モメンタム, 各期間のモメンタム辞書)

        計算式:
            momentum[t, n] = (price[t] - price[t-n]) / price[t-n]
            weighted_momentum = Σ(weight[i] * momentum[t, lookback[i]])
        """
        lookbacks = lookbacks or self.config.abs_lookbacks
        weights = weights or self.config.abs_weights

        if ticker not in prices.columns:
            logger.warning("Ticker %s not found in prices", ticker)
            return 0.0, {}

        price_series = prices[ticker].dropna()
        if len(price_series) < max(lookbacks):
            logger.warning(
                "Insufficient data for ticker %s: %d < %d",
                ticker, len(price_series), max(lookbacks)
            )
            return 0.0, {}

        momentums: Dict[int, float] = {}
        for lb in lookbacks:
            if len(price_series) >= lb:
                current_price = price_series.iloc[-1]
                past_price = price_series.iloc[-lb]
                if past_price > 0:
                    momentums[lb] = (current_price - past_price) / past_price
                else:
                    momentums[lb] = 0.0
            else:
                momentums[lb] = 0.0

        # 加重平均
        weighted_momentum = sum(
            w * momentums.get(lb, 0.0)
            for w, lb in zip(weights, lookbacks)
        )

        return weighted_momentum, momentums

    def calculate_relative_momentum(
        self,
        prices: pd.DataFrame,
        universe: List[str],
        lookback: Optional[int] = None,
    ) -> Dict[str, tuple[float, int]]:
        """相対モメンタムを計算

        ユニバース内での相対的なパフォーマンスを評価し、
        ランキングを付ける。

        Args:
            prices: 価格データ（index=date, columns=tickers）
            universe: 評価対象のティッカーリスト
            lookback: ルックバック期間。Noneの場合は設定値を使用

        Returns:
            {ticker: (relative_momentum, rank)} の辞書
            rankは1が最高（最も高いモメンタム）

        計算式:
            relative_momentum[ticker] = (price[t] - price[t-n]) / price[t-n]
            rank = ユニバース内での順位（降順）
        """
        lookback = lookback or self.config.rel_lookback

        momentum_dict: Dict[str, float] = {}
        for ticker in universe:
            if ticker not in prices.columns:
                continue

            price_series = prices[ticker].dropna()
            if len(price_series) < lookback:
                momentum_dict[ticker] = float("-inf")
                continue

            current_price = price_series.iloc[-1]
            past_price = price_series.iloc[-lookback]
            if past_price > 0:
                momentum_dict[ticker] = (current_price - past_price) / past_price
            else:
                momentum_dict[ticker] = 0.0

        # ランキング付け（降順）
        sorted_tickers = sorted(
            momentum_dict.keys(),
            key=lambda x: momentum_dict[x],
            reverse=True,
        )

        result: Dict[str, tuple[float, int]] = {}
        for rank, ticker in enumerate(sorted_tickers, start=1):
            result[ticker] = (momentum_dict[ticker], rank)

        return result

    def generate_signals(
        self,
        prices: pd.DataFrame,
        universe: Optional[List[str]] = None,
        safe_asset: Optional[str] = None,
    ) -> DualMomentumSignals:
        """デュアルモメンタムシグナルを生成

        絶対モメンタムと相対モメンタムを組み合わせて、
        投資シグナルを生成する。

        Args:
            prices: 価格データ（index=date, columns=tickers）
            universe: 評価対象のティッカーリスト。
                      Noneの場合はpricesの全カラムを使用
            safe_asset: 安全資産のティッカー。Noneの場合は設定値を使用

        Returns:
            DualMomentumSignals: シグナル結果

        戦略ロジック:
        1. 各銘柄の絶対モメンタム（加重平均）を計算
        2. 各銘柄の相対モメンタムとランキングを計算
        3. 絶対モメンタム < 0 の銘柄は除外
        4. 残った銘柄から相対モメンタム上位N銘柄を選択
        5. 全銘柄の絶対モメンタム < 0 なら安全資産へ逃避
        """
        safe_asset = safe_asset or self.config.safe_asset
        if universe is None:
            universe = [
                col for col in prices.columns
                if col != safe_asset
            ]

        if prices.empty:
            logger.warning("Empty prices DataFrame")
            return DualMomentumSignals(
                date=pd.Timestamp.now(),
                metadata={"error": "Empty prices"},
            )

        current_date = prices.index[-1]
        results: List[MomentumResult] = []

        # 1. 各銘柄の絶対モメンタムを計算
        abs_momentums: Dict[str, float] = {}
        for ticker in universe:
            abs_mom, abs_moms = self.calculate_absolute_momentum(
                prices, ticker
            )
            abs_momentums[ticker] = abs_mom

            result = MomentumResult(
                ticker=ticker,
                absolute_momentum=abs_mom,
                absolute_momentums=abs_moms,
            )
            results.append(result)

        # 2. 相対モメンタムを計算
        rel_momentums = self.calculate_relative_momentum(prices, universe)
        for result in results:
            if result.ticker in rel_momentums:
                rel_mom, rank = rel_momentums[result.ticker]
                result.relative_momentum = rel_mom
                result.relative_rank = rank

        # 3. シグナル判定と銘柄選択
        bullish_tickers: List[str] = []
        for result in results:
            if result.absolute_momentum > self.config.momentum_threshold:
                result.signal = MomentumSignal.BULLISH
                bullish_tickers.append(result.ticker)
            elif result.absolute_momentum < -self.config.momentum_threshold:
                result.signal = MomentumSignal.BEARISH
            else:
                result.signal = MomentumSignal.NEUTRAL

        # 4. 上位N銘柄を選択
        selected_tickers: List[str] = []
        safe_asset_selected = False
        allocation: Dict[str, float] = {}

        if bullish_tickers:
            # 相対モメンタムでソート
            bullish_results = [
                r for r in results
                if r.ticker in bullish_tickers
            ]
            bullish_results.sort(key=lambda x: x.relative_rank)

            # 上位N銘柄を選択
            top_n = min(self.config.top_n_selection, len(bullish_results))
            selected_tickers = [r.ticker for r in bullish_results[:top_n]]

            # 均等配分
            weight = 1.0 / len(selected_tickers)
            for ticker in selected_tickers:
                allocation[ticker] = weight

            # 選択フラグを設定
            for result in results:
                if result.ticker in selected_tickers:
                    result.is_selected = True
        else:
            # 全て弱気 → 安全資産へ逃避
            safe_asset_selected = True
            selected_tickers = [safe_asset]
            allocation[safe_asset] = 1.0

        # メタデータ
        metadata = {
            "n_universe": len(universe),
            "n_bullish": len(bullish_tickers),
            "n_selected": len(selected_tickers),
            "avg_abs_momentum": (
                sum(abs_momentums.values()) / len(abs_momentums)
                if abs_momentums else 0.0
            ),
            "config": {
                "abs_lookbacks": self.config.abs_lookbacks,
                "abs_weights": self.config.abs_weights,
                "rel_lookback": self.config.rel_lookback,
                "top_n_selection": self.config.top_n_selection,
            },
        }

        logger.info(
            "Generated dual momentum signals: %d bullish, %d selected, "
            "safe_asset=%s",
            len(bullish_tickers),
            len(selected_tickers),
            safe_asset_selected,
        )

        return DualMomentumSignals(
            date=current_date,
            results=results,
            selected_tickers=selected_tickers,
            safe_asset_selected=safe_asset_selected,
            allocation=allocation,
            metadata=metadata,
        )

    def backtest_signals(
        self,
        prices: pd.DataFrame,
        universe: List[str],
        rebalance_freq: str = "M",
        safe_asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """シグナルをバックテスト

        指定した頻度でリバランスしながら、
        デュアルモメンタム戦略のシグナル履歴を生成する。

        Args:
            prices: 価格データ（index=date, columns=tickers）
            universe: 評価対象のティッカーリスト
            rebalance_freq: リバランス頻度（'M'=月次, 'W'=週次, 'Q'=四半期）
            safe_asset: 安全資産のティッカー

        Returns:
            pd.DataFrame: シグナル履歴
                columns: ['date', 'selected_tickers', 'safe_asset_selected',
                          'allocation', 'avg_abs_momentum']
        """
        safe_asset = safe_asset or self.config.safe_asset

        # リバランス日を取得
        rebalance_dates = prices.resample(rebalance_freq).last().index

        signal_history: List[Dict[str, Any]] = []

        for rebal_date in rebalance_dates:
            # リバランス日までのデータを使用
            prices_subset = prices.loc[:rebal_date]

            if len(prices_subset) < max(self.config.abs_lookbacks):
                continue

            signals = self.generate_signals(
                prices_subset, universe, safe_asset
            )

            signal_history.append({
                "date": rebal_date,
                "selected_tickers": signals.selected_tickers,
                "safe_asset_selected": signals.safe_asset_selected,
                "allocation": signals.allocation,
                "avg_abs_momentum": signals.metadata.get("avg_abs_momentum", 0.0),
                "n_bullish": signals.metadata.get("n_bullish", 0),
            })

        return pd.DataFrame(signal_history)


# ============================================================
# 便利関数
# ============================================================

def create_dual_momentum(
    abs_lookbacks: tuple[int, ...] = (60, 120, 252),
    rel_lookback: int = 60,
    safe_asset: str = "BIL",
    top_n_selection: int = 3,
) -> EnhancedDualMomentum:
    """デュアルモメンタムインスタンスを作成

    Args:
        abs_lookbacks: 絶対モメンタム計算用のルックバック期間リスト
        rel_lookback: 相対モメンタム計算用のルックバック期間
        safe_asset: 安全資産のティッカー
        top_n_selection: 相対モメンタム上位から選択する銘柄数

    Returns:
        EnhancedDualMomentum インスタンス
    """
    # 短期ほど高い重みを自動計算
    n = len(abs_lookbacks)
    weights_raw = [1.0 / (i + 1) for i in range(n)]
    total = sum(weights_raw)
    weights = tuple(w / total for w in weights_raw)

    config = DualMomentumConfig(
        abs_lookbacks=abs_lookbacks,
        abs_weights=weights,
        rel_lookback=rel_lookback,
        safe_asset=safe_asset,
        top_n_selection=top_n_selection,
    )

    return EnhancedDualMomentum(config)


def get_momentum_allocation(
    prices: pd.DataFrame,
    universe: List[str],
    safe_asset: str = "BIL",
    top_n: int = 3,
) -> Dict[str, float]:
    """モメンタムに基づくアロケーションを取得（簡易版）

    Args:
        prices: 価格データ
        universe: ユニバース
        safe_asset: 安全資産
        top_n: 選択する銘柄数

    Returns:
        {ticker: weight} のアロケーション辞書
    """
    dm = create_dual_momentum(
        safe_asset=safe_asset,
        top_n_selection=top_n,
    )
    signals = dm.generate_signals(prices, universe, safe_asset)
    return signals.allocation


def calculate_momentum_scores(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (60, 120, 252),
) -> pd.DataFrame:
    """全銘柄のモメンタムスコアを計算

    Args:
        prices: 価格データ（index=date, columns=tickers）
        lookbacks: ルックバック期間リスト

    Returns:
        pd.DataFrame: 各銘柄のモメンタムスコア
            columns: ['ticker', 'momentum_60', 'momentum_120', 'momentum_252',
                      'weighted_momentum']
    """
    dm = create_dual_momentum(abs_lookbacks=lookbacks)

    scores: List[Dict[str, Any]] = []
    for ticker in prices.columns:
        weighted_mom, moms = dm.calculate_absolute_momentum(prices, ticker)
        row = {"ticker": ticker, "weighted_momentum": weighted_mom}
        for lb, mom in moms.items():
            row[f"momentum_{lb}"] = mom
        scores.append(row)

    return pd.DataFrame(scores)
