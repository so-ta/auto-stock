"""
Pairs Trading Module - 共和分ベースのペアトレーディング

Engle-Granger共和分検定に基づくペアトレーディング戦略。
統計的に関連のある2つの資産間のスプレッドを取引する。

主要コンポーネント:
- CointegrationPairsFinder: 共和分ペアの発見
- PairsTrader: ペアトレーディングシグナル生成
- PairsTradingStrategy: 統合戦略クラス

設計根拠:
- 共和分関係は長期的な均衡を示す
- スプレッドの平均回帰性を利用
- マーケットニュートラル戦略でリスク低減

使用例:
    from src.strategy.pairs_trading import (
        CointegrationPairsFinder,
        PairsTrader,
    )

    # 共和分ペア発見
    finder = CointegrationPairsFinder(significance_level=0.05)
    pairs = finder.find_pairs(prices_df)

    # トレーディングシグナル生成
    trader = PairsTrader(entry_zscore=2.0, exit_zscore=0.5)
    signals = trader.generate_signals(prices_a, prices_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# 推奨ペア定義
# =============================================================================

RECOMMENDED_PAIRS: list[tuple[str, str]] = [
    # インデックスETF
    ("SPY", "IVV"),      # S&P 500 ETF pairs
    ("QQQ", "TQQQ"),     # NASDAQ pairs (leveraged)
    # 債券ETF
    ("TLT", "IEF"),      # Treasury ETF pairs (long vs intermediate)
    # コモディティ
    ("GLD", "GDX"),      # Gold vs Gold Miners
    # セクターETF
    ("XLF", "KBE"),      # Financial sector pairs
    ("XLE", "OIH"),      # Energy sector pairs
    ("VNQ", "IYR"),      # Real estate ETF pairs
]


# =============================================================================
# Enum定義
# =============================================================================

class PairSignal(str, Enum):
    """ペアトレードシグナル"""
    LONG_SPREAD = "long_spread"    # Asset A Long, Asset B Short
    SHORT_SPREAD = "short_spread"  # Asset A Short, Asset B Long
    FLAT = "flat"                  # No position
    HOLD = "hold"                  # Maintain current position


class CointegrationMethod(str, Enum):
    """共和分検定手法"""
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class CointegrationResult:
    """共和分検定結果

    Attributes:
        asset_a: 資産A（被説明変数）
        asset_b: 資産B（説明変数）
        is_cointegrated: 共和分関係があるか
        p_value: 検定のp値
        test_statistic: 検定統計量
        critical_values: 臨界値
        hedge_ratio: ヘッジ比率（β）
        half_life: スプレッドの半減期
        correlation: 相関係数
        metadata: 追加メタデータ
    """
    asset_a: str
    asset_b: str
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]
    hedge_ratio: float
    half_life: float | None = None
    correlation: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "is_cointegrated": self.is_cointegrated,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "critical_values": self.critical_values,
            "hedge_ratio": self.hedge_ratio,
            "half_life": self.half_life,
            "correlation": self.correlation,
            "metadata": self.metadata,
        }


@dataclass
class PairPosition:
    """ペアポジション

    Attributes:
        asset_a: 資産A
        asset_b: 資産B
        signal: 現在のシグナル
        entry_zscore: エントリー時のZスコア
        entry_date: エントリー日
        entry_spread: エントリー時のスプレッド
        holding_days: 保有日数
        hedge_ratio: ヘッジ比率
        pnl: 損益
    """
    asset_a: str
    asset_b: str
    signal: PairSignal
    entry_zscore: float
    entry_date: datetime | None = None
    entry_spread: float = 0.0
    holding_days: int = 0
    hedge_ratio: float = 1.0
    pnl: float = 0.0


@dataclass
class PairsTradingSignal:
    """ペアトレーディングシグナル

    Attributes:
        date: 日付
        signal: シグナル
        spread: 現在のスプレッド
        zscore: 現在のZスコア
        hedge_ratio: ヘッジ比率
        weight_a: 資産Aの重み（正=ロング、負=ショート）
        weight_b: 資産Bの重み
        metadata: 追加メタデータ
    """
    date: datetime
    signal: PairSignal
    spread: float
    zscore: float
    hedge_ratio: float
    weight_a: float = 0.0
    weight_b: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairsTraderConfig:
    """PairsTrader設定

    Attributes:
        entry_zscore: エントリーZスコア閾値
        exit_zscore: エグジットZスコア閾値
        lookback: スプレッド計算期間
        max_holding_days: 最大保有日数
        hedge_ratio_lookback: ヘッジ比率計算期間
        use_dynamic_hedge: 動的ヘッジ比率使用
        stop_loss_zscore: ストップロスZスコア
    """
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    lookback: int = 60
    max_holding_days: int = 20
    hedge_ratio_lookback: int = 60
    use_dynamic_hedge: bool = True
    stop_loss_zscore: float = 4.0

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.entry_zscore <= self.exit_zscore:
            raise ValueError("entry_zscore must be > exit_zscore")
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.max_holding_days <= 0:
            raise ValueError("max_holding_days must be > 0")


# =============================================================================
# 共和分ペア発見クラス
# =============================================================================

class CointegrationPairsFinder:
    """共和分ペア発見クラス

    Engle-Granger検定を使用して共和分関係のあるペアを発見する。

    Usage:
        finder = CointegrationPairsFinder(significance_level=0.05)

        # 全ペア検定
        pairs = finder.find_pairs(prices_df)

        # 特定ペア検定
        result = finder.test_cointegration(prices_a, prices_b)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 252,
        method: CointegrationMethod = CointegrationMethod.ENGLE_GRANGER,
    ) -> None:
        """初期化

        Args:
            significance_level: 有意水準
            min_observations: 最小観測数
            method: 検定手法
        """
        self.significance_level = significance_level
        self.min_observations = min_observations
        self.method = method

    def find_pairs(
        self,
        prices: pd.DataFrame,
        candidate_pairs: list[tuple[str, str]] | None = None,
    ) -> list[CointegrationResult]:
        """共和分ペアを発見

        Args:
            prices: 価格DataFrame（列=資産）
            candidate_pairs: 検定するペアのリスト（Noneの場合は全組み合わせ）

        Returns:
            共和分関係のあるペアのリスト
        """
        if len(prices) < self.min_observations:
            logger.warning(
                "Insufficient observations: %d < %d",
                len(prices), self.min_observations,
            )
            return []

        if candidate_pairs is None:
            # 全組み合わせ生成
            assets = prices.columns.tolist()
            candidate_pairs = [
                (assets[i], assets[j])
                for i in range(len(assets))
                for j in range(i + 1, len(assets))
            ]

        results = []
        for asset_a, asset_b in candidate_pairs:
            if asset_a not in prices.columns or asset_b not in prices.columns:
                logger.debug("Skipping pair %s/%s: not in prices", asset_a, asset_b)
                continue

            result = self.test_cointegration(
                prices[asset_a],
                prices[asset_b],
                asset_a,
                asset_b,
            )
            if result.is_cointegrated:
                results.append(result)

        # p値でソート
        results.sort(key=lambda x: x.p_value)

        logger.info(
            "Found %d cointegrated pairs out of %d candidates",
            len(results), len(candidate_pairs),
        )

        return results

    def test_cointegration(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        name_a: str = "A",
        name_b: str = "B",
    ) -> CointegrationResult:
        """2つの系列の共和分検定

        Args:
            series_a: 系列A
            series_b: 系列B
            name_a: 系列Aの名前
            name_b: 系列Bの名前

        Returns:
            CointegrationResult
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            logger.error("statsmodels not installed. Install with: pip install statsmodels")
            return CointegrationResult(
                asset_a=name_a,
                asset_b=name_b,
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_values={},
                hedge_ratio=1.0,
                metadata={"error": "statsmodels not installed"},
            )

        # 欠損値処理
        df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
        if len(df) < self.min_observations:
            return CointegrationResult(
                asset_a=name_a,
                asset_b=name_b,
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_values={},
                hedge_ratio=1.0,
                metadata={"error": "Insufficient observations"},
            )

        # Engle-Granger検定
        test_stat, p_value, crit_values = coint(df["a"], df["b"])

        critical_values_dict = {
            "1%": crit_values[0],
            "5%": crit_values[1],
            "10%": crit_values[2],
        }

        is_cointegrated = p_value < self.significance_level

        # ヘッジ比率計算（線形回帰）
        hedge_ratio = self._calculate_hedge_ratio(df["a"], df["b"])

        # 相関係数
        correlation = df["a"].corr(df["b"])

        # 半減期計算
        half_life = self._calculate_half_life(df["a"], df["b"], hedge_ratio)

        return CointegrationResult(
            asset_a=name_a,
            asset_b=name_b,
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=test_stat,
            critical_values=critical_values_dict,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            correlation=correlation,
            metadata={
                "n_observations": len(df),
                "method": self.method.value,
            },
        )

    def _calculate_hedge_ratio(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> float:
        """ヘッジ比率を計算（OLS回帰）

        Args:
            series_a: 系列A（被説明変数）
            series_b: 系列B（説明変数）

        Returns:
            ヘッジ比率（β）
        """
        try:
            from sklearn.linear_model import LinearRegression

            X = series_b.values.reshape(-1, 1)
            y = series_a.values

            model = LinearRegression()
            model.fit(X, y)

            return float(model.coef_[0])
        except ImportError:
            # sklearnがない場合は単純な共分散/分散比
            cov = series_a.cov(series_b)
            var = series_b.var()
            return cov / var if var != 0 else 1.0

    def _calculate_half_life(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        hedge_ratio: float,
    ) -> float | None:
        """スプレッドの半減期を計算

        Args:
            series_a: 系列A
            series_b: 系列B
            hedge_ratio: ヘッジ比率

        Returns:
            半減期（日数）
        """
        spread = series_a - hedge_ratio * series_b
        spread_lag = spread.shift(1)
        delta_spread = spread - spread_lag

        # 回帰: Δspread = α + β * spread_lag
        df = pd.DataFrame({
            "delta": delta_spread,
            "lag": spread_lag,
        }).dropna()

        if len(df) < 20:
            return None

        cov = df["delta"].cov(df["lag"])
        var = df["lag"].var()

        if var == 0:
            return None

        beta = cov / var

        if beta >= 0:
            return None  # 平均回帰しない

        half_life = -np.log(2) / beta
        return max(1.0, min(100.0, half_life))


# =============================================================================
# ペアトレーダークラス
# =============================================================================

class PairsTrader:
    """ペアトレーディングシグナル生成クラス

    Zスコアベースのペアトレーディングシグナルを生成する。

    Usage:
        config = PairsTraderConfig(entry_zscore=2.0, exit_zscore=0.5)
        trader = PairsTrader(config)

        signals = trader.generate_signals(prices_a, prices_b, hedge_ratio=1.5)
    """

    def __init__(
        self,
        config: PairsTraderConfig | None = None,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        lookback: int = 60,
        max_holding_days: int = 20,
    ) -> None:
        """初期化

        Args:
            config: 設定（Noneの場合は個別引数を使用）
            entry_zscore: エントリーZスコア閾値
            exit_zscore: エグジットZスコア閾値
            lookback: スプレッド計算期間
            max_holding_days: 最大保有日数
        """
        if config is not None:
            self.config = config
        else:
            self.config = PairsTraderConfig(
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
                lookback=lookback,
                max_holding_days=max_holding_days,
            )

        self._current_position: PairPosition | None = None

    def generate_signals(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float | None = None,
    ) -> pd.DataFrame:
        """ペアトレーディングシグナルを生成

        Args:
            prices_a: 資産Aの価格
            prices_b: 資産Bの価格
            hedge_ratio: ヘッジ比率（Noneの場合は動的計算）

        Returns:
            シグナルDataFrame（signal, spread, zscore, weight_a, weight_b）
        """
        # データ整列
        df = pd.DataFrame({
            "price_a": prices_a,
            "price_b": prices_b,
        }).dropna()

        if len(df) < self.config.lookback:
            logger.warning(
                "Insufficient data: %d < %d",
                len(df), self.config.lookback,
            )
            return pd.DataFrame()

        # ヘッジ比率計算
        if hedge_ratio is None or self.config.use_dynamic_hedge:
            hedge_ratios = self._calculate_rolling_hedge_ratio(
                df["price_a"], df["price_b"]
            )
        else:
            hedge_ratios = pd.Series(hedge_ratio, index=df.index)

        # スプレッド計算
        spread = df["price_a"] - hedge_ratios * df["price_b"]

        # Zスコア計算
        spread_mean = spread.rolling(window=self.config.lookback).mean()
        spread_std = spread.rolling(window=self.config.lookback).std()
        zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        # シグナル生成
        signals = self._generate_signal_series(zscore)

        # 重み計算
        weights_a, weights_b = self._calculate_weights(signals, hedge_ratios)

        result = pd.DataFrame({
            "signal": signals,
            "spread": spread,
            "zscore": zscore,
            "hedge_ratio": hedge_ratios,
            "weight_a": weights_a,
            "weight_b": weights_b,
        }, index=df.index)

        return result

    def _calculate_rolling_hedge_ratio(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> pd.Series:
        """ローリングヘッジ比率を計算

        Args:
            series_a: 系列A
            series_b: 系列B

        Returns:
            ヘッジ比率Series
        """
        lookback = self.config.hedge_ratio_lookback
        hedge_ratios = []

        for i in range(len(series_a)):
            if i < lookback:
                hedge_ratios.append(1.0)
            else:
                window_a = series_a.iloc[i - lookback:i]
                window_b = series_b.iloc[i - lookback:i]
                cov = window_a.cov(window_b)
                var = window_b.var()
                ratio = cov / var if var != 0 else 1.0
                hedge_ratios.append(ratio)

        return pd.Series(hedge_ratios, index=series_a.index)

    def _generate_signal_series(
        self,
        zscore: pd.Series,
    ) -> pd.Series:
        """シグナル系列を生成

        Args:
            zscore: Zスコア系列

        Returns:
            シグナル系列
        """
        signals = []
        current_signal = PairSignal.FLAT
        holding_days = 0

        for z in zscore:
            if pd.isna(z):
                signals.append(PairSignal.FLAT)
                continue

            if current_signal == PairSignal.FLAT:
                # エントリー判定
                if z >= self.config.entry_zscore:
                    current_signal = PairSignal.SHORT_SPREAD
                    holding_days = 1
                elif z <= -self.config.entry_zscore:
                    current_signal = PairSignal.LONG_SPREAD
                    holding_days = 1
            else:
                holding_days += 1

                # エグジット判定
                should_exit = False

                # Zスコアがエグジット閾値に到達
                if current_signal == PairSignal.LONG_SPREAD and z >= -self.config.exit_zscore:
                    should_exit = True
                elif current_signal == PairSignal.SHORT_SPREAD and z <= self.config.exit_zscore:
                    should_exit = True

                # 最大保有日数
                if holding_days >= self.config.max_holding_days:
                    should_exit = True

                # ストップロス
                if abs(z) >= self.config.stop_loss_zscore:
                    should_exit = True

                if should_exit:
                    current_signal = PairSignal.FLAT
                    holding_days = 0

            signals.append(current_signal)

        return pd.Series(signals, index=zscore.index)

    def _calculate_weights(
        self,
        signals: pd.Series,
        hedge_ratios: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """ポジションウェイトを計算

        Args:
            signals: シグナル系列
            hedge_ratios: ヘッジ比率系列

        Returns:
            (weight_a, weight_b)
        """
        weights_a = []
        weights_b = []

        for signal, hr in zip(signals, hedge_ratios):
            if signal == PairSignal.LONG_SPREAD:
                # Long A, Short B
                weights_a.append(1.0)
                weights_b.append(-hr)
            elif signal == PairSignal.SHORT_SPREAD:
                # Short A, Long B
                weights_a.append(-1.0)
                weights_b.append(hr)
            else:
                weights_a.append(0.0)
                weights_b.append(0.0)

        return (
            pd.Series(weights_a, index=signals.index),
            pd.Series(weights_b, index=signals.index),
        )

    def get_current_zscore(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float,
    ) -> float:
        """現在のZスコアを取得

        Args:
            prices_a: 資産Aの価格
            prices_b: 資産Bの価格
            hedge_ratio: ヘッジ比率

        Returns:
            現在のZスコア
        """
        spread = prices_a - hedge_ratio * prices_b
        spread_recent = spread.tail(self.config.lookback)
        mean = spread_recent.mean()
        std = spread_recent.std()

        if std == 0:
            return 0.0

        return (spread.iloc[-1] - mean) / std


# =============================================================================
# 統合戦略クラス
# =============================================================================

class PairsTradingStrategy:
    """ペアトレーディング統合戦略クラス

    共和分ペア発見とトレーディングシグナル生成を統合する。

    Usage:
        strategy = PairsTradingStrategy()

        # ペア発見
        pairs = strategy.find_and_rank_pairs(prices_df)

        # 全ペアのシグナル生成
        all_signals = strategy.generate_all_signals(prices_df, pairs[:5])
    """

    def __init__(
        self,
        finder_config: dict[str, Any] | None = None,
        trader_config: PairsTraderConfig | None = None,
    ) -> None:
        """初期化

        Args:
            finder_config: CointegrationPairsFinder設定
            trader_config: PairsTrader設定
        """
        finder_config = finder_config or {}
        self.finder = CointegrationPairsFinder(**finder_config)
        self.trader = PairsTrader(config=trader_config)

    def find_and_rank_pairs(
        self,
        prices: pd.DataFrame,
        candidate_pairs: list[tuple[str, str]] | None = None,
        min_half_life: float = 5.0,
        max_half_life: float = 60.0,
    ) -> list[CointegrationResult]:
        """共和分ペアを発見してランク付け

        Args:
            prices: 価格DataFrame
            candidate_pairs: 検定するペア（Noneの場合はRECOMMENDED_PAIRS）
            min_half_life: 最小半減期
            max_half_life: 最大半減期

        Returns:
            ランク付けされたペアリスト
        """
        if candidate_pairs is None:
            candidate_pairs = RECOMMENDED_PAIRS

        pairs = self.finder.find_pairs(prices, candidate_pairs)

        # 半減期でフィルタ
        filtered = [
            p for p in pairs
            if p.half_life is not None
            and min_half_life <= p.half_life <= max_half_life
        ]

        # スコアでランク付け（低p値、適切な半減期）
        def score_pair(p: CointegrationResult) -> float:
            p_score = 1 - p.p_value  # 低いp値ほど高スコア
            hl_score = 1 - abs(p.half_life - 20) / 40 if p.half_life else 0  # 20日付近が最適
            return p_score * 0.6 + hl_score * 0.4

        filtered.sort(key=score_pair, reverse=True)

        return filtered

    def generate_all_signals(
        self,
        prices: pd.DataFrame,
        pairs: list[CointegrationResult],
    ) -> dict[str, pd.DataFrame]:
        """全ペアのシグナルを生成

        Args:
            prices: 価格DataFrame
            pairs: 共和分ペアリスト

        Returns:
            {pair_name: signals_df} の辞書
        """
        all_signals = {}

        for pair in pairs:
            if pair.asset_a not in prices.columns or pair.asset_b not in prices.columns:
                continue

            pair_name = f"{pair.asset_a}_{pair.asset_b}"
            signals = self.trader.generate_signals(
                prices[pair.asset_a],
                prices[pair.asset_b],
                hedge_ratio=pair.hedge_ratio,
            )

            if not signals.empty:
                all_signals[pair_name] = signals

        return all_signals


# =============================================================================
# 便利関数
# =============================================================================

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance_level: float = 0.05,
    candidate_pairs: list[tuple[str, str]] | None = None,
) -> list[CointegrationResult]:
    """共和分ペアを発見（便利関数）

    Args:
        prices: 価格DataFrame
        significance_level: 有意水準
        candidate_pairs: 候補ペア

    Returns:
        共和分ペアリスト
    """
    finder = CointegrationPairsFinder(significance_level=significance_level)
    return finder.find_pairs(prices, candidate_pairs)


def generate_pairs_signals(
    prices_a: pd.Series,
    prices_b: pd.Series,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    lookback: int = 60,
) -> pd.DataFrame:
    """ペアトレーディングシグナルを生成（便利関数）

    Args:
        prices_a: 資産A価格
        prices_b: 資産B価格
        entry_zscore: エントリー閾値
        exit_zscore: エグジット閾値
        lookback: ルックバック期間

    Returns:
        シグナルDataFrame
    """
    trader = PairsTrader(
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        lookback=lookback,
    )
    return trader.generate_signals(prices_a, prices_b)


def get_recommended_pairs() -> list[tuple[str, str]]:
    """推奨ペアを取得"""
    return RECOMMENDED_PAIRS.copy()
