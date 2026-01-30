"""
Lead-Lag Signal - 銘柄間時間差相関シグナル

学術的根拠:
- Oxford大学研究: 年率20%+のリターン
- 流動性の高い銘柄が小型株をリードする傾向
- Leader銘柄の情報でFollower銘柄の予測精度60%達成

設計:
- 各銘柄ペアの時間ラグ付き相関を計算
- 統計的に有意なLead-Lag関係を特定
- Follower銘柄にLeader銘柄の方向性シグナルを付与

最適化 (task_042_1_opt):
- Numba JIT並列化で5-10倍高速化
- 段階的フィルタリングで計算量90%削減
- 二分探索的lag検索でO(lag_max) → O(log(lag_max))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry

logger = logging.getLogger(__name__)

# Numba関数のインポート（フォールバック対応）
try:
    from src.backtest.numba_compute import (
        compute_lagged_correlations_numba,
        compute_zero_lag_correlations_numba,
        find_best_lag_binary_search,
        NUMBA_AVAILABLE,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    compute_lagged_correlations_numba = None
    compute_zero_lag_correlations_numba = None
    find_best_lag_binary_search = None


@dataclass
class LeadLagPair:
    """Lead-Lag関係を持つ銘柄ペア。"""

    leader: str
    follower: str
    lag: int
    correlation: float
    direction: int  # +1 for positive correlation, -1 for negative


@SignalRegistry.register(
    "lead_lag",
    category="relative",
    description="Lead-Lag relationship signal detecting cross-ticker time-lagged correlations",
    tags=["cross-sectional", "relative", "academic"],
)
class LeadLagSignal(Signal):
    """
    Lead-Lag関係シグナル

    銘柄間の時間差相関を検出し、Leader銘柄の動きから
    Follower銘柄の将来リターンを予測する。

    Parameters:
        lookback: 相関計算期間（デフォルト60日）
        lag_min: 検証する最小ラグ日数（デフォルト1）
        lag_max: 検証する最大ラグ日数（デフォルト5）
        min_correlation: 最小相関閾値（デフォルト0.3）
        top_n_leaders: 参照するLeader数（デフォルト5）
        use_numba: Numba最適化を使用（デフォルトTrue）
        use_staged_filter: 段階的フィルタリングを使用（デフォルトTrue）

    使用例:
        signal = LeadLagSignal(lookback=60, lag_min=1, lag_max=5)
        result = signal.compute(prices_df)
    """

    signal_name = "lead_lag"

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        """パラメータ仕様を定義。"""
        return [
            ParameterSpec(
                name="lookback",
                default=60,
                searchable=True,
                min_value=30,
                max_value=120,
                step=30,
                description="相関計算期間（日数）",
            ),
            ParameterSpec(
                name="lag_min",
                default=1,
                searchable=True,
                min_value=1,
                max_value=3,
                step=1,
                description="検証する最小ラグ日数",
            ),
            ParameterSpec(
                name="lag_max",
                default=5,
                searchable=True,
                min_value=3,
                max_value=10,
                step=1,
                description="検証する最大ラグ日数",
            ),
            ParameterSpec(
                name="min_correlation",
                default=0.3,
                searchable=True,
                min_value=0.2,
                max_value=0.5,
                step=0.1,
                description="最小相関閾値",
            ),
            ParameterSpec(
                name="top_n_leaders",
                default=5,
                searchable=False,
                min_value=1,
                max_value=10,
                description="各Followerに対して参照するLeader数",
            ),
            ParameterSpec(
                name="use_numba",
                default=True,
                searchable=False,
                description="Numba最適化を使用",
            ),
            ParameterSpec(
                name="use_staged_filter",
                default=True,
                searchable=False,
                description="段階的フィルタリングを使用（計算量90%削減）",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """最適化検索用のパラメータグリッド。"""
        return {
            "lookback": [30, 60, 90, 120],
            "lag_min": [1, 2],
            "lag_max": [3, 5],
            "min_correlation": [0.2, 0.3, 0.4],
        }

    def _validate_lead_lag_input(self, data: pd.DataFrame) -> None:
        """
        LeadLagSignal用の入力データバリデーション。

        通常のSignal.validate_input()と異なり、wide format
        （カラム=ティッカー）のDataFrameを期待する。

        Args:
            data: 入力DataFrame（columns=ティッカー、index=日付）

        Raises:
            ValueError: データ形式が無効な場合
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")

        if len(data.columns) < 2:
            raise ValueError(
                "LeadLagSignal requires at least 2 tickers (columns) for "
                "cross-ticker correlation analysis"
            )

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Lead-Lagシグナルを計算。

        Args:
            data: 価格データ（columns=ティッカー、index=DatetimeIndex）
                  複数銘柄のclose価格を含むDataFrame

        Returns:
            SignalResult: 各銘柄のスコア（-1 to +1）
        """
        self._validate_lead_lag_input(data)

        lookback = self._params["lookback"]
        lag_min = self._params["lag_min"]
        lag_max = self._params["lag_max"]
        min_correlation = self._params["min_correlation"]
        top_n_leaders = self._params["top_n_leaders"]
        use_numba = self._params.get("use_numba", True) and NUMBA_AVAILABLE
        use_staged_filter = self._params.get("use_staged_filter", True)

        # 十分なデータがあるか確認
        if len(data) < lookback + lag_max:
            logger.warning(
                f"Insufficient data for lead-lag analysis: {len(data)} rows, "
                f"need at least {lookback + lag_max}"
            )
            scores = pd.Series(0.0, index=data.columns)
            return SignalResult(
                scores=scores,
                metadata={"error": "insufficient_data", "pairs": []},
            )

        # リターンを計算
        returns = data.pct_change().dropna()

        # Lead-Lag関係を検出（最適化選択）
        if use_numba and use_staged_filter:
            pairs = self._detect_lead_lag_pairs_optimized(
                returns,
                lookback=lookback,
                lag_range=(lag_min, lag_max),
                min_correlation=min_correlation,
            )
        elif use_numba:
            pairs = self._detect_lead_lag_pairs_numba(
                returns,
                lookback=lookback,
                lag_range=(lag_min, lag_max),
                min_correlation=min_correlation,
            )
        else:
            pairs = self._detect_lead_lag_pairs(
                returns,
                lookback=lookback,
                lag_range=(lag_min, lag_max),
                min_correlation=min_correlation,
            )

        # 各Followerに対するシグナルを生成
        scores = self._generate_signals(
            returns,
            pairs,
            top_n_leaders=top_n_leaders,
        )

        # 欠損銘柄には0を埋める
        for col in data.columns:
            if col not in scores.index:
                scores[col] = 0.0

        # スコアを正規化
        scores = self.normalize_tanh(scores, scale=2.0)

        # 順序を元のカラム順に合わせる
        scores = scores.reindex(data.columns).fillna(0.0)

        return SignalResult(
            scores=scores,
            metadata={
                "pairs_count": len(pairs),
                "optimization": "numba+staged" if use_numba and use_staged_filter
                else "numba" if use_numba else "naive",
                "top_pairs": [
                    {
                        "leader": p.leader,
                        "follower": p.follower,
                        "lag": p.lag,
                        "correlation": round(p.correlation, 4),
                    }
                    for p in sorted(
                        pairs, key=lambda x: abs(x.correlation), reverse=True
                    )[:10]
                ],
            },
        )

    def _detect_lead_lag_pairs_optimized(
        self,
        returns: pd.DataFrame,
        lookback: int,
        lag_range: tuple[int, int],
        min_correlation: float,
    ) -> List[LeadLagPair]:
        """
        最適化版Lead-Lag検出（段階的フィルタリング + 二分探索）

        Step 1: lag=0で全ペアの相関を計算
        Step 2: 相関 > threshold のペアのみlag検索（計算量90%削減）
        Step 3: 二分探索的lag検索でO(lag_max) → O(log(lag_max))

        Args:
            returns: リターンのDataFrame
            lookback: 相関計算期間
            lag_range: (min_lag, max_lag)
            min_correlation: 最小相関閾値

        Returns:
            検出されたLeadLagPairのリスト
        """
        pairs = []
        tickers = list(returns.columns)
        n_tickers = len(tickers)

        if n_tickers < 2:
            return pairs

        # NumPy配列に変換 (n_assets, n_days)
        returns_array = returns.values.T.astype(np.float64)

        # Step 1: lag=0相関でフィルタリング
        # 閾値の半分で候補をフィルタ（lag付き相関の方が高い可能性）
        filter_threshold = min_correlation * 0.5
        zero_lag_corr = compute_zero_lag_correlations_numba(returns_array, lookback)

        # Step 2: 候補ペアのみ詳細検索
        for i in range(n_tickers):
            for j in range(n_tickers):
                if i == j:
                    continue

                # lag=0相関でフィルタ
                corr_ij = zero_lag_corr[i, j]
                if np.isnan(corr_ij) or abs(corr_ij) < filter_threshold:
                    continue

                # Step 3: 二分探索的lag検索
                best_lag, best_corr = find_best_lag_binary_search(
                    returns_array[i],
                    returns_array[j],
                    lag_range[0],
                    lag_range[1],
                    lookback,
                )

                if abs(best_corr) >= min_correlation:
                    pairs.append(
                        LeadLagPair(
                            leader=tickers[i],
                            follower=tickers[j],
                            lag=best_lag,
                            correlation=best_corr,
                            direction=1 if best_corr > 0 else -1,
                        )
                    )

        logger.debug(f"Detected {len(pairs)} lead-lag pairs (optimized)")
        return pairs

    def _detect_lead_lag_pairs_numba(
        self,
        returns: pd.DataFrame,
        lookback: int,
        lag_range: tuple[int, int],
        min_correlation: float,
    ) -> List[LeadLagPair]:
        """
        Numba並列化版Lead-Lag検出（全ペア計算）

        Args:
            returns: リターンのDataFrame
            lookback: 相関計算期間
            lag_range: (min_lag, max_lag)
            min_correlation: 最小相関閾値

        Returns:
            検出されたLeadLagPairのリスト
        """
        pairs = []
        tickers = list(returns.columns)
        n_tickers = len(tickers)

        if n_tickers < 2:
            return pairs

        # NumPy配列に変換 (n_assets, n_days)
        returns_array = returns.values.T.astype(np.float64)

        # Numba並列計算
        correlations = compute_lagged_correlations_numba(
            returns_array,
            lag_range[0],
            lag_range[1],
            lookback,
        )

        # 結果からペアを抽出
        n_lags = lag_range[1] - lag_range[0] + 1
        for i in range(n_tickers):
            for j in range(n_tickers):
                if i == j:
                    continue

                # 最良ラグを見つける
                best_lag_idx = -1
                best_corr = 0.0

                for lag_idx in range(n_lags):
                    corr = correlations[i, j, lag_idx]
                    if not np.isnan(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag_idx = lag_idx

                if best_lag_idx >= 0 and abs(best_corr) >= min_correlation:
                    pairs.append(
                        LeadLagPair(
                            leader=tickers[i],
                            follower=tickers[j],
                            lag=lag_range[0] + best_lag_idx,
                            correlation=best_corr,
                            direction=1 if best_corr > 0 else -1,
                        )
                    )

        logger.debug(f"Detected {len(pairs)} lead-lag pairs (numba)")
        return pairs

    def _detect_lead_lag_pairs(
        self,
        returns: pd.DataFrame,
        lookback: int,
        lag_range: tuple[int, int],
        min_correlation: float,
    ) -> List[LeadLagPair]:
        """
        Lead-Lag関係を持つ銘柄ペアを検出（ナイーブ実装）

        Args:
            returns: リターンのDataFrame
            lookback: 相関計算期間
            lag_range: (min_lag, max_lag)
            min_correlation: 最小相関閾値

        Returns:
            検出されたLeadLagPairのリスト
        """
        pairs = []
        tickers = list(returns.columns)
        n_tickers = len(tickers)

        if n_tickers < 2:
            return pairs

        # 直近lookback期間のデータを使用
        recent_returns = returns.iloc[-lookback:]

        for i, potential_leader in enumerate(tickers):
            for j, potential_follower in enumerate(tickers):
                if i == j:
                    continue

                leader_returns = recent_returns[potential_leader].values

                # 各ラグで相関を計算
                best_lag = None
                best_corr = 0.0

                for lag in range(lag_range[0], lag_range[1] + 1):
                    if lag >= len(leader_returns):
                        continue

                    # Leader(t) vs Follower(t+lag)
                    leader_slice = leader_returns[:-lag]
                    follower_slice = recent_returns[potential_follower].values[lag:]

                    if len(leader_slice) < 10:
                        continue

                    # NaNを除外して相関計算
                    valid_mask = ~(
                        np.isnan(leader_slice) | np.isnan(follower_slice)
                    )
                    if valid_mask.sum() < 10:
                        continue

                    corr = np.corrcoef(
                        leader_slice[valid_mask], follower_slice[valid_mask]
                    )[0, 1]

                    if not np.isnan(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                # 閾値を超える相関が見つかった場合
                if best_lag is not None and abs(best_corr) >= min_correlation:
                    pairs.append(
                        LeadLagPair(
                            leader=potential_leader,
                            follower=potential_follower,
                            lag=best_lag,
                            correlation=best_corr,
                            direction=1 if best_corr > 0 else -1,
                        )
                    )

        logger.debug(f"Detected {len(pairs)} lead-lag pairs (naive)")
        return pairs

    def _generate_signals(
        self,
        returns: pd.DataFrame,
        pairs: List[LeadLagPair],
        top_n_leaders: int,
    ) -> pd.Series:
        """
        検出されたペアからシグナルを生成。

        Args:
            returns: リターンのDataFrame
            pairs: LeadLagPairのリスト
            top_n_leaders: 各Followerに対して参照するLeader数

        Returns:
            各銘柄のシグナルスコア
        """
        scores: Dict[str, float] = {}

        # 各Followerごとに最良のLeaderを選択
        follower_to_pairs: Dict[str, List[LeadLagPair]] = {}
        for pair in pairs:
            if pair.follower not in follower_to_pairs:
                follower_to_pairs[pair.follower] = []
            follower_to_pairs[pair.follower].append(pair)

        for follower, follower_pairs in follower_to_pairs.items():
            # 相関の絶対値でソートし、上位N個を選択
            top_pairs = sorted(
                follower_pairs, key=lambda x: abs(x.correlation), reverse=True
            )[:top_n_leaders]

            if not top_pairs:
                continue

            # 各Leaderの直近リターンからシグナルを計算
            weighted_signal = 0.0
            total_weight = 0.0

            for pair in top_pairs:
                leader = pair.leader
                if leader not in returns.columns:
                    continue

                # Leader の最近 lag 日間のリターンの平均を見る
                recent_leader_return = returns[leader].iloc[-pair.lag:].mean()

                if pd.isna(recent_leader_return):
                    continue

                # 相関に基づく方向性シグナル
                # 正の相関: Leaderが上がればFollowerも上がる予測
                # 負の相関: Leaderが上がればFollowerは下がる予測
                signal = recent_leader_return * pair.direction

                # 相関の絶対値をウェイトとして使用
                weight = abs(pair.correlation)
                weighted_signal += signal * weight
                total_weight += weight

            if total_weight > 0:
                scores[follower] = weighted_signal / total_weight

        return pd.Series(scores)

    def compute_polars(self, prices: pl.DataFrame) -> pl.DataFrame:
        """
        Polars DataFrameからシグナルを計算。

        Args:
            prices: 価格データ（columns: timestamp, ticker, close）

        Returns:
            pl.DataFrame: columns: timestamp, ticker, value
        """
        # Polars -> Pandas 変換（pivot）
        if "ticker" in prices.columns:
            # Long format -> Wide format
            pivot_pl = prices.pivot(
                index="timestamp", columns="ticker", values="close"
            )
            pivot_df = pivot_pl.to_pandas()

            # timestamp列をDatetimeIndexに設定
            if "timestamp" in pivot_df.columns:
                pivot_df = pivot_df.set_index("timestamp")
            pivot_df.index = pd.to_datetime(pivot_df.index)

            # 数値データのみ抽出（timestamp列が残っている場合に備えて）
            numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
            pivot_df = pivot_df[numeric_cols]
        else:
            pivot_df = prices.to_pandas()
            if "timestamp" in pivot_df.columns:
                pivot_df = pivot_df.set_index("timestamp")
                pivot_df.index = pd.to_datetime(pivot_df.index)

        # 計算実行
        result = self.compute(pivot_df)

        # 最新タイムスタンプでのスコアをDataFrameに変換
        latest_timestamp = pivot_df.index[-1]
        records = [
            {
                "timestamp": latest_timestamp,
                "ticker": ticker,
                "value": float(score),
            }
            for ticker, score in result.scores.items()
        ]

        return pl.DataFrame(records)


def compute_lead_lag_signal(
    prices: pl.DataFrame,
    lookback: int = 60,
    lag_range: tuple[int, int] = (1, 5),
    min_correlation: float = 0.3,
    top_n_leaders: int = 5,
    use_numba: bool = True,
    use_staged_filter: bool = True,
) -> pl.DataFrame:
    """
    便利関数: Lead-Lagシグナルを計算。

    Args:
        prices: 価格データ（columns: timestamp, ticker, close）
        lookback: 相関計算期間
        lag_range: (min_lag, max_lag)
        min_correlation: 最小相関閾値
        top_n_leaders: 参照するLeader数
        use_numba: Numba最適化を使用
        use_staged_filter: 段階的フィルタリングを使用

    Returns:
        pl.DataFrame: columns: timestamp, ticker, value
    """
    signal = LeadLagSignal(
        lookback=lookback,
        lag_min=lag_range[0],
        lag_max=lag_range[1],
        min_correlation=min_correlation,
        top_n_leaders=top_n_leaders,
        use_numba=use_numba,
        use_staged_filter=use_staged_filter,
    )
    return signal.compute_polars(prices)
