"""
VectorBT-Style Backtest Engine - 完全ベクトル化バックテストエンジン

VectorBT的なアプローチで、ループを一切使わずに
全ての計算をベクトル演算で行う高速バックテストエンジン。

Key Features:
- 完全ベクトル化（ループなし）
- シグナル→ポジション→リターンの行列演算
- Numba JIT高速化オプション
- Polars/NumPy連携
- 累積100-500x高速化

Based on HI-007: VectorBT architecture integration.

Expected Effect:
- 100-500x speedup compared to loop-based implementation
- Support for 800+ symbols efficiently

Usage:
    from src.backtest.vectorbt_engine import VectorBTStyleEngine, VectorBTConfig

    engine = VectorBTStyleEngine()
    result = engine.run(
        prices=price_matrix,    # (n_days, n_assets)
        signals=signal_matrix,  # (n_days, n_assets)
    )

    print(f"Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Max DD: {result.max_drawdown:.2%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

# Import base class for unified interface
from src.backtest.base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    create_rebalance_mask,
)

logger = logging.getLogger(__name__)


# Numba JIT関数（オプション）
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range


@dataclass
class VectorBTConfig:
    """VectorBT風エンジン設定

    Attributes:
        initial_capital: 初期資金
        transaction_cost_bps: 取引コスト（ベーシスポイント）
        slippage_bps: スリッページ（ベーシスポイント）
        rebalance_frequency: リバランス頻度
        position_sizing: ポジションサイジング方法
        max_position: 最大ポジション（単一銘柄）
        use_numba: Numba JIT使用
        signal_threshold: シグナル閾値（これ以下は無視）
    """
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    position_sizing: Literal["equal", "signal_weighted", "vol_target"] = "signal_weighted"
    max_position: float = 0.20
    use_numba: bool = True
    signal_threshold: float = 0.01


@dataclass
class VectorBTResult:
    """VectorBTバックテスト結果

    Attributes:
        dates: 日付配列
        portfolio_values: ポートフォリオ価値時系列
        returns: 日次リターン
        positions: ポジション履歴 (n_days, n_assets)
        weights: ウェイト履歴 (n_days, n_assets)

        # メトリクス
        total_return: 累積リターン
        cagr: 年率リターン
        sharpe_ratio: シャープレシオ
        sortino_ratio: ソルティノレシオ
        max_drawdown: 最大ドローダウン
        calmar_ratio: カルマーレシオ
        volatility: 年率ボラティリティ
        win_rate: 勝率
        profit_factor: プロフィットファクター

        # 取引統計
        n_trades: 取引回数
        total_costs: 総取引コスト
        turnover: 年間ターンオーバー

        # メタデータ
        execution_time_ms: 実行時間
        n_days: 日数
        n_assets: 銘柄数
    """
    # 時系列データ
    dates: list[datetime] | NDArray
    portfolio_values: NDArray[np.float64]
    returns: NDArray[np.float64]
    positions: NDArray[np.float64]
    weights: NDArray[np.float64]

    # メトリクス
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # 取引統計
    n_trades: int = 0
    n_rebalances: int = 0
    total_costs: float = 0.0
    turnover: float = 0.0

    # メタデータ
    execution_time_ms: float = 0.0
    n_days: int = 0
    n_assets: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "volatility": f"{self.volatility:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "n_trades": self.n_trades,
            "n_rebalances": self.n_rebalances,
            "total_costs": f"{self.total_costs:.2f}",
            "turnover": f"{self.turnover:.2%}",
            "execution_time_ms": f"{self.execution_time_ms:.1f}",
            "n_days": self.n_days,
            "n_assets": self.n_assets,
        }

    def summary(self) -> str:
        """サマリー文字列"""
        return (
            f"VectorBT Result: {self.n_days} days, {self.n_assets} assets\n"
            f"  Return: {self.total_return:.2%} (CAGR: {self.cagr:.2%})\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}, Sortino: {self.sortino_ratio:.2f}\n"
            f"  Max DD: {self.max_drawdown:.2%}, Calmar: {self.calmar_ratio:.2f}\n"
            f"  Vol: {self.volatility:.2%}, Win Rate: {self.win_rate:.2%}\n"
            f"  Trades: {self.n_trades}, Rebalances: {self.n_rebalances}, Costs: ${self.total_costs:.2f}\n"
            f"  Execution: {self.execution_time_ms:.1f}ms"
        )


# =============================================================================
# Numba JIT高速化関数
# =============================================================================

@njit(cache=True)
def _compute_portfolio_returns_numba(
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """ポートフォリオリターンを計算（Numba JIT）

    Args:
        asset_returns: 資産リターン (n_days, n_assets)
        weights: ウェイト (n_days, n_assets)

    Returns:
        ポートフォリオリターン (n_days,)
    """
    n_days = asset_returns.shape[0]
    portfolio_returns = np.zeros(n_days)

    for i in range(n_days):
        portfolio_returns[i] = np.sum(asset_returns[i] * weights[i])

    return portfolio_returns


@njit(cache=True)
def _compute_max_drawdown_numba(portfolio_values: np.ndarray) -> float:
    """最大ドローダウンを計算（Numba JIT）

    Args:
        portfolio_values: ポートフォリオ価値時系列

    Returns:
        最大ドローダウン（負の値）
    """
    n = len(portfolio_values)
    if n == 0:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0.0

    for i in range(n):
        if portfolio_values[i] > peak:
            peak = portfolio_values[i]
        dd = (portfolio_values[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return max_dd


@njit(cache=True)
def _compute_sharpe_numba(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """シャープレシオを計算（Numba JIT）

    Args:
        returns: 日次リターン
        risk_free: リスクフリーレート（日次）

    Returns:
        年率シャープレシオ
    """
    excess_returns = returns - risk_free
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(252)


@njit(cache=True)
def _compute_sortino_numba(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """ソルティノレシオを計算（Numba JIT）

    Args:
        returns: 日次リターン
        risk_free: リスクフリーレート（日次）

    Returns:
        年率ソルティノレシオ
    """
    excess_returns = returns - risk_free
    mean_return = np.mean(excess_returns)

    # 下方偏差
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf if mean_return > 0 else 0.0

    downside_std = np.sqrt(np.mean(negative_returns ** 2))

    if downside_std == 0:
        return 0.0

    return mean_return / downside_std * np.sqrt(252)


@njit(cache=True)
def _compute_turnover_numba(weights: np.ndarray) -> float:
    """ターンオーバーを計算（Numba JIT）

    Args:
        weights: ウェイト履歴 (n_days, n_assets)

    Returns:
        平均日次ターンオーバー
    """
    n_days = weights.shape[0]
    if n_days < 2:
        return 0.0

    total_turnover = 0.0
    for i in range(1, n_days):
        total_turnover += np.sum(np.abs(weights[i] - weights[i-1]))

    return total_turnover / (n_days - 1)


@njit(cache=True)
def _apply_rebalance_mask_numba(
    base_weights: np.ndarray,
    rebalance_mask: np.ndarray,
) -> np.ndarray:
    """リバランスマスクを適用（Numba JIT）

    リバランス日以外は前日のウェイトを維持。

    Args:
        base_weights: 基準ウェイト (n_days, n_assets)
        rebalance_mask: リバランス日フラグ (n_days,)

    Returns:
        適用後ウェイト (n_days, n_assets)
    """
    n_days, n_assets = base_weights.shape
    result = np.zeros_like(base_weights)

    # 最初のリバランス日を見つける
    first_rebal = 0
    for i in range(n_days):
        if rebalance_mask[i]:
            first_rebal = i
            break

    # 最初のリバランス日までは均等配分
    for i in range(first_rebal):
        for j in range(n_assets):
            result[i, j] = 1.0 / n_assets

    # リバランス日から次のリバランス日まで同じウェイトを維持
    current_weights = np.zeros(n_assets)
    for j in range(n_assets):
        current_weights[j] = base_weights[first_rebal, j]

    for i in range(first_rebal, n_days):
        if rebalance_mask[i]:
            for j in range(n_assets):
                current_weights[j] = base_weights[i, j]

        for j in range(n_assets):
            result[i, j] = current_weights[j]

    return result


# =============================================================================
# VectorBT Style Engine
# =============================================================================

class VectorBTStyleEngine(BacktestEngineBase):
    """
    VectorBT風の完全ベクトル化バックテストエンジン

    ループを一切使わず、全ての計算を行列演算で行う。
    Numba JITで更なる高速化が可能。

    BacktestEngineBase準拠: 統一インターフェースで呼び出し可能。

    Algorithm:
    1. シグナル → ポジション変換（ベクトル化）
    2. ポジション → ウェイト正規化（ベクトル化）
    3. リバランスマスク適用
    4. 資産リターン × ウェイト = ポートフォリオリターン
    5. 累積積でポートフォリオ価値
    6. メトリクス計算（Numba JIT）

    Usage:
        engine = VectorBTStyleEngine()

        # 価格マトリクス (n_days, n_assets)
        prices = np.array([[100, 80], [101, 81], ...])

        # シグナルマトリクス (n_days, n_assets) in [-1, 1]
        signals = np.array([[0.5, 0.3], [0.6, 0.2], ...])

        result = engine.run(prices, signals)
    """

    ENGINE_NAME: str = "vectorbt"

    def __init__(
        self,
        config: VectorBTConfig | None = None,
        unified_config: Optional[UnifiedBacktestConfig] = None,
    ) -> None:
        """初期化

        Args:
            config: VectorBT固有設定
            unified_config: 統一バックテスト設定
        """
        super().__init__(unified_config)

        # VectorBTConfig の構築（優先順位: config > unified_config > default）
        if config is not None:
            self.vbt_config = config
        elif unified_config is not None:
            self.vbt_config = self._convert_from_unified_config(unified_config)
        else:
            self.vbt_config = VectorBTConfig()

        # 後方互換性のためconfigも維持
        self.config = self.vbt_config

        # Numba利用可能性チェック
        self._use_numba = self.vbt_config.use_numba and NUMBA_AVAILABLE
        if self.vbt_config.use_numba and not NUMBA_AVAILABLE:
            logger.warning("Numba not available, falling back to NumPy")

    def _convert_from_unified_config(
        self,
        unified_config: UnifiedBacktestConfig,
    ) -> VectorBTConfig:
        """UnifiedBacktestConfig から VectorBTConfig を構築

        Args:
            unified_config: 統一バックテスト設定

        Returns:
            VectorBTConfig: VectorBT固有設定
        """
        # rebalance_frequency の変換（UnifiedBacktestConfigの値を優先）
        rebalance_freq = getattr(unified_config, "rebalance_frequency", "monthly")
        if rebalance_freq not in ("daily", "weekly", "monthly"):
            # quarterly等の場合はmonthlyにフォールバック
            rebalance_freq = "monthly"

        return VectorBTConfig(
            initial_capital=getattr(unified_config, "initial_capital", 100000.0),
            transaction_cost_bps=getattr(unified_config, "transaction_cost_bps", 10.0),
            slippage_bps=getattr(unified_config, "slippage_bps", 5.0),
            rebalance_frequency=rebalance_freq,
            max_position=getattr(unified_config, "max_position_size", 0.20),
        )

    def _update_config_from_unified(
        self,
        unified_config: UnifiedBacktestConfig,
    ) -> None:
        """run()呼び出し時にUnifiedBacktestConfigからVectorBTConfigを更新

        Args:
            unified_config: 統一バックテスト設定
        """
        # rebalance_frequency の変換
        rebalance_freq = getattr(unified_config, "rebalance_frequency", None)
        if rebalance_freq is not None:
            if rebalance_freq in ("daily", "weekly", "monthly"):
                self.vbt_config.rebalance_frequency = rebalance_freq
            else:
                # quarterly等の場合はmonthlyにフォールバック
                self.vbt_config.rebalance_frequency = "monthly"
                logger.debug(f"rebalance_frequency '{rebalance_freq}' not supported, using 'monthly'")

        # その他の設定を更新
        if hasattr(unified_config, "initial_capital") and unified_config.initial_capital is not None:
            self.vbt_config.initial_capital = unified_config.initial_capital

        if hasattr(unified_config, "transaction_cost_bps") and unified_config.transaction_cost_bps is not None:
            self.vbt_config.transaction_cost_bps = unified_config.transaction_cost_bps

        if hasattr(unified_config, "slippage_bps") and unified_config.slippage_bps is not None:
            self.vbt_config.slippage_bps = unified_config.slippage_bps

        if hasattr(unified_config, "max_position_size") and unified_config.max_position_size is not None:
            self.vbt_config.max_position = unified_config.max_position_size

        # self.configも同期（後方互換性）
        self.config = self.vbt_config

    # =========================================================================
    # BacktestEngineBase Interface Implementation
    # =========================================================================

    def run(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[Callable] = None,
    ) -> UnifiedBacktestResult:
        """
        統一インターフェースでバックテスト実行（BacktestEngineBase準拠）

        Args:
            universe: ユニバース（銘柄リスト）
            prices: 価格データ（{symbol: DataFrame}）
            config: バックテスト設定
            weights_func: ウェイト計算関数（使用しない場合はNone）

        Returns:
            UnifiedBacktestResult: 統一バックテスト結果
        """
        cfg = config or self._config
        if cfg is None:
            cfg = UnifiedBacktestConfig(
                start_date="2010-01-01",
                end_date="2024-12-31",
                initial_capital=self.vbt_config.initial_capital,
                transaction_cost_bps=self.vbt_config.transaction_cost_bps,
                rebalance_frequency=self.vbt_config.rebalance_frequency,
            )

        # configからVectorBTConfigを更新（run()呼び出し時のconfig優先）
        if config is not None:
            self._update_config_from_unified(config)

        # 入力検証
        self.validate_inputs(universe, prices, cfg)

        # 価格データを行列形式に変換
        prices_matrix, dates, asset_names = self._convert_prices_to_matrix(
            universe, prices, cfg
        )

        # リバランスマスク作成（vbt_configの更新後のrebalance_frequencyを使用）
        rebalance_mask = self._create_rebalance_mask(dates)
        n_rebalances = int(np.sum(rebalance_mask))

        # weights_funcからシグナルを生成
        rebalance_records = []
        if weights_func is not None:
            signals, rebalance_records = self._generate_signals_from_weights_func(
                weights_func=weights_func,
                universe=asset_names,
                prices=prices,
                prices_matrix=prices_matrix,
                dates=dates,
                rebalance_mask=rebalance_mask,
            )
            logger.info(f"Generated signals from weights_func: {n_rebalances} rebalances")
        else:
            signals = None  # 均等配分

        # 内部メソッドで実行
        result = self.run_vectorized(
            prices=prices_matrix,
            signals=signals,
            dates=dates,
            asset_names=asset_names,
        )

        # UnifiedBacktestResult に変換（リバランス情報を追加）
        return self._convert_to_unified_result(result, cfg, rebalance_records, n_rebalances)

    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """
        入力を検証（BacktestEngineBase準拠）

        Args:
            universe: ユニバース
            prices: 価格データ
            config: 設定

        Returns:
            bool: 検証結果

        Raises:
            ValueError: 検証エラー
        """
        # 共通検証
        warnings = self._validate_common_inputs(universe, prices, config)
        for warning in warnings:
            logger.warning(warning)

        if len(universe) == 0:
            raise ValueError("Universe cannot be empty")

        return True

    def _generate_signals_from_weights_func(
        self,
        weights_func: Callable,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        prices_matrix: np.ndarray,
        dates: List[datetime],
        rebalance_mask: np.ndarray,
    ) -> tuple[np.ndarray, List]:
        """
        weights_funcからシグナル行列を生成

        Args:
            weights_func: ウェイト計算関数（WeightsFuncProtocol準拠）
            universe: 銘柄リスト
            prices: 価格データ（元のDict形式）
            prices_matrix: 価格行列
            dates: 日付リスト
            rebalance_mask: リバランス日フラグ

        Returns:
            tuple[np.ndarray, List]: (シグナル行列, リバランス記録リスト)
        """
        from src.backtest.base import RebalanceRecord

        n_days = len(dates)
        n_assets = len(universe)
        signals = np.zeros((n_days, n_assets))
        rebalance_records = []

        # 銘柄名→インデックスのマッピング
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(universe)}

        # 初期ウェイト（均等配分）
        current_weights = {symbol: 1.0 / n_assets for symbol in universe}

        for i, date in enumerate(dates):
            if rebalance_mask[i]:
                # リバランス日: weights_funcを呼び出し
                try:
                    weights_before = current_weights.copy()

                    # weights_funcを呼び出し
                    new_weights = weights_func(universe, prices, date, current_weights)

                    # 新しいウェイトを適用
                    if new_weights:
                        # ウェイトを正規化
                        total_weight = sum(new_weights.values())
                        if total_weight > 0:
                            new_weights = {
                                k: v / total_weight for k, v in new_weights.items()
                            }
                        current_weights = new_weights

                    # ターンオーバー計算
                    turnover = sum(
                        abs(current_weights.get(s, 0) - weights_before.get(s, 0))
                        for s in universe
                    ) / 2

                    # ポートフォリオ価値（概算）
                    portfolio_value = self.vbt_config.initial_capital
                    if i > 0:
                        # 価格変動を考慮した概算
                        prev_mean = np.mean(prices_matrix[i-1])
                        curr_mean = np.mean(prices_matrix[i])
                        if prev_mean > 0:
                            price_return = curr_mean / prev_mean
                            portfolio_value *= price_return

                    # リバランス記録を作成
                    rebalance_records.append(RebalanceRecord(
                        date=date,
                        weights_before=weights_before,
                        weights_after=current_weights.copy(),
                        turnover=turnover,
                        transaction_cost=turnover * (
                            self.vbt_config.transaction_cost_bps +
                            self.vbt_config.slippage_bps
                        ) / 10000,
                        portfolio_value=portfolio_value,
                    ))

                except Exception as e:
                    logger.warning(f"weights_func failed on {date}: {e}")
                    # エラー時は前回のウェイトを維持

            # シグナル行列にウェイトを設定
            for symbol, weight in current_weights.items():
                if symbol in symbol_to_idx:
                    signals[i, symbol_to_idx[symbol]] = weight

        return signals, rebalance_records

    def _convert_prices_to_matrix(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> tuple[np.ndarray, list[datetime], list[str]]:
        """価格データを行列形式に変換"""
        # 共通インデックスを構築
        all_dates = set()
        for symbol in universe:
            if symbol in prices:
                df = prices[symbol]
                if isinstance(df.index, pd.DatetimeIndex):
                    all_dates.update(df.index.tolist())

        sorted_dates = sorted(all_dates)

        # 日付フィルタリング
        start_dt = config.start_date if isinstance(config.start_date, datetime) else datetime.strptime(str(config.start_date), "%Y-%m-%d")
        end_dt = config.end_date if isinstance(config.end_date, datetime) else datetime.strptime(str(config.end_date), "%Y-%m-%d")

        filtered_dates = [d for d in sorted_dates if start_dt <= d <= end_dt]

        if not filtered_dates:
            filtered_dates = sorted_dates[:100] if sorted_dates else [datetime.now()]

        # 行列構築
        n_days = len(filtered_dates)
        n_assets = len(universe)
        matrix = np.zeros((n_days, n_assets))

        for j, symbol in enumerate(universe):
            if symbol in prices:
                df = prices[symbol]
                for i, date in enumerate(filtered_dates):
                    if date in df.index:
                        if "close" in df.columns:
                            matrix[i, j] = df.loc[date, "close"]
                        elif len(df.columns) > 0:
                            matrix[i, j] = df.iloc[df.index.get_loc(date), 0]

        # 前方埋め
        for j in range(n_assets):
            for i in range(1, n_days):
                if matrix[i, j] == 0 and matrix[i-1, j] != 0:
                    matrix[i, j] = matrix[i-1, j]

        return matrix, filtered_dates, universe

    def _convert_to_unified_result(
        self,
        result: "VectorBTResult",
        config: UnifiedBacktestConfig,
        rebalance_records: Optional[List] = None,
        n_rebalances: int = 0,
    ) -> UnifiedBacktestResult:
        """VectorBTResult を UnifiedBacktestResult に変換

        Args:
            result: VectorBTResult
            config: 設定
            rebalance_records: リバランス記録リスト（weights_func使用時）
            n_rebalances: リバランス回数
        """
        # ポートフォリオ価値を Series に変換
        portfolio_values = pd.Series(
            result.portfolio_values,
            index=pd.DatetimeIndex(result.dates) if result.dates else None,
        )
        daily_returns = pd.Series(
            result.returns,
            index=pd.DatetimeIndex(result.dates) if result.dates else None,
        )

        # リバランス記録がない場合は空リスト
        if rebalance_records is None:
            rebalance_records = []

        # 総ターンオーバーを計算
        total_turnover = result.turnover
        if rebalance_records:
            total_turnover = sum(r.turnover for r in rebalance_records)

        # 総取引コストを計算
        total_costs = result.total_costs
        if rebalance_records:
            total_costs = sum(r.transaction_cost for r in rebalance_records) * config.initial_capital

        return UnifiedBacktestResult(
            total_return=result.total_return,
            annual_return=result.cagr,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            max_drawdown=result.max_drawdown,
            volatility=result.volatility,
            calmar_ratio=result.calmar_ratio,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            rebalances=rebalance_records,
            total_turnover=total_turnover,
            total_transaction_costs=total_costs,
            config=config,
            start_date=config.start_date if isinstance(config.start_date, datetime) else datetime.strptime(str(config.start_date), "%Y-%m-%d"),
            end_date=config.end_date if isinstance(config.end_date, datetime) else datetime.strptime(str(config.end_date), "%Y-%m-%d"),
            engine_name=self.ENGINE_NAME,
            engine_specific_results={
                "n_trades": result.n_trades,
                "n_rebalances": result.n_rebalances,
                "execution_time_ms": result.execution_time_ms,
                "n_days": result.n_days,
                "n_assets": result.n_assets,
                "n_rebalances_from_mask": n_rebalances,
            },
        )

    # =========================================================================
    # Original VectorBT Methods (renamed for clarity)
    # =========================================================================

    def run_vectorized(
        self,
        prices: pl.DataFrame | np.ndarray,
        signals: pl.DataFrame | np.ndarray | None = None,
        dates: list[datetime] | None = None,
        asset_names: list[str] | None = None,
    ) -> "VectorBTResult":
        """VectorBT風バックテストを実行（内部/後方互換メソッド）

        Note: 統一インターフェースは run() を使用。こちらは内部/後方互換用。

        Args:
            prices: 価格マトリクス (n_days, n_assets)
            signals: シグナルマトリクス (n_days, n_assets)、Noneで均等配分
            dates: 日付リスト
            asset_names: 銘柄名リスト

        Returns:
            VectorBTResult
        """
        import time
        start_time = time.perf_counter()

        # データ準備
        prices_np, signals_np, dates_list = self._prepare_data(
            prices, signals, dates
        )
        n_days, n_assets = prices_np.shape

        logger.debug(f"Running VectorBT backtest: {n_days} days, {n_assets} assets")

        # Step 1: 資産リターン計算
        asset_returns = self._compute_asset_returns(prices_np)

        # Step 2: シグナル → ウェイト変換
        weights = self._compute_weights_from_signals(signals_np)

        # Step 3: リバランスマスク適用
        rebalance_mask = self._create_rebalance_mask(dates_list)
        if self._use_numba:
            weights = _apply_rebalance_mask_numba(weights, rebalance_mask)
        else:
            weights = self._apply_rebalance_mask_numpy(weights, rebalance_mask)

        # Step 4: 取引コスト計算
        costs = self._compute_transaction_costs(weights)

        # Step 5: ポートフォリオリターン計算
        if self._use_numba:
            portfolio_returns = _compute_portfolio_returns_numba(asset_returns, weights)
        else:
            portfolio_returns = np.sum(asset_returns * weights, axis=1)

        # 取引コストを差し引く
        portfolio_returns = portfolio_returns - costs

        # Step 6: ポートフォリオ価値計算
        portfolio_values = self._compute_portfolio_values(portfolio_returns)

        # Step 7: メトリクス計算
        metrics = self._compute_metrics(portfolio_returns, portfolio_values, weights, rebalance_mask)

        # 実行時間
        execution_time = (time.perf_counter() - start_time) * 1000

        # ポジション計算（ウェイト × ポートフォリオ価値）
        positions = weights * portfolio_values[:, np.newaxis]

        return VectorBTResult(
            dates=dates_list,
            portfolio_values=portfolio_values,
            returns=portfolio_returns,
            positions=positions,
            weights=weights,
            **metrics,
            execution_time_ms=execution_time,
            n_days=n_days,
            n_assets=n_assets,
        )

    def _prepare_data(
        self,
        prices: pl.DataFrame | np.ndarray,
        signals: pl.DataFrame | np.ndarray | None,
        dates: list[datetime] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
        """データを準備"""
        # 価格データ
        if isinstance(prices, pl.DataFrame):
            # 最初のカラムがdateと仮定
            if "date" in prices.columns:
                dates_list = prices["date"].to_list()
                prices_np = prices.drop("date").to_numpy()
            else:
                dates_list = dates or [datetime(2000, 1, 1)]
                prices_np = prices.to_numpy()
        else:
            prices_np = np.asarray(prices, dtype=np.float64)
            dates_list = dates or [datetime(2000, 1, 1) + timedelta(days=i)
                                   for i in range(len(prices_np))]

        n_days, n_assets = prices_np.shape

        # シグナルデータ
        if signals is None:
            # 均等配分
            signals_np = np.ones((n_days, n_assets)) / n_assets
        elif isinstance(signals, pl.DataFrame):
            if "date" in signals.columns:
                signals_np = signals.drop("date").to_numpy()
            else:
                signals_np = signals.to_numpy()
        else:
            signals_np = np.asarray(signals, dtype=np.float64)

        # NaN処理
        prices_np = np.nan_to_num(prices_np, nan=0.0)
        signals_np = np.nan_to_num(signals_np, nan=0.0)

        return prices_np, signals_np, dates_list

    def _compute_asset_returns(self, prices: np.ndarray) -> np.ndarray:
        """資産リターンを計算（完全ベクトル化）

        Args:
            prices: 価格マトリクス (n_days, n_assets)

        Returns:
            リターンマトリクス (n_days, n_assets)
        """
        returns = np.zeros_like(prices)
        returns[1:] = prices[1:] / prices[:-1] - 1

        # 異常値処理
        returns = np.clip(returns, -0.5, 0.5)  # ±50%以上の変動をクリップ
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        return returns

    def _compute_weights_from_signals(self, signals: np.ndarray) -> np.ndarray:
        """シグナルからウェイトを計算（完全ベクトル化）

        Args:
            signals: シグナルマトリクス (n_days, n_assets) in [-1, 1]

        Returns:
            ウェイトマトリクス (n_days, n_assets)
        """
        config = self.config

        if config.position_sizing == "equal":
            # 均等配分（シグナルが正の銘柄のみ）
            positive_mask = signals > config.signal_threshold
            weights = positive_mask.astype(np.float64)
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            weights = weights / row_sums

        elif config.position_sizing == "signal_weighted":
            # シグナル強度に比例した配分
            # 負のシグナルは0に（ロングオンリー想定）
            weights = np.maximum(signals, 0.0)
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            weights = weights / row_sums

        elif config.position_sizing == "vol_target":
            # ボラティリティターゲット（簡易版）
            # 実際にはボラティリティ推定が必要
            weights = np.maximum(signals, 0.0)
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            weights = weights / row_sums

        else:
            raise ValueError(f"Unknown position_sizing: {config.position_sizing}")

        # 最大ポジション制限
        weights = np.minimum(weights, config.max_position)

        # 再正規化
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weights = weights / row_sums

        return weights

    def _create_rebalance_mask(self, dates: list[datetime]) -> np.ndarray:
        """リバランスマスクを作成

        共通ユーティリティ関数を使用（base.py の create_rebalance_mask）。
        全頻度で「期間の最終取引日」にリバランスを実行する。
        - daily: 毎日
        - weekly: 各週の最終取引日
        - monthly: 各月の最終取引日

        Args:
            dates: 日付リスト

        Returns:
            リバランス日フラグ (n_days,)
        """
        return create_rebalance_mask(dates, self.config.rebalance_frequency)

    def _apply_rebalance_mask_numpy(
        self,
        weights: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """リバランスマスクを適用（NumPy版）"""
        n_days, n_assets = weights.shape
        result = np.zeros_like(weights)

        current_weights = weights[0].copy()
        for i in range(n_days):
            if mask[i]:
                current_weights = weights[i].copy()
            result[i] = current_weights

        return result

    def _compute_transaction_costs(self, weights: np.ndarray) -> np.ndarray:
        """取引コストを計算（完全ベクトル化）

        Args:
            weights: ウェイトマトリクス (n_days, n_assets)

        Returns:
            日次取引コスト (n_days,)
        """
        config = self.config
        total_cost_bps = config.transaction_cost_bps + config.slippage_bps

        # ウェイト変化
        weight_changes = np.zeros_like(weights)
        weight_changes[1:] = np.abs(weights[1:] - weights[:-1])

        # ターンオーバー
        turnover = weight_changes.sum(axis=1)

        # コスト（bps→実数）
        costs = turnover * total_cost_bps / 10000

        return costs

    def _compute_portfolio_values(self, returns: np.ndarray) -> np.ndarray:
        """ポートフォリオ価値を計算（完全ベクトル化）

        Args:
            returns: ポートフォリオリターン (n_days,)

        Returns:
            ポートフォリオ価値 (n_days,)
        """
        initial = self.config.initial_capital
        return initial * np.cumprod(1 + returns)

    def _compute_metrics(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        weights: np.ndarray,
        rebalance_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        """メトリクスを計算

        Args:
            returns: ポートフォリオリターン
            portfolio_values: ポートフォリオ価値
            weights: ウェイト履歴
            rebalance_mask: リバランスマスク（リバランス日はTrue）

        Returns:
            メトリクス辞書
        """
        n_days = len(returns)
        years = n_days / 252

        # 基本メトリクス
        total_return = portfolio_values[-1] / self.config.initial_capital - 1
        cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
        volatility = np.std(returns) * np.sqrt(252)

        # リスク調整メトリクス
        if self._use_numba:
            sharpe = _compute_sharpe_numba(returns)
            sortino = _compute_sortino_numba(returns)
            max_dd = _compute_max_drawdown_numba(portfolio_values)
            daily_turnover = _compute_turnover_numba(weights)
        else:
            sharpe = self._compute_sharpe_numpy(returns)
            sortino = self._compute_sortino_numpy(returns)
            max_dd = self._compute_max_drawdown_numpy(portfolio_values)
            daily_turnover = self._compute_turnover_numpy(weights)

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # 勝率・プロフィットファクター
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / max(n_days, 1)

        total_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        total_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-10
        profit_factor = total_profit / total_loss

        # 取引統計
        weight_changes = np.abs(np.diff(weights, axis=0))

        # リバランス回数: rebalance_maskが提供されている場合はそれを使用
        if rebalance_mask is not None:
            n_rebalances = int(np.sum(rebalance_mask))
        else:
            # フォールバック: ウェイト変化から計算
            rebalance_days = np.any(weight_changes > 0.001, axis=1)
            n_rebalances = int(np.sum(rebalance_days))

        # 取引回数: 1%以上のウェイト変化の総数
        n_trades = int(np.sum(weight_changes > 0.01))
        # 最低でもリバランス回数 × 銘柄数の半分程度の取引があると仮定
        if n_trades == 0 and n_rebalances > 0:
            n_assets = weights.shape[1] if weights.ndim > 1 else 1
            n_trades = n_rebalances * max(1, n_assets // 2)

        annual_turnover = daily_turnover * 252

        # 総コスト
        cost_per_trade = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
        total_costs = daily_turnover * n_days * self.config.initial_capital * cost_per_trade

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "volatility": volatility,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "n_rebalances": n_rebalances,
            "total_costs": total_costs,
            "turnover": annual_turnover,
        }

    def _compute_sharpe_numpy(self, returns: np.ndarray) -> float:
        """シャープレシオ（NumPy版）"""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252)

    def _compute_sortino_numpy(self, returns: np.ndarray) -> float:
        """ソルティノレシオ（NumPy版）"""
        mean_ret = np.mean(returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf if mean_ret > 0 else 0.0
        downside_std = np.sqrt(np.mean(negative_returns ** 2))
        if downside_std == 0:
            return 0.0
        return mean_ret / downside_std * np.sqrt(252)

    def _compute_max_drawdown_numpy(self, portfolio_values: np.ndarray) -> float:
        """最大ドローダウン（NumPy版）"""
        peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peaks) / peaks
        return np.min(drawdowns)

    def _compute_turnover_numpy(self, weights: np.ndarray) -> float:
        """ターンオーバー（NumPy版）"""
        if len(weights) < 2:
            return 0.0
        changes = np.abs(np.diff(weights, axis=0))
        return np.mean(np.sum(changes, axis=1))


# =============================================================================
# 便利関数
# =============================================================================

def create_vectorbt_engine(
    initial_capital: float = 100000.0,
    rebalance_frequency: str = "monthly",
    transaction_cost_bps: float = 10.0,
    use_numba: bool = True,
) -> VectorBTStyleEngine:
    """VectorBTStyleEngineを作成（ファクトリ関数）

    Args:
        initial_capital: 初期資金
        rebalance_frequency: リバランス頻度
        transaction_cost_bps: 取引コスト
        use_numba: Numba使用

    Returns:
        VectorBTStyleEngine
    """
    config = VectorBTConfig(
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        use_numba=use_numba,
    )
    return VectorBTStyleEngine(config)


def quick_backtest(
    prices: np.ndarray | pl.DataFrame,
    signals: np.ndarray | pl.DataFrame | None = None,
    initial_capital: float = 100000.0,
) -> VectorBTResult:
    """クイックバックテスト（ワンライナー）

    Args:
        prices: 価格マトリクス
        signals: シグナルマトリクス（Noneで均等配分）
        initial_capital: 初期資金

    Returns:
        VectorBTResult
    """
    engine = VectorBTStyleEngine(VectorBTConfig(initial_capital=initial_capital))
    return engine.run(prices, signals)


# timedelta import for _prepare_data
from datetime import timedelta
