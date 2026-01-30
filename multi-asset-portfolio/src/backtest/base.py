"""
Backtest Engine Base Module - 全バックテストエンジン共通インターフェース

全バックテストエンジンが準拠すべき共通インターフェースを定義。
今後のエンジン追加時も統一されたAPIを強制する。

対象エンジン:
1. BacktestEngine (engine.py) - 標準Walk-Forwardエンジン
2. FastBacktestEngine (fast_engine.py) - 高速Numba/GPUエンジン
3. StreamingBacktestEngine (streaming_engine.py) - ストリーミングエンジン
4. EventDrivenEngine (将来) - イベントドリブンエンジン
5. その他の派生エンジン

設計原則:
- 全エンジンはBacktestEngineBaseを継承
- 共通のConfig/Resultを使用
- run()メソッドのシグネチャを統一
- validate_inputs()で入力検証を標準化

使用例:
    from src.backtest.base import (
        BacktestEngineBase,
        UnifiedBacktestConfig,
        UnifiedBacktestResult,
    )

    class MyCustomEngine(BacktestEngineBase):
        def run(self, universe, prices, config, weights_func=None):
            # 実装
            pass

        def validate_inputs(self, universe, prices, config):
            # 検証
            return True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RebalanceFrequency(str, Enum):
    """リバランス頻度オプション（全エンジン共通）"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @classmethod
    def from_string(cls, value: str) -> "RebalanceFrequency":
        """文字列からRebalanceFrequencyを取得"""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown rebalance frequency: {value}")


# =============================================================================
# リバランス日判定ユーティリティ（全エンジン共通）
# =============================================================================


def _is_period_end(
    i: int,
    dt: datetime,
    dates: List[datetime],
    n_days: int,
    period_key_func: Callable[[datetime], Any],
    fallback_interval: int,
) -> bool:
    """
    期間終了日かどうかを判定するヘルパー関数。

    Args:
        i: 現在のインデックス
        dt: 現在の日付
        dates: 日付リスト
        n_days: 日付リストの長さ
        period_key_func: 期間キーを取得する関数（週番号、月、四半期など）
        fallback_interval: datetimeでない場合のフォールバック間隔

    Returns:
        bool: 期間終了日ならTrue
    """
    if not isinstance(dt, datetime):
        return i % fallback_interval == (fallback_interval - 1) or i == n_days - 1

    if i == n_days - 1:
        return True

    next_dt = dates[i + 1]
    if isinstance(next_dt, datetime):
        return period_key_func(dt) != period_key_func(next_dt)
    return i % fallback_interval == (fallback_interval - 1)


def _get_week_key(dt: datetime) -> int:
    """週番号を取得"""
    return dt.isocalendar()[1]


def _get_month_key(dt: datetime) -> int:
    """月を取得"""
    return dt.month


def _get_quarter_key(dt: datetime) -> int:
    """四半期を取得"""
    return (dt.month - 1) // 3 + 1


def create_rebalance_mask(
    dates: List[datetime],
    frequency: str,
) -> np.ndarray:
    """
    リバランスマスクを作成（全エンジン共通ロジック）

    各頻度で「期間の最終取引日」にリバランスを実行する。
    - daily: 毎日
    - weekly: 各週の最終取引日（金曜日 or 週内最終日）
    - monthly: 各月の最終取引日
    - quarterly: 各四半期の最終取引日

    Args:
        dates: 日付リスト（datetime のリスト）
        frequency: リバランス頻度 ("daily", "weekly", "monthly", "quarterly")

    Returns:
        np.ndarray: リバランス日フラグ (n_days,) boolean配列
    """
    n_days = len(dates)
    if n_days == 0:
        return np.array([], dtype=bool)

    mask = np.zeros(n_days, dtype=bool)
    freq = frequency.lower()

    # 頻度ごとの設定を辞書で管理
    freq_config = {
        "weekly": (_get_week_key, 5),
        "monthly": (_get_month_key, 20),
        "quarterly": (_get_quarter_key, 60),
    }

    if freq == "daily":
        mask[:] = True
    elif freq in freq_config:
        period_key_func, fallback_interval = freq_config[freq]
        for i, dt in enumerate(dates):
            if _is_period_end(i, dt, dates, n_days, period_key_func, fallback_interval):
                mask[i] = True

    # 最初の日は必ずリバランス（初期ポートフォリオ構築）
    if n_days > 0:
        mask[0] = True

    return mask


def get_rebalance_dates(
    dates: List[datetime],
    frequency: str,
) -> List[datetime]:
    """
    リバランス日のリストを取得

    Args:
        dates: 日付リスト
        frequency: リバランス頻度

    Returns:
        List[datetime]: リバランス日のリスト
    """
    mask = create_rebalance_mask(dates, frequency)
    return [d for d, m in zip(dates, mask) if m]


@dataclass
class UnifiedBacktestConfig:
    """
    全エンジン共通のバックテスト設定

    全バックテストエンジンはこの設定クラスを受け入れる必要がある。
    エンジン固有の設定は engine_specific_config に格納する。

    Attributes:
        start_date: バックテスト開始日
        end_date: バックテスト終了日
        initial_capital: 初期資金（デフォルト: 100,000）
        rebalance_frequency: リバランス頻度（daily/weekly/monthly/quarterly）
        transaction_cost_bps: 取引コスト（basis points、デフォルト: 10 = 0.1%）
        slippage_bps: スリッページ（basis points、デフォルト: 5 = 0.05%）
        allow_short: ショートポジション許可（デフォルト: False）
        max_weight: 最大ウェイト制約（デフォルト: 1.0 = 100%）
        min_weight: 最小ウェイト制約（デフォルト: 0.0）
        cash_symbol: キャッシュを表すシンボル（デフォルト: "CASH"）
        risk_free_rate: リスクフリーレート（年率、デフォルト: 0.02 = 2%）
        engine_specific_config: エンジン固有の設定を格納する辞書
    """

    start_date: datetime | str
    end_date: datetime | str
    initial_capital: float = 100000.0
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    allow_short: bool = False
    max_weight: float = 1.0
    min_weight: float = 0.0
    cash_symbol: str = "CASH"
    risk_free_rate: float = 0.02
    engine_specific_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """設定の検証と正規化"""
        # 日付を datetime に変換
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        if isinstance(self.end_date, str):
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        # 検証
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if not (0 <= self.min_weight <= self.max_weight <= 1):
            raise ValueError("weight constraints must satisfy: 0 <= min_weight <= max_weight <= 1")

        # rebalance_frequency正規化（大文字小文字を統一）
        self.rebalance_frequency = self.rebalance_frequency.lower()
        valid_frequencies = {"daily", "weekly", "monthly", "quarterly"}
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid rebalance_frequency: '{self.rebalance_frequency}'. "
                f"Must be one of: {sorted(valid_frequencies)}"
            )

    @property
    def total_cost_bps(self) -> float:
        """往復取引コスト（bps）"""
        return self.transaction_cost_bps + self.slippage_bps

    @property
    def total_cost_rate(self) -> float:
        """往復取引コスト（比率）"""
        return self.total_cost_bps / 10000.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "start_date": self.start_date.isoformat() if isinstance(self.start_date, datetime) else self.start_date,
            "end_date": self.end_date.isoformat() if isinstance(self.end_date, datetime) else self.end_date,
            "initial_capital": self.initial_capital,
            "rebalance_frequency": self.rebalance_frequency,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "allow_short": self.allow_short,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "cash_symbol": self.cash_symbol,
            "risk_free_rate": self.risk_free_rate,
            "engine_specific_config": self.engine_specific_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedBacktestConfig":
        """辞書から生成"""
        return cls(**data)


@dataclass
class TradeRecord:
    """
    取引記録（全エンジン共通）

    Attributes:
        date: 取引日
        symbol: 銘柄シンボル
        side: 売買方向 ("BUY" or "SELL")
        quantity: 数量
        price: 約定価格
        notional: 取引金額（quantity * price）
        cost: 取引コスト
    """

    date: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    notional: float
    cost: float

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "cost": self.cost,
        }


@dataclass
class RebalanceRecord:
    """
    リバランス記録（全エンジン共通）

    Attributes:
        date: リバランス日
        weights_before: リバランス前ウェイト
        weights_after: リバランス後ウェイト
        turnover: ターンオーバー（片道）
        transaction_cost: 取引コスト
        portfolio_value: リバランス時のポートフォリオ価値
        trades: 取引リスト
        metadata: 追加メタデータ
    """

    date: datetime
    weights_before: Dict[str, float]
    weights_after: Dict[str, float]
    turnover: float
    transaction_cost: float
    portfolio_value: float
    trades: List[TradeRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "weights_before": self.weights_before,
            "weights_after": self.weights_after,
            "turnover": self.turnover,
            "transaction_cost": self.transaction_cost,
            "portfolio_value": self.portfolio_value,
            "trades": [t.to_dict() for t in self.trades],
            "metadata": self.metadata,
        }


@dataclass
class UnifiedBacktestResult:
    """
    全エンジン共通のバックテスト結果

    全バックテストエンジンはこの結果クラスを返す必要がある。
    エンジン固有の結果は engine_specific_results に格納する。

    Attributes:
        # 基本メトリクス
        total_return: 累積リターン（例: 0.5 = 50%）
        annual_return: 年率リターン
        sharpe_ratio: シャープレシオ（年率化）
        sortino_ratio: ソルティノレシオ（年率化）
        max_drawdown: 最大ドローダウン（負の値、例: -0.15 = -15%）
        volatility: 年率ボラティリティ
        calmar_ratio: カルマーレシオ（年率リターン / |最大DD|）

        # 時系列データ
        daily_returns: 日次リターンSeries
        portfolio_values: ポートフォリオ価値Series

        # 取引統計
        trades: 取引リスト
        rebalances: リバランス記録リスト
        total_turnover: 総ターンオーバー
        total_transaction_costs: 総取引コスト

        # 設定・メタデータ
        config: 使用した設定
        start_date: 実際の開始日
        end_date: 実際の終了日
        engine_name: 使用したエンジン名
        engine_specific_results: エンジン固有の結果
        warnings: 警告メッセージ
        errors: エラーメッセージ
    """

    # 基本メトリクス
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0

    # 追加メトリクス
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0

    # 時系列データ
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # 取引統計
    trades: List[TradeRecord] = field(default_factory=list)
    rebalances: List[RebalanceRecord] = field(default_factory=list)
    total_turnover: float = 0.0
    total_transaction_costs: float = 0.0

    # 設定・メタデータ
    config: Optional[UnifiedBacktestConfig] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    engine_name: str = "unknown"
    engine_specific_results: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def final_value(self) -> float:
        """最終ポートフォリオ価値"""
        if len(self.portfolio_values) > 0:
            return float(self.portfolio_values.iloc[-1])
        if self.config:
            return self.config.initial_capital * (1 + self.total_return)
        return 0.0

    @property
    def initial_value(self) -> float:
        """初期ポートフォリオ価値"""
        if self.config:
            return self.config.initial_capital
        if len(self.portfolio_values) > 0:
            return float(self.portfolio_values.iloc[0])
        return 0.0

    @property
    def n_days(self) -> int:
        """取引日数"""
        return len(self.daily_returns)

    @property
    def n_rebalances(self) -> int:
        """リバランス回数"""
        return len(self.rebalances)

    @property
    def n_trades(self) -> int:
        """取引回数"""
        return len(self.trades)

    @property
    def avg_turnover(self) -> float:
        """平均ターンオーバー"""
        if self.n_rebalances == 0:
            return 0.0
        return self.total_turnover / self.n_rebalances

    @property
    def is_successful(self) -> bool:
        """バックテストが成功したか"""
        return len(self.errors) == 0 and len(self.portfolio_values) > 0

    def calculate_metrics(self, risk_free_rate: float = 0.02) -> None:
        """
        時系列データから全メトリクスを計算

        Args:
            risk_free_rate: 年率リスクフリーレート（デフォルト: 2%）
        """
        if len(self.daily_returns) == 0:
            return

        returns = self.daily_returns.values
        n_days = len(returns)

        # 基本統計
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if n_days > 1 else 0.0

        # Total return
        self.total_return = float(np.prod(1 + returns) - 1)

        # Annual return
        if n_days > 0:
            self.annual_return = float((1 + self.total_return) ** (252 / n_days) - 1)

        # Volatility (annualized)
        self.volatility = float(std_return * np.sqrt(252))

        # Sharpe ratio
        daily_rf = risk_free_rate / 252
        if std_return > 0:
            self.sharpe_ratio = float((mean_return - daily_rf) / std_return * np.sqrt(252))

        # Sortino ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns, ddof=1)
            if downside_std > 0:
                self.sortino_ratio = float((mean_return - daily_rf) / downside_std * np.sqrt(252))

        # Max Drawdown
        if len(self.portfolio_values) > 0:
            values = self.portfolio_values.values
            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max
            self.max_drawdown = float(np.min(drawdowns))

        # Calmar ratio
        if self.max_drawdown < 0:
            self.calmar_ratio = float(self.annual_return / abs(self.max_drawdown))

        # Win rate
        winning_days = np.sum(returns > 0)
        self.win_rate = float(winning_days / n_days) if n_days > 0 else 0.0

        # Profit factor
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        if losses > 0:
            self.profit_factor = float(gains / losses)
        elif gains > 0:
            self.profit_factor = float("inf")

        # VaR (95%)
        self.var_95 = float(np.percentile(returns, 5))

        # Expected Shortfall (CVaR 95%)
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            self.expected_shortfall = float(np.mean(tail_returns))

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "metrics": {
                "total_return": self.total_return,
                "annual_return": self.annual_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "calmar_ratio": self.calmar_ratio,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "var_95": self.var_95,
                "expected_shortfall": self.expected_shortfall,
            },
            "trading_stats": {
                "n_days": self.n_days,
                "n_rebalances": self.n_rebalances,
                "n_trades": self.n_trades,
                "total_turnover": self.total_turnover,
                "avg_turnover": self.avg_turnover,
                "total_transaction_costs": self.total_transaction_costs,
            },
            "values": {
                "initial_value": self.initial_value,
                "final_value": self.final_value,
            },
            "config": self.config.to_dict() if self.config else None,
            "engine_name": self.engine_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "warnings": self.warnings,
            "errors": self.errors,
            "is_successful": self.is_successful,
        }

    def summary(self) -> str:
        """人間可読なサマリ文字列を生成"""
        lines = [
            "=" * 60,
            f"  BACKTEST RESULT SUMMARY ({self.engine_name})",
            "=" * 60,
            "",
            f"  Period: {self.start_date.strftime('%Y-%m-%d') if self.start_date else 'N/A'} to "
            f"{self.end_date.strftime('%Y-%m-%d') if self.end_date else 'N/A'}",
            f"  Initial Capital: ${self.initial_value:,.2f}",
            f"  Final Value: ${self.final_value:,.2f}",
            "",
            "-" * 60,
            "  RETURNS",
            "-" * 60,
            f"  Total Return:      {self.total_return * 100:>10.2f}%",
            f"  Annual Return:     {self.annual_return * 100:>10.2f}%",
            f"  Volatility (ann.): {self.volatility * 100:>10.2f}%",
            "",
            "-" * 60,
            "  RISK-ADJUSTED METRICS",
            "-" * 60,
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:      {self.calmar_ratio:>10.3f}",
            "",
            "-" * 60,
            "  RISK METRICS",
            "-" * 60,
            f"  Max Drawdown:      {self.max_drawdown * 100:>10.2f}%",
            f"  VaR (95%):         {self.var_95 * 100:>10.2f}%",
            f"  Win Rate:          {self.win_rate * 100:>10.2f}%",
            "",
            "-" * 60,
            "  TRADING STATISTICS",
            "-" * 60,
            f"  Days:              {self.n_days:>10d}",
            f"  Rebalances:        {self.n_rebalances:>10d}",
            f"  Trades:            {self.n_trades:>10d}",
            f"  Total Turnover:    {self.total_turnover * 100:>10.2f}%",
            f"  Transaction Costs: ${self.total_transaction_costs:>10.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"UnifiedBacktestResult("
            f"engine={self.engine_name}, "
            f"return={self.total_return * 100:.2f}%, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown * 100:.2f}%)"
        )


class WeightsFuncProtocol(Protocol):
    """
    ウェイト計算関数の統一プロトコル（全エンジン共通）

    全バックテストエンジン（FastBacktestEngine, StreamingBacktestEngine,
    RayBacktestEngine, VectorBTStyleEngine）で同一のシグネチャを使用。

    エンジン固有の内部形式（signals, cov_matrix 等）を使用する場合は、
    各エンジンが内部でアダプター変換を行う。

    Example:
        def my_weights_func(
            universe: List[str],
            prices: Dict[str, pd.DataFrame],
            date: datetime,
            current_weights: Dict[str, float],
        ) -> Dict[str, float]:
            # 各銘柄の新しいウェイトを計算
            n = len(universe)
            return {symbol: 1.0 / n for symbol in universe}

        # 全エンジンで同一関数を使用可能
        result = engine.run(prices=prices, weights_func=my_weights_func)
    """

    def __call__(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        date: datetime,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        ウェイトを計算

        Args:
            universe: ユニバース（銘柄リスト）
            prices: 価格データ（{symbol: DataFrame with 'close' column}）
                    各DataFrameはDatetimeIndexを持つ
            date: 現在のリバランス日（この日までのデータで計算）
            current_weights: 現在のウェイト（{symbol: weight}）
                            初回リバランス時は空の辞書または等ウェイト

        Returns:
            新しいウェイト（{symbol: weight}）
            - 合計は1.0になるべき（エンジンが正規化する場合もある）
            - 負の値はショートポジション（allow_short=True時のみ）
        """
        ...


class BacktestEngineBase(ABC):
    """
    全バックテストエンジンの抽象基底クラス

    全てのバックテストエンジンはこのクラスを継承し、
    run() と validate_inputs() を実装する必要がある。

    Usage:
        class MyEngine(BacktestEngineBase):
            def run(self, universe, prices, config, weights_func=None):
                # 実装
                pass

            def validate_inputs(self, universe, prices, config):
                # 検証
                return True
    """

    # エンジン名（サブクラスでオーバーライド）
    ENGINE_NAME: str = "base"

    def __init__(self, config: Optional[UnifiedBacktestConfig] = None) -> None:
        """
        初期化

        Args:
            config: バックテスト設定（後でrun()で指定も可）
        """
        self._config = config
        self._logger = logger

    @property
    def config(self) -> Optional[UnifiedBacktestConfig]:
        """現在の設定を取得"""
        return self._config

    @config.setter
    def config(self, value: UnifiedBacktestConfig) -> None:
        """設定を更新"""
        self._config = value

    @abstractmethod
    def run(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[Callable] = None,
    ) -> UnifiedBacktestResult:
        """
        バックテストを実行

        全エンジンはこのシグネチャに準拠する必要がある。

        Args:
            universe: ユニバース（銘柄リスト）
            prices: 価格データ（{symbol: DataFrame}、またはDataFrame with symbol columns）
            config: バックテスト設定（Noneの場合はコンストラクタの設定を使用）
            weights_func: ウェイト計算関数（Noneの場合はエンジンデフォルト）

        Returns:
            UnifiedBacktestResult: バックテスト結果

        Raises:
            ValueError: 入力が無効な場合
            RuntimeError: バックテスト実行エラー
        """
        pass

    @abstractmethod
    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """
        入力を検証

        Args:
            universe: ユニバース
            prices: 価格データ
            config: 設定

        Returns:
            bool: 検証結果（True=有効）

        Raises:
            ValueError: 検証エラーの詳細
        """
        pass

    def _validate_common_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> List[str]:
        """
        共通の入力検証（全エンジンで使用可能）

        Args:
            universe: ユニバース
            prices: 価格データ
            config: 設定

        Returns:
            List[str]: 警告メッセージのリスト
        """
        warnings = []

        # ユニバースチェック
        if not universe:
            raise ValueError("Universe cannot be empty")

        # 価格データチェック
        if not prices:
            raise ValueError("Price data cannot be empty")

        # ユニバースと価格データの整合性
        missing_symbols = [s for s in universe if s not in prices]
        if missing_symbols:
            warnings.append(f"Missing price data for symbols: {missing_symbols[:5]}...")

        # 日付範囲チェック
        for symbol, df in prices.items():
            if df.empty:
                warnings.append(f"Empty price data for {symbol}")
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                min_date = df.index.min()
                max_date = df.index.max()

                if min_date > config.start_date:
                    warnings.append(
                        f"{symbol}: data starts at {min_date.date()}, "
                        f"after backtest start {config.start_date}"
                    )

        return warnings

    def _create_empty_result(
        self,
        config: UnifiedBacktestConfig,
        error_message: str,
    ) -> UnifiedBacktestResult:
        """
        エラー時の空結果を生成

        Args:
            config: 設定
            error_message: エラーメッセージ

        Returns:
            UnifiedBacktestResult: 空の結果
        """
        return UnifiedBacktestResult(
            config=config,
            start_date=config.start_date if isinstance(config.start_date, datetime) else None,
            end_date=config.end_date if isinstance(config.end_date, datetime) else None,
            engine_name=self.ENGINE_NAME,
            errors=[error_message],
        )

    def _calculate_turnover(
        self,
        weights_before: Dict[str, float],
        weights_after: Dict[str, float],
    ) -> float:
        """
        ターンオーバーを計算（片道）

        Args:
            weights_before: 変更前ウェイト
            weights_after: 変更後ウェイト

        Returns:
            float: ターンオーバー（0-1、片道）
        """
        all_symbols = set(weights_before.keys()) | set(weights_after.keys())
        total_change = sum(
            abs(weights_after.get(s, 0.0) - weights_before.get(s, 0.0))
            for s in all_symbols
        )
        return total_change / 2  # 片道

    def _apply_weight_constraints(
        self,
        weights: Dict[str, float],
        config: UnifiedBacktestConfig,
    ) -> Dict[str, float]:
        """
        ウェイト制約を適用

        Args:
            weights: 元のウェイト
            config: 設定

        Returns:
            Dict[str, float]: 制約適用後のウェイト
        """
        # クリップ
        clipped = {
            symbol: max(config.min_weight, min(config.max_weight, w))
            for symbol, w in weights.items()
        }

        # 正規化（合計を1に）
        total = sum(clipped.values())
        if total > 0:
            return {symbol: w / total for symbol, w in clipped.items()}
        else:
            # 全て0の場合はキャッシュ100%
            return {config.cash_symbol: 1.0}


# 型エイリアス
PriceData = Union[Dict[str, pd.DataFrame], pd.DataFrame]
WeightsFunc = Callable[
    [List[str], Dict[str, pd.DataFrame], datetime, Dict[str, float]],
    Dict[str, float],
]
