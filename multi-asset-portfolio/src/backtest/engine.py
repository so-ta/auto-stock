"""
Backtest Engine - Walk-Forward Historical Simulation

This module implements a proper walk-forward backtesting framework that
ensures no data leakage. Each rebalance decision uses ONLY data available
at that point in time.

Key Design Principles:
1. Data Cutoff: At each rebalance date, only use data up to that date
2. Walk-Forward: Train on past data, evaluate on holdout, allocate
3. No Lookahead: Never peek at future data for any decision
4. Realistic Costs: Apply transaction costs at each rebalance

Usage:
    from datetime import datetime
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.config.settings import Settings

    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        rebalance_frequency="monthly",
        train_period_days=504,
        initial_capital=10000.0,
        transaction_cost_bps=10.0,
    )

    engine = BacktestEngine(config, Settings())
    result = engine.run(universe=["AAPL", "MSFT", "GOOGL"])
    print(result.total_return)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from src.config.settings import Settings
    from src.orchestrator.pipeline import Pipeline

from .base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    RebalanceRecord as UnifiedRebalanceRecord,
    TradeRecord,
)
from .cache import (
    SignalCache,
    batch_compute_signals,
    vectorized_momentum,
    vectorized_sharpe,
)
from .rebalance_scheduler import (
    EventDrivenRebalanceScheduler,
    PortfolioState,
    MarketData,
    TriggerType,
    create_default_scheduler,
)

logger = logging.getLogger(__name__)

# Number of parallel workers (default to CPU count)
DEFAULT_WORKERS = min(os.cpu_count() or 4, 8)


class RebalanceFrequency(str, Enum):
    """Rebalance frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution.

    Attributes:
        start_date: Backtest start date (first potential rebalance)
        end_date: Backtest end date
        rebalance_frequency: How often to rebalance (daily/weekly/monthly)
        train_period_days: Days of historical data for training (default: 504 = 2 years)
        initial_capital: Starting capital (default: 10000)
        transaction_cost_bps: Transaction cost in basis points (default: 10)
        slippage_bps: Slippage in basis points (default: 5)
        allow_short: Allow short positions (default: False)
        cash_symbol: Symbol for cash position (default: "CASH")
        parallel_workers: Number of parallel workers for data fetching (default: auto)
        enable_signal_cache: Enable signal computation caching (default: True)
        cache_dir: Directory for cache files (default: ./cache/backtest)
        use_fast_mode: Enable fast backtest mode using FastBacktestEngine (default: True)
        precompute_signals: Precompute all signals before backtest (default: True)
        use_incremental_cov: Use incremental covariance updates (default: True)
    """

    start_date: datetime
    end_date: datetime
    rebalance_frequency: str = "monthly"
    train_period_days: int = 504
    initial_capital: float = 10000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    allow_short: bool = False
    cash_symbol: str = "CASH"
    parallel_workers: int = 0  # 0 = auto (CPU count)
    enable_signal_cache: bool = True
    cache_dir: str = "./cache/backtest"
    # 高速化オプション（新規追加）
    use_fast_mode: bool = True
    precompute_signals: bool = True
    use_incremental_cov: bool = True
    # イベントドリブンリバランス
    use_event_driven: bool = False
    event_min_interval_days: int = 5
    event_position_threshold: float = 0.05
    event_vix_threshold: float = 0.30
    event_drawdown_threshold: float = 0.10
    event_max_triggers_per_month: int = 3

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.train_period_days <= 0:
            raise ValueError("train_period_days must be positive")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")

    @property
    def data_start_date(self) -> datetime:
        """Calculate the earliest date needed for training data."""
        return self.start_date - timedelta(days=self.train_period_days + 30)

    @property
    def total_cost_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return self.transaction_cost_bps + self.slippage_bps

    def to_unified_config(self) -> UnifiedBacktestConfig:
        """
        Convert to UnifiedBacktestConfig for INT-001 compatibility.

        Returns:
            UnifiedBacktestConfig: Unified configuration object
        """
        return UnifiedBacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            rebalance_frequency=self.rebalance_frequency,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps,
            allow_short=self.allow_short,
            cash_symbol=self.cash_symbol,
            engine_specific_config={
                "train_period_days": self.train_period_days,
                "parallel_workers": self.parallel_workers,
                "enable_signal_cache": self.enable_signal_cache,
                "cache_dir": self.cache_dir,
                "use_fast_mode": self.use_fast_mode,
                "precompute_signals": self.precompute_signals,
                "use_incremental_cov": self.use_incremental_cov,
                "use_event_driven": self.use_event_driven,
                "event_min_interval_days": self.event_min_interval_days,
                "event_position_threshold": self.event_position_threshold,
                "event_vix_threshold": self.event_vix_threshold,
                "event_drawdown_threshold": self.event_drawdown_threshold,
                "event_max_triggers_per_month": self.event_max_triggers_per_month,
            },
        )

    @classmethod
    def from_unified_config(cls, unified: UnifiedBacktestConfig) -> "BacktestConfig":
        """
        Create BacktestConfig from UnifiedBacktestConfig.

        Args:
            unified: Unified configuration object

        Returns:
            BacktestConfig: Engine-specific configuration
        """
        engine_config = unified.engine_specific_config or {}
        return cls(
            start_date=unified.start_date if isinstance(unified.start_date, datetime) else datetime.fromisoformat(str(unified.start_date)),
            end_date=unified.end_date if isinstance(unified.end_date, datetime) else datetime.fromisoformat(str(unified.end_date)),
            rebalance_frequency=unified.rebalance_frequency,
            train_period_days=engine_config.get("train_period_days", 504),
            initial_capital=unified.initial_capital,
            transaction_cost_bps=unified.transaction_cost_bps,
            slippage_bps=unified.slippage_bps,
            allow_short=unified.allow_short,
            cash_symbol=unified.cash_symbol,
            parallel_workers=engine_config.get("parallel_workers", 0),
            enable_signal_cache=engine_config.get("enable_signal_cache", True),
            cache_dir=engine_config.get("cache_dir", "./cache/backtest"),
            use_fast_mode=engine_config.get("use_fast_mode", True),
            precompute_signals=engine_config.get("precompute_signals", True),
            use_incremental_cov=engine_config.get("use_incremental_cov", True),
            use_event_driven=engine_config.get("use_event_driven", False),
            event_min_interval_days=engine_config.get("event_min_interval_days", 5),
            event_position_threshold=engine_config.get("event_position_threshold", 0.05),
            event_vix_threshold=engine_config.get("event_vix_threshold", 0.30),
            event_drawdown_threshold=engine_config.get("event_drawdown_threshold", 0.10),
            event_max_triggers_per_month=engine_config.get("event_max_triggers_per_month", 3),
        )


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event.

    Attributes:
        date: Rebalance date
        weights_before: Weights before rebalance
        weights_after: Weights after rebalance
        turnover: Total turnover (sum of absolute weight changes)
        transaction_cost: Total transaction cost incurred
        portfolio_value: Portfolio value at rebalance
        signals_used: Number of signals used in decision
        strategies_adopted: Number of strategies adopted
    """

    date: datetime
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    turnover: float
    transaction_cost: float
    portfolio_value: float
    signals_used: int = 0
    strategies_adopted: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from backtest execution.

    Attributes:
        config: Backtest configuration used
        portfolio_values: Time series of portfolio values
        returns: Time series of returns
        weights_history: History of weights at each rebalance
        rebalance_records: Detailed records of each rebalance
        metrics: Performance metrics (Sharpe, MDD, etc.)
        errors: Any errors encountered
        warnings: Any warnings generated
    """

    config: BacktestConfig
    portfolio_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights_history: list[dict[str, float]] = field(default_factory=list)
    rebalance_records: list[RebalanceRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """Calculate total return."""
        if len(self.portfolio_values) < 2:
            return 0.0
        return (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1

    @property
    def annualized_return(self) -> float:
        """Calculate annualized return."""
        if len(self.portfolio_values) < 2:
            return 0.0
        days = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        total_ret = self.total_return
        if total_ret <= -1:
            return -1.0
        return (1 + total_ret) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_values) < 2:
            return 0.0
        cummax = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cummax) / cummax
        return float(drawdown.min())

    @property
    def volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        return float(self.returns.std() * np.sqrt(252))

    @property
    def total_turnover(self) -> float:
        """Calculate total turnover across all rebalances."""
        return sum(r.turnover for r in self.rebalance_records)

    @property
    def total_transaction_costs(self) -> float:
        """Calculate total transaction costs."""
        return sum(r.transaction_cost for r in self.rebalance_records)

    @property
    def rebalance_count(self) -> int:
        """Return number of rebalances."""
        return len(self.rebalance_records)

    @property
    def avg_turnover(self) -> float:
        """Calculate average turnover per rebalance."""
        if not self.rebalance_records:
            return 0.0
        return self.total_turnover / len(self.rebalance_records)

    @property
    def total_costs(self) -> float:
        """Alias for total_transaction_costs."""
        return self.total_transaction_costs

    @property
    def final_value(self) -> float:
        """Get final portfolio value."""
        if len(self.portfolio_values) == 0:
            return self.config.initial_capital
        return float(self.portfolio_values.iloc[-1])

    @property
    def final_weights(self) -> dict[str, float]:
        """Get final portfolio weights."""
        if self.rebalance_records:
            return self.rebalance_records[-1].weights_after
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "rebalance_frequency": self.config.rebalance_frequency,
                "train_period_days": self.config.train_period_days,
                "initial_capital": self.config.initial_capital,
                "transaction_cost_bps": self.config.transaction_cost_bps,
            },
            "metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "total_turnover": self.total_turnover,
                "total_transaction_costs": self.total_transaction_costs,
                "n_rebalances": len(self.rebalance_records),
            },
            "portfolio_values": self.portfolio_values.to_dict() if len(self.portfolio_values) > 0 else {},
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_unified_result(self) -> UnifiedBacktestResult:
        """
        Convert to UnifiedBacktestResult for INT-001 compatibility.

        Returns:
            UnifiedBacktestResult: Unified result object
        """
        # Convert RebalanceRecord to unified format
        unified_rebalances = []
        for rec in self.rebalance_records:
            unified_rec = UnifiedRebalanceRecord(
                date=rec.date,
                weights_before=rec.weights_before,
                weights_after=rec.weights_after,
                turnover=rec.turnover,
                transaction_cost=rec.transaction_cost,
                portfolio_value=rec.portfolio_value,
                trades=[],  # Original doesn't track individual trades
                metadata=rec.metadata if hasattr(rec, 'metadata') else {},
            )
            unified_rebalances.append(unified_rec)

        # Calculate Sortino ratio if not available
        sortino = 0.0
        if len(self.returns) > 0:
            negative_returns = self.returns[self.returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 0:
                    mean_return = self.returns.mean()
                    sortino = (mean_return * np.sqrt(252)) / (downside_std * np.sqrt(252))

        # Calculate Calmar ratio
        calmar = 0.0
        if self.max_drawdown < 0:
            calmar = self.annualized_return / abs(self.max_drawdown)

        # Calculate win rate
        win_rate = 0.0
        if len(self.returns) > 0:
            win_rate = (self.returns > 0).mean()

        result = UnifiedBacktestResult(
            total_return=self.total_return,
            annual_return=self.annualized_return,
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=sortino,
            max_drawdown=self.max_drawdown,
            volatility=self.volatility,
            calmar_ratio=calmar,
            win_rate=win_rate,
            daily_returns=self.returns,
            portfolio_values=self.portfolio_values,
            trades=[],  # Original doesn't track individual trades
            rebalances=unified_rebalances,
            total_turnover=self.total_turnover,
            total_transaction_costs=self.total_transaction_costs,
            config=self.config.to_unified_config() if self.config else None,
            start_date=self.config.start_date if self.config else None,
            end_date=self.config.end_date if self.config else None,
            engine_name="BacktestEngine",
            warnings=self.warnings,
            errors=self.errors,
        )
        return result


class BacktestEngine(BacktestEngineBase):
    """
    Walk-forward backtest engine with strict data leakage prevention.

    This engine ensures that each rebalance decision is made using ONLY
    data that would have been available at that point in time. No future
    data is ever used in any decision.

    Architecture:
    1. Fetch all historical data upfront (for efficiency)
    2. For each rebalance date:
       a. Filter data to cutoff date (exclude future)
       b. Run signal generation on filtered data
       c. Run strategy evaluation on filtered data
       d. Compute optimal weights
       e. Apply transaction costs
       f. Update portfolio

    Usage:
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            rebalance_frequency="monthly",
        )
        engine = BacktestEngine(config, settings)
        result = engine.run(["AAPL", "MSFT", "GOOGL"])

    INT-001 Unified Interface:
        # Using unified config
        unified_config = UnifiedBacktestConfig(...)
        engine = BacktestEngine.from_unified_config(unified_config)
        result = engine.run_unified(universe, prices, unified_config)
    """

    # INT-001: エンジン名（BacktestEngineBase準拠）
    ENGINE_NAME: str = "BacktestEngine"

    def __init__(
        self,
        config: BacktestConfig | None = None,
        settings: "Settings | None" = None,
        unified_config: UnifiedBacktestConfig | None = None,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration (legacy API)
            settings: Application settings (optional, uses defaults if None)
            unified_config: Unified configuration (INT-001 API)
        """
        # INT-001: Initialize base class
        super().__init__(unified_config)

        # Handle both legacy and unified config
        if unified_config is not None and config is None:
            config = BacktestConfig.from_unified_config(unified_config)
        elif config is not None and unified_config is None:
            self._config = config.to_unified_config()

        self.config = config
        self._settings = settings
        self._logger = logger.bind(component="backtest_engine") if hasattr(logger, "bind") else logger

        # Data storage (Polars DataFrames for performance)
        self._all_data: dict[str, pl.DataFrame] = {}
        self._price_matrix: pl.DataFrame | None = None
        # Pandas compatibility cache
        self._price_matrix_pd: pd.DataFrame | None = None

        # State
        self._current_weights: dict[str, float] = {}
        initial_capital = config.initial_capital if config else 10000.0
        self._portfolio_value: float = initial_capital
        self._cash: float = initial_capital

        # Parallel processing
        self._n_workers = (config.parallel_workers if config and config.parallel_workers > 0 else DEFAULT_WORKERS)

        # Signal cache for performance
        self._signal_cache: SignalCache | None = None
        if config and config.enable_signal_cache:
            self._signal_cache = SignalCache(
                cache_dir=config.cache_dir,
                max_memory_entries=500,
                max_memory_mb=256,
                enable_disk_cache=True,
            )

        # Event-driven rebalance scheduler
        self._rebalance_scheduler: EventDrivenRebalanceScheduler | None = None
        if config and config.use_event_driven:
            self._rebalance_scheduler = create_default_scheduler(
                position_threshold=config.event_position_threshold,
                vix_threshold=config.event_vix_threshold,
                drawdown_threshold=config.event_drawdown_threshold,
                min_interval_days=config.event_min_interval_days,
            )
            self._rebalance_scheduler.max_triggers_per_month = config.event_max_triggers_per_month

        # VIX data cache for event-driven mode
        self._vix_data: pd.Series | None = None
        self._peak_value: float = initial_capital

    @property
    def settings(self) -> "Settings":
        """Get settings instance."""
        if self._settings is None:
            from src.config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    def run(self, universe: list[str]) -> BacktestResult:
        """
        Execute the backtest.

        Args:
            universe: List of asset symbols to trade

        Returns:
            BacktestResult with performance metrics and history
        """
        self._logger.info(
            "Starting backtest",
            start_date=self.config.start_date.isoformat(),
            end_date=self.config.end_date.isoformat(),
            universe_size=len(universe),
            rebalance_frequency=self.config.rebalance_frequency,
            fast_mode=self.config.use_fast_mode,
        )

        # 高速モードの場合はFastBacktestEngineを使用
        if self.config.use_fast_mode:
            try:
                from .fast_engine import FastBacktestEngine
                fast_engine = FastBacktestEngine(self.config, self._settings)
                return fast_engine.run(universe)
            except ImportError:
                self._logger.warning(
                    "FastBacktestEngine not available, falling back to standard mode"
                )
            except (RuntimeError, TypeError, ValueError) as e:
                self._logger.warning(
                    f"Fast mode failed: {e}, falling back to standard mode"
                )

        # イベントドリブンモード
        if self.config.use_event_driven:
            return self._run_event_driven(universe)

        # 標準モード
        return self._run_standard(universe)

    # ═══════════════════════════════════════════════════════════════
    # INT-001: BacktestEngineBase 統一インターフェース実装
    # ═══════════════════════════════════════════════════════════════

    def run_unified(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[Callable] = None,
    ) -> UnifiedBacktestResult:
        """
        INT-001: BacktestEngineBase準拠の統一run()メソッド

        既存のrun()メソッドとの後方互換性を保ちつつ、
        統一インターフェースを提供する。

        Args:
            universe: ユニバース（銘柄リスト）
            prices: 価格データ（{symbol: DataFrame}）
            config: 統一設定（Noneの場合はコンストラクタの設定を使用）
            weights_func: ウェイト計算関数（Noneの場合は内蔵スコアリング）

        Returns:
            UnifiedBacktestResult: 統一結果オブジェクト
        """
        # 設定の処理
        if config is not None:
            self.config = BacktestConfig.from_unified_config(config)
            self._config = config
        elif self._config is None and self.config is not None:
            self._config = self.config.to_unified_config()

        # 入力検証
        effective_config = config or self._config
        if effective_config is not None:
            self.validate_inputs(universe, prices, effective_config)

        # 外部から価格データが渡された場合は内部キャッシュに設定
        if prices:
            self._inject_external_prices(prices)

        # 既存のrunメソッドを実行
        legacy_result = self.run(universe)

        # 結果を統一フォーマットに変換
        return legacy_result.to_unified_result()

    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """
        INT-001: 入力検証

        Args:
            universe: ユニバース
            prices: 価格データ
            config: 設定

        Returns:
            bool: 検証結果（True=有効）

        Raises:
            ValueError: 検証エラーの詳細
        """
        # 共通検証を使用
        warnings = self._validate_common_inputs(universe, prices, config)

        # 警告をログに出力
        for warning in warnings:
            self._logger.warning(warning)

        # BacktestEngine固有の検証
        engine_config = config.engine_specific_config or {}
        train_period_days = engine_config.get("train_period_days", 504)

        # 訓練期間の検証
        for symbol, df in prices.items():
            if df.empty:
                continue
            if len(df) < train_period_days // 2:
                self._logger.warning(
                    f"{symbol}: insufficient data ({len(df)} rows) for "
                    f"train_period_days={train_period_days}"
                )

        return True

    def _inject_external_prices(self, prices: Dict[str, pd.DataFrame]) -> None:
        """
        外部から渡された価格データを内部キャッシュに設定

        Args:
            prices: 価格データ（{symbol: DataFrame}）
        """
        for symbol, df in prices.items():
            if df.empty:
                continue

            # DataFrameをPolars形式に変換
            if isinstance(df.index, pd.DatetimeIndex):
                df_reset = df.reset_index()
                if df_reset.columns[0] == "index":
                    df_reset = df_reset.rename(columns={"index": "timestamp"})
                pl_df = pl.from_pandas(df_reset)
            else:
                pl_df = pl.from_pandas(df.reset_index())

            self._all_data[symbol] = pl_df

        # 価格マトリクスを再構築
        if self._all_data:
            self._build_price_matrix()

    @classmethod
    def from_unified_config(
        cls,
        unified_config: UnifiedBacktestConfig,
        settings: "Settings | None" = None,
    ) -> "BacktestEngine":
        """
        INT-001: UnifiedBacktestConfigからエンジンを生成

        Args:
            unified_config: 統一設定
            settings: アプリケーション設定

        Returns:
            BacktestEngine: 設定済みエンジンインスタンス
        """
        return cls(
            config=None,
            settings=settings,
            unified_config=unified_config,
        )

    # ═══════════════════════════════════════════════════════════════

    def _run_standard(self, universe: list[str]) -> BacktestResult:
        """
        Execute the standard (non-fast) backtest.

        Args:
            universe: List of asset symbols to trade

        Returns:
            BacktestResult with performance metrics and history
        """
        result = BacktestResult(config=self.config)

        try:
            # Step 1: Fetch all historical data
            self._fetch_all_data(universe)

            if not self._all_data:
                result.errors.append("No data fetched for any symbol")
                return result

            # Step 2: Build price matrix for portfolio valuation
            self._build_price_matrix()

            if self._price_matrix is None or self._price_matrix.is_empty():
                result.errors.append("Failed to build price matrix")
                return result

            # Step 3: Get rebalance dates
            rebalance_dates = self._get_rebalance_dates()

            if not rebalance_dates:
                result.errors.append("No valid rebalance dates in range")
                return result

            self._logger.info(f"Found {len(rebalance_dates)} rebalance dates")

            # Step 4: Initialize portfolio
            self._initialize_portfolio(universe)

            # Step 5: Run simulation loop
            portfolio_values: list[tuple[datetime, float]] = []
            daily_returns: list[tuple[datetime, float]] = []

            prev_date = None
            prev_value = self.config.initial_capital

            for rebalance_date in rebalance_dates:
                try:
                    # Run pipeline with data cutoff
                    new_weights = self._run_pipeline_with_cutoff(
                        universe=universe,
                        cutoff_date=rebalance_date,
                    )

                    # Calculate turnover and costs
                    turnover = self._calculate_turnover(self._current_weights, new_weights)
                    transaction_cost = self._calculate_transaction_cost(turnover)

                    # Apply transaction cost
                    self._portfolio_value -= transaction_cost

                    # Record rebalance
                    record = RebalanceRecord(
                        date=rebalance_date,
                        weights_before=self._current_weights.copy(),
                        weights_after=new_weights.copy(),
                        turnover=turnover,
                        transaction_cost=transaction_cost,
                        portfolio_value=self._portfolio_value,
                    )
                    result.rebalance_records.append(record)

                    # Update weights
                    self._current_weights = new_weights.copy()

                    # Calculate portfolio value using prices
                    self._portfolio_value = self._calculate_portfolio_value(
                        weights=self._current_weights,
                        date=rebalance_date,
                    )

                    portfolio_values.append((rebalance_date, self._portfolio_value))

                    # Calculate return since previous rebalance
                    if prev_date is not None and prev_value > 0:
                        period_return = (self._portfolio_value / prev_value) - 1
                        daily_returns.append((rebalance_date, period_return))

                    prev_date = rebalance_date
                    prev_value = self._portfolio_value

                    self._logger.debug(
                        f"Rebalance {rebalance_date.date()}: value={self._portfolio_value:.2f}, turnover={turnover:.4f}"
                    )

                except (ValueError, ZeroDivisionError, KeyError) as e:
                    self._logger.warning(f"Rebalance failed at {rebalance_date}: {e}")
                    result.warnings.append(f"Rebalance failed at {rebalance_date}: {e}")
                    continue

            # Step 6: Build result time series
            if portfolio_values:
                dates, values = zip(*portfolio_values)
                result.portfolio_values = pd.Series(values, index=pd.DatetimeIndex(dates))

            if daily_returns:
                dates, rets = zip(*daily_returns)
                result.returns = pd.Series(rets, index=pd.DatetimeIndex(dates))

            # Step 7: Calculate final metrics
            result.metrics = self._calculate_metrics(result)

            self._logger.info(
                "Backtest completed",
                total_return=f"{result.total_return:.2%}",
                sharpe_ratio=f"{result.sharpe_ratio:.2f}",
                max_drawdown=f"{result.max_drawdown:.2%}",
                n_rebalances=len(result.rebalance_records),
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self._logger.exception(f"Backtest failed: {e}")
            result.errors.append(f"Backtest failed: {e}")
        except Exception as e:
            self._logger.exception(f"Unexpected backtest error: {e}")
            result.errors.append(f"Unexpected backtest error: {e}")
            raise

        return result

    def _run_event_driven(self, universe: list[str]) -> BacktestResult:
        """
        Execute event-driven backtest.

        月次リバランスをベースに、イベントトリガーで追加リバランスを実行。
        週次のコスト削減と月次の機会損失のバランスを取る。

        Args:
            universe: List of asset symbols to trade

        Returns:
            BacktestResult with performance metrics and history
        """
        result = BacktestResult(config=self.config)

        try:
            # Step 1: Fetch all historical data
            self._fetch_all_data(universe)

            if not self._all_data:
                result.errors.append("No data fetched for any symbol")
                return result

            # Step 2: Build price matrix
            self._build_price_matrix()

            if self._price_matrix is None:
                result.errors.append("Failed to build price matrix")
                return result

            # Step 3: Fetch VIX data for event triggers
            self._fetch_vix_data()

            # Step 4: Get ALL trading days (not just rebalance dates)
            all_trading_days = self._get_all_trading_days()

            if not all_trading_days:
                result.errors.append("No valid trading days in range")
                return result

            self._logger.info(
                f"Event-driven mode: {len(all_trading_days)} trading days",
            )

            # Step 5: Initialize portfolio
            self._initialize_portfolio(universe)
            self._peak_value = self.config.initial_capital

            # Step 6: Run simulation loop
            portfolio_values: list[tuple[datetime, float]] = []
            daily_returns: list[tuple[datetime, float]] = []

            prev_value = self.config.initial_capital
            rebalance_count = 0

            # Build returns DataFrame for momentum trigger
            returns_df = self._build_returns_dataframe()

            for trading_day in all_trading_days:
                try:
                    # Update portfolio value based on price changes
                    current_value = self._calculate_portfolio_value(
                        weights=self._current_weights,
                        date=trading_day,
                    )
                    self._portfolio_value = current_value

                    # Update peak value for drawdown calculation
                    if current_value > self._peak_value:
                        self._peak_value = current_value

                    # Build portfolio state for scheduler
                    target_weights = self._run_simple_pipeline(universe, trading_day)
                    portfolio_state = PortfolioState(
                        current_weights=self._current_weights.copy(),
                        target_weights=target_weights,
                        portfolio_value=current_value,
                        last_rebalance_date=self._rebalance_scheduler._last_rebalance_date,
                        peak_value=self._peak_value,
                    )

                    # Build market data for scheduler
                    market_data = self._build_market_data(
                        date=trading_day,
                        returns_df=returns_df,
                    )

                    # Check if should rebalance
                    decision = self._rebalance_scheduler.should_rebalance(
                        date=trading_day,
                        portfolio_state=portfolio_state,
                        market_data=market_data,
                    )

                    if decision.should_rebalance:
                        # Execute rebalance
                        new_weights = target_weights

                        # Calculate turnover and costs
                        turnover = self._calculate_turnover(self._current_weights, new_weights)
                        transaction_cost = self._calculate_transaction_cost(turnover)

                        # Apply transaction cost
                        self._portfolio_value -= transaction_cost

                        # Record rebalance
                        record = RebalanceRecord(
                            date=trading_day,
                            weights_before=self._current_weights.copy(),
                            weights_after=new_weights.copy(),
                            turnover=turnover,
                            transaction_cost=transaction_cost,
                            portfolio_value=self._portfolio_value,
                            metadata={
                                "trigger_type": decision.primary_trigger.value if decision.primary_trigger else "unknown",
                                "reason": decision.reason,
                                "severity": decision.combined_severity,
                            },
                        )
                        result.rebalance_records.append(record)

                        # Update weights
                        self._current_weights = new_weights.copy()
                        rebalance_count += 1

                        self._logger.debug(
                            f"Event rebalance {trading_day.date()}: "
                            f"trigger={decision.primary_trigger}, "
                            f"turnover={turnover:.4f}"
                        )

                    # Record portfolio value
                    portfolio_values.append((trading_day, self._portfolio_value))

                    # Calculate daily return
                    if prev_value > 0:
                        daily_return = (self._portfolio_value / prev_value) - 1
                        daily_returns.append((trading_day, daily_return))

                    prev_value = self._portfolio_value

                except (ValueError, ZeroDivisionError, KeyError) as e:
                    self._logger.warning(f"Day processing failed at {trading_day}: {e}")
                    result.warnings.append(f"Day processing failed at {trading_day}: {e}")
                    continue

            # Step 7: Build result time series
            if portfolio_values:
                dates, values = zip(*portfolio_values)
                result.portfolio_values = pd.Series(values, index=pd.DatetimeIndex(dates))

            if daily_returns:
                dates, rets = zip(*daily_returns)
                result.returns = pd.Series(rets, index=pd.DatetimeIndex(dates))

            # Step 8: Calculate final metrics
            result.metrics = self._calculate_metrics(result)

            # Add event-driven specific stats
            if self._rebalance_scheduler:
                scheduler_stats = self._rebalance_scheduler.get_stats()
                result.metrics["scheduled_rebalances"] = scheduler_stats["scheduled_rebalances"]
                result.metrics["triggered_rebalances"] = scheduler_stats["triggered_rebalances"]
                result.metrics["trigger_breakdown"] = scheduler_stats["trigger_counts"]

            self._logger.info(
                "Event-driven backtest completed",
                total_return=f"{result.total_return:.2%}",
                sharpe_ratio=f"{result.sharpe_ratio:.2f}",
                max_drawdown=f"{result.max_drawdown:.2%}",
                n_rebalances=len(result.rebalance_records),
                scheduled=scheduler_stats.get("scheduled_rebalances", 0) if self._rebalance_scheduler else 0,
                triggered=scheduler_stats.get("triggered_rebalances", 0) if self._rebalance_scheduler else 0,
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self._logger.exception(f"Event-driven backtest failed: {e}")
            result.errors.append(f"Event-driven backtest failed: {e}")
        except Exception as e:
            self._logger.exception(f"Unexpected event-driven backtest error: {e}")
            result.errors.append(f"Unexpected event-driven backtest error: {e}")
            raise

        return result

    def _get_all_trading_days(self) -> list[datetime]:
        """Get all trading days in the backtest period."""
        if self._price_matrix is None:
            return []

        # Get trading days from price matrix
        if "timestamp" in self._price_matrix.columns:
            ts_col = self._price_matrix["timestamp"].to_list()
        elif self._price_matrix_pd is not None:
            ts_col = self._price_matrix_pd.index.tolist()
        else:
            return []

        trading_days = [
            d if isinstance(d, datetime) else pd.Timestamp(d).to_pydatetime()
            for d in ts_col
        ]

        # Normalize timezone
        def to_naive(dt: datetime) -> datetime:
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt

        trading_days = [to_naive(d) for d in trading_days]
        start_naive = to_naive(self.config.start_date)
        end_naive = to_naive(self.config.end_date)

        # Filter to backtest range
        return [d for d in trading_days if start_naive <= d <= end_naive]

    def _fetch_vix_data(self) -> None:
        """Fetch VIX data for event triggers."""
        try:
            from src.data.adapters import StockAdapter

            adapter = StockAdapter()
            start = self.config.data_start_date
            end = self.config.end_date

            ohlcv = adapter.fetch_ohlcv("^VIX", start, end)

            if ohlcv and hasattr(ohlcv, "data"):
                if isinstance(ohlcv.data, pl.DataFrame):
                    df = ohlcv.data.to_pandas()
                else:
                    df = ohlcv.data

                if "close" in df.columns:
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                    self._vix_data = df["close"]
                    self._logger.debug(f"Fetched VIX data: {len(self._vix_data)} rows")

        except (ValueError, KeyError, TypeError, IOError) as e:
            self._logger.warning(f"Failed to fetch VIX data: {e}")
            self._vix_data = None

    def _build_returns_dataframe(self) -> pd.DataFrame | None:
        """Build returns DataFrame from price data."""
        if self._price_matrix_pd is None:
            return None

        try:
            returns = self._price_matrix_pd.pct_change().dropna()
            return returns
        except Exception as e:
            self._logger.warning(f"Failed to build returns DataFrame: {e}")
            return None

    def _build_market_data(
        self,
        date: datetime,
        returns_df: pd.DataFrame | None,
    ) -> MarketData:
        """Build MarketData object for scheduler."""
        vix = None
        vix_change = None

        if self._vix_data is not None:
            try:
                # Find VIX value at or before date
                valid_dates = self._vix_data.index[self._vix_data.index <= date]
                if len(valid_dates) > 0:
                    vix = float(self._vix_data.loc[valid_dates[-1]])

                    # Calculate VIX change
                    if len(valid_dates) > 1:
                        prev_vix = float(self._vix_data.loc[valid_dates[-2]])
                        if prev_vix > 0:
                            vix_change = (vix - prev_vix) / prev_vix
            except Exception:
                pass

        # Get returns up to date
        filtered_returns = None
        if returns_df is not None:
            try:
                filtered_returns = returns_df[returns_df.index <= date].tail(30)
            except Exception:
                pass

        return MarketData(
            date=date,
            vix=vix,
            vix_change=vix_change,
            returns=filtered_returns,
        )

    def _fetch_all_data(self, universe: list[str]) -> None:
        """
        Fetch all historical data for the backtest period using parallel execution.

        Data is fetched from data_start_date (including training period)
        through end_date. Uses ThreadPoolExecutor for I/O-bound parallelism.
        Data is stored as Polars DataFrames for better performance.

        Args:
            universe: List of asset symbols
        """
        from src.data.adapters import CryptoAdapter, DataFrequency, StockAdapter
        from src.data.cache import DataCache

        # Calculate date range
        start = self.config.data_start_date
        end = self.config.end_date

        self._logger.info(
            "Fetching data (parallel, Polars)",
            data_start=start.isoformat(),
            data_end=end.isoformat(),
            symbols=len(universe),
            workers=self._n_workers,
        )

        # Initialize cache
        cache = DataCache(max_age_days=7)

        def fetch_single_symbol(symbol: str) -> tuple[str, pl.DataFrame | None]:
            """Fetch data for a single symbol, returning Polars DataFrame."""
            try:
                # Check cache first
                cached = cache.get(symbol, DataFrequency.DAILY, start, end)
                if cached is not None:
                    # Convert to Polars if needed
                    if isinstance(cached.data, pl.DataFrame):
                        return symbol, cached.data
                    elif hasattr(cached.data, "to_pandas"):
                        return symbol, pl.from_pandas(cached.data.to_pandas().reset_index())
                    else:
                        return symbol, pl.from_pandas(cached.data.reset_index())

                # Determine adapter type
                is_crypto = "/" in symbol or symbol.upper().endswith(("USD", "USDT"))

                if is_crypto:
                    adapter = CryptoAdapter()
                    fetch_symbol = symbol if "/" in symbol else f"{symbol[:-3]}/{symbol[-3:]}"
                    ohlcv = adapter.fetch_ohlcv(fetch_symbol, start, end)
                else:
                    adapter = StockAdapter()
                    ohlcv = adapter.fetch_ohlcv(symbol, start, end)

                # Get data as Polars DataFrame
                if isinstance(ohlcv.data, pl.DataFrame):
                    df = ohlcv.data
                elif hasattr(ohlcv.data, "to_pandas"):
                    df = pl.from_pandas(ohlcv.data.to_pandas().reset_index())
                else:
                    df = pl.from_pandas(ohlcv.data.reset_index())

                if len(df) > 0:
                    # Cache for future use
                    cache.put(ohlcv, start, end)
                    return symbol, df

                return symbol, None

            except Exception as e:
                logger.debug(f"Failed to fetch {symbol}: {e}")
                return symbol, None

        # Parallel fetch using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {executor.submit(fetch_single_symbol, sym): sym for sym in universe}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, df = future.result()
                    if df is not None:
                        self._all_data[sym] = df
                        self._logger.debug(f"Fetched {sym}: {len(df)} rows")
                except Exception as e:
                    self._logger.warning(f"Failed to fetch {symbol}: {e}")

        self._logger.info(f"Fetched data for {len(self._all_data)}/{len(universe)} symbols")

    def _build_price_matrix(self) -> None:
        """Build a price matrix from all fetched data using Polars.

        Creates a Polars DataFrame with timestamp as a column and each symbol
        as additional columns containing close prices. Uses Polars lazy
        evaluation for optimal performance.
        """
        if not self._all_data:
            return

        # Build price matrix using Polars joins (much faster than Pandas)
        price_dfs = []
        for symbol, df in self._all_data.items():
            if "close" not in df.columns:
                continue

            # Extract timestamp and close, rename close to symbol name
            if "timestamp" in df.columns:
                price_df = df.select([
                    pl.col("timestamp"),
                    pl.col("close").alias(symbol)
                ])
            elif "index" in df.columns:
                price_df = df.select([
                    pl.col("index").alias("timestamp"),
                    pl.col("close").alias(symbol)
                ])
            else:
                # Assume first column is timestamp
                first_col = df.columns[0]
                price_df = df.select([
                    pl.col(first_col).alias("timestamp"),
                    pl.col("close").alias(symbol)
                ])

            price_dfs.append(price_df)

        if not price_dfs:
            return

        # Join all price DataFrames on timestamp
        # Note: Polars outer join creates 'timestamp_right' column that must be removed
        # to avoid DuplicateError on subsequent joins (BT-FIX1)
        result = price_dfs[0]
        for price_df in price_dfs[1:]:
            result = result.join(price_df, on="timestamp", how="outer")
            # Remove timestamp_right column created by outer join
            if "timestamp_right" in result.columns:
                result = result.drop("timestamp_right")

        # Sort by timestamp and forward fill missing values
        self._price_matrix = (
            result
            .sort("timestamp")
            .fill_null(strategy="forward")
        )

        # Cache Pandas version for compatibility
        self._price_matrix_pd = self._price_matrix.to_pandas()
        if "timestamp" in self._price_matrix_pd.columns:
            self._price_matrix_pd = self._price_matrix_pd.set_index("timestamp")

    def _get_rebalance_dates(self) -> list[datetime]:
        """
        Generate list of rebalance dates based on frequency.

        Returns:
            List of datetime objects for each rebalance
        """
        if self._price_matrix is None:
            return []

        # Get trading days from Polars price matrix
        if "timestamp" in self._price_matrix.columns:
            ts_col = self._price_matrix["timestamp"].to_list()
        else:
            # Fallback to Pandas cached version
            if self._price_matrix_pd is not None:
                ts_col = self._price_matrix_pd.index.tolist()
            else:
                return []

        trading_days = [
            d if isinstance(d, datetime) else pd.Timestamp(d).to_pydatetime()
            for d in ts_col
        ]

        # Normalize timezone - remove timezone info for comparison
        def to_naive(dt: datetime) -> datetime:
            """Convert datetime to timezone-naive for comparison."""
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt

        trading_days = [to_naive(d) for d in trading_days]
        start_naive = to_naive(self.config.start_date)
        end_naive = to_naive(self.config.end_date)

        # Filter to backtest range
        trading_days = [
            d for d in trading_days
            if start_naive <= d <= end_naive
        ]

        if not trading_days:
            return []

        frequency = self.config.rebalance_frequency.lower()

        if frequency == "daily":
            return trading_days

        elif frequency == "weekly":
            # Last trading day of each week
            weekly_dates = []
            current_week = None
            for d in trading_days:
                week = d.isocalendar()[1]
                if current_week != week:
                    if weekly_dates:
                        # Keep the last date of previous week
                        pass
                    current_week = week
                weekly_dates.append(d)

            # Take last day of each week
            result = []
            for i, d in enumerate(weekly_dates):
                if i == len(weekly_dates) - 1:
                    result.append(d)
                elif weekly_dates[i + 1].isocalendar()[1] != d.isocalendar()[1]:
                    result.append(d)
            return result

        elif frequency == "monthly":
            # Last trading day of each month
            monthly_dates = []
            current_month = None
            for d in trading_days:
                if current_month != (d.year, d.month):
                    if current_month is not None and monthly_dates:
                        pass  # Keep last date of previous month
                    current_month = (d.year, d.month)
                monthly_dates.append(d)

            # Take last day of each month
            result = []
            for i, d in enumerate(monthly_dates):
                if i == len(monthly_dates) - 1:
                    result.append(d)
                elif (monthly_dates[i + 1].year, monthly_dates[i + 1].month) != (d.year, d.month):
                    result.append(d)
            return result

        elif frequency == "quarterly":
            # Last trading day of each quarter
            quarterly_dates = []
            current_quarter = None
            for d in trading_days:
                quarter = (d.year, (d.month - 1) // 3)
                if current_quarter != quarter:
                    current_quarter = quarter
                quarterly_dates.append(d)

            # Take last day of each quarter
            result = []
            for i, d in enumerate(quarterly_dates):
                if i == len(quarterly_dates) - 1:
                    result.append(d)
                else:
                    next_q = (quarterly_dates[i + 1].year, (quarterly_dates[i + 1].month - 1) // 3)
                    curr_q = (d.year, (d.month - 1) // 3)
                    if next_q != curr_q:
                        result.append(d)
            return result

        else:
            self._logger.warning(f"Unknown frequency '{frequency}', using monthly")
            return self._get_rebalance_dates_monthly(trading_days)

    def _get_rebalance_dates_monthly(self, trading_days: list[datetime]) -> list[datetime]:
        """Helper for monthly rebalance dates."""
        result = []
        current_month = None
        for d in trading_days:
            if current_month != (d.year, d.month):
                if result:
                    pass
                current_month = (d.year, d.month)
            result.append(d)

        final = []
        for i, d in enumerate(result):
            if i == len(result) - 1:
                final.append(d)
            elif (result[i + 1].year, result[i + 1].month) != (d.year, d.month):
                final.append(d)
        return final

    def _initialize_portfolio(self, universe: list[str]) -> None:
        """Initialize portfolio with equal weights or cash."""
        # Start with 100% cash
        self._current_weights = {self.config.cash_symbol: 1.0}
        self._portfolio_value = self.config.initial_capital

    def _run_pipeline_with_cutoff(
        self,
        universe: list[str],
        cutoff_date: datetime,
    ) -> dict[str, float]:
        """
        Run the investment pipeline using ONLY data up to cutoff_date.

        This is the critical method that ensures no data leakage.
        All decisions are made using only historical data.

        Two modes:
        1. Full Pipeline mode: Uses the complete Pipeline orchestrator
        2. Simple mode: Uses built-in momentum scoring (fallback)

        Args:
            universe: List of asset symbols
            cutoff_date: Date cutoff (exclusive of future data)

        Returns:
            Dictionary of symbol -> weight
        """
        # Try full Pipeline integration first
        try:
            return self._run_full_pipeline(universe, cutoff_date)
        except Exception as e:
            self._logger.warning(
                f"Full pipeline failed at {cutoff_date}, falling back to simple scoring: {e}"
            )
            return self._run_simple_pipeline(universe, cutoff_date)

    def _run_full_pipeline(
        self,
        universe: list[str],
        cutoff_date: datetime,
    ) -> dict[str, float]:
        """
        Run the full Pipeline orchestrator with data cutoff.

        The Pipeline will:
        1. Use data only up to cutoff_date (data_cutoff_date parameter)
        2. Generate signals, evaluate strategies, compute weights
        3. Return optimized portfolio weights

        Args:
            universe: List of asset symbols
            cutoff_date: Date cutoff for data filtering

        Returns:
            Dictionary of symbol -> weight
        """
        from src.orchestrator.pipeline import Pipeline, PipelineConfig, PipelineStatus

        # Create pipeline with appropriate config
        pipeline_config = PipelineConfig(
            skip_data_fetch=True,  # We already have data
        )
        pipeline = Pipeline(settings=self._settings, config=pipeline_config)

        # Inject pre-fetched data into pipeline
        # This avoids duplicate API calls
        self._inject_data_into_pipeline(pipeline, universe, cutoff_date)

        # Run pipeline with cutoff date
        result = pipeline.run(
            universe=universe,
            previous_weights=self._current_weights,
            as_of_date=cutoff_date,
            data_cutoff_date=cutoff_date,
        )

        # Check result status
        if result.status == PipelineStatus.FAILED:
            raise RuntimeError(f"Pipeline failed: {result.errors}")

        # Extract weights from result
        if result.weights:
            return result.weights
        else:
            self._logger.warning("Pipeline returned no weights, using cash")
            return {self.config.cash_symbol: 1.0}

    def _inject_data_into_pipeline(
        self,
        pipeline: "Pipeline",
        universe: list[str],
        cutoff_date: datetime,
    ) -> None:
        """
        Inject pre-fetched Polars data into pipeline to avoid duplicate API calls.

        The data is filtered to cutoff_date before injection using Polars
        lazy evaluation for optimal performance.

        Args:
            pipeline: Pipeline instance
            universe: List of symbols
            cutoff_date: Date cutoff
        """
        for symbol in universe:
            if symbol not in self._all_data:
                continue

            df = self._all_data[symbol]

            # Get timestamp column name
            ts_col = "timestamp" if "timestamp" in df.columns else "index" if "index" in df.columns else df.columns[0]

            # Filter to cutoff date using Polars lazy evaluation
            try:
                filtered = (
                    df.lazy()
                    .filter(pl.col(ts_col) <= cutoff_date)
                    .collect()
                )

                if len(filtered) < self.config.train_period_days // 2:
                    continue

                # Rename timestamp column if needed
                if ts_col != "timestamp":
                    filtered = filtered.rename({ts_col: "timestamp"})

                # Inject directly (already Polars)
                pipeline._raw_data[symbol] = filtered

            except Exception as e:
                self._logger.debug(f"Failed to inject {symbol} data: {e}")

    def _run_simple_pipeline(
        self,
        universe: list[str],
        cutoff_date: datetime,
    ) -> dict[str, float]:
        """
        Fallback simple pipeline using momentum scoring with optimization.

        Uses Polars lazy evaluation and optional parallel processing.
        Used when full Pipeline is not available or fails.

        Args:
            universe: List of asset symbols
            cutoff_date: Date cutoff

        Returns:
            Dictionary of symbol -> weight
        """
        # Filter data to cutoff date using Polars
        filtered_data: dict[str, pl.DataFrame] = {}
        for symbol, df in self._all_data.items():
            if symbol not in universe:
                continue

            # Get timestamp column name
            ts_col = "timestamp" if "timestamp" in df.columns else "index" if "index" in df.columns else df.columns[0]

            # Strictly filter to data available at cutoff using Polars lazy evaluation
            filtered = (
                df.lazy()
                .filter(pl.col(ts_col) <= cutoff_date)
                .collect()
            )

            if len(filtered) >= self.config.train_period_days // 2:
                filtered_data[symbol] = filtered

        if not filtered_data:
            self._logger.warning(f"No valid data at cutoff {cutoff_date}")
            return {self.config.cash_symbol: 1.0}

        # Choose computation method based on data size
        if len(filtered_data) > 20:
            # Use parallel computation for larger universes
            scores = self._compute_scores_parallel(filtered_data, cutoff_date)
        else:
            # Use vectorized sequential for smaller universes
            scores = self._compute_simple_scores(filtered_data)

        if not scores:
            return {self.config.cash_symbol: 1.0}

        # Convert scores to weights using softmax
        weights = self._scores_to_weights(scores)

        return weights

    def _compute_simple_scores(
        self,
        data: dict[str, pl.DataFrame],
    ) -> dict[str, float]:
        """
        Compute simple momentum scores using Polars and NumPy vectorization.

        Uses Polars for data extraction and NumPy for numerical computation.

        Args:
            data: Dictionary of symbol -> Polars DataFrame (filtered to cutoff)

        Returns:
            Dictionary of symbol -> score
        """
        scores = {}
        lookback = min(60, self.config.train_period_days // 4)

        # Vectorized batch computation
        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            try:
                # Extract close prices using Polars (faster than pandas)
                close = df["close"].to_numpy()

                # Vectorized momentum calculation
                momentum_arr = vectorized_momentum(close, lookback)

                # Get last valid momentum value
                valid_momentum = momentum_arr[~np.isnan(momentum_arr)]
                if len(valid_momentum) == 0:
                    continue

                # Vectorized returns for volatility
                returns = np.diff(close) / close[:-1]

                if len(returns) < lookback:
                    continue

                # Annualized momentum and volatility
                recent_returns = returns[-lookback:]
                momentum = np.mean(recent_returns) * np.sqrt(252)
                vol = np.std(recent_returns, ddof=1) * np.sqrt(252)

                if vol > 0:
                    sharpe = momentum / vol
                else:
                    sharpe = 0.0

                scores[symbol] = float(sharpe)

            except Exception as e:
                self._logger.debug(f"Score computation failed for {symbol}: {e}")
                continue

        return scores

    def _compute_scores_parallel(
        self,
        data: dict[str, pl.DataFrame],
        cutoff_date: datetime,
    ) -> dict[str, float]:
        """
        Compute scores using parallel processing with caching.

        Uses ProcessPoolExecutor for CPU-bound score calculation.
        Data is extracted from Polars DataFrames before parallel processing.

        Args:
            data: Dictionary of symbol -> Polars DataFrame
            cutoff_date: Date cutoff for caching

        Returns:
            Dictionary of symbol -> score
        """
        lookback = min(60, self.config.train_period_days // 4)

        def compute_single_score(args: tuple) -> tuple[str, float | None]:
            """Compute score for a single symbol."""
            symbol, close_values = args
            if len(close_values) < lookback:
                return symbol, None

            try:
                returns = np.diff(close_values) / close_values[:-1]
                if len(returns) < lookback:
                    return symbol, None

                recent_returns = returns[-lookback:]
                momentum = np.mean(recent_returns) * np.sqrt(252)
                vol = np.std(recent_returns, ddof=1) * np.sqrt(252)

                sharpe = momentum / vol if vol > 0 else 0.0
                return symbol, float(sharpe)

            except Exception:
                return symbol, None

        # Prepare data for parallel processing (extract NumPy arrays from Polars)
        items = [(sym, df["close"].to_numpy()) for sym, df in data.items()]

        scores = {}

        # Use ProcessPoolExecutor for CPU-bound work
        # Only use parallel if enough symbols
        if len(items) > 10 and self._n_workers > 1:
            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                results = list(executor.map(compute_single_score, items))
                for symbol, score in results:
                    if score is not None:
                        scores[symbol] = score
        else:
            # Sequential for small datasets
            for item in items:
                symbol, score = compute_single_score(item)
                if score is not None:
                    scores[symbol] = score

        return scores

    def _scores_to_weights(
        self,
        scores: dict[str, float],
        temperature: float = 2.0,
    ) -> dict[str, float]:
        """
        Convert scores to portfolio weights using softmax.

        Args:
            scores: Dictionary of symbol -> score
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Dictionary of symbol -> weight
        """
        if not scores:
            return {self.config.cash_symbol: 1.0}

        # Filter to positive scores only (long-only constraint)
        positive_scores = {k: v for k, v in scores.items() if v > 0}

        if not positive_scores:
            # All negative scores -> go to cash
            return {self.config.cash_symbol: 1.0}

        # Softmax
        symbols = list(positive_scores.keys())
        score_values = np.array([positive_scores[s] for s in symbols])

        # Normalize for numerical stability
        score_values = score_values - score_values.max()
        exp_scores = np.exp(score_values / temperature)
        weights = exp_scores / exp_scores.sum()

        # Apply max weight constraint
        max_weight = self.settings.asset_allocation.w_asset_max
        weights = np.minimum(weights, max_weight)

        # Renormalize
        weights = weights / weights.sum()

        result = {symbols[i]: float(weights[i]) for i in range(len(symbols))}

        # Any remaining weight goes to cash
        total = sum(result.values())
        if total < 1.0:
            result[self.config.cash_symbol] = 1.0 - total

        return result

    def _calculate_turnover(
        self,
        weights_before: dict[str, float],
        weights_after: dict[str, float],
    ) -> float:
        """Calculate portfolio turnover."""
        all_symbols = set(weights_before.keys()) | set(weights_after.keys())
        turnover = 0.0
        for symbol in all_symbols:
            w_before = weights_before.get(symbol, 0.0)
            w_after = weights_after.get(symbol, 0.0)
            turnover += abs(w_after - w_before)
        return turnover / 2  # One-way turnover

    def _calculate_transaction_cost(self, turnover: float) -> float:
        """Calculate transaction cost for given turnover."""
        cost_rate = self.config.total_cost_bps / 10000
        return self._portfolio_value * turnover * cost_rate

    def _calculate_portfolio_value(
        self,
        weights: dict[str, float],
        date: datetime,
    ) -> float:
        """
        Calculate portfolio value at given date.

        Uses Polars for efficient date filtering when available,
        falls back to Pandas cached version for compatibility.

        Args:
            weights: Current portfolio weights
            date: Date for valuation

        Returns:
            Portfolio value
        """
        if self._price_matrix is None:
            return self._portfolio_value

        # Use Pandas cached version for index operations (faster for this use case)
        if self._price_matrix_pd is not None:
            valid_dates = self._price_matrix_pd.index[self._price_matrix_pd.index <= date]
            if len(valid_dates) == 0:
                return self._portfolio_value

            # closest_date = valid_dates[-1]  # Not currently used
            price_cols = set(self._price_matrix_pd.columns)
        else:
            # Fallback to Polars
            price_cols = set(self._price_matrix.columns) - {"timestamp"}

        value = 0.0
        for symbol, weight in weights.items():
            if symbol == self.config.cash_symbol:
                value += self._portfolio_value * weight
            elif symbol in price_cols:
                # For simplicity, assume weights represent value allocation
                value += self._portfolio_value * weight

        return value if value > 0 else self._portfolio_value

    def _calculate_metrics(self, result: BacktestResult) -> dict[str, float]:
        """Calculate final performance metrics."""
        metrics = {
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_turnover": result.total_turnover,
            "total_costs": result.total_transaction_costs,
            "n_rebalances": len(result.rebalance_records),
        }

        # Calmar ratio
        if result.max_drawdown < 0:
            metrics["calmar_ratio"] = result.annualized_return / abs(result.max_drawdown)
        else:
            metrics["calmar_ratio"] = 0.0

        # Win rate
        if len(result.returns) > 0:
            metrics["win_rate"] = (result.returns > 0).mean()
        else:
            metrics["win_rate"] = 0.0

        return metrics

    def _ensure_precomputed(
        self,
        prices: pd.DataFrame,
        signal_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure signals are precomputed for fast mode.

        If precompute_signals is enabled and cache is not valid,
        precompute all signals for the entire price history.

        Args:
            prices: Price DataFrame
            signal_config: Signal configuration (optional)
        """
        if not self.config.precompute_signals:
            return

        try:
            from .signal_precompute import SignalPrecomputer

            precomputer = SignalPrecomputer(self.config.cache_dir)

            if not precomputer.is_cache_valid(prices):
                self._logger.info("Precomputing signals for fast mode...")
                precomputer.precompute_all(prices, signal_config)
                self._logger.info("Signal precomputation completed")

        except ImportError:
            self._logger.debug("SignalPrecomputer not available, skipping precomputation")
        except Exception as e:
            self._logger.warning(f"Signal precomputation failed: {e}")
