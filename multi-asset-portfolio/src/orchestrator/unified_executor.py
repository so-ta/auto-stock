"""
UnifiedExecutor - Pipeline/Backtest統一実行エンジン (v2.1 高速化版)

PipelineとBacktestEngineを統一し、Pipelineの最適化ロジックを
バックテストで検証可能にする。

v2.1 高速化:
- skip_data_fetch=True でAPI呼び出しをスキップ
- 価格データを事前にPipelineに注入
- データ取得は1回のみ（バックテスト開始時）

Usage:
    from src.orchestrator.unified_executor import UnifiedExecutor

    executor = UnifiedExecutor()

    # 単一日付モード（本番用）
    result = executor.run_single(
        universe=["AAPL", "MSFT", ...],
        as_of_date=datetime.now(),
    )

    # バックテストモード（検証用）- 高速化版
    result = executor.run_backtest(
        universe=["AAPL", "MSFT", ...],
        start_date="2020-01-01",
        end_date="2025-01-01",
        frequency="monthly",
    )
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd
import polars as pl

from src.backtest.base import UnifiedBacktestConfig, UnifiedBacktestResult
from src.backtest.factory import create_engine
from src.config.settings import Settings
from src.orchestrator.pipeline import Pipeline, PipelineConfig, PipelineResult

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageConfig

# Lightweight + skip_data_fetch mode for maximum backtest performance
# v2.2 (task_040_3): Added incremental covariance for 10-20% speedup
_BACKTEST_PIPELINE_CONFIG = PipelineConfig(
    lightweight_mode=True,
    skip_diagnostics=True,
    skip_audit_log=True,
    skip_data_fetch=True,  # Critical: Skip API calls, use injected data
    use_incremental_covariance=True,  # v2.2: Enable incremental update
    covariance_halflife=60,  # v2.2: 60-day halflife
)

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint Support (task_040_4)
# =============================================================================

@dataclass
class BacktestCheckpoint:
    """Checkpoint data for resuming interrupted backtests (task_040_4).

    Enables warm-start capability for long-running backtests.
    All state needed to resume from a specific rebalance point is captured.

    Attributes:
        rebalance_index: Current rebalance iteration index
        current_date: Date of the checkpoint
        equity_curve: List of portfolio values up to checkpoint
        weights_history: List of weight dictionaries for each rebalance
        random_state: Numpy random state for reproducibility
        config_hash: Hash of backtest config for validation
        metadata: Additional metadata (version, timestamps, etc.)
    """
    rebalance_index: int
    current_date: datetime
    equity_curve: List[float]
    weights_history: List[Dict[str, float]]
    random_state: Optional[Any] = None
    config_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
            }


class UnifiedExecutor:
    """
    Pipeline と BacktestEngine を統一する実行エンジン

    Pipelineの計算ロジック（シグナル生成、配分最適化、レジーム適応等）を
    BacktestEngineのweights_funcとして使用することで、
    最適化の効果をバックテストで検証可能にする。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        config_path: Optional[str] = None,
    ):
        """
        初期化

        Args:
            settings: Settings オブジェクト（省略時は自動読み込み）
            config_path: 設定ファイルパス（省略時はデフォルト）
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self._pipeline: Optional[Pipeline] = None
        self._backtest_pipeline: Optional[Pipeline] = None  # Lightweight mode for backtest
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._storage_config: Optional["StorageConfig"] = None  # S3/local storage config

        logger.info("UnifiedExecutor initialized")

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline インスタンス（遅延初期化）- 本番モード"""
        if self._pipeline is None:
            self._pipeline = Pipeline(settings=self.settings)
        return self._pipeline

    @property
    def backtest_pipeline(self) -> Pipeline:
        """Pipeline インスタンス（遅延初期化）- バックテスト高速モード

        High-performance mode for backtesting:
        - skip_data_fetch=True: No API calls (use pre-injected data)
        - lightweight_mode=True: Skip diagnostics and audit logs
        - Data is injected once, filtered by date for each rebalance
        """
        if self._backtest_pipeline is None:
            self._backtest_pipeline = Pipeline(
                settings=self.settings,
                config=_BACKTEST_PIPELINE_CONFIG,
            )
        return self._backtest_pipeline

    @property
    def storage_config(self) -> Optional["StorageConfig"]:
        """現在のストレージ設定を取得。

        run_backtest() 呼び出し時に設定されるか、Settings から取得。
        キャッシュクラス（CovarianceCache 等）への伝播に使用。
        """
        if self._storage_config is None and self.settings is not None:
            self._storage_config = self.settings.storage.to_storage_config()
        return self._storage_config

    def run_single(
        self,
        universe: List[str],
        as_of_date: Optional[datetime] = None,
        previous_weights: Optional[Dict[str, float]] = None,
    ) -> PipelineResult:
        """
        単一日付モード（本番用）

        Pipelineを1回実行し、その日のウェイトを返す。

        Args:
            universe: 資産リスト
            as_of_date: 評価日（省略時は現在）
            previous_weights: 前期のウェイト

        Returns:
            PipelineResult: ウェイトと診断情報
        """
        logger.info(f"Running single mode: {len(universe)} assets, as_of={as_of_date}")

        return self.pipeline.run(
            universe=universe,
            previous_weights=previous_weights,
            as_of_date=as_of_date,
            data_cutoff_date=as_of_date,
        )

    def run_backtest(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        frequency: str = "monthly",
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        max_weight: float = 0.20,
        risk_free_rate: float = 0.02,
        storage_config: Optional["StorageConfig"] = None,
    ) -> UnifiedBacktestResult:
        """
        バックテストモード（検証用）

        Pipelineのweights計算をweights_funcとして使用し、
        時系列バックテストを実行。

        Args:
            universe: 資産リスト
            prices: 価格データ Dict[symbol, DataFrame]
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            frequency: リバランス頻度 (daily/weekly/monthly)
            initial_capital: 初期資本
            transaction_cost_bps: 取引コスト (bps)
            slippage_bps: スリッページ (bps)
            max_weight: 最大ウェイト
            risk_free_rate: リスクフリーレート
            storage_config: ストレージ設定（S3/ローカル）。省略時はSettingsから取得。

        Returns:
            UnifiedBacktestResult: バックテスト結果
        """
        # Storage config: 明示的指定 or Settings から自動取得
        if storage_config is None and self.settings is not None:
            storage_config = self.settings.storage.to_storage_config()
        self._storage_config = storage_config

        logger.info(
            f"Running backtest mode (v2.1 optimized): {len(universe)} assets, "
            f"{start_date} to {end_date}, {frequency}"
        )
        if storage_config:
            logger.info(f"Storage backend: {storage_config.backend}")

        # 価格データをキャッシュ
        self._prices_cache = prices

        # 価格データをPolars DataFrameに変換してPipelineに事前注入
        # これにより毎リバランスでのAPI呼び出しをスキップ
        self._inject_price_data(universe, prices)
        logger.info("Price data pre-injected into Pipeline (skip_data_fetch=True)")

        # Pipelineベースのweights_func (v2.1: 高速化版)
        def pipeline_weights_func(
            universe: List[str],
            prices: Dict[str, pd.DataFrame],
            date: datetime,
            current_weights: Dict[str, float],
        ) -> Dict[str, float]:
            """
            Pipelineを呼び出してウェイトを計算（高速化版）

            v2.1 Optimization:
            - skip_data_fetch=True: API呼び出しスキップ
            - データは事前注入済み、date_cutoffで未来データを除外
            - 毎リバランスでデータ取得なし → 大幅な高速化
            """
            try:
                # 日付でフィルタしたデータを再注入
                self._inject_filtered_data(universe, prices, date)

                result = self.backtest_pipeline.run(
                    universe=universe,
                    previous_weights=current_weights,
                    as_of_date=date,
                    data_cutoff_date=date,
                )

                if result.weights:
                    # ウェイトの正規化
                    total = sum(result.weights.values())
                    if total > 0:
                        return {k: v / total for k, v in result.weights.items()}

                # Fallback: 等ウェイト
                logger.warning(f"Pipeline returned empty weights for {date}, using equal weight")
                n = len(universe)
                return {symbol: 1.0 / n for symbol in universe}

            except Exception as e:
                logger.error(f"Pipeline error at {date}: {e}")
                # Fallback: 等ウェイト
                n = len(universe)
                return {symbol: 1.0 / n for symbol in universe}

        # バックテスト設定
        config = UnifiedBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_frequency=frequency,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            max_weight=max_weight,
            min_weight=0.0,
            risk_free_rate=risk_free_rate,
        )

        # エンジン作成
        engine = create_engine(config=config)
        logger.info(f"Using engine: {engine.ENGINE_NAME}")

        # バックテスト実行（run_unified使用: 外部形式weights_funcを内部形式に変換）
        result = engine.run_unified(
            universe=universe,
            prices=prices,  # Dict形式をそのまま渡す（run_unified内で変換）
            config=config,
            weights_func=pipeline_weights_func,
        )

        logger.info(
            f"Backtest completed: Sharpe={result.sharpe_ratio:.3f}, "
            f"Return={result.annual_return*100:.2f}%"
        )

        return result

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._prices_cache.clear()
        if self._pipeline is not None:
            self._pipeline = None
        if self._backtest_pipeline is not None:
            self._backtest_pipeline = None
        logger.info("Cache cleared")

    def _inject_price_data(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
    ) -> None:
        """
        価格データをPipelineに事前注入（高速化用）

        Args:
            universe: 銘柄リスト
            prices: 価格データ Dict[symbol, DataFrame]
        """
        raw_data: Dict[str, pl.DataFrame] = {}

        for symbol in universe:
            if symbol not in prices:
                continue

            df = prices[symbol]
            if df.empty:
                continue

            # pandas → polars変換
            # Pipelineが期待する形式に変換
            if isinstance(df.index, pd.DatetimeIndex):
                df_reset = df.reset_index()
                df_reset.columns = ["timestamp"] + list(df.columns)
            else:
                df_reset = df.copy()

            try:
                pl_df = pl.from_pandas(df_reset)
                raw_data[symbol] = pl_df
            except Exception as e:
                logger.warning(f"Failed to convert {symbol} to Polars: {e}")
                continue

        # Pipelineの_raw_dataに直接注入
        self.backtest_pipeline._raw_data = raw_data
        logger.debug(f"Injected {len(raw_data)} symbols into Pipeline")

    def _inject_filtered_data(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        cutoff_date: datetime,
    ) -> None:
        """
        日付でフィルタした価格データをPipelineに注入

        Args:
            universe: 銘柄リスト
            prices: 価格データ
            cutoff_date: カットオフ日（この日以前のデータのみ使用）
        """
        raw_data: Dict[str, pl.DataFrame] = {}
        cutoff_ts = pd.Timestamp(cutoff_date)

        for symbol in universe:
            if symbol not in prices:
                continue

            df = prices[symbol]
            if df.empty:
                continue

            # 日付でフィルタ
            if isinstance(df.index, pd.DatetimeIndex):
                filtered = df[df.index <= cutoff_ts]
                if filtered.empty:
                    continue
                df_reset = filtered.reset_index()
                df_reset.columns = ["timestamp"] + list(filtered.columns)
            else:
                filtered = df[df["timestamp"] <= cutoff_ts] if "timestamp" in df.columns else df
                df_reset = filtered.copy()

            try:
                pl_df = pl.from_pandas(df_reset)
                raw_data[symbol] = pl_df
            except Exception:
                continue

        # Pipelineの_raw_dataに注入
        self.backtest_pipeline._raw_data = raw_data

    # =========================================================================
    # Checkpoint Methods (task_040_4)
    # =========================================================================

    def save_checkpoint(
        self,
        checkpoint: BacktestCheckpoint,
        path: Path,
    ) -> None:
        """Save checkpoint to file (task_040_4).

        Args:
            checkpoint: BacktestCheckpoint object to save
            path: Path to save the checkpoint file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Checkpoint saved: {path} ({file_size_mb:.2f} MB)")

    def load_checkpoint(self, path: Path) -> BacktestCheckpoint:
        """Load checkpoint from file (task_040_4).

        Args:
            path: Path to the checkpoint file

        Returns:
            BacktestCheckpoint object

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        if not isinstance(checkpoint, BacktestCheckpoint):
            raise ValueError(f"Invalid checkpoint format: {type(checkpoint)}")

        logger.info(
            f"Checkpoint loaded: rebalance_index={checkpoint.rebalance_index}, "
            f"date={checkpoint.current_date}"
        )
        return checkpoint

    def run_backtest_with_checkpoint(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        frequency: str = "monthly",
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        max_weight: float = 0.20,
        risk_free_rate: float = 0.02,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
        resume_from: Optional[Path] = None,
    ) -> UnifiedBacktestResult:
        """Run backtest with checkpoint support (task_040_4).

        Enables saving checkpoints at regular intervals and resuming
        from a previous checkpoint.

        Args:
            universe: Asset list
            prices: Price data Dict[symbol, DataFrame]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Rebalance frequency
            initial_capital: Initial capital
            transaction_cost_bps: Transaction cost in bps
            slippage_bps: Slippage in bps
            max_weight: Maximum weight per asset
            risk_free_rate: Risk-free rate
            checkpoint_interval: Save checkpoint every N rebalances (0=disabled)
            checkpoint_dir: Directory for checkpoint files
            resume_from: Path to checkpoint file to resume from

        Returns:
            UnifiedBacktestResult
        """
        import hashlib
        import numpy as np

        logger.info(
            f"Running backtest with checkpoint support: "
            f"interval={checkpoint_interval}, resume={resume_from}"
        )

        # Create config hash for validation
        config_str = f"{start_date}_{end_date}_{frequency}_{initial_capital}_{transaction_cost_bps}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints")
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        equity_curve: List[float] = [initial_capital]
        weights_history: List[Dict[str, float]] = []
        start_index = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            checkpoint = self.load_checkpoint(resume_from)

            # Validate config hash
            if checkpoint.config_hash and checkpoint.config_hash != config_hash:
                logger.warning(
                    f"Config hash mismatch: checkpoint={checkpoint.config_hash}, "
                    f"current={config_hash}. Results may differ."
                )

            # Restore state
            start_index = checkpoint.rebalance_index
            equity_curve = checkpoint.equity_curve.copy()
            weights_history = checkpoint.weights_history.copy()

            if checkpoint.random_state is not None:
                np.random.set_state(checkpoint.random_state)

            logger.info(f"Resuming from rebalance index {start_index}")

        # Price data injection
        self._prices_cache = prices
        self._inject_price_data(universe, prices)

        # Generate rebalance dates
        from src.backtest.base import UnifiedBacktestConfig

        config = UnifiedBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_frequency=frequency,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            max_weight=max_weight,
            min_weight=0.0,
            risk_free_rate=risk_free_rate,
        )

        # Get rebalance dates from price data
        sample_prices = next(iter(prices.values()))
        if isinstance(sample_prices.index, pd.DatetimeIndex):
            all_dates = sample_prices.index
        else:
            all_dates = pd.DatetimeIndex(sample_prices["timestamp"])

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        date_range = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]

        # Create rebalance mask
        from src.backtest.base import create_rebalance_mask
        rebalance_mask = create_rebalance_mask(date_range, frequency)
        rebalance_dates = date_range[rebalance_mask]

        logger.info(f"Total rebalance dates: {len(rebalance_dates)}, starting from index {start_index}")

        # Run backtest with checkpoint support
        current_weights: Dict[str, float] = {}
        if weights_history:
            current_weights = weights_history[-1]

        current_equity = equity_curve[-1]

        for i, date in enumerate(rebalance_dates):
            if i < start_index:
                continue

            # Calculate weights using Pipeline
            self._inject_filtered_data(universe, prices, date.to_pydatetime())

            try:
                result = self.backtest_pipeline.run(
                    universe=universe,
                    previous_weights=current_weights,
                    as_of_date=date.to_pydatetime(),
                    data_cutoff_date=date.to_pydatetime(),
                )

                if result.weights:
                    total = sum(result.weights.values())
                    if total > 0:
                        current_weights = {k: v / total for k, v in result.weights.items()}
                    else:
                        n = len(universe)
                        current_weights = {s: 1.0 / n for s in universe}
                else:
                    n = len(universe)
                    current_weights = {s: 1.0 / n for s in universe}
            except Exception as e:
                logger.warning(f"Pipeline error at {date}: {e}, using equal weight")
                n = len(universe)
                current_weights = {s: 1.0 / n for s in universe}

            weights_history.append(current_weights.copy())

            # Simple equity update (for checkpoint tracking)
            # Note: Actual returns are calculated by the backtest engine
            equity_curve.append(current_equity)

            # Save checkpoint at interval
            if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                checkpoint = BacktestCheckpoint(
                    rebalance_index=i + 1,
                    current_date=date.to_pydatetime(),
                    equity_curve=equity_curve.copy(),
                    weights_history=weights_history.copy(),
                    random_state=np.random.get_state(),
                    config_hash=config_hash,
                    metadata={
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0",
                        "frequency": frequency,
                        "universe_size": len(universe),
                    },
                )
                cp_path = checkpoint_dir / f"cp_{i+1:04d}.pkl"
                self.save_checkpoint(checkpoint, cp_path)

        # Run final backtest with collected weights
        # Use standard run_backtest with pre-computed weights
        def precomputed_weights_func(
            universe: List[str],
            prices: Dict[str, pd.DataFrame],
            date: datetime,
            current_weights: Dict[str, float],
        ) -> Dict[str, float]:
            """Return pre-computed weights for the given date."""
            # Find closest rebalance date
            date_ts = pd.Timestamp(date)
            for i, rd in enumerate(rebalance_dates):
                if rd >= date_ts and i < len(weights_history):
                    return weights_history[i]
            return weights_history[-1] if weights_history else {s: 1.0/len(universe) for s in universe}

        # Use backtest engine for final result calculation
        from src.backtest.factory import create_engine as _create_engine
        engine = _create_engine(config=config)

        # バックテスト実行（run_unified使用: 外部形式weights_funcを内部形式に変換）
        result = engine.run_unified(
            universe=universe,
            prices=prices,  # Dict形式をそのまま渡す（run_unified内で変換）
            config=config,
            weights_func=precomputed_weights_func,
        )

        logger.info(
            f"Backtest with checkpoint completed: Sharpe={result.sharpe_ratio:.3f}, "
            f"Return={result.annual_return*100:.2f}%"
        )

        return result
