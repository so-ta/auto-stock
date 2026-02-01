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
from src.config.settings import Settings, get_settings
from src.orchestrator.exceptions import BacktestError
from src.orchestrator.pipeline import Pipeline, PipelineConfig, PipelineResult

# Enhanced Alpha Scoring support (Phase A-D integration)
try:
    from src.meta.enhanced_alpha_scorer import (
        EnhancedAlphaScorer,
        EnhancedAlphaResult,
        OptimizableParams,
        create_enhanced_alpha_scorer,
    )
    from src.meta.alpha_param_optimizer import (
        AlphaParamOptimizer,
        AlphaOptimizationError,
        create_alpha_param_optimizer,
    )
    ENHANCED_ALPHA_AVAILABLE = True
except ImportError:
    ENHANCED_ALPHA_AVAILABLE = False

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageConfig
    from src.utils.progress_tracker import ProgressTracker
    from src.allocation.transaction_cost import TransactionCostSchedule
    from src.data.asset_master import AssetMaster

# Lightweight + skip_data_fetch mode for maximum backtest performance
# v2.2 (task_040_3): Added incremental covariance for 10-20% speedup
# v2.3: Precomputed signals for 40x speedup (now always enabled)
# v3.0: use_precomputed_signals removed - SignalPrecomputer is now required
_BACKTEST_PIPELINE_CONFIG = PipelineConfig(
    lightweight_mode=True,
    skip_diagnostics=True,
    skip_audit_log=True,
    skip_data_fetch=True,  # Critical: Skip API calls, use injected data
    use_incremental_covariance=True,  # v2.2: Enable incremental update
    covariance_halflife=60,  # v2.2: 60-day halflife
    # NOTE: use_precomputed_signals removed - precomputed signals are always used
)

logger = logging.getLogger(__name__)


# =============================================================================
# Top-N Asset Filter (v2.4 - Large Universe Optimization)
# =============================================================================

def compute_signal_scores(
    prices: Dict[str, pd.DataFrame],
    universe: List[str],
    date: datetime,
    lookback_days: int = 252,
) -> Dict[str, float]:
    """
    シグナルスコアを計算（モメンタム + Sharpe複合スコア）

    大規模ユニバースから Top-N 銘柄を選択するための高速スコアリング。
    NCO/HRP計算の前にフィルタリングし、計算量を削減。

    Args:
        prices: 価格データ Dict[symbol, DataFrame]
        universe: 評価対象銘柄リスト
        date: 評価日
        lookback_days: ルックバック期間（日数）

    Returns:
        Dict[symbol, score]: シグナルスコア辞書（高いほど良い）
    """
    import numpy as np

    scores = {}
    date_ts = pd.Timestamp(date)

    for symbol in universe:
        if symbol not in prices:
            continue

        df = prices[symbol]
        if df.empty:
            continue

        # 日付でフィルタ
        if isinstance(df.index, pd.DatetimeIndex):
            df_filtered = df[df.index <= date_ts].tail(lookback_days)
        elif "timestamp" in df.columns:
            df_filtered = df[df["timestamp"] <= date_ts].tail(lookback_days)
        else:
            continue

        if len(df_filtered) < 60:  # 最低60日のデータが必要
            continue

        # 終値を取得
        close_col = "close" if "close" in df_filtered.columns else "Close"
        if close_col not in df_filtered.columns:
            continue

        close = df_filtered[close_col].values

        # リターンを計算
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 20:
            continue

        # モメンタムスコア（252日リターン）
        total_return = (close[-1] / close[0]) - 1 if close[0] > 0 else 0

        # Sharpeスコア（日次リターンから推定）
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # 複合スコア = 0.5 * モメンタム + 0.5 * Sharpe
        # 両方を正規化するため、簡易的に合算
        score = 0.5 * total_return + 0.5 * sharpe

        scores[symbol] = score

    return scores


def filter_top_assets(
    universe: List[str],
    prices: Dict[str, pd.DataFrame],
    date: datetime,
    max_assets: int,
    lookback_days: int = 252,
) -> List[str]:
    """
    シグナルスコア上位N銘柄にフィルタリング

    大規模ユニバース（16,000銘柄等）で NCO/HRP 計算の前に
    銘柄数を絞り込み、計算量とメモリ使用量を削減。

    計算量:
    - フィルタなし: O(n²) for covariance + O(n²〜n³) for NCO
    - フィルタ後: O(max_assets²) for covariance

    Args:
        universe: 全銘柄リスト
        prices: 価格データ
        date: 評価日
        max_assets: 最大銘柄数
        lookback_days: スコア計算のルックバック期間

    Returns:
        フィルタ後の銘柄リスト（上位N銘柄）
    """
    if len(universe) <= max_assets:
        return universe

    # シグナルスコアを計算
    scores = compute_signal_scores(prices, universe, date, lookback_days)

    if not scores:
        # スコア計算に失敗した場合はランダムサンプリング
        logger.warning(f"Signal score computation failed, using random sampling for {max_assets} assets")
        import random
        return random.sample(universe, min(max_assets, len(universe)))

    # スコアで降順ソート
    sorted_symbols = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)

    # 上位N銘柄を選択
    top_n = sorted_symbols[:max_assets]

    logger.debug(
        f"Top-N filter: {len(universe)} -> {len(top_n)} assets "
        f"(top score: {scores.get(top_n[0], 0):.3f}, "
        f"cutoff: {scores.get(top_n[-1], 0):.3f})"
    )

    return top_n


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
            settings = get_settings()

        self.settings = settings
        self._pipeline: Optional[Pipeline] = None
        self._backtest_pipeline: Optional[Pipeline] = None  # Lightweight mode for backtest
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._storage_config: Optional["StorageConfig"] = None  # S3/local storage config

        # Signal precomputation support (v2.3: 40x speedup for 15-year backtests)
        self._signal_precomputer: Optional[Any] = None

        # Enhanced Alpha Scoring support (Phase A-D integration)
        self._enhanced_alpha_scorer: Optional[Any] = None  # EnhancedAlphaScorer
        self._alpha_param_optimizer: Optional[Any] = None  # AlphaParamOptimizer
        self._use_enhanced_alpha: bool = ENHANCED_ALPHA_AVAILABLE  # Default: enabled if available

        logger.info("UnifiedExecutor initialized")

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline インスタンス（遅延初期化）- 本番モード"""
        if self._pipeline is None:
            storage_backend = self._get_storage_backend()
            self._pipeline = Pipeline(
                settings=self.settings,
                storage_backend=storage_backend,
            )
        return self._pipeline

    @property
    def log_collector(self) -> "Optional[Any]":
        """Get the log collector from the backtest pipeline (if available).

        Returns the PipelineLogCollector that was used during the last
        pipeline execution, or None if not available.
        """
        if self._backtest_pipeline is not None:
            return self._backtest_pipeline.log_collector
        if self._pipeline is not None:
            return self._pipeline.log_collector
        return None

    @property
    def backtest_pipeline(self) -> Pipeline:
        """Pipeline インスタンス（遅延初期化）- バックテスト高速モード

        High-performance mode for backtesting:
        - skip_data_fetch=True: No API calls (use pre-injected data)
        - lightweight_mode=True: Skip diagnostics and audit logs
        - Data is injected once, filtered by date for each rebalance
        - use_precomputed_signals=True: Load signals from cache (v2.3)

        NOTE: signal_precomputer is passed at construction time if available.
        This ensures StrategyEvaluator receives the precomputer on first access,
        preventing cache miss issues caused by late injection (task_cache_fix).
        """
        if self._backtest_pipeline is None:
            storage_backend = self._get_storage_backend()
            self._backtest_pipeline = Pipeline(
                settings=self.settings,
                config=_BACKTEST_PIPELINE_CONFIG,
                storage_backend=storage_backend,
                signal_precomputer=self._signal_precomputer,
            )
        # Update precomputer if it was set after pipeline creation
        elif self._signal_precomputer is not None:
            self._backtest_pipeline._signal_precomputer = self._signal_precomputer
        return self._backtest_pipeline

    def _get_storage_backend(self) -> "Optional[Any]":
        """Get StorageBackend instance from storage_config."""
        storage_config = self.storage_config
        if storage_config is None:
            return None
        try:
            from src.utils.storage_backend import StorageBackend
            return StorageBackend(storage_config)
        except Exception as e:
            logger.warning(f"Failed to create StorageBackend: {e}")
            return None

    def _reset_backtest_pipeline(self) -> None:
        """Reset backtest pipeline for config changes."""
        self._backtest_pipeline = None

    @property
    def enhanced_alpha_scorer(self) -> Optional[Any]:
        """EnhancedAlphaScorer インスタンス（遅延初期化）

        Returns:
            EnhancedAlphaScorer or None if not available
        """
        if not ENHANCED_ALPHA_AVAILABLE or not self._use_enhanced_alpha:
            return None
        if self._enhanced_alpha_scorer is None:
            self._enhanced_alpha_scorer = create_enhanced_alpha_scorer(
                param_optimizer=self.alpha_param_optimizer
            )
        return self._enhanced_alpha_scorer

    @property
    def alpha_param_optimizer(self) -> Optional[Any]:
        """AlphaParamOptimizer インスタンス（遅延初期化）

        Returns:
            AlphaParamOptimizer or None if not available
        """
        if not ENHANCED_ALPHA_AVAILABLE or not self._use_enhanced_alpha:
            return None
        if self._alpha_param_optimizer is None:
            self._alpha_param_optimizer = create_alpha_param_optimizer(
                n_iterations=30,  # Reduced for backtest speed
                n_cv_folds=3,
            )
        return self._alpha_param_optimizer

    def compute_enhanced_alpha(
        self,
        returns_df: pd.DataFrame,
        signal_scores: Optional[Dict[str, float]] = None,
        covariance_matrix: Optional[Any] = None,
        use_optimization: bool = False,
    ) -> Optional[Dict[str, float]]:
        """EnhancedAlphaScorerでリスク調整済みアルファスコアを計算

        Args:
            returns_df: リターンデータ（列=銘柄、行=日付）
            signal_scores: シグナルスコア辞書（オプション）
            covariance_matrix: 共分散行列（オプション）
            use_optimization: パラメータ最適化を使用するか

        Returns:
            アルファスコア辞書、または利用不可の場合はNone
        """
        if self.enhanced_alpha_scorer is None:
            return None

        try:
            result = self.enhanced_alpha_scorer.compute(
                returns_df=returns_df,
                signal_scores=signal_scores,
                covariance_matrix=covariance_matrix,
                use_optimization=use_optimization,
            )
            return result.alpha_scores
        except Exception as e:
            logger.warning(f"Enhanced alpha computation failed: {e}")
            return None

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
        trading_universe: Optional[List[str]] = None,
        max_assets: Optional[int] = None,
        progress_tracker: Optional["ProgressTracker"] = None,
        cost_schedule: Optional["TransactionCostSchedule"] = None,
        asset_master: Optional["AssetMaster"] = None,
    ) -> UnifiedBacktestResult:
        """
        バックテストモード（検証用）

        Pipelineのweights計算をweights_funcとして使用し、
        時系列バックテストを実行。

        Args:
            universe: 資産リスト（シグナル計算用の全銘柄）
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
            trading_universe: 取引対象銘柄リスト（Noneの場合はuniverse全体）
                シグナル計算は全universで行い、ウェイト配分はtrading_universeのみ
            max_assets: 最大アセット数（Top-Nフィルター）。大規模ユニバース向け最適化。
                指定時、各リバランス日でシグナルスコア上位N銘柄のみを配分対象にする。
                推奨値: 500-1000（16,000銘柄ユニバースで2-4時間実行可能）
            progress_tracker: 進捗追跡インスタンス（オプション）。
                指定時、各フェーズとリバランス進捗をファイルに書き込む。
            cost_schedule: カテゴリ別取引コストスケジュール。
                指定時、銘柄カテゴリに応じた手数料を適用（米国株$2固定、日本株0.1%等）。
                未指定の場合はtransaction_cost_bpsの一律コストを適用。
            asset_master: アセットマスタ。銘柄カテゴリ情報を提供。
                cost_scheduleと組み合わせてカテゴリ別コストを計算。

        Returns:
            UnifiedBacktestResult: バックテスト結果
        """
        # Storage config: 明示的指定 or Settings から自動取得
        if storage_config is None and self.settings is not None:
            storage_config = self.settings.storage.to_storage_config()
        self._storage_config = storage_config

        # Progress tracking: set initial phase
        if progress_tracker:
            progress_tracker.set_phase("initializing")

        logger.info(
            f"Running backtest mode (v2.4): {len(universe)} assets, "
            f"{start_date} to {end_date}, {frequency}"
        )
        if max_assets:
            logger.info(f"Top-N filter: max_assets={max_assets} (large universe optimization)")
        if storage_config:
            logger.info(f"Storage backend: s3_bucket={storage_config.s3_bucket}")

        # 価格データをキャッシュ
        self._prices_cache = prices

        # === Signal Precomputation Phase (v3.0 unified mode) ===
        # IMPORTANT: This must be called BEFORE _inject_price_data()
        # because _inject_price_data() accesses backtest_pipeline property,
        # which creates Pipeline instance. We need _signal_precomputer to be set
        # before Pipeline creation so it can be passed to constructor.
        # This fixes the cache miss issue caused by late precomputer injection.
        from src.backtest.signal_precompute import SignalPrecomputer

        # Get signal count estimate (v3.0: unified mode with period variants)
        # Base signals × enabled variants ≈ 60 × 4 = ~240 signal variants
        try:
            from src.signals import SignalRegistry
            from src.config.settings import get_settings
            base_signal_count = len(SignalRegistry.list_all())
            # Estimate: ~80% of signals have period params, average 4-5 variants each
            settings = get_settings()
            num_variants = len(settings.signal_precompute.enabled_variants)
            signal_count = int(base_signal_count * 0.8 * num_variants + base_signal_count * 0.2)
        except Exception:
            signal_count = 230  # Fallback to expected count (~60 signals × ~4 variants)
        if progress_tracker:
            progress_tracker.set_phase("signal_precompute", total_steps=signal_count)
        self._precompute_signals(
            universe, prices, start_date, end_date, storage_config,
            progress_tracker=progress_tracker,
        )
        if progress_tracker and self._signal_precomputer:
            # Update signal cache stats
            stats = self._signal_precomputer.cache_stats
            progress_tracker.update_cache_stats({
                "signal": {
                    "hits": stats.get("hits", 0),
                    "misses": stats.get("misses", 0),
                }
            })
            # Mark signal precompute phase as completed
            # (Important for cache hit case where progress_callback is not called)
            progress_tracker.complete_phase("signal_precompute")

        # 価格データをPolars DataFrameに変換してPipelineに事前注入
        # これにより毎リバランスでのAPI呼び出しをスキップ
        if progress_tracker:
            progress_tracker.set_phase("data_loading")
        self._inject_price_data(universe, prices)
        logger.info("Price data pre-injected into Pipeline (skip_data_fetch=True)")

        # 取引対象銘柄セット（フィルタリング用）
        _trading_universe = trading_universe if trading_universe else universe
        _trading_set = set(_trading_universe)

        if trading_universe and len(trading_universe) != len(universe):
            logger.info(
                f"Trading filter active: {len(trading_universe)} tradable / {len(universe)} total"
            )

        # max_assets をクロージャでキャプチャ
        _max_assets = max_assets

        # Progress tracking: リバランス回数を見積もる
        _rebalance_count = [0]  # クロージャでキャプチャするためリスト
        _progress_tracker = progress_tracker  # クロージャでキャプチャ

        # === エクイティカーブ追跡（リアルタイム表示用） ===
        # ProgressTrackerに初期資本を設定
        if progress_tracker:
            progress_tracker.set_initial_capital(initial_capital)

        # 簡易ポートフォリオ追跡用（クロージャでキャプチャ）
        _last_weights: Dict[str, float] = {}  # 前回のウェイト
        _last_date: List[Optional[datetime]] = [None]  # 前回の日付
        _portfolio_value: List[float] = [initial_capital]  # 現在のポートフォリオ価値

        # Pipelineベースのweights_func (v2.4: 高速化版 + Top-Nフィルター)
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

            v2.4: Top-N filter for large universes
            - max_assets指定時、シグナルスコア上位N銘柄のみで最適化
            - 16,000銘柄 → 1,000銘柄: 計算量 568倍削減

            v3.0: trading_universe support
            - シグナル計算は全universeで実行
            - ウェイト配分はtrading_universeのみ

            v3.1: Real-time equity curve tracking
            - ProgressTrackerにスナップショットを送信
            - フロントエンドでリアルタイムエクイティカーブ表示
            """
            # Progress tracking: update rebalance count
            _rebalance_count[0] += 1

            # === エクイティカーブ追跡（リアルタイム表示用） ===
            # 前回からの期間リターンを計算してポートフォリオ価値を更新
            if _last_date[0] is not None and _last_weights:
                period_return = self._calculate_period_return(
                    weights=_last_weights,
                    prices=prices,
                    start_date=_last_date[0],
                    end_date=date,
                )
                _portfolio_value[0] *= (1.0 + period_return)

            # スナップショット付きで進捗更新
            if _progress_tracker:
                _progress_tracker.update_rebalance_with_snapshot(
                    index=_rebalance_count[0] - 1,
                    date=date,
                    portfolio_value=_portfolio_value[0],
                    weights=_last_weights if _last_weights else None,
                )

            try:
                # Top-N フィルター適用（大規模ユニバース最適化）
                pipeline_universe = universe
                if _max_assets and len(universe) > _max_assets:
                    pipeline_universe = filter_top_assets(
                        universe=universe,
                        prices=prices,
                        date=date,
                        max_assets=_max_assets,
                    )
                    logger.debug(
                        f"Top-N filter at {date}: {len(universe)} -> {len(pipeline_universe)} assets"
                    )

                # 日付でフィルタしたデータを再注入
                self._inject_filtered_data(pipeline_universe, prices, date)

                result = self.backtest_pipeline.run(
                    universe=pipeline_universe,
                    previous_weights=current_weights,
                    as_of_date=date,
                    data_cutoff_date=date,
                )

                if result.weights:
                    # 取引対象銘柄のみにフィルタリング
                    filtered_weights = {
                        k: v for k, v in result.weights.items()
                        if k in _trading_set and v > 0
                    }

                    if filtered_weights:
                        # フィルタ後のウェイトを正規化
                        total = sum(filtered_weights.values())
                        if total > 0:
                            normalized_weights = {k: v / total for k, v in filtered_weights.items()}
                            # 次回のエクイティ計算用にウェイトと日付を保存
                            _last_weights.clear()
                            _last_weights.update(normalized_weights)
                            _last_date[0] = date
                            return normalized_weights

                # Fallback: trading_universe内で等ウェイト
                logger.warning(f"Pipeline returned empty weights for {date}, using equal weight")
                fallback_universe = pipeline_universe if _max_assets else _trading_universe
                n = len(fallback_universe)
                fallback_weights = {symbol: 1.0 / n for symbol in fallback_universe}
                # フォールバック時もウェイトと日付を保存
                _last_weights.clear()
                _last_weights.update(fallback_weights)
                _last_date[0] = date
                return fallback_weights

            except BacktestError as e:
                # Strict mode: record error and stop backtest immediately
                error_msg = str(e)
                logger.error(f"Backtest error at {date}: {error_msg}")
                if _progress_tracker:
                    _progress_tracker.fail(error_msg)
                raise  # Re-raise to stop the entire backtest

            except Exception as e:
                # Unexpected errors also stop the backtest in strict mode
                error_msg = f"Unexpected pipeline error at {date}: {e}"
                logger.error(error_msg)
                if _progress_tracker:
                    _progress_tracker.fail(error_msg)
                raise BacktestError(error_msg) from e

        # バックテスト設定
        # エンジン固有設定（カテゴリ別コスト等）
        engine_specific = {}
        if cost_schedule is not None:
            engine_specific["cost_schedule"] = cost_schedule
        if asset_master is not None:
            engine_specific["asset_master"] = asset_master

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
            engine_specific_config=engine_specific,
        )

        # エンジン作成
        engine = create_engine(config=config)
        logger.info(f"Using engine: {engine.ENGINE_NAME}")

        # Progress tracking: set backtest phase with estimated total rebalances
        if progress_tracker:
            # リバランス日数を見積もる（frequencyに基づく）
            import pandas as pd
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            if frequency == "monthly":
                estimated_rebalances = len(date_range) // 30 + 1
            elif frequency == "weekly":
                estimated_rebalances = len(date_range) // 7 + 1
            else:  # daily
                estimated_rebalances = len(date_range)
            progress_tracker.set_phase("backtest", total_steps=estimated_rebalances)

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

        # Progress tracking: mark as complete
        if progress_tracker:
            progress_tracker.set_phase("report")
            # Collect final cache stats
            cache_stats = self._collect_all_cache_stats()
            if cache_stats:
                progress_tracker.update_cache_stats(cache_stats)
            progress_tracker.complete()

        return result

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._prices_cache.clear()
        if self._pipeline is not None:
            self._pipeline = None
        if self._backtest_pipeline is not None:
            self._backtest_pipeline = None
        self._signal_precomputer = None
        logger.info("Cache cleared")

    def _collect_all_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        全キャッシュから統計を収集

        Returns:
            キャッシュ統計辞書
        """
        stats: Dict[str, Dict[str, Any]] = {}

        # Signal precomputer stats
        if self._signal_precomputer is not None:
            try:
                signal_stats = self._signal_precomputer.cache_stats
                if signal_stats:
                    stats["signal"] = {
                        "hits": signal_stats.get("hits", 0),
                        "misses": signal_stats.get("misses", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to collect signal cache stats: {e}")

        # Pipeline covariance cache stats (if available)
        if self._backtest_pipeline is not None:
            try:
                # Try to access covariance cache from pipeline
                if hasattr(self._backtest_pipeline, "_risk_estimator"):
                    risk_est = self._backtest_pipeline._risk_estimator
                    if hasattr(risk_est, "_covariance_cache"):
                        cov_cache = risk_est._covariance_cache
                        if hasattr(cov_cache, "get_stats"):
                            cov_stats = cov_cache.get_stats()
                            stats["covariance"] = {
                                "hits": cov_stats.get("hits", 0),
                                "misses": cov_stats.get("misses", 0),
                            }
            except Exception as e:
                logger.debug(f"Failed to collect covariance cache stats: {e}")

        return stats

    def _precompute_signals(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        storage_config: Optional["StorageConfig"] = None,
        progress_tracker: Optional["ProgressTracker"] = None,
    ) -> None:
        """
        Pre-compute all signals for the entire backtest period.

        This is the key optimization for 15-year backtests:
        - Computes all signals once at the beginning
        - Stores in Parquet files (S3 or local)
        - Subsequent rebalances load from cache

        Expected speedup: 40x (21 hours -> 30 minutes for 15-year monthly backtest)

        Args:
            universe: List of asset tickers
            prices: Price data Dict[symbol, DataFrame]
            start_date: Backtest start date
            end_date: Backtest end date
            storage_config: Storage configuration (S3/local)
        """
        from src.backtest.signal_precompute import SignalPrecomputer

        logger.info("=" * 60)
        logger.info("SIGNAL PRECOMPUTATION PHASE (v2.3 Optimization)")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Universe: {len(universe)} assets")
        logger.info("=" * 60)

        # Convert prices to Polars DataFrame for SignalPrecomputer
        prices_records = []
        for symbol, df in prices.items():
            if symbol not in universe:
                continue
            if df.empty:
                continue

            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                df_copy.columns = ["timestamp"] + list(df.columns)
            elif "timestamp" not in df_copy.columns and "Date" in df_copy.columns:
                df_copy = df_copy.rename(columns={"Date": "timestamp"})

            # Ensure required columns exist
            if "timestamp" not in df_copy.columns:
                continue

            # Normalize column names
            col_map = {"Close": "close", "High": "high", "Low": "low", "Volume": "volume", "Open": "open"}
            df_copy = df_copy.rename(columns=col_map)

            if "close" not in df_copy.columns:
                continue

            df_copy["ticker"] = symbol
            prices_records.append(df_copy[["timestamp", "ticker", "close", "high", "low", "volume"]].dropna())

        if not prices_records:
            logger.warning("No valid price data for signal precomputation")
            return

        # Combine all prices into single DataFrame
        combined_prices = pd.concat(prices_records, ignore_index=True)
        prices_pl = pl.from_pandas(combined_prices)

        logger.info(f"Combined price data: {len(prices_pl)} rows, {prices_pl.select('ticker').unique().height} tickers")

        # Initialize SignalPrecomputer with StorageBackend (S3 required)
        from src.utils.storage_backend import StorageBackend
        if storage_config is not None:
            backend = StorageBackend(storage_config)
        else:
            from src.utils.storage_backend import get_storage_backend
            backend = get_storage_backend()

        # Progress callback for UI integration
        def on_signal_computed(signal_name: str) -> None:
            if progress_tracker:
                progress_tracker.increment_phase("signal_precompute")

        self._signal_precomputer = SignalPrecomputer(
            storage_backend=backend,
            progress_callback=on_signal_computed,
        )
        logger.info(f"Using storage: s3://{backend._s3_base_path}, local={backend._base_path}")

        # Precompute all signals (uses intelligent caching)
        import time
        start_time = time.time()

        computed = self._signal_precomputer.precompute_all(prices_pl)

        elapsed = time.time() - start_time

        if computed:
            logger.info(f"Signal precomputation completed in {elapsed:.2f}s (computed)")
        else:
            logger.info(f"Signal precomputation completed in {elapsed:.2f}s (cache hit)")

        # NOTE: backtest_pipeline property will use self._signal_precomputer at construction
        # time since we've now set it. If pipeline was already created, update it.
        # This ensures StrategyEvaluator gets the precomputer on first property access.
        if self._backtest_pipeline is not None:
            self._backtest_pipeline._signal_precomputer = self._signal_precomputer

        # Log cache stats
        stats = self._signal_precomputer.cache_stats
        logger.info(f"Cache stats: {stats.get('num_signals', 0)} signals cached")
        logger.info("=" * 60)

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
            except Exception as e:
                logger.debug(f"Failed to convert {symbol} to Polars (filtered): {e}")
                continue

        # Pipelineの_raw_dataに注入
        self.backtest_pipeline._raw_data = raw_data

    def _calculate_period_return(
        self,
        weights: Dict[str, float],
        prices: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> float:
        """
        指定期間のポートフォリオリターンを計算

        リアルタイムエクイティカーブ追跡用。各銘柄の期間リターンをウェイト加重平均。

        Args:
            weights: ウェイト辞書 {symbol: weight}
            prices: 価格データ {symbol: DataFrame}
            start_date: 開始日
            end_date: 終了日

        Returns:
            期間リターン（例: 0.05 = 5%）
        """
        if not weights:
            return 0.0

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        total_return = 0.0
        total_weight = 0.0

        for symbol, weight in weights.items():
            if symbol not in prices or weight <= 0:
                continue

            df = prices[symbol]
            if df.empty:
                continue

            try:
                # 価格データから終値を取得
                if isinstance(df.index, pd.DatetimeIndex):
                    close_col = "Close" if "Close" in df.columns else "close"
                    if close_col not in df.columns:
                        continue

                    # 開始日と終了日の終値を取得
                    start_prices = df[df.index <= start_ts][close_col]
                    end_prices = df[df.index <= end_ts][close_col]

                    if start_prices.empty or end_prices.empty:
                        continue

                    start_price = start_prices.iloc[-1]
                    end_price = end_prices.iloc[-1]
                else:
                    close_col = "Close" if "Close" in df.columns else "close"
                    ts_col = "timestamp" if "timestamp" in df.columns else "Date"
                    if close_col not in df.columns or ts_col not in df.columns:
                        continue

                    start_df = df[df[ts_col] <= start_ts]
                    end_df = df[df[ts_col] <= end_ts]

                    if start_df.empty or end_df.empty:
                        continue

                    start_price = start_df[close_col].iloc[-1]
                    end_price = end_df[close_col].iloc[-1]

                if start_price <= 0:
                    continue

                # 銘柄リターンを計算
                asset_return = (end_price / start_price) - 1.0
                total_return += weight * asset_return
                total_weight += weight

            except Exception as e:
                logger.debug(f"Failed to calculate return for {symbol}: {e}")
                continue

        # ウェイト合計で正規化（一部の銘柄がない場合の対応）
        if total_weight > 0 and total_weight < 0.99:
            # 一部の銘柄がない場合、残りを現金と見なす（0%リターン）
            total_return = total_return  # そのまま

        return total_return

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
            f"Running backtest with checkpoint support (v2.3): "
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

        # === Signal Precomputation Phase (v2.3) ===
        storage_config = self._storage_config
        if storage_config is None and self.settings is not None:
            storage_config = self.settings.storage.to_storage_config()
        self._precompute_signals(universe, prices, start_date, end_date, storage_config)

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
            except BacktestError:
                # Strict mode: re-raise to stop backtest immediately
                raise
            except Exception as e:
                # Wrap unexpected errors and re-raise
                raise BacktestError(f"Pipeline error at {date}: {e}") from e

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
