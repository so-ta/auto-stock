"""
Fast Backtest Engine - 高速バックテストエンジン

Phase 1 で作成したモジュールを統合した高速バックテストエンジン。
事前計算シグナル、インクリメンタル共分散、ベクトル化計算を組み合わせて
3-5倍の高速化を実現。

Phase 2（task_022_7）: Numba/GPU統合により累積50-100倍高速化。

主な高速化ポイント:
- シグナル: 事前計算済みをメモリマップで読み込み
- 共分散: インクリメンタル更新（毎回再計算しない）
- ポートフォリオ計算: ベクトル化
- メインループ: NumPy配列で効率的に計算
- Numba: @njit(parallel=True) でJITコンパイル高速化
- GPU: CuPy対応時にGPUで行列演算

使用例:
    from src.backtest.fast_engine import FastBacktestEngine, FastBacktestConfig
    from datetime import datetime

    config = FastBacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        rebalance_frequency="monthly",
        initial_capital=100000.0,
        use_numba=True,   # Numba JIT高速化
        use_gpu=False,    # GPU使用（CuPy必要）
    )

    engine = FastBacktestEngine(config)
    result = engine.run(prices_df, asset_names=["SPY", "QQQ", "TLT", "GLD"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from src.utils.storage_backend import StorageBackend, StorageConfig

import numpy as np
import pandas as pd
import polars as pl

# Polars is now required for this module
HAS_POLARS = True

from .result import BacktestResult, DailySnapshot
from .covariance_cache import (
    IncrementalCovarianceEstimator,
    CovarianceCache,
)
from .base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
    RebalanceRecord as BaseRebalanceRecord,
    TradeRecord,
)

# RegimeDetector import (optional)
try:
    from ..signals.regime_detector_v2 import (
        EnhancedRegimeDetector,
        RegimeAdaptiveStrategy,
        RegimeType,
        RegimeResult,
    )
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    EnhancedRegimeDetector = None
    RegimeAdaptiveStrategy = None
    RegimeType = None
    RegimeResult = None

# DrawdownProtection import (SYNC-002)
try:
    from ..risk.drawdown_protection import (
        DrawdownProtector,
        DrawdownProtectorConfig,
        ProtectionStatus,
    )
    DRAWDOWN_PROTECTION_AVAILABLE = True
except ImportError:
    DRAWDOWN_PROTECTION_AVAILABLE = False
    DrawdownProtector = None
    DrawdownProtectorConfig = None
    ProtectionStatus = None

# VIX Signal import (optional)
try:
    from ..signals.vix_signal import (
        EnhancedVIXSignal,
        VIXSignalConfig,
        VIXSignalResult,
        get_vix_cash_allocation,
        calculate_vix_adjusted_weights,
    )
    VIX_SIGNAL_AVAILABLE = True
except ImportError:
    VIX_SIGNAL_AVAILABLE = False
    EnhancedVIXSignal = None
    VIXSignalConfig = None
    VIXSignalResult = None
    get_vix_cash_allocation = None
    calculate_vix_adjusted_weights = None

# TransactionCostOptimizer import (optional - SYNC-001)
try:
    from ..allocation.transaction_cost_optimizer import (
        TransactionCostOptimizer,
        TransactionCostConfig,
        OptimizationResult as TCOptResult,
        TurnoverConstrainedOptimizer,
    )
    TRANSACTION_COST_OPTIMIZER_AVAILABLE = True
except ImportError:
    TRANSACTION_COST_OPTIMIZER_AVAILABLE = False
    TransactionCostOptimizer = None
    TransactionCostConfig = None
    TCOptResult = None
    TurnoverConstrainedOptimizer = None

# DynamicWeighter import (optional - SYNC-005)
try:
    from ..meta.dynamic_weighter import (
        DynamicWeighter,
        DynamicWeightingConfig,
        DynamicWeightingResult,
    )
    DYNAMIC_WEIGHTER_AVAILABLE = True
except ImportError:
    DYNAMIC_WEIGHTER_AVAILABLE = False
    DynamicWeighter = None
    DynamicWeightingConfig = None
    DynamicWeightingResult = None

# DividendHandler import (optional - SYNC-006)
try:
    from ..data.dividend_handler import (
        DividendHandler,
        DividendConfig,
        DividendData,
        TotalReturnResult,
    )
    DIVIDEND_HANDLER_AVAILABLE = True
except ImportError:
    DIVIDEND_HANDLER_AVAILABLE = False
    DividendHandler = None
    DividendConfig = None
    DividendData = None
    TotalReturnResult = None

logger = logging.getLogger(__name__)


class RebalanceFrequency(str, Enum):
    """リバランス頻度"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class FastBacktestConfig:
    """高速バックテスト設定

    Attributes:
        start_date: 開始日
        end_date: 終了日
        rebalance_frequency: リバランス頻度
        initial_capital: 初期資金
        transaction_cost_bps: 取引コスト（bps）
        cov_halflife: 共分散半減期（日）
        use_signal_cache: シグナルキャッシュを使用
        signal_cache_dir: シグナルキャッシュディレクトリ
        cov_cache_dir: 共分散キャッシュディレクトリ
        min_weight: 最小ウェイト
        max_weight: 最大ウェイト
        use_numba: Numba JIT高速化を使用
        use_gpu: GPU計算を使用（CuPy必要）
        numba_parallel: Numba並列化を有効化
        warmup_jit: JITコンパイル事前ウォームアップ
    """

    start_date: datetime
    end_date: datetime
    rebalance_frequency: str = "monthly"
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0  # スリッページ（bps）- SYNC統一規格対応
    cov_halflife: int = 60
    use_signal_cache: bool = True
    signal_cache_dir: str = ".cache/signals"
    cov_cache_dir: str = ".cache/covariance"
    min_weight: float = 0.0
    max_weight: float = 1.0
    # Numba/GPU設定
    # GPU計算を有効にするには:
    #   1. CuPyをインストール: pip install cupy-cuda12x (CUDA 12.x) or cupy-cuda11x (CUDA 11.x)
    #   2. use_gpu=True を設定
    #   3. scripts/check_gpu.py で環境確認可能
    # ResourceConfigからの自動検出: FastBacktestConfig.from_resource_config() を使用
    use_numba: bool = True
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.8  # GPU使用時のメモリ使用率（0.0-1.0）
    numba_parallel: bool = True
    warmup_jit: bool = True
    # レジーム検出設定
    use_regime_detection: bool = False
    regime_lookback: int = 60
    regime_adaptive_weights: bool = True  # レジームに応じてウェイト調整
    regime_risk_scaling: bool = True  # レジームに応じてリスク調整
    # VIX動的キャッシュ配分設定
    vix_cash_enabled: bool = True
    vix_config_path: Optional[str] = None
    # TransactionCostOptimizer設定 (SYNC-001)
    use_cost_optimizer: bool = False
    cost_aversion: float = 1.0
    risk_aversion: float = 2.0
    max_turnover: float = 0.20
    # DrawdownProtection設定 (SYNC-002)
    use_drawdown_protection: bool = False
    dd_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    dd_risk_reductions: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.3])
    dd_recovery_threshold: float = 0.5
    dd_emergency_level: float = 0.25
    dd_emergency_cash_ratio: float = 0.8
    # DynamicWeighter設定 (SYNC-005)
    use_dynamic_weighting: bool = False
    target_volatility: float = 0.15
    dd_protection_threshold: float = 0.10
    vol_scaling_enabled: bool = True
    dd_protection_enabled: bool = True
    # DividendHandler設定 (SYNC-006)
    use_dividends: bool = True
    reinvest_dividends: bool = True
    withholding_tax_rate: float = 0.0
    # StorageBackend設定 (S3対応)
    storage_config: Optional["StorageConfig"] = None

    @classmethod
    def from_resource_config(
        cls,
        start_date: datetime,
        end_date: datetime,
        **kwargs,
    ) -> "FastBacktestConfig":
        """
        ResourceConfigからインスタンスを生成

        システムリソースに基づいた最適な設定でFastBacktestConfigを作成する。
        start_date/end_dateは必須。その他のパラメータはkwargsでオーバーライド可能。

        Parameters
        ----------
        start_date : datetime
            バックテスト開始日
        end_date : datetime
            バックテスト終了日
        **kwargs
            その他のオーバーライドパラメータ

        Returns
        -------
        FastBacktestConfig
            ResourceConfigベースの設定

        Example
        -------
        >>> config = FastBacktestConfig.from_resource_config(
        ...     start_date=datetime(2020, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     rebalance_frequency="weekly",
        ... )
        """
        from src.config.resource_config import get_current_resource_config

        rc = get_current_resource_config()

        # ResourceConfigからの設定を適用
        # GPU設定はResourceConfigが自動検出した値を使用
        config_dict = {
            "start_date": start_date,
            "end_date": end_date,
            "use_numba": rc.use_numba,
            "numba_parallel": rc.numba_parallel,
            "use_gpu": rc.use_gpu,
            "gpu_memory_fraction": rc.gpu_memory_fraction,
        }

        # kwargsでオーバーライド
        config_dict.update(kwargs)

        return cls(**config_dict)


@dataclass
class SimulationState:
    """シミュレーション状態

    Attributes:
        date: 現在日
        portfolio_value: ポートフォリオ価値
        weights: 現在のウェイト
        cash: キャッシュ
    """

    date: datetime
    portfolio_value: float
    weights: np.ndarray
    cash: float = 0.0


@dataclass
class SimulationResult:
    """シミュレーション結果

    Attributes:
        dates: 日付配列
        portfolio_values: ポートフォリオ価値配列
        weights_history: ウェイト履歴
        returns: 日次リターン
        rebalance_dates: リバランス日
        transaction_costs: 取引コスト合計
    """

    dates: List[datetime]
    portfolio_values: np.ndarray
    weights_history: np.ndarray
    returns: np.ndarray
    rebalance_dates: List[datetime]
    transaction_costs: float


class _WeightsFuncAdapter:
    """
    外部weights_funcを内部形式に変換するアダプター

    WeightsFuncProtocol準拠の外部関数を、FastBacktestEngine内部の
    (signals, cov_matrix) -> np.ndarray 形式に変換する。

    リバランス時に正確な日付と現在ウェイトを外部関数に渡すため、
    状態を保持する。

    Usage:
        adapter = _WeightsFuncAdapter(external_func, universe, prices)
        adapter.update_state(current_date, current_weights_dict)
        new_weights = adapter(signals, cov_matrix)
    """

    def __init__(
        self,
        external_func: callable,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
    ):
        """
        初期化

        Parameters
        ----------
        external_func : callable
            外部weights_func（WeightsFuncProtocol準拠）
            シグネチャ: (universe, prices, date, current_weights) -> Dict[str, float]
        universe : List[str]
            ユニバース（銘柄リスト）
        prices : Dict[str, pd.DataFrame]
            価格データ
        """
        self.external_func = external_func
        self.universe = universe
        self.prices = prices
        self.current_date: Optional[datetime] = None
        self.current_weights: Dict[str, float] = {s: 1.0 / len(universe) for s in universe}

    def update_state(self, date: datetime, weights: Dict[str, float]) -> None:
        """
        状態を更新（各リバランス前に呼び出し）

        Parameters
        ----------
        date : datetime
            現在のリバランス日
        weights : Dict[str, float]
            現在のウェイト
        """
        self.current_date = date
        self.current_weights = weights

    def __call__(self, signals: Dict[str, float], cov_matrix: np.ndarray) -> np.ndarray:
        """
        内部形式で呼び出し

        Parameters
        ----------
        signals : Dict[str, float]
            シグナル（内部形式、外部関数には渡さない）
        cov_matrix : np.ndarray
            共分散行列（内部形式、外部関数には渡さない）

        Returns
        -------
        np.ndarray
            新しいウェイト（universe順の配列）
        """
        # 外部関数を呼び出し（WeightsFuncProtocol準拠シグネチャ）
        new_weights_dict = self.external_func(
            self.universe,
            self.prices,
            self.current_date,
            self.current_weights,
        )

        # Dict形式からnp.array形式に変換（universe順）
        new_weights = np.array([
            new_weights_dict.get(symbol, 0.0)
            for symbol in self.universe
        ])

        # 正規化
        total = np.sum(new_weights)
        if total > 0:
            new_weights = new_weights / total

        # 状態を更新
        self.current_weights = new_weights_dict

        return new_weights


class FastBacktestEngine(BacktestEngineBase):
    """
    高速バックテストエンジン（BacktestEngineBase準拠）

    Phase 1 で作成した以下のモジュールを統合:
    - SignalPrecomputer: 事前計算シグナル
    - IncrementalCovarianceEstimator: インクリメンタル共分散
    - IncrementalSignalEngine: インクリメンタルシグナル（オプション）

    Phase 2: Numba/GPU計算バックエンド統合
    - use_gpu=True: GPU(CuPy)で高速行列演算
    - use_numba=True: Numba JITで高速化
    - フォールバック: NumPy

    INT-003: BacktestEngineBase準拠（共通インターフェース）
    - run()とvalidate_inputs()を実装
    - FastBacktestConfigとUnifiedBacktestConfigの両方に対応

    Usage:
        config = FastBacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 12, 31),
            use_numba=True,
            use_gpu=False,
        )
        engine = FastBacktestEngine(config)
        result = engine.run(prices_df)
    """

    # エンジン名（BacktestEngineBase準拠）
    ENGINE_NAME: str = "FastBacktestEngine"

    def __init__(
        self,
        config: Union[FastBacktestConfig, UnifiedBacktestConfig],
        signal_precomputer: Optional[Any] = None,
        incremental_engine: Optional[Any] = None,
    ):
        """
        初期化

        Parameters
        ----------
        config : FastBacktestConfig | UnifiedBacktestConfig
            バックテスト設定（両方の設定形式に対応）
        signal_precomputer : SignalPrecomputer, optional
            シグナル事前計算器（Noneの場合は新規作成）
        incremental_engine : IncrementalSignalEngine, optional
            インクリメンタルシグナルエンジン
        """
        # UnifiedBacktestConfigの場合はFastBacktestConfigに変換
        if isinstance(config, UnifiedBacktestConfig):
            fast_config = self._convert_from_unified_config(config)
            self._unified_config = config
        else:
            fast_config = config
            self._unified_config = None

        # 親クラス初期化
        super().__init__(self._unified_config)

        self.config = fast_config

        # StorageBackend初期化（S3対応）
        self._storage_backend: Optional["StorageBackend"] = None
        if fast_config.storage_config is not None:
            from src.utils.storage_backend import get_storage_backend
            self._storage_backend = get_storage_backend(fast_config.storage_config)

        # 共分散推定器（後で初期化）
        self.cov_estimator: Optional[IncrementalCovarianceEstimator] = None
        self.cov_cache = CovarianceCache(
            cache_dir=fast_config.cov_cache_dir,
            storage_backend=self._storage_backend,
        )

        # シグナル関連（オプション）
        self.signal_precomputer = signal_precomputer
        self.incremental_engine = incremental_engine

        # 内部状態
        self._n_assets = 0
        self._asset_names: List[str] = []
        self._rebalance_dates: List[datetime] = []

        # 計算バックエンド関数（_setup_compute_backend で設定）
        self._cov_func: Optional[callable] = None
        self._matmul_func: Optional[callable] = None
        self._momentum_func: Optional[callable] = None
        self._zscore_func: Optional[callable] = None
        self._volatility_func: Optional[callable] = None
        self._compute_backend: str = "numpy"  # "gpu", "numba", "numpy"

        # 計算バックエンドをセットアップ
        self._setup_compute_backend()

        # レジーム検出器（オプション）
        self._regime_detector: Optional[Any] = None
        self._current_regime: Optional[str] = None
        self._regime_params: Dict[str, Any] = {}
        if config.use_regime_detection and REGIME_DETECTOR_AVAILABLE:
            self._regime_detector = EnhancedRegimeDetector({
                "lookback": config.regime_lookback,
            })
            logger.info("RegimeDetector enabled with lookback=%d", config.regime_lookback)

        # DrawdownProtector（SYNC-002）
        self._dd_protector: Optional[Any] = None
        if config.use_drawdown_protection and DRAWDOWN_PROTECTION_AVAILABLE:
            dd_config = DrawdownProtectorConfig(
                dd_levels=config.dd_levels,
                risk_reductions=config.dd_risk_reductions,
                recovery_threshold=config.dd_recovery_threshold,
                emergency_dd_level=config.dd_emergency_level,
                emergency_cash_ratio=config.dd_emergency_cash_ratio,
            )
            self._dd_protector = DrawdownProtector(dd_config, initial_value=config.initial_capital)
            logger.info(
                "DrawdownProtector enabled: levels=%s, reductions=%s",
                config.dd_levels, config.dd_risk_reductions,
            )

        # VIXシグナル（SYNC-004）
        self._vix_signal: Optional[Any] = None
        self._vix_data: Optional[np.ndarray] = None
        self._vix_dates: Optional[List[datetime]] = None
        self._last_vix_cash_allocation: float = 0.0
        if config.vix_cash_enabled and VIX_SIGNAL_AVAILABLE:
            if config.vix_config_path:
                self._vix_signal = EnhancedVIXSignal.from_config(config.vix_config_path)
            else:
                self._vix_signal = EnhancedVIXSignal()
            logger.info("VIX dynamic cash allocation enabled")

        # DynamicWeighter（SYNC-005）
        self._dynamic_weighter: Optional[Any] = None
        self._dynamic_weighting_result: Optional[Dict[str, Any]] = None
        self._portfolio_peak_value: float = config.initial_capital
        if config.use_dynamic_weighting and DYNAMIC_WEIGHTER_AVAILABLE:
            dw_config = DynamicWeightingConfig(
                target_volatility=config.target_volatility,
                max_drawdown_trigger=config.dd_protection_threshold,
                vol_scaling_enabled=config.vol_scaling_enabled,
                dd_protection_enabled=config.dd_protection_enabled,
                regime_weighting_enabled=config.use_regime_detection,
            )
            self._dynamic_weighter = DynamicWeighter(dw_config)
            logger.info(
                "DynamicWeighter enabled: target_vol=%.2f, dd_trigger=%.2f",
                config.target_volatility, config.dd_protection_threshold,
            )

        # DividendHandler（SYNC-006）
        self._dividend_handler: Optional[Any] = None
        self._dividend_data: Optional[Dict[str, Any]] = None
        if config.use_dividends and DIVIDEND_HANDLER_AVAILABLE:
            div_config = DividendConfig(
                reinvest_dividends=config.reinvest_dividends,
                withholding_tax_rate=config.withholding_tax_rate,
            )
            self._dividend_handler = DividendHandler(div_config)
            logger.info(
                "DividendHandler enabled: reinvest=%s, tax_rate=%.2f%%",
                config.reinvest_dividends, config.withholding_tax_rate * 100,
            )

        logger.info(
            "FastBacktestEngine initialized: %s to %s, freq=%s, backend=%s, regime=%s, dd=%s, vix=%s, dw=%s, div=%s",
            config.start_date,
            config.end_date,
            config.rebalance_frequency,
            self._compute_backend,
            "enabled" if self._regime_detector else "disabled",
            "enabled" if self._dd_protector else "disabled",
            "enabled" if self._vix_signal else "disabled",
            "enabled" if self._dynamic_weighter else "disabled",
            "enabled" if self._dividend_handler else "disabled",
        )

    @staticmethod
    def _convert_from_unified_config(config: UnifiedBacktestConfig) -> "FastBacktestConfig":
        """
        UnifiedBacktestConfigからFastBacktestConfigに変換（INT-003）

        Parameters
        ----------
        config : UnifiedBacktestConfig
            共通設定

        Returns
        -------
        FastBacktestConfig
            エンジン固有設定
        """
        # engine_specific_configからエンジン固有設定を取得
        specific = config.engine_specific_config or {}

        return FastBacktestConfig(
            start_date=config.start_date if isinstance(config.start_date, datetime) else datetime.strptime(config.start_date, "%Y-%m-%d"),
            end_date=config.end_date if isinstance(config.end_date, datetime) else datetime.strptime(config.end_date, "%Y-%m-%d"),
            initial_capital=config.initial_capital,
            rebalance_frequency=config.rebalance_frequency,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            max_weight=config.max_weight,
            min_weight=config.min_weight,
            # エンジン固有設定
            use_numba=specific.get("use_numba", True),
            use_gpu=specific.get("use_gpu", False),
            use_regime_detection=specific.get("use_regime_detection", False),
            regime_lookback=specific.get("regime_lookback", 60),
            regime_risk_scaling=specific.get("regime_risk_scaling", False),
            regime_adaptive_weights=specific.get("regime_adaptive_weights", False),
            use_cost_optimizer=specific.get("use_cost_optimizer", False),
            max_turnover=specific.get("max_turnover", 0.20),
            use_drawdown_protection=specific.get("use_dd_protection", False),
            dd_recovery_threshold=specific.get("dd_recovery_threshold", 0.5),
            vix_cash_enabled=specific.get("vix_cash_enabled", False),
            use_dynamic_weighting=specific.get("use_dynamic_weighting", False),
            use_dividends=specific.get("use_dividend_adjustment", False),
        )

    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """
        入力を検証（BacktestEngineBase準拠）

        Parameters
        ----------
        universe : List[str]
            ユニバース（銘柄リスト）
        prices : Dict[str, pd.DataFrame]
            価格データ
        config : UnifiedBacktestConfig
            設定

        Returns
        -------
        bool
            検証結果（True=有効）

        Raises
        ------
        ValueError
            検証エラー
        """
        # 共通検証
        warnings = self._validate_common_inputs(universe, prices, config)
        for warning in warnings:
            logger.warning(warning)

        # FastBacktestEngine固有の検証
        if config.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        # リバランス頻度のチェック
        valid_frequencies = {"daily", "weekly", "monthly", "quarterly"}
        if config.rebalance_frequency.lower() not in valid_frequencies:
            raise ValueError(f"Invalid rebalance_frequency: {config.rebalance_frequency}")

        return True

    def _setup_compute_backend(self) -> None:
        """
        計算バックエンドをセットアップ

        優先順位:
        1. GPU (use_gpu=True かつ CuPy利用可能)
        2. Numba (use_numba=True かつ Numba利用可能)
        3. NumPy (フォールバック)
        """
        # GPU優先
        if self.config.use_gpu:
            try:
                from .gpu_compute import (
                    GPU_AVAILABLE,
                    covariance_gpu,
                    matrix_multiply_gpu,
                    is_gpu_available,
                )

                if GPU_AVAILABLE and is_gpu_available():
                    self._cov_func = covariance_gpu
                    self._matmul_func = matrix_multiply_gpu
                    self._compute_backend = "gpu"
                    logger.info("GPU compute backend enabled")
                    return
                else:
                    logger.warning("GPU requested but not available, falling back")
            except ImportError:
                logger.warning("gpu_compute module not found, falling back")

        # Numba
        if self.config.use_numba:
            try:
                from .numba_compute import (
                    NUMBA_AVAILABLE,
                    covariance_matrix as numba_cov,
                    momentum_batch,
                    volatility_batch,
                    zscore_batch,
                    warmup_jit,
                    check_numba_available,
                )

                if NUMBA_AVAILABLE or check_numba_available():
                    self._cov_func = numba_cov
                    self._momentum_func = momentum_batch
                    self._volatility_func = volatility_batch
                    self._zscore_func = zscore_batch
                    self._compute_backend = "numba"

                    # JITウォームアップ
                    if self.config.warmup_jit:
                        warmup_jit()
                        logger.info("Numba JIT warmup completed")

                    logger.info("Numba compute backend enabled")
                    return
                else:
                    logger.warning("Numba requested but not available, falling back")
            except ImportError:
                logger.warning("numba_compute module not found, falling back")

        # NumPyフォールバック
        self._cov_func = np.cov
        self._matmul_func = np.dot
        self._compute_backend = "numpy"
        logger.info("Using NumPy compute backend (fallback)")

    def _compute_signals(
        self,
        prices: np.ndarray,
        periods: Optional[List[int]] = None,
        zscore_window: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        シグナルを計算（バックエンドに応じて切り替え）

        Parameters
        ----------
        prices : np.ndarray
            価格行列 (n_days, n_assets)
        periods : List[int], optional
            モメンタム期間リスト
        zscore_window : int
            Z-Scoreウィンドウサイズ

        Returns
        -------
        Dict[str, np.ndarray]
            シグナル辞書 {'momentum': ..., 'zscore': ..., 'volatility': ...}
        """
        if periods is None:
            periods = [20, 60, 120]

        result = {}

        if self._compute_backend == "numba" and self._momentum_func is not None:
            # Numbaバックエンド
            periods_arr = np.array(periods, dtype=np.int64)
            # (n_days, n_assets) -> (n_assets, n_days) for numba functions
            prices_t = prices.T.astype(np.float64)

            result['momentum'] = self._momentum_func(prices_t, periods_arr)
            result['zscore'] = self._zscore_func(prices_t, zscore_window)

            # リターンからボラティリティ
            returns_t = np.zeros_like(prices_t)
            returns_t[:, 1:] = prices_t[:, 1:] / prices_t[:, :-1] - 1
            result['volatility'] = self._volatility_func(returns_t, zscore_window)

        else:
            # Pure NumPyフォールバック（vectorized_computeはPolars DataFrameを期待）
            n_days, n_assets = prices.shape

            # 簡易モメンタム計算
            momentum_list = []
            for period in periods:
                if n_days > period:
                    mom = prices[period:] / prices[:-period] - 1
                    padded = np.vstack([
                        np.full((period, n_assets), np.nan),
                        mom
                    ])
                    momentum_list.append(padded)
                else:
                    momentum_list.append(np.full((n_days, n_assets), np.nan))

            result['momentum'] = np.stack(momentum_list, axis=-1)

            # 簡易Z-Score
            result['zscore'] = np.zeros((n_days, n_assets))
            for i in range(zscore_window, n_days):
                window = prices[i - zscore_window:i]
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0)
                std = np.where(std > 0, std, 1.0)
                result['zscore'][i] = (prices[i] - mean) / std

            # 簡易ボラティリティ
            returns = np.zeros_like(prices)
            returns[1:] = prices[1:] / prices[:-1] - 1
            result['volatility'] = np.zeros((n_days, n_assets))
            for i in range(zscore_window, n_days):
                result['volatility'][i] = np.std(
                    returns[i - zscore_window:i], axis=0
                ) * np.sqrt(252)

        return result

    def _compute_covariance_fast(self, returns: np.ndarray) -> np.ndarray:
        """
        高速共分散計算（バックエンドに応じて切り替え）

        Parameters
        ----------
        returns : np.ndarray
            リターン行列 (n_days, n_assets)

        Returns
        -------
        np.ndarray
            共分散行列 (n_assets, n_assets)
        """
        if self._compute_backend == "gpu" and self._cov_func is not None:
            # GPU
            return self._cov_func(returns.T)
        elif self._compute_backend == "numba" and self._cov_func is not None:
            # Numba（半減期付き）
            return self._cov_func(returns.T, halflife=self.config.cov_halflife)
        else:
            # NumPy
            return np.cov(returns.T)

    def run_unified(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[callable] = None,
    ) -> UnifiedBacktestResult:
        """
        バックテストを実行（BacktestEngineBase準拠インターフェース）

        INT-003: 全エンジン共通インターフェース

        Parameters
        ----------
        universe : List[str]
            ユニバース（銘柄リスト）
        prices : Dict[str, pd.DataFrame]
            価格データ（{symbol: DataFrame}）
        config : UnifiedBacktestConfig, optional
            設定（Noneの場合はコンストラクタの設定を使用）
        weights_func : callable, optional
            ウェイト計算関数

        Returns
        -------
        UnifiedBacktestResult
            共通形式のバックテスト結果
        """
        # 設定を更新
        if config is not None:
            self.config = self._convert_from_unified_config(config)
            self._unified_config = config
        elif self._unified_config is not None:
            config = self._unified_config
        else:
            # FastBacktestConfigからUnifiedBacktestConfigを作成
            config = UnifiedBacktestConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=self.config.initial_capital,
                rebalance_frequency=self.config.rebalance_frequency,
                transaction_cost_bps=self.config.transaction_cost_bps,
            )

        # 入力検証
        self.validate_inputs(universe, prices, config)

        # prices DictをDataFrameに変換
        price_df = self._convert_prices_dict_to_df(prices, universe)

        # weights_funcをアダプター経由で変換
        # WeightsFuncAdapter: 状態を保持し、外部形式→内部形式の変換を行う
        adapter = None
        if weights_func is not None:
            adapter = _WeightsFuncAdapter(weights_func, universe, prices)

        # 内部run()を呼び出し（アダプター付き）
        result = self._run_with_weights_adapter(price_df, adapter, universe)

        # BacktestResultをUnifiedBacktestResultに変換
        return self._convert_to_unified_result(result, config)

    def _run_with_weights_adapter(
        self,
        prices: pd.DataFrame,
        adapter: Optional["_WeightsFuncAdapter"],
        universe: List[str],
    ) -> "BacktestResult":
        """
        weights_funcアダプター付きでバックテストを実行

        外部weights_func（WeightsFuncProtocol準拠）を使用する場合、
        各リバランス時に正確な日付と現在ウェイトをアダプターに渡す。

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ
        adapter : _WeightsFuncAdapter, optional
            weights_funcアダプター
        universe : List[str]
            ユニバース

        Returns
        -------
        BacktestResult
            バックテスト結果
        """
        # データ準備
        prices = self._prepare_prices(prices, None)
        self._asset_names = list(prices.columns)
        self._n_assets = len(self._asset_names)

        # 共分散推定器を初期化
        self.cov_estimator = IncrementalCovarianceEstimator(
            n_assets=self._n_assets,
            halflife=self.config.cov_halflife,
            asset_names=self._asset_names,
        )

        # リバランス日を計算
        self._rebalance_dates = self._get_rebalance_dates(prices.index)

        n_days = len(prices)
        n_assets = self._n_assets

        # 配列を事前確保
        portfolio_values = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))
        daily_returns = np.zeros(n_days)

        # 初期状態
        portfolio_values[0] = self.config.initial_capital
        current_weights = np.ones(n_assets) / n_assets
        weights_history[0] = current_weights

        # 価格をNumPy配列に変換
        price_matrix = prices.values
        dates = prices.index.tolist()

        # リバランス日をセットに変換
        rebalance_set = set(self._rebalance_dates)

        # 取引コスト
        cost_rate = self.config.transaction_cost_bps / 10000.0
        total_costs = 0.0
        actual_rebalance_dates = []

        for i in range(1, n_days):
            current_date = dates[i]
            prev_prices = price_matrix[i - 1]
            curr_prices = price_matrix[i]

            # 日次リターンを計算
            price_returns = np.where(
                prev_prices > 0,
                curr_prices / prev_prices - 1,
                0.0
            )
            asset_returns = price_returns

            # 共分散推定器を更新
            self.cov_estimator.update(asset_returns)

            # リバランス判定
            should_rebalance = current_date in rebalance_set

            if should_rebalance and i > 0:
                # 新しいウェイトを計算
                if adapter is not None:
                    # アダプター状態を更新（正確な日付と現在ウェイト）
                    current_weights_dict = {
                        self._asset_names[j]: float(current_weights[j])
                        for j in range(n_assets)
                    }
                    adapter.update_state(current_date, current_weights_dict)

                    # アダプター経由でウェイトを取得
                    cov = self.cov_estimator.get_covariance()
                    signals = self._get_current_signals(current_date)
                    new_weights = adapter(signals, cov)
                else:
                    # 等ウェイト
                    new_weights = np.ones(n_assets) / n_assets

                # ウェイト制約を適用
                new_weights = self._apply_weight_constraints(new_weights)

                # 取引コストを計算
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = turnover * cost_rate * portfolio_values[i - 1]
                total_costs += cost

                current_weights = new_weights
                actual_rebalance_dates.append(current_date)

            # ポートフォリオリターンを計算
            portfolio_return = np.dot(current_weights, asset_returns)
            daily_returns[i] = portfolio_return

            # ポートフォリオ価値を更新
            portfolio_values[i] = portfolio_values[i - 1] * (1 + portfolio_return)

            # ウェイト履歴を保存
            weights_history[i] = current_weights

        sim_result = SimulationResult(
            dates=dates,
            portfolio_values=portfolio_values,
            weights_history=weights_history,
            returns=daily_returns,
            rebalance_dates=actual_rebalance_dates,
            transaction_costs=total_costs,
        )

        # BacktestResultに変換
        return self._convert_to_backtest_result(prices, sim_result)

    def _convert_prices_dict_to_df(
        self,
        prices: Dict[str, pd.DataFrame],
        universe: List[str],
    ) -> pd.DataFrame:
        """
        価格辞書をDataFrameに変換

        Parameters
        ----------
        prices : Dict[str, pd.DataFrame]
            価格データ（{symbol: DataFrame with Close column}）
        universe : List[str]
            ユニバース

        Returns
        -------
        pd.DataFrame
            価格DataFrame（列=シンボル）
        """
        result = {}
        common_index = None

        for symbol in universe:
            if symbol not in prices:
                logger.warning("Missing price data for %s", symbol)
                continue

            df = prices[symbol]
            if "Close" in df.columns:
                series = df["Close"]
            elif "close" in df.columns:
                series = df["close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = df.iloc[:, 0]

            result[symbol] = series

            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("No common dates found in price data")

        # 共通インデックスでDataFrameを作成
        price_df = pd.DataFrame({
            symbol: result[symbol].reindex(common_index)
            for symbol in result
        })

        return price_df

    def _convert_to_unified_result(
        self,
        result: BacktestResult,
        config: UnifiedBacktestConfig,
    ) -> UnifiedBacktestResult:
        """
        BacktestResultをUnifiedBacktestResultに変換

        Parameters
        ----------
        result : BacktestResult
            内部形式の結果
        config : UnifiedBacktestConfig
            設定

        Returns
        -------
        UnifiedBacktestResult
            共通形式の結果
        """
        # 日次リターンSeries
        daily_returns = pd.Series(
            [s.daily_return for s in result.snapshots],
            index=[s.date for s in result.snapshots],
        )

        # ポートフォリオ価値Series
        portfolio_values = pd.Series(
            [s.portfolio_value for s in result.snapshots],
            index=[s.date for s in result.snapshots],
        )

        # リバランス記録
        rebalances = []
        for i, snapshot in enumerate(result.snapshots):
            if i > 0:
                prev_weights = result.snapshots[i - 1].weights
                if snapshot.weights != prev_weights:
                    rebalances.append(BaseRebalanceRecord(
                        date=snapshot.date,
                        weights_before=prev_weights,
                        weights_after=snapshot.weights,
                        turnover=sum(
                            abs(snapshot.weights.get(k, 0) - prev_weights.get(k, 0))
                            for k in set(snapshot.weights) | set(prev_weights)
                        ) / 2,
                        transaction_cost=0.0,  # 個別取引コストは不明
                        portfolio_value=snapshot.portfolio_value,
                    ))

        unified_result = UnifiedBacktestResult(
            total_return=result.total_return,
            annual_return=getattr(result, "annualized_return", getattr(result, "annual_return", 0.0)),
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=getattr(result, "sortino_ratio", 0.0),
            max_drawdown=result.max_drawdown,
            volatility=result.volatility,
            calmar_ratio=getattr(result, "calmar_ratio", 0.0),
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            rebalances=rebalances,
            total_turnover=getattr(result, "total_turnover", getattr(result, "turnover", 0.0)),
            total_transaction_costs=result.transaction_costs,
            config=config,
            start_date=result.start_date,
            end_date=result.end_date,
            engine_name=self.ENGINE_NAME,
        )

        # メトリクスを計算
        unified_result.calculate_metrics(config.risk_free_rate)

        return unified_result

    def run(
        self,
        prices: pd.DataFrame | pl.DataFrame,
        asset_names: Optional[List[str]] = None,
        weights_func: Optional[callable] = None,
        vix_data: Optional[pd.DataFrame | pl.DataFrame] = None,
        dividend_data: Optional[Union[pd.DataFrame, pl.DataFrame, Dict[str, Any]]] = None,
    ) -> BacktestResult:
        """
        バックテストを実行（Polars/Pandas両対応）

        Parameters
        ----------
        prices : pd.DataFrame | pl.DataFrame
            価格データ。Polars: timestamp列 + アセット列、Pandas: インデックスはDatetime、列はアセット名。
        asset_names : List[str], optional
            アセット名。Noneの場合はpricesの列名を使用。
        weights_func : callable, optional
            ウェイト計算関数。シグネチャ: (signals, cov_matrix) -> weights
            Noneの場合は等ウェイト。
        vix_data : pd.DataFrame | pl.DataFrame, optional
            VIXデータ（close列を含む）。VIX動的キャッシュ配分に使用。
        dividend_data : pd.DataFrame | pl.DataFrame | Dict, optional
            配当データ。配当込みトータルリターン計算に使用。(SYNC-006)

        Returns
        -------
        BacktestResult
            バックテスト結果
        """
        # データ準備
        prices = self._prepare_prices(prices, asset_names)
        self._asset_names = list(prices.columns)
        self._n_assets = len(self._asset_names)

        # 共分散推定器を初期化
        self.cov_estimator = IncrementalCovarianceEstimator(
            n_assets=self._n_assets,
            halflife=self.config.cov_halflife,
            asset_names=self._asset_names,
        )

        # VIXデータを準備（SYNC-004）
        if vix_data is not None and self._vix_signal is not None:
            self._prepare_vix_data(vix_data, prices.index)
        else:
            self._vix_data = None
            self._vix_dates = None

        # 配当データを準備（SYNC-006）
        if dividend_data is not None and self._dividend_handler is not None:
            self._dividend_data = dividend_data
            logger.info("Dividend data provided for total return calculation")
        else:
            self._dividend_data = None

        # リバランス日を計算
        self._rebalance_dates = self._get_rebalance_dates(prices.index)

        logger.info(
            "Starting fast backtest: %d assets, %d days, %d rebalances",
            self._n_assets,
            len(prices),
            len(self._rebalance_dates),
        )

        # 高速シミュレーション実行
        sim_result = self._run_fast_simulation(
            prices=prices,
            weights_func=weights_func,
        )

        # BacktestResultに変換
        result = self._convert_to_backtest_result(prices, sim_result)

        return result

    def _prepare_prices(
        self,
        prices: pd.DataFrame | pl.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        価格データを準備（Polars/Pandas両対応）

        Parameters
        ----------
        prices : pd.DataFrame | pl.DataFrame
            価格データ（Polars or Pandas）
        asset_names : List[str], optional
            アセット名

        Returns
        -------
        pd.DataFrame
            準備済み価格データ（最終出力はPandas for NumPy compatibility）
        """
        # Convert Polars to Pandas if needed
        if isinstance(prices, pl.DataFrame):
            # Handle timestamp column
            if "timestamp" in prices.columns:
                prices_pd = prices.to_pandas()
                prices_pd = prices_pd.set_index("timestamp")
            else:
                prices_pd = prices.to_pandas()
                if prices_pd.index.name is None and prices_pd.columns[0] in ["index", "date", "Date"]:
                    prices_pd = prices_pd.set_index(prices_pd.columns[0])
            prices = prices_pd

        # インデックスをDatetimeに変換
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        # アセット名でフィルタ
        if asset_names is not None:
            available = [a for a in asset_names if a in prices.columns]
            prices = prices[available]

        # 期間でフィルタ
        mask = (prices.index >= self.config.start_date) & (
            prices.index <= self.config.end_date
        )
        prices = prices.loc[mask]

        # 欠損値を前方補完
        prices = prices.ffill().bfill()

        return prices

    def _prepare_vix_data(
        self,
        vix_data: pd.DataFrame | pl.DataFrame,
        price_dates: pd.DatetimeIndex,
    ) -> None:
        """
        VIXデータを準備（SYNC-004）

        Parameters
        ----------
        vix_data : pd.DataFrame | pl.DataFrame
            VIXデータ（close列を含む）
        price_dates : pd.DatetimeIndex
            価格データの日付インデックス
        """
        # Polars -> Pandas変換
        if isinstance(vix_data, pl.DataFrame):
            if "timestamp" in vix_data.columns:
                vix_df = vix_data.to_pandas().set_index("timestamp")
            else:
                vix_df = vix_data.to_pandas()
        else:
            vix_df = vix_data.copy()

        # インデックス処理
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            if "timestamp" in vix_df.columns:
                vix_df = vix_df.set_index("timestamp")
            elif "date" in vix_df.columns:
                vix_df = vix_df.set_index("date")
            vix_df.index = pd.to_datetime(vix_df.index)

        # close列を取得
        if "close" in vix_df.columns:
            vix_series = vix_df["close"]
        elif "Close" in vix_df.columns:
            vix_series = vix_df["Close"]
        else:
            logger.warning("VIX data has no 'close' column, disabling VIX cash allocation")
            self._vix_data = None
            self._vix_dates = None
            return

        # 価格データの日付にリインデックス
        vix_series = vix_series.reindex(price_dates, method="ffill")
        vix_series = vix_series.fillna(method="bfill")

        # 配列として保存
        self._vix_data = vix_series.values.astype(np.float64)
        self._vix_dates = list(price_dates)

        logger.info(
            "VIX data prepared: %d days, VIX range [%.1f, %.1f]",
            len(self._vix_data),
            np.nanmin(self._vix_data),
            np.nanmax(self._vix_data),
        )

    def _get_vix_cash_allocation(
        self,
        day_idx: int,
        prev_day_idx: int = -1,
    ) -> float:
        """
        指定日のVIXキャッシュ配分を取得（SYNC-004）

        Parameters
        ----------
        day_idx : int
            日付インデックス
        prev_day_idx : int
            前日のインデックス（変化率計算用）

        Returns
        -------
        float
            キャッシュ配分（0.0〜1.0）
        """
        if self._vix_signal is None or self._vix_data is None:
            return 0.0

        if day_idx < 0 or day_idx >= len(self._vix_data):
            return 0.0

        vix = self._vix_data[day_idx]
        if np.isnan(vix):
            return self._last_vix_cash_allocation

        # 日次変化率を計算
        vix_change = 0.0
        if prev_day_idx >= 0 and prev_day_idx < len(self._vix_data):
            prev_vix = self._vix_data[prev_day_idx]
            if not np.isnan(prev_vix) and prev_vix > 0:
                vix_change = (vix - prev_vix) / prev_vix

        # VIXシグナルからキャッシュ配分を取得
        result = self._vix_signal.get_cash_allocation(
            vix=vix,
            vix_change=vix_change,
        )

        self._last_vix_cash_allocation = result.cash_allocation
        return result.cash_allocation

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[datetime]:
        """
        リバランス日を計算

        Parameters
        ----------
        dates : pd.DatetimeIndex
            取引日

        Returns
        -------
        List[datetime]
            リバランス日のリスト
        """
        freq = self.config.rebalance_frequency.lower()
        rebalance_dates = []

        if freq == "daily":
            rebalance_dates = list(dates)
        elif freq == "weekly":
            # 週初めにリバランス
            for date in dates:
                if date.weekday() == 0:  # Monday
                    rebalance_dates.append(date)
            # 最初の日が月曜でない場合は追加
            if dates[0] not in rebalance_dates:
                rebalance_dates.insert(0, dates[0])
        elif freq == "monthly":
            # 月初めにリバランス
            current_month = None
            for date in dates:
                if current_month != (date.year, date.month):
                    rebalance_dates.append(date)
                    current_month = (date.year, date.month)
        elif freq == "quarterly":
            # 四半期初めにリバランス
            current_quarter = None
            for date in dates:
                quarter = (date.year, (date.month - 1) // 3)
                if current_quarter != quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
        else:
            raise ValueError(f"Unknown rebalance frequency: {freq}")

        return [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
                for d in rebalance_dates]

    def _run_fast_simulation(
        self,
        prices: pd.DataFrame,
        weights_func: Optional[callable] = None,
    ) -> SimulationResult:
        """
        高速シミュレーションを実行

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ
        weights_func : callable, optional
            ウェイト計算関数

        Returns
        -------
        SimulationResult
            シミュレーション結果
        """
        n_days = len(prices)
        n_assets = self._n_assets

        # 配列を事前確保
        portfolio_values = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))
        daily_returns = np.zeros(n_days)

        # 初期状態
        portfolio_values[0] = self.config.initial_capital
        current_weights = np.ones(n_assets) / n_assets  # 等ウェイトで開始
        weights_history[0] = current_weights

        # 価格をNumPy配列に変換
        price_matrix = prices.values
        dates = prices.index.tolist()

        # リバランス日をセットに変換（高速検索）
        rebalance_set = set(self._rebalance_dates)

        # 取引コスト
        cost_rate = self.config.transaction_cost_bps / 10000.0
        total_costs = 0.0
        actual_rebalance_dates = []

        for i in range(1, n_days):
            current_date = dates[i]
            prev_prices = price_matrix[i - 1]
            curr_prices = price_matrix[i]

            # 日次リターンを計算（価格リターン）
            price_returns = np.where(
                prev_prices > 0,
                curr_prices / prev_prices - 1,
                0.0
            )

            # 配当リターンを加算（SYNC-006）
            dividend_returns = self._get_dividend_returns(
                current_date, prev_prices, i
            )
            asset_returns = price_returns + dividend_returns

            # 共分散推定器を更新
            self.cov_estimator.update(asset_returns)

            # リバランス判定
            should_rebalance = current_date in rebalance_set

            if should_rebalance and i > 0:
                # レジーム検出（有効な場合）
                regime_params = self._detect_and_get_regime_params(prices, i)

                # 新しいウェイトを計算
                if weights_func is not None:
                    cov = self.cov_estimator.get_covariance()
                    signals = self._get_current_signals(current_date)
                    new_weights = weights_func(signals, cov)
                else:
                    # 等ウェイト
                    new_weights = np.ones(n_assets) / n_assets

                # レジームに応じたリスク調整
                if regime_params and self.config.regime_risk_scaling:
                    risk_budget = regime_params.get("risk_budget", 1.0)
                    cash_allocation = regime_params.get("cash_allocation", 0.0)
                    # ウェイトをリスクバジェットでスケール
                    new_weights = new_weights * risk_budget
                    # 残りをキャッシュに配分（ウェイト合計を1に保つ）
                    if cash_allocation > 0:
                        new_weights = new_weights * (1 - cash_allocation)

                # ウェイト制約を適用
                new_weights = self._apply_weight_constraints(new_weights)

                # TransactionCostOptimizer適用 (SYNC-001)
                if self.config.use_cost_optimizer and TRANSACTION_COST_OPTIMIZER_AVAILABLE:
                    new_weights = self._apply_cost_optimizer(
                        new_weights,
                        current_weights,
                        prices,
                        i,
                    )

                # DrawdownProtection適用 (SYNC-002)
                if self._dd_protector is not None:
                    new_weights = self._apply_dd_protection(new_weights)

                # VIXキャッシュ配分適用 (SYNC-004)
                if self._vix_signal is not None and self._vix_data is not None:
                    vix_cash = self._get_vix_cash_allocation(i, i - 1)
                    if vix_cash > 0:
                        # ウェイトをスケールダウンしてキャッシュに配分
                        new_weights = new_weights * (1.0 - vix_cash)
                        # ウェイト合計が1未満になる（キャッシュ分）

                # DynamicWeighter適用 (SYNC-005)
                if self._dynamic_weighter is not None:
                    new_weights = self._apply_dynamic_weighting(
                        new_weights,
                        prices,
                        i,
                        portfolio_values[i - 1],
                    )

                # 取引コストを計算
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = turnover * cost_rate * portfolio_values[i - 1]
                total_costs += cost

                current_weights = new_weights
                actual_rebalance_dates.append(current_date)

            # ポートフォリオリターンを計算
            portfolio_return = np.dot(current_weights, asset_returns)
            daily_returns[i] = portfolio_return

            # ポートフォリオ価値を更新
            portfolio_values[i] = portfolio_values[i - 1] * (1 + portfolio_return)

            # DrawdownProtector更新 (SYNC-002)
            if self._dd_protector is not None:
                self._dd_protector.update(portfolio_values[i])

            # ウェイト履歴を保存
            weights_history[i] = current_weights

        return SimulationResult(
            dates=dates,
            portfolio_values=portfolio_values,
            weights_history=weights_history,
            returns=daily_returns,
            rebalance_dates=actual_rebalance_dates,
            transaction_costs=total_costs,
        )

    def _detect_and_get_regime_params(
        self,
        prices: pd.DataFrame,
        current_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        レジームを検出してパラメータを取得

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ
        current_idx : int
            現在のインデックス

        Returns
        -------
        Optional[Dict[str, Any]]
            レジームパラメータ（レジーム検出無効時はNone）
        """
        if self._regime_detector is None:
            return None

        # ルックバック期間分のデータを取得
        lookback = self.config.regime_lookback
        start_idx = max(0, current_idx - lookback)

        # 代表銘柄（最初の銘柄）でレジーム検出
        price_series = prices.iloc[start_idx:current_idx + 1, 0]

        if len(price_series) < lookback // 2:
            return None

        try:
            result = self._regime_detector.detect_regime(price_series)
            self._current_regime = result.regime.value

            # レジームに応じたパラメータを取得
            if self.config.regime_adaptive_weights and RegimeAdaptiveStrategy:
                self._regime_params = RegimeAdaptiveStrategy.interpolate_params(
                    result.probabilities
                )
            else:
                self._regime_params = RegimeAdaptiveStrategy.get_params(
                    result.regime
                ) if RegimeAdaptiveStrategy else {}

            return self._regime_params

        except (ValueError, KeyError, RuntimeError, AttributeError) as e:
            logger.warning("Regime detection failed: %s", e)
            return None

    def get_current_regime(self) -> Optional[str]:
        """
        現在のレジームを取得

        Returns
        -------
        Optional[str]
            レジーム名（未検出時はNone）
        """
        return self._current_regime

    def get_regime_params(self) -> Dict[str, Any]:
        """
        現在のレジームパラメータを取得

        Returns
        -------
        Dict[str, Any]
            レジームパラメータ
        """
        return self._regime_params.copy()


    def _apply_dynamic_weighting(
        self,
        weights: np.ndarray,
        prices: pd.DataFrame,
        current_idx: int,
        portfolio_value: float,
    ) -> np.ndarray:
        """
        DynamicWeighterを適用してウェイトを動的に調整

        Parameters
        ----------
        weights : np.ndarray
            現在のウェイト
        prices : pd.DataFrame
            価格データ
        current_idx : int
            現在のインデックス
        portfolio_value : float
            現在のポートフォリオ価値

        Returns
        -------
        np.ndarray
            調整後のウェイト
        """
        if self._dynamic_weighter is None:
            return weights

        # ピーク値を更新
        self._portfolio_peak_value = max(self._portfolio_peak_value, portfolio_value)

        # ウェイトを辞書形式に変換
        weights_dict = {
            self._asset_names[j]: float(weights[j])
            for j in range(len(weights))
        }

        # 市場データを準備
        lookback = min(current_idx, self.config.regime_lookback)
        if lookback < 20:
            return weights

        price_matrix = prices.values
        returns = np.zeros(lookback)
        for j in range(lookback):
            idx = current_idx - lookback + j + 1
            if idx > 0:
                daily_ret = np.nanmean(
                    (price_matrix[idx] - price_matrix[idx - 1]) / price_matrix[idx - 1]
                )
                returns[j] = daily_ret if not np.isnan(daily_ret) else 0.0

        market_data = {
            "returns": pd.Series(returns),
            "portfolio_value": portfolio_value,
            "peak_value": self._portfolio_peak_value,
        }

        # レジーム情報を準備
        regime_info = None
        if self._current_regime:
            vol_regime = "medium"
            trend_regime = "range"
            if self._current_regime == "crisis":
                vol_regime = "high"
                trend_regime = "downtrend"
            elif self._current_regime == "high_vol":
                vol_regime = "high"
            elif self._current_regime == "low_vol":
                vol_regime = "low"
            elif self._current_regime == "trending":
                trend_regime = "uptrend"

            regime_info = {
                "current_vol_regime": vol_regime,
                "current_trend_regime": trend_regime,
            }

        # DynamicWeighterを適用
        try:
            adjusted_weights_dict = self._dynamic_weighter.adjust_weights(
                weights_dict,
                market_data,
                regime_info,
            )

            # 辞書から配列に変換
            adjusted_weights = np.array([
                adjusted_weights_dict.get(self._asset_names[j], 0.0)
                for j in range(len(weights))
            ])

            self._dynamic_weighting_result = {
                "original": weights_dict,
                "adjusted": adjusted_weights_dict,
                "portfolio_value": portfolio_value,
                "peak_value": self._portfolio_peak_value,
            }

            return adjusted_weights

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("DynamicWeighter failed: %s", e)
            return weights

    def get_dynamic_weighting_result(self) -> Optional[Dict[str, Any]]:
        """
        直近のDynamicWeighting結果を取得

        Returns
        -------
        Optional[Dict[str, Any]]
            DynamicWeighting結果
        """
        return self._dynamic_weighting_result

    def _get_current_signals(self, date: datetime) -> Dict[str, float]:
        """
        現在日のシグナルを取得

        Parameters
        ----------
        date : datetime
            現在日

        Returns
        -------
        Dict[str, float]
            アセット名 -> シグナル値
        """
        # SignalPrecomputerが利用可能な場合
        if self.signal_precomputer is not None:
            try:
                return self.signal_precomputer.get_signals_at_date(date)
            except (KeyError, ValueError, AttributeError):
                pass

        # IncrementalSignalEngineが利用可能な場合
        if self.incremental_engine is not None:
            try:
                return self.incremental_engine.get_all_signals()
            except (KeyError, ValueError, AttributeError):
                pass

        # デフォルト: 空のシグナル
        return {}

    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        ウェイト制約を適用

        Parameters
        ----------
        weights : np.ndarray
            元のウェイト

        Returns
        -------
        np.ndarray
            制約適用後のウェイト
        """
        # 最小・最大制約
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        # 正規化（合計を1に）
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(len(weights)) / len(weights)

        return weights

    def _apply_cost_optimizer(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        prices: pd.DataFrame,
        current_idx: int,
        lookback: int = 60,
    ) -> np.ndarray:
        """
        TransactionCostOptimizerを適用して取引コスト考慮のウェイトを計算 (SYNC-001)

        期待リターン vs 取引コストのトレードオフを最適化。
        U = μ'w - (λ/2)*w'Σw - κ*TC(w, w_current)

        Parameters
        ----------
        target_weights : np.ndarray
            目標ウェイト（コスト考慮前）
        current_weights : np.ndarray
            現在のウェイト
        prices : pd.DataFrame
            価格データ
        current_idx : int
            現在のインデックス
        lookback : int
            期待リターン/共分散計算のルックバック期間

        Returns
        -------
        np.ndarray
            コスト最適化後のウェイト
        """
        if not TRANSACTION_COST_OPTIMIZER_AVAILABLE:
            return target_weights

        n_assets = len(target_weights)
        asset_names = list(prices.columns)

        # ルックバック期間の制限
        start_idx = max(0, current_idx - lookback)
        if current_idx - start_idx < 10:
            return target_weights

        # リターンをDataFrameとして計算
        price_slice = prices.iloc[start_idx:current_idx + 1]
        returns_df = price_slice.pct_change().dropna()

        if len(returns_df) < 5:
            return target_weights

        # ウェイトをDict形式に変換
        current_weights_dict = {
            asset_names[i]: float(current_weights[i])
            for i in range(n_assets)
        }
        target_weights_dict = {
            asset_names[i]: float(target_weights[i])
            for i in range(n_assets)
        }

        try:
            # TurnoverConstrainedOptimizer: ターンオーバー制約内でターゲットに近づける
            optimizer = TurnoverConstrainedOptimizer(
                max_turnover=self.config.max_turnover,
                max_weight=self.config.max_weight,
            )
            result = optimizer.optimize(
                returns=returns_df,
                current_weights=current_weights_dict,
                target_weights=target_weights_dict,
            )

            if result.converged:
                # Dict形式からnp.array形式に変換
                optimized_weights = np.array([
                    result.optimal_weights.get(asset, 0.0)
                    for asset in asset_names
                ])
                # 正規化
                total = np.sum(optimized_weights)
                if total > 0:
                    optimized_weights = optimized_weights / total
                logger.debug("Cost optimizer: turnover=%.4f", result.turnover)
                return optimized_weights
            else:
                logger.warning("Cost optimizer did not converge")
                return target_weights

        except Exception as e:
            logger.warning("Cost optimizer error: %s", e)
            return target_weights

    def _get_dividend_returns(
        self,
        current_date: datetime,
        prev_prices: np.ndarray,
        day_idx: int,
    ) -> np.ndarray:
        """
        指定日の配当リターンを取得 (SYNC-006)

        Parameters
        ----------
        current_date : datetime
            現在日
        prev_prices : np.ndarray
            前日の価格
        day_idx : int
            日付インデックス

        Returns
        -------
        np.ndarray
            配当リターン（配当なし時は0）
        """
        if self._dividend_handler is None or self._dividend_data is None:
            return np.zeros(len(prev_prices))

        try:
            n_assets = len(prev_prices)
            dividend_returns = np.zeros(n_assets)

            # 配当データの形式に応じて処理
            if isinstance(self._dividend_data, dict):
                # Dict[str, DividendData] 形式
                for asset_idx, (symbol, div_data) in enumerate(self._dividend_data.items()):
                    if asset_idx >= n_assets:
                        break
                    # 当日が配落日かチェック
                    for ex_date, amount in zip(div_data.ex_dates, div_data.amounts):
                        if self._is_same_date(current_date, ex_date):
                            if prev_prices[asset_idx] > 0:
                                # 税引後配当
                                net_amount = amount * (1 - self.config.withholding_tax_rate)
                                dividend_returns[asset_idx] = net_amount / prev_prices[asset_idx]

            elif isinstance(self._dividend_data, (pd.DataFrame, pl.DataFrame)):
                # DataFrame形式
                if isinstance(self._dividend_data, pl.DataFrame):
                    div_df = self._dividend_data.to_pandas()
                else:
                    div_df = self._dividend_data

                # 日付列と配当額列を特定
                date_col = self._find_column(div_df, ["ex_date", "date", "Date", "timestamp"])
                amount_col = self._find_column(div_df, ["amount", "dividend_amount", "dividend"])

                if date_col and amount_col:
                    div_df[date_col] = pd.to_datetime(div_df[date_col])

                    # 当日の配当を検索
                    current_date_normalized = pd.Timestamp(current_date).normalize()
                    day_divs = div_df[div_df[date_col].dt.normalize() == current_date_normalized]

                    for _, row in day_divs.iterrows():
                        amount = row[amount_col]
                        # symbol列がある場合はアセット別
                        if "symbol" in div_df.columns:
                            symbol = row["symbol"]
                            if symbol in self._asset_names:
                                asset_idx = self._asset_names.index(symbol)
                                if prev_prices[asset_idx] > 0:
                                    net_amount = amount * (1 - self.config.withholding_tax_rate)
                                    dividend_returns[asset_idx] = net_amount / prev_prices[asset_idx]
                        else:
                            # 全アセットに同一配当（ETFなど）
                            for asset_idx in range(n_assets):
                                if prev_prices[asset_idx] > 0:
                                    net_amount = amount * (1 - self.config.withholding_tax_rate)
                                    dividend_returns[asset_idx] = net_amount / prev_prices[asset_idx]

            return dividend_returns

        except Exception as e:
            logger.debug("Dividend return calculation error: %s", e)
            return np.zeros(len(prev_prices))

    def _is_same_date(self, date1: datetime, date2: datetime) -> bool:
        """日付が同じかチェック"""
        if hasattr(date1, 'date'):
            d1 = date1.date()
        else:
            d1 = date1
        if hasattr(date2, 'date'):
            d2 = date2.date()
        else:
            d2 = date2
        return d1 == d2

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """DataFrameから候補列を検索"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _apply_dd_protection(self, weights: np.ndarray) -> np.ndarray:
        """
        DrawdownProtectorを適用してリスク削減ウェイトを計算 (SYNC-002)

        Parameters
        ----------
        weights : np.ndarray
            元のウェイト

        Returns
        -------
        np.ndarray
            ドローダウン保護適用後のウェイト
        """
        if self._dd_protector is None:
            return weights

        # リスク乗数を取得（0.0〜1.0、DDが深いほど小さい値）
        risk_multiplier = self._dd_protector.get_risk_multiplier()

        if risk_multiplier >= 1.0:
            # 通常時はそのまま
            return weights

        # ウェイトをリスク乗数でスケール
        adjusted_weights = weights * risk_multiplier

        total = np.sum(adjusted_weights)
        if total < 1.0:
            logger.debug(
                "DD protection: multiplier=%.2f, cash_ratio=%.2f",
                risk_multiplier, 1.0 - total
            )
        elif total > 1.0:
            # 合計が1を超える場合は正規化
            adjusted_weights = adjusted_weights / total

        return adjusted_weights

    def _convert_to_backtest_result(
        self,
        prices: pd.DataFrame,
        sim_result: SimulationResult,
    ) -> BacktestResult:
        """
        SimulationResultをBacktestResultに変換

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ
        sim_result : SimulationResult
            シミュレーション結果

        Returns
        -------
        BacktestResult
            バックテスト結果
        """
        # スナップショットを作成
        snapshots = []
        cumulative_return = 0.0

        for i, date in enumerate(sim_result.dates):
            if i > 0:
                cumulative_return = (
                    sim_result.portfolio_values[i] / self.config.initial_capital - 1
                )

            weights_dict = {
                self._asset_names[j]: float(sim_result.weights_history[i, j])
                for j in range(self._n_assets)
            }

            snapshot = DailySnapshot(
                date=date,
                weights=weights_dict,
                portfolio_value=float(sim_result.portfolio_values[i]),
                daily_return=float(sim_result.returns[i]),
                cumulative_return=cumulative_return,
                cash_weight=0.0,
            )
            snapshots.append(snapshot)

        # BacktestResultを作成
        result = BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            rebalance_frequency=self.config.rebalance_frequency,
            initial_capital=self.config.initial_capital,
            snapshots=snapshots,
            transaction_costs=sim_result.transaction_costs,
            num_rebalances=len(sim_result.rebalance_dates),
        )

        # メトリクスを計算
        result.calculate_metrics()

        return result


# ショートカット関数
def run_fast_backtest(
    prices: pd.DataFrame | pl.DataFrame,
    start_date: datetime,
    end_date: datetime,
    rebalance_frequency: str = "monthly",
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 10.0,
    weights_func: Optional[callable] = None,
    use_numba: bool = True,
    use_gpu: bool = False,
    vix_cash_enabled: bool = True,
    vix_data: Optional[pd.DataFrame | pl.DataFrame] = None,
) -> BacktestResult:
    """
    高速バックテストを実行するショートカット関数（Polars/Pandas両対応）

    Parameters
    ----------
    prices : pd.DataFrame | pl.DataFrame
        価格データ（Polars or Pandas）
    start_date : datetime
        開始日
    end_date : datetime
        終了日
    rebalance_frequency : str
        リバランス頻度
    initial_capital : float
        初期資金
    transaction_cost_bps : float
        取引コスト（bps）
    weights_func : callable, optional
        ウェイト計算関数
    use_numba : bool
        Numba JIT高速化を使用（デフォルトTrue）
    use_gpu : bool
        GPU計算を使用（デフォルトFalse）
    vix_cash_enabled : bool
        VIX動的キャッシュ配分を有効化（デフォルトTrue）
    vix_data : pd.DataFrame | pl.DataFrame, optional
        VIXデータ（close列を含む）

    Returns
    -------
    BacktestResult
        バックテスト結果
    """
    config = FastBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        use_numba=use_numba,
        use_gpu=use_gpu,
        vix_cash_enabled=vix_cash_enabled,
    )

    engine = FastBacktestEngine(config)
    return engine.run(prices, weights_func=weights_func, vix_data=vix_data)
