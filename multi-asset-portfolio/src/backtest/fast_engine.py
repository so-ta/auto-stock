"""
Backtest Engine - Numba JIT高速化バックテストエンジン（1137x高速化）

事前計算シグナル、インクリメンタル共分散、ベクトル化計算、Numba JITを組み合わせた
高性能バックテストエンジン。

主な高速化ポイント:
- シグナル: 事前計算済みをメモリマップで読み込み
- 共分散: インクリメンタル更新（毎回再計算しない）
- ポートフォリオ計算: ベクトル化
- メインループ: NumPy配列で効率的に計算
- Numba: @njit(parallel=True) でJITコンパイル高速化（デフォルト有効）
- GPU: CuPy対応時にGPUで行列演算（オプション）

使用例:
    from src.backtest.fast_engine import BacktestEngine, BacktestConfig
    from datetime import datetime

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        rebalance_frequency="monthly",
        initial_capital=100000.0,
    )

    engine = BacktestEngine(config)
    result = engine.run(prices_df, asset_names=["SPY", "QQQ", "TLT", "GLD"])

後方互換性:
    FastBacktestEngine, FastBacktestConfig は非推奨エイリアスとして残存。
    新規コードでは BacktestEngine, BacktestConfig を使用すること。
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
    from src.allocation.transaction_cost import TransactionCostSchedule
    from src.data.asset_master import AssetMaster

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
class BacktestConfig:
    """バックテスト設定（Numba JIT高速化対応）

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
        use_numba: Numba JIT高速化を使用（デフォルト: True）
        use_gpu: GPU計算を使用（デフォルト: False、CuPy必要）
        numba_parallel: Numba並列化を有効化（デフォルト: True）
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
    signal_cache_dir: Optional[str] = None  # Set in __post_init__ from Settings
    cov_cache_dir: Optional[str] = None  # Set in __post_init__ from Settings
    min_weight: float = 0.0
    max_weight: float = 1.0
    # Numba/GPU設定
    # GPU計算を有効にするには:
    #   1. CuPyをインストール: pip install cupy-cuda12x (CUDA 12.x) or cupy-cuda11x (CUDA 11.x)
    #   2. use_gpu=True を設定
    #   3. scripts/check_gpu.py で環境確認可能
    # ResourceConfigからの自動検出: BacktestConfig.from_resource_config() を使用
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
    # カテゴリ別取引コスト設定
    cost_schedule: Optional["TransactionCostSchedule"] = None
    asset_master: Optional["AssetMaster"] = None

    def __post_init__(self):
        """Initialize cache paths from Settings if not provided."""
        from src.config.settings import get_cache_path

        if self.signal_cache_dir is None:
            self.signal_cache_dir = get_cache_path("signals")
        if self.cov_cache_dir is None:
            self.cov_cache_dir = get_cache_path("covariance")

    @classmethod
    def from_resource_config(
        cls,
        start_date: datetime,
        end_date: datetime,
        **kwargs,
    ) -> "BacktestConfig":
        """
        ResourceConfigからインスタンスを生成

        システムリソースに基づいた最適な設定でBacktestConfigを作成する。
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
        BacktestConfig
            ResourceConfigベースの設定

        Example
        -------
        >>> config = BacktestConfig.from_resource_config(
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

    WeightsFuncProtocol準拠の外部関数を、BacktestEngine内部の
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


class BacktestEngine(BacktestEngineBase):
    """
    Numba JIT高速化バックテストエンジン（1137x高速化）

    以下のモジュールを統合した高性能バックテストエンジン:
    - SignalPrecomputer: 事前計算シグナル
    - IncrementalCovarianceEstimator: インクリメンタル共分散
    - IncrementalSignalEngine: インクリメンタルシグナル（オプション）

    高速化機能:
    - use_numba=True（デフォルト）: Numba JITで高速化
    - use_gpu=True（オプション）: GPU(CuPy)で高速行列演算
    - フォールバック: NumPy

    INT-003: BacktestEngineBase準拠（共通インターフェース）
    - run()とvalidate_inputs()を実装
    - BacktestConfigとUnifiedBacktestConfigの両方に対応

    Usage:
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        engine = BacktestEngine(config)
        result = engine.run(prices_df)
    """

    # エンジン名（BacktestEngineBase準拠）
    ENGINE_NAME: str = "BacktestEngine"

    def __init__(
        self,
        config: Union[BacktestConfig, UnifiedBacktestConfig],
        signal_precomputer: Optional[Any] = None,
        incremental_engine: Optional[Any] = None,
    ):
        """
        初期化

        Parameters
        ----------
        config : BacktestConfig | UnifiedBacktestConfig
            バックテスト設定（両方の設定形式に対応）
        signal_precomputer : SignalPrecomputer, optional
            シグナル事前計算器（Noneの場合は新規作成）
        incremental_engine : IncrementalSignalEngine, optional
            インクリメンタルシグナルエンジン
        """
        # UnifiedBacktestConfigの場合はBacktestConfigに変換
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
        if fast_config.use_regime_detection and REGIME_DETECTOR_AVAILABLE:
            self._regime_detector = EnhancedRegimeDetector({
                "lookback": fast_config.regime_lookback,
            })
            logger.info("RegimeDetector enabled with lookback=%d", fast_config.regime_lookback)

        # DrawdownProtector（SYNC-002）
        self._dd_protector: Optional[Any] = None
        if fast_config.use_drawdown_protection and DRAWDOWN_PROTECTION_AVAILABLE:
            dd_config = DrawdownProtectorConfig(
                dd_levels=fast_config.dd_levels,
                risk_reductions=fast_config.dd_risk_reductions,
                recovery_threshold=fast_config.dd_recovery_threshold,
                emergency_dd_level=fast_config.dd_emergency_level,
                emergency_cash_ratio=fast_config.dd_emergency_cash_ratio,
            )
            self._dd_protector = DrawdownProtector(dd_config, initial_value=fast_config.initial_capital)
            logger.info(
                "DrawdownProtector enabled: levels=%s, reductions=%s",
                fast_config.dd_levels, fast_config.dd_risk_reductions,
            )

        # VIXシグナル（SYNC-004）
        self._vix_signal: Optional[Any] = None
        self._vix_data: Optional[np.ndarray] = None
        self._vix_dates: Optional[List[datetime]] = None
        self._last_vix_cash_allocation: float = 0.0
        if fast_config.vix_cash_enabled and VIX_SIGNAL_AVAILABLE:
            if fast_config.vix_config_path:
                self._vix_signal = EnhancedVIXSignal.from_config(fast_config.vix_config_path)
            else:
                self._vix_signal = EnhancedVIXSignal()
            logger.info("VIX dynamic cash allocation enabled")

        # DynamicWeighter（SYNC-005）
        self._dynamic_weighter: Optional[Any] = None
        self._dynamic_weighting_result: Optional[Dict[str, Any]] = None
        self._portfolio_peak_value: float = fast_config.initial_capital
        if fast_config.use_dynamic_weighting and DYNAMIC_WEIGHTER_AVAILABLE:
            dw_config = DynamicWeightingConfig(
                target_volatility=fast_config.target_volatility,
                max_drawdown_trigger=fast_config.dd_protection_threshold,
                vol_scaling_enabled=fast_config.vol_scaling_enabled,
                dd_protection_enabled=fast_config.dd_protection_enabled,
                regime_weighting_enabled=fast_config.use_regime_detection,
            )
            self._dynamic_weighter = DynamicWeighter(dw_config)
            logger.info(
                "DynamicWeighter enabled: target_vol=%.2f, dd_trigger=%.2f",
                fast_config.target_volatility, fast_config.dd_protection_threshold,
            )

        # DividendHandler（SYNC-006）
        self._dividend_handler: Optional[Any] = None
        self._dividend_data: Optional[Dict[str, Any]] = None
        if fast_config.use_dividends and DIVIDEND_HANDLER_AVAILABLE:
            div_config = DividendConfig(
                reinvest_dividends=fast_config.reinvest_dividends,
                withholding_tax_rate=fast_config.withholding_tax_rate,
            )
            self._dividend_handler = DividendHandler(div_config)
            logger.info(
                "DividendHandler enabled: reinvest=%s, tax_rate=%.2f%%",
                fast_config.reinvest_dividends, fast_config.withholding_tax_rate * 100,
            )

        logger.info(
            "BacktestEngine initialized: %s to %s, freq=%s, backend=%s, regime=%s, dd=%s, vix=%s, dw=%s, div=%s",
            fast_config.start_date,
            fast_config.end_date,
            fast_config.rebalance_frequency,
            self._compute_backend,
            "enabled" if self._regime_detector else "disabled",
            "enabled" if self._dd_protector else "disabled",
            "enabled" if self._vix_signal else "disabled",
            "enabled" if self._dynamic_weighter else "disabled",
            "enabled" if self._dividend_handler else "disabled",
        )

    @staticmethod
    def _convert_from_unified_config(config: UnifiedBacktestConfig) -> "BacktestConfig":
        """
        UnifiedBacktestConfigからBacktestConfigに変換（INT-003）

        Parameters
        ----------
        config : UnifiedBacktestConfig
            共通設定

        Returns
        -------
        BacktestConfig
            エンジン固有設定
        """
        # engine_specific_configからエンジン固有設定を取得
        specific = config.engine_specific_config or {}

        return BacktestConfig(
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
            # カテゴリ別取引コスト設定
            cost_schedule=specific.get("cost_schedule"),
            asset_master=specific.get("asset_master"),
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

        # BacktestEngine固有の検証
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

    def _calculate_per_symbol_costs(
        self,
        asset_names: List[str],
        new_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """カテゴリ別コストで総取引コストを計算

        各銘柄のカテゴリに応じた取引コストを適用して、
        リバランスの総取引コストを計算する。

        Args:
            asset_names: 銘柄名リスト
            new_weights: 新しいウェイト配列
            current_weights: 現在のウェイト配列
            portfolio_value: ポートフォリオ総額

        Returns:
            float: 総取引コスト（金額）
        """
        from src.allocation.transaction_cost import (
            TransactionCostSchedule,
            load_cost_schedule,
        )
        from src.data.asset_master import load_asset_master

        # cost_scheduleが未設定なら自動ロード
        cost_schedule = self.config.cost_schedule
        if cost_schedule is None:
            cost_schedule = load_cost_schedule()

        # asset_masterが未設定なら自動ロード
        asset_master = self.config.asset_master
        if asset_master is None:
            asset_master = load_asset_master()

        # スリッページも加算（cost_scheduleは手数料のみなので）
        slippage_rate = self.config.slippage_bps / 10000.0

        total_cost = 0.0
        for i, symbol in enumerate(asset_names):
            weight_change = abs(new_weights[i] - current_weights[i])
            if weight_change < 1e-8:
                continue

            # カテゴリを取得
            category = None
            if asset_master is not None:
                info = asset_master.get(symbol)
                if info and info.category:
                    category = info.category

            # カテゴリ別コスト計算（ウェイトベースのコスト率を返す）
            cost_rate = cost_schedule.calculate_cost(
                symbol=symbol,
                weight_change=weight_change,
                portfolio_value=portfolio_value,
                category=category,
            )

            # スリッページを加算
            slippage_cost = weight_change * slippage_rate

            # 総コスト = (カテゴリ別コスト + スリッページ) × ポートフォリオ価値
            total_cost += (cost_rate + slippage_cost) * portfolio_value

        return total_cost

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
            # BacktestConfigからUnifiedBacktestConfigを作成
            config = UnifiedBacktestConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=self.config.initial_capital,
                rebalance_frequency=self.config.rebalance_frequency,
                transaction_cost_bps=self.config.transaction_cost_bps,
            )

        # 入力検証
        self.validate_inputs(universe, prices, config)

        # prices DictをDataFrameに変換（Close と Open）
        close_df, open_df = self._convert_prices_dict_to_df(prices, universe)

        # weights_funcをアダプター経由で変換
        # WeightsFuncAdapter: 状態を保持し、外部形式→内部形式の変換を行う
        adapter = None
        if weights_func is not None:
            adapter = _WeightsFuncAdapter(weights_func, universe, prices)

        # 内部run()を呼び出し（アダプター付き）
        result = self._run_with_weights_adapter(close_df, open_df, adapter, universe)

        # BacktestResultをUnifiedBacktestResultに変換
        return self._convert_to_unified_result(result, config)

    def _run_with_weights_adapter(
        self,
        close_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
        adapter: Optional["_WeightsFuncAdapter"],
        universe: List[str],
    ) -> "BacktestResult":
        """
        weights_funcアダプター付きでバックテストを実行（翌日初値執行対応）

        外部weights_func（WeightsFuncProtocol準拠）を使用する場合、
        各リバランス時に正確な日付と現在ウェイトをアダプターに渡す。

        処理フロー:
        - Day T (リバランス判定日):
          - Close[T] でシグナル計算
          - 新しいウェイトを計算
          - pending_rebalance = True（翌日執行予約）
        - Day T+1 (執行日):
          - Open[T+1] で売買執行
          - オーバーナイトリターン: Open[T+1] / Close[T] - 1
          - pending_rebalance = False

        Parameters
        ----------
        close_prices : pd.DataFrame
            終値データ
        open_prices : pd.DataFrame
            始値データ
        adapter : _WeightsFuncAdapter, optional
            weights_funcアダプター
        universe : List[str]
            ユニバース

        Returns
        -------
        BacktestResult
            バックテスト結果
        """
        # データ準備（Close価格ベースで期間フィルタなど）
        close_prices = self._prepare_prices(close_prices, None)
        # Open価格も同じインデックスに合わせる
        open_prices = open_prices.reindex(close_prices.index)

        self._asset_names = list(close_prices.columns)
        self._n_assets = len(self._asset_names)

        # 共分散推定器を初期化
        self.cov_estimator = IncrementalCovarianceEstimator(
            n_assets=self._n_assets,
            halflife=self.config.cov_halflife,
            asset_names=self._asset_names,
        )

        # リバランス日を計算
        self._rebalance_dates = self._get_rebalance_dates(close_prices.index)

        n_days = len(close_prices)
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
        close_matrix = close_prices.values
        open_matrix = open_prices.values
        dates = close_prices.index.tolist()

        # リバランス日をセットに変換
        rebalance_set = set(self._rebalance_dates)

        # 取引コスト設定
        # cost_scheduleが設定されている場合はカテゴリ別コスト、そうでなければ一律コスト
        use_category_costs = self.config.cost_schedule is not None or self.config.asset_master is not None
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0
        total_costs = 0.0
        actual_rebalance_dates = []

        # 保留リバランス状態（翌日初値執行用）
        pending_rebalance = False
        pending_weights = None
        cost = 0.0  # 当日の取引コスト

        for i in range(1, n_days):
            current_date = dates[i]
            prev_close = close_matrix[i - 1]
            curr_close = close_matrix[i]
            curr_open = open_matrix[i]
            cost = 0.0  # 当日の取引コストをリセット

            # Step 1: 保留中のリバランスを今日の初値で執行
            if pending_rebalance and pending_weights is not None:
                # Openが NaN の場合は Close を使用
                execution_prices = np.where(
                    np.isnan(curr_open) | (curr_open <= 0),
                    curr_close,
                    curr_open
                )

                # オーバーナイトリターン（前日Close → 当日Open）
                valid_mask = (prev_close > 0) & ~np.isnan(execution_prices)
                overnight_returns = np.where(
                    valid_mask,
                    execution_prices / prev_close - 1,
                    0.0
                )

                # オーバーナイトリターンを現在ウェイトで計算
                overnight_portfolio_return = np.dot(current_weights, overnight_returns)

                # 寄り付き時点のポートフォリオ価値
                open_value = portfolio_values[i - 1] * (1 + overnight_portfolio_return)

                # 売買コスト計算・執行
                if use_category_costs:
                    cost = self._calculate_per_symbol_costs(
                        self._asset_names, pending_weights, current_weights, open_value
                    )
                else:
                    turnover = np.sum(np.abs(pending_weights - current_weights))
                    cost = turnover * cost_rate * open_value
                total_costs += cost

                # リバランス執行日: 正確な2段階計算
                # オーバーナイトリターン(旧ウェイト) + イントラデイリターン(新ウェイト)

                # イントラデイリターン計算（Open[i] → Close[i]）
                valid_intraday = (curr_open > 0) & ~np.isnan(curr_close) & ~np.isnan(curr_open)
                intraday_returns = np.where(
                    valid_intraday,
                    curr_close / curr_open - 1,
                    0.0
                )

                # ウェイトを更新（イントラデイリターン計算後）
                current_weights = pending_weights
                actual_rebalance_dates.append(current_date)  # 執行日を記録

                # イントラデイポートフォリオリターン（新ウェイトで）
                intraday_portfolio_return = np.dot(current_weights, intraday_returns)

                # ポートフォリオ価値計算: Open価値 × (1 + イントラデイリターン) - コスト
                portfolio_values[i] = open_value * (1 + intraday_portfolio_return) - cost

                # 日次リターンを記録（終値ベース）
                if portfolio_values[i - 1] > 0:
                    daily_returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1
                else:
                    daily_returns[i] = 0.0

                # 保留状態をリセット
                pending_rebalance = False
                pending_weights = None

                # 共分散推定器を更新（Close-to-Closeリターンで）
                valid_mask = (prev_close > 0) & ~np.isnan(curr_close)
                price_returns = np.where(valid_mask, curr_close / prev_close - 1, 0.0)
                self.cov_estimator.update(price_returns)
            else:
                # 非リバランス日: 通常のClose-to-Close計算
                # Step 2: 日次リターン計算（Close to Close）
                valid_mask = (prev_close > 0) & ~np.isnan(curr_close)
                price_returns = np.where(
                    valid_mask,
                    curr_close / prev_close - 1,
                    0.0
                )
                asset_returns = price_returns

                # 共分散推定器を更新
                self.cov_estimator.update(asset_returns)

                # ポートフォリオリターンを計算
                portfolio_return = np.dot(current_weights, asset_returns)
                daily_returns[i] = portfolio_return

                # ポートフォリオ価値を更新
                portfolio_values[i] = portfolio_values[i - 1] * (1 + portfolio_return)

            # Step 3: リバランスシグナル確認（翌日執行を予約）
            should_rebalance = current_date in rebalance_set

            if should_rebalance and i < n_days - 1:  # 最終日はリバランスしない（翌日がない）
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

                # 翌日執行を予約
                pending_rebalance = True
                pending_weights = new_weights

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
        return self._convert_to_backtest_result(close_prices, sim_result)

    def _convert_prices_dict_to_df(
        self,
        prices: Dict[str, pd.DataFrame],
        universe: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        価格辞書をDataFrameに変換（Close と Open の両方を抽出）

        途中上場/廃止銘柄をサポートするため、全銘柄の日付をunionで結合し、
        データがない期間はNaNとする。

        Parameters
        ----------
        prices : Dict[str, pd.DataFrame]
            価格データ（{symbol: DataFrame with Close/Open columns}）
        universe : List[str]
            ユニバース

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (close_df, open_df) - 価格DataFrame（列=シンボル）、データがない期間はNaN
        """
        close_result = {}
        open_result = {}
        union_index = None

        for symbol in universe:
            if symbol not in prices:
                logger.warning("Missing price data for %s", symbol)
                continue

            df = prices[symbol]

            # Close価格を抽出
            if "Close" in df.columns:
                close_series = df["Close"]
            elif "close" in df.columns:
                close_series = df["close"]
            elif "Adj Close" in df.columns:
                close_series = df["Adj Close"]
            else:
                close_series = df.iloc[:, 0]

            close_result[symbol] = close_series

            # Open価格を抽出（なければCloseをフォールバック）
            if "Open" in df.columns:
                open_series = df["Open"]
            elif "open" in df.columns:
                open_series = df["open"]
            else:
                # Open価格がない場合はClose価格を使用
                logger.debug("No Open price for %s, using Close as fallback", symbol)
                open_series = close_series

            open_result[symbol] = open_series

            if union_index is None:
                union_index = close_series.index
            else:
                # intersection → union に変更（途中上場/廃止銘柄サポート）
                union_index = union_index.union(close_series.index)

        if union_index is None or len(union_index) == 0:
            raise ValueError("No dates found in price data")

        # unionインデックスでDataFrameを作成（データがない期間はNaN）
        close_df = pd.DataFrame({
            symbol: close_result[symbol].reindex(union_index)
            for symbol in close_result
        })
        open_df = pd.DataFrame({
            symbol: open_result[symbol].reindex(union_index)
            for symbol in open_result
        })

        # 日付順にソート
        close_df = close_df.sort_index()
        open_df = open_df.sort_index()

        return close_df, open_df

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
        use_category_costs = self.config.cost_schedule is not None or self.config.asset_master is not None
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0
        for i, snapshot in enumerate(result.snapshots):
            if i > 0:
                prev_weights = result.snapshots[i - 1].weights
                if snapshot.weights != prev_weights:
                    # ウェイト変化量を計算
                    all_symbols = list(set(snapshot.weights) | set(prev_weights))
                    turnover = sum(
                        abs(snapshot.weights.get(k, 0) - prev_weights.get(k, 0))
                        for k in all_symbols
                    ) / 2

                    # 取引コスト計算
                    if use_category_costs:
                        # カテゴリ別コストを使用
                        new_w = np.array([snapshot.weights.get(s, 0.0) for s in all_symbols])
                        prev_w = np.array([prev_weights.get(s, 0.0) for s in all_symbols])
                        transaction_cost = self._calculate_per_symbol_costs(
                            all_symbols, new_w, prev_w, snapshot.portfolio_value
                        )
                    else:
                        # 一律コスト
                        transaction_cost = turnover * cost_rate * snapshot.portfolio_value

                    rebalances.append(BaseRebalanceRecord(
                        date=snapshot.date,
                        weights_before=prev_weights,
                        weights_after=snapshot.weights,
                        turnover=turnover,
                        transaction_cost=transaction_cost,
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
        open_prices: Optional[pd.DataFrame | pl.DataFrame] = None,
    ) -> BacktestResult:
        """
        バックテストを実行（Polars/Pandas両対応、翌日初値執行対応）

        Parameters
        ----------
        prices : pd.DataFrame | pl.DataFrame
            終値データ。Polars: timestamp列 + アセット列、Pandas: インデックスはDatetime、列はアセット名。
        asset_names : List[str], optional
            アセット名。Noneの場合はpricesの列名を使用。
        weights_func : callable, optional
            ウェイト計算関数。シグネチャ: (signals, cov_matrix) -> weights
            Noneの場合は等ウェイト。
        vix_data : pd.DataFrame | pl.DataFrame, optional
            VIXデータ（close列を含む）。VIX動的キャッシュ配分に使用。
        dividend_data : pd.DataFrame | pl.DataFrame | Dict, optional
            配当データ。配当込みトータルリターン計算に使用。(SYNC-006)
        open_prices : pd.DataFrame | pl.DataFrame, optional
            始値データ（翌日初値執行用）。Noneの場合は終値を使用。

        Returns
        -------
        BacktestResult
            バックテスト結果
        """
        # データ準備（Close価格）
        close_prices = self._prepare_prices(prices, asset_names)
        self._asset_names = list(close_prices.columns)
        self._n_assets = len(self._asset_names)

        # Open価格を準備（翌日初値執行用）
        if open_prices is not None:
            open_prices_df = self._prepare_prices(open_prices, asset_names)
            # Close価格と同じインデックスに揃える
            open_prices_df = open_prices_df.reindex(close_prices.index)
        else:
            # Open価格がない場合はClose価格を使用（フォールバック）
            logger.debug("No open_prices provided, using close_prices for execution")
            open_prices_df = close_prices.copy()

        # 共分散推定器を初期化
        self.cov_estimator = IncrementalCovarianceEstimator(
            n_assets=self._n_assets,
            halflife=self.config.cov_halflife,
            asset_names=self._asset_names,
        )

        # VIXデータを準備（SYNC-004）
        if vix_data is not None and self._vix_signal is not None:
            self._prepare_vix_data(vix_data, close_prices.index)
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
        self._rebalance_dates = self._get_rebalance_dates(close_prices.index)

        logger.info(
            "Starting fast backtest: %d assets, %d days, %d rebalances (next-day open execution)",
            self._n_assets,
            len(close_prices),
            len(self._rebalance_dates),
        )

        # 部分期間カバレッジのアセットをログに出力
        self._log_partial_coverage_assets()

        # 高速シミュレーション実行（翌日初値執行）
        sim_result = self._run_fast_simulation(
            close_prices=close_prices,
            open_prices=open_prices_df,
            weights_func=weights_func,
        )

        # BacktestResultに変換
        result = self._convert_to_backtest_result(close_prices, sim_result)

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

        # 部分期間アセット情報を計算（ffill/bfill前に計算）
        self._asset_coverage = self._compute_asset_coverage(prices)

        # 欠損値を前方補完（各アセットのデータ範囲内のみ）
        # 注意: 上場前/廃止後のNaNは保持し、途中の欠損のみ補完
        for col in prices.columns:
            series = prices[col]
            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            if first_valid is not None and last_valid is not None:
                # データ範囲内のみffill
                mask = (prices.index >= first_valid) & (prices.index <= last_valid)
                prices.loc[mask, col] = prices.loc[mask, col].ffill()

        return prices

    def _compute_asset_coverage(
        self,
        prices: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        各アセットのデータカバレッジ情報を計算

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ（NaNを含む可能性あり）

        Returns
        -------
        Dict[str, Dict[str, Any]]
            アセットごとのカバレッジ情報
            - first_date: 最初の有効データ日
            - last_date: 最後の有効データ日
            - coverage_days: 有効データ日数
            - total_days: 全期間の日数
            - coverage_ratio: カバレッジ比率
        """
        total_days = len(prices)
        coverage = {}

        for col in prices.columns:
            series = prices[col]
            valid_mask = series.notna()
            valid_count = valid_mask.sum()

            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()

            coverage[col] = {
                "first_date": first_valid,
                "last_date": last_valid,
                "coverage_days": int(valid_count),
                "total_days": total_days,
                "coverage_ratio": valid_count / total_days if total_days > 0 else 0.0,
            }

        return coverage

    def _log_partial_coverage_assets(self) -> None:
        """
        部分期間カバレッジを持つアセットをログに出力
        """
        if not hasattr(self, "_asset_coverage") or self._asset_coverage is None:
            return

        partial_assets = []
        for asset, info in self._asset_coverage.items():
            if info["coverage_ratio"] < 1.0:
                partial_assets.append((asset, info))

        if partial_assets:
            logger.info("Partial coverage detected for %d assets:", len(partial_assets))
            for asset, info in partial_assets:
                first_str = info["first_date"].strftime("%Y-%m-%d") if info["first_date"] else "N/A"
                last_str = info["last_date"].strftime("%Y-%m-%d") if info["last_date"] else "N/A"
                logger.info(
                    "  %s: %s to %s (%.1f%% coverage)",
                    asset,
                    first_str,
                    last_str,
                    info["coverage_ratio"] * 100,
                )

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
        close_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
        weights_func: Optional[callable] = None,
    ) -> SimulationResult:
        """
        高速シミュレーションを実行（翌日初値執行対応）

        処理フロー:
        - Day T (リバランス判定日):
          - Close[T] でシグナル計算
          - 新しいウェイトを計算
          - pending_rebalance = True（翌日執行予約）
        - Day T+1 (執行日):
          - Open[T+1] で売買執行
          - オーバーナイトリターン反映
          - pending_rebalance = False

        Parameters
        ----------
        close_prices : pd.DataFrame
            終値データ
        open_prices : pd.DataFrame
            始値データ
        weights_func : callable, optional
            ウェイト計算関数

        Returns
        -------
        SimulationResult
            シミュレーション結果
        """
        n_days = len(close_prices)
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
        close_matrix = close_prices.values
        open_matrix = open_prices.values
        dates = close_prices.index.tolist()

        # リバランス日をセットに変換（高速検索）
        rebalance_set = set(self._rebalance_dates)

        # 取引コスト設定
        # cost_scheduleが設定されている場合はカテゴリ別コスト、そうでなければ一律コスト
        use_category_costs = self.config.cost_schedule is not None or self.config.asset_master is not None
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0
        total_costs = 0.0
        actual_rebalance_dates = []

        # 保留リバランス状態（翌日初値執行用）
        pending_rebalance = False
        pending_weights = None
        today_cost = 0.0  # 当日発生したコスト

        for i in range(1, n_days):
            current_date = dates[i]
            prev_close = close_matrix[i - 1]
            curr_close = close_matrix[i]
            curr_open = open_matrix[i]
            today_cost = 0.0

            # Step 1: 保留中のリバランスを今日の初値で執行
            if pending_rebalance and pending_weights is not None:
                # Openが NaN または 0以下の場合は Close を使用
                execution_prices = np.where(
                    np.isnan(curr_open) | (curr_open <= 0),
                    curr_close,
                    curr_open
                )

                # オーバーナイトリターン（前日Close → 当日Open）
                valid_overnight = (prev_close > 0) & ~np.isnan(execution_prices) & (execution_prices > 0)
                overnight_returns = np.where(
                    valid_overnight,
                    execution_prices / prev_close - 1,
                    0.0
                )

                # オーバーナイトリターンを現在ウェイトで計算
                overnight_portfolio_return = np.dot(current_weights, overnight_returns)

                # 寄り付き時点のポートフォリオ価値
                open_value = portfolio_values[i - 1] * (1 + overnight_portfolio_return)

                # 売買コスト計算
                if use_category_costs:
                    today_cost = self._calculate_per_symbol_costs(
                        self._asset_names, pending_weights, current_weights, open_value
                    )
                else:
                    turnover = np.sum(np.abs(pending_weights - current_weights))
                    today_cost = turnover * cost_rate * open_value
                total_costs += today_cost

                # リバランス執行日: 正確な2段階計算
                # オーバーナイトリターン(旧ウェイト) + イントラデイリターン(新ウェイト)

                # イントラデイリターン計算（Open[i] → Close[i]）
                valid_intraday = (
                    ~np.isnan(curr_open) &
                    ~np.isnan(curr_close) &
                    (curr_open > 0)
                )
                intraday_returns = np.where(
                    valid_intraday,
                    curr_close / curr_open - 1,
                    0.0
                )

                # 配当リターンを加算（SYNC-006）- イントラデイ部分に加算
                dividend_returns = self._get_dividend_returns(
                    current_date, curr_open, i
                )
                intraday_returns = np.where(
                    valid_intraday,
                    intraday_returns + dividend_returns,
                    intraday_returns
                )

                # ウェイトを更新（イントラデイリターン計算後）
                current_weights = pending_weights
                actual_rebalance_dates.append(current_date)  # 執行日を記録

                # イントラデイポートフォリオリターン（新ウェイトで、NaN対応）
                available_mask = ~np.isnan(intraday_returns) & valid_intraday
                available_weights = np.where(available_mask, current_weights, 0.0)
                weight_sum = np.sum(available_weights)

                if weight_sum > 0:
                    normalized_weights = available_weights / weight_sum
                    safe_returns = np.where(available_mask, intraday_returns, 0.0)
                    intraday_portfolio_return = np.dot(normalized_weights, safe_returns)
                else:
                    intraday_portfolio_return = 0.0

                # ポートフォリオ価値計算: Open価値 × (1 + イントラデイリターン) - コスト
                portfolio_values[i] = open_value * (1 + intraday_portfolio_return) - today_cost

                # 日次リターンを記録（終値ベース）
                if portfolio_values[i - 1] > 0:
                    daily_returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1
                else:
                    daily_returns[i] = 0.0

                # 保留状態をリセット
                pending_rebalance = False
                pending_weights = None

                # 共分散推定器を更新（Close-to-Closeリターンで）
                valid_mask = (prev_close > 0) & ~np.isnan(curr_close)
                price_returns_for_cov = np.where(valid_mask, curr_close / prev_close - 1, 0.0)
                self._update_covariance_with_nan(price_returns_for_cov, valid_mask)

                # DrawdownProtector更新 (SYNC-002)
                if self._dd_protector is not None:
                    self._dd_protector.update(portfolio_values[i])
            else:
                # 非リバランス日: 通常のClose-to-Close計算
                # Step 2: アセット可用性マスク（NaNでない & 価格が正）
                valid_mask = (
                    ~np.isnan(prev_close) &
                    ~np.isnan(curr_close) &
                    (prev_close > 0)
                )

                # 日次リターンを計算（Close to Close）- NaN対応
                price_returns = np.where(
                    valid_mask,
                    curr_close / prev_close - 1,
                    np.nan  # 利用不可アセットはNaN
                )

                # 配当リターンを加算（SYNC-006）
                dividend_returns = self._get_dividend_returns(
                    current_date, prev_close, i
                )
                # NaNアセットには配当を加算しない
                asset_returns = np.where(
                    valid_mask,
                    price_returns + dividend_returns,
                    np.nan
                )

                # 共分散推定器を更新（NaN対応版）
                self._update_covariance_with_nan(asset_returns, valid_mask)

                # ポートフォリオリターンを計算（NaN対応）
                # 利用可能なアセットのみでウェイトを再配分
                available_mask = ~np.isnan(asset_returns)
                available_weights = np.where(available_mask, current_weights, 0.0)
                weight_sum = np.sum(available_weights)

                if weight_sum > 0:
                    # 利用可能アセットでウェイトを正規化
                    normalized_weights = available_weights / weight_sum
                    # NaNリターンを0として扱い、正規化ウェイトで計算
                    safe_returns = np.where(available_mask, asset_returns, 0.0)
                    portfolio_return = np.dot(normalized_weights, safe_returns)
                else:
                    # 全アセットが利用不可の場合はリターン0
                    portfolio_return = 0.0

                daily_returns[i] = portfolio_return

                # ポートフォリオ価値を更新
                portfolio_values[i] = portfolio_values[i - 1] * (1 + portfolio_return)

                # DrawdownProtector更新 (SYNC-002)
                if self._dd_protector is not None:
                    self._dd_protector.update(portfolio_values[i])

            # Step 3: リバランスシグナル確認（翌日執行を予約）
            should_rebalance = current_date in rebalance_set

            if should_rebalance and i < n_days - 1:  # 最終日はリバランスしない（翌日がない）
                # レジーム検出（有効な場合）
                regime_params = self._detect_and_get_regime_params(close_prices, i)

                # リバランス時のアセット可用性マスクを計算
                # 当日の価格がNaNでないアセットが利用可能
                rebalance_available_mask = ~np.isnan(curr_close)

                # 新しいウェイトを計算
                if weights_func is not None:
                    cov = self.cov_estimator.get_covariance()
                    signals = self._get_current_signals(current_date)
                    new_weights = weights_func(signals, cov)
                else:
                    # 等ウェイト（利用可能アセットのみ）
                    if np.any(rebalance_available_mask):
                        n_available = np.sum(rebalance_available_mask)
                        new_weights = np.where(
                            rebalance_available_mask,
                            1.0 / n_available,
                            0.0
                        )
                    else:
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

                # ウェイト制約を適用（利用可能マスクを渡す）
                new_weights = self._apply_weight_constraints(
                    new_weights, rebalance_available_mask
                )

                # TransactionCostOptimizer適用 (SYNC-001)
                if self.config.use_cost_optimizer and TRANSACTION_COST_OPTIMIZER_AVAILABLE:
                    new_weights = self._apply_cost_optimizer(
                        new_weights,
                        current_weights,
                        close_prices,
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
                        close_prices,
                        i,
                        portfolio_values[i],
                    )

                # 翌日執行を予約
                pending_rebalance = True
                pending_weights = new_weights

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

    def _update_covariance_with_nan(
        self,
        returns: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        """
        NaNを含むリターンで共分散推定器を更新

        利用可能なアセットのみで更新し、利用不可アセットの
        共分散/平均は前回値を維持する。

        Parameters
        ----------
        returns : np.ndarray
            1日分のリターン（NaNを含む可能性あり）
        valid_mask : np.ndarray
            有効なアセットのマスク
        """
        if np.all(valid_mask):
            # 全アセットが有効な場合は通常の更新
            self.cov_estimator.update(returns)
        elif np.any(valid_mask):
            # 一部のアセットのみ有効な場合
            # NaNを0に置き換えて更新（ただしマスクを使用）
            safe_returns = np.where(valid_mask, returns, 0.0)
            self.cov_estimator.update_with_mask(safe_returns, valid_mask)
        # 全アセットが利用不可の場合は更新しない

    def _apply_weight_constraints(
        self,
        weights: np.ndarray,
        available_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        ウェイト制約を適用

        Parameters
        ----------
        weights : np.ndarray
            元のウェイト
        available_mask : np.ndarray, optional
            利用可能なアセットのマスク。Noneの場合は全アセット利用可能とみなす。

        Returns
        -------
        np.ndarray
            制約適用後のウェイト
        """
        weights = weights.copy()

        # 利用不可アセットのウェイトを0に設定
        if available_mask is not None:
            weights = np.where(available_mask, weights, 0.0)

        # 最小・最大制約
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        # 利用不可アセットは再度0に（clipで復活しないように）
        if available_mask is not None:
            weights = np.where(available_mask, weights, 0.0)

        # 正規化（合計を1に）
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            # 全アセットが利用不可の場合、利用可能なアセットで等ウェイト
            if available_mask is not None and np.any(available_mask):
                n_available = np.sum(available_mask)
                weights = np.where(available_mask, 1.0 / n_available, 0.0)
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
    open_prices: Optional[pd.DataFrame | pl.DataFrame] = None,
) -> BacktestResult:
    """
    高速バックテストを実行するショートカット関数（Polars/Pandas両対応、翌日初値執行対応）

    Parameters
    ----------
    prices : pd.DataFrame | pl.DataFrame
        終値データ（Polars or Pandas）
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
    open_prices : pd.DataFrame | pl.DataFrame, optional
        始値データ（翌日初値執行用）。Noneの場合は終値を使用。

    Returns
    -------
    BacktestResult
        バックテスト結果
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency=rebalance_frequency,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        use_numba=use_numba,
        use_gpu=use_gpu,
        vix_cash_enabled=vix_cash_enabled,
    )

    engine = BacktestEngine(config)
    return engine.run(
        prices,
        weights_func=weights_func,
        vix_data=vix_data,
        open_prices=open_prices,
    )


# ==============================================================================
# 後方互換エイリアス（非推奨）
# ==============================================================================
# 新規コードでは BacktestConfig, BacktestEngine を使用すること
FastBacktestConfig = BacktestConfig
FastBacktestEngine = BacktestEngine
