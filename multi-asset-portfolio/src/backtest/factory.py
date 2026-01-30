"""
Backtest Engine Factory - エンジンファクトリ＆自動選択【INT-005】

状況に応じて最適なバックテストエンジンを自動選択する仕組みを提供。

使用例:
    from src.backtest.factory import BacktestEngineFactory
    from src.backtest.base import UnifiedBacktestConfig

    # 自動選択
    config = UnifiedBacktestConfig(...)
    engine = BacktestEngineFactory.create(mode='auto', config=config)

    # 明示的選択
    engine = BacktestEngineFactory.create(mode='fast', config=config)

    # 利用可能エンジン一覧
    available = BacktestEngineFactory.list_available()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from src.backtest.base import BacktestEngineBase, UnifiedBacktestConfig

logger = logging.getLogger(__name__)


class EngineMode(str, Enum):
    """エンジン選択モード"""
    AUTO = "auto"
    STANDARD = "standard"
    FAST = "fast"
    STREAMING = "streaming"
    RAY = "ray"
    VECTORBT = "vectorbt"


@dataclass
class EngineInfo:
    """エンジン情報"""
    mode: str
    class_name: str
    module_path: str
    description: str
    recommended_for: str
    available: bool = False
    requires_optional_deps: List[str] = None

    def __post_init__(self):
        if self.requires_optional_deps is None:
            self.requires_optional_deps = []


# エンジンレジストリ
ENGINE_REGISTRY: Dict[str, EngineInfo] = {
    "standard": EngineInfo(
        mode="standard",
        class_name="BacktestEngine",
        module_path="src.backtest.engine",
        description="標準Walk-Forwardエンジン（データ自動取得対応）",
        recommended_for="小〜中規模、フルパイプライン統合",
    ),
    "fast": EngineInfo(
        mode="fast",
        class_name="FastBacktestEngine",
        module_path="src.backtest.fast_engine",
        description="高速Numba/GPUエンジン",
        recommended_for="中規模、高速実行が必要な場合",
        requires_optional_deps=["numba"],
    ),
    "streaming": EngineInfo(
        mode="streaming",
        class_name="StreamingBacktestEngine",
        module_path="src.backtest.streaming_engine",
        description="ストリーミングエンジン（低メモリ）",
        recommended_for="大規模データ、メモリ制約がある場合",
    ),
    "ray": EngineInfo(
        mode="ray",
        class_name="RayBacktestEngine",
        module_path="src.backtest.ray_engine",
        description="Ray分散エンジン",
        recommended_for="超大規模、クラスター環境",
        requires_optional_deps=["ray"],
    ),
    "vectorbt": EngineInfo(
        mode="vectorbt",
        class_name="VectorBTStyleEngine",
        module_path="src.backtest.vectorbt_engine",
        description="VectorBTスタイル完全ベクトル化エンジン",
        recommended_for="大規模、高速ベクトル計算",
    ),
}


@dataclass
class AutoSelectCriteria:
    """自動選択の判断基準"""
    universe_size: int = 0
    period_days: int = 0
    memory_constrained: bool = False
    distributed_available: bool = False
    gpu_available: bool = False


class BacktestEngineFactory:
    """
    バックテストエンジンファクトリ

    状況に応じて最適なエンジンを自動選択、または明示的に選択可能。

    使用例:
        # 自動選択
        engine = BacktestEngineFactory.create(mode='auto', config=config)

        # 明示的選択
        engine = BacktestEngineFactory.create(mode='fast', config=config)

        # ユニバースサイズを指定して自動選択
        engine = BacktestEngineFactory.create(
            mode='auto',
            config=config,
            universe_size=500,
        )
    """

    # 自動選択の閾値
    LARGE_UNIVERSE_THRESHOLD = 500  # 大規模ユニバース
    MEDIUM_UNIVERSE_THRESHOLD = 50   # 中規模ユニバース
    LONG_PERIOD_YEARS = 10           # 長期間

    @classmethod
    def create(
        cls,
        mode: str = "auto",
        config: "UnifiedBacktestConfig | None" = None,
        universe_size: int = 0,
        **kwargs,
    ) -> "BacktestEngineBase":
        """
        エンジンを生成

        Args:
            mode: エンジンモード ('auto', 'standard', 'fast', 'streaming', 'ray', 'vectorbt')
            config: 統一設定
            universe_size: ユニバースサイズ（自動選択時に使用）
            **kwargs: エンジン固有の引数

        Returns:
            BacktestEngineBase: 生成されたエンジン

        Raises:
            ValueError: 不明なモード
            ImportError: エンジンが利用不可
        """
        mode = mode.lower()

        if mode == "auto":
            return cls._auto_select(config, universe_size, **kwargs)

        if mode not in ENGINE_REGISTRY:
            available_modes = list(ENGINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown engine mode: {mode}. "
                f"Available modes: {available_modes}"
            )

        return cls._create_engine(mode, config, **kwargs)

    @classmethod
    def _auto_select(
        cls,
        config: "UnifiedBacktestConfig | None",
        universe_size: int = 0,
        **kwargs,
    ) -> "BacktestEngineBase":
        """
        状況に応じて最適なエンジンを自動選択

        選択ロジック:
        1. 超大規模（銘柄数 > 500）かつRay利用可能 → ray
        2. 大規模（銘柄数 > 500）または長期間（> 10年） → vectorbt or streaming
        3. 中規模（銘柄数 50-500） → fast
        4. 小規模（銘柄数 < 50） → standard

        Args:
            config: 統一設定
            universe_size: ユニバースサイズ
            **kwargs: エンジン固有の引数

        Returns:
            BacktestEngineBase: 選択されたエンジン
        """
        criteria = cls._build_criteria(config, universe_size)

        logger.info(
            f"Auto-selecting engine: universe={criteria.universe_size}, "
            f"period_days={criteria.period_days}, "
            f"memory_constrained={criteria.memory_constrained}"
        )

        # 選択ロジック
        selected_mode = cls._determine_optimal_mode(criteria)

        logger.info(f"Selected engine mode: {selected_mode}")

        return cls._create_engine(selected_mode, config, **kwargs)

    @classmethod
    def _build_criteria(
        cls,
        config: "UnifiedBacktestConfig | None",
        universe_size: int,
    ) -> AutoSelectCriteria:
        """選択基準を構築"""
        criteria = AutoSelectCriteria(universe_size=universe_size)

        if config:
            # 期間を計算
            start = config.start_date
            end = config.end_date
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            criteria.period_days = (end - start).days

        # Ray利用可能性チェック
        try:
            from src.backtest.ray_engine import RAY_AVAILABLE
            criteria.distributed_available = RAY_AVAILABLE
        except ImportError:
            criteria.distributed_available = False

        # GPU利用可能性チェック
        try:
            from src.backtest.gpu_compute import GPU_AVAILABLE
            criteria.gpu_available = GPU_AVAILABLE
        except ImportError:
            criteria.gpu_available = False

        return criteria

    @classmethod
    def _determine_optimal_mode(cls, criteria: AutoSelectCriteria) -> str:
        """最適なモードを決定"""
        universe_size = criteria.universe_size
        period_days = criteria.period_days
        period_years = period_days / 365.25 if period_days > 0 else 0

        # メモリ制約がある場合はストリーミング
        if criteria.memory_constrained:
            if cls._is_available("streaming"):
                return "streaming"

        # 超大規模かつRay利用可能
        if universe_size > cls.LARGE_UNIVERSE_THRESHOLD and criteria.distributed_available:
            if cls._is_available("ray"):
                return "ray"

        # 大規模または長期間
        if universe_size > cls.LARGE_UNIVERSE_THRESHOLD or period_years > cls.LONG_PERIOD_YEARS:
            # VectorBTを優先（ベクトル化で高速）
            if cls._is_available("vectorbt"):
                return "vectorbt"
            # フォールバック: ストリーミング（メモリ効率）
            if cls._is_available("streaming"):
                return "streaming"

        # 中規模
        if universe_size >= cls.MEDIUM_UNIVERSE_THRESHOLD:
            if cls._is_available("fast"):
                return "fast"

        # 小規模またはデフォルト
        if cls._is_available("standard"):
            return "standard"

        # 何も利用できない場合はエラー
        raise RuntimeError("No backtest engine available")

    @classmethod
    def _create_engine(
        cls,
        mode: str,
        config: "UnifiedBacktestConfig | None",
        **kwargs,
    ) -> "BacktestEngineBase":
        """エンジンを実際に生成"""
        info = ENGINE_REGISTRY[mode]

        # 動的インポート
        engine_class = cls._import_engine_class(info)

        # エンジン生成
        if config is not None:
            # INT-001対応エンジン: from_unified_configクラスメソッドを使用
            if hasattr(engine_class, "from_unified_config"):
                return engine_class.from_unified_config(config, **kwargs)

            # フォールバック: configなしで生成を試みる
            logger.warning(
                f"{info.class_name} does not support unified config. "
                f"Creating without config."
            )
            return engine_class(**kwargs)
        else:
            return engine_class(**kwargs)

    @classmethod
    def _import_engine_class(cls, info: EngineInfo) -> Type["BacktestEngineBase"]:
        """エンジンクラスを動的インポート"""
        import importlib

        try:
            module = importlib.import_module(info.module_path)
            engine_class = getattr(module, info.class_name)
            return engine_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import {info.class_name} from {info.module_path}: {e}. "
                f"Required dependencies: {info.requires_optional_deps}"
            )
        except AttributeError as e:
            raise ImportError(
                f"Class {info.class_name} not found in {info.module_path}: {e}"
            )

    @classmethod
    def _is_available(cls, mode: str) -> bool:
        """エンジンが利用可能かチェック"""
        if mode not in ENGINE_REGISTRY:
            return False

        info = ENGINE_REGISTRY[mode]

        try:
            cls._import_engine_class(info)
            return True
        except ImportError:
            return False

    @classmethod
    def list_available(cls) -> List[str]:
        """
        利用可能なエンジン一覧を取得

        Returns:
            List[str]: 利用可能なエンジンモードのリスト
        """
        available = []
        for mode in ENGINE_REGISTRY:
            if cls._is_available(mode):
                available.append(mode)
        return available

    @classmethod
    def list_all(cls) -> List[EngineInfo]:
        """
        全エンジン情報を取得（利用可否を含む）

        Returns:
            List[EngineInfo]: エンジン情報のリスト
        """
        result = []
        for mode, info in ENGINE_REGISTRY.items():
            info.available = cls._is_available(mode)
            result.append(info)
        return result

    @classmethod
    def get_info(cls, mode: str) -> EngineInfo:
        """
        特定エンジンの情報を取得

        Args:
            mode: エンジンモード

        Returns:
            EngineInfo: エンジン情報

        Raises:
            ValueError: 不明なモード
        """
        if mode not in ENGINE_REGISTRY:
            raise ValueError(f"Unknown engine mode: {mode}")

        info = ENGINE_REGISTRY[mode]
        info.available = cls._is_available(mode)
        return info

    @classmethod
    def recommend(
        cls,
        universe_size: int,
        period_days: int = 0,
        memory_mb: int = 0,
    ) -> str:
        """
        推奨エンジンを取得（生成せず）

        Args:
            universe_size: ユニバースサイズ
            period_days: 期間（日数）
            memory_mb: 利用可能メモリ（MB）

        Returns:
            str: 推奨エンジンモード
        """
        criteria = AutoSelectCriteria(
            universe_size=universe_size,
            period_days=period_days,
            memory_constrained=memory_mb > 0 and memory_mb < 4096,
        )

        # 分散/GPU利用可能性チェック
        try:
            from src.backtest.ray_engine import RAY_AVAILABLE
            criteria.distributed_available = RAY_AVAILABLE
        except ImportError:
            pass

        try:
            from src.backtest.gpu_compute import GPU_AVAILABLE
            criteria.gpu_available = GPU_AVAILABLE
        except ImportError:
            pass

        return cls._determine_optimal_mode(criteria)


# 便利関数
def create_engine(
    mode: str = "auto",
    config: "UnifiedBacktestConfig | None" = None,
    **kwargs,
) -> "BacktestEngineBase":
    """
    エンジンを生成する便利関数

    BacktestEngineFactory.create() のショートカット。

    Args:
        mode: エンジンモード
        config: 統一設定
        **kwargs: エンジン固有の引数

    Returns:
        BacktestEngineBase: 生成されたエンジン
    """
    return BacktestEngineFactory.create(mode=mode, config=config, **kwargs)


def list_engines() -> List[str]:
    """
    利用可能なエンジン一覧を取得する便利関数

    Returns:
        List[str]: 利用可能なエンジンモードのリスト
    """
    return BacktestEngineFactory.list_available()


def recommend_engine(universe_size: int, period_days: int = 0) -> str:
    """
    推奨エンジンを取得する便利関数

    Args:
        universe_size: ユニバースサイズ
        period_days: 期間（日数）

    Returns:
        str: 推奨エンジンモード
    """
    return BacktestEngineFactory.recommend(universe_size, period_days)


# =============================================================================
# Ray分散処理 + ResourceConfig統合
# =============================================================================


def init_ray(
    n_workers: Optional[int] = None,
    address: Optional[str] = None,
    object_store_memory: Optional[int] = None,
) -> bool:
    """
    Rayを初期化するヘルパー関数

    ResourceConfigと連携して最適なワーカー数でRayを初期化。

    Args:
        n_workers: ワーカー数（Noneでリソース設定から自動取得）
        address: 既存Rayクラスタのアドレス（"auto"で自動検出）
        object_store_memory: オブジェクトストアメモリ（バイト）

    Returns:
        bool: 初期化成功したかどうか

    Example:
        from src.backtest.factory import init_ray
        from src.config.resource_config import get_current_resource_config

        # リソース設定から自動取得
        init_ray()

        # ワーカー数を明示指定
        rc = get_current_resource_config()
        init_ray(n_workers=rc.ray_workers)
    """
    try:
        import ray
    except ImportError:
        logger.warning("Ray is not installed. Install with: pip install ray")
        return False

    # 既に初期化済みならスキップ
    if ray.is_initialized():
        logger.info("Ray is already initialized")
        return True

    # ワーカー数をResourceConfigから取得
    if n_workers is None:
        try:
            from src.config.resource_config import get_current_resource_config
            rc = get_current_resource_config()
            n_workers = rc.ray_workers
            logger.info(f"Using ray_workers from ResourceConfig: {n_workers}")
        except Exception as e:
            logger.warning(f"Failed to get ResourceConfig: {e}, using default")
            n_workers = max(1, (import_cpu_count() or 4) - 1)

    try:
        init_kwargs: Dict[str, Any] = {
            "ignore_reinit_error": True,
            "num_cpus": n_workers,
        }

        if address:
            init_kwargs["address"] = address

        if object_store_memory:
            init_kwargs["object_store_memory"] = object_store_memory

        ray.init(**init_kwargs)
        logger.info(f"Ray initialized with {n_workers} workers")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return False


def import_cpu_count() -> Optional[int]:
    """CPUコア数を取得（osモジュールから）"""
    import os
    return os.cpu_count()


def create_ray_engine(
    config: "UnifiedBacktestConfig | None" = None,
    n_workers: Optional[int] = None,
    auto_init_ray: bool = True,
    **kwargs,
) -> "BacktestEngineBase":
    """
    Rayバックテストエンジンを作成（ResourceConfig統合）

    ResourceConfigからワーカー数を自動取得してRayエンジンを生成。
    これにより、システムリソースに応じた最適な並列処理が可能。

    Args:
        config: 統一バックテスト設定
        n_workers: ワーカー数（Noneでリソース設定から自動取得）
        auto_init_ray: Rayを自動初期化するか（デフォルト: True）
        **kwargs: その他のRayBacktestConfig設定

    Returns:
        BacktestEngineBase: 生成されたRayエンジン

    Raises:
        ImportError: Rayが利用不可の場合

    Example:
        from src.backtest.factory import create_ray_engine
        from src.config.resource_config import get_current_resource_config

        # リソース設定から自動取得
        engine = create_ray_engine(config)

        # ワーカー数を明示指定
        rc = get_current_resource_config()
        engine = create_ray_engine(config, n_workers=rc.ray_workers)
    """
    # ワーカー数をResourceConfigから取得
    if n_workers is None:
        try:
            from src.config.resource_config import get_current_resource_config
            rc = get_current_resource_config()
            n_workers = rc.ray_workers
            logger.info(f"Using ray_workers from ResourceConfig: {n_workers}")
        except Exception as e:
            logger.warning(f"Failed to get ResourceConfig: {e}, using default")
            n_workers = max(1, (import_cpu_count() or 4) - 1)

    return BacktestEngineFactory.create(
        mode="ray",
        config=config,
        n_workers=n_workers,
        auto_init_ray=auto_init_ray,
        **kwargs,
    )
