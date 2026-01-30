"""
Resource Configuration - システムリソースの動的検出と設定

サーバーのスペック（CPU、メモリ、GPU）を自動検出し、
最適なリソース設定を生成する。

Usage:
    from src.config.resource_config import get_resource_config, ResourceConfig

    # 自動検出
    config = get_resource_config()
    print(f"Workers: {config.max_workers}")
    print(f"Cache memory: {config.cache_max_memory_mb} MB")

    # 手動設定（オーバーライド）
    config = get_resource_config(
        max_workers=32,
        memory_usage_ratio=0.8,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cupy = None


@dataclass
class GPUInfo:
    """GPU情報"""
    available: bool = False
    device_count: int = 0
    devices: list[dict[str, Any]] = field(default_factory=list)
    total_memory_gb: float = 0.0


@dataclass
class SystemResources:
    """システムリソース情報"""
    # CPU
    cpu_physical_cores: int = 1
    cpu_logical_cores: int = 1

    # Memory (bytes)
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0

    # Disk (bytes)
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0

    # GPU
    gpu: GPUInfo = field(default_factory=GPUInfo)

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_bytes / (1024**3)

    @property
    def memory_available_gb(self) -> float:
        return self.memory_available_bytes / (1024**3)

    @property
    def disk_total_gb(self) -> float:
        return self.disk_total_bytes / (1024**3)

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_bytes / (1024**3)


@dataclass
class ResourceConfig:
    """
    リソース設定

    システムリソースに基づいて自動計算された設定値。
    手動でオーバーライド可能。
    """
    # === CPU設定 ===
    max_workers: int = 4
    """並列ワーカー数（ProcessPool/ThreadPool）"""

    ray_workers: int = 4
    """Ray分散処理のワーカー数"""

    # === メモリ設定 ===
    cache_max_memory_mb: int = 512
    """キャッシュ最大メモリ（MB）"""

    cache_max_entries: int = 1000
    """キャッシュ最大エントリ数"""

    signal_cache_max_size: int = 50
    """シグナルキャッシュ最大サイズ"""

    # === チャンク処理設定 ===
    disable_chunking: bool = False
    """チャンク処理を無効化（全データ一括処理）"""

    chunk_size: int = 100
    """チャンクサイズ（行数）"""

    streaming_chunk_size: int = 50
    """ストリーミング処理のチャンクサイズ（銘柄数）"""

    # === ディスク設定 ===
    cache_max_disk_mb: Optional[int] = 500
    """ディスクキャッシュ最大サイズ（MB）。Noneで無制限"""

    # === GPU設定 ===
    use_gpu: bool = False
    """GPU計算を使用"""

    gpu_memory_fraction: float = 0.8
    """GPU使用時のメモリ使用率"""

    # === Numba設定 ===
    use_numba: bool = True
    """Numba JITを使用"""

    numba_parallel: bool = True
    """Numba並列化を使用"""

    # === データ取得設定 ===
    batch_size: int = 50
    """データ取得バッチサイズ"""

    parallel_fetchers: int = 4
    """並列データ取得数"""

    # === 自動検出情報 ===
    auto_detected: bool = False
    """自動検出されたかどうか"""

    system_resources: Optional[SystemResources] = None
    """検出されたシステムリソース"""

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書に変換"""
        return {
            "cpu": {
                "max_workers": self.max_workers,
                "ray_workers": self.ray_workers,
            },
            "memory": {
                "cache_max_memory_mb": self.cache_max_memory_mb,
                "cache_max_entries": self.cache_max_entries,
                "signal_cache_max_size": self.signal_cache_max_size,
            },
            "chunking": {
                "disable_chunking": self.disable_chunking,
                "chunk_size": self.chunk_size,
                "streaming_chunk_size": self.streaming_chunk_size,
            },
            "disk": {
                "cache_max_disk_mb": self.cache_max_disk_mb,
            },
            "gpu": {
                "use_gpu": self.use_gpu,
                "gpu_memory_fraction": self.gpu_memory_fraction,
            },
            "numba": {
                "use_numba": self.use_numba,
                "numba_parallel": self.numba_parallel,
            },
            "data_fetching": {
                "batch_size": self.batch_size,
                "parallel_fetchers": self.parallel_fetchers,
            },
            "auto_detected": self.auto_detected,
        }


def detect_system_resources() -> SystemResources:
    """
    システムリソースを検出

    Returns:
        SystemResources: 検出されたリソース情報
    """
    resources = SystemResources()

    # CPU
    resources.cpu_logical_cores = os.cpu_count() or 1
    if HAS_PSUTIL:
        resources.cpu_physical_cores = psutil.cpu_count(logical=False) or 1
    else:
        resources.cpu_physical_cores = resources.cpu_logical_cores

    # Memory
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        resources.memory_total_bytes = mem.total
        resources.memory_available_bytes = mem.available
    else:
        # フォールバック: 8GB想定
        resources.memory_total_bytes = 8 * 1024**3
        resources.memory_available_bytes = 4 * 1024**3

    # Disk
    if HAS_PSUTIL:
        try:
            disk = psutil.disk_usage('/')
            resources.disk_total_bytes = disk.total
            resources.disk_free_bytes = disk.free
        except Exception:
            pass

    # GPU
    gpu_info = GPUInfo()
    if HAS_CUPY:
        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
            if device_count > 0:
                gpu_info.available = True
                gpu_info.device_count = device_count

                total_gpu_memory = 0
                for i in range(device_count):
                    props = cupy.cuda.runtime.getDeviceProperties(i)
                    mem_info = cupy.cuda.Device(i).mem_info
                    total_gpu_memory += mem_info[1]
                    gpu_info.devices.append({
                        "id": i,
                        "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                        "memory_gb": mem_info[1] / (1024**3),
                    })
                gpu_info.total_memory_gb = total_gpu_memory / (1024**3)
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")

    resources.gpu = gpu_info

    return resources


def calculate_optimal_config(
    resources: SystemResources,
    memory_usage_ratio: float = 0.7,
    dedicated_server: bool = True,
) -> ResourceConfig:
    """
    システムリソースから最適な設定を計算

    Args:
        resources: システムリソース情報
        memory_usage_ratio: メモリ使用率（0.0-1.0）
        dedicated_server: 専用サーバーモード（True=制限緩和）

    Returns:
        ResourceConfig: 計算された設定
    """
    config = ResourceConfig()
    config.auto_detected = True
    config.system_resources = resources

    # === CPU設定 ===
    if dedicated_server:
        # 専用サーバー: 全コア使用
        config.max_workers = resources.cpu_logical_cores
        config.ray_workers = resources.cpu_logical_cores
    else:
        # 共有環境: コア数-1（最低1）
        config.max_workers = max(1, resources.cpu_logical_cores - 1)
        config.ray_workers = max(1, resources.cpu_logical_cores - 1)

    # === メモリ設定 ===
    available_memory_mb = int(resources.memory_available_bytes / (1024**2))
    usable_memory_mb = int(available_memory_mb * memory_usage_ratio)

    if dedicated_server:
        # 専用サーバー: 利用可能メモリの大部分をキャッシュに
        config.cache_max_memory_mb = usable_memory_mb
        # エントリ数: 1MBあたり100エントリ想定
        config.cache_max_entries = max(10000, usable_memory_mb * 100)
        config.signal_cache_max_size = max(500, usable_memory_mb // 10)
    else:
        # 共有環境: 控えめな設定
        config.cache_max_memory_mb = min(2048, usable_memory_mb // 4)
        config.cache_max_entries = 5000
        config.signal_cache_max_size = 100

    # === チャンク処理設定 ===
    # メモリが十分（16GB以上）ならチャンク処理を無効化
    if resources.memory_available_gb >= 16 and dedicated_server:
        config.disable_chunking = True
        config.chunk_size = 10000  # 大きな値（実質無効）
        config.streaming_chunk_size = 500
    else:
        config.disable_chunking = False
        # メモリに応じてチャンクサイズを調整
        config.chunk_size = min(1000, max(100, int(resources.memory_available_gb * 50)))
        config.streaming_chunk_size = min(200, max(50, int(resources.memory_available_gb * 10)))

    # === ディスク設定 ===
    if dedicated_server:
        # 専用サーバー: 無制限（Noneで表現）
        config.cache_max_disk_mb = None
    else:
        # 共有環境: 空き容量の10%まで
        config.cache_max_disk_mb = int(resources.disk_free_gb * 100)  # GB→MB * 10%

    # === GPU設定 ===
    config.use_gpu = resources.gpu.available
    if resources.gpu.available:
        config.gpu_memory_fraction = 0.8 if dedicated_server else 0.5

    # === Numba設定 ===
    config.use_numba = True
    config.numba_parallel = True  # 常に有効化

    # === データ取得設定 ===
    # ワーカー数に応じて調整
    config.batch_size = min(200, max(50, config.max_workers * 10))
    config.parallel_fetchers = min(config.max_workers, 8)

    return config


def get_resource_config(
    max_workers: Optional[int] = None,
    memory_usage_ratio: float = 0.7,
    dedicated_server: bool = True,
    **overrides: Any,
) -> ResourceConfig:
    """
    リソース設定を取得（メイン関数）

    システムリソースを自動検出し、最適な設定を生成。
    必要に応じて手動でオーバーライド可能。

    Args:
        max_workers: ワーカー数（Noneで自動検出）
        memory_usage_ratio: メモリ使用率
        dedicated_server: 専用サーバーモード
        **overrides: その他のオーバーライド設定

    Returns:
        ResourceConfig: リソース設定

    Example:
        # 完全自動
        config = get_resource_config()

        # ワーカー数のみ指定
        config = get_resource_config(max_workers=16)

        # 共有環境モード
        config = get_resource_config(dedicated_server=False)

        # 詳細オーバーライド
        config = get_resource_config(
            max_workers=32,
            cache_max_memory_mb=8192,
            use_gpu=False,
        )
    """
    # システムリソース検出
    resources = detect_system_resources()

    logger.info(
        f"System resources detected: "
        f"CPU={resources.cpu_logical_cores} cores, "
        f"Memory={resources.memory_total_gb:.1f}GB (available={resources.memory_available_gb:.1f}GB), "
        f"GPU={'Yes' if resources.gpu.available else 'No'}"
    )

    # 最適設定を計算
    config = calculate_optimal_config(
        resources,
        memory_usage_ratio=memory_usage_ratio,
        dedicated_server=dedicated_server,
    )

    # 手動オーバーライド
    if max_workers is not None:
        config.max_workers = max_workers
        config.ray_workers = max_workers

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key: {key}")

    logger.info(
        f"Resource config: "
        f"workers={config.max_workers}, "
        f"cache_memory={config.cache_max_memory_mb}MB, "
        f"gpu={config.use_gpu}, "
        f"chunking={'disabled' if config.disable_chunking else f'size={config.chunk_size}'}"
    )

    return config


# グローバルインスタンス（シングルトン）
_resource_config: Optional[ResourceConfig] = None


def init_resource_config(**kwargs: Any) -> ResourceConfig:
    """
    リソース設定を初期化（アプリケーション起動時に一度だけ呼ぶ）

    Args:
        **kwargs: get_resource_config()への引数

    Returns:
        ResourceConfig: 初期化された設定
    """
    global _resource_config
    _resource_config = get_resource_config(**kwargs)
    return _resource_config


def get_current_resource_config() -> ResourceConfig:
    """
    現在のリソース設定を取得

    初期化されていない場合は自動で初期化。

    Returns:
        ResourceConfig: 現在の設定
    """
    global _resource_config
    if _resource_config is None:
        _resource_config = get_resource_config()
    return _resource_config


def print_resource_summary() -> None:
    """リソース設定のサマリーを表示"""
    config = get_current_resource_config()
    resources = config.system_resources

    print("=" * 60)
    print("System Resource Configuration")
    print("=" * 60)

    if resources:
        print(f"\n[System Detected]")
        print(f"  CPU Cores: {resources.cpu_logical_cores} (physical: {resources.cpu_physical_cores})")
        print(f"  Memory: {resources.memory_total_gb:.1f} GB total, {resources.memory_available_gb:.1f} GB available")
        print(f"  Disk: {resources.disk_total_gb:.1f} GB total, {resources.disk_free_gb:.1f} GB free")
        if resources.gpu.available:
            print(f"  GPU: {resources.gpu.device_count} device(s), {resources.gpu.total_memory_gb:.1f} GB total")
            for dev in resources.gpu.devices:
                print(f"    - {dev['name']}: {dev['memory_gb']:.1f} GB")
        else:
            print(f"  GPU: Not available")

    print(f"\n[Calculated Config]")
    print(f"  Max Workers: {config.max_workers}")
    print(f"  Cache Memory: {config.cache_max_memory_mb} MB")
    print(f"  Cache Entries: {config.cache_max_entries}")
    print(f"  Chunking: {'Disabled' if config.disable_chunking else f'Enabled (size={config.chunk_size})'}")
    print(f"  Disk Cache: {'Unlimited' if config.cache_max_disk_mb is None else f'{config.cache_max_disk_mb} MB'}")
    print(f"  GPU: {'Enabled' if config.use_gpu else 'Disabled'}")
    print(f"  Numba Parallel: {'Enabled' if config.numba_parallel else 'Disabled'}")

    print("=" * 60)
