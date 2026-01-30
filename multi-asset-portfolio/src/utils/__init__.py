"""
Utility modules for the Multi-Asset Portfolio System.

This package provides:
- logger: Structured logging with structlog
- reproducibility: Seed management and version tracking
- cache_manager: Unified cache manager with adapters (task_032_16)
- memory_profiler: Memory profiling with psutil (P3)
- dataframe_utils: Polars/Pandas conversion utilities (task_013_6)
"""

from src.utils.logger import get_logger, setup_logging
from src.utils.reproducibility import SeedManager, get_run_info
from src.utils.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryAlert,
    memory_profiler,
    log_memory,
    get_memory_summary,
    get_current_memory_mb,
    check_memory_threshold,
)
from src.utils.cache_manager import (
    UnifiedCacheManager,
    unified_cache_manager,
    CacheType,
    CachePolicy,
    UnifiedCacheStats,
    get_signal_cache,
    get_data_cache,
    get_dataframe_cache,
    clear_all_caches,
    get_cache_summary,
)
from src.utils.dataframe_utils import (
    ensure_polars,
    ensure_pandas,
    extract_numeric_pandas,
    extract_numeric_numpy,
    is_polars,
    is_pandas,
    ConversionTracker,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "SeedManager",
    "get_run_info",
    # Unified Cache Manager (task_032_16)
    "UnifiedCacheManager",
    "unified_cache_manager",
    "CacheType",
    "CachePolicy",
    "UnifiedCacheStats",
    "get_signal_cache",
    "get_data_cache",
    "get_dataframe_cache",
    "clear_all_caches",
    "get_cache_summary",
    # Memory Profiler (P3)
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryAlert",
    "memory_profiler",
    "log_memory",
    "get_memory_summary",
    "get_current_memory_mb",
    "check_memory_threshold",
    # DataFrame Utilities (task_013_6)
    "ensure_polars",
    "ensure_pandas",
    "extract_numeric_pandas",
    "extract_numeric_numpy",
    "is_polars",
    "is_pandas",
    "ConversionTracker",
]
