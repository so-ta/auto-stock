"""
Memory Profiler - メモリプロファイリング強化【P3】

psutilを使用した常時メモリモニタリング。
バックテスト各フェーズでメモリ記録し、スパイク検出時にアラート。

Usage:
    from src.utils.memory_profiler import memory_profiler, log_memory, MemoryAlert

    # デコレータで関数のメモリ使用を追跡
    @memory_profiler.track("data_loading")
    def load_data():
        ...

    # 手動でメモリ記録
    log_memory("after_signal_computation")

    # プロファイラのサマリー
    print(memory_profiler.summary())

    # メモリアラートの設定
    memory_profiler.set_threshold(warning_mb=1024, critical_mb=2048)
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

if not PSUTIL_AVAILABLE:
    logger.warning(
        "psutil not installed. Memory profiling will be disabled. "
        "Install with: pip install psutil"
    )

T = TypeVar("T")


# デフォルトのメモリ閾値（MB）
DEFAULT_WARNING_THRESHOLD_MB = 1024  # 1GB
DEFAULT_CRITICAL_THRESHOLD_MB = 2048  # 2GB
DEFAULT_SPIKE_THRESHOLD_MB = 256  # 256MB increase


@dataclass
class MemorySnapshot:
    """メモリスナップショット"""

    phase: str
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # メモリ使用率（システム全体）
    delta_mb: float = 0.0  # 前回からの差分

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "rss_mb": round(self.rss_mb, 2),
            "vms_mb": round(self.vms_mb, 2),
            "percent": round(self.percent, 2),
            "delta_mb": round(self.delta_mb, 2),
        }


@dataclass
class MemoryAlert:
    """メモリアラート"""

    level: str  # "warning", "critical", "spike"
    phase: str
    message: str
    current_mb: float
    threshold_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryProfiler:
    """
    メモリプロファイラー

    バックテスト各フェーズでメモリ使用量を記録し、
    閾値超過時にアラートを発行。
    """

    def __init__(
        self,
        warning_threshold_mb: float = DEFAULT_WARNING_THRESHOLD_MB,
        critical_threshold_mb: float = DEFAULT_CRITICAL_THRESHOLD_MB,
        spike_threshold_mb: float = DEFAULT_SPIKE_THRESHOLD_MB,
        enabled: bool = True,
    ):
        """
        初期化

        Args:
            warning_threshold_mb: WARNING閾値（MB）
            critical_threshold_mb: CRITICAL閾値（MB）
            spike_threshold_mb: スパイク検出閾値（前回比、MB）
            enabled: プロファイリング有効化
        """
        self._warning_threshold = warning_threshold_mb
        self._critical_threshold = critical_threshold_mb
        self._spike_threshold = spike_threshold_mb
        self._enabled = enabled and PSUTIL_AVAILABLE

        self._snapshots: List[MemorySnapshot] = []
        self._alerts: List[MemoryAlert] = []
        self._lock = threading.RLock()
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None

        # 初期スナップショット
        if self._enabled:
            self.record("init")

    def set_threshold(
        self,
        warning_mb: Optional[float] = None,
        critical_mb: Optional[float] = None,
        spike_mb: Optional[float] = None,
    ) -> None:
        """
        閾値を設定

        Args:
            warning_mb: WARNING閾値
            critical_mb: CRITICAL閾値
            spike_mb: スパイク検出閾値
        """
        if warning_mb is not None:
            self._warning_threshold = warning_mb
        if critical_mb is not None:
            self._critical_threshold = critical_mb
        if spike_mb is not None:
            self._spike_threshold = spike_mb

    def enable(self) -> None:
        """プロファイリングを有効化"""
        self._enabled = True

    def disable(self) -> None:
        """プロファイリングを無効化"""
        self._enabled = False

    def record(self, phase: str) -> Optional[MemorySnapshot]:
        """
        現在のメモリ状態を記録

        Args:
            phase: フェーズ名（例: "data_loading", "signal_computation"）

        Returns:
            MemorySnapshot（無効時はNone）
        """
        if not self._enabled or not PSUTIL_AVAILABLE or self._process is None:
            return None

        with self._lock:
            mem_info = self._process.memory_info()
            system_percent = psutil.virtual_memory().percent  # type: ignore

            rss_mb = mem_info.rss / 1024 / 1024
            vms_mb = mem_info.vms / 1024 / 1024

            # 前回からの差分を計算
            delta_mb = 0.0
            if self._snapshots:
                delta_mb = rss_mb - self._snapshots[-1].rss_mb

            snapshot = MemorySnapshot(
                phase=phase,
                timestamp=datetime.now(),
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                percent=system_percent,
                delta_mb=delta_mb,
            )

            self._snapshots.append(snapshot)

            # ログ出力
            logger.info(
                f"[{phase}] Memory: {rss_mb:.1f} MB (delta: {delta_mb:+.1f} MB, system: {system_percent:.1f}%)"
            )

            # アラートチェック
            self._check_alerts(snapshot)

            return snapshot

    def _check_alerts(self, snapshot: MemorySnapshot) -> None:
        """アラートをチェックして発行"""
        # CRITICALチェック
        if snapshot.rss_mb >= self._critical_threshold:
            alert = MemoryAlert(
                level="critical",
                phase=snapshot.phase,
                message=f"CRITICAL: Memory usage {snapshot.rss_mb:.1f} MB exceeds threshold {self._critical_threshold:.1f} MB",
                current_mb=snapshot.rss_mb,
                threshold_mb=self._critical_threshold,
            )
            self._alerts.append(alert)
            logger.critical(alert.message)

        # WARNINGチェック
        elif snapshot.rss_mb >= self._warning_threshold:
            alert = MemoryAlert(
                level="warning",
                phase=snapshot.phase,
                message=f"WARNING: Memory usage {snapshot.rss_mb:.1f} MB exceeds threshold {self._warning_threshold:.1f} MB",
                current_mb=snapshot.rss_mb,
                threshold_mb=self._warning_threshold,
            )
            self._alerts.append(alert)
            logger.warning(alert.message)

        # スパイクチェック
        if snapshot.delta_mb >= self._spike_threshold:
            alert = MemoryAlert(
                level="spike",
                phase=snapshot.phase,
                message=f"SPIKE: Memory increased by {snapshot.delta_mb:.1f} MB in phase '{snapshot.phase}'",
                current_mb=snapshot.rss_mb,
                threshold_mb=self._spike_threshold,
            )
            self._alerts.append(alert)
            logger.warning(alert.message)

    def track(self, phase: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        関数のメモリ使用を追跡するデコレータ

        Args:
            phase: フェーズ名

        Usage:
            @memory_profiler.track("data_loading")
            def load_data():
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                self.record(f"{phase}_start")
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.record(f"{phase}_end")

            return wrapper

        return decorator

    def get_current_memory(self) -> float:
        """現在のメモリ使用量（MB）を取得"""
        if not PSUTIL_AVAILABLE or self._process is None:
            return 0.0
        return self._process.memory_info().rss / 1024 / 1024

    def get_peak_memory(self) -> float:
        """ピークメモリ使用量（MB）を取得"""
        if not self._snapshots:
            return self.get_current_memory()
        return max(s.rss_mb for s in self._snapshots)

    @property
    def snapshots(self) -> List[MemorySnapshot]:
        """スナップショット一覧"""
        with self._lock:
            return self._snapshots.copy()

    @property
    def alerts(self) -> List[MemoryAlert]:
        """アラート一覧"""
        with self._lock:
            return self._alerts.copy()

    def summary(self) -> str:
        """
        プロファイリングサマリー文字列を生成

        Returns:
            人間可読なサマリー
        """
        with self._lock:
            if not self._snapshots:
                return "No memory snapshots recorded."

            lines = [
                "=" * 60,
                "  MEMORY PROFILER SUMMARY",
                "=" * 60,
                "",
            ]

            # 基本統計
            peak = max(s.rss_mb for s in self._snapshots)
            current = self._snapshots[-1].rss_mb if self._snapshots else 0
            total_delta = current - self._snapshots[0].rss_mb if self._snapshots else 0

            lines.append(f"  Peak Memory:    {peak:.1f} MB")
            lines.append(f"  Current Memory: {current:.1f} MB")
            lines.append(f"  Total Delta:    {total_delta:+.1f} MB")
            lines.append(f"  Snapshots:      {len(self._snapshots)}")
            lines.append(f"  Alerts:         {len(self._alerts)}")
            lines.append("")

            # フェーズ別サマリー
            lines.append("-" * 60)
            lines.append("  Phase History:")
            lines.append("-" * 60)

            for snapshot in self._snapshots[-10:]:  # 最新10件
                lines.append(
                    f"  [{snapshot.phase}] {snapshot.rss_mb:.1f} MB "
                    f"(delta: {snapshot.delta_mb:+.1f} MB)"
                )

            if len(self._snapshots) > 10:
                lines.append(f"  ... and {len(self._snapshots) - 10} more")

            # アラートサマリー
            if self._alerts:
                lines.append("")
                lines.append("-" * 60)
                lines.append("  Alerts:")
                lines.append("-" * 60)

                for alert in self._alerts[-5:]:  # 最新5件
                    lines.append(f"  [{alert.level.upper()}] {alert.message}")

            lines.append("=" * 60)

            return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        with self._lock:
            return {
                "enabled": self._enabled,
                "thresholds": {
                    "warning_mb": self._warning_threshold,
                    "critical_mb": self._critical_threshold,
                    "spike_mb": self._spike_threshold,
                },
                "peak_mb": self.get_peak_memory(),
                "current_mb": self.get_current_memory(),
                "snapshot_count": len(self._snapshots),
                "alert_count": len(self._alerts),
                "snapshots": [s.to_dict() for s in self._snapshots],
                "alerts": [
                    {
                        "level": a.level,
                        "phase": a.phase,
                        "message": a.message,
                        "current_mb": a.current_mb,
                        "threshold_mb": a.threshold_mb,
                    }
                    for a in self._alerts
                ],
            }

    def reset(self) -> None:
        """プロファイラをリセット"""
        with self._lock:
            self._snapshots.clear()
            self._alerts.clear()
            self.record("reset")


# グローバルシングルトンインスタンス
memory_profiler = MemoryProfiler()


# 便利関数
def log_memory(phase: str) -> Optional[MemorySnapshot]:
    """
    現在のメモリ状態を記録（ショートカット）

    Args:
        phase: フェーズ名

    Returns:
        MemorySnapshot
    """
    return memory_profiler.record(phase)


def get_memory_summary() -> str:
    """メモリサマリーを取得（ショートカット）"""
    return memory_profiler.summary()


def get_current_memory_mb() -> float:
    """現在のメモリ使用量を取得（MB）"""
    return memory_profiler.get_current_memory()


def check_memory_threshold(threshold_mb: float) -> bool:
    """
    メモリ使用量が閾値以下かチェック

    Args:
        threshold_mb: 閾値（MB）

    Returns:
        閾値以下ならTrue
    """
    current = memory_profiler.get_current_memory()
    if current > threshold_mb:
        logger.warning(
            f"Memory usage {current:.1f} MB exceeds threshold {threshold_mb:.1f} MB"
        )
        return False
    return True
