"""
Correlation Break Detection - 相関ブレイク検出

このモジュールは、ポートフォリオ内の相関構造の崩壊を検出し、
適切なリスク対応を提案する機能を提供する。

主要コンポーネント:
- CorrelationBreakDetector: 相関ブレイク検出クラス
- WarningLevel: 警告レベル（NORMAL, WARNING, CRITICAL）
- CorrelationBreakResult: 検出結果

設計根拠:
- 市場ストレス時に相関構造が崩壊することがある
- 相関構造の変化は分散効果の減少を意味する
- 早期検出によりリスク管理の改善が可能
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WarningLevel(str, Enum):
    """警告レベル。"""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CorrelationChangeResult:
    """相関変化の計測結果。

    Attributes:
        change_magnitude: 変化量（フロベニウスノルム）
        short_window_corr: 短期相関行列
        long_window_corr: 長期相関行列
        difference_matrix: 差分行列
        max_change_pair: 最大変化ペア（asset1, asset2, change）
        avg_correlation_short: 短期平均相関
        avg_correlation_long: 長期平均相関
    """

    change_magnitude: float
    short_window_corr: pd.DataFrame
    long_window_corr: pd.DataFrame
    difference_matrix: pd.DataFrame
    max_change_pair: tuple[str, str, float] | None = None
    avg_correlation_short: float = 0.0
    avg_correlation_long: float = 0.0


@dataclass
class CorrelationBreakResult:
    """相関ブレイク検出結果。

    Attributes:
        warning_level: 警告レベル
        change_magnitude: 変化量
        is_break_detected: ブレイクが検出されたか
        baseline_correlation: ベースライン相関行列
        current_correlation: 現在の相関行列
        affected_pairs: 影響を受けたペアのリスト
        timestamp: 検出日時
        metadata: 追加情報
    """

    warning_level: WarningLevel
    change_magnitude: float
    is_break_detected: bool
    baseline_correlation: pd.DataFrame | None = None
    current_correlation: pd.DataFrame | None = None
    affected_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "warning_level": self.warning_level.value,
            "change_magnitude": self.change_magnitude,
            "is_break_detected": self.is_break_detected,
            "affected_pairs": [
                {"asset1": p[0], "asset2": p[1], "change": p[2]}
                for p in self.affected_pairs
            ],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AdjustmentResult:
    """配分調整結果。

    Attributes:
        original_weights: 元のウェイト
        adjusted_weights: 調整後ウェイト
        warning_level: 適用された警告レベル
        adjustment_reason: 調整理由
        cash_added: 追加されたキャッシュ比率
    """

    original_weights: dict[str, float]
    adjusted_weights: dict[str, float]
    warning_level: WarningLevel
    adjustment_reason: str
    cash_added: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return {
            "original_weights": self.original_weights,
            "adjusted_weights": self.adjusted_weights,
            "warning_level": self.warning_level.value,
            "adjustment_reason": self.adjustment_reason,
            "cash_added": self.cash_added,
        }


@dataclass
class CorrelationBreakConfig:
    """相関ブレイク検出の設定。

    Attributes:
        warning_threshold: 警告レベル閾値
        critical_threshold: 危険レベル閾値
        short_window: 短期ウィンドウ
        long_window: 長期ウィンドウ
        baseline_lookback: ベースライン計算期間
        min_observations: 最小観測数
    """

    warning_threshold: float = 0.3
    critical_threshold: float = 0.5
    short_window: int = 20
    long_window: int = 60
    baseline_lookback: int = 252
    min_observations: int = 30

    def __post_init__(self) -> None:
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("warning_threshold must be less than critical_threshold")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")


class CorrelationBreakDetector:
    """相関ブレイク検出クラス。

    相関構造の崩壊を検出し、適切なリスク対応を提案する。

    Usage:
        detector = CorrelationBreakDetector()

        # 相関ブレイク検出
        result = detector.detect_correlation_break(returns)
        if result.is_break_detected:
            print(f"Warning: {result.warning_level.value}")

        # 配分調整
        adjustment = detector.adjust_for_correlation_break(
            base_weights=weights,
            warning_level=result.warning_level,
        )
    """

    def __init__(self, config: CorrelationBreakConfig | None = None) -> None:
        """初期化。

        Args:
            config: 設定
        """
        self.config = config or CorrelationBreakConfig()

    def detect_correlation_break(
        self,
        returns: pd.DataFrame,
        lookback: int | None = None,
        baseline_lookback: int | None = None,
    ) -> CorrelationBreakResult:
        """相関ブレイクを検出する。

        Args:
            returns: リターンデータ（columns=銘柄）
            lookback: 直近の計算期間
            baseline_lookback: ベースライン計算期間

        Returns:
            検出結果
        """
        lookback = lookback or self.config.long_window
        baseline_lookback = baseline_lookback or self.config.baseline_lookback

        if len(returns) < baseline_lookback:
            logger.warning(
                f"Insufficient data for baseline_lookback={baseline_lookback}"
            )
            baseline_lookback = len(returns)

        if len(returns) < lookback:
            lookback = len(returns)

        # ベースライン相関（長期）
        baseline_returns = returns.iloc[-baseline_lookback:]
        baseline_corr = baseline_returns.corr()

        # 現在の相関（直近）
        current_returns = returns.iloc[-lookback:]
        current_corr = current_returns.corr()

        # 変化量の計算
        change_result = self._calculate_correlation_change_internal(
            baseline_corr, current_corr
        )

        # 警告レベルの判定
        warning_level = self.get_warning_level(change_result.change_magnitude)

        # 影響を受けたペアの特定
        affected_pairs = self._get_affected_pairs(
            change_result.difference_matrix,
            threshold=self.config.warning_threshold,
        )

        is_break_detected = warning_level != WarningLevel.NORMAL

        logger.info(
            f"Correlation break detection: level={warning_level.value}, "
            f"magnitude={change_result.change_magnitude:.4f}, "
            f"affected_pairs={len(affected_pairs)}"
        )

        return CorrelationBreakResult(
            warning_level=warning_level,
            change_magnitude=change_result.change_magnitude,
            is_break_detected=is_break_detected,
            baseline_correlation=baseline_corr,
            current_correlation=current_corr,
            affected_pairs=affected_pairs,
            metadata={
                "lookback": lookback,
                "baseline_lookback": baseline_lookback,
                "avg_corr_baseline": change_result.avg_correlation_long,
                "avg_corr_current": change_result.avg_correlation_short,
            },
        )

    def calculate_correlation_change(
        self,
        returns: pd.DataFrame,
        short_window: int | None = None,
        long_window: int | None = None,
    ) -> CorrelationChangeResult:
        """相関変化を計算する。

        Args:
            returns: リターンデータ
            short_window: 短期ウィンドウ
            long_window: 長期ウィンドウ

        Returns:
            相関変化の計測結果
        """
        short_window = short_window or self.config.short_window
        long_window = long_window or self.config.long_window

        if len(returns) < long_window:
            long_window = len(returns)
        if len(returns) < short_window:
            short_window = len(returns)

        # 短期相関
        short_returns = returns.iloc[-short_window:]
        short_corr = short_returns.corr()

        # 長期相関
        long_returns = returns.iloc[-long_window:]
        long_corr = long_returns.corr()

        return self._calculate_correlation_change_internal(long_corr, short_corr)

    def _calculate_correlation_change_internal(
        self,
        baseline_corr: pd.DataFrame,
        current_corr: pd.DataFrame,
    ) -> CorrelationChangeResult:
        """相関変化を内部計算する。"""
        # 差分行列
        diff_matrix = current_corr - baseline_corr

        # フロベニウスノルムで変化量を計測
        # 対角成分（=1）を除外して計算
        n = len(diff_matrix)
        mask = ~np.eye(n, dtype=bool)
        off_diagonal_diff = diff_matrix.values[mask]
        change_magnitude = np.sqrt(np.sum(off_diagonal_diff ** 2)) / (n * (n - 1))

        # 最大変化ペア
        max_change_pair = None
        if n > 1:
            # 対角成分を除いた最大変化を探す
            diff_abs = np.abs(diff_matrix.values)
            np.fill_diagonal(diff_abs, 0)
            max_idx = np.unravel_index(np.argmax(diff_abs), diff_abs.shape)
            max_change = diff_matrix.iloc[max_idx[0], max_idx[1]]
            max_change_pair = (
                diff_matrix.index[max_idx[0]],
                diff_matrix.columns[max_idx[1]],
                float(max_change),
            )

        # 平均相関（対角成分除外）
        baseline_off_diag = baseline_corr.values[mask]
        current_off_diag = current_corr.values[mask]
        avg_correlation_long = np.mean(baseline_off_diag)
        avg_correlation_short = np.mean(current_off_diag)

        return CorrelationChangeResult(
            change_magnitude=float(change_magnitude),
            short_window_corr=current_corr,
            long_window_corr=baseline_corr,
            difference_matrix=diff_matrix,
            max_change_pair=max_change_pair,
            avg_correlation_short=float(avg_correlation_short),
            avg_correlation_long=float(avg_correlation_long),
        )

    def get_warning_level(
        self,
        change_magnitude: float,
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
    ) -> WarningLevel:
        """警告レベルを判定する。

        Args:
            change_magnitude: 変化量
            warning_threshold: 警告閾値
            critical_threshold: 危険閾値

        Returns:
            警告レベル
        """
        warning_threshold = warning_threshold or self.config.warning_threshold
        critical_threshold = critical_threshold or self.config.critical_threshold

        if change_magnitude >= critical_threshold:
            return WarningLevel.CRITICAL
        elif change_magnitude >= warning_threshold:
            return WarningLevel.WARNING
        else:
            return WarningLevel.NORMAL

    def _get_affected_pairs(
        self,
        difference_matrix: pd.DataFrame,
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """影響を受けたペアを特定する。"""
        affected = []
        n = len(difference_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                change = difference_matrix.iloc[i, j]
                if abs(change) >= threshold:
                    affected.append((
                        difference_matrix.index[i],
                        difference_matrix.columns[j],
                        float(change),
                    ))

        # 変化量の絶対値でソート
        affected.sort(key=lambda x: abs(x[2]), reverse=True)
        return affected

    def adjust_for_correlation_break(
        self,
        base_weights: dict[str, float],
        warning_level: WarningLevel,
        returns: pd.DataFrame | None = None,
        cash_ticker: str = "SHY",
        warning_cash_increase: float = 0.10,
        critical_cash_increase: float = 0.25,
    ) -> AdjustmentResult:
        """相関ブレイクに応じて配分を調整する。

        Args:
            base_weights: ベースウェイト
            warning_level: 警告レベル
            returns: リターンデータ（分散度調整用）
            cash_ticker: キャッシュのティッカー
            warning_cash_increase: WARNING時のキャッシュ増加率
            critical_cash_increase: CRITICAL時のキャッシュ増加率

        Returns:
            調整結果
        """
        if warning_level == WarningLevel.NORMAL:
            return AdjustmentResult(
                original_weights=base_weights,
                adjusted_weights=base_weights.copy(),
                warning_level=warning_level,
                adjustment_reason="No adjustment needed",
                cash_added=0.0,
            )

        adjusted = base_weights.copy()
        cash_added = 0.0

        if warning_level == WarningLevel.WARNING:
            # WARNING: 分散度を高める
            if returns is not None:
                adjusted = self._increase_diversification(
                    adjusted, returns
                )
            adjustment_reason = "Increased diversification due to correlation instability"

        elif warning_level == WarningLevel.CRITICAL:
            # CRITICAL: キャッシュ比率を上げる
            cash_added = critical_cash_increase

            # 既存のキャッシュウェイトを取得
            current_cash = adjusted.get(cash_ticker, 0.0)

            # 全ウェイトを縮小してキャッシュを追加
            scale_factor = (1.0 - cash_added) / (1.0 - current_cash) if current_cash < 1.0 else 0.0

            for ticker in adjusted:
                if ticker != cash_ticker:
                    adjusted[ticker] *= scale_factor

            adjusted[cash_ticker] = current_cash + cash_added * (1.0 - current_cash)

            adjustment_reason = f"Increased cash allocation by {cash_added:.1%} due to critical correlation break"

        # 正規化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        logger.info(
            f"Correlation break adjustment: level={warning_level.value}, "
            f"cash_added={cash_added:.2%}"
        )

        return AdjustmentResult(
            original_weights=base_weights,
            adjusted_weights=adjusted,
            warning_level=warning_level,
            adjustment_reason=adjustment_reason,
            cash_added=cash_added,
        )

    def _increase_diversification(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """分散度を高める調整を行う。

        相関が低い資産のウェイトを増やし、高い資産のウェイトを減らす。
        """
        corr = returns.corr()
        available_tickers = [t for t in weights.keys() if t in corr.columns]

        if len(available_tickers) < 2:
            return weights

        # 各資産の平均相関を計算
        avg_correlations = {}
        for ticker in available_tickers:
            other_tickers = [t for t in available_tickers if t != ticker]
            if other_tickers:
                avg_corr = corr.loc[ticker, other_tickers].mean()
                avg_correlations[ticker] = avg_corr

        if not avg_correlations:
            return weights

        # 相関が低い資産を増やし、高い資産を減らす
        adjusted = weights.copy()
        corr_values = list(avg_correlations.values())
        corr_mean = np.mean(corr_values)
        corr_std = np.std(corr_values) if len(corr_values) > 1 else 1.0

        for ticker, avg_corr in avg_correlations.items():
            if corr_std > 0:
                # zスコアに基づく調整
                z_score = (avg_corr - corr_mean) / corr_std
                # 高相関（z > 0）: ウェイト減少、低相関（z < 0）: ウェイト増加
                adjustment_factor = 1.0 - z_score * 0.1  # 最大±10%調整
                adjustment_factor = np.clip(adjustment_factor, 0.8, 1.2)
                adjusted[ticker] *= adjustment_factor

        return adjusted

    def compute_rolling_correlation_change(
        self,
        returns: pd.DataFrame,
        short_window: int | None = None,
        long_window: int | None = None,
        step: int = 1,
    ) -> pd.Series:
        """ローリング相関変化を計算する。

        Args:
            returns: リターンデータ
            short_window: 短期ウィンドウ
            long_window: 長期ウィンドウ
            step: ステップサイズ

        Returns:
            相関変化の時系列
        """
        short_window = short_window or self.config.short_window
        long_window = long_window or self.config.long_window

        if len(returns) < long_window:
            return pd.Series(dtype=float)

        changes = []
        dates = []

        for i in range(long_window, len(returns), step):
            window_returns = returns.iloc[:i]
            result = self.calculate_correlation_change(
                window_returns,
                short_window=short_window,
                long_window=long_window,
            )
            changes.append(result.change_magnitude)
            dates.append(returns.index[i - 1])

        return pd.Series(changes, index=dates)


def create_correlation_break_detector(
    warning_threshold: float = 0.3,
    critical_threshold: float = 0.5,
    short_window: int = 20,
    long_window: int = 60,
) -> CorrelationBreakDetector:
    """CorrelationBreakDetector のファクトリ関数。

    Args:
        warning_threshold: 警告閾値
        critical_threshold: 危険閾値
        short_window: 短期ウィンドウ
        long_window: 長期ウィンドウ

    Returns:
        初期化された CorrelationBreakDetector
    """
    config = CorrelationBreakConfig(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        short_window=short_window,
        long_window=long_window,
    )
    return CorrelationBreakDetector(config=config)


def quick_detect_correlation_break(
    returns: pd.DataFrame,
    warning_threshold: float = 0.3,
    critical_threshold: float = 0.5,
) -> tuple[WarningLevel, float]:
    """便利関数: 相関ブレイクを簡易検出する。

    Args:
        returns: リターンデータ
        warning_threshold: 警告閾値
        critical_threshold: 危険閾値

    Returns:
        (警告レベル, 変化量)
    """
    detector = create_correlation_break_detector(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    result = detector.detect_correlation_break(returns)
    return result.warning_level, result.change_magnitude
