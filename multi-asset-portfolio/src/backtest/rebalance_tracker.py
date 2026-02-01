"""
Rebalance Tracker - リバランス追跡と予測精度分析

リバランス時の予測リターンと実績リターンを記録し、
予測精度を分析するためのモジュール。

主な機能:
- リバランス毎の予測・実績リターンを記録
- 予測誤差の統計分析（平均誤差、標準偏差、相関係数）
- 取引コストの累計追跡
- レポート用データ出力

使用例:
    from src.backtest.rebalance_tracker import RebalanceTracker, RebalanceRecord

    tracker = RebalanceTracker()

    # リバランス時に記録
    record = RebalanceRecord(
        date=datetime(2024, 1, 31),
        weights_before={"AAPL": 0.3, "GOOG": 0.2},
        weights_after={"AAPL": 0.4, "GOOG": 0.3},
        expected_returns={"AAPL": 0.01, "GOOG": 0.008},
        expected_portfolio_return=0.009,
        transaction_costs={"AAPL": 0.001, "GOOG": 0.001},
        turnover=0.15,
    )
    tracker.record_rebalance(record)

    # 次回リバランス時に実績を記録
    tracker.update_actual_return(
        date=datetime(2024, 1, 31),
        actual_return=0.012,
    )

    # メトリクス取得
    metrics = tracker.get_forecast_metrics()
    print(f"予測精度相関: {metrics.correlation:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RebalanceRecord:
    """リバランス記録

    リバランス時点での配分変更と予測情報を保持。
    次回リバランス時に実績リターンが記録される。

    Attributes:
        date: リバランス日
        weights_before: リバランス前ウェイト
        weights_after: リバランス後ウェイト
        expected_returns: 各銘柄の期待リターン（リバランス期間ベース）
        expected_portfolio_return: ポートフォリオ期待リターン
        transaction_costs: 各銘柄の取引コスト
        turnover: ターンオーバー（片道）
        actual_return: 実績ポートフォリオリターン（後から記録）
        actual_returns_per_asset: 各銘柄の実績リターン（後から記録）
        metadata: 追加メタデータ
    """

    date: datetime
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    expected_returns: dict[str, float]
    expected_portfolio_return: float
    transaction_costs: dict[str, float]
    turnover: float

    # 次回リバランス時に更新
    actual_return: float | None = None
    actual_returns_per_asset: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def forecast_error(self) -> float | None:
        """予測誤差（実績 - 予測）"""
        if self.actual_return is None:
            return None
        return self.actual_return - self.expected_portfolio_return

    @property
    def total_transaction_cost(self) -> float:
        """合計取引コスト"""
        return sum(self.transaction_costs.values())

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "expected_return": self.expected_portfolio_return,
            "actual_return": self.actual_return,
            "forecast_error": self.forecast_error,
            "turnover": self.turnover,
            "total_cost": self.total_transaction_cost,
            "n_assets": len([w for w in self.weights_after.values() if w > 0]),
        }

    def to_detailed_dict(self) -> dict[str, Any]:
        """詳細辞書に変換"""
        base = self.to_dict()
        base.update({
            "weights_before": self.weights_before,
            "weights_after": self.weights_after,
            "expected_returns_per_asset": self.expected_returns,
            "actual_returns_per_asset": self.actual_returns_per_asset,
            "transaction_costs": self.transaction_costs,
            "metadata": self.metadata,
        })
        return base


@dataclass
class ForecastMetrics:
    """予測精度メトリクス

    Attributes:
        mean_expected: 平均予測リターン
        mean_actual: 平均実績リターン
        mean_error: 平均予測誤差
        std_error: 予測誤差の標準偏差
        correlation: 予測と実績の相関係数
        total_cost: 累計取引コスト
        total_turnover: 累計ターンオーバー
        n_rebalances: リバランス回数
    """

    mean_expected: float
    mean_actual: float
    mean_error: float
    std_error: float
    correlation: float
    total_cost: float
    total_turnover: float
    n_rebalances: int

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "mean_expected": self.mean_expected,
            "mean_actual": self.mean_actual,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "correlation": self.correlation,
            "total_cost": self.total_cost,
            "total_turnover": self.total_turnover,
            "n_rebalances": self.n_rebalances,
        }


class RebalanceTracker:
    """リバランス追跡

    リバランス毎の予測・実績を記録し、予測精度を分析する。

    Example:
        >>> tracker = RebalanceTracker()
        >>> record = RebalanceRecord(...)
        >>> tracker.record_rebalance(record)
        >>> tracker.update_actual_return(date, 0.015)
        >>> metrics = tracker.get_forecast_metrics()
    """

    def __init__(self) -> None:
        self._records: list[RebalanceRecord] = []
        self._date_index: dict[datetime, int] = {}

    def record_rebalance(self, record: RebalanceRecord) -> None:
        """リバランスを記録

        Args:
            record: リバランス記録
        """
        self._date_index[record.date] = len(self._records)
        self._records.append(record)

    def update_actual_return(
        self,
        date: datetime,
        actual_return: float,
        actual_returns_per_asset: dict[str, float] | None = None,
    ) -> bool:
        """実績リターンを更新

        Args:
            date: リバランス日
            actual_return: 実績ポートフォリオリターン
            actual_returns_per_asset: 各銘柄の実績リターン

        Returns:
            bool: 更新成功ならTrue
        """
        idx = self._date_index.get(date)
        if idx is None:
            return False

        self._records[idx].actual_return = actual_return
        if actual_returns_per_asset:
            self._records[idx].actual_returns_per_asset = actual_returns_per_asset
        return True

    def update_previous_actual_return(
        self,
        actual_return: float,
        actual_returns_per_asset: dict[str, float] | None = None,
    ) -> bool:
        """直前のリバランス記録の実績リターンを更新

        次回リバランス時に前回の実績を記録する際に使用。

        Args:
            actual_return: 実績ポートフォリオリターン
            actual_returns_per_asset: 各銘柄の実績リターン

        Returns:
            bool: 更新成功ならTrue
        """
        if len(self._records) < 2:
            return False

        # 最新から2番目（前回リバランス）を更新
        prev_idx = len(self._records) - 2
        self._records[prev_idx].actual_return = actual_return
        if actual_returns_per_asset:
            self._records[prev_idx].actual_returns_per_asset = actual_returns_per_asset
        return True

    def get_records(self) -> list[RebalanceRecord]:
        """全リバランス記録を取得

        Returns:
            list[RebalanceRecord]: リバランス記録リスト
        """
        return self._records.copy()

    def get_completed_records(self) -> list[RebalanceRecord]:
        """実績が記録済みのリバランス記録を取得

        Returns:
            list[RebalanceRecord]: 実績記録済みのリバランス記録リスト
        """
        return [r for r in self._records if r.actual_return is not None]

    def get_forecast_metrics(self) -> ForecastMetrics | None:
        """予測精度メトリクスを計算

        Returns:
            ForecastMetrics: 予測精度メトリクス。データ不足の場合はNone。
        """
        completed = self.get_completed_records()
        if not completed:
            return None

        expected = [r.expected_portfolio_return for r in completed]
        actual = [r.actual_return for r in completed]  # type: ignore
        errors = [r.forecast_error for r in completed]  # type: ignore

        # 相関係数（2件以上必要）
        if len(completed) > 1:
            correlation = float(np.corrcoef(expected, actual)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        return ForecastMetrics(
            mean_expected=float(np.mean(expected)),
            mean_actual=float(np.mean(actual)),
            mean_error=float(np.mean(errors)),
            std_error=float(np.std(errors)) if len(errors) > 1 else 0.0,
            correlation=correlation,
            total_cost=sum(r.total_transaction_cost for r in completed),
            total_turnover=sum(r.turnover for r in completed),
            n_rebalances=len(completed),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrameに変換

        Returns:
            pd.DataFrame: リバランス記録のDataFrame
        """
        return pd.DataFrame([r.to_dict() for r in self._records])

    def to_detailed_dataframe(self) -> pd.DataFrame:
        """詳細DataFrameに変換

        Returns:
            pd.DataFrame: 詳細リバランス記録のDataFrame
        """
        return pd.DataFrame([r.to_detailed_dict() for r in self._records])

    def clear(self) -> None:
        """全記録をクリア"""
        self._records.clear()
        self._date_index.clear()

    def __len__(self) -> int:
        """記録数を返す"""
        return len(self._records)

    def __iter__(self):
        """記録をイテレート"""
        return iter(self._records)
