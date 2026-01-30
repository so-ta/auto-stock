"""
Portfolio Specific Parameters - ポートフォリオ別パラメータ管理

このモジュールは、ポートフォリオ固有のパラメータを管理し、
必要に応じて動的に再計算する機能を提供する。

主要コンポーネント:
- PortfolioSpecificParams: パラメータを保持するデータクラス
- AdaptiveParameterManager: パラメータの取得・更新・永続化を管理
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Any

import pandas as pd

logger = logging.getLogger(__name__)


class ThresholdCalculatorProtocol(Protocol):
    """DynamicThresholdCalculator のプロトコル定義。

    実際の実装は別モジュールで提供される。
    このプロトコルを満たす任意のクラスを使用可能。
    """

    def calculate_rebalance_threshold(
        self,
        returns: pd.Series,
        transaction_costs: float,
    ) -> float:
        """リバランス閾値を計算する。"""
        ...

    def calculate_vix_thresholds(
        self,
        returns: pd.Series,
    ) -> dict[str, float]:
        """VIX閾値を計算する。"""
        ...

    def calculate_correlation_thresholds(
        self,
        returns: pd.Series,
    ) -> dict[str, float]:
        """相関閾値を計算する。"""
        ...

    def calculate_kelly_params(
        self,
        returns: pd.Series,
        transaction_costs: float,
    ) -> dict[str, float]:
        """Kelly基準パラメータを計算する。"""
        ...


@dataclass
class PortfolioSpecificParams:
    """ポートフォリオ固有のパラメータ。

    各ポートフォリオの特性に応じた閾値やパラメータを保持する。

    Attributes:
        portfolio_id: ポートフォリオの一意識別子
        rebalance_threshold: リバランストリガー閾値（例: 0.05 = 5%乖離）
        vix_thresholds: VIX水準の閾値辞書
            - low: 低ボラティリティ閾値
            - high: 高ボラティリティ閾値
            - extreme: 極端な高ボラティリティ閾値
        correlation_thresholds: 相関閾値辞書
            - warning: 警告レベル
            - critical: 危険レベル
            - baseline: 基準レベル
        kelly_params: Kelly基準関連パラメータ
            - full: フルKelly係数
            - half: ハーフKelly係数
            - quarter: クォーターKelly係数
        lookback_start: パラメータ計算に使用したデータの開始日
        lookback_end: パラメータ計算に使用したデータの終了日
        last_updated: 最終更新日時
        update_frequency: 更新頻度（"daily", "weekly", "monthly", "quarterly"）
    """

    portfolio_id: str
    rebalance_threshold: float
    vix_thresholds: dict[str, float]
    correlation_thresholds: dict[str, float]
    kelly_params: dict[str, float]
    lookback_start: datetime
    lookback_end: datetime
    last_updated: datetime
    update_frequency: str = "monthly"

    def __post_init__(self) -> None:
        """バリデーション。"""
        valid_frequencies = {"daily", "weekly", "monthly", "quarterly"}
        if self.update_frequency not in valid_frequencies:
            raise ValueError(
                f"update_frequency must be one of {valid_frequencies}, "
                f"got '{self.update_frequency}'"
            )

        required_vix_keys = {"low", "high", "extreme"}
        if not required_vix_keys.issubset(self.vix_thresholds.keys()):
            missing = required_vix_keys - set(self.vix_thresholds.keys())
            raise ValueError(f"vix_thresholds missing keys: {missing}")

        required_corr_keys = {"warning", "critical", "baseline"}
        if not required_corr_keys.issubset(self.correlation_thresholds.keys()):
            missing = required_corr_keys - set(self.correlation_thresholds.keys())
            raise ValueError(f"correlation_thresholds missing keys: {missing}")

        required_kelly_keys = {"full", "half", "quarter"}
        if not required_kelly_keys.issubset(self.kelly_params.keys()):
            missing = required_kelly_keys - set(self.kelly_params.keys())
            raise ValueError(f"kelly_params missing keys: {missing}")

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        result = asdict(self)
        # datetime を ISO 形式文字列に変換
        result["lookback_start"] = self.lookback_start.isoformat()
        result["lookback_end"] = self.lookback_end.isoformat()
        result["last_updated"] = self.last_updated.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PortfolioSpecificParams:
        """辞書から生成する。"""
        # 文字列を datetime に変換
        data = data.copy()
        for key in ["lookback_start", "lookback_end", "last_updated"]:
            if isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class DefaultThresholdCalculator:
    """デフォルトの閾値計算機。

    DynamicThresholdCalculator が提供されない場合のフォールバック実装。
    シンプルな統計ベースの計算を行う。
    """

    def calculate_rebalance_threshold(
        self,
        returns: pd.Series,
        transaction_costs: float,
    ) -> float:
        """リバランス閾値を計算する。

        取引コストとボラティリティに基づいて閾値を決定。
        """
        volatility = returns.std() * (252 ** 0.5)  # 年率換算
        # 取引コストの2倍 + ボラティリティの10%を閾値とする
        threshold = max(transaction_costs * 2, volatility * 0.1)
        return min(max(threshold, 0.02), 0.15)  # 2%〜15%の範囲に制限

    def calculate_vix_thresholds(
        self,
        returns: pd.Series,
    ) -> dict[str, float]:
        """VIX閾値を計算する。

        リターンのボラティリティ分布に基づいて閾値を決定。
        """
        rolling_vol = returns.rolling(window=21).std() * (252 ** 0.5) * 100
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 21:
            # データ不足の場合はデフォルト値
            return {"low": 15.0, "high": 25.0, "extreme": 35.0}

        return {
            "low": float(rolling_vol.quantile(0.25)),
            "high": float(rolling_vol.quantile(0.75)),
            "extreme": float(rolling_vol.quantile(0.95)),
        }

    def calculate_correlation_thresholds(
        self,
        returns: pd.Series,
    ) -> dict[str, float]:
        """相関閾値を計算する。

        リターンの自己相関に基づいて基準を決定。
        """
        # ラグ1の自己相関を基準として使用
        if len(returns) < 22:
            return {"warning": 0.7, "critical": 0.85, "baseline": 0.5}

        autocorr = returns.autocorr(lag=1)
        baseline = max(0.3, min(0.6, abs(autocorr) + 0.4))

        return {
            "warning": min(baseline + 0.15, 0.85),
            "critical": min(baseline + 0.30, 0.95),
            "baseline": baseline,
        }

    def calculate_kelly_params(
        self,
        returns: pd.Series,
        transaction_costs: float,
    ) -> dict[str, float]:
        """Kelly基準パラメータを計算する。

        Kelly基準: f* = (μ - r) / σ² ≈ μ / σ² (リスクフリーレートを無視)
        取引コストを考慮して調整。
        """
        if len(returns) < 22:
            return {"full": 0.25, "half": 0.125, "quarter": 0.0625}

        mean_return = returns.mean() * 252  # 年率換算
        variance = returns.var() * 252

        if variance <= 0:
            return {"full": 0.25, "half": 0.125, "quarter": 0.0625}

        # Kelly係数の計算（取引コストを控除）
        adjusted_return = mean_return - transaction_costs * 12  # 月次取引を仮定
        full_kelly = max(0.0, adjusted_return / variance)
        full_kelly = min(full_kelly, 1.0)  # 最大100%に制限

        return {
            "full": full_kelly,
            "half": full_kelly * 0.5,
            "quarter": full_kelly * 0.25,
        }


class AdaptiveParameterManager:
    """ポートフォリオパラメータの適応的管理クラス。

    各ポートフォリオの固有パラメータを管理し、
    必要に応じて再計算する機能を提供する。
    キャッシュと永続化をサポート。

    Attributes:
        calculator: 閾値計算を行うオブジェクト
        params_cache: ポートフォリオIDをキーとするパラメータキャッシュ
    """

    # 更新頻度と日数のマッピング
    UPDATE_INTERVALS = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 90,
    }

    def __init__(
        self,
        threshold_calculator: ThresholdCalculatorProtocol | None = None,
    ) -> None:
        """初期化。

        Args:
            threshold_calculator: 閾値計算オブジェクト。
                Noneの場合はデフォルト実装を使用。
        """
        self.calculator = threshold_calculator or DefaultThresholdCalculator()
        self.params_cache: dict[str, PortfolioSpecificParams] = {}

    def get_or_update_params(
        self,
        portfolio_id: str,
        returns: pd.Series,
        transaction_costs: float,
        force_update: bool = False,
    ) -> PortfolioSpecificParams:
        """パラメータを取得する（必要に応じて再計算）。

        Args:
            portfolio_id: ポートフォリオの一意識別子
            returns: リターン系列
            transaction_costs: 取引コスト率
            force_update: 強制的に再計算するか

        Returns:
            ポートフォリオ固有のパラメータ
        """
        if portfolio_id in self.params_cache and not force_update:
            params = self.params_cache[portfolio_id]
            if not self._needs_update(params):
                logger.debug(
                    f"Using cached params for portfolio '{portfolio_id}'"
                )
                return params

        logger.info(f"Calculating params for portfolio '{portfolio_id}'")
        params = self._calculate_params(portfolio_id, returns, transaction_costs)
        self.params_cache[portfolio_id] = params
        return params

    def _needs_update(self, params: PortfolioSpecificParams) -> bool:
        """更新が必要か判定する。

        Args:
            params: 現在のパラメータ

        Returns:
            更新が必要な場合True
        """
        interval_days = self.UPDATE_INTERVALS.get(params.update_frequency, 30)
        elapsed = datetime.now() - params.last_updated
        return elapsed > timedelta(days=interval_days)

    def _calculate_params(
        self,
        portfolio_id: str,
        returns: pd.Series,
        transaction_costs: float,
    ) -> PortfolioSpecificParams:
        """パラメータを計算する。

        Args:
            portfolio_id: ポートフォリオの一意識別子
            returns: リターン系列
            transaction_costs: 取引コスト率

        Returns:
            計算されたパラメータ
        """
        now = datetime.now()

        # ルックバック期間の決定
        if hasattr(returns.index, "min") and hasattr(returns.index, "max"):
            try:
                lookback_start = pd.Timestamp(returns.index.min()).to_pydatetime()
                lookback_end = pd.Timestamp(returns.index.max()).to_pydatetime()
            except Exception:
                lookback_start = now - timedelta(days=len(returns))
                lookback_end = now
        else:
            lookback_start = now - timedelta(days=len(returns))
            lookback_end = now

        return PortfolioSpecificParams(
            portfolio_id=portfolio_id,
            rebalance_threshold=self.calculator.calculate_rebalance_threshold(
                returns, transaction_costs
            ),
            vix_thresholds=self.calculator.calculate_vix_thresholds(returns),
            correlation_thresholds=self.calculator.calculate_correlation_thresholds(
                returns
            ),
            kelly_params=self.calculator.calculate_kelly_params(
                returns, transaction_costs
            ),
            lookback_start=lookback_start,
            lookback_end=lookback_end,
            last_updated=now,
            update_frequency="monthly",
        )

    def save_to_file(self, filepath: str | Path) -> None:
        """パラメータをファイルに保存する。

        Args:
            filepath: 保存先ファイルパス（JSON形式）
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            portfolio_id: params.to_dict()
            for portfolio_id, params in self.params_cache.items()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} portfolio params to {filepath}")

    def load_from_file(self, filepath: str | Path) -> None:
        """パラメータをファイルから読み込む。

        Args:
            filepath: 読み込み元ファイルパス（JSON形式）
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.params_cache.clear()
        for portfolio_id, params_dict in data.items():
            try:
                self.params_cache[portfolio_id] = PortfolioSpecificParams.from_dict(
                    params_dict
                )
            except Exception as e:
                logger.error(f"Failed to load params for '{portfolio_id}': {e}")

        logger.info(f"Loaded {len(self.params_cache)} portfolio params from {filepath}")

    def get_all_portfolios(self) -> list[str]:
        """キャッシュされているポートフォリオIDの一覧を取得する。"""
        return list(self.params_cache.keys())

    def clear_cache(self, portfolio_id: str | None = None) -> None:
        """キャッシュをクリアする。

        Args:
            portfolio_id: 特定のポートフォリオのみクリアする場合に指定。
                Noneの場合は全てクリア。
        """
        if portfolio_id is None:
            self.params_cache.clear()
            logger.info("Cleared all cached params")
        elif portfolio_id in self.params_cache:
            del self.params_cache[portfolio_id]
            logger.info(f"Cleared cached params for '{portfolio_id}'")


def create_adaptive_parameter_manager(
    threshold_calculator: ThresholdCalculatorProtocol | None = None,
    params_file: str | Path | None = None,
) -> AdaptiveParameterManager:
    """AdaptiveParameterManager のファクトリ関数。

    Args:
        threshold_calculator: 閾値計算オブジェクト（オプション）
        params_file: 既存のパラメータファイルパス（オプション）

    Returns:
        初期化されたAdaptiveParameterManager
    """
    manager = AdaptiveParameterManager(threshold_calculator)
    if params_file:
        manager.load_from_file(params_file)
    return manager
