"""
Asset Allocator Module - 配分統合

Covariance + HRP/RP + Constraints + Smoother を統合した
配分計算の中核モジュール。

処理フロー:
1. リターンデータから共分散推定
2. HRP または Risk Parity で基本配分を計算
3. 制約を適用（上限、下限、ターンオーバー）
4. スムージング処理
5. 品質チェックとフォールバック

設計根拠:
- 要求.md §8: 全アセットの重み付け
- 要求.md §8.5: 品質NG時: w=0 または前日維持
- 要求.md §12: 実行手順 Step 7-8

使用方法:
    allocator = AssetAllocator()
    result = allocator.allocate(
        returns=returns_df,
        expected_returns=mu,
        quality_flags=quality,
        previous_weights=prev_w,
    )
    final_weights = result.weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .constraints import ConstraintConfig, ConstraintProcessor, ConstraintResult
from .covariance import CovarianceConfig, CovarianceEstimator, CovarianceMethod, CovarianceResult
from .hrp import HierarchicalRiskParity, HRPConfig, HRPResult
from .risk_parity import RiskParity, RiskParityConfig, RiskParityResult
from .smoother import SmootherConfig, SmoothingResult, WeightSmoother

logger = logging.getLogger(__name__)


class AllocationMethod(str, Enum):
    """配分手法"""

    HRP = "HRP"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VARIANCE = "inverse_variance"


class FallbackReason(str, Enum):
    """フォールバック理由"""

    NONE = "none"
    DATA_QUALITY = "data_quality"
    ESTIMATION_FAILED = "estimation_failed"
    OPTIMIZATION_FAILED = "optimization_failed"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ALL_ASSETS_EXCLUDED = "all_assets_excluded"


@dataclass(frozen=True)
class AllocatorConfig:
    """配分設定

    Attributes:
        method: 配分手法
        covariance_method: 共分散推定手法
        w_asset_max: 単一アセット上限
        w_asset_min: 単一アセット下限
        delta_max: 最大変更量
        smooth_alpha: スムージング係数
        allow_short: ショート許可
        fallback_to_previous: 失敗時に前回の重みを使用
        fallback_to_equal: 前回もない場合に均等配分
        min_assets_required: 最低必要アセット数
    """

    method: AllocationMethod = AllocationMethod.HRP
    covariance_method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF
    w_asset_max: float = 0.2
    w_asset_min: float = 0.0
    delta_max: float = 0.05
    smooth_alpha: float = 0.3
    allow_short: bool = False
    fallback_to_previous: bool = True
    fallback_to_equal: bool = True
    min_assets_required: int = 2

    def __post_init__(self) -> None:
        """バリデーション"""
        if not 0 < self.w_asset_max <= 1:
            raise ValueError("w_asset_max must be in (0, 1]")
        if not 0 <= self.smooth_alpha <= 1:
            raise ValueError("smooth_alpha must be in [0, 1]")


@dataclass
class AllocationResult:
    """配分結果

    Attributes:
        weights: 最終アセット重み（Series）
        raw_weights: 制約・スムージング前の重み
        timestamp: 計算時刻
        method_used: 使用した配分手法
        fallback_reason: フォールバック理由（あれば）
        excluded_assets: 除外されたアセット
        covariance_result: 共分散推定結果
        allocation_result: 配分計算結果（HRP/RP）
        constraint_result: 制約適用結果
        smoothing_result: スムージング結果
        portfolio_metrics: ポートフォリオ指標
        metadata: 追加メタデータ
    """

    weights: pd.Series
    raw_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    method_used: str = ""
    fallback_reason: FallbackReason = FallbackReason.NONE
    excluded_assets: list[str] = field(default_factory=list)
    covariance_result: CovarianceResult | None = None
    allocation_result: HRPResult | RiskParityResult | None = None
    constraint_result: ConstraintResult | None = None
    smoothing_result: SmoothingResult | None = None
    portfolio_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """有効な結果かどうか"""
        if self.weights.empty:
            return False
        if self.weights.isna().any():
            return False
        total = self.weights.sum()
        return np.isclose(total, 1.0, atol=1e-4)

    @property
    def is_fallback(self) -> bool:
        """フォールバックが発生したか"""
        return self.fallback_reason != FallbackReason.NONE

    @property
    def turnover(self) -> float:
        """ターンオーバー"""
        if self.smoothing_result:
            return self.smoothing_result.turnover_after
        if self.constraint_result:
            return self.constraint_result.turnover
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "method_used": self.method_used,
            "fallback_reason": self.fallback_reason.value,
            "excluded_assets": self.excluded_assets,
            "is_valid": self.is_valid,
            "is_fallback": self.is_fallback,
            "turnover": self.turnover,
            "portfolio_metrics": self.portfolio_metrics,
            "metadata": self.metadata,
        }


class AssetAllocator:
    """アセット配分統合クラス

    共分散推定、配分計算、制約処理、スムージングを統合して
    最終的なポートフォリオ重みを計算する。

    Usage:
        config = AllocatorConfig(method=AllocationMethod.HRP)
        allocator = AssetAllocator(config)

        result = allocator.allocate(
            returns=daily_returns_df,
            expected_returns=mu_series,
            quality_flags=quality_df,
            previous_weights=prev_weights,
        )

        print(result.weights)
        print(result.portfolio_metrics)

    Dynamic Parameters:
        use_dynamic=True でデータに基づく動的パラメータを使用可能。
        動的パラメータが計算できない場合はデフォルト値にフォールバック。
    """

    def __init__(
        self,
        config: AllocatorConfig | None = None,
        use_dynamic: bool = False,
        num_assets: int | None = None,
        returns: pd.Series | None = None,
        transaction_costs: float = 0.001,
        concentration_limit: float = 0.20,
    ) -> None:
        """初期化

        Args:
            config: 配分設定。Noneの場合はデフォルト値を使用
            use_dynamic: 動的パラメータを使用するかどうか
            num_assets: アセット数（動的パラメータ計算用）
            returns: リターン系列（動的パラメータ計算用）
            transaction_costs: 取引コスト（動的パラメータ計算用）
            concentration_limit: 集中度制限（動的パラメータ計算用）
        """
        self._use_dynamic = use_dynamic
        self._num_assets = num_assets
        self._returns = returns
        self._transaction_costs = transaction_costs
        self._concentration_limit = concentration_limit

        if use_dynamic and config is None:
            self.config = self._compute_dynamic_config()
        else:
            self.config = config or AllocatorConfig()

        self._init_subcomponents()

    def _compute_dynamic_config(self) -> AllocatorConfig:
        """動的パラメータからConfigを計算

        Returns:
            動的に計算されたAllocatorConfig
        """
        try:
            from .dynamic_allocation_params import calculate_allocation_params

            if self._num_assets is None or self._num_assets <= 0:
                logger.warning(
                    "Dynamic allocator params requested but num_assets not provided. "
                    "Using default config."
                )
                return AllocatorConfig()

            params = calculate_allocation_params(
                num_assets=self._num_assets,
                returns=self._returns,
                transaction_costs=self._transaction_costs,
                concentration_limit=self._concentration_limit,
            )
            return AllocatorConfig(
                w_asset_max=params.w_asset_max,
                delta_max=params.delta_max,
                smooth_alpha=params.smooth_alpha,
            )

        except Exception as e:
            logger.warning(
                "Failed to compute dynamic allocator config: %s. Using defaults.", e
            )
            return AllocatorConfig()

    def _init_subcomponents(self) -> None:
        """サブコンポーネントの初期化"""
        # サブコンポーネントの初期化
        self.cov_estimator = CovarianceEstimator(
            CovarianceConfig(method=self.config.covariance_method)
        )
        self.hrp = HierarchicalRiskParity()
        self.risk_parity = RiskParity()
        self.constraint_processor = ConstraintProcessor(
            ConstraintConfig(
                w_max=self.config.w_asset_max,
                w_min=self.config.w_asset_min,
                delta_max=self.config.delta_max,
                allow_short=self.config.allow_short,
            )
        )
        self.smoother = WeightSmoother(
            SmootherConfig(
                alpha=self.config.smooth_alpha,
                max_single_change=self.config.delta_max,
            )
        )

    def allocate(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        quality_flags: pd.DataFrame | None = None,
        previous_weights: pd.Series | None = None,
        covariance: pd.DataFrame | None = None,
    ) -> AllocationResult:
        """アセット配分を計算

        Args:
            returns: 日次リターン (T x N)
            expected_returns: 期待リターン推定 (N,)。Mean-Variance用
            quality_flags: データ品質フラグ（OK/NG）
            previous_weights: 前期の重み
            covariance: 事前計算済み共分散行列（Noneの場合はreturnsから推定）

        Returns:
            AllocationResult: 配分結果
        """
        timestamp = datetime.utcnow()
        all_assets = returns.columns.tolist()

        # Step 1: 品質フラグでアセットをフィルタ
        valid_assets, excluded_assets = self._filter_by_quality(
            all_assets, quality_flags
        )

        if len(valid_assets) < self.config.min_assets_required:
            logger.warning(
                "Insufficient valid assets: %d < %d",
                len(valid_assets),
                self.config.min_assets_required,
            )
            return self._create_fallback_result(
                all_assets,
                previous_weights,
                excluded_assets,
                FallbackReason.ALL_ASSETS_EXCLUDED,
                timestamp,
            )

        # フィルタ後のリターン
        filtered_returns = returns[valid_assets]

        # Step 2: 共分散推定
        if covariance is not None:
            cov_result = CovarianceResult(
                covariance=covariance.loc[valid_assets, valid_assets],
                correlation=covariance.loc[valid_assets, valid_assets],  # 簡略化
                volatilities=pd.Series(
                    np.sqrt(np.diag(covariance.loc[valid_assets, valid_assets].values)),
                    index=valid_assets,
                ),
            )
        else:
            cov_result = self.cov_estimator.estimate(filtered_returns)

        if not cov_result.is_valid:
            logger.warning("Covariance estimation failed")
            return self._create_fallback_result(
                all_assets,
                previous_weights,
                excluded_assets,
                FallbackReason.ESTIMATION_FAILED,
                timestamp,
                cov_result=cov_result,
            )

        # Step 3: 配分計算
        alloc_result: HRPResult | RiskParityResult
        method = self.config.method

        try:
            if method == AllocationMethod.HRP:
                alloc_result = self.hrp.allocate(
                    cov_result.covariance, cov_result.correlation
                )
            elif method == AllocationMethod.RISK_PARITY:
                alloc_result = self.risk_parity.allocate(cov_result.covariance)
            elif method == AllocationMethod.INVERSE_VARIANCE:
                alloc_result = self._inverse_variance_allocation(cov_result)
            elif method == AllocationMethod.EQUAL_WEIGHT:
                alloc_result = self._equal_weight_allocation(valid_assets)
            elif method == AllocationMethod.MEAN_VARIANCE:
                if expected_returns is None:
                    logger.warning("Mean-Variance requires expected_returns")
                    method = AllocationMethod.HRP
                    alloc_result = self.hrp.allocate(
                        cov_result.covariance, cov_result.correlation
                    )
                else:
                    alloc_result = self._mean_variance_allocation(
                        expected_returns.loc[valid_assets],
                        cov_result.covariance,
                    )
            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            logger.error("Allocation optimization failed: %s", e)
            return self._create_fallback_result(
                all_assets,
                previous_weights,
                excluded_assets,
                FallbackReason.OPTIMIZATION_FAILED,
                timestamp,
                cov_result=cov_result,
            )

        if not alloc_result.is_valid:
            logger.warning("Allocation result invalid")
            return self._create_fallback_result(
                all_assets,
                previous_weights,
                excluded_assets,
                FallbackReason.OPTIMIZATION_FAILED,
                timestamp,
                cov_result=cov_result,
            )

        raw_weights = alloc_result.weights

        # 除外アセットを0で追加
        for asset in excluded_assets:
            raw_weights[asset] = 0.0
        raw_weights = raw_weights.reindex(all_assets, fill_value=0.0)

        # Step 4: 制約適用
        constraint_result = self.constraint_processor.apply(
            raw_weights, previous_weights
        )

        # Step 5: スムージング
        smoothing_result = self.smoother.smooth(
            constraint_result.weights, previous_weights
        )

        final_weights = smoothing_result.weights

        # Step 6: ポートフォリオ指標計算
        portfolio_metrics = self._compute_portfolio_metrics(
            final_weights.loc[valid_assets],
            cov_result.covariance,
            expected_returns.loc[valid_assets] if expected_returns is not None else None,
        )

        logger.info(
            "Allocation completed: %s, %d assets, vol=%.4f",
            method.value,
            len(valid_assets),
            portfolio_metrics.get("volatility", 0),
        )

        return AllocationResult(
            weights=final_weights,
            raw_weights=raw_weights,
            timestamp=timestamp,
            method_used=method.value,
            fallback_reason=FallbackReason.NONE,
            excluded_assets=excluded_assets,
            covariance_result=cov_result,
            allocation_result=alloc_result,
            constraint_result=constraint_result,
            smoothing_result=smoothing_result,
            portfolio_metrics=portfolio_metrics,
        )

    def _filter_by_quality(
        self,
        assets: list[str],
        quality_flags: pd.DataFrame | None,
    ) -> tuple[list[str], list[str]]:
        """品質フラグでアセットをフィルタ

        Args:
            assets: 全アセットリスト
            quality_flags: 品質フラグ（True=OK, False=NG）

        Returns:
            (有効アセット, 除外アセット)
        """
        if quality_flags is None:
            return assets, []

        valid = []
        excluded = []

        for asset in assets:
            if asset in quality_flags.columns:
                # 最新の品質フラグを確認
                is_ok = quality_flags[asset].iloc[-1] if not quality_flags.empty else True
                if is_ok:
                    valid.append(asset)
                else:
                    excluded.append(asset)
            else:
                valid.append(asset)

        return valid, excluded

    def _inverse_variance_allocation(
        self, cov_result: CovarianceResult
    ) -> RiskParityResult:
        """逆分散配分（ナイーブRP）

        Args:
            cov_result: 共分散推定結果

        Returns:
            RiskParityResult
        """
        from .risk_parity import NaiveRiskParity

        naive_rp = NaiveRiskParity()
        return naive_rp.allocate(cov_result.covariance)

    def _equal_weight_allocation(self, assets: list[str]) -> HRPResult:
        """均等配分

        Args:
            assets: アセットリスト

        Returns:
            HRPResult（均等配分）
        """
        n = len(assets)
        weights = pd.Series(np.ones(n) / n, index=assets)
        return HRPResult(
            weights=weights,
            metadata={"method": "equal_weight"},
        )

    def _mean_variance_allocation(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> HRPResult:
        """平均分散最適化（正則化付き）

        Args:
            expected_returns: 期待リターン
            covariance: 共分散行列

        Returns:
            HRPResult
        """
        from scipy.optimize import minimize

        n = len(expected_returns)
        mu = expected_returns.values
        cov = covariance.values

        # 正則化: μを0方向に縮小
        shrinkage = 0.5
        mu_shrunk = mu * (1 - shrinkage)

        # 目的関数: 期待効用 = μ'w - λ/2 * w'Σw
        risk_aversion = 2.0

        def objective(w: np.ndarray) -> float:
            return -(mu_shrunk @ w - risk_aversion / 2 * w @ cov @ w)

        # 制約と境界
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, self.config.w_asset_max) for _ in range(n)]

        # 初期値: 均等配分
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return HRPResult(
            weights=pd.Series(weights, index=expected_returns.index),
            metadata={
                "method": "mean_variance",
                "risk_aversion": risk_aversion,
                "shrinkage": shrinkage,
            },
        )

    def _compute_portfolio_metrics(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame,
        expected_returns: pd.Series | None,
    ) -> dict[str, float]:
        """ポートフォリオ指標を計算

        Args:
            weights: 重み
            covariance: 共分散行列
            expected_returns: 期待リターン

        Returns:
            指標辞書
        """
        w = weights.values
        cov = covariance.loc[weights.index, weights.index].values

        # ボラティリティ
        port_var = w @ cov @ w
        port_vol = np.sqrt(max(0, port_var))

        metrics: dict[str, float] = {
            "volatility": port_vol,
            "variance": port_var,
            "effective_n": 1.0 / (w**2).sum() if (w**2).sum() > 0 else 0,
            "max_weight": float(w.max()),
            "min_weight": float(w[w > 0].min()) if (w > 0).any() else 0,
            "herfindahl": float((w**2).sum()),
        }

        if expected_returns is not None:
            mu = expected_returns.loc[weights.index].values
            port_return = mu @ w
            metrics["expected_return"] = port_return
            if port_vol > 0:
                metrics["sharpe_ratio"] = port_return / port_vol

        return metrics

    def _create_fallback_result(
        self,
        all_assets: list[str],
        previous_weights: pd.Series | None,
        excluded_assets: list[str],
        reason: FallbackReason,
        timestamp: datetime,
        cov_result: CovarianceResult | None = None,
    ) -> AllocationResult:
        """フォールバック結果を作成

        Args:
            all_assets: 全アセット
            previous_weights: 前回の重み
            excluded_assets: 除外アセット
            reason: フォールバック理由
            timestamp: タイムスタンプ
            cov_result: 共分散結果（あれば）

        Returns:
            AllocationResult
        """
        # フォールバック戦略
        if self.config.fallback_to_previous and previous_weights is not None:
            weights = previous_weights.reindex(all_assets, fill_value=0.0)
            method = "previous_weights"
        elif self.config.fallback_to_equal:
            n = len(all_assets)
            weights = pd.Series(np.ones(n) / n, index=all_assets)
            # 除外アセットは0に
            for asset in excluded_assets:
                weights[asset] = 0.0
            if weights.sum() > 0:
                weights = weights / weights.sum()
            method = "equal_weight"
        else:
            # 全てキャッシュ
            weights = pd.Series(np.zeros(len(all_assets)), index=all_assets)
            method = "cash"

        logger.warning(
            "Fallback applied: %s -> %s",
            reason.value,
            method,
        )

        return AllocationResult(
            weights=weights,
            timestamp=timestamp,
            method_used=method,
            fallback_reason=reason,
            excluded_assets=excluded_assets,
            covariance_result=cov_result,
            metadata={"fallback_method": method},
        )


def create_allocator_from_settings() -> AssetAllocator:
    """グローバル設定からAllocatorを生成

    Returns:
        設定済みのAssetAllocator
    """
    try:
        from src.config.settings import AllocationMethod as SettingsAllocationMethod
        from src.config.settings import get_settings

        settings = get_settings()

        # 設定のAllocationMethodをローカルのenumに変換
        method_map = {
            SettingsAllocationMethod.HRP: AllocationMethod.HRP,
            SettingsAllocationMethod.RISK_PARITY: AllocationMethod.RISK_PARITY,
            SettingsAllocationMethod.MEAN_VARIANCE: AllocationMethod.MEAN_VARIANCE,
            SettingsAllocationMethod.EQUAL_WEIGHT: AllocationMethod.EQUAL_WEIGHT,
        }

        config = AllocatorConfig(
            method=method_map.get(
                settings.asset_allocation.method, AllocationMethod.HRP
            ),
            w_asset_max=settings.asset_allocation.w_asset_max,
            w_asset_min=settings.asset_allocation.w_asset_min,
            delta_max=settings.asset_allocation.delta_max,
            smooth_alpha=settings.asset_allocation.smooth_alpha,
            allow_short=settings.asset_allocation.allow_short,
        )
        return AssetAllocator(config)
    except ImportError:
        logger.warning("Settings not available, using default AllocatorConfig")
        return AssetAllocator()
