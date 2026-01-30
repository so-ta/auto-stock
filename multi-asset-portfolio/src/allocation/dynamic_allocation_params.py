"""
Dynamic Allocation Parameters - アロケーターパラメータの動的計算

AllocatorConfigのパラメータ（w_asset_max, delta_max, smooth_alpha）を
市場環境に応じて動的に計算する。

主要コンポーネント:
- DynamicAllocationParams: 動的パラメータデータクラス
- AllocationDynamicParamsCalculator: パラメータ計算クラス
- calculate_allocation_params: 便利関数

設計根拠:
- 静的パラメータでは市場環境の変化に対応できない
- ボラティリティレジームに応じた適応的なパラメータ調整
- 取引コストを考慮したターンオーバー制御

使用例:
    from src.allocation.dynamic_allocation_params import (
        calculate_allocation_params,
        DynamicAllocationParams,
    )

    # 動的パラメータ計算
    params = calculate_allocation_params(
        num_assets=50,
        returns=returns_series,
        transaction_costs=0.001,
    )
    print(f"w_asset_max: {params.w_asset_max:.4f}")
    print(f"delta_max: {params.delta_max:.4f}")
    print(f"smooth_alpha: {params.smooth_alpha:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ボラティリティレジーム
# =============================================================================

class VolatilityRegime(str, Enum):
    """ボラティリティレジーム"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


def detect_volatility_regime(
    returns: pd.Series,
    lookback_days: int = 60,
) -> VolatilityRegime:
    """ボラティリティレジームを検出

    Args:
        returns: リターン系列
        lookback_days: ルックバック期間

    Returns:
        VolatilityRegime
    """
    if len(returns) < lookback_days:
        return VolatilityRegime.NORMAL

    recent_vol = returns.tail(20).std() * np.sqrt(252)
    long_vol = returns.tail(lookback_days).std() * np.sqrt(252)

    if long_vol == 0:
        return VolatilityRegime.NORMAL

    vol_ratio = recent_vol / long_vol

    if vol_ratio < 0.7:
        return VolatilityRegime.LOW
    elif vol_ratio < 1.3:
        return VolatilityRegime.NORMAL
    elif vol_ratio < 2.0:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.EXTREME


# =============================================================================
# データクラス定義
# =============================================================================

@dataclass
class DynamicAllocationParams:
    """動的アロケーションパラメータ

    Attributes:
        w_asset_max: 単一アセット最大ウェイト
        delta_max: 最大変更量（ターンオーバー制限）
        smooth_alpha: スムージング係数
        num_assets: アセット数
        concentration_limit: 集中度制限
        transaction_costs: 取引コスト
        regime: ボラティリティレジーム
        lookback_days: ルックバック日数
        calculated_at: 計算日時
        metadata: 追加メタデータ
    """

    w_asset_max: float
    delta_max: float
    smooth_alpha: float
    num_assets: int
    concentration_limit: float
    transaction_costs: float
    regime: str
    lookback_days: int
    calculated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "w_asset_max": self.w_asset_max,
            "delta_max": self.delta_max,
            "smooth_alpha": self.smooth_alpha,
            "num_assets": self.num_assets,
            "concentration_limit": self.concentration_limit,
            "transaction_costs": self.transaction_costs,
            "regime": self.regime,
            "lookback_days": self.lookback_days,
            "calculated_at": self.calculated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_allocator_config_kwargs(self) -> dict[str, Any]:
        """AllocatorConfigの初期化引数として使用可能な辞書を返す"""
        return {
            "w_asset_max": self.w_asset_max,
            "delta_max": self.delta_max,
            "smooth_alpha": self.smooth_alpha,
        }


@dataclass
class WAssetMaxResult:
    """w_asset_max計算結果

    Attributes:
        w_asset_max: 計算されたw_asset_max
        num_assets: アセット数
        concentration_limit: 集中度制限
        diversification_limit: 分散化制限（3/num_assets）
        binding_constraint: どの制約が有効か
    """

    w_asset_max: float
    num_assets: int
    concentration_limit: float
    diversification_limit: float
    binding_constraint: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeltaMaxResult:
    """delta_max計算結果

    Attributes:
        delta_max: 計算されたdelta_max
        base_delta: 基準delta
        vol_adjusted_delta: ボラティリティ調整後
        cost_adjusted_delta: コスト調整後
        regime: ボラティリティレジーム
        binding_factor: 制約要因
    """

    delta_max: float
    base_delta: float
    vol_adjusted_delta: float
    cost_adjusted_delta: float
    regime: str
    binding_factor: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SmoothAlphaResult:
    """smooth_alpha計算結果

    Attributes:
        smooth_alpha: 計算されたalpha
        base_alpha: 基準alpha
        regime_multiplier: レジーム乗数
        regime: ボラティリティレジーム
    """

    smooth_alpha: float
    base_alpha: float
    regime_multiplier: float
    regime: str
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# パラメータ計算クラス
# =============================================================================

class AllocationDynamicParamsCalculator:
    """アロケーション動的パラメータ計算クラス

    AllocatorConfigのパラメータを市場環境に応じて動的に計算する。

    Usage:
        calculator = AllocationDynamicParamsCalculator()
        params = calculator.calculate_all(
            num_assets=50,
            returns=returns_series,
            transaction_costs=0.001,
        )
        print(f"w_asset_max: {params.w_asset_max}")
    """

    def __init__(
        self,
        base_delta: float = 0.05,
        base_alpha: float = 0.3,
        default_concentration_limit: float = 0.20,
        vol_sensitivity: float = 0.3,
        cost_sensitivity: float = 50.0,
    ) -> None:
        """初期化

        Args:
            base_delta: 基準delta_max
            base_alpha: 基準smooth_alpha
            default_concentration_limit: デフォルト集中度制限
            vol_sensitivity: ボラティリティ感応度（0〜1）
            cost_sensitivity: コスト感応度
        """
        self.base_delta = base_delta
        self.base_alpha = base_alpha
        self.default_concentration_limit = default_concentration_limit
        self.vol_sensitivity = vol_sensitivity
        self.cost_sensitivity = cost_sensitivity

    def calculate_all(
        self,
        num_assets: int,
        returns: pd.Series | None = None,
        transaction_costs: float = 0.001,
        concentration_limit: float | None = None,
        lookback_days: int = 60,
    ) -> DynamicAllocationParams:
        """全パラメータを計算

        Args:
            num_assets: アセット数
            returns: リターン系列（ボラティリティレジーム検出用）
            transaction_costs: 取引コスト（片道）
            concentration_limit: 集中度制限（Noneの場合はデフォルト使用）
            lookback_days: ルックバック日数

        Returns:
            DynamicAllocationParams
        """
        if concentration_limit is None:
            concentration_limit = self.default_concentration_limit

        # ボラティリティレジーム検出
        if returns is not None and len(returns) >= 20:
            regime = detect_volatility_regime(returns, lookback_days)
        else:
            regime = VolatilityRegime.NORMAL

        # 各パラメータ計算
        w_max_result = self.calculate_w_asset_max(num_assets, concentration_limit)
        delta_result = self.calculate_delta_max(
            returns, transaction_costs, regime
        )
        alpha_result = self.calculate_smooth_alpha(regime)

        return DynamicAllocationParams(
            w_asset_max=w_max_result.w_asset_max,
            delta_max=delta_result.delta_max,
            smooth_alpha=alpha_result.smooth_alpha,
            num_assets=num_assets,
            concentration_limit=concentration_limit,
            transaction_costs=transaction_costs,
            regime=regime.value,
            lookback_days=lookback_days,
            metadata={
                "w_max_details": {
                    "diversification_limit": w_max_result.diversification_limit,
                    "binding_constraint": w_max_result.binding_constraint,
                },
                "delta_details": {
                    "base_delta": delta_result.base_delta,
                    "vol_adjusted": delta_result.vol_adjusted_delta,
                    "cost_adjusted": delta_result.cost_adjusted_delta,
                    "binding_factor": delta_result.binding_factor,
                },
                "alpha_details": {
                    "base_alpha": alpha_result.base_alpha,
                    "regime_multiplier": alpha_result.regime_multiplier,
                },
            },
        )

    def calculate_w_asset_max(
        self,
        num_assets: int,
        concentration_limit: float,
    ) -> WAssetMaxResult:
        """単一アセット最大ウェイトを計算

        w_max = min(concentration_limit, 3 / num_assets)

        - concentration_limit: ポリシーによる上限
        - 3 / num_assets: 分散化による上限（Herfindahl指数を考慮）

        Args:
            num_assets: アセット数
            concentration_limit: 集中度制限

        Returns:
            WAssetMaxResult
        """
        if num_assets <= 0:
            raise ValueError("num_assets must be positive")

        # 分散化制限: 1銘柄への過度な集中を防ぐ
        # 3/N は Herfindahl指数が約0.33以下になる水準
        diversification_limit = 3.0 / num_assets

        # 両方の制限のうち厳しい方を採用
        w_asset_max = min(concentration_limit, diversification_limit)

        # 最低限の値を保証（1銘柄で100%は許可）
        w_asset_max = max(w_asset_max, 1.0 / num_assets)

        # どちらの制約が有効か
        if concentration_limit <= diversification_limit:
            binding = "concentration_limit"
        else:
            binding = "diversification_limit"

        return WAssetMaxResult(
            w_asset_max=w_asset_max,
            num_assets=num_assets,
            concentration_limit=concentration_limit,
            diversification_limit=diversification_limit,
            binding_constraint=binding,
            metadata={
                "effective_max_concentration": num_assets * w_asset_max,
            },
        )

    def calculate_delta_max(
        self,
        returns: pd.Series | None,
        transaction_costs: float,
        regime: VolatilityRegime,
    ) -> DeltaMaxResult:
        """最大変更量（ターンオーバー制限）を計算

        - 高ボラ時: 小さいdelta_max（頻繁な変更を抑制）
        - 低ボラ時: 大きいdelta_max（機会を逃さない）
        - 取引コストが高い場合: delta_maxを下げる

        Args:
            returns: リターン系列
            transaction_costs: 取引コスト（片道）
            regime: ボラティリティレジーム

        Returns:
            DeltaMaxResult
        """
        base_delta = self.base_delta

        # ボラティリティ調整
        # 高ボラ時は急な変更を避ける（ホイップソー対策）
        vol_multiplier_map = {
            VolatilityRegime.LOW: 1.2,      # 低ボラ: やや大きく
            VolatilityRegime.NORMAL: 1.0,   # 通常
            VolatilityRegime.HIGH: 0.7,     # 高ボラ: 控えめに
            VolatilityRegime.EXTREME: 0.5,  # 極端: かなり控えめに
        }
        vol_multiplier = vol_multiplier_map.get(regime, 1.0)
        vol_adjusted_delta = base_delta * vol_multiplier

        # コスト調整
        # 取引コストが高いほどターンオーバーを抑制
        # cost_adjusted = base × (1 - cost_sensitivity × transaction_costs)
        cost_adjustment = 1.0 - self.cost_sensitivity * transaction_costs
        cost_adjustment = max(0.3, cost_adjustment)  # 最低30%は維持
        cost_adjusted_delta = base_delta * cost_adjustment

        # 両方の調整のうち厳しい方を採用
        delta_max = min(vol_adjusted_delta, cost_adjusted_delta)

        # 最低限の変更は許可
        delta_max = max(delta_max, 0.01)

        # どちらの制約が有効か
        if vol_adjusted_delta <= cost_adjusted_delta:
            binding = "volatility"
        else:
            binding = "transaction_cost"

        return DeltaMaxResult(
            delta_max=delta_max,
            base_delta=base_delta,
            vol_adjusted_delta=vol_adjusted_delta,
            cost_adjusted_delta=cost_adjusted_delta,
            regime=regime.value,
            binding_factor=binding,
            metadata={
                "vol_multiplier": vol_multiplier,
                "cost_adjustment": cost_adjustment,
            },
        )

    def calculate_smooth_alpha(
        self,
        regime: VolatilityRegime,
    ) -> SmoothAlphaResult:
        """スムージング係数を計算

        - 高ボラ時: alpha低め（安定重視、急な変更を抑制）
        - 低ボラ時: alpha高め（追従重視、変化に素早く対応）

        alpha = base_alpha × regime_multiplier

        Args:
            regime: ボラティリティレジーム

        Returns:
            SmoothAlphaResult
        """
        base_alpha = self.base_alpha

        # レジームに応じた乗数
        multiplier_map = {
            VolatilityRegime.LOW: 1.5,      # 低ボラ: 追従重視
            VolatilityRegime.NORMAL: 1.0,   # 通常
            VolatilityRegime.HIGH: 0.5,     # 高ボラ: 安定重視
            VolatilityRegime.EXTREME: 0.3,  # 極端: かなり安定重視
        }
        multiplier = multiplier_map.get(regime, 1.0)

        smooth_alpha = base_alpha * multiplier

        # [0, 1]の範囲にクリップ
        smooth_alpha = max(0.0, min(1.0, smooth_alpha))

        return SmoothAlphaResult(
            smooth_alpha=smooth_alpha,
            base_alpha=base_alpha,
            regime_multiplier=multiplier,
            regime=regime.value,
            metadata={
                "effective_half_life": np.log(0.5) / np.log(1 - smooth_alpha)
                if smooth_alpha > 0 and smooth_alpha < 1 else float("inf"),
            },
        )


# =============================================================================
# ユーティリティ関数
# =============================================================================

def calculate_allocation_params(
    num_assets: int,
    returns: pd.Series | None = None,
    transaction_costs: float = 0.001,
    concentration_limit: float = 0.20,
    lookback_days: int = 60,
    base_delta: float = 0.05,
    base_alpha: float = 0.3,
) -> DynamicAllocationParams:
    """アロケーションパラメータを計算（便利関数）

    Args:
        num_assets: アセット数
        returns: リターン系列
        transaction_costs: 取引コスト（片道）
        concentration_limit: 集中度制限
        lookback_days: ルックバック日数
        base_delta: 基準delta_max
        base_alpha: 基準smooth_alpha

    Returns:
        DynamicAllocationParams
    """
    calculator = AllocationDynamicParamsCalculator(
        base_delta=base_delta,
        base_alpha=base_alpha,
        default_concentration_limit=concentration_limit,
    )
    return calculator.calculate_all(
        num_assets=num_assets,
        returns=returns,
        transaction_costs=transaction_costs,
        concentration_limit=concentration_limit,
        lookback_days=lookback_days,
    )


def get_w_asset_max(
    num_assets: int,
    concentration_limit: float = 0.20,
) -> float:
    """w_asset_maxのみを取得

    Args:
        num_assets: アセット数
        concentration_limit: 集中度制限

    Returns:
        w_asset_max
    """
    calculator = AllocationDynamicParamsCalculator()
    result = calculator.calculate_w_asset_max(num_assets, concentration_limit)
    return result.w_asset_max


def get_delta_max(
    returns: pd.Series | None = None,
    transaction_costs: float = 0.001,
    regime: VolatilityRegime | str | None = None,
    base_delta: float = 0.05,
) -> float:
    """delta_maxのみを取得

    Args:
        returns: リターン系列
        transaction_costs: 取引コスト
        regime: ボラティリティレジーム（Noneの場合はreturnsから検出）
        base_delta: 基準delta

    Returns:
        delta_max
    """
    calculator = AllocationDynamicParamsCalculator(base_delta=base_delta)

    if regime is None:
        if returns is not None and len(returns) >= 20:
            vol_regime = detect_volatility_regime(returns)
        else:
            vol_regime = VolatilityRegime.NORMAL
    elif isinstance(regime, str):
        vol_regime = VolatilityRegime(regime)
    else:
        vol_regime = regime

    result = calculator.calculate_delta_max(returns, transaction_costs, vol_regime)
    return result.delta_max


def get_smooth_alpha(
    regime: VolatilityRegime | str | None = None,
    returns: pd.Series | None = None,
    base_alpha: float = 0.3,
) -> float:
    """smooth_alphaのみを取得

    Args:
        regime: ボラティリティレジーム
        returns: リターン系列（regimeがNoneの場合に使用）
        base_alpha: 基準alpha

    Returns:
        smooth_alpha
    """
    calculator = AllocationDynamicParamsCalculator(base_alpha=base_alpha)

    if regime is None:
        if returns is not None and len(returns) >= 20:
            vol_regime = detect_volatility_regime(returns)
        else:
            vol_regime = VolatilityRegime.NORMAL
    elif isinstance(regime, str):
        vol_regime = VolatilityRegime(regime)
    else:
        vol_regime = regime

    result = calculator.calculate_smooth_alpha(vol_regime)
    return result.smooth_alpha


def create_dynamic_allocator_config(
    num_assets: int,
    returns: pd.Series | None = None,
    transaction_costs: float = 0.001,
    concentration_limit: float = 0.20,
    **extra_config: Any,
) -> dict[str, Any]:
    """動的パラメータを使用したAllocatorConfigの引数を生成

    Usage:
        from src.allocation import AllocatorConfig
        from src.allocation.dynamic_allocation_params import create_dynamic_allocator_config

        config_kwargs = create_dynamic_allocator_config(
            num_assets=50,
            returns=returns_series,
            method=AllocationMethod.HRP,  # 追加の引数
        )
        config = AllocatorConfig(**config_kwargs)

    Args:
        num_assets: アセット数
        returns: リターン系列
        transaction_costs: 取引コスト
        concentration_limit: 集中度制限
        **extra_config: AllocatorConfigに渡す追加の引数

    Returns:
        AllocatorConfigの初期化引数辞書
    """
    params = calculate_allocation_params(
        num_assets=num_assets,
        returns=returns,
        transaction_costs=transaction_costs,
        concentration_limit=concentration_limit,
    )

    config_kwargs = params.to_allocator_config_kwargs()
    config_kwargs.update(extra_config)

    return config_kwargs
