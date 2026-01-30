"""
Dynamic Weighter Module - 動的重み調整

市場状況に応じて戦略の重みを動的に調整する。

機能:
1. VolatilityScaling: ボラティリティに応じた配分調整
2. DrawdownProtection: ドローダウン保護
3. RegimeBasedWeighting: レジームに応じた戦略選択

設計根拠:
- 高ボラ時はリスク縮小、低ボラ時はリスク拡大
- ドローダウンが閾値を超えたら防御モード
- RegimeDetector（ensemble.py）の出力を活用

計算式:
    VolatilityScaling: scale_factor = target_vol / realized_vol
    DrawdownProtection: cash_increase = (dd - threshold) * sensitivity
    RegimeWeighting: strategy_weight *= regime_modifier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamicWeightingConfig:
    """動的重み調整の設定

    Attributes:
        target_volatility: 目標ボラティリティ（年率）
        max_drawdown_trigger: ドローダウン防御モード発動閾値
        regime_lookback_days: レジーム判定期間（日）
        vol_scaling_enabled: ボラティリティスケーリング有効
        dd_protection_enabled: ドローダウン保護有効
        regime_weighting_enabled: レジームベース重み付け有効
        vol_floor: ボラティリティスケールの下限（過大なレバレッジ防止）
        vol_cap: ボラティリティスケールの上限
        dd_recovery_rate: ドローダウン回復時の戻し速度（日次）
        cash_symbol: 現金のシンボル名
        adaptive_regime_enabled: レジーム適応アロケーション有効
    """

    target_volatility: float = 0.15
    max_drawdown_trigger: float = 0.10
    regime_lookback_days: int = 60
    vol_scaling_enabled: bool = True
    dd_protection_enabled: bool = True
    regime_weighting_enabled: bool = True
    vol_floor: float = 0.5
    vol_cap: float = 2.0
    dd_recovery_rate: float = 0.02
    cash_symbol: str = "CASH"
    adaptive_regime_enabled: bool = True

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be > 0")
        if self.max_drawdown_trigger <= 0 or self.max_drawdown_trigger >= 1:
            raise ValueError("max_drawdown_trigger must be in (0, 1)")
        if self.regime_lookback_days <= 0:
            raise ValueError("regime_lookback_days must be > 0")
        if self.vol_floor <= 0 or self.vol_floor > self.vol_cap:
            raise ValueError("vol_floor must be > 0 and <= vol_cap")


# =============================================================================
# Regime Adaptive Allocation - レジーム適応アロケーション
# =============================================================================
@dataclass(frozen=True)
class RegimeCondition:
    """レジーム判定条件

    Attributes:
        momentum_20d_threshold: 20日モメンタム閾値
        momentum_60d_threshold: 60日モメンタム閾値（オプション）
        breadth_threshold: 市場幅（上昇銘柄比率）閾値
        vix_threshold: VIX閾値
        condition_type: 条件タイプ（"all" = AND, "any" = OR）
    """

    momentum_20d_threshold: float | None = None
    momentum_60d_threshold: float | None = None
    breadth_threshold: float | None = None
    vix_threshold: float | None = None
    condition_type: str = "all"  # "all" or "any"


@dataclass(frozen=True)
class RegimeAdjustment:
    """レジーム適応調整パラメータ

    Attributes:
        momentum_weight: モメンタム戦略の重み倍率
        defensive_weight: ディフェンシブ戦略の重み倍率
        cash_target: キャッシュ目標比率
        description: レジームの説明
    """

    momentum_weight: float = 1.0
    defensive_weight: float = 1.0
    cash_target: float = 0.10
    description: str = ""


@dataclass
class AdaptiveRegimeConfig:
    """レジーム適応アロケーション設定

    計画の task_012_7 に基づくレジーム定義:
    - bull_trend: momentum_20d > 0.05, breadth > 0.6, vix < 20
    - bear_market: momentum_60d < -0.10, vix > 25
    """

    regimes: dict[str, tuple[RegimeCondition, RegimeAdjustment]]

    @classmethod
    def default(cls) -> "AdaptiveRegimeConfig":
        """デフォルト設定を作成"""
        return cls(
            regimes={
                # Bull Trend: 強気トレンド
                "bull_trend": (
                    RegimeCondition(
                        momentum_20d_threshold=0.05,  # > 5%
                        breadth_threshold=0.60,  # > 60%
                        vix_threshold=20.0,  # < 20
                        condition_type="all",
                    ),
                    RegimeAdjustment(
                        momentum_weight=1.3,  # モメンタム重視
                        defensive_weight=0.7,
                        cash_target=0.05,  # キャッシュ最小化
                        description="Bull trend: favor momentum, minimize cash",
                    ),
                ),
                # Bear Market: 弱気相場
                "bear_market": (
                    RegimeCondition(
                        momentum_60d_threshold=-0.10,  # < -10%
                        vix_threshold=25.0,  # > 25
                        condition_type="all",
                    ),
                    RegimeAdjustment(
                        momentum_weight=0.5,  # モメンタム縮小
                        defensive_weight=1.5,  # ディフェンシブ重視
                        cash_target=0.30,  # キャッシュ大幅増
                        description="Bear market: reduce risk, increase cash",
                    ),
                ),
                # High Volatility Crisis: 高ボラクライシス
                "crisis": (
                    RegimeCondition(
                        vix_threshold=30.0,  # > 30
                        condition_type="any",
                    ),
                    RegimeAdjustment(
                        momentum_weight=0.3,
                        defensive_weight=1.2,
                        cash_target=0.40,
                        description="Crisis: maximum defense",
                    ),
                ),
                # Recovery: 回復局面
                "recovery": (
                    RegimeCondition(
                        momentum_20d_threshold=0.03,  # > 3%
                        momentum_60d_threshold=-0.05,  # > -5% (改善中)
                        vix_threshold=25.0,  # < 25 (落ち着き始め)
                        condition_type="all",
                    ),
                    RegimeAdjustment(
                        momentum_weight=1.1,
                        defensive_weight=1.0,
                        cash_target=0.15,
                        description="Recovery: cautious optimism",
                    ),
                ),
                # Range Bound: レンジ相場
                "range_bound": (
                    RegimeCondition(
                        momentum_20d_threshold=0.02,  # |momentum| < 2%
                        vix_threshold=18.0,  # < 18
                        condition_type="all",
                    ),
                    RegimeAdjustment(
                        momentum_weight=0.8,
                        defensive_weight=1.2,
                        cash_target=0.10,
                        description="Range: favor mean reversion",
                    ),
                ),
            }
        )


class RegimeAdaptiveAllocator:
    """レジーム適応アロケータ

    市場レジームを検出し、レジームに応じたアロケーション調整を行う。

    使用例:
        config = AdaptiveRegimeConfig.default()
        allocator = RegimeAdaptiveAllocator(config)

        market_state = {
            "momentum_20d": 0.08,
            "momentum_60d": 0.15,
            "breadth": 0.65,
            "vix": 15,
        }

        regime, adjustments = allocator.detect_and_adjust(market_state)
        adjusted_weights = allocator.apply_adjustments(weights, adjustments)
    """

    def __init__(self, config: AdaptiveRegimeConfig | None = None):
        """初期化

        Args:
            config: レジーム設定（None = デフォルト）
        """
        self.config = config or AdaptiveRegimeConfig.default()

    def detect_regime(self, market_state: dict[str, float]) -> str | None:
        """現在のレジームを検出

        Args:
            market_state: 市場状態
                - momentum_20d: 20日モメンタム
                - momentum_60d: 60日モメンタム
                - breadth: 市場幅（0-1）
                - vix: VIX指数

        Returns:
            検出されたレジーム名（None = 該当なし）
        """
        momentum_20d = market_state.get("momentum_20d", 0.0)
        momentum_60d = market_state.get("momentum_60d", 0.0)
        breadth = market_state.get("breadth", 0.5)
        vix = market_state.get("vix", 20.0)

        # 優先順位でレジームをチェック（より厳しい条件から）
        priority_order = ["crisis", "bear_market", "bull_trend", "recovery", "range_bound"]

        for regime_name in priority_order:
            if regime_name not in self.config.regimes:
                continue

            condition, _ = self.config.regimes[regime_name]

            if self._check_condition(condition, momentum_20d, momentum_60d, breadth, vix):
                logger.info(f"Detected regime: {regime_name}")
                return regime_name

        return None  # デフォルトレジーム（調整なし）

    def _check_condition(
        self,
        condition: RegimeCondition,
        momentum_20d: float,
        momentum_60d: float,
        breadth: float,
        vix: float,
    ) -> bool:
        """条件をチェック

        Args:
            condition: レジーム条件
            momentum_20d, momentum_60d, breadth, vix: 市場状態

        Returns:
            条件を満たすかどうか
        """
        checks = []

        # 各条件をチェック
        if condition.momentum_20d_threshold is not None:
            if condition.momentum_20d_threshold > 0:
                checks.append(momentum_20d > condition.momentum_20d_threshold)
            else:
                checks.append(momentum_20d < condition.momentum_20d_threshold)

        if condition.momentum_60d_threshold is not None:
            if condition.momentum_60d_threshold > 0:
                checks.append(momentum_60d > condition.momentum_60d_threshold)
            else:
                checks.append(momentum_60d < condition.momentum_60d_threshold)

        if condition.breadth_threshold is not None:
            checks.append(breadth > condition.breadth_threshold)

        if condition.vix_threshold is not None:
            # VIXはレジームによって上下どちらかの閾値
            # bull_trend: vix < threshold (低い方が良い)
            # bear_market/crisis: vix > threshold (高い方が悪い)
            # この判定は呼び出し側で適切に設定する必要がある
            # デフォルトでは閾値との比較方法を推測
            if condition.vix_threshold <= 20:  # 低い閾値 = 低VIXを期待
                checks.append(vix < condition.vix_threshold)
            else:  # 高い閾値 = 高VIXを検出
                checks.append(vix > condition.vix_threshold)

        if not checks:
            return False

        if condition.condition_type == "all":
            return all(checks)
        else:  # "any"
            return any(checks)

    def get_adjustments(self, regime_name: str | None) -> RegimeAdjustment:
        """レジームに対応する調整パラメータを取得

        Args:
            regime_name: レジーム名

        Returns:
            RegimeAdjustment
        """
        if regime_name is None or regime_name not in self.config.regimes:
            # デフォルト（調整なし）
            return RegimeAdjustment(
                momentum_weight=1.0,
                defensive_weight=1.0,
                cash_target=0.10,
                description="Default: no regime adjustment",
            )

        _, adjustment = self.config.regimes[regime_name]
        return adjustment

    def apply_adjustments(
        self,
        weights: dict[str, float],
        adjustment: RegimeAdjustment,
        strategy_types: dict[str, str] | None = None,
        cash_symbol: str = "CASH",
    ) -> dict[str, float]:
        """調整をウェイトに適用

        Args:
            weights: 元のウェイト
            adjustment: 適用する調整
            strategy_types: アセット -> 戦略タイプ のマッピング
                ("momentum", "defensive", "neutral")
            cash_symbol: キャッシュのシンボル

        Returns:
            調整後のウェイト
        """
        strategy_types = strategy_types or {}
        adjusted = {}

        # 非キャッシュ資産の調整
        for asset, weight in weights.items():
            if asset == cash_symbol:
                continue

            strategy_type = strategy_types.get(asset, "neutral")

            if strategy_type == "momentum":
                adjusted[asset] = weight * adjustment.momentum_weight
            elif strategy_type == "defensive":
                adjusted[asset] = weight * adjustment.defensive_weight
            else:
                # neutral: 平均的な調整
                avg_modifier = (adjustment.momentum_weight + adjustment.defensive_weight) / 2
                adjusted[asset] = weight * avg_modifier

        # キャッシュ目標への調整
        current_cash = weights.get(cash_symbol, 0.0)
        target_cash = adjustment.cash_target

        if target_cash > current_cash:
            # キャッシュ増加 → 他のアセットを縮小
            cash_increase = target_cash - current_cash
            non_cash_total = sum(adjusted.values())

            if non_cash_total > 0:
                shrink_factor = (1.0 - target_cash) / non_cash_total
                adjusted = {k: v * shrink_factor for k, v in adjusted.items()}

            adjusted[cash_symbol] = target_cash
        else:
            # キャッシュ維持または減少
            adjusted[cash_symbol] = current_cash

        # 正規化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def detect_and_adjust(
        self,
        market_state: dict[str, float],
        weights: dict[str, float],
        strategy_types: dict[str, str] | None = None,
        cash_symbol: str = "CASH",
    ) -> tuple[str | None, dict[str, float]]:
        """レジーム検出と調整を一括実行

        Args:
            market_state: 市場状態
            weights: 元のウェイト
            strategy_types: 戦略タイプマッピング
            cash_symbol: キャッシュシンボル

        Returns:
            (検出されたレジーム, 調整後のウェイト)
        """
        regime = self.detect_regime(market_state)
        adjustment = self.get_adjustments(regime)
        adjusted_weights = self.apply_adjustments(
            weights, adjustment, strategy_types, cash_symbol
        )

        logger.info(
            f"Regime adaptive allocation: regime={regime}, "
            f"cash_target={adjustment.cash_target:.1%}, "
            f"description={adjustment.description}"
        )

        return regime, adjusted_weights


def compute_market_state_from_prices(
    prices: pd.DataFrame,
    vix_column: str | None = "vix",
) -> dict[str, float]:
    """価格データから市場状態を計算

    Args:
        prices: 価格DataFrame（'close'カラム必須、'vix'オプション）
        vix_column: VIXデータのカラム名（None = 推定）

    Returns:
        market_state辞書
    """
    close = prices["close"]

    # モメンタム計算
    momentum_20d = close.pct_change(periods=20).iloc[-1]
    momentum_60d = close.pct_change(periods=60).iloc[-1]

    # 市場幅の推定（単一銘柄の場合は過去のリターンの正の比率で代用）
    returns_20d = close.pct_change().tail(20)
    breadth = (returns_20d > 0).mean()

    # VIX
    if vix_column and vix_column in prices.columns:
        vix = prices[vix_column].iloc[-1]
    else:
        # 実現ボラティリティから推定（年率 * 100でVIX相当に変換）
        realized_vol = close.pct_change().tail(20).std() * np.sqrt(252) * 100
        vix = realized_vol

    return {
        "momentum_20d": float(momentum_20d) if not np.isnan(momentum_20d) else 0.0,
        "momentum_60d": float(momentum_60d) if not np.isnan(momentum_60d) else 0.0,
        "breadth": float(breadth),
        "vix": float(vix) if not np.isnan(vix) else 20.0,
    }


@dataclass
class DynamicWeightingResult:
    """動的重み調整の結果

    Attributes:
        weights: 調整後の重み
        original_weights: 調整前の重み
        adjustments_applied: 適用された調整のリスト
        scale_factor: ボラティリティスケール係数
        dd_protection_level: ドローダウン保護レベル（0-1）
        regime_info: レジーム情報
    """

    weights: dict[str, float]
    original_weights: dict[str, float]
    adjustments_applied: list[str] = field(default_factory=list)
    scale_factor: float = 1.0
    dd_protection_level: float = 0.0
    regime_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights,
            "original_weights": self.original_weights,
            "adjustments_applied": self.adjustments_applied,
            "scale_factor": self.scale_factor,
            "dd_protection_level": self.dd_protection_level,
            "regime_info": self.regime_info,
        }


class DynamicWeighter:
    """動的重み調整クラス

    市場状況に応じて戦略の重みを動的に調整する。

    使用例:
        config = DynamicWeightingConfig(target_volatility=0.15)
        weighter = DynamicWeighter(config)
        result = weighter.adjust_weights(
            weights={"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2},
            market_data={"returns": returns_series, "portfolio_value": 100000, "peak_value": 110000},
            regime_info={"current_vol_regime": "high", "current_trend_regime": "downtrend"}
        )
    """

    # レジーム別の戦略重み調整係数
    REGIME_MODIFIERS: dict[tuple[str, str], dict[str, float]] = {
        # (vol_regime, trend_regime): {strategy_type: modifier}
        # high vol scenarios - 現金重視
        ("high", "uptrend"): {"momentum": 0.6, "mean_reversion": 0.3, "default": 0.5},
        ("high", "range"): {"momentum": 0.3, "mean_reversion": 0.4, "default": 0.4},
        ("high", "downtrend"): {"momentum": 0.2, "mean_reversion": 0.3, "default": 0.3},
        # medium vol scenarios - バランス
        ("medium", "uptrend"): {"momentum": 1.2, "mean_reversion": 0.7, "default": 1.0},
        ("medium", "range"): {"momentum": 0.8, "mean_reversion": 1.1, "default": 0.9},
        ("medium", "downtrend"): {"momentum": 0.5, "mean_reversion": 0.9, "default": 0.7},
        # low vol scenarios - リスク資産重視
        ("low", "uptrend"): {"momentum": 1.5, "mean_reversion": 0.6, "default": 1.2},
        ("low", "range"): {"momentum": 1.0, "mean_reversion": 1.3, "default": 1.1},
        ("low", "downtrend"): {"momentum": 0.7, "mean_reversion": 1.0, "default": 0.8},
    }

    def __init__(self, config: DynamicWeightingConfig | None = None) -> None:
        """初期化

        Args:
            config: 動的重み調整設定（Noneの場合デフォルト）
        """
        self.config = config or DynamicWeightingConfig()
        self._dd_protection_state: float = 0.0  # 現在のDD保護レベル

    def adjust_weights(
        self,
        weights: dict[str, float],
        market_data: dict[str, Any],
        regime_info: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """重みを動的に調整する

        Args:
            weights: 現在の重み（asset -> weight）
            market_data: 市場データ
                - returns: pd.Series（リターン系列）
                - portfolio_value: float（現在のポートフォリオ価値）
                - peak_value: float（ピーク価値、DD計算用）
            regime_info: レジーム情報（RegimeDetectorの出力）
                - current_vol_regime: "low" | "medium" | "high"
                - current_trend_regime: "uptrend" | "range" | "downtrend"

        Returns:
            調整後の重み（dict）
        """
        if not weights:
            logger.warning("Empty weights provided, returning as-is")
            return weights

        original_weights = weights.copy()
        adjusted_weights = weights.copy()
        adjustments: list[str] = []
        scale_factor = 1.0
        dd_level = 0.0

        # 1. ボラティリティスケーリング
        if self.config.vol_scaling_enabled:
            adjusted_weights, scale_factor = self._apply_vol_scaling(
                adjusted_weights, market_data
            )
            if scale_factor != 1.0:
                adjustments.append(f"vol_scaling:{scale_factor:.2f}")

        # 2. ドローダウン保護
        if self.config.dd_protection_enabled:
            adjusted_weights, dd_level = self._apply_dd_protection(
                adjusted_weights, market_data
            )
            if dd_level > 0:
                adjustments.append(f"dd_protection:{dd_level:.2f}")

        # 3. レジームベース重み付け
        if self.config.regime_weighting_enabled and regime_info:
            adjusted_weights = self._apply_regime_weighting(
                adjusted_weights, regime_info
            )
            adjustments.append("regime_weighting")

        # 重みの正規化（合計1.0）
        adjusted_weights = self._normalize_weights(adjusted_weights)

        logger.info(
            f"DynamicWeighter: adjustments={adjustments}, "
            f"scale={scale_factor:.2f}, dd_level={dd_level:.2f}"
        )

        return adjusted_weights

    def _apply_vol_scaling(
        self, weights: dict[str, float], market_data: dict[str, Any]
    ) -> tuple[dict[str, float], float]:
        """ボラティリティスケーリングを適用

        高ボラ時 → 現金比率UP、各銘柄比率DOWN
        低ボラ時 → リスク資産比率UP

        計算式: scale_factor = target_vol / realized_vol

        Args:
            weights: 現在の重み
            market_data: 市場データ（returnsを含む）

        Returns:
            (調整後の重み, スケール係数)
        """
        returns = market_data.get("returns")
        if returns is None or len(returns) < 20:
            return weights, 1.0

        # 直近のリターンでボラティリティを計算（年率換算）
        if isinstance(returns, pd.Series):
            recent_returns = returns.tail(self.config.regime_lookback_days)
        else:
            recent_returns = pd.Series(returns[-self.config.regime_lookback_days :])

        realized_vol = recent_returns.std() * np.sqrt(252)

        if realized_vol <= 0 or np.isnan(realized_vol):
            return weights, 1.0

        # スケール係数を計算
        raw_scale = self.config.target_volatility / realized_vol
        scale_factor = np.clip(raw_scale, self.config.vol_floor, self.config.vol_cap)

        # CASHを除く資産の重みをスケーリング
        cash_symbol = self.config.cash_symbol
        cash_weight = weights.get(cash_symbol, 0.0)
        non_cash_weight = 1.0 - cash_weight

        scaled_weights = {}
        for asset, w in weights.items():
            if asset == cash_symbol:
                # CASH重みは後で調整
                continue
            # 非CASH資産をスケーリング
            scaled_weights[asset] = w * scale_factor

        # スケーリング後の非CASH合計
        scaled_non_cash = sum(scaled_weights.values())

        if scaled_non_cash > 0:
            # 非CASH重みを元の非CASH比率に正規化
            for asset in scaled_weights:
                scaled_weights[asset] = (
                    scaled_weights[asset] / scaled_non_cash * non_cash_weight
                )
        else:
            # 全てCASHの場合
            scaled_weights = {k: v for k, v in weights.items() if k != cash_symbol}

        # スケールダウンした分をCASHに追加
        if scale_factor < 1.0:
            # リスク縮小 → CASH増加
            cash_increase = (1.0 - scale_factor) * non_cash_weight
            new_cash = cash_weight + cash_increase
            # 非CASH資産を縮小
            remaining = 1.0 - new_cash
            if sum(scaled_weights.values()) > 0:
                factor = remaining / sum(scaled_weights.values())
                scaled_weights = {k: v * factor for k, v in scaled_weights.items()}
            scaled_weights[cash_symbol] = new_cash
        else:
            # リスク拡大 → CASH減少（ただし0未満にはしない）
            cash_decrease = min(cash_weight, (scale_factor - 1.0) * non_cash_weight * 0.5)
            new_cash = max(0.0, cash_weight - cash_decrease)
            scaled_weights[cash_symbol] = new_cash

        return scaled_weights, scale_factor

    def _apply_dd_protection(
        self, weights: dict[str, float], market_data: dict[str, Any]
    ) -> tuple[dict[str, float], float]:
        """ドローダウン保護を適用

        直近の損失が閾値を超えたら現金比率UP
        回復したら徐々にリスク資産に戻す

        Args:
            weights: 現在の重み
            market_data: 市場データ（portfolio_value, peak_valueを含む）

        Returns:
            (調整後の重み, 保護レベル)
        """
        portfolio_value = market_data.get("portfolio_value")
        peak_value = market_data.get("peak_value")

        if portfolio_value is None or peak_value is None or peak_value <= 0:
            return weights, 0.0

        # ドローダウンを計算
        drawdown = (peak_value - portfolio_value) / peak_value

        cash_symbol = self.config.cash_symbol
        trigger = self.config.max_drawdown_trigger

        if drawdown >= trigger:
            # DD閾値超過 → 保護モード強化
            excess_dd = drawdown - trigger
            # 保護レベル: DD超過分に比例（最大1.0）
            protection_level = min(1.0, excess_dd / trigger + 0.5)
            self._dd_protection_state = protection_level
        else:
            # 回復中 → 徐々に保護レベルを下げる
            self._dd_protection_state = max(
                0.0, self._dd_protection_state - self.config.dd_recovery_rate
            )
            protection_level = self._dd_protection_state

        if protection_level <= 0:
            return weights, 0.0

        # 保護レベルに応じてCASH比率を増加
        current_cash = weights.get(cash_symbol, 0.0)
        target_cash = current_cash + protection_level * (1.0 - current_cash) * 0.5

        protected_weights = {}
        non_cash_factor = (1.0 - target_cash) / (1.0 - current_cash) if current_cash < 1.0 else 0.0

        for asset, w in weights.items():
            if asset == cash_symbol:
                protected_weights[asset] = target_cash
            else:
                protected_weights[asset] = w * non_cash_factor

        return protected_weights, protection_level

    def _apply_regime_weighting(
        self, weights: dict[str, float], regime_info: dict[str, Any]
    ) -> dict[str, float]:
        """レジームに応じた重み調整

        トレンド相場 → モメンタム戦略重視
        レンジ相場 → 平均回帰戦略重視
        高ボラ相場 → 現金重視

        Args:
            weights: 現在の重み
            regime_info: レジーム情報

        Returns:
            調整後の重み
        """
        vol_regime = regime_info.get("current_vol_regime", "medium")
        trend_regime = regime_info.get("current_trend_regime", "range")

        # レジーム修正係数を取得
        regime_key = (vol_regime, trend_regime)
        modifiers = self.REGIME_MODIFIERS.get(
            regime_key, {"momentum": 1.0, "mean_reversion": 1.0, "default": 1.0}
        )

        cash_symbol = self.config.cash_symbol
        adjusted_weights = {}

        for asset, w in weights.items():
            if asset == cash_symbol:
                adjusted_weights[asset] = w
                continue

            # 戦略タイプを推定（シンプルな実装）
            # 実際にはassetのメタデータから取得すべき
            strategy_type = self._infer_strategy_type(asset)
            modifier = modifiers.get(strategy_type, modifiers.get("default", 1.0))

            adjusted_weights[asset] = w * modifier

        return adjusted_weights

    def _infer_strategy_type(self, asset: str) -> str:
        """アセット名から戦略タイプを推定

        NOTE: 実際の実装では、アセットのメタデータや
        シグナル情報から判断すべき。
        ここではシンプルなヒューリスティックを使用。

        Args:
            asset: アセット名

        Returns:
            戦略タイプ ("momentum", "mean_reversion", "default")
        """
        asset_lower = asset.lower()

        # モメンタム系のキーワード
        momentum_keywords = ["momentum", "trend", "breakout", "growth"]
        for kw in momentum_keywords:
            if kw in asset_lower:
                return "momentum"

        # 平均回帰系のキーワード
        reversion_keywords = ["value", "reversion", "mean", "rsi", "contrarian"]
        for kw in reversion_keywords:
            if kw in asset_lower:
                return "mean_reversion"

        return "default"

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """重みを正規化して合計1.0にする

        Args:
            weights: 重み

        Returns:
            正規化された重み
        """
        total = sum(weights.values())
        if total <= 0:
            # 全てゼロの場合はCASH 100%
            return {self.config.cash_symbol: 1.0}

        return {k: v / total for k, v in weights.items()}

    def get_adjustment_summary(
        self, market_data: dict[str, Any], regime_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """現在の市場状況に対する調整サマリを取得

        デバッグ・監視用。

        Args:
            market_data: 市場データ
            regime_info: レジーム情報

        Returns:
            調整サマリ
        """
        returns = market_data.get("returns")
        realized_vol = None
        if returns is not None and len(returns) >= 20:
            if isinstance(returns, pd.Series):
                recent = returns.tail(self.config.regime_lookback_days)
            else:
                recent = pd.Series(returns[-self.config.regime_lookback_days :])
            realized_vol = recent.std() * np.sqrt(252)

        portfolio_value = market_data.get("portfolio_value")
        peak_value = market_data.get("peak_value")
        drawdown = None
        if portfolio_value is not None and peak_value is not None and peak_value > 0:
            drawdown = (peak_value - portfolio_value) / peak_value

        return {
            "config": {
                "target_volatility": self.config.target_volatility,
                "max_drawdown_trigger": self.config.max_drawdown_trigger,
                "vol_scaling_enabled": self.config.vol_scaling_enabled,
                "dd_protection_enabled": self.config.dd_protection_enabled,
                "regime_weighting_enabled": self.config.regime_weighting_enabled,
            },
            "market_state": {
                "realized_volatility": realized_vol,
                "current_drawdown": drawdown,
                "vol_regime": regime_info.get("current_vol_regime") if regime_info else None,
                "trend_regime": regime_info.get("current_trend_regime") if regime_info else None,
            },
            "protection_state": {
                "dd_protection_level": self._dd_protection_state,
            },
        }
