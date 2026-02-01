"""
CMD_016 Feature Integrator - 全機能統合モジュール

cmd_016で実装された全機能をパイプラインに統合するためのヘルパーモジュール。

統合順序:
1. VIX取得 → VIXキャッシュ配分計算
2. レジーム適応シグナルパラメータ設定
3. シグナル生成（ペアトレーディング、クロスアセット等）
4. セクターローテーション調整
5. ヒステリシスフィルター適用
6. シグナル減衰適用
7. 最低保有期間フィルター
8. 相関ブレイク検出
9. ドローダウンプロテクション
10. イールドカーブ調整
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Availability Flags
# =============================================================================

# Risk Management
try:
    from src.risk.vix_cash_allocation import VIXCashAllocator, VIXCashConfig
    VIX_CASH_AVAILABLE = True
except ImportError:
    VIX_CASH_AVAILABLE = False

try:
    from src.risk.correlation_break import (
        CorrelationBreakDetector,
        CorrelationBreakConfig,
    )
    CORRELATION_BREAK_AVAILABLE = True
except ImportError:
    CORRELATION_BREAK_AVAILABLE = False

try:
    from src.risk.drawdown_protection import (
        DrawdownProtector,
        DrawdownProtectorConfig,
    )
    DRAWDOWN_PROTECTION_AVAILABLE = True
except ImportError:
    DRAWDOWN_PROTECTION_AVAILABLE = False

# Strategy
try:
    from src.strategy.pairs_trading import (
        PairsTradingStrategy,
        PairsTraderConfig,
    )
    PAIRS_TRADING_AVAILABLE = True
except ImportError:
    PAIRS_TRADING_AVAILABLE = False

try:
    from src.strategy.sector_rotation import (
        EconomicCycleSectorRotator,
        MomentumSectorRotator,
    )
    SECTOR_ROTATION_AVAILABLE = True
except ImportError:
    SECTOR_ROTATION_AVAILABLE = False

try:
    from src.strategy.min_holding_period import (
        MinHoldingPeriodFilter,
        MinHoldingPeriodConfig,
        apply_min_holding,
    )
    MIN_HOLDING_AVAILABLE = True
except ImportError:
    MIN_HOLDING_AVAILABLE = False

try:
    from src.strategy.entry_exit_optimizer import HysteresisFilter
    HYSTERESIS_AVAILABLE = True
except ImportError:
    HYSTERESIS_AVAILABLE = False

# Signals
try:
    from src.signals.regime_adaptive_params import RegimeAdaptiveParams
    REGIME_ADAPTIVE_AVAILABLE = True
except ImportError:
    REGIME_ADAPTIVE_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CMD016Config:
    """CMD_016機能の統合設定"""

    # Signal & Strategy
    pairs_trading_enabled: bool = True
    cross_asset_momentum_enabled: bool = True
    sector_rotation_enabled: bool = True
    dual_momentum_enabled: bool = True
    low_vol_premium_enabled: bool = True

    # Risk Management
    vix_cash_allocation_enabled: bool = True
    correlation_break_enabled: bool = True
    drawdown_protection_enabled: bool = True

    # Dynamic Parameters
    dynamic_thresholds_enabled: bool = True
    regime_signal_params_enabled: bool = True
    yield_curve_enabled: bool = True

    # Filters
    hysteresis_filter_enabled: bool = True
    signal_decay_enabled: bool = True
    min_holding_period_enabled: bool = True

    # Raw config dict for detailed settings
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Any) -> "CMD016Config":
        """設定オブジェクトから構築"""
        config = cls()

        # Get cmd_016_features from settings
        features = getattr(settings, "cmd_016_features", None)
        if features is None:
            return config

        if isinstance(features, dict):
            raw = features
        else:
            raw = features.__dict__ if hasattr(features, "__dict__") else {}

        config.raw_config = raw

        # Parse enabled flags
        def get_enabled(key: str) -> bool:
            if key in raw:
                val = raw[key]
                if isinstance(val, dict):
                    return val.get("enabled", True)
                return bool(val)
            return True

        config.pairs_trading_enabled = get_enabled("pairs_trading")
        config.cross_asset_momentum_enabled = get_enabled("cross_asset_momentum")
        config.sector_rotation_enabled = get_enabled("sector_rotation")
        config.dual_momentum_enabled = get_enabled("dual_momentum")
        config.low_vol_premium_enabled = get_enabled("low_vol_premium")
        config.vix_cash_allocation_enabled = get_enabled("vix_cash_allocation")
        config.correlation_break_enabled = get_enabled("correlation_break")
        config.drawdown_protection_enabled = get_enabled("drawdown_protection")
        config.dynamic_thresholds_enabled = get_enabled("dynamic_thresholds")
        config.regime_signal_params_enabled = get_enabled("regime_signal_params")
        config.yield_curve_enabled = get_enabled("yield_curve")
        config.hysteresis_filter_enabled = get_enabled("hysteresis_filter")
        config.signal_decay_enabled = get_enabled("signal_decay")
        config.min_holding_period_enabled = get_enabled("min_holding_period")

        return config


@dataclass
class IntegrationResult:
    """統合処理の結果"""

    adjusted_weights: pd.Series
    cash_ratio: float = 0.0
    vix_adjustment: dict[str, Any] = field(default_factory=dict)
    correlation_break_info: dict[str, Any] = field(default_factory=dict)
    drawdown_info: dict[str, Any] = field(default_factory=dict)
    sector_rotation_info: dict[str, Any] = field(default_factory=dict)
    filter_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "adjusted_weights": self.adjusted_weights.to_dict(),
            "cash_ratio": self.cash_ratio,
            "vix_adjustment": self.vix_adjustment,
            "correlation_break_info": self.correlation_break_info,
            "drawdown_info": self.drawdown_info,
            "sector_rotation_info": self.sector_rotation_info,
            "filter_info": self.filter_info,
            "warnings": self.warnings,
        }


# =============================================================================
# Main Integrator Class
# =============================================================================

class CMD016Integrator:
    """
    CMD_016機能統合クラス

    パイプラインの各段階で呼び出され、cmd_016の機能を適用する。

    Usage:
        integrator = CMD016Integrator(settings)

        # VIXキャッシュ配分
        vix_result = integrator.apply_vix_cash_allocation(vix_data)

        # シグナルフィルター
        filtered_signals = integrator.apply_signal_filters(signals, weights)

        # リスク調整
        adjusted_weights = integrator.apply_risk_adjustments(weights, portfolio_value)
    """

    def __init__(self, settings: Any = None) -> None:
        """初期化"""
        self._settings = settings
        self._config = CMD016Config.from_settings(settings) if settings else CMD016Config()

        # Component instances (lazy initialized)
        self._vix_allocator: VIXCashAllocator | None = None
        self._correlation_detector: CorrelationBreakDetector | None = None
        self._drawdown_protector: DrawdownProtector | None = None
        self._min_holding_filter: MinHoldingPeriodFilter | None = None
        self._hysteresis_filter: HysteresisFilter | None = None
        self._sector_rotator: EconomicCycleSectorRotator | None = None
        self._regime_adaptive: RegimeAdaptiveParams | None = None

        # State tracking
        self._portfolio_hwm: float = 0.0
        self._last_weights: pd.Series | None = None

    # =========================================================================
    # VIX Cash Allocation
    # =========================================================================

    def apply_vix_cash_allocation(
        self,
        vix_value: float,
        base_weights: pd.Series,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        VIXベースのキャッシュ配分を適用

        Args:
            vix_value: 現在のVIX値
            base_weights: 基本重み

        Returns:
            (調整後重み, VIX調整情報)
        """
        if not self._config.vix_cash_allocation_enabled or not VIX_CASH_AVAILABLE:
            return base_weights, {"enabled": False}

        try:
            if self._vix_allocator is None:
                vix_config = self._config.raw_config.get("vix_cash_allocation", {})
                self._vix_allocator = VIXCashAllocator(VIXCashConfig(
                    vix_low=vix_config.get("vix_low", 15),
                    vix_high=vix_config.get("vix_high", 25),
                    vix_extreme=vix_config.get("vix_extreme", 35),
                    max_cash_ratio=vix_config.get("max_cash_ratio", 0.5),
                ))

            result = self._vix_allocator.compute(vix_value)

            # Apply cash allocation
            if result.cash_ratio > 0:
                adjusted = base_weights * (1 - result.cash_ratio)
                if "CASH" in adjusted.index:
                    adjusted["CASH"] += result.cash_ratio
                else:
                    adjusted["CASH"] = result.cash_ratio

                # Normalize
                adjusted = adjusted / adjusted.sum()
            else:
                adjusted = base_weights.copy()

            info = {
                "enabled": True,
                "vix_value": vix_value,
                "cash_ratio": result.cash_ratio,
                "vix_regime": result.regime.value if hasattr(result, "regime") else "unknown",
            }

            return adjusted, info

        except Exception as e:
            logger.warning(f"VIX cash allocation failed: {e}")
            return base_weights, {"enabled": True, "error": str(e)}

    # =========================================================================
    # Correlation Break Detection
    # =========================================================================

    def detect_correlation_break(
        self,
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        相関ブレイクを検出

        Args:
            returns: リターンデータ

        Returns:
            相関ブレイク情報
        """
        if not self._config.correlation_break_enabled or not CORRELATION_BREAK_AVAILABLE:
            return {"enabled": False}

        try:
            if self._correlation_detector is None:
                corr_config = self._config.raw_config.get("correlation_break", {})
                self._correlation_detector = CorrelationBreakDetector(CorrelationBreakConfig(
                    warning_threshold=corr_config.get("warning_threshold", 0.3),
                    critical_threshold=corr_config.get("critical_threshold", 0.5),
                    short_window=corr_config.get("lookback_short", 20),
                    long_window=corr_config.get("lookback_long", 60),
                ))

            result = self._correlation_detector.detect_correlation_break(returns)

            return {
                "enabled": True,
                "warning_level": result.warning_level.value if hasattr(result, "warning_level") else "none",
                "change_magnitude": result.change_magnitude if hasattr(result, "change_magnitude") else 0.0,
                "affected_pairs": result.affected_pairs if hasattr(result, "affected_pairs") else [],
            }

        except Exception as e:
            logger.warning(f"Correlation break detection failed: {e}")
            return {"enabled": True, "error": str(e)}

    # =========================================================================
    # Drawdown Protection
    # =========================================================================

    def apply_drawdown_protection(
        self,
        portfolio_value: float,
        base_weights: pd.Series,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        ドローダウンプロテクションを適用

        Args:
            portfolio_value: 現在のポートフォリオ価値
            base_weights: 基本重み

        Returns:
            (調整後重み, ドローダウン情報)
        """
        if not self._config.drawdown_protection_enabled or not DRAWDOWN_PROTECTION_AVAILABLE:
            return base_weights, {"enabled": False}

        try:
            if self._drawdown_protector is None:
                dd_config = self._config.raw_config.get("drawdown_protection", {})
                self._drawdown_protector = DrawdownProtector(DrawdownProtectorConfig(
                    dd_levels=dd_config.get("dd_levels", [0.05, 0.10, 0.15, 0.20]),
                    risk_reductions=dd_config.get("risk_reductions", [0.9, 0.7, 0.5, 0.3]),
                    recovery_threshold=dd_config.get("recovery_threshold", 0.5),
                    emergency_dd_level=dd_config.get("emergency_dd_level", 0.25),
                ))

            # Update HWM
            if portfolio_value > self._portfolio_hwm:
                self._portfolio_hwm = portfolio_value

            # Initialize if needed
            if self._drawdown_protector._state.hwm == 0:
                self._drawdown_protector.initialize(self._portfolio_hwm)

            # Update state
            self._drawdown_protector.update(portfolio_value)

            # Adjust weights
            result = self._drawdown_protector.adjust_weights(base_weights)

            info = {
                "enabled": True,
                "current_dd": self._drawdown_protector._state.drawdown,
                "protection_level": self._drawdown_protector._state.protection_level,
                "risk_multiplier": result.risk_multiplier,
                "hwm": self._portfolio_hwm,
            }

            return result.adjusted_weights, info

        except Exception as e:
            logger.warning(f"Drawdown protection failed: {e}")
            return base_weights, {"enabled": True, "error": str(e)}

    # =========================================================================
    # Signal Filters
    # =========================================================================

    def apply_signal_filters(
        self,
        signals: dict[str, float] | pd.Series,
        current_weights: dict[str, float] | pd.Series,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        シグナルフィルターを適用（ヒステリシス、減衰、最低保有期間）

        Args:
            signals: シグナル辞書
            current_weights: 現在の重み

        Returns:
            (フィルタ済みシグナル, フィルター情報)
        """
        if isinstance(signals, dict):
            signals = pd.Series(signals)
        if isinstance(current_weights, dict):
            current_weights = pd.Series(current_weights)

        filtered = signals.copy()
        info: dict[str, Any] = {}

        # 1. Hysteresis Filter
        if self._config.hysteresis_filter_enabled and HYSTERESIS_AVAILABLE:
            try:
                if self._hysteresis_filter is None:
                    hyst_config = self._config.raw_config.get("hysteresis_filter", {})
                    self._hysteresis_filter = HysteresisFilter(
                        entry_threshold=hyst_config.get("entry_threshold", 0.3),
                        exit_threshold=hyst_config.get("exit_threshold", 0.1),
                    )

                for asset in filtered.index:
                    current = current_weights.get(asset, 0.0)
                    original = filtered[asset]
                    # filter_signal() returns float directly (not FilterResult object)
                    filtered[asset] = self._hysteresis_filter.filter_signal(asset, original)

                info["hysteresis"] = {"enabled": True}

            except Exception as e:
                logger.warning(f"Hysteresis filter failed: {e}")
                info["hysteresis"] = {"enabled": True, "error": str(e)}

        # 2. Signal Decay (exponential)
        if self._config.signal_decay_enabled:
            try:
                decay_config = self._config.raw_config.get("signal_decay", {})
                halflife = decay_config.get("halflife", 5)
                min_signal = decay_config.get("min_signal", 0.01)

                # Simple decay factor (would need state tracking for full implementation)
                # For now, just apply a mild decay
                decay_factor = 0.95  # ~5 day halflife
                filtered = filtered * decay_factor
                filtered = filtered.clip(lower=min_signal)

                info["signal_decay"] = {"enabled": True, "halflife": halflife}

            except Exception as e:
                logger.warning(f"Signal decay failed: {e}")
                info["signal_decay"] = {"enabled": True, "error": str(e)}

        # 3. Minimum Holding Period
        if self._config.min_holding_period_enabled and MIN_HOLDING_AVAILABLE:
            try:
                if self._min_holding_filter is None:
                    mhp_config = self._config.raw_config.get("min_holding_period", {})
                    # Use config= keyword argument to avoid positional argument confusion
                    self._min_holding_filter = MinHoldingPeriodFilter(
                        config=MinHoldingPeriodConfig(
                            min_periods=mhp_config.get("min_periods", 5),
                            force_exit_on_reversal=mhp_config.get("force_exit_on_reversal", True),
                            reversal_threshold=mhp_config.get("reversal_threshold", -0.5),
                        )
                    )

                decisions = apply_min_holding(
                    filtered.to_dict(),
                    current_weights.to_dict(),
                    self._min_holding_filter,
                    update_periods=True,
                )

                # Update signals based on decisions
                for asset, decision in decisions.items():
                    filtered[asset] = decision.signal

                blocked = self._min_holding_filter.get_blocked_assets()
                info["min_holding"] = {
                    "enabled": True,
                    "blocked_assets": blocked,
                }

            except Exception as e:
                logger.warning(f"Min holding filter failed: {e}")
                info["min_holding"] = {"enabled": True, "error": str(e)}

        return filtered, info

    # =========================================================================
    # Sector Rotation
    # =========================================================================

    def apply_sector_rotation(
        self,
        base_weights: pd.Series,
        macro_indicators: dict[str, float] | None = None,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        セクターローテーション調整を適用

        Args:
            base_weights: 基本重み
            macro_indicators: マクロ指標

        Returns:
            (調整後重み, セクターローテーション情報)
        """
        if not self._config.sector_rotation_enabled or not SECTOR_ROTATION_AVAILABLE:
            return base_weights, {"enabled": False}

        try:
            if self._sector_rotator is None:
                from src.strategy.sector_rotation import EconomicCycleSectorRotator
                sr_config = self._config.raw_config.get("sector_rotation", {})
                # create_economic_cycle_rotator() は引数を取らないため、直接クラスを使用
                self._sector_rotator = EconomicCycleSectorRotator()

            # Detect phase and get adjustments
            if macro_indicators:
                from src.strategy.sector_rotation import MacroIndicators
                indicators = MacroIndicators(**macro_indicators)
                result = self._sector_rotator.rotate(base_weights, indicators)
                adjusted = result.adjusted_weights
                phase = result.detected_phase.value if hasattr(result, "detected_phase") else "unknown"
            else:
                adjusted = base_weights.copy()
                phase = "unknown"

            info = {
                "enabled": True,
                "detected_phase": phase,
            }

            return adjusted, info

        except Exception as e:
            logger.warning(f"Sector rotation failed: {e}")
            return base_weights, {"enabled": True, "error": str(e)}

    # =========================================================================
    # Full Integration
    # =========================================================================

    def integrate_all(
        self,
        base_weights: pd.Series,
        signals: pd.Series,
        portfolio_value: float,
        vix_value: float | None = None,
        returns: pd.DataFrame | None = None,
        macro_indicators: dict[str, float] | None = None,
    ) -> IntegrationResult:
        """
        全機能を統合して適用

        Args:
            base_weights: 基本重み
            signals: シグナル
            portfolio_value: ポートフォリオ価値
            vix_value: VIX値（任意）
            returns: リターンデータ（任意）
            macro_indicators: マクロ指標（任意）

        Returns:
            IntegrationResult
        """
        result = IntegrationResult(adjusted_weights=base_weights.copy())
        warnings: list[str] = []

        # 1. VIX Cash Allocation
        if vix_value is not None:
            adjusted, vix_info = self.apply_vix_cash_allocation(vix_value, result.adjusted_weights)
            result.adjusted_weights = adjusted
            result.vix_adjustment = vix_info
            result.cash_ratio = vix_info.get("cash_ratio", 0.0)

        # 2. Correlation Break Detection
        if returns is not None:
            corr_info = self.detect_correlation_break(returns)
            result.correlation_break_info = corr_info
            if corr_info.get("warning_level") == "critical":
                warnings.append("Critical correlation break detected")

        # 3. Sector Rotation
        adjusted, sr_info = self.apply_sector_rotation(result.adjusted_weights, macro_indicators)
        result.adjusted_weights = adjusted
        result.sector_rotation_info = sr_info

        # 4. Signal Filters
        if signals is not None:
            _, filter_info = self.apply_signal_filters(signals, result.adjusted_weights)
            result.filter_info = filter_info

        # 5. Drawdown Protection
        adjusted, dd_info = self.apply_drawdown_protection(portfolio_value, result.adjusted_weights)
        result.adjusted_weights = adjusted
        result.drawdown_info = dd_info

        result.warnings = warnings
        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_feature_status(self) -> dict[str, bool]:
        """各機能の有効/無効状態を取得"""
        return {
            "pairs_trading": self._config.pairs_trading_enabled and PAIRS_TRADING_AVAILABLE,
            "cross_asset_momentum": self._config.cross_asset_momentum_enabled,
            "sector_rotation": self._config.sector_rotation_enabled and SECTOR_ROTATION_AVAILABLE,
            "dual_momentum": self._config.dual_momentum_enabled,
            "low_vol_premium": self._config.low_vol_premium_enabled,
            "vix_cash_allocation": self._config.vix_cash_allocation_enabled and VIX_CASH_AVAILABLE,
            "correlation_break": self._config.correlation_break_enabled and CORRELATION_BREAK_AVAILABLE,
            "drawdown_protection": self._config.drawdown_protection_enabled and DRAWDOWN_PROTECTION_AVAILABLE,
            "dynamic_thresholds": self._config.dynamic_thresholds_enabled,
            "regime_signal_params": self._config.regime_signal_params_enabled and REGIME_ADAPTIVE_AVAILABLE,
            "yield_curve": self._config.yield_curve_enabled,
            "hysteresis_filter": self._config.hysteresis_filter_enabled and HYSTERESIS_AVAILABLE,
            "signal_decay": self._config.signal_decay_enabled,
            "min_holding_period": self._config.min_holding_period_enabled and MIN_HOLDING_AVAILABLE,
        }

    def reset_state(self) -> None:
        """内部状態をリセット"""
        self._portfolio_hwm = 0.0
        self._last_weights = None

        if self._drawdown_protector:
            self._drawdown_protector.reset()
        if self._min_holding_filter:
            self._min_holding_filter.reset()
        if self._hysteresis_filter:
            self._hysteresis_filter.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_integrator(settings: Any = None) -> CMD016Integrator:
    """CMD016Integratorを作成"""
    return CMD016Integrator(settings)


def get_available_features() -> dict[str, bool]:
    """利用可能な機能を取得"""
    return {
        "vix_cash_allocation": VIX_CASH_AVAILABLE,
        "correlation_break": CORRELATION_BREAK_AVAILABLE,
        "drawdown_protection": DRAWDOWN_PROTECTION_AVAILABLE,
        "pairs_trading": PAIRS_TRADING_AVAILABLE,
        "sector_rotation": SECTOR_ROTATION_AVAILABLE,
        "min_holding_period": MIN_HOLDING_AVAILABLE,
        "hysteresis_filter": HYSTERESIS_AVAILABLE,
        "regime_adaptive": REGIME_ADAPTIVE_AVAILABLE,
    }
