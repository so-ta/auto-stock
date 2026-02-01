"""
Signal Generator Module.

This module provides various signal generators for trading strategies.
All signals output normalized values in the range [-1, +1].

Available Signal Categories:
- momentum: Trend-following signals (MomentumReturnSignal, ROCSignal, etc.)
- mean_reversion: Counter-trend signals (BollingerReversionSignal, RSISignal, etc.)
- volatility: Volatility-based signals (ATRSignal, VolatilityBreakoutSignal, etc.)
- breakout: Breakout signals (DonchianChannelSignal, HighLowBreakoutSignal, etc.)
- seasonality: Calendar-based signals (DayOfWeekSignal, MonthEffectSignal, etc.)
- factor: Cross-sectional factor signals (ValueFactor, QualityFactor, LowVolFactor, etc.)
- sector: Sector-based signals (SectorMomentum, SectorRelativeStrength, SectorBreadth)
- volume: Order flow signals (OBVMomentum, MFI, VWAP, AccumulationDistribution)
- correlation: Regime detection signals (CorrelationRegime, ReturnDispersion)
- advanced_technical: Adaptive indicators (KAMA, KeltnerChannel)

Usage:
    from signals import SignalRegistry, Signal

    # List all available signals
    all_signals = SignalRegistry.list_all()

    # Create a signal by name
    signal = SignalRegistry.create("momentum_return", lookback=20)

    # Compute signal scores
    result = signal.compute(price_data)
    scores = result.scores  # pd.Series with values in [-1, +1]
"""

from src.signals.base import ParameterSpec, Signal, SignalResult
from src.signals.registry import SignalRegistry, SignalRegistryError

# Momentum signals
from src.signals.momentum import (
    MomentumReturnSignal,
    ROCSignal,
    MomentumCompositeSignal,
    MomentumAccelerationSignal,
)

# Mean reversion signals
from src.signals.mean_reversion import (
    BollingerReversionSignal,
    RSISignal,
    ZScoreReversionSignal,
    StochasticReversionSignal,
)

# Volatility signals
from src.signals.volatility import (
    ATRSignal,
    VolatilityBreakoutSignal,
    VolatilityRegimeSignal,
)

# Breakout signals
from src.signals.breakout import (
    DonchianChannelSignal,
    HighLowBreakoutSignal,
    RangeBreakoutSignal,
)

# Seasonality signals
from src.signals.seasonality import (
    DayOfWeekSignal,
    MonthEffectSignal,
    TurnOfMonthSignal,
)

# Factor signals
from src.signals.factor import (
    ValueFactorSignal,
    QualityFactorSignal,
    LowVolFactorSignal,
    MomentumFactorSignal,
    SizeFactorSignal,
)

# Sector signals
from src.signals.sector import (
    SectorMomentumSignal,
    SectorRelativeStrengthSignal,
    SectorBreadthSignal,
    SECTOR_ETFS,
    get_sector_etf,
    get_all_sector_etfs,
)

# Ensemble signals
from src.signals.ensemble import (
    MomentumEnsemble,
    MeanReversionEnsemble,
    TrendStrength,
    RegimeDetector,
)

# Trend signals (advanced)
from src.signals.trend import (
    DualMomentumSignal,
    TrendFollowingSignal,
    AdaptiveTrendSignal,
    CrossAssetMomentumSignal,
    MultiTimeframeMomentumSignal,
    TimeframeConsensusSignal,
)

# Sentiment signals
from src.signals.sentiment import (
    VIXSentimentSignal,
    PutCallRatioSignal,
    MarketBreadthSignal,
    FearGreedCompositeSignal,
    VIXTermStructureSignal,
)

# Macro economic signals
from src.signals.macro import (
    YieldCurveSignal,
    InflationExpectationSignal,
    CreditSpreadSignal,
    DollarStrengthSignal,
    MacroRegimeCompositeSignal,
)

# Dynamic parameters
from src.signals.dynamic_params import (
    SignalDynamicParamsCalculator,
    MomentumDynamicParams,
    BollingerDynamicParams,
    RSIDynamicParams,
    ZScoreDynamicParams,
    SignalParamsBundle,
    VolatilityRegime,
    calculate_signal_params,
    get_momentum_params,
    get_bollinger_params,
    get_rsi_params,
    get_zscore_params,
    detect_volatility_regime,
)

# Regime adaptive parameters
from src.signals.regime_adaptive_params import (
    RegimeAdaptiveParams,
    SignalParamSet,
    RegimeParamAdjustment,
    MarketRegime,
    REGIME_SIGNAL_PARAMS,
    get_regime_params,
    adjust_signal_params,
    get_param_set,
    list_available_regimes,
    list_signal_types,
    create_regime_adaptive_params,
)

# Cross asset momentum
from src.signals.cross_asset import (
    CrossAssetMomentumRanker,
    AssetClass,
    AssetClassType,
    AssetClassScore,
    RankingResult,
    AllocationAdjustment,
    DEFAULT_ASSET_CLASSES,
    create_cross_asset_ranker,
    quick_rank_asset_classes,
)

# Regime signal parameters (flat structure)
from src.signals.regime_signal_params import (
    RegimeSignalParamSelector,
    SignalRegime,
    RegimeParams,
    AppliedParams,
    REGIME_SIGNAL_PARAMS as REGIME_SIGNAL_PARAMS_FLAT,
    get_regime_signal_params,
    interpolate_regime_params,
    apply_regime_params,
    apply_regime_params_full,
    list_signal_regimes,
    create_regime_signal_param_selector,
)

# VIX signal (IMP-003)
from src.signals.vix_signal import (
    EnhancedVIXSignal,
    VIXSignalConfig,
    VIXSignalResult,
    VIXTier,
    EmergencyTrigger,
    get_vix_cash_allocation,
    calculate_vix_adjusted_weights,
)

# Enhanced Mean Reversion (IMP-006)
from src.signals.mean_reversion_enhanced import (
    EnhancedMeanReversion,
    MeanReversionConfig,
    MomentumMeanReversionBlender,
    calculate_mean_reversion_score,
)

# Enhanced Regime Detector V2 (IMP-007)
from src.signals.regime_detector_v2 import (
    EnhancedRegimeDetector,
    RegimeDetectorConfig,
    RegimeResult,
    RegimeType,
    RegimeAdaptiveStrategy,
    detect_market_regime,
    get_regime_adjusted_weights,
)

# Enhanced Sector Rotation V2 (IMP-008)
from src.signals.sector_rotation_v2 import (
    EnhancedSectorRotation,
    EconomicCycleDetector,
    EconomicCycle,
    SectorMomentumCalculator,
    RelativeStrengthCalculator,
    RotationResult,
    SectorScore,
    ECONOMIC_CYCLE_SECTORS,
    SECTOR_CYCLICALITY,
    create_sector_rotation_strategy,
    get_current_cycle_sectors,
)

# Short Interest Signal (task_042_5)
from src.signals.short_interest import (
    ShortInterestSignal,
    ShortInterestChangeSignal,
    compute_short_interest_signal,
)

# Insider Trading Signal (task_042_4)
from src.signals.insider_trading import (
    InsiderTradingSignal,
    InsiderSignalConfig,
    compute_insider_signal,
)

# Short-Term Reversal Signal (task_042_3)
from src.signals.short_term_reversal import (
    ShortTermReversalSignal,
    WeeklyShortTermReversalSignal,
    MonthlyShortTermReversalSignal,
)

# Volume signals (Order Flow)
from src.signals.volume import (
    OBVMomentumSignal,
    MoneyFlowIndexSignal,
    VWAPDeviationSignal,
    AccumulationDistributionSignal,
)

# Correlation signals (Regime Detection)
from src.signals.correlation import (
    CorrelationRegimeSignal,
    ReturnDispersionSignal,
    CrossAssetCorrelationSignal,
)

# Advanced Technical signals (Adaptive Indicators)
from src.signals.advanced_technical import (
    KAMASignal,
    KeltnerChannelSignal,
)

# Fifty Two Week High (George & Hwang 2004)
from src.signals.fifty_two_week_high import (
    FiftyTwoWeekHighMomentumSignal,
)

# Lead-Lag Signal (Oxford研究ベース)
from src.signals.lead_lag import (
    LeadLagSignal,
    LeadLagPair,
    compute_lead_lag_signal,
)

# Low Volatility Premium (ファクター投資)
from src.signals.low_vol_premium import (
    LowVolPremiumSignal,
    LowVolPremiumStrategy,
    LowVolPremiumConfig,
    calculate_volatility_rank,
    apply_low_vol_premium,
)

# Enhanced Yield Curve Signal (マクロ)
from src.signals.yield_curve_signal import (
    EnhancedYieldCurveRegisteredSignal,
    EnhancedYieldCurveSignal,
    CurveShape,
    get_current_yield_curve_shape,
    get_yield_curve_allocation_adjustment,
)

# Meta Validation (Optimization Level Selection)
from src.signals.meta_validation import (
    MetaValidationCache,
    MetaValidator,
    MetaValidationResult,
    LevelValidationResult,
    AdaptiveParameterCalculator,
    OptimizationLevel,
    create_adaptive_calculator,
)

__all__ = [
    # Base
    "Signal",
    "SignalResult",
    "ParameterSpec",
    "SignalRegistry",
    "SignalRegistryError",
    # Momentum
    "MomentumReturnSignal",
    "ROCSignal",
    "MomentumCompositeSignal",
    "MomentumAccelerationSignal",
    # Mean Reversion
    "BollingerReversionSignal",
    "RSISignal",
    "ZScoreReversionSignal",
    "StochasticReversionSignal",
    # Volatility
    "ATRSignal",
    "VolatilityBreakoutSignal",
    "VolatilityRegimeSignal",
    # Breakout
    "DonchianChannelSignal",
    "HighLowBreakoutSignal",
    "RangeBreakoutSignal",
    # Seasonality
    "DayOfWeekSignal",
    "MonthEffectSignal",
    "TurnOfMonthSignal",
    # Factor
    "ValueFactorSignal",
    "QualityFactorSignal",
    "LowVolFactorSignal",
    "MomentumFactorSignal",
    "SizeFactorSignal",
    # Sector
    "SectorMomentumSignal",
    "SectorRelativeStrengthSignal",
    "SectorBreadthSignal",
    "SECTOR_ETFS",
    "get_sector_etf",
    "get_all_sector_etfs",
    # Ensemble
    "MomentumEnsemble",
    "MeanReversionEnsemble",
    "TrendStrength",
    "RegimeDetector",
    # Trend (advanced)
    "DualMomentumSignal",
    "TrendFollowingSignal",
    "AdaptiveTrendSignal",
    "CrossAssetMomentumSignal",
    "MultiTimeframeMomentumSignal",
    "TimeframeConsensusSignal",
    # Sentiment
    "VIXSentimentSignal",
    "PutCallRatioSignal",
    "MarketBreadthSignal",
    "FearGreedCompositeSignal",
    "VIXTermStructureSignal",
    # Macro
    "YieldCurveSignal",
    "InflationExpectationSignal",
    "CreditSpreadSignal",
    "DollarStrengthSignal",
    "MacroRegimeCompositeSignal",
    # Dynamic parameters
    "SignalDynamicParamsCalculator",
    "MomentumDynamicParams",
    "BollingerDynamicParams",
    "RSIDynamicParams",
    "ZScoreDynamicParams",
    "SignalParamsBundle",
    "VolatilityRegime",
    "calculate_signal_params",
    "get_momentum_params",
    "get_bollinger_params",
    "get_rsi_params",
    "get_zscore_params",
    "detect_volatility_regime",
    # Regime adaptive parameters
    "RegimeAdaptiveParams",
    "SignalParamSet",
    "RegimeParamAdjustment",
    "MarketRegime",
    "REGIME_SIGNAL_PARAMS",
    "get_regime_params",
    "adjust_signal_params",
    "get_param_set",
    "list_available_regimes",
    "list_signal_types",
    "create_regime_adaptive_params",
    # Cross asset momentum
    "CrossAssetMomentumRanker",
    "AssetClass",
    "AssetClassType",
    "AssetClassScore",
    "RankingResult",
    "AllocationAdjustment",
    "DEFAULT_ASSET_CLASSES",
    "create_cross_asset_ranker",
    "quick_rank_asset_classes",
    # Regime signal parameters (flat structure)
    "RegimeSignalParamSelector",
    "SignalRegime",
    "RegimeParams",
    "AppliedParams",
    "REGIME_SIGNAL_PARAMS_FLAT",
    "get_regime_signal_params",
    "interpolate_regime_params",
    "apply_regime_params",
    "apply_regime_params_full",
    "list_signal_regimes",
    "create_regime_signal_param_selector",
    # VIX signal (IMP-003)
    "EnhancedVIXSignal",
    "VIXSignalConfig",
    "VIXSignalResult",
    "VIXTier",
    "EmergencyTrigger",
    "get_vix_cash_allocation",
    "calculate_vix_adjusted_weights",
    # Enhanced Mean Reversion (IMP-006)
    "EnhancedMeanReversion",
    "MeanReversionConfig",
    "MomentumMeanReversionBlender",
    "calculate_mean_reversion_score",
    # Enhanced Regime Detector V2 (IMP-007)
    "EnhancedRegimeDetector",
    "RegimeDetectorConfig",
    "RegimeResult",
    "RegimeType",
    "RegimeAdaptiveStrategy",
    "detect_market_regime",
    "get_regime_adjusted_weights",
    # Enhanced Sector Rotation V2 (IMP-008)
    "EnhancedSectorRotation",
    "EconomicCycleDetector",
    "EconomicCycle",
    "SectorMomentumCalculator",
    "RelativeStrengthCalculator",
    "RotationResult",
    "SectorScore",
    "ECONOMIC_CYCLE_SECTORS",
    "SECTOR_CYCLICALITY",
    "create_sector_rotation_strategy",
    "get_current_cycle_sectors",
    # Short Interest Signal (task_042_5)
    "ShortInterestSignal",
    "ShortInterestChangeSignal",
    "compute_short_interest_signal",
    # Insider Trading Signal (task_042_4)
    "InsiderTradingSignal",
    "InsiderSignalConfig",
    "compute_insider_signal",
    # Short-Term Reversal Signal (task_042_3)
    "ShortTermReversalSignal",
    "WeeklyShortTermReversalSignal",
    "MonthlyShortTermReversalSignal",
    # Volume signals (Order Flow)
    "OBVMomentumSignal",
    "MoneyFlowIndexSignal",
    "VWAPDeviationSignal",
    "AccumulationDistributionSignal",
    # Correlation signals (Regime Detection)
    "CorrelationRegimeSignal",
    "ReturnDispersionSignal",
    "CrossAssetCorrelationSignal",
    # Advanced Technical signals (Adaptive Indicators)
    "KAMASignal",
    "KeltnerChannelSignal",
    # Fifty Two Week High (George & Hwang 2004)
    "FiftyTwoWeekHighMomentumSignal",
    # Meta Validation (Optimization Level Selection)
    "MetaValidationCache",
    "MetaValidator",
    "MetaValidationResult",
    "LevelValidationResult",
    "AdaptiveParameterCalculator",
    "OptimizationLevel",
    "create_adaptive_calculator",
    # Lead-Lag Signal (Oxford研究ベース)
    "LeadLagSignal",
    "LeadLagPair",
    "compute_lead_lag_signal",
    # Low Volatility Premium (ファクター投資)
    "LowVolPremiumSignal",
    "LowVolPremiumStrategy",
    "LowVolPremiumConfig",
    "calculate_volatility_rank",
    "apply_low_vol_premium",
    # Enhanced Yield Curve Signal (マクロ)
    "EnhancedYieldCurveRegisteredSignal",
    "EnhancedYieldCurveSignal",
    "CurveShape",
    "get_current_yield_curve_shape",
    "get_yield_curve_allocation_adjustment",
]

__version__ = "1.0.0"
