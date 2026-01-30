"""
Machine Learning Module - 機械学習モジュール

戦略のスタッキングやアンサンブル学習を提供する。

主な機能:
- XGBoostスタッキング
- LightGBMスタッキング
- アンサンブルスタッキング（複数モデルの組み合わせ）
- リターン予測（LightGBM/RandomForest）
"""

from src.ml.xgboost_stacking import (
    XGBoostStacker,
    LightGBMStacker,
    EnsembleStacker,
    RidgeStacker,
    create_stacker,
    get_available_stackers,
)

from src.ml.return_predictor import (
    ReturnPredictor,
    MultiAssetPredictor,
    PredictorConfig,
    PredictionResult,
    create_return_predictor,
    predict_returns,
    get_return_features,
)

from src.ml.dynamic_ensemble_weights import (
    DynamicEnsembleWeightLearner,
    RegimeAwareEnsemble,
    ModelPerformance,
    EnsembleWeights,
    create_dynamic_ensemble_learner,
    create_regime_aware_ensemble,
)

from src.ml.ppo_allocator import (
    RLConfig,
    EnvironmentConfig,
    TradingEnvironment,
    SimplePPOAgent,
    train_ppo_allocator,
    create_ppo_allocator,
)

__all__ = [
    # Stacking
    "XGBoostStacker",
    "LightGBMStacker",
    "EnsembleStacker",
    "RidgeStacker",
    "create_stacker",
    "get_available_stackers",
    # Return Prediction
    "ReturnPredictor",
    "MultiAssetPredictor",
    "PredictorConfig",
    "PredictionResult",
    "create_return_predictor",
    "predict_returns",
    "get_return_features",
    # Dynamic Ensemble Weights
    "DynamicEnsembleWeightLearner",
    "RegimeAwareEnsemble",
    "ModelPerformance",
    "EnsembleWeights",
    "create_dynamic_ensemble_learner",
    "create_regime_aware_ensemble",
    # PPO Allocator
    "RLConfig",
    "EnvironmentConfig",
    "TradingEnvironment",
    "SimplePPOAgent",
    "train_ppo_allocator",
    "create_ppo_allocator",
]
