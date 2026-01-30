"""
XGBoost Stacking Module - 勾配ブースティングによるスタッキング

複数の戦略シグナルをスタッキングして最終予測を生成する。
XGBoost, LightGBM, Ridge のスタッカーを提供。

設計根拠:
- 要求.md §7: Meta層
- 線形モデル（Ridge）からGBDTへのアップグレード
- 特徴量重要度による戦略貢献度分析

主な機能:
- XGBoostStacker: XGBoostによるスタッキング
- LightGBMStacker: LightGBMによるスタッキング
- EnsembleStacker: 複数モデルのアンサンブル
- RidgeStacker: Ridgeによるフォールバック

注意:
- xgboost/lightgbm がインストールされていない場合は
  RidgeStacker にフォールバック
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# オプショナル依存のチェック
_HAS_XGBOOST = False
_HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    logger.info("xgboost not available, XGBoostStacker will fallback to Ridge")

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    logger.info("lightgbm not available, LightGBMStacker will fallback to Ridge")


@dataclass
class StackerResult:
    """スタッカー結果

    Attributes:
        predictions: 予測値
        feature_importance: 特徴量重要度（あれば）
        model_info: モデル情報
    """
    predictions: np.ndarray
    feature_importance: Optional[pd.Series] = None
    model_info: Dict[str, Any] = field(default_factory=dict)


class BaseStacker(ABC):
    """スタッカー基底クラス

    全てのスタッカーが継承する抽象基底クラス。
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "BaseStacker":
        """モデルを学習

        Args:
            X: 特徴量（各戦略のシグナル）
            y: 目的変数（リターン等）
            eval_set: 検証用データセット（early stopping用）

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を生成

        Args:
            X: 特徴量

        Returns:
            予測値の配列
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """特徴量重要度を取得

        Returns:
            特徴量名をインデックスとした重要度のSeries（降順）
        """
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """モデルが学習済みかどうか"""
        pass


class RidgeStacker(BaseStacker):
    """Ridgeスタッカー

    Ridge回帰によるスタッキング。
    XGBoost/LightGBMがない場合のフォールバック。

    Attributes:
        alpha: 正則化強度
        normalize: 正規化するか
    """

    def __init__(
        self,
        alpha: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """初期化

        Args:
            alpha: 正則化強度
            normalize: 正規化するか
        """
        self.alpha = alpha
        self.normalize = normalize
        self._model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "RidgeStacker":
        """モデルを学習"""
        self._feature_names = list(X.columns)

        X_arr = X.values
        y_arr = y.values

        if self.normalize:
            self._scaler = StandardScaler()
            X_arr = self._scaler.fit_transform(X_arr)

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_arr, y_arr)
        self._is_fitted = True

        logger.info(
            "RidgeStacker fitted: alpha=%.4f, n_features=%d",
            self.alpha, len(self._feature_names)
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を生成"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)

        return self._model.predict(X_arr)

    def get_feature_importance(self) -> pd.Series:
        """特徴量重要度を取得（係数の絶対値）"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        )
        return importance.sort_values(ascending=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class XGBoostStacker(BaseStacker):
    """XGBoostスタッカー

    XGBoostによるスタッキング。
    特徴量重要度の取得とearly stoppingをサポート。

    XGBoostがインストールされていない場合はRidgeにフォールバック。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 10,
        random_state: int = 42,
    ) -> None:
        """初期化

        Args:
            n_estimators: ブースティングラウンド数
            max_depth: 木の最大深度
            learning_rate: 学習率
            subsample: サンプリング率
            colsample_bytree: 列サンプリング率
            early_stopping_rounds: early stoppingラウンド数
            random_state: 乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._model: Any = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._use_fallback = not _HAS_XGBOOST
        self._fallback_model: Optional[RidgeStacker] = None

        if self._use_fallback:
            logger.warning(
                "XGBoost not available, using Ridge fallback"
            )
            self._fallback_model = RidgeStacker()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "XGBoostStacker":
        """モデルを学習"""
        if self._use_fallback:
            self._fallback_model.fit(X, y)
            self._is_fitted = True
            return self

        self._feature_names = list(X.columns)

        # XGBRegressorの設定
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )

        # 学習
        fit_params: Dict[str, Any] = {}

        if eval_set is not None:
            # 検証セットがある場合はearly stopping使用
            eval_data = [(df.values, s.values) for df, s in eval_set]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(
                    X.values,
                    y.values,
                    eval_set=eval_data,
                    verbose=False,
                )
        else:
            self._model.fit(X.values, y.values)

        self._is_fitted = True

        logger.info(
            "XGBoostStacker fitted: n_estimators=%d, max_depth=%d, "
            "n_features=%d",
            self.n_estimators, self.max_depth, len(self._feature_names)
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を生成"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._use_fallback:
            return self._fallback_model.predict(X)

        return self._model.predict(X.values)

    def get_feature_importance(self) -> pd.Series:
        """特徴量重要度を取得"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._use_fallback:
            return self._fallback_model.get_feature_importance()

        importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        )
        return importance.sort_values(ascending=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class LightGBMStacker(BaseStacker):
    """LightGBMスタッカー

    LightGBMによるスタッキング。
    XGBoostより高速で、カテゴリカル特徴量をネイティブサポート。

    LightGBMがインストールされていない場合はRidgeにフォールバック。
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        """初期化

        Args:
            n_estimators: ブースティングラウンド数
            max_depth: 木の最大深度
            learning_rate: 学習率
            num_leaves: 葉の数
            subsample: サンプリング率
            colsample_bytree: 列サンプリング率
            random_state: 乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self._model: Any = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._use_fallback = not _HAS_LIGHTGBM
        self._fallback_model: Optional[RidgeStacker] = None

        if self._use_fallback:
            logger.warning(
                "LightGBM not available, using Ridge fallback"
            )
            self._fallback_model = RidgeStacker()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "LightGBMStacker":
        """モデルを学習"""
        if self._use_fallback:
            self._fallback_model.fit(X, y)
            self._is_fitted = True
            return self

        self._feature_names = list(X.columns)

        # LGBMRegressorの設定
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=-1,
        )

        # 学習
        self._model.fit(X.values, y.values)
        self._is_fitted = True

        logger.info(
            "LightGBMStacker fitted: n_estimators=%d, num_leaves=%d, "
            "n_features=%d",
            self.n_estimators, self.num_leaves, len(self._feature_names)
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を生成"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._use_fallback:
            return self._fallback_model.predict(X)

        return self._model.predict(X.values)

    def get_feature_importance(self) -> pd.Series:
        """特徴量重要度を取得"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._use_fallback:
            return self._fallback_model.get_feature_importance()

        importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        )
        return importance.sort_values(ascending=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class EnsembleStacker(BaseStacker):
    """アンサンブルスタッカー

    複数のスタッカーを組み合わせてアンサンブル予測を生成する。
    デフォルトでは Ridge + XGBoost を使用。

    Usage:
        ensemble = EnsembleStacker()
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        models: Optional[List[BaseStacker]] = None,
        weights: Optional[List[float]] = None,
    ) -> None:
        """初期化

        Args:
            models: スタッカーのリスト。Noneの場合はRidge + XGBoost
            weights: 各モデルの重み。Noneの場合は均等
        """
        if models is None:
            models = [
                RidgeStacker(),
                XGBoostStacker(),
            ]

        self.models = models

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        elif len(weights) != len(models):
            raise ValueError("weights length must match models length")
        else:
            # 正規化
            total = sum(weights)
            weights = [w / total for w in weights]

        self.weights = weights
        self._feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "EnsembleStacker":
        """全モデルを学習"""
        self._feature_names = list(X.columns)

        for i, model in enumerate(self.models):
            logger.info(
                "Fitting ensemble model %d/%d: %s",
                i + 1, len(self.models), model.__class__.__name__
            )
            model.fit(X, y, eval_set)

        self._is_fitted = True

        logger.info(
            "EnsembleStacker fitted: %d models, weights=%s",
            len(self.models),
            [f"{w:.2f}" for w in self.weights]
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """加重平均による予測を生成"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = np.zeros(len(X))

        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)

        return predictions

    def get_feature_importance(self) -> pd.Series:
        """加重平均した特徴量重要度を取得"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = pd.Series(0.0, index=self._feature_names)

        for model, weight in zip(self.models, self.weights):
            model_importance = model.get_feature_importance()
            # 正規化してから加重
            if model_importance.sum() > 0:
                normalized = model_importance / model_importance.sum()
                importance += weight * normalized

        return importance.sort_values(ascending=False)

    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """各モデルの個別予測を取得

        Args:
            X: 特徴量

        Returns:
            {モデル名: 予測値} の辞書
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return {
            model.__class__.__name__: model.predict(X)
            for model in self.models
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ============================================================
# 便利関数
# ============================================================

def create_stacker(
    stacker_type: str = "xgboost",
    **kwargs: Any,
) -> BaseStacker:
    """スタッカーを作成

    Args:
        stacker_type: スタッカータイプ（'xgboost', 'lightgbm', 'ridge', 'ensemble'）
        **kwargs: スタッカー固有のパラメータ

    Returns:
        スタッカーインスタンス
    """
    stacker_map = {
        "xgboost": XGBoostStacker,
        "xgb": XGBoostStacker,
        "lightgbm": LightGBMStacker,
        "lgbm": LightGBMStacker,
        "ridge": RidgeStacker,
        "ensemble": EnsembleStacker,
    }

    stacker_type_lower = stacker_type.lower()
    if stacker_type_lower not in stacker_map:
        raise ValueError(
            f"Unknown stacker type: {stacker_type}. "
            f"Available: {list(stacker_map.keys())}"
        )

    return stacker_map[stacker_type_lower](**kwargs)


def get_available_stackers() -> Dict[str, bool]:
    """利用可能なスタッカーを取得

    Returns:
        {スタッカー名: 利用可能かどうか} の辞書
    """
    return {
        "xgboost": _HAS_XGBOOST,
        "lightgbm": _HAS_LIGHTGBM,
        "ridge": True,  # 常に利用可能
        "ensemble": True,  # 常に利用可能
    }


def stack_signals(
    X: pd.DataFrame,
    y: pd.Series,
    stacker_type: str = "ensemble",
    **kwargs: Any,
) -> Tuple[BaseStacker, StackerResult]:
    """シグナルをスタッキング（便利関数）

    Args:
        X: 特徴量（各戦略のシグナル）
        y: 目的変数（リターン等）
        stacker_type: スタッカータイプ
        **kwargs: スタッカー固有のパラメータ

    Returns:
        (スタッカー, 結果) のタプル
    """
    stacker = create_stacker(stacker_type, **kwargs)
    stacker.fit(X, y)

    predictions = stacker.predict(X)
    importance = stacker.get_feature_importance()

    result = StackerResult(
        predictions=predictions,
        feature_importance=importance,
        model_info={
            "stacker_type": stacker_type,
            "n_features": len(X.columns),
            "n_samples": len(X),
        },
    )

    return stacker, result


def cross_validate_stacker(
    X: pd.DataFrame,
    y: pd.Series,
    stacker: BaseStacker,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
) -> Dict[str, float]:
    """スタッカーをクロスバリデーション

    Args:
        X: 特徴量
        y: 目的変数
        stacker: スタッカーインスタンス
        cv: 分割数
        scoring: スコアリング方法

    Returns:
        {mean_score, std_score} の辞書
    """
    # sklearn互換のラッパー
    from sklearn.base import BaseEstimator, RegressorMixin

    class StackerWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, stacker: BaseStacker):
            self.stacker = stacker

        def fit(self, X: np.ndarray, y: np.ndarray) -> "StackerWrapper":
            X_df = pd.DataFrame(X)
            y_s = pd.Series(y)
            self.stacker.fit(X_df, y_s)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X)
            return self.stacker.predict(X_df)

    wrapper = StackerWrapper(stacker)
    scores = cross_val_score(wrapper, X.values, y.values, cv=cv, scoring=scoring)

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "scores": scores.tolist(),
    }
