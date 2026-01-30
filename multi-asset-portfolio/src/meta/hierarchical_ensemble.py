"""
Hierarchical Ensemble Module - 階層的アンサンブル

3レイヤー構成のシグナル統合システム:
- Trend Layer: MultiTimeframe, DualMomentum, AdaptiveTrend, TrendFollowing
- Reversion Layer: Bollinger, RSI, ZScore, Stochastic
- Macro Layer: MacroRegimeComposite, FearGreedComposite, CreditSpread, YieldCurve

各レイヤー内でスタッキングモデル（Ridge/XGBoost）を学習し、
レイヤー間はレジーム適応重み付けで統合する。

設計根拠:
- 異なる市場環境で異なるシグナルが有効
- レジームに応じてレイヤーの重みを動的に調整
- スタッキングで各レイヤー内のシグナルを最適に組み合わせ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StackingModelType(str, Enum):
    """スタッキングモデルのタイプ"""
    RIDGE = "ridge"
    XGBOOST = "xgboost"
    SIMPLE_AVG = "simple_avg"


@dataclass(frozen=True)
class LayerWeights:
    """レジームごとのレイヤー重み"""
    trend: float
    reversion: float
    macro: float

    def to_dict(self) -> Dict[str, float]:
        return {"trend": self.trend, "reversion": self.reversion, "macro": self.macro}

    def validate(self) -> bool:
        """重みの合計が1.0かチェック"""
        return abs(self.trend + self.reversion + self.macro - 1.0) < 1e-6


# レジーム別のレイヤー重み設定
REGIME_LAYER_WEIGHTS: Dict[str, LayerWeights] = {
    "bull_trend": LayerWeights(trend=0.50, reversion=0.20, macro=0.30),
    "bear_market": LayerWeights(trend=0.20, reversion=0.30, macro=0.50),
    "high_vol": LayerWeights(trend=0.25, reversion=0.25, macro=0.50),
    "low_vol": LayerWeights(trend=0.45, reversion=0.35, macro=0.20),
    "range": LayerWeights(trend=0.25, reversion=0.50, macro=0.25),
    "default": LayerWeights(trend=0.33, reversion=0.34, macro=0.33),
}


class SignalProvider(Protocol):
    """シグナルプロバイダーのプロトコル"""
    def compute(self, data: pd.DataFrame) -> Any:
        ...


@dataclass
class SignalConfig:
    """シグナル設定"""
    name: str
    signal_class: Optional[Type] = None
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    required: bool = False  # Trueの場合、利用不可でエラー


@dataclass
class LayerConfig:
    """レイヤー設定"""
    name: str
    signals: List[SignalConfig] = field(default_factory=list)
    stacking_model_type: StackingModelType = StackingModelType.RIDGE
    fallback_method: str = "equal_weight"  # シグナルが足りない場合のフォールバック


@dataclass
class HierarchicalEnsembleConfig:
    """階層的アンサンブルの設定"""
    trend_layer: LayerConfig = field(default_factory=lambda: LayerConfig(name="trend"))
    reversion_layer: LayerConfig = field(default_factory=lambda: LayerConfig(name="reversion"))
    macro_layer: LayerConfig = field(default_factory=lambda: LayerConfig(name="macro"))
    regime_weights: Dict[str, LayerWeights] = field(default_factory=lambda: REGIME_LAYER_WEIGHTS.copy())
    default_regime: str = "default"
    min_signals_per_layer: int = 1


@dataclass
class LayerResult:
    """レイヤーの計算結果"""
    name: str
    score: pd.Series
    signal_scores: Dict[str, pd.Series] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    signals_available: int = 0
    signals_total: int = 0
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score_mean": float(self.score.mean()) if self.is_valid else None,
            "score_std": float(self.score.std()) if self.is_valid else None,
            "weights_used": self.weights_used,
            "signals_available": self.signals_available,
            "signals_total": self.signals_total,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


@dataclass
class HierarchicalEnsembleResult:
    """階層的アンサンブルの結果"""
    final_score: pd.Series
    regime: str
    layer_weights: LayerWeights
    trend_result: LayerResult
    reversion_result: LayerResult
    macro_result: LayerResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_score_mean": float(self.final_score.mean()),
            "final_score_std": float(self.final_score.std()),
            "regime": self.regime,
            "layer_weights": self.layer_weights.to_dict(),
            "trend": self.trend_result.to_dict(),
            "reversion": self.reversion_result.to_dict(),
            "macro": self.macro_result.to_dict(),
            "metadata": self.metadata,
        }


class BaseStackingModel:
    """スタッキングモデルの基底クラス"""

    def __init__(self):
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.weights: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseStackingModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class SimpleAverageModel(BaseStackingModel):
    """シンプルな平均モデル"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleAverageModel":
        self.feature_names = list(X.columns)
        n = len(self.feature_names)
        self.weights = {name: 1.0 / n for name in self.feature_names}
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return X.mean(axis=1)


class RidgeStackingModel(BaseStackingModel):
    """Ridgeベースのスタッキングモデル"""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self._model = None
        self._fallback_model: Optional[SimpleAverageModel] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeStackingModel":
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            logger.warning("sklearn not available, falling back to simple average")
            self._fallback_model = SimpleAverageModel().fit(X, y)
            self.feature_names = self._fallback_model.feature_names
            self.weights = self._fallback_model.weights
            self.is_fitted = True
            return self

        self.feature_names = list(X.columns)

        # Handle NaN values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)

        # Fit Ridge model
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_clean.values, y_clean.values)

        # Store weights
        self.weights = dict(zip(self.feature_names, self._model.coef_))
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Check if using fallback model
        if self._fallback_model is not None:
            return self._fallback_model.predict(X)

        if self._model is None:
            raise RuntimeError("Model not fitted")

        X_clean = X.fillna(0)
        predictions = self._model.predict(X_clean.values)
        return pd.Series(predictions, index=X.index)


class XGBoostStackingModel(BaseStackingModel):
    """XGBoostベースのスタッキングモデル"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None
        self._fallback_model: Optional[BaseStackingModel] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostStackingModel":
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not available, falling back to Ridge")
            self._fallback_model = RidgeStackingModel().fit(X, y)
            self.feature_names = self._fallback_model.feature_names
            self.weights = self._fallback_model.weights
            self.is_fitted = True
            return self

        self.feature_names = list(X.columns)

        # Handle NaN values
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)

        # Fit XGBoost model
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
            verbosity=0,
        )
        self._model.fit(X_clean.values, y_clean.values)

        # Store feature importances as weights
        importances = self._model.feature_importances_
        total = importances.sum()
        if total > 0:
            self.weights = dict(zip(self.feature_names, importances / total))
        else:
            self.weights = {name: 1.0 / len(self.feature_names) for name in self.feature_names}

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Check if using fallback model
        if self._fallback_model is not None:
            return self._fallback_model.predict(X)

        if self._model is None:
            raise RuntimeError("Model not fitted")

        X_clean = X.fillna(0)
        predictions = self._model.predict(X_clean.values)
        return pd.Series(predictions, index=X.index)


def create_stacking_model(model_type: StackingModelType) -> BaseStackingModel:
    """スタッキングモデルを生成"""
    if model_type == StackingModelType.RIDGE:
        return RidgeStackingModel()
    elif model_type == StackingModelType.XGBOOST:
        return XGBoostStackingModel()
    else:
        return SimpleAverageModel()


class SignalLayer:
    """シグナルレイヤー

    複数のシグナルを統合してレイヤースコアを計算する。
    """

    def __init__(self, config: LayerConfig):
        self.config = config
        self.name = config.name
        self.signals: Dict[str, Any] = {}
        self.stacking_model: Optional[BaseStackingModel] = None
        self._is_fitted = False

    def register_signal(self, name: str, signal: Any, weight: float = 1.0) -> None:
        """シグナルを登録"""
        self.signals[name] = {"instance": signal, "weight": weight}
        logger.debug(f"Registered signal '{name}' to layer '{self.name}'")

    def compute_signal_scores(
        self,
        data: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.Series]:
        """登録されたシグナルからスコアを計算"""
        scores = {}
        additional_data = additional_data or {}

        for signal_name, signal_info in self.signals.items():
            signal = signal_info["instance"]
            try:
                # シグナルのcomputeメソッドを呼び出し
                if hasattr(signal, "compute"):
                    result = signal.compute(data, **additional_data)
                    if hasattr(result, "scores"):
                        scores[signal_name] = result.scores
                    elif isinstance(result, pd.Series):
                        scores[signal_name] = result
                    else:
                        logger.warning(f"Signal '{signal_name}' returned unexpected type")
                else:
                    logger.warning(f"Signal '{signal_name}' has no compute method")
            except Exception as e:
                logger.warning(f"Error computing signal '{signal_name}': {e}")
                # フォールバック: ゼロスコアを返す
                scores[signal_name] = pd.Series(0.0, index=data.index)

        return scores

    def fit(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> "SignalLayer":
        """スタッキングモデルを学習"""
        # シグナルスコアを計算
        signal_scores = self.compute_signal_scores(data, additional_data)

        if not signal_scores:
            logger.warning(f"No signals available for layer '{self.name}', using dummy model")
            self.stacking_model = SimpleAverageModel()
            self._is_fitted = True
            return self

        # 特徴量DataFrameを構築
        X = pd.DataFrame(signal_scores)

        # アライメント
        common_index = X.index.intersection(target.index)
        X = X.loc[common_index]
        y = target.loc[common_index]

        # スタッキングモデルを作成して学習
        self.stacking_model = create_stacking_model(self.config.stacking_model_type)
        self.stacking_model.fit(X, y)
        self._is_fitted = True

        logger.info(
            f"Layer '{self.name}' fitted with {len(signal_scores)} signals, "
            f"model type: {self.config.stacking_model_type}"
        )

        return self

    def predict(
        self,
        data: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> LayerResult:
        """レイヤースコアを予測"""
        signal_scores = self.compute_signal_scores(data, additional_data)

        if not signal_scores:
            # シグナルがない場合はゼロスコア
            return LayerResult(
                name=self.name,
                score=pd.Series(0.0, index=data.index),
                is_valid=False,
                metadata={"error": "No signals available"},
            )

        X = pd.DataFrame(signal_scores)

        # スタッキングモデルで予測
        if self._is_fitted and self.stacking_model is not None:
            score = self.stacking_model.predict(X)
            weights_used = self.stacking_model.weights
        else:
            # 未学習の場合は単純平均
            score = X.mean(axis=1)
            n = len(signal_scores)
            weights_used = {name: 1.0 / n for name in signal_scores.keys()}

        # スコアを[-1, +1]にクリップ
        score = score.clip(-1, 1)

        return LayerResult(
            name=self.name,
            score=score,
            signal_scores=signal_scores,
            weights_used=weights_used,
            signals_available=len(signal_scores),
            signals_total=len(self.signals),
            is_valid=True,
        )


class HierarchicalEnsemble:
    """階層的アンサンブル

    3つのシグナルレイヤーを統合し、レジームに応じて重み付けする。

    Usage:
        ensemble = HierarchicalEnsemble()

        # シグナルを登録
        ensemble.register_trend_signal("momentum", MomentumSignal())
        ensemble.register_reversion_signal("rsi", RSISignal())
        ensemble.register_macro_signal("yield_curve", YieldCurveSignal())

        # 学習
        ensemble.fit(prices, returns)

        # 予測
        result = ensemble.predict(prices, regime="bull_trend")
        final_score = result.final_score
    """

    def __init__(
        self,
        config: Optional[HierarchicalEnsembleConfig] = None,
        stacking_model_type: str = "ridge",
    ):
        """初期化

        Args:
            config: アンサンブル設定
            stacking_model_type: スタッキングモデルタイプ ("ridge", "xgboost", "simple_avg")
        """
        self.config = config or HierarchicalEnsembleConfig()

        # モデルタイプを設定に反映
        model_type = StackingModelType(stacking_model_type)
        self.config.trend_layer.stacking_model_type = model_type
        self.config.reversion_layer.stacking_model_type = model_type
        self.config.macro_layer.stacking_model_type = model_type

        # レイヤーを初期化
        self.trend_layer = SignalLayer(self.config.trend_layer)
        self.reversion_layer = SignalLayer(self.config.reversion_layer)
        self.macro_layer = SignalLayer(self.config.macro_layer)

        self._is_fitted = False

    def register_trend_signal(
        self,
        name: str,
        signal: Any,
        weight: float = 1.0,
    ) -> None:
        """トレンドレイヤーにシグナルを登録"""
        self.trend_layer.register_signal(name, signal, weight)

    def register_reversion_signal(
        self,
        name: str,
        signal: Any,
        weight: float = 1.0,
    ) -> None:
        """リバージョンレイヤーにシグナルを登録"""
        self.reversion_layer.register_signal(name, signal, weight)

    def register_macro_signal(
        self,
        name: str,
        signal: Any,
        weight: float = 1.0,
    ) -> None:
        """マクロレイヤーにシグナルを登録"""
        self.macro_layer.register_signal(name, signal, weight)

    def fit(
        self,
        prices: pd.DataFrame,
        returns: pd.Series,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> "HierarchicalEnsemble":
        """各レイヤーのスタッキングモデルを学習

        Args:
            prices: 価格DataFrame (OHLCV)
            returns: ターゲットリターン
            additional_data: 追加データ（マクロ指標など）

        Returns:
            self
        """
        additional_data = additional_data or {}

        # 各レイヤーを学習
        self.trend_layer.fit(prices, returns, additional_data)
        self.reversion_layer.fit(prices, returns, additional_data)
        self.macro_layer.fit(prices, returns, additional_data)

        self._is_fitted = True
        logger.info("HierarchicalEnsemble fitted successfully")

        return self

    def predict(
        self,
        prices: pd.DataFrame,
        regime: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalEnsembleResult:
        """最終スコアを予測

        Args:
            prices: 価格DataFrame (OHLCV)
            regime: 現在のレジーム名
            additional_data: 追加データ

        Returns:
            HierarchicalEnsembleResult
        """
        additional_data = additional_data or {}

        # 各レイヤーのスコアを計算
        trend_result = self.trend_layer.predict(prices, additional_data)
        reversion_result = self.reversion_layer.predict(prices, additional_data)
        macro_result = self.macro_layer.predict(prices, additional_data)

        # レジーム重みを取得
        regime = regime or self.config.default_regime
        layer_weights = self.config.regime_weights.get(
            regime,
            self.config.regime_weights.get("default", REGIME_LAYER_WEIGHTS["default"]),
        )

        # レイヤースコアを統合
        # 無効なレイヤーは重みを0にして再正規化
        valid_weights = {}
        layer_scores = {}

        if trend_result.is_valid:
            valid_weights["trend"] = layer_weights.trend
            layer_scores["trend"] = trend_result.score
        if reversion_result.is_valid:
            valid_weights["reversion"] = layer_weights.reversion
            layer_scores["reversion"] = reversion_result.score
        if macro_result.is_valid:
            valid_weights["macro"] = layer_weights.macro
            layer_scores["macro"] = macro_result.score

        # 重みを正規化
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
        else:
            # 全レイヤーが無効な場合はゼロスコア
            return HierarchicalEnsembleResult(
                final_score=pd.Series(0.0, index=prices.index),
                regime=regime,
                layer_weights=layer_weights,
                trend_result=trend_result,
                reversion_result=reversion_result,
                macro_result=macro_result,
                metadata={"error": "All layers invalid"},
            )

        # 最終スコアを計算
        final_score = pd.Series(0.0, index=prices.index)
        for layer_name, weight in normalized_weights.items():
            if layer_name in layer_scores:
                final_score += weight * layer_scores[layer_name]

        # クリップ
        final_score = final_score.clip(-1, 1)

        return HierarchicalEnsembleResult(
            final_score=final_score,
            regime=regime,
            layer_weights=layer_weights,
            trend_result=trend_result,
            reversion_result=reversion_result,
            macro_result=macro_result,
            metadata={
                "effective_weights": normalized_weights,
                "is_fitted": self._is_fitted,
            },
        )

    def detect_regime(
        self,
        prices: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """価格データからレジームを検出

        Args:
            prices: 価格DataFrame
            additional_data: VIXなどの追加データ

        Returns:
            検出されたレジーム名
        """
        additional_data = additional_data or {}
        close = prices["close"]

        # モメンタムを計算
        momentum_20d = close.pct_change(periods=20).iloc[-1]
        momentum_60d = close.pct_change(periods=60).iloc[-1]

        # ボラティリティを計算
        returns = close.pct_change()
        vol_20d = returns.tail(20).std() * np.sqrt(252)

        # VIXがあれば使用
        vix = additional_data.get("vix", vol_20d * 100)

        # レジーム判定
        # Crisis: VIX > 30
        if vix > 30:
            return "high_vol"

        # Bear Market: momentum_60d < -10%, VIX > 25
        if momentum_60d < -0.10 and vix > 25:
            return "bear_market"

        # Bull Trend: momentum_20d > 5%, VIX < 20
        if momentum_20d > 0.05 and vix < 20:
            return "bull_trend"

        # Low Vol: VIX < 15
        if vix < 15:
            return "low_vol"

        # Range: default
        return "range"


def create_default_hierarchical_ensemble(
    stacking_model_type: str = "ridge",
) -> HierarchicalEnsemble:
    """デフォルト設定で階層的アンサンブルを作成

    シグナルは登録されていない状態で返す。
    使用前に各シグナルを登録する必要がある。

    Args:
        stacking_model_type: "ridge", "xgboost", "simple_avg"

    Returns:
        HierarchicalEnsemble
    """
    return HierarchicalEnsemble(stacking_model_type=stacking_model_type)


def create_hierarchical_ensemble_with_signals(
    stacking_model_type: str = "ridge",
) -> HierarchicalEnsemble:
    """シグナル付きで階層的アンサンブルを作成

    利用可能なシグナルを自動的に登録する。
    SignalRegistry経由でシグナルを取得し、循環依存を回避。

    Args:
        stacking_model_type: "ridge", "xgboost", "simple_avg"

    Returns:
        HierarchicalEnsemble
    """
    ensemble = HierarchicalEnsemble(stacking_model_type=stacking_model_type)

    # シグナルモジュールをインポートしてRegistryに登録
    # Note: src.signals パッケージのインポートで全シグナルがRegistryに登録される
    import src.signals  # noqa: F401 - triggers signal registration

    # SignalRegistry経由でシグナルを取得（循環依存を回避）
    from src.signals.registry import SignalRegistry, SignalRegistryError

    # トレンドレイヤーのシグナルを登録
    trend_signals = [
        ("multi_timeframe", "multi_timeframe_momentum"),
        ("dual_momentum", "dual_momentum"),
        ("adaptive_trend", "adaptive_trend"),
        ("trend_following", "trend_following"),
    ]
    for ensemble_name, registry_name in trend_signals:
        try:
            signal = SignalRegistry.create(registry_name)
            ensemble.register_trend_signal(ensemble_name, signal)
        except SignalRegistryError as e:
            logger.warning(f"Trend signal '{registry_name}' not found in registry: {e}")
    if ensemble.trend_layer.signals:
        logger.info(f"Trend layer signals registered: {list(ensemble.trend_layer.signals.keys())}")

    # リバージョンレイヤーのシグナルを登録
    reversion_signals = [
        ("bollinger", "bollinger_reversion"),
        ("rsi", "rsi"),
        ("zscore", "zscore_reversion"),
        ("stochastic", "stochastic_reversion"),
    ]
    for ensemble_name, registry_name in reversion_signals:
        try:
            signal = SignalRegistry.create(registry_name)
            ensemble.register_reversion_signal(ensemble_name, signal)
        except SignalRegistryError as e:
            logger.warning(f"Reversion signal '{registry_name}' not found in registry: {e}")
    if ensemble.reversion_layer.signals:
        logger.info(f"Reversion layer signals registered: {list(ensemble.reversion_layer.signals.keys())}")

    # マクロレイヤーのシグナルを登録
    macro_signals = [
        ("macro_regime", "macro_regime_composite"),
        ("fear_greed", "fear_greed_composite"),
        ("credit_spread", "credit_spread"),
        ("yield_curve", "yield_curve"),
    ]
    for ensemble_name, registry_name in macro_signals:
        try:
            signal = SignalRegistry.create(registry_name)
            ensemble.register_macro_signal(ensemble_name, signal)
        except SignalRegistryError as e:
            logger.warning(f"Macro signal '{registry_name}' not found in registry: {e}")
    if ensemble.macro_layer.signals:
        logger.info(f"Macro layer signals registered: {list(ensemble.macro_layer.signals.keys())}")

    return ensemble
