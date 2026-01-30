"""
Dynamic Ensemble Weight Learning - 動的アンサンブル重み学習

モデルのパフォーマンスに基づいて動的にアンサンブル重みを調整する。
各モデルの過去の予測精度を追跡し、精度の高いモデルに高い重みを割り当てる。

主要コンポーネント:
- DynamicEnsembleWeightLearner: パフォーマンスベースの動的重み学習
- RegimeAwareEnsemble: レジーム別の重み管理

設計根拠:
- 指数加重平均: 直近のパフォーマンスを重視
- 逆誤差重み付け: 誤差の小さいモデルに高い重み
- 最小重み保証: どのモデルも完全に無視しない（多様性維持）

使用例:
    from src.ml.dynamic_ensemble_weights import DynamicEnsembleWeightLearner

    learner = DynamicEnsembleWeightLearner(
        models=["momentum", "reversion", "macro"],
        lookback=60,
        decay_factor=0.95,
    )

    # パフォーマンス更新
    learner.update_performance(
        model_predictions={"momentum": 0.02, "reversion": -0.01, "macro": 0.01},
        actual_return=0.015,
    )

    # 重み計算
    weights = learner.compute_weights()
    print(f"Weights: {weights}")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class ModelPerformance:
    """モデルパフォーマンス記録

    Attributes:
        model_name: モデル名
        predictions: 予測値の履歴
        actuals: 実績値の履歴
        errors: 誤差の履歴
        squared_errors: 二乗誤差の履歴
        timestamps: タイムスタンプ（オプション）
    """
    model_name: str
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    squared_errors: List[float] = field(default_factory=list)
    timestamps: List[Any] = field(default_factory=list)

    def add_observation(
        self,
        prediction: float,
        actual: float,
        timestamp: Optional[Any] = None,
    ) -> None:
        """観測を追加"""
        error = prediction - actual
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.errors.append(error)
        self.squared_errors.append(error ** 2)
        if timestamp is not None:
            self.timestamps.append(timestamp)

    def trim_to_lookback(self, lookback: int) -> None:
        """履歴をlookbackに制限"""
        if len(self.predictions) > lookback:
            self.predictions = self.predictions[-lookback:]
            self.actuals = self.actuals[-lookback:]
            self.errors = self.errors[-lookback:]
            self.squared_errors = self.squared_errors[-lookback:]
            if self.timestamps:
                self.timestamps = self.timestamps[-lookback:]

    def get_mse(self) -> float:
        """平均二乗誤差を計算"""
        if not self.squared_errors:
            return float('inf')
        return np.mean(self.squared_errors)

    def get_ewma_mse(self, decay_factor: float) -> float:
        """指数加重平均二乗誤差を計算"""
        if not self.squared_errors:
            return float('inf')

        n = len(self.squared_errors)
        weights = np.array([decay_factor ** (n - 1 - i) for i in range(n)])
        weights = weights / np.sum(weights)

        return float(np.dot(weights, self.squared_errors))

    def get_mae(self) -> float:
        """平均絶対誤差を計算"""
        if not self.errors:
            return float('inf')
        return np.mean(np.abs(self.errors))

    def get_hit_rate(self) -> float:
        """方向一致率を計算"""
        if len(self.predictions) < 2:
            return 0.5

        hits = 0
        for pred, actual in zip(self.predictions, self.actuals):
            if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or (pred == 0 and actual == 0):
                hits += 1

        return hits / len(self.predictions)


@dataclass
class EnsembleWeights:
    """アンサンブル重み結果

    Attributes:
        weights: モデル別重み
        performance_metrics: パフォーマンス指標
        n_observations: 観測数
        regime: レジーム（RegimeAwareEnsembleの場合）
    """
    weights: Dict[str, float]
    performance_metrics: Dict[str, Dict[str, float]]
    n_observations: int
    regime: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "performance_metrics": self.performance_metrics,
            "n_observations": self.n_observations,
            "regime": self.regime,
        }

    def get_weight(self, model_name: str) -> float:
        """モデルの重みを取得"""
        return self.weights.get(model_name, 0.0)


# =============================================================================
# DynamicEnsembleWeightLearner クラス
# =============================================================================

class DynamicEnsembleWeightLearner:
    """
    動的アンサンブル重み学習

    各モデルの過去のパフォーマンスに基づいて、アンサンブル重みを動的に調整する。
    指数加重平均二乗誤差の逆数に比例した重みを計算する。

    Usage:
        learner = DynamicEnsembleWeightLearner(
            models=["momentum", "reversion", "macro"],
            lookback=60,
            decay_factor=0.95,
            min_weight=0.05,
        )

        # パフォーマンス更新（毎期実行）
        learner.update_performance(
            model_predictions={"momentum": 0.02, "reversion": -0.01, "macro": 0.01},
            actual_return=0.015,
        )

        # 重み計算
        weights = learner.compute_weights()
        # {"momentum": 0.45, "reversion": 0.25, "macro": 0.30}

        # アンサンブル予測
        ensemble_pred = learner.ensemble_predict(next_predictions)
    """

    def __init__(
        self,
        models: List[str],
        lookback: int = 60,
        decay_factor: float = 0.95,
        min_weight: float = 0.05,
        initial_weight: Optional[float] = None,
    ) -> None:
        """
        初期化

        Args:
            models: モデル名のリスト
            lookback: パフォーマンス履歴の最大長
            decay_factor: 指数加重の減衰係数（0-1、1に近いほど過去を重視）
            min_weight: 各モデルの最小重み（多様性維持）
            initial_weight: 初期重み（Noneで均等配分）
        """
        self.models = models
        self.lookback = lookback
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.n_models = len(models)

        # 初期重み
        if initial_weight is None:
            self.initial_weight = 1.0 / self.n_models
        else:
            self.initial_weight = initial_weight

        # パフォーマンス記録
        self.performance: Dict[str, ModelPerformance] = {
            model: ModelPerformance(model_name=model) for model in models
        }

        # 観測カウンタ
        self.n_observations = 0

        logger.info(
            f"DynamicEnsembleWeightLearner initialized: "
            f"models={models}, lookback={lookback}, decay={decay_factor}"
        )

    def update_performance(
        self,
        model_predictions: Dict[str, float],
        actual_return: float,
        timestamp: Optional[Any] = None,
    ) -> None:
        """
        各モデルの予測精度を記録

        Args:
            model_predictions: モデル名 -> 予測値
            actual_return: 実績リターン
            timestamp: タイムスタンプ（オプション）
        """
        for model in self.models:
            if model in model_predictions:
                prediction = model_predictions[model]
                self.performance[model].add_observation(
                    prediction=prediction,
                    actual=actual_return,
                    timestamp=timestamp,
                )
                # 履歴制限
                self.performance[model].trim_to_lookback(self.lookback)

        self.n_observations += 1

        logger.debug(
            f"Performance updated: n_observations={self.n_observations}"
        )

    def compute_weights(self) -> Dict[str, float]:
        """
        パフォーマンスに基づいて重みを計算

        重み ∝ 1 / (指数加重平均二乗誤差)
        正規化 + 最小重み保証

        Returns:
            モデル名 -> 重み
        """
        if self.n_observations == 0:
            # データがない場合は均等配分
            return {model: self.initial_weight for model in self.models}

        # EWMA MSEを計算
        ewma_mse = {}
        for model in self.models:
            perf = self.performance[model]
            if len(perf.squared_errors) > 0:
                ewma_mse[model] = perf.get_ewma_mse(self.decay_factor)
            else:
                ewma_mse[model] = float('inf')

        # 逆誤差重み（MSEの逆数）
        inverse_weights = {}
        for model, mse in ewma_mse.items():
            if mse > 0 and mse != float('inf'):
                inverse_weights[model] = 1.0 / mse
            else:
                # MSEが0または無限大の場合は小さい値を割り当て
                inverse_weights[model] = 1e-10

        # 正規化
        total = sum(inverse_weights.values())
        if total > 0:
            weights = {model: w / total for model, w in inverse_weights.items()}
        else:
            weights = {model: self.initial_weight for model in self.models}

        # 最小重み保証
        weights = self._apply_min_weight(weights)

        return weights

    def _apply_min_weight(self, weights: Dict[str, float]) -> Dict[str, float]:
        """最小重みを保証して再正規化"""
        # 最小重み未満のモデルを引き上げ
        adjusted = {}
        below_min = []
        above_min = []

        for model, w in weights.items():
            if w < self.min_weight:
                adjusted[model] = self.min_weight
                below_min.append(model)
            else:
                above_min.append(model)

        # 残りの重みを配分
        total_below = len(below_min) * self.min_weight
        remaining = 1.0 - total_below

        if above_min:
            # 最小重み以上のモデルで残りを配分
            above_total = sum(weights[m] for m in above_min)
            if above_total > 0:
                for model in above_min:
                    adjusted[model] = weights[model] / above_total * remaining
            else:
                for model in above_min:
                    adjusted[model] = remaining / len(above_min)
        else:
            # 全モデルが最小重み未満の場合は均等配分
            equal_weight = 1.0 / self.n_models
            adjusted = {model: equal_weight for model in self.models}

        return adjusted

    def get_weights_result(self) -> EnsembleWeights:
        """重み結果を詳細に取得"""
        weights = self.compute_weights()

        # パフォーマンス指標
        metrics = {}
        for model in self.models:
            perf = self.performance[model]
            metrics[model] = {
                "mse": perf.get_mse(),
                "ewma_mse": perf.get_ewma_mse(self.decay_factor),
                "mae": perf.get_mae(),
                "hit_rate": perf.get_hit_rate(),
                "n_observations": len(perf.predictions),
            }

        return EnsembleWeights(
            weights=weights,
            performance_metrics=metrics,
            n_observations=self.n_observations,
        )

    def ensemble_predict(
        self,
        model_predictions: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        アンサンブル予測を計算

        Args:
            model_predictions: モデル名 -> 予測値
            weights: 使用する重み（Noneで自動計算）

        Returns:
            加重平均予測
        """
        if weights is None:
            weights = self.compute_weights()

        ensemble = 0.0
        for model in self.models:
            if model in model_predictions and model in weights:
                ensemble += weights[model] * model_predictions[model]

        return ensemble

    def reset(self) -> None:
        """パフォーマンス履歴をリセット"""
        self.performance = {
            model: ModelPerformance(model_name=model) for model in self.models
        }
        self.n_observations = 0
        logger.info("Performance history reset")


# =============================================================================
# RegimeAwareEnsemble クラス
# =============================================================================

class RegimeAwareEnsemble:
    """
    レジーム別アンサンブル重み管理

    市場レジームごとに異なる重みを学習・管理する。
    各レジームでのモデルパフォーマンスを個別に追跡し、
    現在のレジームに最適な重みを提供する。

    Usage:
        ensemble = RegimeAwareEnsemble(
            models=["momentum", "reversion", "macro"],
            regimes=["bull", "bear", "high_vol", "low_vol", "range"],
        )

        # レジーム別にパフォーマンス更新
        ensemble.update(
            regime="bull",
            model_predictions={"momentum": 0.02, "reversion": -0.01, "macro": 0.01},
            actual_return=0.015,
        )

        # レジームに応じた重み取得
        weights = ensemble.get_weights_for_regime("bull")
    """

    DEFAULT_REGIMES = ["bull", "bear", "high_vol", "low_vol", "range"]

    def __init__(
        self,
        models: List[str],
        regimes: Optional[List[str]] = None,
        lookback: int = 60,
        decay_factor: float = 0.95,
        min_weight: float = 0.05,
    ) -> None:
        """
        初期化

        Args:
            models: モデル名のリスト
            regimes: レジーム名のリスト
            lookback: パフォーマンス履歴の最大長
            decay_factor: 指数加重の減衰係数
            min_weight: 各モデルの最小重み
        """
        self.models = models
        self.regimes = regimes or self.DEFAULT_REGIMES
        self.lookback = lookback
        self.decay_factor = decay_factor
        self.min_weight = min_weight

        # レジーム別のDynamicEnsembleWeightLearner
        self.regime_learners: Dict[str, DynamicEnsembleWeightLearner] = {
            regime: DynamicEnsembleWeightLearner(
                models=models,
                lookback=lookback,
                decay_factor=decay_factor,
                min_weight=min_weight,
            )
            for regime in self.regimes
        }

        # グローバル（レジーム非依存）のlearner
        self.global_learner = DynamicEnsembleWeightLearner(
            models=models,
            lookback=lookback,
            decay_factor=decay_factor,
            min_weight=min_weight,
        )

        # レジーム別観測カウント
        self.regime_counts: Dict[str, int] = defaultdict(int)

        logger.info(
            f"RegimeAwareEnsemble initialized: "
            f"models={models}, regimes={self.regimes}"
        )

    def update(
        self,
        regime: str,
        model_predictions: Dict[str, float],
        actual_return: float,
        timestamp: Optional[Any] = None,
    ) -> None:
        """
        レジーム別にパフォーマンスを記録

        Args:
            regime: 現在のレジーム
            model_predictions: モデル名 -> 予測値
            actual_return: 実績リターン
            timestamp: タイムスタンプ（オプション）
        """
        # レジーム別の更新
        if regime in self.regime_learners:
            self.regime_learners[regime].update_performance(
                model_predictions=model_predictions,
                actual_return=actual_return,
                timestamp=timestamp,
            )
            self.regime_counts[regime] += 1
        else:
            logger.warning(f"Unknown regime: {regime}, updating global only")

        # グローバルも更新
        self.global_learner.update_performance(
            model_predictions=model_predictions,
            actual_return=actual_return,
            timestamp=timestamp,
        )

    def get_weights_for_regime(
        self,
        regime: str,
        min_observations: int = 10,
    ) -> Dict[str, float]:
        """
        現在のレジームに適した重みを返す

        Args:
            regime: 現在のレジーム
            min_observations: 最小観測数（これ未満ならグローバル重みを使用）

        Returns:
            モデル名 -> 重み
        """
        if regime in self.regime_learners:
            learner = self.regime_learners[regime]
            if learner.n_observations >= min_observations:
                return learner.compute_weights()
            else:
                # 観測数不足の場合はグローバル重みを使用
                logger.debug(
                    f"Insufficient observations for regime '{regime}' "
                    f"({learner.n_observations} < {min_observations}), "
                    f"using global weights"
                )
                return self.global_learner.compute_weights()
        else:
            logger.warning(f"Unknown regime: {regime}, using global weights")
            return self.global_learner.compute_weights()

    def get_weights_result_for_regime(
        self,
        regime: str,
        min_observations: int = 10,
    ) -> EnsembleWeights:
        """レジーム別の重み結果を詳細に取得"""
        weights = self.get_weights_for_regime(regime, min_observations)

        # パフォーマンス指標
        if regime in self.regime_learners:
            learner = self.regime_learners[regime]
        else:
            learner = self.global_learner

        metrics = {}
        for model in self.models:
            perf = learner.performance[model]
            metrics[model] = {
                "mse": perf.get_mse(),
                "ewma_mse": perf.get_ewma_mse(self.decay_factor),
                "mae": perf.get_mae(),
                "hit_rate": perf.get_hit_rate(),
                "n_observations": len(perf.predictions),
            }

        return EnsembleWeights(
            weights=weights,
            performance_metrics=metrics,
            n_observations=learner.n_observations,
            regime=regime,
        )

    def ensemble_predict(
        self,
        regime: str,
        model_predictions: Dict[str, float],
        min_observations: int = 10,
    ) -> float:
        """
        レジームに応じたアンサンブル予測

        Args:
            regime: 現在のレジーム
            model_predictions: モデル名 -> 予測値
            min_observations: 最小観測数

        Returns:
            加重平均予測
        """
        weights = self.get_weights_for_regime(regime, min_observations)

        ensemble = 0.0
        for model in self.models:
            if model in model_predictions and model in weights:
                ensemble += weights[model] * model_predictions[model]

        return ensemble

    def get_all_regime_weights(self) -> Dict[str, Dict[str, float]]:
        """全レジームの重みを取得"""
        return {
            regime: learner.compute_weights()
            for regime, learner in self.regime_learners.items()
        }

    def get_regime_statistics(self) -> Dict[str, Dict[str, Any]]:
        """レジーム別の統計を取得"""
        stats = {}
        for regime in self.regimes:
            learner = self.regime_learners[regime]
            stats[regime] = {
                "n_observations": learner.n_observations,
                "weights": learner.compute_weights(),
                "model_metrics": {
                    model: {
                        "ewma_mse": learner.performance[model].get_ewma_mse(
                            self.decay_factor
                        ),
                        "hit_rate": learner.performance[model].get_hit_rate(),
                    }
                    for model in self.models
                },
            }
        return stats

    def reset(self, regime: Optional[str] = None) -> None:
        """
        パフォーマンス履歴をリセット

        Args:
            regime: リセットするレジーム（Noneで全レジーム）
        """
        if regime is None:
            for learner in self.regime_learners.values():
                learner.reset()
            self.global_learner.reset()
            self.regime_counts.clear()
            logger.info("All regime histories reset")
        elif regime in self.regime_learners:
            self.regime_learners[regime].reset()
            self.regime_counts[regime] = 0
            logger.info(f"Regime '{regime}' history reset")


# =============================================================================
# 便利関数
# =============================================================================

def create_dynamic_ensemble_learner(
    models: List[str],
    lookback: int = 60,
    decay_factor: float = 0.95,
    min_weight: float = 0.05,
) -> DynamicEnsembleWeightLearner:
    """
    DynamicEnsembleWeightLearnerを作成（ファクトリ関数）

    Args:
        models: モデル名のリスト
        lookback: パフォーマンス履歴の最大長
        decay_factor: 指数加重の減衰係数
        min_weight: 最小重み

    Returns:
        DynamicEnsembleWeightLearner
    """
    return DynamicEnsembleWeightLearner(
        models=models,
        lookback=lookback,
        decay_factor=decay_factor,
        min_weight=min_weight,
    )


def create_regime_aware_ensemble(
    models: List[str],
    regimes: Optional[List[str]] = None,
    lookback: int = 60,
    decay_factor: float = 0.95,
    min_weight: float = 0.05,
) -> RegimeAwareEnsemble:
    """
    RegimeAwareEnsembleを作成（ファクトリ関数）

    Args:
        models: モデル名のリスト
        regimes: レジーム名のリスト
        lookback: パフォーマンス履歴の最大長
        decay_factor: 指数加重の減衰係数
        min_weight: 最小重み

    Returns:
        RegimeAwareEnsemble
    """
    return RegimeAwareEnsemble(
        models=models,
        regimes=regimes,
        lookback=lookback,
        decay_factor=decay_factor,
        min_weight=min_weight,
    )
