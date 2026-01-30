"""
Parameter Stability Filter Module - パラメータ安定性フィルタ

パラメータの急激な変更を抑制し、安定した最適化を実現する。
過学習や短期ノイズによる過剰な変動を防ぐ。

主な機能:
- パラメータ変化率の制限
- 指数移動平均によるスムージング
- 安定性スコアの算出
- パフォーマンス改善に基づく受け入れ判定
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilteredParameterResult:
    """フィルタリング結果

    Attributes:
        original_value: 元の値
        filtered_value: フィルタ後の値
        change_rate: 変化率
        was_clamped: 変化率制限が適用されたか
        was_smoothed: スムージングが適用されたか
    """

    original_value: float
    filtered_value: float
    change_rate: float
    was_clamped: bool
    was_smoothed: bool


@dataclass
class StabilityMetrics:
    """安定性メトリクス

    Attributes:
        param_name: パラメータ名
        stability_score: 安定性スコア (0-1)
        coefficient_of_variation: 変動係数
        history_length: 履歴の長さ
        mean: 平均値
        std: 標準偏差
    """

    param_name: str
    stability_score: float
    coefficient_of_variation: float
    history_length: int
    mean: float
    std: float


@dataclass
class AcceptanceDecision:
    """パラメータ受け入れ判定結果

    Attributes:
        accepted: 受け入れるか
        reason: 判定理由
        required_improvement: 必要だった改善率
        actual_improvement: 実際の改善率
        avg_stability_score: 平均安定性スコア
    """

    accepted: bool
    reason: str
    required_improvement: float
    actual_improvement: float
    avg_stability_score: float


class ParameterStabilityFilter:
    """
    パラメータ安定性フィルタ

    パラメータの急激な変更を抑制し、安定した最適化を実現する。
    履歴を保持し、過去のパラメータとの比較・スムージングを行う。

    Usage:
        filter = ParameterStabilityFilter(max_change_rate=0.3)

        # パラメータをフィルタリング
        new_params = {"momentum_window": 25, "volatility_scale": 1.5}
        filtered = filter.filter_parameters(new_params)

        # 安定性スコアを取得
        score = filter.get_stability_score("momentum_window")

        # パラメータ変更を受け入れるか判定
        should_accept = filter.should_accept_new_params(
            new_params, performance_improvement=0.08
        )
    """

    def __init__(
        self,
        max_change_rate: float = 0.3,
        history_length: int = 5,
        smoothing_factor: float = 0.3,
    ):
        """
        初期化

        Parameters
        ----------
        max_change_rate : float
            許容する最大変化率（0.3 = 30%）
        history_length : int
            保持する履歴の長さ
        smoothing_factor : float
            スムージング係数（alpha）。0に近いほど過去重視。
        """
        if not 0 < max_change_rate <= 1:
            raise ValueError("max_change_rate must be in (0, 1]")
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        if not 0 < smoothing_factor <= 1:
            raise ValueError("smoothing_factor must be in (0, 1]")

        self.max_change_rate = max_change_rate
        self.history_length = history_length
        self.smoothing_factor = smoothing_factor

        # パラメータ履歴: param_name -> deque of values
        self._history: Dict[str, deque] = {}

    def filter_parameters(
        self,
        new_params: Dict[str, float],
        param_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        パラメータをフィルタリング

        過去との変化率をチェックし、大きすぎる変化は抑制する。
        その後、スムージングを適用。

        Parameters
        ----------
        new_params : Dict[str, float]
            新しいパラメータ値
        param_names : List[str], optional
            フィルタリング対象のパラメータ名。Noneの場合は全パラメータ。

        Returns
        -------
        Dict[str, float]
            フィルタリング後のパラメータ
        """
        if param_names is None:
            param_names = list(new_params.keys())

        filtered_params = {}

        for name, value in new_params.items():
            if name not in param_names:
                # フィルタリング対象外はそのまま
                filtered_params[name] = value
                continue

            result = self._filter_single_parameter(name, value)
            filtered_params[name] = result.filtered_value

            # 履歴を更新
            self._update_history(name, result.filtered_value)

            if result.was_clamped or result.was_smoothed:
                logger.debug(
                    "Parameter %s filtered: %.4f -> %.4f (clamped=%s, smoothed=%s)",
                    name,
                    result.original_value,
                    result.filtered_value,
                    result.was_clamped,
                    result.was_smoothed,
                )

        return filtered_params

    def filter_parameters_detailed(
        self,
        new_params: Dict[str, float],
        param_names: Optional[List[str]] = None,
    ) -> Dict[str, FilteredParameterResult]:
        """
        パラメータをフィルタリング（詳細結果付き）

        Parameters
        ----------
        new_params : Dict[str, float]
            新しいパラメータ値
        param_names : List[str], optional
            フィルタリング対象のパラメータ名

        Returns
        -------
        Dict[str, FilteredParameterResult]
            各パラメータのフィルタリング結果
        """
        if param_names is None:
            param_names = list(new_params.keys())

        results = {}

        for name, value in new_params.items():
            if name not in param_names:
                results[name] = FilteredParameterResult(
                    original_value=value,
                    filtered_value=value,
                    change_rate=0.0,
                    was_clamped=False,
                    was_smoothed=False,
                )
                continue

            result = self._filter_single_parameter(name, value)
            results[name] = result

            # 履歴を更新
            self._update_history(name, result.filtered_value)

        return results

    def _filter_single_parameter(
        self, name: str, value: float
    ) -> FilteredParameterResult:
        """
        単一パラメータのフィルタリング

        Parameters
        ----------
        name : str
            パラメータ名
        value : float
            新しい値

        Returns
        -------
        FilteredParameterResult
            フィルタリング結果
        """
        # 履歴がない場合は初期値としてそのまま返す
        if name not in self._history or len(self._history[name]) == 0:
            return FilteredParameterResult(
                original_value=value,
                filtered_value=value,
                change_rate=0.0,
                was_clamped=False,
                was_smoothed=False,
            )

        last_value = self._history[name][-1]
        was_clamped = False
        was_smoothed = False

        # ゼロ除算を防ぐ
        if last_value == 0:
            if value == 0:
                change_rate = 0.0
            else:
                change_rate = float("inf")
        else:
            change_rate = (value - last_value) / abs(last_value)

        # 変化率を制限
        clamped_value = value
        if abs(change_rate) > self.max_change_rate:
            direction = 1.0 if change_rate > 0 else -1.0
            clamped_value = last_value * (1 + direction * self.max_change_rate)
            was_clamped = True

        # スムージングを適用: alpha * new + (1-alpha) * last
        smoothed_value = (
            self.smoothing_factor * clamped_value
            + (1 - self.smoothing_factor) * last_value
        )
        was_smoothed = True

        return FilteredParameterResult(
            original_value=value,
            filtered_value=smoothed_value,
            change_rate=change_rate,
            was_clamped=was_clamped,
            was_smoothed=was_smoothed,
        )

    def _update_history(self, name: str, value: float) -> None:
        """
        履歴を更新

        Parameters
        ----------
        name : str
            パラメータ名
        value : float
            新しい値
        """
        if name not in self._history:
            self._history[name] = deque(maxlen=self.history_length)
        self._history[name].append(value)

    def get_stability_score(self, param_name: str) -> float:
        """
        パラメータの安定性スコアを取得

        安定性スコアは変動係数（CV）の逆数に基づく。
        stability = 1 / (1 + CV)
        CV = std / |mean|

        Parameters
        ----------
        param_name : str
            パラメータ名

        Returns
        -------
        float
            安定性スコア（0-1、1が最も安定）
        """
        metrics = self.get_stability_metrics(param_name)
        return metrics.stability_score

    def get_stability_metrics(self, param_name: str) -> StabilityMetrics:
        """
        パラメータの安定性メトリクスを取得

        Parameters
        ----------
        param_name : str
            パラメータ名

        Returns
        -------
        StabilityMetrics
            安定性メトリクス
        """
        if param_name not in self._history or len(self._history[param_name]) < 2:
            return StabilityMetrics(
                param_name=param_name,
                stability_score=1.0,  # 履歴不足は安定とみなす
                coefficient_of_variation=0.0,
                history_length=len(self._history.get(param_name, [])),
                mean=self._history[param_name][-1] if param_name in self._history else 0.0,
                std=0.0,
            )

        values = list(self._history[param_name])
        mean_val = np.mean(values)
        std_val = np.std(values)

        # 変動係数を計算
        if mean_val == 0:
            cv = 0.0 if std_val == 0 else float("inf")
        else:
            cv = std_val / abs(mean_val)

        # 安定性スコア: 1 / (1 + CV)
        stability_score = 1 / (1 + cv) if not np.isinf(cv) else 0.0

        return StabilityMetrics(
            param_name=param_name,
            stability_score=stability_score,
            coefficient_of_variation=cv,
            history_length=len(values),
            mean=float(mean_val),
            std=float(std_val),
        )

    def should_accept_new_params(
        self,
        new_params: Dict[str, float],
        performance_improvement: float,
        min_improvement: float = 0.05,
    ) -> bool:
        """
        新しいパラメータを受け入れるべきか判定

        パフォーマンス改善が閾値未満の場合は拒否。
        安定性が低い場合は、より高い改善率が必要。

        Parameters
        ----------
        new_params : Dict[str, float]
            新しいパラメータ
        performance_improvement : float
            パフォーマンス改善率（0.05 = 5%改善）
        min_improvement : float
            最小必要改善率

        Returns
        -------
        bool
            受け入れるべきか
        """
        decision = self.should_accept_new_params_detailed(
            new_params, performance_improvement, min_improvement
        )
        return decision.accepted

    def should_accept_new_params_detailed(
        self,
        new_params: Dict[str, float],
        performance_improvement: float,
        min_improvement: float = 0.05,
    ) -> AcceptanceDecision:
        """
        新しいパラメータを受け入れるべきか判定（詳細結果付き）

        Parameters
        ----------
        new_params : Dict[str, float]
            新しいパラメータ
        performance_improvement : float
            パフォーマンス改善率
        min_improvement : float
            最小必要改善率

        Returns
        -------
        AcceptanceDecision
            判定結果
        """
        # 既知のパラメータの平均安定性スコアを計算
        known_params = [p for p in new_params.keys() if p in self._history]

        if not known_params:
            # 履歴がない場合は受け入れ
            return AcceptanceDecision(
                accepted=True,
                reason="No history available, accepting initial parameters",
                required_improvement=min_improvement,
                actual_improvement=performance_improvement,
                avg_stability_score=1.0,
            )

        stability_scores = [self.get_stability_score(p) for p in known_params]
        avg_stability = np.mean(stability_scores)

        # 安定性が低いほど、高い改善率が必要
        # required = min_improvement / avg_stability
        # 例: stability=0.5 なら、required = 0.05 / 0.5 = 0.10
        if avg_stability > 0:
            required_improvement = min_improvement / avg_stability
        else:
            required_improvement = float("inf")

        accepted = performance_improvement >= required_improvement

        if accepted:
            reason = (
                f"Performance improvement ({performance_improvement:.2%}) "
                f">= required ({required_improvement:.2%})"
            )
        else:
            reason = (
                f"Performance improvement ({performance_improvement:.2%}) "
                f"< required ({required_improvement:.2%}) "
                f"due to low stability ({avg_stability:.2f})"
            )

        logger.debug(
            "Parameter acceptance decision: %s (improvement=%.2%%, required=%.2%%, stability=%.2f)",
            "ACCEPT" if accepted else "REJECT",
            performance_improvement * 100,
            required_improvement * 100,
            avg_stability,
        )

        return AcceptanceDecision(
            accepted=accepted,
            reason=reason,
            required_improvement=required_improvement,
            actual_improvement=performance_improvement,
            avg_stability_score=avg_stability,
        )

    def get_all_stability_scores(self) -> Dict[str, float]:
        """
        全パラメータの安定性スコアを取得

        Returns
        -------
        Dict[str, float]
            パラメータ名 -> 安定性スコア
        """
        return {name: self.get_stability_score(name) for name in self._history.keys()}

    def reset(self) -> None:
        """履歴をリセット"""
        self._history.clear()

    def reset_parameter(self, param_name: str) -> None:
        """
        特定パラメータの履歴をリセット

        Parameters
        ----------
        param_name : str
            パラメータ名
        """
        if param_name in self._history:
            del self._history[param_name]

    @property
    def tracked_parameters(self) -> List[str]:
        """追跡中のパラメータ名のリスト"""
        return list(self._history.keys())

    def get_history(self, param_name: str) -> List[float]:
        """
        パラメータの履歴を取得

        Parameters
        ----------
        param_name : str
            パラメータ名

        Returns
        -------
        List[float]
            履歴のリスト
        """
        if param_name not in self._history:
            return []
        return list(self._history[param_name])


# ショートカット関数
def filter_parameters(
    new_params: Dict[str, float],
    last_params: Dict[str, float],
    max_change_rate: float = 0.3,
    smoothing_factor: float = 0.3,
) -> Dict[str, float]:
    """
    パラメータをフィルタリングするショートカット関数

    Parameters
    ----------
    new_params : Dict[str, float]
        新しいパラメータ
    last_params : Dict[str, float]
        前回のパラメータ
    max_change_rate : float
        最大変化率
    smoothing_factor : float
        スムージング係数

    Returns
    -------
    Dict[str, float]
        フィルタリング後のパラメータ
    """
    filter_instance = ParameterStabilityFilter(
        max_change_rate=max_change_rate,
        smoothing_factor=smoothing_factor,
        history_length=2,
    )

    # 前回のパラメータを履歴に追加
    for name, value in last_params.items():
        filter_instance._update_history(name, value)

    return filter_instance.filter_parameters(new_params)
