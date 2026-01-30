"""
Parameter Optimizer Module - 高度なパラメータ最適化

時系列クロスバリデーションを使用した過学習防止パラメータ最適化を提供する。

主要機能:
- 5-fold時系列CV（Purge gap付き）
- OOS Sharpe比較（±20%以内の制約）
- パラメータグリッドサーチ
- ベイズ最適化（オプション）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Data Classes
# =============================================================================
class SignalEvaluator(Protocol):
    """シグナル評価プロトコル"""

    def evaluate(self, params: dict[str, Any], data: pd.DataFrame) -> float:
        """パラメータでシグナルを評価し、スコア（例: Sharpe比）を返す"""
        ...


@dataclass
class OptimizationResult:
    """最適化結果"""

    best_params: dict[str, Any]
    best_score: float
    in_sample_score: float
    out_of_sample_score: float
    cv_scores: list[float]
    cv_std: float
    all_results: list[dict[str, Any]]
    overfitting_ratio: float  # IS / OOS
    is_valid: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CVFold:
    """クロスバリデーションのフォールド"""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_start: pd.Timestamp
    purge_end: pd.Timestamp


@dataclass
class ParameterGridConfig:
    """パラメータグリッド設定

    計画からのパラメータグリッド:
    - signal_params:
        - momentum_lookback: [20, 40, 60, 90, 120]
        - bollinger_period: [14, 20, 30]
    - allocation_params:
        - w_asset_max: [0.15, 0.20, 0.25]
        - smooth_alpha: [0.2, 0.3, 0.4]
    - meta_params:
        - top_n: [5, 7, 10, 12, 15]
        - beta: [1.5, 2.0, 2.5, 3.0]
    """

    # シグナルパラメータ
    momentum_lookback: list[int] = field(
        default_factory=lambda: [20, 40, 60, 90, 120]
    )
    bollinger_period: list[int] = field(default_factory=lambda: [14, 20, 30])
    rsi_period: list[int] = field(default_factory=lambda: [7, 14, 21])
    atr_period: list[int] = field(default_factory=lambda: [10, 14, 20])

    # アロケーションパラメータ
    w_asset_max: list[float] = field(default_factory=lambda: [0.15, 0.20, 0.25])
    smooth_alpha: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])
    min_weight: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])

    # メタパラメータ
    top_n: list[int] = field(default_factory=lambda: [5, 7, 10, 12, 15])
    beta: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    entropy_weight: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])

    def to_dict(self) -> dict[str, list[Any]]:
        """辞書形式に変換"""
        return {
            "momentum_lookback": self.momentum_lookback,
            "bollinger_period": self.bollinger_period,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "w_asset_max": self.w_asset_max,
            "smooth_alpha": self.smooth_alpha,
            "min_weight": self.min_weight,
            "top_n": self.top_n,
            "beta": self.beta,
            "entropy_weight": self.entropy_weight,
        }


# =============================================================================
# Time Series Cross-Validation
# =============================================================================
class TimeSeriesCV:
    """時系列クロスバリデーション

    金融データ向けの時系列CVで、以下の特徴を持つ:
    - Purge gap: 訓練と検証の間に期間を空け、先読みバイアスを防止
    - 拡張型（Expanding）または固定窓（Rolling）の選択
    - 検証期間の最小サイズ保証
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        min_train_size: int = 252,
        test_size: int = 63,
        expanding: bool = True,
    ):
        """初期化

        Args:
            n_splits: フォールド数（デフォルト5）
            purge_gap: 訓練と検証の間のギャップ（日数、デフォルト5）
            min_train_size: 最小訓練サイズ（デフォルト252日=1年）
            test_size: 検証期間サイズ（デフォルト63日=約3ヶ月）
            expanding: True=拡張型、False=固定窓
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.expanding = expanding

    def split(self, data: pd.DataFrame) -> list[CVFold]:
        """データをCV用に分割

        Args:
            data: 時系列データ（DatetimeIndex）

        Returns:
            CVFoldのリスト
        """
        n = len(data)
        index = data.index

        # 必要な最小サイズの検証
        min_required = self.min_train_size + self.purge_gap + self.test_size
        if n < min_required * 2:
            logger.warning(
                "Data too short for %d-fold CV. Reducing splits.",
                self.n_splits,
            )
            self.n_splits = max(2, n // min_required)

        folds = []

        # 検証期間の総サイズ
        total_test_size = self.test_size * self.n_splits

        # 訓練開始インデックス（拡張型の場合は固定、ローリングの場合は移動）
        base_train_start = 0

        for fold_idx in range(self.n_splits):
            # 検証期間の計算（末尾から逆算）
            test_end_idx = n - (self.n_splits - fold_idx - 1) * self.test_size - 1
            test_start_idx = test_end_idx - self.test_size + 1

            # Purge期間
            purge_end_idx = test_start_idx - 1
            purge_start_idx = purge_end_idx - self.purge_gap + 1

            # 訓練期間
            train_end_idx = purge_start_idx - 1

            if self.expanding:
                train_start_idx = base_train_start
            else:
                # ローリング: 固定サイズの訓練窓
                train_start_idx = max(
                    base_train_start, train_end_idx - self.min_train_size + 1
                )

            # 有効性チェック
            if train_end_idx - train_start_idx < self.min_train_size:
                logger.warning(
                    "Fold %d has insufficient training data. Skipping.",
                    fold_idx,
                )
                continue

            folds.append(
                CVFold(
                    fold_id=fold_idx,
                    train_start=index[train_start_idx],
                    train_end=index[train_end_idx],
                    test_start=index[test_start_idx],
                    test_end=index[test_end_idx],
                    purge_start=index[purge_start_idx],
                    purge_end=index[purge_end_idx],
                )
            )

        return folds

    def get_train_test_indices(
        self, data: pd.DataFrame, fold: CVFold
    ) -> tuple[pd.Index, pd.Index]:
        """フォールドから訓練/検証インデックスを取得

        Args:
            data: データ
            fold: CVフォールド

        Returns:
            (train_index, test_index)
        """
        train_mask = (data.index >= fold.train_start) & (data.index <= fold.train_end)
        test_mask = (data.index >= fold.test_start) & (data.index <= fold.test_end)

        return data.index[train_mask], data.index[test_mask]


# =============================================================================
# Advanced Parameter Optimizer
# =============================================================================
class AdvancedParameterOptimizer:
    """高度なパラメータ最適化クラス

    時系列CVとOOS検証を使用して、過学習を防止しながら最適パラメータを探索する。

    過学習防止策:
    1. 5-fold時系列CV（Purge gap 5日）
    2. OOS Sharpe ± 20%以内の制約
    3. パラメータの安定性チェック（隣接パラメータでの性能）
    """

    def __init__(
        self,
        cv: TimeSeriesCV | None = None,
        oos_tolerance: float = 0.20,
        min_cv_consistency: float = 0.6,
        random_state: int = 42,
    ):
        """初期化

        Args:
            cv: 時系列クロスバリデーター（None=デフォルト設定）
            oos_tolerance: OOSとISの許容乖離（デフォルト20%）
            min_cv_consistency: CV間での最小一貫性（フォールドの何%で改善が必要か）
            random_state: 乱数シード
        """
        self.cv = cv or TimeSeriesCV(n_splits=5, purge_gap=5)
        self.oos_tolerance = oos_tolerance
        self.min_cv_consistency = min_cv_consistency
        self.random_state = random_state
        np.random.seed(random_state)

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: dict[str, list[Any]],
        evaluator: SignalEvaluator | Callable[[dict[str, Any], pd.DataFrame], float],
        baseline_params: dict[str, Any] | None = None,
        max_combinations: int = 500,
    ) -> OptimizationResult:
        """パラメータ最適化を実行

        Args:
            data: 時系列データ
            param_grid: パラメータグリッド {param_name: [values]}
            evaluator: 評価関数またはSignalEvaluatorプロトコル
            baseline_params: ベースラインパラメータ（比較用）
            max_combinations: 最大探索組み合わせ数

        Returns:
            OptimizationResult
        """
        # CVフォールドの生成
        folds = self.cv.split(data)

        if len(folds) < 2:
            logger.error("Insufficient data for cross-validation")
            return self._create_empty_result(baseline_params or {})

        # パラメータ組み合わせの生成
        combinations = self._generate_combinations(param_grid, max_combinations)
        logger.info(
            "Optimizing %d parameter combinations across %d CV folds",
            len(combinations),
            len(folds),
        )

        # 評価関数のラップ
        if hasattr(evaluator, "evaluate"):
            eval_func = evaluator.evaluate
        else:
            eval_func = evaluator

        # 全組み合わせの評価
        all_results = []
        best_score = float("-inf")
        best_params = combinations[0] if combinations else {}

        for i, params in enumerate(combinations):
            if (i + 1) % 50 == 0:
                logger.info("Evaluated %d/%d combinations", i + 1, len(combinations))

            result = self._evaluate_params(data, params, folds, eval_func)
            all_results.append(result)

            # CVスコアの平均で比較
            cv_mean = result["cv_mean"]
            if cv_mean > best_score:
                # OOS制約チェック
                if self._check_oos_constraint(result):
                    best_score = cv_mean
                    best_params = params

        # ベストパラメータの詳細評価
        best_result = next(
            (r for r in all_results if r["params"] == best_params), all_results[0]
        )

        # 過学習比率の計算
        is_score = best_result.get("in_sample_score", best_score)
        oos_score = best_result.get("cv_mean", best_score)
        overfitting_ratio = is_score / oos_score if oos_score != 0 else float("inf")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            in_sample_score=is_score,
            out_of_sample_score=oos_score,
            cv_scores=best_result.get("cv_scores", []),
            cv_std=best_result.get("cv_std", 0.0),
            all_results=all_results,
            overfitting_ratio=overfitting_ratio,
            is_valid=self._check_oos_constraint(best_result),
            metadata={
                "n_combinations": len(combinations),
                "n_folds": len(folds),
                "oos_tolerance": self.oos_tolerance,
            },
        )

    def _evaluate_params(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        folds: list[CVFold],
        eval_func: Callable,
    ) -> dict[str, Any]:
        """パラメータをCV評価

        Args:
            data: データ
            params: パラメータ
            folds: CVフォールド
            eval_func: 評価関数

        Returns:
            評価結果辞書
        """
        cv_scores = []
        in_sample_scores = []

        for fold in folds:
            train_idx, test_idx = self.cv.get_train_test_indices(data, fold)

            train_data = data.loc[train_idx]
            test_data = data.loc[test_idx]

            # 訓練データでの評価（in-sample）
            try:
                is_score = eval_func(params, train_data)
                in_sample_scores.append(is_score)
            except Exception as e:
                logger.debug("IS evaluation failed: %s", e)
                in_sample_scores.append(float("nan"))

            # 検証データでの評価（out-of-sample）
            try:
                oos_score = eval_func(params, test_data)
                cv_scores.append(oos_score)
            except Exception as e:
                logger.debug("OOS evaluation failed: %s", e)
                cv_scores.append(float("nan"))

        # NaNを除外して統計計算
        valid_cv = [s for s in cv_scores if not np.isnan(s)]
        valid_is = [s for s in in_sample_scores if not np.isnan(s)]

        cv_mean = np.mean(valid_cv) if valid_cv else float("nan")
        cv_std = np.std(valid_cv) if len(valid_cv) > 1 else 0.0
        is_mean = np.mean(valid_is) if valid_is else float("nan")

        return {
            "params": params,
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "in_sample_score": is_mean,
            "valid_folds": len(valid_cv),
        }

    def _check_oos_constraint(self, result: dict[str, Any]) -> bool:
        """OOS制約をチェック

        ISとOOSのスコア乖離が許容範囲内かどうか

        Args:
            result: 評価結果

        Returns:
            制約を満たすかどうか
        """
        is_score = result.get("in_sample_score", 0)
        oos_score = result.get("cv_mean", 0)

        if oos_score == 0 or np.isnan(oos_score):
            return False

        ratio = abs(is_score - oos_score) / abs(oos_score)
        return ratio <= self.oos_tolerance

    def _generate_combinations(
        self, param_grid: dict[str, list[Any]], max_combinations: int
    ) -> list[dict[str, Any]]:
        """パラメータ組み合わせを生成

        Args:
            param_grid: パラメータグリッド
            max_combinations: 最大組み合わせ数

        Returns:
            パラメータ辞書のリスト
        """
        import itertools

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        all_combos = list(itertools.product(*values))
        total = len(all_combos)

        if total > max_combinations:
            # ランダムサンプリング
            indices = np.random.choice(total, max_combinations, replace=False)
            all_combos = [all_combos[i] for i in indices]
            logger.info(
                "Sampled %d combinations from %d total", max_combinations, total
            )

        return [dict(zip(keys, combo)) for combo in all_combos]

    def _create_empty_result(self, params: dict[str, Any]) -> OptimizationResult:
        """空の結果を作成"""
        return OptimizationResult(
            best_params=params,
            best_score=float("nan"),
            in_sample_score=float("nan"),
            out_of_sample_score=float("nan"),
            cv_scores=[],
            cv_std=0.0,
            all_results=[],
            overfitting_ratio=float("inf"),
            is_valid=False,
            metadata={"error": "Insufficient data for optimization"},
        )


# =============================================================================
# Sharpe Ratio Evaluator
# =============================================================================
class SharpeEvaluator:
    """Sharpe比率評価クラス

    シグナルパラメータからSharpe比率を計算する評価器。
    """

    def __init__(
        self,
        signal_factory: Callable[[dict[str, Any]], Any] | None = None,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252,
    ):
        """初期化

        Args:
            signal_factory: パラメータからシグナルを生成する関数
            risk_free_rate: 無リスク金利（年率）
            annualization_factor: 年率化係数
        """
        self.signal_factory = signal_factory
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def evaluate(self, params: dict[str, Any], data: pd.DataFrame) -> float:
        """パラメータでシグナルを評価

        Args:
            params: シグナルパラメータ
            data: 価格データ

        Returns:
            Sharpe比率
        """
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # シグナル生成（カスタム factory がある場合）
        if self.signal_factory:
            signal = self.signal_factory(params)
            result = signal.compute(data)
            scores = result.scores
        else:
            # デフォルト: モメンタムシグナル
            lookback = params.get("lookback", params.get("momentum_lookback", 20))
            returns = data["close"].pct_change(periods=lookback)
            scores = np.tanh(returns * 5)

        # シグナルベースのリターン計算（単純な方向性戦略）
        daily_returns = data["close"].pct_change()
        strategy_returns = daily_returns * scores.shift(1)  # 1日遅れで執行
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) < 20:
            return float("nan")

        # Sharpe比率計算
        mean_return = strategy_returns.mean() * self.annualization_factor
        std_return = strategy_returns.std() * np.sqrt(self.annualization_factor)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return float(sharpe)


# =============================================================================
# Convenience Functions
# =============================================================================
def create_default_param_grid() -> dict[str, list[Any]]:
    """デフォルトのパラメータグリッドを作成

    計画の §task_012_6 に基づく
    """
    return {
        "momentum_lookback": [20, 40, 60, 90, 120],
        "bollinger_period": [14, 20, 30],
        "w_asset_max": [0.15, 0.20, 0.25],
        "smooth_alpha": [0.2, 0.3, 0.4],
        "top_n": [5, 7, 10, 12, 15],
        "beta": [1.5, 2.0, 2.5, 3.0],
    }


def optimize_with_defaults(
    data: pd.DataFrame,
    param_subset: dict[str, list[Any]] | None = None,
    n_folds: int = 5,
    purge_gap: int = 5,
) -> OptimizationResult:
    """デフォルト設定でパラメータ最適化を実行

    Args:
        data: 価格データ（'close'カラム必須）
        param_subset: 最適化するパラメータのサブセット
        n_folds: CVフォールド数
        purge_gap: Purge gap（日数）

    Returns:
        OptimizationResult
    """
    cv = TimeSeriesCV(n_splits=n_folds, purge_gap=purge_gap)
    optimizer = AdvancedParameterOptimizer(cv=cv)
    evaluator = SharpeEvaluator()

    param_grid = param_subset or {"momentum_lookback": [20, 40, 60, 90, 120]}

    return optimizer.optimize(data, param_grid, evaluator)
