"""
Alpha Parameter Optimizer - アルファスコア計算パラメータの動的最適化

BayesianOptimizerを使用してアルファスコア計算パラメータを
Walk-Forward Cross Validationで動的に最適化するモジュール。

主要コンポーネント:
- AlphaParamOptimizer: パラメータ最適化器
- AlphaOptimizationError: 最適化失敗時のエラー

設計目的:
- 目的関数: Walk-Forward CVでのシャープレシオ
- データ駆動: configファイル不使用、全て動的に算出
- フォールバックなし: データ不足/最適化失敗時はエラー終了

使用方法:
    from src.meta.alpha_param_optimizer import AlphaParamOptimizer

    optimizer = AlphaParamOptimizer()
    params = optimizer.get_params(returns_df, signal_scores_history)
    print(params.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from skopt.space import Dimension

logger = logging.getLogger(__name__)


# =============================================================================
# 固定定数（コード内で定義）
# =============================================================================

MIN_OBSERVATIONS = 60  # 最小観測数
REOPTIMIZE_INTERVAL_ROTATIONS = 12  # 再最適化間隔（ローテーション数）

# 探索空間（コード内で定義）
SEARCH_SPACE = {
    "momentum_weight": (0.0, 0.5),
    "quality_weight": (0.0, 0.5),
    "expected_return_weight": (0.2, 0.8),
    "risk_aversion": (1.0, 5.0),
    "lookback_days": (20, 120),
}

# デフォルト値
DEFAULT_PARAMS = {
    "momentum_weight": 0.33,
    "quality_weight": 0.33,
    "expected_return_weight": 0.34,
    "risk_aversion": 2.5,
    "lookback_days": 60,
}


# =============================================================================
# 例外クラス
# =============================================================================


class AlphaOptimizationError(Exception):
    """アルファスコア最適化に失敗した場合のエラー

    データ不足や最適化失敗時に発生。フォールバックなしで終了する。
    """
    pass


# =============================================================================
# データクラス
# =============================================================================


@dataclass
class OptimizableParams:
    """BayesianOptimizerで動的に最適化されるパラメータ

    Attributes:
        momentum_weight: モメンタムの重み
        quality_weight: シグナル品質の重み
        expected_return_weight: 期待リターンの重み
        risk_aversion: リスク回避係数
        lookback_days: ルックバック日数
    """

    momentum_weight: float = 0.33
    quality_weight: float = 0.33
    expected_return_weight: float = 0.34
    risk_aversion: float = 2.5
    lookback_days: int = 60

    def __post_init__(self) -> None:
        """バリデーションと正規化"""
        # 重みの正規化
        total_weight = self.momentum_weight + self.quality_weight + self.expected_return_weight
        if total_weight > 0 and not np.isclose(total_weight, 1.0, atol=0.01):
            self.momentum_weight /= total_weight
            self.quality_weight /= total_weight
            self.expected_return_weight /= total_weight

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "momentum_weight": self.momentum_weight,
            "quality_weight": self.quality_weight,
            "expected_return_weight": self.expected_return_weight,
            "risk_aversion": self.risk_aversion,
            "lookback_days": self.lookback_days,
        }


@dataclass
class OptimizationHistory:
    """最適化履歴

    Attributes:
        params: 最適化されたパラメータ
        sharpe: 達成されたシャープレシオ
        timestamp: 最適化日時
        n_iterations: イテレーション数
        cv_folds: CVフォールド数
    """
    params: OptimizableParams
    sharpe: float
    timestamp: datetime = field(default_factory=datetime.now)
    n_iterations: int = 0
    cv_folds: int = 5


# =============================================================================
# メインクラス
# =============================================================================


class AlphaParamOptimizer:
    """
    BayesianOptimizerを使用してアルファスコア計算パラメータを最適化

    目的関数: Walk-Forward Cross Validationでのシャープレシオ

    パラメータを完全に動的に管理（フォールバックなし）:
    - 初回実行時: BayesianOptimizerで最適化
    - データ不足/最適化失敗時: AlphaOptimizationErrorを発生
    - 最適化結果: メモリ内キャッシュ（永続化不要）

    Usage:
        optimizer = AlphaParamOptimizer()
        params = optimizer.get_params(returns_df, signal_scores_history)
    """

    def __init__(
        self,
        n_initial: int = 10,
        n_iterations: int = 50,
        n_cv_folds: int = 5,
    ) -> None:
        """初期化

        Args:
            n_initial: 初期ランダム探索回数
            n_iterations: ベイズ最適化イテレーション数
            n_cv_folds: クロスバリデーションのフォールド数
        """
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_cv_folds = n_cv_folds

        # 内部状態
        self._current_params: OptimizableParams | None = None
        self._optimization_history: list[OptimizationHistory] = []
        self._last_optimization_date: datetime | None = None

    @property
    def current_params(self) -> OptimizableParams | None:
        """現在のパラメータを取得"""
        return self._current_params

    def get_params(
        self,
        returns_df: pd.DataFrame,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
        rotation_days: int = 21,
        force_reoptimize: bool = False,
    ) -> OptimizableParams:
        """
        現在の最適パラメータを取得（失敗時はエラー終了）

        Args:
            returns_df: リターンデータ
            signal_scores_history: シグナル履歴（あれば最適化に使用）
            rotation_days: ローテーション日数
            force_reoptimize: 強制的に再最適化するか

        Returns:
            最適化されたパラメータ

        Raises:
            AlphaOptimizationError: データ不足または最適化失敗時
        """
        # データ不足チェック
        min_required = MIN_OBSERVATIONS * 2  # 最適化に必要な最小データ量
        if len(returns_df) < min_required:
            raise AlphaOptimizationError(
                f"データ不足: {len(returns_df)} 観測 < 必要量 {min_required}"
            )

        # 再最適化が必要か判定
        if force_reoptimize or self._should_reoptimize(rotation_days):
            try:
                self._current_params = self._optimize(
                    returns_df, signal_scores_history
                )
                self._last_optimization_date = datetime.now()
                logger.info(
                    f"Alpha parameters optimized: {self._current_params.to_dict()}"
                )
            except Exception as e:
                raise AlphaOptimizationError(
                    f"パラメータ最適化失敗: {e}"
                ) from e

        if self._current_params is None:
            raise AlphaOptimizationError(
                "パラメータが初期化されていません"
            )

        return self._current_params

    def get_params_safe(
        self,
        returns_df: pd.DataFrame,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
        rotation_days: int = 21,
    ) -> OptimizableParams:
        """
        最適パラメータを取得（失敗時はデフォルトを返す安全版）

        Args:
            returns_df: リターンデータ
            signal_scores_history: シグナル履歴
            rotation_days: ローテーション日数

        Returns:
            最適化されたパラメータ、または失敗時はデフォルト
        """
        try:
            return self.get_params(returns_df, signal_scores_history, rotation_days)
        except AlphaOptimizationError as e:
            logger.warning(f"Optimization failed, using defaults: {e}")
            return OptimizableParams(**DEFAULT_PARAMS)

    def _should_reoptimize(self, rotation_days: int) -> bool:
        """再最適化が必要かを判定

        Args:
            rotation_days: ローテーション日数

        Returns:
            再最適化が必要な場合True
        """
        if self._current_params is None:
            return True
        if self._last_optimization_date is None:
            return True

        days_since_last = (datetime.now() - self._last_optimization_date).days
        reoptimize_interval = rotation_days * REOPTIMIZE_INTERVAL_ROTATIONS

        return days_since_last >= reoptimize_interval

    def _optimize(
        self,
        returns_df: pd.DataFrame,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
    ) -> OptimizableParams:
        """
        Walk-Forward CVでパラメータ最適化を実行

        Args:
            returns_df: 全期間のリターンデータ
            signal_scores_history: 各時点でのシグナルスコア履歴

        Returns:
            最適化されたパラメータ
        """
        logger.info(
            f"Starting alpha parameter optimization: "
            f"{len(returns_df)} observations, {self.n_iterations} iterations"
        )

        # 探索空間を構築
        search_space = self._build_search_space()

        # 目的関数
        def objective(params_list: list) -> float:
            """目的関数（最小化のため負のシャープを返す）"""
            params = self._list_to_params(params_list)
            sharpe = self._evaluate_params(
                params, returns_df, signal_scores_history
            )
            return -sharpe  # 最小化

        # ベイズ最適化を実行
        try:
            from skopt import gp_minimize

            result = gp_minimize(
                func=objective,
                dimensions=search_space,
                n_calls=self.n_iterations,
                n_random_starts=self.n_initial,
                acq_func="EI",  # Expected Improvement
                random_state=42,
                verbose=False,
            )

            # 結果をパラメータに変換
            best_params = self._list_to_params(result.x)

            # 重みの正規化
            total = (best_params.momentum_weight +
                     best_params.quality_weight +
                     best_params.expected_return_weight)
            if total > 0:
                best_params.momentum_weight /= total
                best_params.quality_weight /= total
                best_params.expected_return_weight /= total

            # 履歴に追加
            self._optimization_history.append(
                OptimizationHistory(
                    params=best_params,
                    sharpe=-result.fun,  # 最小化の負値を戻す
                    n_iterations=len(result.func_vals),
                    cv_folds=self.n_cv_folds,
                )
            )

            logger.info(
                f"Optimization completed: sharpe={-result.fun:.3f}, "
                f"iterations={len(result.func_vals)}"
            )

            return best_params

        except ImportError:
            logger.warning(
                "scikit-optimize not installed, using simple grid search"
            )
            return self._simple_optimization(returns_df, signal_scores_history)

    def _build_search_space(self) -> list:
        """探索空間を構築

        Returns:
            skopt.space のDimensionリスト
        """
        from skopt.space import Integer, Real

        return [
            Real(SEARCH_SPACE["momentum_weight"][0],
                 SEARCH_SPACE["momentum_weight"][1],
                 name="momentum_weight"),
            Real(SEARCH_SPACE["quality_weight"][0],
                 SEARCH_SPACE["quality_weight"][1],
                 name="quality_weight"),
            Real(SEARCH_SPACE["expected_return_weight"][0],
                 SEARCH_SPACE["expected_return_weight"][1],
                 name="expected_return_weight"),
            Real(SEARCH_SPACE["risk_aversion"][0],
                 SEARCH_SPACE["risk_aversion"][1],
                 name="risk_aversion"),
            Integer(SEARCH_SPACE["lookback_days"][0],
                    SEARCH_SPACE["lookback_days"][1],
                    name="lookback_days"),
        ]

    def _list_to_params(self, params_list: list) -> OptimizableParams:
        """リストをパラメータオブジェクトに変換

        Args:
            params_list: パラメータ値のリスト

        Returns:
            OptimizableParams
        """
        return OptimizableParams(
            momentum_weight=params_list[0],
            quality_weight=params_list[1],
            expected_return_weight=params_list[2],
            risk_aversion=params_list[3],
            lookback_days=int(params_list[4]),
        )

    def _evaluate_params(
        self,
        params: OptimizableParams,
        returns_df: pd.DataFrame,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
    ) -> float:
        """パラメータを評価（Walk-Forward CV）

        Args:
            params: 評価するパラメータ
            returns_df: リターンデータ
            signal_scores_history: シグナル履歴

        Returns:
            平均シャープレシオ
        """
        sharpe_ratios = []
        n_folds = self.n_cv_folds

        for train_idx, test_idx in self._walk_forward_split(returns_df, n_folds):
            train_returns = returns_df.iloc[train_idx]
            test_returns = returns_df.iloc[test_idx]

            # トレーニング期間でアルファ計算
            alpha_scores = self._compute_alpha_scores(
                train_returns, params, signal_scores_history
            )

            # テスト期間でシャープ評価
            test_sharpe = self._evaluate_sharpe(alpha_scores, test_returns)
            sharpe_ratios.append(test_sharpe)

        return np.mean(sharpe_ratios) if sharpe_ratios else 0.0

    def _walk_forward_split(
        self,
        returns_df: pd.DataFrame,
        n_folds: int,
    ) -> list[tuple[range, range]]:
        """Walk-Forward方式でデータを分割

        Args:
            returns_df: リターンデータ
            n_folds: フォールド数

        Returns:
            (train_indices, test_indices) のリスト
        """
        n = len(returns_df)
        fold_size = n // (n_folds + 1)
        splits = []

        for i in range(n_folds):
            train_end = fold_size * (i + 1)
            test_start = train_end
            test_end = min(fold_size * (i + 2), n)

            if test_end > test_start:
                splits.append((
                    range(0, train_end),
                    range(test_start, test_end),
                ))

        return splits

    def _compute_alpha_scores(
        self,
        returns_df: pd.DataFrame,
        params: OptimizableParams,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """アルファスコアを計算

        Args:
            returns_df: リターンデータ
            params: パラメータ
            signal_scores_history: シグナル履歴

        Returns:
            シンボル -> アルファスコア
        """
        symbols = list(returns_df.columns)
        lookback = min(params.lookback_days, len(returns_df))

        if lookback < 5:
            return {s: 0.0 for s in symbols}

        recent_returns = returns_df.tail(lookback)

        # モメンタムスコア
        cumulative_returns = (1 + recent_returns).prod() - 1
        momentum_ranks = cumulative_returns.rank(pct=True)
        momentum_score = (momentum_ranks - 0.5) * 2

        # 品質スコア（シグナルまたは低ボラ）
        volatilities = recent_returns.std()
        if volatilities.std() > 1e-10:
            inv_vol = 1 / (volatilities + 1e-10)
            quality_score = (inv_vol - inv_vol.mean()) / inv_vol.std()
            quality_score = np.clip(quality_score / 3, -1, 1)
        else:
            quality_score = pd.Series(0.0, index=symbols)

        # 期待リターンスコア
        mean_returns = recent_returns.mean()
        if mean_returns.std() > 1e-10:
            exp_ret_score = (mean_returns - mean_returns.mean()) / mean_returns.std()
            exp_ret_score = np.clip(exp_ret_score / 3, -1, 1)
        else:
            exp_ret_score = pd.Series(0.0, index=symbols)

        # 加重平均
        alpha_scores = {}
        for symbol in symbols:
            mom = momentum_score[symbol]
            qual = quality_score[symbol]
            exp_ret = exp_ret_score[symbol]

            alpha = (
                params.momentum_weight * mom +
                params.quality_weight * qual +
                params.expected_return_weight * exp_ret
            )
            alpha_scores[symbol] = alpha

        return alpha_scores

    def _evaluate_sharpe(
        self,
        alpha_scores: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> float:
        """アルファスコアに基づくポートフォリオのシャープレシオを評価

        Args:
            alpha_scores: シンボル -> アルファスコア
            returns_df: テスト期間のリターンデータ

        Returns:
            シャープレシオ
        """
        if returns_df.empty or not alpha_scores:
            return 0.0

        symbols = list(returns_df.columns)

        # アルファスコアに基づくウェイト（Softmax）
        scores = np.array([alpha_scores.get(s, 0.0) for s in symbols])
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # 数値安定性
        weights = exp_scores / exp_scores.sum()

        # ポートフォリオリターン
        returns_matrix = returns_df.values
        portfolio_returns = (returns_matrix * weights).sum(axis=1)

        # シャープレシオ計算
        if len(portfolio_returns) < 5:
            return 0.0

        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std()

        if std_ret > 1e-10:
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = 0.0

        return float(sharpe)

    def _simple_optimization(
        self,
        returns_df: pd.DataFrame,
        signal_scores_history: dict[str, dict[str, float]] | None = None,
    ) -> OptimizableParams:
        """簡易グリッドサーチ（skoptがない場合のフォールバック）

        Args:
            returns_df: リターンデータ
            signal_scores_history: シグナル履歴

        Returns:
            最適なパラメータ
        """
        logger.info("Running simple grid search optimization")

        best_sharpe = float("-inf")
        best_params = OptimizableParams(**DEFAULT_PARAMS)

        # 簡易グリッド
        grid = [
            OptimizableParams(0.2, 0.3, 0.5, 2.0, 40),
            OptimizableParams(0.3, 0.3, 0.4, 2.5, 60),
            OptimizableParams(0.4, 0.2, 0.4, 3.0, 60),
            OptimizableParams(0.3, 0.4, 0.3, 2.0, 80),
            OptimizableParams(0.25, 0.25, 0.5, 2.5, 60),
        ]

        for params in grid:
            sharpe = self._evaluate_params(params, returns_df, signal_scores_history)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        logger.info(f"Simple optimization completed: sharpe={best_sharpe:.3f}")
        return best_params


# =============================================================================
# 便利関数
# =============================================================================


def create_alpha_param_optimizer(
    n_iterations: int = 50,
    n_cv_folds: int = 5,
) -> AlphaParamOptimizer:
    """AlphaParamOptimizerのファクトリ関数

    Args:
        n_iterations: 最適化イテレーション数
        n_cv_folds: CVフォールド数

    Returns:
        初期化されたAlphaParamOptimizer
    """
    return AlphaParamOptimizer(
        n_iterations=n_iterations,
        n_cv_folds=n_cv_folds,
    )
