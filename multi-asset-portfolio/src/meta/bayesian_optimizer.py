"""
Bayesian Optimizer - ハイパーパラメータ最適化モジュール

scikit-optimize (skopt) を用いたベイズ最適化で、
バックテストのシャープレシオを最大化するパラメータを探索する。

設計特徴:
1. ガウス過程 + Expected Improvement (EI) 獲得関数
2. 過学習ペナルティ付き目的関数
3. 時系列CVによる堅牢なシャープ推定
4. 早期停止による効率化

依存パッケージ: scikit-optimize (skopt)

使用例:
    from src.meta.bayesian_optimizer import BayesianOptimizer, OptimizerConfig

    optimizer = BayesianOptimizer(config=OptimizerConfig(n_calls=50))
    result = optimizer.optimize(
        train_data=data,
        universe=["SPY", "QQQ", "TLT"],
    )
    print(f"Best Sharpe: {result.best_score:.3f}")
    print(f"Best Params: {result.best_params}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from skopt import gp_minimize
    from skopt.space import Dimension

logger = logging.getLogger(__name__)


# =============================================================================
# 探索空間定義
# =============================================================================

def get_default_search_space() -> list[Any]:
    """デフォルトの探索空間を取得

    Returns:
        skopt.space のDimensionリスト
    """
    from skopt.space import Categorical, Integer, Real

    return [
        # シグナルパラメータ
        Integer(10, 120, name="momentum_lookback"),
        Integer(5, 30, name="rsi_period"),
        Integer(10, 60, name="bollinger_period"),

        # 戦略重み
        Real(0.1, 0.6, name="trend_weight"),
        Real(0.1, 0.5, name="reversion_weight"),
        Real(0.1, 0.4, name="macro_weight"),

        # Meta層パラメータ
        Real(1.0, 4.0, name="beta"),
        Integer(3, 15, name="top_n"),
        Real(0.1, 0.3, name="w_asset_max"),
    ]


def get_signal_search_space() -> list[Any]:
    """シグナルパラメータのみの探索空間

    Returns:
        skopt.space のDimensionリスト（シグナル系のみ）
    """
    from skopt.space import Integer, Real

    return [
        Integer(5, 120, name="momentum_lookback"),
        Integer(5, 50, name="rsi_period"),
        Integer(5, 60, name="bollinger_period"),
        Real(1.0, 5.0, name="atr_multiplier"),
        Integer(5, 30, name="macd_fast"),
        Integer(12, 50, name="macd_slow"),
    ]


def get_allocation_search_space() -> list[Any]:
    """アロケーションパラメータのみの探索空間

    Returns:
        skopt.space のDimensionリスト（アロケーション系のみ）
    """
    from skopt.space import Integer, Real

    return [
        Real(1.0, 5.0, name="beta"),
        Integer(3, 20, name="top_n"),
        Real(0.05, 0.4, name="w_asset_max"),
        Real(0.5, 2.0, name="entropy_min"),
        Real(0.0, 0.5, name="cash_weight_min"),
    ]


# =============================================================================
# 設定とデータクラス
# =============================================================================

@dataclass
class OptimizerConfig:
    """ベイズ最適化設定

    Attributes:
        n_calls: 最適化試行回数
        n_random_starts: 初期ランダム探索回数
        acq_func: 獲得関数 (EI, LCB, PI, gp_hedge)
        cv_folds: 時系列CVのfold数
        overfitting_penalty: 過学習ペナルティ係数
        early_stop_patience: 早期停止の待機回数
        random_state: 乱数シード
        verbose: 詳細ログ出力
        n_jobs: 並列ジョブ数 (-1 で全コア)
    """

    n_calls: int = 100
    n_random_starts: int = 20
    acq_func: str = "EI"
    cv_folds: int = 5
    overfitting_penalty: float = 0.1
    early_stop_patience: int = 15
    random_state: int = 42
    verbose: bool = True
    n_jobs: int = 1


@dataclass
class OptimizationResult:
    """最適化結果

    Attributes:
        best_params: 最適パラメータ辞書
        best_score: 最良スコア（シャープレシオ）
        convergence_curve: 収束曲線（各イテレーションのベストスコア）
        param_importance: パラメータ重要度（相対寄与）
        all_results: 全試行の結果リスト
        cv_scores: クロスバリデーションスコア
        raw_result: skoptの生の結果オブジェクト
        elapsed_time: 実行時間（秒）
        n_iterations: イテレーション数
        metadata: 追加メタデータ
    """

    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    convergence_curve: list[float] = field(default_factory=list)
    param_importance: dict[str, float] = field(default_factory=dict)
    all_results: list[dict[str, Any]] = field(default_factory=list)
    cv_scores: list[float] = field(default_factory=list)
    raw_result: Any = None
    elapsed_time: float = 0.0
    n_iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "convergence_curve": self.convergence_curve,
            "param_importance": self.param_importance,
            "elapsed_time": self.elapsed_time,
            "n_iterations": self.n_iterations,
            "cv_scores": self.cv_scores,
            "metadata": self.metadata,
        }


@dataclass
class CrossValidationResult:
    """時系列CV結果

    Attributes:
        mean_sharpe: 平均シャープレシオ
        std_sharpe: シャープレシオの標準偏差
        fold_sharpes: 各foldのシャープレシオ
        overfitting_score: 過学習スコア（train/test差分）
    """

    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    fold_sharpes: list[float] = field(default_factory=list)
    overfitting_score: float = 0.0


# =============================================================================
# ベイズ最適化クラス
# =============================================================================

class BayesianOptimizer:
    """ベイズ最適化によるハイパーパラメータ探索

    scikit-optimize (skopt) の gp_minimize をラップし、
    バックテストのシャープレシオを最大化するパラメータを探索する。

    特徴:
    - ガウス過程回帰によるサロゲートモデル
    - Expected Improvement (EI) 獲得関数
    - 時系列CVによる堅牢なスコア推定
    - 過学習ペナルティ

    Usage:
        optimizer = BayesianOptimizer(
            config=OptimizerConfig(n_calls=50)
        )
        result = optimizer.optimize(
            train_data=data,
            universe=["SPY", "QQQ", "TLT"],
        )
        print(f"Best: {result.best_params}")
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        search_space: list[Any] | None = None,
    ) -> None:
        """初期化

        Args:
            config: 最適化設定
            search_space: 探索空間（Noneでデフォルト使用）
        """
        self.config = config or OptimizerConfig()
        self._search_space = search_space or get_default_search_space()
        self._param_names = [dim.name for dim in self._search_space]

        # 内部状態
        self._best_score = float("-inf")
        self._no_improvement_count = 0
        self._all_results: list[dict[str, Any]] = []
        self._convergence: list[float] = []

        # データとバックテスト設定
        self._train_data: pd.DataFrame | None = None
        self._universe: list[str] = []
        self._backtest_config: dict[str, Any] = {}

    @property
    def search_space(self) -> list[Any]:
        """探索空間を取得"""
        return self._search_space

    @property
    def param_names(self) -> list[str]:
        """パラメータ名リストを取得"""
        return self._param_names

    def optimize(
        self,
        train_data: pd.DataFrame,
        universe: list[str],
        backtest_config: dict[str, Any] | None = None,
        custom_objective: Callable[[list[Any]], float] | None = None,
    ) -> OptimizationResult:
        """最適化を実行

        Args:
            train_data: 学習データ（OHLCVデータフレーム）
            universe: 銘柄リスト
            backtest_config: バックテスト設定の上書き
            custom_objective: カスタム目的関数（省略時は内蔵）

        Returns:
            最適化結果
        """
        from skopt import gp_minimize
        from skopt.callbacks import DeltaYStopper
        import time

        self._train_data = train_data
        self._universe = universe
        self._backtest_config = backtest_config or {}

        # 状態リセット
        self._best_score = float("-inf")
        self._no_improvement_count = 0
        self._all_results = []
        self._convergence = []

        # 目的関数
        objective = custom_objective or self._objective

        logger.info(
            "Starting Bayesian optimization: n_calls=%d, n_random=%d, acq=%s",
            self.config.n_calls,
            self.config.n_random_starts,
            self.config.acq_func,
        )

        start_time = time.time()

        # 早期停止コールバック
        callbacks = []
        if self.config.early_stop_patience > 0:
            callbacks.append(
                DeltaYStopper(delta=0.001, n_best=self.config.early_stop_patience)
            )

        # 最適化実行
        result = gp_minimize(
            func=objective,
            dimensions=self._search_space,
            n_calls=self.config.n_calls,
            n_random_starts=self.config.n_random_starts,
            acq_func=self.config.acq_func,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            n_jobs=self.config.n_jobs,
            callback=callbacks if callbacks else None,
        )

        elapsed = time.time() - start_time

        # 結果を構築
        best_params = dict(zip(self._param_names, result.x))
        best_score = -result.fun  # 最小化を最大化に変換

        # 収束曲線（累積最小値を最大値に変換）
        convergence = [-min(result.func_vals[:i + 1]) for i in range(len(result.func_vals))]

        # パラメータ重要度を計算
        param_importance = self._compute_param_importance(result)

        opt_result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            convergence_curve=convergence,
            param_importance=param_importance,
            all_results=self._all_results,
            cv_scores=[r.get("cv_mean", 0.0) for r in self._all_results],
            raw_result=result,
            elapsed_time=elapsed,
            n_iterations=len(result.func_vals),
            metadata={
                "acq_func": self.config.acq_func,
                "n_random_starts": self.config.n_random_starts,
                "early_stopped": len(result.func_vals) < self.config.n_calls,
            },
        )

        logger.info(
            "Optimization completed: best_sharpe=%.3f, iterations=%d, time=%.1fs",
            best_score,
            opt_result.n_iterations,
            elapsed,
        )

        return opt_result

    def _objective(self, params: list[Any]) -> float:
        """目的関数（最小化）

        Args:
            params: パラメータ値のリスト

        Returns:
            負のスコア（最小化のため）
        """
        param_dict = dict(zip(self._param_names, params))

        try:
            # パラメータを設定に反映
            self._set_params(param_dict)

            # 時系列CVでシャープレシオを計算
            cv_result = self._cross_validate(param_dict)

            # 過学習ペナルティ
            penalty = self.config.overfitting_penalty * cv_result.overfitting_score

            # 最終スコア
            score = cv_result.mean_sharpe - penalty

            # 結果を記録
            self._all_results.append({
                "params": param_dict.copy(),
                "score": score,
                "cv_mean": cv_result.mean_sharpe,
                "cv_std": cv_result.std_sharpe,
                "penalty": penalty,
                "overfitting": cv_result.overfitting_score,
            })

            # 収束追跡
            if score > self._best_score:
                self._best_score = score
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            self._convergence.append(self._best_score)

            if self.config.verbose:
                logger.debug(
                    "Trial: sharpe=%.3f, penalty=%.3f, score=%.3f",
                    cv_result.mean_sharpe,
                    penalty,
                    score,
                )

            return -score  # 最小化

        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return 0.0  # ペナルティとして0を返す

    def _set_params(self, params: dict[str, Any]) -> None:
        """パラメータをシステムに反映

        Args:
            params: パラメータ辞書
        """
        # 注: 実際の実装では Settings やパイプラインに反映
        # ここでは内部状態として保持
        self._current_params = params

    def _cross_validate(self, params: dict[str, Any]) -> CrossValidationResult:
        """時系列クロスバリデーション

        Walk-forward方式で時系列分割し、各foldでバックテストを実行。

        Args:
            params: テストするパラメータ

        Returns:
            CV結果
        """
        if self._train_data is None or self._train_data.empty:
            return CrossValidationResult()

        n_folds = self.config.cv_folds
        data_len = len(self._train_data)
        fold_size = data_len // (n_folds + 1)

        fold_sharpes = []
        train_sharpes = []

        for fold_idx in range(n_folds):
            # 時系列分割: 前半をtrain、後半をtest
            train_end = fold_size * (fold_idx + 1)
            test_end = fold_size * (fold_idx + 2)

            if test_end > data_len:
                break

            train_fold = self._train_data.iloc[:train_end]
            test_fold = self._train_data.iloc[train_end:test_end]

            # 各foldでバックテスト
            train_sharpe = self._run_backtest(train_fold, params)
            test_sharpe = self._run_backtest(test_fold, params)

            train_sharpes.append(train_sharpe)
            fold_sharpes.append(test_sharpe)

        if not fold_sharpes:
            return CrossValidationResult()

        mean_sharpe = float(np.mean(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))

        # 過学習スコア: train vs test の乖離
        if train_sharpes:
            mean_train = float(np.mean(train_sharpes))
            overfitting = max(0, mean_train - mean_sharpe)
        else:
            overfitting = 0.0

        return CrossValidationResult(
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            fold_sharpes=fold_sharpes,
            overfitting_score=overfitting,
        )

    def _run_backtest(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
    ) -> float:
        """単一期間のバックテストを実行

        Args:
            data: バックテストデータ
            params: パラメータ

        Returns:
            シャープレシオ
        """
        try:
            # 簡易バックテスト: リターンベースのシャープ計算
            if "close" not in data.columns:
                return 0.0

            returns = data["close"].pct_change().dropna()
            if len(returns) < 20:
                return 0.0

            # パラメータに基づくシグナル生成（簡易版）
            lookback = params.get("momentum_lookback", 20)
            momentum = data["close"].pct_change(periods=min(lookback, len(data) - 1))

            # モメンタムベースの仮想リターン
            signal = np.sign(momentum.shift(1))
            strategy_returns = returns * signal.fillna(0)

            # シャープレシオ計算
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()

            if std_ret > 0:
                sharpe = mean_ret / std_ret * np.sqrt(252)
            else:
                sharpe = 0.0

            return float(sharpe)

        except Exception as e:
            logger.debug(f"Backtest failed: {e}")
            return 0.0

    def _compute_param_importance(self, result: Any) -> dict[str, float]:
        """パラメータ重要度を計算

        Args:
            result: skopt の最適化結果

        Returns:
            パラメータ名 -> 重要度（0-1）の辞書
        """
        try:
            from skopt.plots import partial_dependence

            # 簡易的な重要度: パラメータごとの偏微分の分散
            importance = {}
            total_var = 0.0

            for i, name in enumerate(self._param_names):
                # 偏依存性から分散を推定
                try:
                    xi, yi = partial_dependence(
                        result.space,
                        result.models[-1],
                        i,
                        n_points=20,
                    )
                    var = float(np.var(yi))
                    importance[name] = var
                    total_var += var
                except Exception:
                    importance[name] = 0.0

            # 正規化
            if total_var > 0:
                importance = {k: v / total_var for k, v in importance.items()}

            return importance

        except Exception as e:
            logger.debug(f"Importance calculation failed: {e}")
            return {name: 1.0 / len(self._param_names) for name in self._param_names}

    def plot_convergence(self, result: OptimizationResult) -> None:
        """収束曲線をプロット

        Args:
            result: 最適化結果
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(result.convergence_curve, "b-", linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Best Sharpe Ratio")
            plt.title("Bayesian Optimization Convergence")
            plt.grid(True, alpha=0.3)
            plt.axhline(y=result.best_score, color="r", linestyle="--", label=f"Best: {result.best_score:.3f}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")

    def plot_param_importance(self, result: OptimizationResult) -> None:
        """パラメータ重要度をプロット

        Args:
            result: 最適化結果
        """
        try:
            import matplotlib.pyplot as plt

            names = list(result.param_importance.keys())
            values = list(result.param_importance.values())

            plt.figure(figsize=(10, 6))
            plt.barh(names, values, color="steelblue")
            plt.xlabel("Relative Importance")
            plt.title("Parameter Importance")
            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")


# =============================================================================
# ユーティリティ関数
# =============================================================================

def create_optimizer_from_settings() -> BayesianOptimizer:
    """グローバル設定からオプティマイザを生成

    Returns:
        設定済みのBayesianOptimizer
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()

        config = OptimizerConfig(
            n_calls=100,
            n_random_starts=20,
            acq_func="EI",
            cv_folds=5,
            verbose=True,
        )

        return BayesianOptimizer(config=config)

    except ImportError:
        logger.warning("Settings not available, using defaults")
        return BayesianOptimizer()


def quick_optimize(
    data: pd.DataFrame,
    universe: list[str],
    n_calls: int = 30,
) -> OptimizationResult:
    """クイック最適化（少ないイテレーションで実行）

    Args:
        data: OHLCVデータ
        universe: 銘柄リスト
        n_calls: 試行回数

    Returns:
        最適化結果
    """
    config = OptimizerConfig(
        n_calls=n_calls,
        n_random_starts=min(10, n_calls // 3),
        early_stop_patience=10,
    )
    optimizer = BayesianOptimizer(config=config)
    return optimizer.optimize(train_data=data, universe=universe)
