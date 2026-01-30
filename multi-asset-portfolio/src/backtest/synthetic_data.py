"""
Synthetic Data Generation Module - 合成データ生成

バックテストの統計的検証のためのブートストラップ・モンテカルロシミュレーション。
戦略のロバスト性と統計的有意性を検証する。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class BootstrapResult:
    """ブートストラップ結果"""
    samples: List[np.ndarray]
    n_samples: int
    sample_size: int
    method: str


@dataclass
class SimulationResult:
    """シミュレーション結果"""
    paths: List[np.ndarray]
    n_simulations: int
    n_days: int
    method: str
    parameters: Dict


@dataclass
class SignificanceTestResult:
    """統計的有意性検定結果"""
    observed_sharpe: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    confidence_level: float
    n_bootstrap: int


class SyntheticDataGenerator:
    """
    合成データ生成器

    ブートストラップ法とモンテカルロシミュレーションを用いて
    戦略検証用の合成データを生成する。
    """

    def __init__(self, random_seed: int = 42):
        """
        初期化

        Parameters
        ----------
        random_seed : int
            乱数シード（再現性確保）
        """
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

    def reset_seed(self, seed: Optional[int] = None):
        """乱数シードをリセット"""
        if seed is None:
            seed = self.random_seed
        self._rng = np.random.RandomState(seed)

    def bootstrap_returns(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_samples: int = 1000,
        sample_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        単純ブートストラップ（時系列依存性なし）

        リターンデータからランダムに復元抽出してサンプルを生成。
        時系列の自己相関は保持されない。

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            元のリターン系列
        n_samples : int
            生成するサンプル数
        sample_size : int, optional
            各サンプルのサイズ（デフォルトは元データと同じ）

        Returns
        -------
        List[np.ndarray]
            ブートストラップサンプルのリスト
        """
        returns_arr = np.asarray(returns)
        n = len(returns_arr)

        if sample_size is None:
            sample_size = n

        samples = []
        for _ in range(n_samples):
            # 復元抽出
            indices = self._rng.randint(0, n, size=sample_size)
            sample = returns_arr[indices]
            samples.append(sample)

        return samples

    def block_bootstrap(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_samples: int = 1000,
        block_size: int = 20,
    ) -> List[np.ndarray]:
        """
        ブロックブートストラップ（自己相関保持）

        連続するブロック単位でリサンプリングし、時系列の自己相関構造を保持。

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            元のリターン系列
        n_samples : int
            生成するサンプル数
        block_size : int
            ブロックサイズ（自己相関の持続期間に応じて設定）

        Returns
        -------
        List[np.ndarray]
            ブロックブートストラップサンプルのリスト
        """
        returns_arr = np.asarray(returns)
        n = len(returns_arr)

        if block_size > n:
            block_size = n

        # ブロック開始位置の候補
        n_blocks_available = n - block_size + 1

        # 必要なブロック数
        n_blocks_needed = (n + block_size - 1) // block_size

        samples = []
        for _ in range(n_samples):
            # ランダムにブロック開始位置を選択
            block_starts = self._rng.randint(0, n_blocks_available, size=n_blocks_needed)

            # ブロックを連結
            blocks = []
            for start in block_starts:
                blocks.append(returns_arr[start:start + block_size])

            sample = np.concatenate(blocks)[:n]  # 元のサイズにトリミング
            samples.append(sample)

        return samples

    def monte_carlo_simulation(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_simulations: int = 1000,
        n_days: int = 252,
    ) -> List[np.ndarray]:
        """
        モンテカルロシミュレーション

        過去のリターンの統計量（平均・標準偏差）を用いて
        将来のリターンパスを生成。正規分布を仮定。

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            過去のリターン系列
        n_simulations : int
            シミュレーション回数
        n_days : int
            シミュレーション日数

        Returns
        -------
        List[np.ndarray]
            シミュレーションパスのリスト
        """
        returns_arr = np.asarray(returns)

        # 統計量を計算
        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr)

        paths = []
        for _ in range(n_simulations):
            # 正規分布からサンプリング
            simulated_returns = self._rng.normal(mu, sigma, size=n_days)
            paths.append(simulated_returns)

        return paths

    def regime_aware_simulation(
        self,
        returns: Union[pd.Series, np.ndarray],
        regime_labels: Union[pd.Series, np.ndarray],
        n_simulations: int = 1000,
        n_days: int = 252,
    ) -> List[np.ndarray]:
        """
        レジーム別シミュレーション

        各レジームの統計量を個別に計算し、レジーム遷移を考慮した
        シミュレーションを実行。

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            過去のリターン系列
        regime_labels : pd.Series or np.ndarray
            各時点のレジームラベル（returns と同じ長さ）
        n_simulations : int
            シミュレーション回数
        n_days : int
            シミュレーション日数

        Returns
        -------
        List[np.ndarray]
            シミュレーションパスのリスト
        """
        returns_arr = np.asarray(returns)
        regime_arr = np.asarray(regime_labels)

        if len(returns_arr) != len(regime_arr):
            raise ValueError("returns and regime_labels must have the same length")

        # レジーム別統計量を計算
        unique_regimes = np.unique(regime_arr)
        regime_stats = {}

        for regime in unique_regimes:
            mask = regime_arr == regime
            regime_returns = returns_arr[mask]
            if len(regime_returns) > 0:
                regime_stats[regime] = {
                    'mu': np.mean(regime_returns),
                    'sigma': np.std(regime_returns) if len(regime_returns) > 1 else 0.01,
                    'count': len(regime_returns),
                }

        # レジーム遷移確率を計算
        transition_matrix = self._compute_transition_matrix(regime_arr, unique_regimes)

        # レジーム出現頻度（初期レジーム選択用）
        regime_counts = {r: stats['count'] for r, stats in regime_stats.items()}
        total_count = sum(regime_counts.values())
        regime_probs = {r: c / total_count for r, c in regime_counts.items()}

        paths = []
        for _ in range(n_simulations):
            # 初期レジームを頻度に応じて選択
            current_regime = self._rng.choice(
                list(regime_probs.keys()),
                p=list(regime_probs.values())
            )

            simulated_returns = []
            for _ in range(n_days):
                # 現在のレジームの統計量でサンプリング
                stats = regime_stats[current_regime]
                ret = self._rng.normal(stats['mu'], stats['sigma'])
                simulated_returns.append(ret)

                # レジーム遷移
                current_regime = self._sample_next_regime(
                    current_regime, transition_matrix, unique_regimes
                )

            paths.append(np.array(simulated_returns))

        return paths

    def _compute_transition_matrix(
        self,
        regime_arr: np.ndarray,
        unique_regimes: np.ndarray,
    ) -> Dict:
        """レジーム遷移確率行列を計算"""
        n_regimes = len(unique_regimes)
        regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}

        # 遷移カウント
        transition_counts = np.zeros((n_regimes, n_regimes))

        for i in range(len(regime_arr) - 1):
            from_regime = regime_arr[i]
            to_regime = regime_arr[i + 1]
            from_idx = regime_to_idx[from_regime]
            to_idx = regime_to_idx[to_regime]
            transition_counts[from_idx, to_idx] += 1

        # 確率に変換
        transition_probs = {}
        for regime in unique_regimes:
            idx = regime_to_idx[regime]
            row_sum = transition_counts[idx].sum()
            if row_sum > 0:
                probs = transition_counts[idx] / row_sum
            else:
                # 遷移データがない場合は均等
                probs = np.ones(n_regimes) / n_regimes
            transition_probs[regime] = dict(zip(unique_regimes, probs))

        return transition_probs

    def _sample_next_regime(
        self,
        current_regime,
        transition_matrix: Dict,
        unique_regimes: np.ndarray,
    ):
        """遷移確率に基づいて次のレジームをサンプリング"""
        probs = transition_matrix[current_regime]
        regimes = list(probs.keys())
        probabilities = list(probs.values())
        return self._rng.choice(regimes, p=probabilities)


class StatisticalSignificanceTester:
    """
    統計的有意性検定器

    ブートストラップ法を用いて戦略のシャープレシオの
    統計的有意性を検定する。
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ):
        """
        初期化

        Parameters
        ----------
        n_bootstrap : int
            ブートストラップ回数
        confidence_level : float
            信頼水準（0-1）
        random_seed : int
            乱数シード
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self._generator = SyntheticDataGenerator(random_seed=random_seed)

    def test_sharpe_significance(
        self,
        strategy_returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict:
        """
        シャープレシオの統計的有意性を検定

        Parameters
        ----------
        strategy_returns : pd.Series or np.ndarray
            戦略のリターン系列
        benchmark_returns : pd.Series or np.ndarray, optional
            ベンチマークのリターン系列（差分のシャープを検定）

        Returns
        -------
        dict
            検定結果
            - observed_sharpe: 観測されたシャープレシオ
            - ci_lower: 信頼区間下限
            - ci_upper: 信頼区間上限
            - p_value: p値（シャープ > 0 の検定）
            - significant: 統計的に有意か
        """
        strategy_arr = np.asarray(strategy_returns)

        # ベンチマークがある場合は差分を取る
        if benchmark_returns is not None:
            benchmark_arr = np.asarray(benchmark_returns)
            if len(strategy_arr) != len(benchmark_arr):
                raise ValueError("strategy and benchmark must have the same length")
            returns_to_test = strategy_arr - benchmark_arr
        else:
            returns_to_test = strategy_arr

        # 観測されたシャープレシオを計算
        observed_sharpe = self._compute_annualized_sharpe(returns_to_test)

        # ブートストラップでシャープレシオの分布を推定
        bootstrap_samples = self._generator.bootstrap_returns(
            returns_to_test,
            n_samples=self.n_bootstrap,
        )

        bootstrap_sharpes = []
        for sample in bootstrap_samples:
            sharpe = self._compute_annualized_sharpe(sample)
            if not np.isnan(sharpe) and not np.isinf(sharpe):
                bootstrap_sharpes.append(sharpe)

        bootstrap_sharpes = np.array(bootstrap_sharpes)

        # 信頼区間を計算
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

        # p値を計算（シャープ > 0 の検定）
        p_value = np.mean(bootstrap_sharpes <= 0)

        # 有意性判定
        significant = p_value < alpha

        return {
            'observed_sharpe': observed_sharpe,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': significant,
            'confidence_level': self.confidence_level,
            'n_bootstrap': self.n_bootstrap,
        }

    def test_sharpe_significance_detailed(
        self,
        strategy_returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> SignificanceTestResult:
        """詳細な結果オブジェクトを返すバージョン"""
        result = self.test_sharpe_significance(strategy_returns, benchmark_returns)
        return SignificanceTestResult(**result)

    def _compute_annualized_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """年率シャープレシオを計算"""
        if len(returns) == 0:
            return np.nan

        excess_returns = returns - risk_free_rate / periods_per_year
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0 or np.isnan(std_return):
            return np.nan

        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        return sharpe


# ショートカット関数
def bootstrap_returns(
    returns: Union[pd.Series, np.ndarray],
    n_samples: int = 1000,
    block_size: Optional[int] = None,
    random_seed: int = 42,
) -> List[np.ndarray]:
    """
    ブートストラップサンプルを生成するショートカット関数

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        元のリターン系列
    n_samples : int
        サンプル数
    block_size : int, optional
        指定時はブロックブートストラップを使用
    random_seed : int
        乱数シード

    Returns
    -------
    List[np.ndarray]
        ブートストラップサンプル
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)

    if block_size is not None:
        return generator.block_bootstrap(returns, n_samples, block_size)
    else:
        return generator.bootstrap_returns(returns, n_samples)


def test_sharpe_significance(
    strategy_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict:
    """
    シャープレシオの有意性検定ショートカット関数

    Returns
    -------
    dict
        検定結果（observed_sharpe, ci_lower, ci_upper, p_value, significant）
    """
    tester = StatisticalSignificanceTester(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    return tester.test_sharpe_significance(strategy_returns, benchmark_returns)
