"""
PPO (Proximal Policy Optimization) Allocator - 深層強化学習による配分最適化

PPOを用いた強化学習ベースのポートフォリオ配分。
PyTorchなしで動作する軽量実装と、stable-baselines3を使った本格実装の両方をサポート。

主要コンポーネント:
- RLConfig: 強化学習設定
- TradingEnvironment: 取引環境（OpenAI Gym互換）
- SimplePPOAgent: PyTorchなしで動作するシンプルPPOエージェント
- train_ppo_allocator: 学習関数

設計根拠:
- 強化学習は非定常な金融市場での適応的な意思決定に有効
- PPOは安定した学習と実装のシンプルさで人気
- 軽量実装により、依存関係を最小限に抑えつつ動作確認可能

使用例:
    from src.ml.ppo_allocator import (
        TradingEnvironment,
        SimplePPOAgent,
        train_ppo_allocator,
    )

    # 環境作成
    env = TradingEnvironment(prices_df, lookback=20)

    # エージェント学習
    agent = train_ppo_allocator(prices_df, n_episodes=100)

    # 行動取得
    state = env.reset()
    action = agent.get_action(state, deterministic=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# 設定データクラス
# =============================================================================

@dataclass
class RLConfig:
    """強化学習設定

    Attributes:
        state_dim: 状態空間の次元
        action_dim: 行動空間の次元（アセット数）
        hidden_dim: 隠れ層の次元
        learning_rate: 学習率
        gamma: 割引率
        gae_lambda: GAE (Generalized Advantage Estimation) のλ
        clip_ratio: PPOのクリッピング比率
        epochs: 1回の更新でのエポック数
        batch_size: バッチサイズ
        entropy_coef: エントロピー係数（探索促進）
        value_coef: 価値関数の係数
        max_grad_norm: 勾配クリッピングの最大ノルム
    """
    state_dim: int = 60  # lookback * n_features
    action_dim: int = 5   # アセット数
    hidden_dim: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1)")


@dataclass
class EnvironmentConfig:
    """取引環境設定

    Attributes:
        lookback: ルックバック期間
        transaction_cost: 取引コスト（比率）
        initial_balance: 初期残高
        max_position: 最大ポジションサイズ
        reward_scaling: 報酬スケーリング
    """
    lookback: int = 20
    transaction_cost: float = 0.001
    initial_balance: float = 100000.0
    max_position: float = 1.0
    reward_scaling: float = 100.0


# =============================================================================
# 取引環境
# =============================================================================

class TradingEnvironment:
    """取引環境（OpenAI Gym互換）

    ポートフォリオ配分の強化学習環境。

    状態空間:
    - 過去のリターン（各アセット × lookback）
    - 過去のボラティリティ（各アセット × 1）
    - 現在のポジション（各アセット × 1）

    行動空間:
    - 各アセットへの配分比率（合計1に正規化）

    報酬:
    - ポートフォリオリターン - 取引コスト

    Usage:
        env = TradingEnvironment(prices_df, lookback=20)
        state = env.reset()

        for _ in range(n_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
        transaction_cost: float = 0.001,
        config: EnvironmentConfig | None = None,
    ) -> None:
        """初期化

        Args:
            prices: 価格データ（列=アセット、行=日付）
            lookback: ルックバック期間
            transaction_cost: 取引コスト
            config: 環境設定（指定時は他の引数は無視）
        """
        if config is not None:
            self.config = config
        else:
            self.config = EnvironmentConfig(
                lookback=lookback,
                transaction_cost=transaction_cost,
            )

        self.prices = prices
        self.assets = list(prices.columns)
        self.n_assets = len(self.assets)

        # リターン計算
        self.returns = prices.pct_change().dropna()
        self.n_steps = len(self.returns) - self.config.lookback

        if self.n_steps <= 0:
            raise ValueError("Insufficient data for given lookback")

        # 状態・行動空間の次元
        self.state_dim = self.config.lookback * self.n_assets + self.n_assets * 2
        self.action_dim = self.n_assets

        # 内部状態
        self._current_step = 0
        self._position = np.zeros(self.n_assets)
        self._balance = self.config.initial_balance
        self._portfolio_value = self.config.initial_balance

        # 履歴
        self._episode_returns: list[float] = []
        self._episode_actions: list[NDArray[np.float64]] = []

    def reset(self) -> NDArray[np.float64]:
        """環境をリセット

        Returns:
            初期状態
        """
        self._current_step = 0
        self._position = np.zeros(self.n_assets)
        self._balance = self.config.initial_balance
        self._portfolio_value = self.config.initial_balance
        self._episode_returns = []
        self._episode_actions = []

        return self._get_state()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
        """1ステップ進める

        Args:
            action: 行動（配分比率、合計1に正規化される）

        Returns:
            (次状態, 報酬, 終了フラグ, 追加情報)
        """
        # 行動を正規化（合計1、非負）
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.n_assets) / self.n_assets

        # 取引コスト計算
        position_change = np.abs(action - self._position)
        transaction_cost = position_change.sum() * self.config.transaction_cost

        # ポジション更新
        old_position = self._position.copy()
        self._position = action

        # 市場リターン取得
        step_idx = self.config.lookback + self._current_step
        if step_idx >= len(self.returns):
            return self._get_state(), 0.0, True, {"error": "Out of data"}

        market_returns = self.returns.iloc[step_idx].values

        # ポートフォリオリターン計算
        portfolio_return = np.dot(self._position, market_returns)

        # 報酬計算（リターン - コスト）
        reward = (portfolio_return - transaction_cost) * self.config.reward_scaling

        # ポートフォリオ価値更新
        self._portfolio_value *= (1 + portfolio_return - transaction_cost)

        # 履歴記録
        self._episode_returns.append(portfolio_return)
        self._episode_actions.append(action.copy())

        # 次ステップへ
        self._current_step += 1
        done = self._current_step >= self.n_steps

        info = {
            "portfolio_return": portfolio_return,
            "transaction_cost": transaction_cost,
            "portfolio_value": self._portfolio_value,
            "position": self._position.copy(),
            "step": self._current_step,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> NDArray[np.float64]:
        """現在の状態を取得

        Returns:
            状態ベクトル
        """
        step_idx = self.config.lookback + self._current_step

        # 過去リターン
        start_idx = step_idx - self.config.lookback
        end_idx = step_idx
        past_returns = self.returns.iloc[start_idx:end_idx].values.flatten()

        # ボラティリティ（過去リターンの標準偏差）
        volatility = self.returns.iloc[start_idx:end_idx].std().values

        # 現在のポジション
        position = self._position

        # 状態ベクトルを構築
        state = np.concatenate([past_returns, volatility, position])

        return state.astype(np.float32)

    def get_episode_stats(self) -> dict[str, float]:
        """エピソードの統計情報を取得

        Returns:
            統計情報辞書
        """
        if not self._episode_returns:
            return {}

        returns_array = np.array(self._episode_returns)
        total_return = (1 + returns_array).prod() - 1
        sharpe = returns_array.mean() / (returns_array.std() + 1e-8) * np.sqrt(252)

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "n_steps": len(self._episode_returns),
            "final_value": float(self._portfolio_value),
            "avg_position_change": float(np.mean([
                np.abs(self._episode_actions[i] - self._episode_actions[i-1]).sum()
                for i in range(1, len(self._episode_actions))
            ])) if len(self._episode_actions) > 1 else 0.0,
        }


# =============================================================================
# シンプルPPOエージェント（PyTorchなし）
# =============================================================================

class SimplePPOAgent:
    """シンプルPPOエージェント（PyTorchなしで動作）

    線形ポリシーネットワークを使用したシンプルな実装。
    学習用途というより、動作確認やベースライン用。

    本格的な学習にはstable-baselines3を推奨。

    Usage:
        agent = SimplePPOAgent(state_dim=60, action_dim=5)

        # 行動取得
        action = agent.get_action(state)

        # 学習
        agent.update(states, actions, rewards, values, log_probs)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        config: RLConfig | None = None,
    ) -> None:
        """初期化

        Args:
            state_dim: 状態空間の次元
            action_dim: 行動空間の次元
            learning_rate: 学習率
            config: RL設定（指定時は他の引数は無視）
        """
        if config is not None:
            self.config = config
            state_dim = config.state_dim
            action_dim = config.action_dim
            learning_rate = config.learning_rate
        else:
            self.config = RLConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # ポリシーネットワークの重み（線形）
        # state -> hidden -> action_mean
        hidden_dim = self.config.hidden_dim

        self._w1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self._b1 = np.zeros(hidden_dim)
        self._w2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self._b2 = np.zeros(action_dim)

        # 行動の標準偏差（学習可能パラメータ）
        self._log_std = np.zeros(action_dim)

        # 価値ネットワークの重み
        self._v_w1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self._v_b1 = np.zeros(hidden_dim)
        self._v_w2 = np.random.randn(hidden_dim, 1) * 0.1
        self._v_b2 = np.zeros(1)

        # 学習統計
        self._update_count = 0
        self._episode_rewards: list[float] = []

    def get_action(
        self,
        state: NDArray[np.float64],
        deterministic: bool = False,
    ) -> NDArray[np.float64]:
        """行動を取得

        Args:
            state: 状態ベクトル
            deterministic: 決定的に行動するか

        Returns:
            行動ベクトル（配分比率）
        """
        # フォワードパス
        mean = self._forward_policy(state)

        if deterministic:
            action = mean
        else:
            # ガウスノイズを追加
            std = np.exp(self._log_std)
            action = mean + std * np.random.randn(self.action_dim)

        # ソフトマックスで正規化（合計1、非負）
        action = self._softmax(action)

        return action

    def get_value(self, state: NDArray[np.float64]) -> float:
        """状態価値を取得

        Args:
            state: 状態ベクトル

        Returns:
            状態価値
        """
        return float(self._forward_value(state))

    def get_action_and_value(
        self,
        state: NDArray[np.float64],
        deterministic: bool = False,
    ) -> tuple[NDArray[np.float64], float, float]:
        """行動と価値を取得

        Args:
            state: 状態ベクトル
            deterministic: 決定的に行動するか

        Returns:
            (行動, 価値, ログ確率)
        """
        action = self.get_action(state, deterministic)
        value = self.get_value(state)

        # ログ確率を計算
        mean = self._forward_policy(state)
        std = np.exp(self._log_std)
        log_prob = self._gaussian_log_prob(action, mean, std)

        return action, value, log_prob

    def update(
        self,
        states: NDArray[np.float64],
        actions: NDArray[np.float64],
        rewards: NDArray[np.float64],
        values: NDArray[np.float64] | None = None,
        log_probs: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """パラメータを更新

        Args:
            states: 状態の配列
            actions: 行動の配列
            rewards: 報酬の配列
            values: 価値の配列（任意）
            log_probs: ログ確率の配列（任意）

        Returns:
            学習統計
        """
        n_samples = len(states)

        # リターンとアドバンテージを計算
        if values is None:
            values = np.array([self.get_value(s) for s in states])

        returns = self._compute_returns(rewards)
        advantages = returns - values

        # アドバンテージの正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        policy_loss_total = 0.0
        value_loss_total = 0.0

        for _ in range(self.config.epochs):
            # ミニバッチで更新
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # ポリシー損失
                policy_loss = self._compute_policy_loss(
                    batch_states, batch_actions, batch_advantages
                )
                policy_loss_total += policy_loss

                # 価値損失
                value_loss = self._compute_value_loss(batch_states, batch_returns)
                value_loss_total += value_loss

                # 勾配更新（シンプルなSGD）
                self._update_policy_gradients(
                    batch_states, batch_actions, batch_advantages
                )
                self._update_value_gradients(batch_states, batch_returns)

        self._update_count += 1
        self._episode_rewards.extend(rewards.tolist())

        return {
            "policy_loss": policy_loss_total / self.config.epochs,
            "value_loss": value_loss_total / self.config.epochs,
            "update_count": self._update_count,
        }

    def _forward_policy(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """ポリシーネットワークのフォワードパス

        Args:
            state: 状態

        Returns:
            行動の平均
        """
        # 隠れ層
        h = np.tanh(state @ self._w1 + self._b1)
        # 出力層
        mean = h @ self._w2 + self._b2
        return mean

    def _forward_value(self, state: NDArray[np.float64]) -> float:
        """価値ネットワークのフォワードパス

        Args:
            state: 状態

        Returns:
            状態価値
        """
        # 隠れ層
        h = np.tanh(state @ self._v_w1 + self._v_b1)
        # 出力層
        value = h @ self._v_w2 + self._v_b2
        return float(value[0])

    def _softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """ソフトマックス関数

        Args:
            x: 入力

        Returns:
            ソフトマックス出力
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _gaussian_log_prob(
        self,
        action: NDArray[np.float64],
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
    ) -> float:
        """ガウス分布のログ確率

        Args:
            action: 行動
            mean: 平均
            std: 標準偏差

        Returns:
            ログ確率
        """
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var +
            np.log(var) +
            np.log(2 * np.pi)
        )
        return float(log_prob.sum())

    def _compute_returns(self, rewards: NDArray[np.float64]) -> NDArray[np.float64]:
        """割引リターンを計算

        Args:
            rewards: 報酬配列

        Returns:
            割引リターン
        """
        returns = np.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def _compute_policy_loss(
        self,
        states: NDArray[np.float64],
        actions: NDArray[np.float64],
        advantages: NDArray[np.float64],
    ) -> float:
        """ポリシー損失を計算

        Args:
            states: 状態配列
            actions: 行動配列
            advantages: アドバンテージ配列

        Returns:
            ポリシー損失
        """
        loss = 0.0
        for state, action, adv in zip(states, actions, advantages):
            mean = self._forward_policy(state)
            std = np.exp(self._log_std)
            log_prob = self._gaussian_log_prob(action, mean, std)
            loss -= log_prob * adv  # ポリシー勾配

        return loss / len(states)

    def _compute_value_loss(
        self,
        states: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> float:
        """価値損失を計算

        Args:
            states: 状態配列
            returns: リターン配列

        Returns:
            価値損失（MSE）
        """
        loss = 0.0
        for state, ret in zip(states, returns):
            value = self._forward_value(state)
            loss += (value - ret) ** 2

        return loss / len(states)

    def _update_policy_gradients(
        self,
        states: NDArray[np.float64],
        actions: NDArray[np.float64],
        advantages: NDArray[np.float64],
    ) -> None:
        """ポリシー勾配を更新（シンプルなSGD）

        Args:
            states: 状態配列
            actions: 行動配列
            advantages: アドバンテージ配列
        """
        # 数値勾配で近似（シンプルな実装）
        eps = 1e-5

        for state, action, adv in zip(states, actions, advantages):
            # w2の勾配
            h = np.tanh(state @ self._w1 + self._b1)
            mean = h @ self._w2 + self._b2
            std = np.exp(self._log_std)

            # 勾配計算（解析的）
            diff = (action - mean) / (std ** 2)
            grad_mean = diff * adv

            # w2更新
            self._w2 += self.learning_rate * np.outer(h, grad_mean)
            self._b2 += self.learning_rate * grad_mean

    def _update_value_gradients(
        self,
        states: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> None:
        """価値勾配を更新

        Args:
            states: 状態配列
            returns: リターン配列
        """
        for state, ret in zip(states, returns):
            # フォワードパス
            h = np.tanh(state @ self._v_w1 + self._v_b1)
            value = float((h @ self._v_w2 + self._v_b2)[0])

            # 勾配計算
            error = value - ret

            # v_w2更新
            self._v_w2 -= self.learning_rate * error * h.reshape(-1, 1)
            self._v_b2 -= self.learning_rate * error

    def get_stats(self) -> dict[str, Any]:
        """統計情報を取得

        Returns:
            統計情報辞書
        """
        return {
            "update_count": self._update_count,
            "total_rewards": sum(self._episode_rewards),
            "avg_reward": np.mean(self._episode_rewards) if self._episode_rewards else 0,
        }

    def save(self, path: str) -> None:
        """パラメータを保存

        Args:
            path: 保存パス
        """
        params = {
            "w1": self._w1,
            "b1": self._b1,
            "w2": self._w2,
            "b2": self._b2,
            "log_std": self._log_std,
            "v_w1": self._v_w1,
            "v_b1": self._v_b1,
            "v_w2": self._v_w2,
            "v_b2": self._v_b2,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
            },
        }
        np.savez(path, **params)
        logger.info(f"Saved agent to {path}")

    @classmethod
    def load(cls, path: str) -> "SimplePPOAgent":
        """パラメータを読み込み

        Args:
            path: 読み込みパス

        Returns:
            エージェント
        """
        data = np.load(path, allow_pickle=True)
        config = data["config"].item()

        agent = cls(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            learning_rate=config["learning_rate"],
        )

        agent._w1 = data["w1"]
        agent._b1 = data["b1"]
        agent._w2 = data["w2"]
        agent._b2 = data["b2"]
        agent._log_std = data["log_std"]
        agent._v_w1 = data["v_w1"]
        agent._v_b1 = data["v_b1"]
        agent._v_w2 = data["v_w2"]
        agent._v_b2 = data["v_b2"]

        logger.info(f"Loaded agent from {path}")
        return agent


# =============================================================================
# 学習関数
# =============================================================================

def train_ppo_allocator(
    prices: pd.DataFrame,
    n_episodes: int = 100,
    lookback: int = 20,
    transaction_cost: float = 0.001,
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> SimplePPOAgent:
    """PPOエージェントを学習

    Args:
        prices: 価格データ
        n_episodes: エピソード数
        lookback: ルックバック期間
        transaction_cost: 取引コスト
        learning_rate: 学習率
        verbose: 詳細出力

    Returns:
        学習済みエージェント
    """
    # 環境作成
    env = TradingEnvironment(
        prices=prices,
        lookback=lookback,
        transaction_cost=transaction_cost,
    )

    # エージェント作成
    agent = SimplePPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=learning_rate,
    )

    # 学習ループ
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        done = False
        total_reward = 0.0

        while not done:
            action, value, log_prob = agent.get_action_and_value(state)
            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            total_reward += reward
            state = next_state

        # エピソード終了後に更新
        if len(states) > 0:
            agent.update(
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                values=np.array(values),
                log_probs=np.array(log_probs),
            )

        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            stats = env.get_episode_stats()
            logger.info(
                f"Episode {episode + 1}/{n_episodes}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Sharpe: {stats.get('sharpe_ratio', 0):.3f}"
            )

    return agent


def create_ppo_allocator(
    prices: pd.DataFrame,
    config: RLConfig | None = None,
    env_config: EnvironmentConfig | None = None,
) -> tuple[SimplePPOAgent, TradingEnvironment]:
    """PPOアロケータを作成（学習なし）

    Args:
        prices: 価格データ
        config: RL設定
        env_config: 環境設定

    Returns:
        (エージェント, 環境)
    """
    env = TradingEnvironment(prices=prices, config=env_config)

    if config is None:
        config = RLConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
        )

    agent = SimplePPOAgent(config=config)

    return agent, env


# =============================================================================
# Stable-Baselines3 ラッパー（オプショナル）
# =============================================================================

def create_sb3_environment(
    prices: pd.DataFrame,
    lookback: int = 20,
    transaction_cost: float = 0.001,
) -> Any:
    """Stable-Baselines3用の環境を作成

    Args:
        prices: 価格データ
        lookback: ルックバック期間
        transaction_cost: 取引コスト

    Returns:
        Gym環境（SB3互換）

    Raises:
        ImportError: gym/gymnasiumがインストールされていない場合
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        try:
            import gym
            from gym import spaces
        except ImportError:
            raise ImportError(
                "gymnasium or gym is required for SB3 integration. "
                "Install with: pip install gymnasium"
            )

    class GymTradingEnv(gym.Env):
        """Gym互換の取引環境"""

        def __init__(self) -> None:
            super().__init__()
            self._env = TradingEnvironment(
                prices=prices,
                lookback=lookback,
                transaction_cost=transaction_cost,
            )

            # 行動空間: 配分比率（0-1）
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._env.action_dim,),
                dtype=np.float32,
            )

            # 状態空間
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._env.state_dim,),
                dtype=np.float32,
            )

        def reset(self, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                np.random.seed(seed)
            state = self._env.reset()
            return state, {}

        def step(self, action):
            state, reward, done, info = self._env.step(action)
            return state, reward, done, False, info

    return GymTradingEnv()


def train_with_sb3(
    prices: pd.DataFrame,
    total_timesteps: int = 10000,
    lookback: int = 20,
    transaction_cost: float = 0.001,
) -> Any:
    """Stable-Baselines3でPPOを学習

    Args:
        prices: 価格データ
        total_timesteps: 総タイムステップ数
        lookback: ルックバック期間
        transaction_cost: 取引コスト

    Returns:
        学習済みSB3モデル

    Raises:
        ImportError: stable-baselines3がインストールされていない場合
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required. "
            "Install with: pip install stable-baselines3"
        )

    env = create_sb3_environment(prices, lookback, transaction_cost)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    return model
