"""
Kelly Allocator Module - Kelly係数ベースのポジションサイジング

Kelly公式に基づく最適ポジションサイズの計算を提供する。

Kelly公式:
    f* = (p*b - q) / b
    where:
        p = 勝率 (win rate)
        q = 1 - p = 敗率 (loss rate)
        b = avg_win / avg_loss = ペイオフ比 (payoff ratio)

Fractional Kelly:
    実運用では過度なレバレッジを避けるため、
    Full Kelly の一部（1/4 = Quarter Kelly など）を使用。

設計根拠:
- 要求.md §7: Strategyの採用と重み
- Kelly基準は理論上最適だが、推定誤差に敏感
- Fractional Kelly でリスク低減

使用例:
    from src.allocation.kelly_allocator import KellyAllocator

    allocator = KellyAllocator(fraction=0.25, max_weight=0.25)

    # 単一戦略のKelly計算
    kelly_result = allocator.calculate_strategy_kelly(strategy_returns)
    print(f"Adjusted Kelly: {kelly_result.adjusted_kelly:.2%}")

    # ポートフォリオ統合
    final_weights = allocator.allocate(
        strategy_returns={"momentum": mom_ret, "reversion": rev_ret},
        base_weights={"momentum": 0.6, "reversion": 0.4},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly計算結果

    Attributes:
        win_rate: 勝率 (0-1)
        avg_win: 平均勝ちリターン
        avg_loss: 平均負けリターン（絶対値）
        payoff_ratio: ペイオフ比 (avg_win / avg_loss)
        full_kelly: フルKelly比率 (0-1+)
        adjusted_kelly: Fractional Kelly比率 (0-max_weight)
        n_trades: 分析対象取引数
        edge: エッジ (expected_value / avg_loss)
    """

    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    full_kelly: float
    adjusted_kelly: float
    n_trades: int = 0
    edge: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "payoff_ratio": self.payoff_ratio,
            "full_kelly": self.full_kelly,
            "adjusted_kelly": self.adjusted_kelly,
            "n_trades": self.n_trades,
            "edge": self.edge,
        }

    @property
    def has_edge(self) -> bool:
        """正のエッジを持つか"""
        return self.full_kelly > 0


@dataclass
class KellyAllocationResult:
    """Kelly配分結果

    Attributes:
        weights: 最終配分重み
        kelly_results: 各戦略のKelly計算結果
        base_weights: 入力された基本配分
        blend_ratio: Kelly比重（0=基本配分のみ, 1=Kellyのみ）
        total_kelly_weight: Kelly計算による総重み（正規化前）
        metadata: 追加メタデータ
    """

    weights: Dict[str, float]
    kelly_results: Dict[str, KellyResult] = field(default_factory=dict)
    base_weights: Dict[str, float] = field(default_factory=dict)
    blend_ratio: float = 0.5
    total_kelly_weight: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "weights": self.weights,
            "kelly_results": {k: v.to_dict() for k, v in self.kelly_results.items()},
            "base_weights": self.base_weights,
            "blend_ratio": self.blend_ratio,
            "total_kelly_weight": self.total_kelly_weight,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class KellyConfig:
    """Kelly配分設定

    Attributes:
        fraction: Kelly比率（0.25 = Quarter Kelly）
        max_weight: 単一戦略の最大重み
        min_trades: 最低必要取引数
        lookback_days: 計算対象期間（日数）
        blend_ratio: Kelly比重（0-1, 0.5 = 基本配分とKellyの50/50）
        use_geometric: 幾何平均を使用するか（対数リターン向け）
        regularization: 正則化係数（0-1, Kelly推定の安定化）
    """

    fraction: float = 0.25
    max_weight: float = 0.25
    min_trades: int = 20
    lookback_days: int = 252
    blend_ratio: float = 0.5
    use_geometric: bool = False
    regularization: float = 0.1

    def __post_init__(self) -> None:
        """バリデーション"""
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if not 0 <= self.blend_ratio <= 1:
            raise ValueError("blend_ratio must be in [0, 1]")
        if not 0 <= self.regularization <= 1:
            raise ValueError("regularization must be in [0, 1]")


class KellyAllocator:
    """
    Kelly係数ベースのポジションサイジング

    Kelly公式を使用して各戦略の最適配分比率を計算する。
    実運用では Fractional Kelly（Quarter Kelly など）を使用して
    推定誤差によるリスクを低減する。

    Usage:
        allocator = KellyAllocator(fraction=0.25, max_weight=0.25)

        # 単一戦略
        result = allocator.calculate_strategy_kelly(returns)
        print(f"Kelly Weight: {result.adjusted_kelly:.2%}")

        # ポートフォリオ
        weights = allocator.allocate(
            strategy_returns={"mom": mom_ret, "rev": rev_ret},
            base_weights={"mom": 0.6, "rev": 0.4},
        )

    Note:
        - Kelly公式は理論上最適だが、推定誤差に非常に敏感
        - 実運用では必ず Fractional Kelly を使用
        - 負のKellyは「ベットするな」を意味（重み0）
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_weight: float = 0.25,
        config: Optional[KellyConfig] = None,
    ) -> None:
        """
        初期化

        Args:
            fraction: Kelly比率（0.25 = Quarter Kelly）
            max_weight: 単一戦略の最大重み
            config: 詳細設定（指定時はfraction/max_weightを上書き）
        """
        if config is not None:
            self.config = config
        else:
            self.config = KellyConfig(
                fraction=fraction,
                max_weight=max_weight,
            )

        logger.info(
            f"KellyAllocator initialized: fraction={self.config.fraction}, "
            f"max_weight={self.config.max_weight}"
        )

    @property
    def fraction(self) -> float:
        """Kelly比率"""
        return self.config.fraction

    @property
    def max_weight(self) -> float:
        """最大重み"""
        return self.config.max_weight

    def calculate_kelly_weight(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly公式で最適重みを計算

        Kelly公式: f* = (p*b - q) / b
        where:
            p = win_rate
            q = 1 - win_rate
            b = avg_win / avg_loss (payoff ratio)

        Args:
            win_rate: 勝率 (0-1)
            avg_win: 平均勝ちリターン（正の値）
            avg_loss: 平均負けリターン（正の値、絶対値）

        Returns:
            調整済みKelly重み (0 - max_weight)
        """
        # バリデーション
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        # ペイオフ比
        b = avg_win / avg_loss

        # Kelly公式
        q = 1 - win_rate
        f_star = (win_rate * b - q) / b

        # 負のKellyは0に（ベットするな）
        f_star = max(0.0, f_star)

        # 正則化（推定誤差の影響を緩和）
        if self.config.regularization > 0:
            # 勝率50%、ペイオフ1:1（期待値0）に向けて収縮
            reg = self.config.regularization
            f_star = f_star * (1 - reg)

        # Fractional Kelly
        f_adjusted = f_star * self.fraction

        # 上限適用
        return min(f_adjusted, self.max_weight)

    def calculate_strategy_kelly(
        self,
        strategy_returns: pd.Series,
        lookback_days: Optional[int] = None,
    ) -> KellyResult:
        """
        戦略リターン系列からKelly係数を計算

        Args:
            strategy_returns: 日次リターン系列
            lookback_days: 計算対象期間（None=設定値使用）

        Returns:
            KellyResult
        """
        lookback = lookback_days or self.config.lookback_days

        # 直近データを使用
        returns = strategy_returns.tail(lookback).dropna()

        if len(returns) < self.config.min_trades:
            logger.warning(
                f"Insufficient trades: {len(returns)} < {self.config.min_trades}"
            )
            return KellyResult(
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                payoff_ratio=0.0,
                full_kelly=0.0,
                adjusted_kelly=0.0,
                n_trades=len(returns),
                edge=0.0,
            )

        # 勝ち/負けを分離
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        n_wins = len(wins)
        n_losses = len(losses)
        n_trades = len(returns)

        # 勝率
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        # 平均勝ち/負けリターン
        if self.config.use_geometric:
            # 幾何平均（対数リターン向け）
            avg_win = np.exp(np.mean(np.log(1 + wins))) - 1 if n_wins > 0 else 0.0
            avg_loss = 1 - np.exp(np.mean(np.log(1 - abs(losses)))) if n_losses > 0 else 0.01
        else:
            # 算術平均
            avg_win = wins.mean() if n_wins > 0 else 0.0
            avg_loss = abs(losses.mean()) if n_losses > 0 else 0.01

        # ペイオフ比
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # フルKelly
        if avg_loss > 0:
            b = payoff_ratio
            q = 1 - win_rate
            full_kelly = (win_rate * b - q) / b if b > 0 else 0.0
            full_kelly = max(0.0, full_kelly)
        else:
            full_kelly = 0.0

        # 調整済みKelly
        adjusted_kelly = self.calculate_kelly_weight(win_rate, avg_win, avg_loss)

        # エッジ（期待値 / 平均負け）
        expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss
        edge = expected_value / avg_loss if avg_loss > 0 else 0.0

        return KellyResult(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=payoff_ratio,
            full_kelly=full_kelly,
            adjusted_kelly=adjusted_kelly,
            n_trades=n_trades,
            edge=edge,
        )

    def allocate(
        self,
        strategy_returns: Dict[str, pd.Series],
        base_weights: Dict[str, float],
        blend_ratio: Optional[float] = None,
    ) -> KellyAllocationResult:
        """
        基本配分をKelly係数で調整してポートフォリオ配分を計算

        Args:
            strategy_returns: 戦略ID -> リターン系列のマッピング
            base_weights: 基本配分 {strategy_id: weight}
            blend_ratio: Kelly比重（None=設定値使用）

        Returns:
            KellyAllocationResult
        """
        blend = blend_ratio if blend_ratio is not None else self.config.blend_ratio

        kelly_results: Dict[str, KellyResult] = {}
        kelly_weights: Dict[str, float] = {}

        # 各戦略のKellyを計算
        for strategy_id, returns in strategy_returns.items():
            kelly_result = self.calculate_strategy_kelly(returns)
            kelly_results[strategy_id] = kelly_result

            base_w = base_weights.get(strategy_id, 0.0)
            kelly_w = kelly_result.adjusted_kelly

            # 基本配分とKelly配分の加重平均
            blended_w = (1 - blend) * base_w + blend * kelly_w
            kelly_weights[strategy_id] = blended_w

            logger.debug(
                f"Strategy {strategy_id}: base={base_w:.3f}, "
                f"kelly={kelly_w:.3f}, blended={blended_w:.3f}"
            )

        # 重みがない戦略は基本配分のみ使用
        for strategy_id, base_w in base_weights.items():
            if strategy_id not in kelly_weights:
                kelly_weights[strategy_id] = base_w

        # 総Kelly重み（正規化前）
        total_kelly = sum(kelly_weights.values())

        # 正規化
        if total_kelly > 0:
            normalized_weights = {k: v / total_kelly for k, v in kelly_weights.items()}
        else:
            # Kelly計算結果が全て0の場合は基本配分にフォールバック
            logger.warning("All Kelly weights are zero, falling back to base weights")
            total_base = sum(base_weights.values())
            if total_base > 0:
                normalized_weights = {k: v / total_base for k, v in base_weights.items()}
            else:
                # 均等配分にフォールバック
                n = len(base_weights)
                normalized_weights = {k: 1.0 / n for k in base_weights}

        return KellyAllocationResult(
            weights=normalized_weights,
            kelly_results=kelly_results,
            base_weights=base_weights.copy(),
            blend_ratio=blend,
            total_kelly_weight=total_kelly,
            metadata={
                "fraction": self.fraction,
                "max_weight": self.max_weight,
                "n_strategies": len(strategy_returns),
            },
        )

    def calculate_optimal_fraction(
        self,
        strategy_returns: pd.Series,
        target_max_drawdown: float = 0.20,
        n_simulations: int = 1000,
    ) -> float:
        """
        目標ドローダウンに対する最適Kelly比率を計算

        モンテカルロシミュレーションで、指定のドローダウン制約を
        満たすKelly比率を推定する。

        Args:
            strategy_returns: 戦略リターン系列
            target_max_drawdown: 目標最大ドローダウン（例: 0.20 = 20%）
            n_simulations: シミュレーション回数

        Returns:
            推奨Kelly比率
        """
        kelly_result = self.calculate_strategy_kelly(strategy_returns)
        full_kelly = kelly_result.full_kelly

        if full_kelly <= 0:
            return 0.0

        returns = strategy_returns.dropna().values
        n_periods = len(returns)

        if n_periods < 20:
            return self.fraction  # デフォルト値

        best_fraction = 0.0
        fractions_to_test = np.arange(0.05, 1.05, 0.05)

        for frac in fractions_to_test:
            kelly_weight = full_kelly * frac
            max_dds = []

            for _ in range(n_simulations):
                # リサンプリング
                sampled_returns = np.random.choice(returns, size=n_periods, replace=True)

                # ポートフォリオリターン
                portfolio_returns = sampled_returns * kelly_weight

                # 累積リターン
                cumulative = np.cumprod(1 + portfolio_returns)

                # ドローダウン計算
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_dd = abs(np.min(drawdowns))
                max_dds.append(max_dd)

            # 95パーセンタイルのドローダウン
            dd_95 = np.percentile(max_dds, 95)

            if dd_95 <= target_max_drawdown:
                best_fraction = frac
            else:
                break

        return best_fraction


def create_kelly_allocator_from_settings() -> KellyAllocator:
    """
    グローバル設定からKellyAllocatorを生成

    Returns:
        設定済みのKellyAllocator
    """
    try:
        from src.config.settings import get_settings

        settings = get_settings()

        # 設定からKellyパラメータを取得
        # （settings.yamlにkelly_fractionなどが定義されている場合）
        fraction = getattr(settings, "kelly_fraction", 0.25)
        max_weight = getattr(settings, "kelly_max_weight", 0.25)

        return KellyAllocator(fraction=fraction, max_weight=max_weight)

    except ImportError:
        logger.warning("Settings not available, using default KellyAllocator")
        return KellyAllocator()
