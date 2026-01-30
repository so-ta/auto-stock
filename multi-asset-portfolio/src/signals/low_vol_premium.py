"""
Low Volatility Premium Strategy - 低ボラティリティプレミアム

低ボラティリティ銘柄のアウトパフォーマンス傾向を活用した戦略。

学術的背景:
- 低ボラティリティアノマリー（Low Volatility Anomaly）
- 高ボラティリティ銘柄は期待ほどリターンが高くない
- 低ボラティリティ銘柄はリスク調整後リターンが優れる傾向

実装:
- ボラティリティでユニバースをランク付け
- 低ボラ銘柄をオーバーウェイト
- 高ボラ銘柄をアンダーウェイト

使用例:
    from src.signals.low_vol_premium import LowVolPremiumStrategy

    strategy = LowVolPremiumStrategy(
        lookback=60,
        target_percentile=0.3,
        adjustment_strength=0.3,
    )

    # ボラティリティランク計算
    vol_ranks = strategy.calculate_volatility_rank(returns_df)

    # 調整係数を生成
    adjustments = strategy.generate_low_vol_overweight(vol_ranks)

    # 基本配分に適用
    adjusted_weights = strategy.apply_adjustment(base_weights, adjustments)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.signals.base import ParameterSpec, Signal, SignalResult
from src.signals.registry import SignalRegistry

logger = logging.getLogger(__name__)


@dataclass
class LowVolPremiumConfig:
    """低ボラプレミアム戦略の設定

    Attributes:
        lookback: ボラティリティ計算のルックバック期間（日数）
        target_percentile: オーバーウェイト対象の下位パーセンタイル
        adjustment_strength: 調整の強さ（0-1）
        underweight_high_vol: 高ボラ銘柄をアンダーウェイトするか
        min_observations: 最小必要観測数
        volatility_type: ボラティリティの種類 (std | ewm | parkinson)
    """

    lookback: int = 60
    target_percentile: float = 0.3
    adjustment_strength: float = 0.3
    underweight_high_vol: bool = True
    min_observations: int = 20
    volatility_type: str = "std"

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if not 0 < self.target_percentile < 1:
            raise ValueError("target_percentile must be in (0, 1)")
        if not 0 <= self.adjustment_strength <= 1:
            raise ValueError("adjustment_strength must be in [0, 1]")
        if self.volatility_type not in ("std", "ewm", "parkinson"):
            raise ValueError("volatility_type must be 'std', 'ewm', or 'parkinson'")


@dataclass
class VolatilityRankResult:
    """ボラティリティランク結果

    Attributes:
        ranks: 銘柄別ランク（0=最低ボラ、1=最高ボラ）
        volatilities: 銘柄別ボラティリティ
        low_vol_assets: 低ボラ銘柄リスト
        high_vol_assets: 高ボラ銘柄リスト
        threshold_low: 低ボラ閾値
        threshold_high: 高ボラ閾値
        metadata: 追加情報
    """

    ranks: pd.Series
    volatilities: pd.Series
    low_vol_assets: List[str]
    high_vol_assets: List[str]
    threshold_low: float
    threshold_high: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ranks": self.ranks.to_dict(),
            "volatilities": self.volatilities.to_dict(),
            "low_vol_assets": self.low_vol_assets,
            "high_vol_assets": self.high_vol_assets,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "metadata": self.metadata,
        }


@dataclass
class LowVolAdjustmentResult:
    """低ボラ調整結果

    Attributes:
        adjustments: 銘柄別調整係数
        original_weights: 元の重み
        adjusted_weights: 調整後の重み
        overweight_total: オーバーウェイト合計
        underweight_total: アンダーウェイト合計
        metadata: 追加情報
    """

    adjustments: Dict[str, float]
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    overweight_total: float
    underweight_total: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "adjustments": self.adjustments,
            "original_weights": self.original_weights,
            "adjusted_weights": self.adjusted_weights,
            "overweight_total": self.overweight_total,
            "underweight_total": self.underweight_total,
            "metadata": self.metadata,
        }


class LowVolPremiumStrategy:
    """低ボラティリティプレミアム戦略

    低ボラティリティ銘柄のアウトパフォーマンス傾向を活用する。

    戦略ロジック:
    1. 過去N日間のリターンからボラティリティを計算
    2. ボラティリティでユニバースをランク付け
    3. 下位X%（低ボラ）をオーバーウェイト
    4. 上位X%（高ボラ）をアンダーウェイト

    Example:
        strategy = LowVolPremiumStrategy(lookback=60)

        # リターンデータ（columns=銘柄, index=日付）
        returns_df = pd.DataFrame(...)

        # ボラランク計算
        rank_result = strategy.calculate_volatility_rank(returns_df)

        # 調整係数
        adjustments = strategy.generate_low_vol_overweight(rank_result.ranks)

        # 配分に適用
        base_weights = {"AAPL": 0.2, "MSFT": 0.2, ...}
        result = strategy.apply_adjustment(base_weights, adjustments)
        print(result.adjusted_weights)
    """

    def __init__(
        self,
        config: LowVolPremiumConfig | None = None,
        lookback: int = 60,
        target_percentile: float = 0.3,
        adjustment_strength: float = 0.3,
    ) -> None:
        """初期化

        Args:
            config: 設定オブジェクト（優先）
            lookback: ボラティリティ計算のルックバック期間
            target_percentile: オーバーウェイト対象の下位パーセンタイル
            adjustment_strength: 調整の強さ
        """
        if config is not None:
            self.config = config
        else:
            self.config = LowVolPremiumConfig(
                lookback=lookback,
                target_percentile=target_percentile,
                adjustment_strength=adjustment_strength,
            )

    def calculate_volatility_rank(
        self,
        returns: pd.DataFrame,
        lookback: int | None = None,
    ) -> VolatilityRankResult:
        """ボラティリティでランク付け

        過去リターンのボラティリティを計算し、
        ボラティリティでランク付け（低→高: 0→1）。

        Args:
            returns: リターンデータ（columns=銘柄, index=日付）
            lookback: ルックバック期間（Noneでconfig値使用）

        Returns:
            VolatilityRankResult
        """
        lookback = lookback or self.config.lookback

        # ルックバック期間でフィルタ
        if len(returns) > lookback:
            returns = returns.tail(lookback)

        # ボラティリティ計算
        if self.config.volatility_type == "std":
            volatilities = returns.std()
        elif self.config.volatility_type == "ewm":
            volatilities = returns.ewm(span=lookback).std().iloc[-1]
        else:
            # Parkinson volatility (高値・安値が必要なので、ここではstdにフォールバック)
            volatilities = returns.std()

        # NaN除去
        volatilities = volatilities.dropna()

        if len(volatilities) == 0:
            logger.warning("No valid volatility data")
            return VolatilityRankResult(
                ranks=pd.Series(dtype=float),
                volatilities=pd.Series(dtype=float),
                low_vol_assets=[],
                high_vol_assets=[],
                threshold_low=0.0,
                threshold_high=0.0,
                metadata={"error": "No valid data"},
            )

        # パーセンタイルランク（0=最低ボラ、1=最高ボラ）
        ranks = volatilities.rank(pct=True)

        # 閾値計算
        threshold_low = self.config.target_percentile
        threshold_high = 1 - self.config.target_percentile

        # 低ボラ・高ボラ銘柄を特定
        low_vol_assets = list(ranks[ranks <= threshold_low].index)
        high_vol_assets = list(ranks[ranks >= threshold_high].index)

        logger.debug(
            "Volatility rank calculated: n_assets=%d, low_vol=%d, high_vol=%d",
            len(ranks),
            len(low_vol_assets),
            len(high_vol_assets),
        )

        return VolatilityRankResult(
            ranks=ranks,
            volatilities=volatilities,
            low_vol_assets=low_vol_assets,
            high_vol_assets=high_vol_assets,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            metadata={
                "lookback": lookback,
                "volatility_type": self.config.volatility_type,
                "n_assets": len(ranks),
                "vol_mean": float(volatilities.mean()),
                "vol_std": float(volatilities.std()),
            },
        )

    def generate_low_vol_overweight(
        self,
        vol_ranks: pd.Series | VolatilityRankResult,
        target_percentile: float | None = None,
    ) -> Dict[str, float]:
        """低ボラオーバーウェイト調整係数を生成

        下位X%（低ボラ）をオーバーウェイト、
        上位X%（高ボラ）をアンダーウェイトする調整係数を生成。

        Args:
            vol_ranks: ボラティリティランク（0-1）またはVolatilityRankResult
            target_percentile: オーバーウェイト対象のパーセンタイル

        Returns:
            銘柄別調整係数（正=オーバーウェイト、負=アンダーウェイト）
        """
        target_pct = target_percentile or self.config.target_percentile

        # VolatilityRankResultの場合はranksを抽出
        if isinstance(vol_ranks, VolatilityRankResult):
            ranks = vol_ranks.ranks
        else:
            ranks = vol_ranks

        adjustments: Dict[str, float] = {}

        for asset, rank in ranks.items():
            if rank <= target_pct:
                # 低ボラ: オーバーウェイト
                # rank=0 で最大オーバーウェイト、rank=target_pctで0
                adjustment = (target_pct - rank) / target_pct
                adjustments[str(asset)] = adjustment
            elif rank >= (1 - target_pct) and self.config.underweight_high_vol:
                # 高ボラ: アンダーウェイト
                # rank=1で最大アンダーウェイト、rank=1-target_pctで0
                adjustment = -((rank - (1 - target_pct)) / target_pct)
                adjustments[str(asset)] = adjustment
            else:
                # 中間: 調整なし
                adjustments[str(asset)] = 0.0

        logger.debug(
            "Generated adjustments: overweight=%d, underweight=%d",
            sum(1 for v in adjustments.values() if v > 0),
            sum(1 for v in adjustments.values() if v < 0),
        )

        return adjustments

    def apply_adjustment(
        self,
        base_weights: Dict[str, float],
        vol_adjustment: Dict[str, float],
        adjustment_strength: float | None = None,
    ) -> LowVolAdjustmentResult:
        """基本配分にボラティリティ調整を適用

        調整式:
            adjusted_weight = base_weight * (1 + strength * adjustment)

        その後、合計が1になるように正規化。

        Args:
            base_weights: 基本配分（合計1.0）
            vol_adjustment: 調整係数（-1〜+1）
            adjustment_strength: 調整の強さ（Noneでconfig値使用）

        Returns:
            LowVolAdjustmentResult
        """
        strength = adjustment_strength or self.config.adjustment_strength

        adjusted_weights: Dict[str, float] = {}
        overweight_total = 0.0
        underweight_total = 0.0

        for asset, base_weight in base_weights.items():
            adjustment = vol_adjustment.get(asset, 0.0)

            # 調整適用
            adjusted = base_weight * (1 + strength * adjustment)

            # 負の重みを防止
            adjusted = max(adjusted, 0.0)

            adjusted_weights[asset] = adjusted

            # 統計
            if adjustment > 0:
                overweight_total += base_weight * strength * adjustment
            elif adjustment < 0:
                underweight_total += abs(base_weight * strength * adjustment)

        # 正規化（合計1.0）
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        logger.info(
            "Low vol adjustment applied: strength=%.2f, overweight=%.3f, underweight=%.3f",
            strength,
            overweight_total,
            underweight_total,
        )

        return LowVolAdjustmentResult(
            adjustments=vol_adjustment,
            original_weights=base_weights,
            adjusted_weights=adjusted_weights,
            overweight_total=overweight_total,
            underweight_total=underweight_total,
            metadata={
                "adjustment_strength": strength,
                "n_overweight": sum(1 for v in vol_adjustment.values() if v > 0),
                "n_underweight": sum(1 for v in vol_adjustment.values() if v < 0),
            },
        )

    def process(
        self,
        returns: pd.DataFrame,
        base_weights: Dict[str, float],
    ) -> LowVolAdjustmentResult:
        """一括処理: ランク計算 → 調整生成 → 適用

        Args:
            returns: リターンデータ
            base_weights: 基本配分

        Returns:
            LowVolAdjustmentResult
        """
        # Step 1: ボラティリティランク
        rank_result = self.calculate_volatility_rank(returns)

        # Step 2: 調整係数生成
        adjustments = self.generate_low_vol_overweight(rank_result)

        # Step 3: 配分に適用
        result = self.apply_adjustment(base_weights, adjustments)

        # メタデータにランク情報を追加
        result.metadata["volatility_ranks"] = rank_result.metadata

        return result


# =============================================================================
# Signal クラスとしての実装（SignalRegistry互換）
# =============================================================================


@SignalRegistry.register(
    "low_vol_premium",
    category="factor",
    description="Low volatility premium signal - overweight low vol assets",
    tags=["volatility", "factor", "anomaly", "premium"],
)
class LowVolPremiumSignal(Signal):
    """低ボラティリティプレミアムシグナル

    低ボラティリティ銘柄を高スコア、高ボラティリティ銘柄を低スコアとして出力。

    Output interpretation:
    - +1: 最も低いボラティリティ（強いオーバーウェイト推奨）
    - 0: 平均的なボラティリティ
    - -1: 最も高いボラティリティ（アンダーウェイト推奨）
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback",
                default=60,
                searchable=True,
                min_value=20,
                max_value=120,
                step=10,
                description="Volatility calculation lookback period",
            ),
            ParameterSpec(
                name="target_percentile",
                default=0.3,
                searchable=True,
                min_value=0.1,
                max_value=0.5,
                step=0.1,
                description="Target percentile for overweight",
            ),
        ]

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """シグナルを計算

        Args:
            data: DataFrame with 'close' column

        Returns:
            SignalResult with low vol premium scores
        """
        self.validate_input(data)

        if "close" not in data.columns:
            raise ValueError("Missing required column: close")

        lookback = self._params["lookback"]
        target_pct = self._params["target_percentile"]

        # リターン計算
        returns = data["close"].pct_change().dropna()

        if len(returns) < lookback:
            logger.warning(
                "Insufficient data for low vol premium: %d < %d",
                len(returns),
                lookback,
            )
            return SignalResult(
                scores=pd.Series(0.0, index=data.index),
                metadata={"error": "Insufficient data"},
            )

        # ボラティリティ計算（ルックバック期間）
        rolling_vol = returns.rolling(lookback).std()

        # 最新のボラティリティを取得
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.nan

        if pd.isna(current_vol):
            return SignalResult(
                scores=pd.Series(0.0, index=data.index),
                metadata={"error": "Invalid volatility"},
            )

        # ボラティリティの長期分布からランク化
        vol_rank = rolling_vol.rank(pct=True)

        # スコア変換: 低ボラ=+1、高ボラ=-1
        # rank 0 -> score +1, rank 1 -> score -1
        scores = 1 - 2 * vol_rank

        # 端のパーセンタイルを強調
        scores = scores.apply(
            lambda x: np.clip(x * (1 / target_pct), -1, 1) if not pd.isna(x) else 0.0
        )

        return SignalResult(
            scores=scores,
            metadata={
                "lookback": lookback,
                "target_percentile": target_pct,
                "current_vol": float(current_vol) if not pd.isna(current_vol) else None,
                "vol_mean": float(rolling_vol.mean()),
                "vol_std": float(rolling_vol.std()),
            },
        )


# =============================================================================
# 便利関数
# =============================================================================


def calculate_volatility_rank(
    returns: pd.DataFrame,
    lookback: int = 60,
) -> VolatilityRankResult:
    """ボラティリティランクを計算（ショートカット関数）

    Args:
        returns: リターンデータ
        lookback: ルックバック期間

    Returns:
        VolatilityRankResult
    """
    strategy = LowVolPremiumStrategy(lookback=lookback)
    return strategy.calculate_volatility_rank(returns)


def apply_low_vol_premium(
    returns: pd.DataFrame,
    base_weights: Dict[str, float],
    lookback: int = 60,
    target_percentile: float = 0.3,
    adjustment_strength: float = 0.3,
) -> Dict[str, float]:
    """低ボラプレミアム調整を適用（ショートカット関数）

    Args:
        returns: リターンデータ
        base_weights: 基本配分
        lookback: ルックバック期間
        target_percentile: オーバーウェイト対象パーセンタイル
        adjustment_strength: 調整の強さ

    Returns:
        調整後の重み
    """
    strategy = LowVolPremiumStrategy(
        lookback=lookback,
        target_percentile=target_percentile,
        adjustment_strength=adjustment_strength,
    )
    result = strategy.process(returns, base_weights)
    return result.adjusted_weights
