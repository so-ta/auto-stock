"""
Strategy Base Module - 戦略基底クラス

Signalを用いたポジション提案生成器の抽象基底クラスを提供する。

設計根拠:
- 要求.md §4: Signal/Strategyの定義
- 要求.md §1: Strategy = Signalを用いたポジション提案の生成器
- 出力は [-1, +1] または [0, 1] に正規化

主要概念:
- Strategy: 1つ以上のSignalを組み合わせてポジション提案を生成
- PositionProposal: 時系列のポジション提案（-1=フルショート, +1=フルロング）
- StrategyConfig: 戦略のパラメータ設定
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


class PositionMode(Enum):
    """ポジションモード

    LONG_SHORT: [-1, +1] ショートからロングまで
    LONG_ONLY: [0, 1] ロングのみ
    """

    LONG_SHORT = "long_short"
    LONG_ONLY = "long_only"


@dataclass(frozen=True)
class StrategyConfig:
    """戦略設定

    Attributes:
        name: 戦略名
        position_mode: ポジションモード
        max_position: 最大ポジションサイズ（絶対値）
        min_position: 最小ポジションサイズ（絶対値）
        signal_threshold: シグナル閾値（この値以下は0ポジション）
        smoothing_factor: ポジション変更のスムージング係数（0-1）
        position_sizing: ポジションサイジング方法
    """

    name: str = "base_strategy"
    position_mode: PositionMode = PositionMode.LONG_SHORT
    max_position: float = 1.0
    min_position: float = 0.0
    signal_threshold: float = 0.0
    smoothing_factor: float = 0.0
    position_sizing: str = "signal_proportional"

    def __post_init__(self) -> None:
        """設定の検証"""
        if not 0.0 <= self.max_position <= 1.0:
            raise ValueError("max_position must be in [0, 1]")
        if not 0.0 <= self.min_position <= self.max_position:
            raise ValueError("min_position must be in [0, max_position]")
        if not 0.0 <= self.smoothing_factor <= 1.0:
            raise ValueError("smoothing_factor must be in [0, 1]")


@dataclass
class PositionProposal:
    """ポジション提案コンテナ

    Attributes:
        positions: ポジション提案の時系列（-1〜+1 or 0〜1）
        timestamps: 対応するタイムスタンプ
        raw_signal: 正規化前の生シグナル値
        metadata: 追加メタデータ
    """

    positions: pd.Series
    timestamps: pd.DatetimeIndex
    raw_signal: Optional[pd.Series] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """検証"""
        if len(self.positions) != len(self.timestamps):
            raise ValueError(
                f"Length mismatch: positions={len(self.positions)}, "
                f"timestamps={len(self.timestamps)}"
            )

    @property
    def is_valid(self) -> bool:
        """有効なポジション提案かどうか"""
        return not self.positions.isna().all()

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "positions": self.positions.to_dict(),
            "timestamps": [str(ts) for ts in self.timestamps],
            "metadata": self.metadata,
        }

    def apply_smoothing(self, factor: float) -> "PositionProposal":
        """ポジション変更をスムージング

        Args:
            factor: スムージング係数（0-1）。0=変更なし、1=完全平滑化

        Returns:
            スムージング適用後のPositionProposal
        """
        if factor <= 0:
            return self

        smoothed = self.positions.ewm(alpha=1 - factor, adjust=False).mean()
        return PositionProposal(
            positions=smoothed,
            timestamps=self.timestamps,
            raw_signal=self.raw_signal,
            metadata={**self.metadata, "smoothing_applied": factor},
        )

    def clip_to_mode(self, mode: PositionMode) -> "PositionProposal":
        """ポジションモードに応じてクリップ

        Args:
            mode: ポジションモード

        Returns:
            クリップ後のPositionProposal
        """
        if mode == PositionMode.LONG_ONLY:
            clipped = self.positions.clip(0, 1)
        else:
            clipped = self.positions.clip(-1, 1)

        return PositionProposal(
            positions=clipped,
            timestamps=self.timestamps,
            raw_signal=self.raw_signal,
            metadata={**self.metadata, "clipped_to": mode.value},
        )


class StrategyRegistry:
    """戦略レジストリ

    プラグインアーキテクチャで戦略を登録・検索可能にする。
    """

    _strategies: ClassVar[Dict[str, Type["Strategy"]]] = {}
    _metadata: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        category: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """戦略を登録するデコレータ

        Args:
            name: 戦略の一意識別子
            category: カテゴリ（momentum, mean_reversion等）
            description: 説明文
            tags: タグリスト

        Example:
            @StrategyRegistry.register("momentum_crossover", category="momentum")
            class MomentumCrossoverStrategy(Strategy):
                ...
        """

        def decorator(strategy_cls: Type["Strategy"]) -> Type["Strategy"]:
            if name in cls._strategies:
                raise ValueError(
                    f"Strategy '{name}' is already registered: "
                    f"{cls._strategies[name].__name__}"
                )

            if not issubclass(strategy_cls, Strategy):
                raise TypeError(
                    f"Class {strategy_cls.__name__} must be subclass of Strategy"
                )

            cls._strategies[name] = strategy_cls
            cls._metadata[name] = {
                "category": category,
                "description": description or strategy_cls.__doc__,
                "tags": tags or [],
                "class_name": strategy_cls.__name__,
            }

            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type["Strategy"]:
        """戦略クラスを名前で取得"""
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> "Strategy":
        """戦略インスタンスを名前で作成"""
        strategy_cls = cls.get(name)
        return strategy_cls(**kwargs)

    @classmethod
    def list_all(cls) -> List[str]:
        """登録済み戦略名一覧"""
        return sorted(cls._strategies.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """カテゴリで絞り込み"""
        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("category") == category
        ]

    @classmethod
    def clear(cls) -> None:
        """全登録をクリア（テスト用）"""
        cls._strategies.clear()
        cls._metadata.clear()


class Strategy(ABC):
    """戦略抽象基底クラス

    1つ以上のSignalを組み合わせてポジション提案を生成する。

    設計原則:
    1. 純関数設計: 入力（過去データ）→出力（ポジション提案）
    2. 出力正規化: [-1, +1] (Long/Short) または [0, 1] (Long Only)
    3. データリーク防止: 未来データを参照しない
    4. パラメータ分離: 探索対象と固定を明確化

    Example:
        class MyStrategy(Strategy):
            def __init__(self, lookback: int = 20):
                super().__init__()
                self.lookback = lookback
                self.signal = MomentumReturnSignal(lookback=lookback)

            def generate_positions(self, data: pd.DataFrame) -> PositionProposal:
                signal_result = self.signal.compute(data)
                positions = self._signal_to_position(signal_result.scores)
                return PositionProposal(
                    positions=positions,
                    timestamps=data.index,
                )
    """

    name: str = "base_strategy"

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        **kwargs: Any,
    ) -> None:
        """戦略を初期化

        Args:
            config: 戦略設定（Noneの場合はデフォルト）
            **kwargs: サブクラス固有パラメータ
        """
        self.config = config or StrategyConfig()
        self._signals: List[Signal] = []
        self._params = kwargs
        self._fitted = False

    def add_signal(self, signal: Signal) -> None:
        """シグナルを追加

        Args:
            signal: 追加するSignalインスタンス
        """
        self._signals.append(signal)

    @property
    def signals(self) -> List[Signal]:
        """登録済みシグナル一覧"""
        return self._signals.copy()

    @abstractmethod
    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> PositionProposal:
        """ポジション提案を生成

        サブクラスで実装必須。Signalを使ってポジション提案を計算する。

        Args:
            data: OHLCV DataFrame (DatetimeIndex必須)

        Returns:
            PositionProposal with normalized positions
        """
        pass

    def __call__(
        self,
        data: pd.DataFrame,
    ) -> PositionProposal:
        """generate_positionsのショートカット"""
        return self.generate_positions(data)

    def _signal_to_position(
        self,
        signal_scores: pd.Series,
    ) -> pd.Series:
        """シグナルスコアをポジションに変換

        Args:
            signal_scores: シグナルスコア（-1〜+1）

        Returns:
            ポジション（設定に応じて正規化）
        """
        config = self.config

        # 閾値以下は0ポジション
        positions = signal_scores.copy()
        positions = positions.where(positions.abs() > config.signal_threshold, 0.0)

        # ポジションサイジング
        if config.position_sizing == "signal_proportional":
            # シグナル比例
            positions = positions * config.max_position
        elif config.position_sizing == "binary":
            # バイナリ（符号のみ）
            positions = np.sign(positions) * config.max_position
        elif config.position_sizing == "scaled_binary":
            # スケールドバイナリ
            positions = np.where(
                positions.abs() > config.signal_threshold,
                np.sign(positions) * config.max_position,
                0.0,
            )
            positions = pd.Series(positions, index=signal_scores.index)

        # ポジションモードに応じてクリップ
        if config.position_mode == PositionMode.LONG_ONLY:
            positions = positions.clip(0, config.max_position)
        else:
            positions = positions.clip(-config.max_position, config.max_position)

        # 最小ポジション適用（絶対値がmin_position未満は0に）
        if config.min_position > 0:
            positions = positions.where(
                positions.abs() >= config.min_position, 0.0
            )

        return positions

    def _combine_signals(
        self,
        signal_results: List[SignalResult],
        weights: Optional[List[float]] = None,
    ) -> pd.Series:
        """複数シグナルを統合

        Args:
            signal_results: SignalResultのリスト
            weights: 各シグナルの重み（Noneの場合は均等）

        Returns:
            統合されたシグナルスコア
        """
        if not signal_results:
            raise ValueError("No signals to combine")

        if weights is None:
            weights = [1.0 / len(signal_results)] * len(signal_results)

        if len(weights) != len(signal_results):
            raise ValueError("Weights length must match signals length")

        # 重みを正規化
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 加重平均
        combined = sum(
            result.scores * weight
            for result, weight in zip(signal_results, weights)
        )

        # 出力を[-1, +1]にクリップ
        return combined.clip(-1, 1)

    def validate_data(self, data: pd.DataFrame) -> None:
        """入力データを検証

        Args:
            data: 入力DataFrame

        Raises:
            ValueError: 検証失敗時
        """
        if data is None or data.empty:
            raise ValueError("Input data is empty")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        required_cols = ["close"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_config(self) -> Dict[str, Any]:
        """設定を辞書で取得（再現性用）"""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "config": {
                "position_mode": self.config.position_mode.value,
                "max_position": self.config.max_position,
                "min_position": self.config.min_position,
                "signal_threshold": self.config.signal_threshold,
                "smoothing_factor": self.config.smoothing_factor,
                "position_sizing": self.config.position_sizing,
            },
            "params": self._params,
            "signals": [signal.get_config() for signal in self._signals],
        }

    def __repr__(self) -> str:
        signals_str = ", ".join(s.__class__.__name__ for s in self._signals)
        return (
            f"{self.__class__.__name__}("
            f"mode={self.config.position_mode.value}, "
            f"signals=[{signals_str}])"
        )


class SingleSignalStrategy(Strategy):
    """単一シグナル戦略

    1つのSignalをそのままポジション提案に変換する最もシンプルな戦略。
    """

    name = "single_signal"

    def __init__(
        self,
        signal: Signal,
        config: Optional[StrategyConfig] = None,
        **kwargs: Any,
    ) -> None:
        """単一シグナル戦略を初期化

        Args:
            signal: 使用するSignalインスタンス
            config: 戦略設定
            **kwargs: 追加パラメータ
        """
        super().__init__(config=config, **kwargs)
        self.signal = signal
        self.add_signal(signal)

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> PositionProposal:
        """ポジション提案を生成

        Args:
            data: OHLCV DataFrame

        Returns:
            PositionProposal
        """
        self.validate_data(data)

        # シグナル計算
        signal_result = self.signal.compute(data)

        # ポジションに変換
        positions = self._signal_to_position(signal_result.scores)

        # スムージング適用
        proposal = PositionProposal(
            positions=positions,
            timestamps=data.index,
            raw_signal=signal_result.scores,
            metadata={
                "strategy": self.name,
                "signal": self.signal.name,
                "signal_metadata": signal_result.metadata,
            },
        )

        if self.config.smoothing_factor > 0:
            proposal = proposal.apply_smoothing(self.config.smoothing_factor)

        return proposal.clip_to_mode(self.config.position_mode)


class CompositeStrategy(Strategy):
    """複合戦略

    複数のSignalを組み合わせてポジション提案を生成する。
    """

    name = "composite"

    def __init__(
        self,
        signals: List[Signal],
        weights: Optional[List[float]] = None,
        config: Optional[StrategyConfig] = None,
        **kwargs: Any,
    ) -> None:
        """複合戦略を初期化

        Args:
            signals: 使用するSignalインスタンスのリスト
            weights: 各シグナルの重み（Noneの場合は均等）
            config: 戦略設定
            **kwargs: 追加パラメータ
        """
        super().__init__(config=config, **kwargs)

        if not signals:
            raise ValueError("At least one signal is required")

        self._signal_weights = weights
        for signal in signals:
            self.add_signal(signal)

    @property
    def weights(self) -> List[float]:
        """正規化された重み"""
        if self._signal_weights is None:
            n = len(self._signals)
            return [1.0 / n] * n

        total = sum(self._signal_weights)
        return [w / total for w in self._signal_weights]

    def generate_positions(
        self,
        data: pd.DataFrame,
    ) -> PositionProposal:
        """ポジション提案を生成

        Args:
            data: OHLCV DataFrame

        Returns:
            PositionProposal
        """
        self.validate_data(data)

        # 各シグナルを計算
        signal_results = [signal.compute(data) for signal in self._signals]

        # シグナルを統合
        combined_scores = self._combine_signals(signal_results, self._signal_weights)

        # ポジションに変換
        positions = self._signal_to_position(combined_scores)

        # スムージング適用
        proposal = PositionProposal(
            positions=positions,
            timestamps=data.index,
            raw_signal=combined_scores,
            metadata={
                "strategy": self.name,
                "signals": [s.name for s in self._signals],
                "weights": self.weights,
            },
        )

        if self.config.smoothing_factor > 0:
            proposal = proposal.apply_smoothing(self.config.smoothing_factor)

        return proposal.clip_to_mode(self.config.position_mode)
