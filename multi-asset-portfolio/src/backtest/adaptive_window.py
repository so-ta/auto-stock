"""
Adaptive Window Size Module - 適応的ウィンドウサイズ

市場レジームに応じて最適なルックバックウィンドウサイズを決定する。
高ボラティリティ時は短いウィンドウ、安定期は長いウィンドウを使用。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """ボラティリティレジーム"""
    CRISIS = "crisis"
    HIGH_VOL = "high_vol"
    NORMAL = "normal"
    LOW_VOL = "low_vol"


@dataclass
class WindowResult:
    """ウィンドウサイズ計算結果"""
    optimal_window: int
    base_window: int
    regime_multiplier: float
    regime_change_multiplier: float
    volatility_regime: VolatilityRegime
    regime_change_detected: bool


@dataclass
class RegimeChangeResult:
    """レジーム変化検出結果"""
    detected: bool
    recent_vol: float
    long_vol: float
    ratio: float
    threshold: float


@dataclass
class WindowSplit:
    """ウィンドウ分割情報"""
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def __iter__(self):
        """タプルとしてイテレート可能にする"""
        return iter((self.train_start, self.train_end, self.test_start, self.test_end))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """タプルに変換"""
        return (self.train_start, self.train_end, self.test_start, self.test_end)


class AdaptiveWindowSelector:
    """
    適応的ウィンドウサイズ選択器

    市場のボラティリティレジームに応じてルックバックウィンドウサイズを
    動的に調整する。危機時は短いウィンドウで最新の市場状態を重視し、
    安定期は長いウィンドウで統計的な安定性を確保する。
    """

    # レジーム別乗数（要件に準拠）
    REGIME_MULTIPLIERS: Dict[VolatilityRegime, float] = {
        VolatilityRegime.CRISIS: 0.5,
        VolatilityRegime.HIGH_VOL: 0.7,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.LOW_VOL: 1.3,
    }

    # レジーム変化時の追加乗数
    REGIME_CHANGE_MULTIPLIER: float = 0.6

    def __init__(
        self,
        min_window: int = 126,
        max_window: int = 756,
        default_window: int = 504,
    ):
        """
        初期化

        Parameters
        ----------
        min_window : int
            最小ウィンドウサイズ（約半年: 126営業日）
        max_window : int
            最大ウィンドウサイズ（約3年: 756営業日）
        default_window : int
            デフォルトウィンドウサイズ（約2年: 504営業日）
        """
        if min_window <= 0:
            raise ValueError("min_window must be positive")
        if max_window < min_window:
            raise ValueError("max_window must be >= min_window")
        if not (min_window <= default_window <= max_window):
            raise ValueError("default_window must be between min_window and max_window")

        self.min_window = min_window
        self.max_window = max_window
        self.default_window = default_window

    def compute_optimal_window(
        self,
        returns: Optional[pd.Series] = None,
        volatility_regime: Optional[VolatilityRegime] = None,
        regime_change_detected: bool = False,
    ) -> int:
        """
        最適なウィンドウサイズを計算

        Parameters
        ----------
        returns : pd.Series, optional
            リターン系列（レジーム変化検出に使用）
        volatility_regime : VolatilityRegime, optional
            現在のボラティリティレジーム
        regime_change_detected : bool
            レジーム変化が検出されたか

        Returns
        -------
        int
            最適なウィンドウサイズ
        """
        result = self.compute_optimal_window_detailed(
            returns=returns,
            volatility_regime=volatility_regime,
            regime_change_detected=regime_change_detected,
        )
        return result.optimal_window

    def compute_optimal_window_detailed(
        self,
        returns: Optional[pd.Series] = None,
        volatility_regime: Optional[VolatilityRegime] = None,
        regime_change_detected: bool = False,
    ) -> WindowResult:
        """
        最適なウィンドウサイズを詳細に計算

        Parameters
        ----------
        returns : pd.Series, optional
            リターン系列（レジーム変化検出に使用）
        volatility_regime : VolatilityRegime, optional
            現在のボラティリティレジーム
        regime_change_detected : bool
            レジーム変化が検出されたか（引数で指定した場合）

        Returns
        -------
        WindowResult
            計算結果の詳細
        """
        # デフォルトレジームを設定
        if volatility_regime is None:
            volatility_regime = VolatilityRegime.NORMAL

        # レジーム変化検出（returnsが提供され、明示的に検出されていない場合）
        if returns is not None and not regime_change_detected:
            regime_change_result = self.detect_regime_change(returns)
            regime_change_detected = regime_change_result.detected

        # レジーム乗数を取得
        regime_multiplier = self.REGIME_MULTIPLIERS[volatility_regime]

        # レジーム変化時の乗数
        regime_change_mult = self.REGIME_CHANGE_MULTIPLIER if regime_change_detected else 1.0

        # ウィンドウサイズを計算
        adjusted_window = self.default_window * regime_multiplier * regime_change_mult

        # 制約を適用
        optimal_window = int(np.clip(adjusted_window, self.min_window, self.max_window))

        return WindowResult(
            optimal_window=optimal_window,
            base_window=self.default_window,
            regime_multiplier=regime_multiplier,
            regime_change_multiplier=regime_change_mult,
            volatility_regime=volatility_regime,
            regime_change_detected=regime_change_detected,
        )

    def detect_regime_change(
        self,
        returns: pd.Series,
        lookback: int = 60,
        threshold: float = 2.0,
    ) -> RegimeChangeResult:
        """
        レジーム変化を検出

        直近のボラティリティと長期ボラティリティの比率が閾値を超えた場合、
        レジーム変化と判定する。

        Parameters
        ----------
        returns : pd.Series
            リターン系列
        lookback : int
            直近ボラティリティ計算用のルックバック期間
        threshold : float
            レジーム変化判定の閾値

        Returns
        -------
        RegimeChangeResult
            検出結果
        """
        if len(returns) < lookback * 2:
            # データ不足の場合は変化なしとする
            return RegimeChangeResult(
                detected=False,
                recent_vol=np.nan,
                long_vol=np.nan,
                ratio=np.nan,
                threshold=threshold,
            )

        # 直近のボラティリティ
        recent_vol = returns.iloc[-lookback:].std()

        # 長期ボラティリティ（全期間）
        long_vol = returns.std()

        # ゼロ除算を防ぐ
        if long_vol == 0 or np.isnan(long_vol):
            return RegimeChangeResult(
                detected=False,
                recent_vol=recent_vol,
                long_vol=long_vol,
                ratio=np.nan,
                threshold=threshold,
            )

        # 比率を計算
        ratio = recent_vol / long_vol

        # 閾値を超えた場合はレジーム変化と判定
        detected = ratio > threshold

        return RegimeChangeResult(
            detected=detected,
            recent_vol=recent_vol,
            long_vol=long_vol,
            ratio=ratio,
            threshold=threshold,
        )

    def classify_regime(
        self,
        current_vol: float,
        vol_percentile: float,
    ) -> VolatilityRegime:
        """
        ボラティリティパーセンタイルからレジームを分類

        Parameters
        ----------
        current_vol : float
            現在のボラティリティ
        vol_percentile : float
            ボラティリティのパーセンタイル（0-100）

        Returns
        -------
        VolatilityRegime
            分類されたレジーム
        """
        if vol_percentile >= 90:
            return VolatilityRegime.CRISIS
        elif vol_percentile >= 70:
            return VolatilityRegime.HIGH_VOL
        elif vol_percentile >= 30:
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.LOW_VOL


class ExpandingWindowOptimizer:
    """
    Expanding Window 最適化器

    時系列交差検証用のExpanding Window分割を生成する。
    訓練データは時間とともに拡大し、テストデータは固定サイズで前進する。
    """

    def __init__(
        self,
        min_train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
    ):
        """
        初期化

        Parameters
        ----------
        min_train_size : int
            最小訓練データサイズ（約1年: 252営業日）
        test_size : int
            テストデータサイズ（約3ヶ月: 63営業日）
        step_size : int
            ステップサイズ（約1ヶ月: 21営業日）
        """
        if min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size

    def generate_splits(
        self,
        n_samples: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Expanding Windowの分割を生成

        Parameters
        ----------
        n_samples : int
            総サンプル数

        Returns
        -------
        List[Tuple[int, int, int, int]]
            分割のリスト。各タプルは (train_start, train_end, test_start, test_end)
            インデックスは0始まりで、endは排他的（Pythonスライス形式）
        """
        splits = []

        # 最小データ要件をチェック
        min_required = self.min_train_size + self.test_size
        if n_samples < min_required:
            return splits

        # 訓練データの終了位置から開始
        train_end = self.min_train_size

        while train_end + self.test_size <= n_samples:
            train_start = 0  # Expanding window: 常に最初から
            test_start = train_end
            test_end = train_end + self.test_size

            splits.append((train_start, train_end, test_start, test_end))

            # 次のウィンドウへ
            train_end += self.step_size

        return splits

    def generate_splits_detailed(
        self,
        n_samples: int,
    ) -> List[WindowSplit]:
        """
        Expanding Windowの分割を詳細オブジェクトで生成

        Parameters
        ----------
        n_samples : int
            総サンプル数

        Returns
        -------
        List[WindowSplit]
            分割のリスト
        """
        tuples = self.generate_splits(n_samples)
        return [
            WindowSplit(
                train_start=t[0],
                train_end=t[1],
                test_start=t[2],
                test_end=t[3],
            )
            for t in tuples
        ]

    def get_split_count(self, n_samples: int) -> int:
        """
        生成される分割数を計算

        Parameters
        ----------
        n_samples : int
            総サンプル数

        Returns
        -------
        int
            分割数
        """
        min_required = self.min_train_size + self.test_size
        if n_samples < min_required:
            return 0

        available = n_samples - self.min_train_size - self.test_size
        return 1 + (available // self.step_size)


# ショートカット関数
def compute_adaptive_window(
    returns: pd.Series,
    volatility_regime: Optional[VolatilityRegime] = None,
    min_window: int = 126,
    max_window: int = 756,
    default_window: int = 504,
) -> int:
    """
    適応的ウィンドウサイズを計算するショートカット関数

    Parameters
    ----------
    returns : pd.Series
        リターン系列
    volatility_regime : VolatilityRegime, optional
        現在のボラティリティレジーム
    min_window : int
        最小ウィンドウサイズ
    max_window : int
        最大ウィンドウサイズ
    default_window : int
        デフォルトウィンドウサイズ

    Returns
    -------
    int
        最適なウィンドウサイズ
    """
    selector = AdaptiveWindowSelector(
        min_window=min_window,
        max_window=max_window,
        default_window=default_window,
    )
    return selector.compute_optimal_window(
        returns=returns,
        volatility_regime=volatility_regime,
    )


def generate_expanding_splits(
    n_samples: int,
    min_train_size: int = 252,
    test_size: int = 63,
    step_size: int = 21,
) -> List[Tuple[int, int, int, int]]:
    """
    Expanding Window分割を生成するショートカット関数

    Parameters
    ----------
    n_samples : int
        総サンプル数
    min_train_size : int
        最小訓練データサイズ
    test_size : int
        テストデータサイズ
    step_size : int
        ステップサイズ

    Returns
    -------
    List[Tuple[int, int, int, int]]
        分割のリスト
    """
    optimizer = ExpandingWindowOptimizer(
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    return optimizer.generate_splits(n_samples)
