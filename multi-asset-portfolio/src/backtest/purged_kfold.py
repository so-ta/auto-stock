"""
Purged K-Fold Cross-Validation - データリーク防止CV

時系列データ用のクロスバリデーション。
テスト前後にpurge期間を設けてデータリークを防止。

主要コンポーネント:
- PurgedKFold: 基本的なPurged K-Fold CV
- CombinatorialPurgedCV: 組み合わせベースのPurged CV

設計根拠:
- 時系列データでは未来の情報が過去に漏れる可能性がある
- 例: ラベルに未来情報が含まれる場合、テスト直前のデータは汚染される
- Purge: テスト前のデータを除外して汚染を防止
- Embargo: テスト後のデータを除外して逆方向の漏れを防止

参考文献:
- López de Prado, M. (2018). Advances in Financial Machine Learning

使用例:
    from src.backtest.purged_kfold import PurgedKFold, CombinatorialPurgedCV

    # 基本的なPurged K-Fold
    cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(data):
        # 学習と評価
        pass

    # 組み合わせCV
    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)
    for train_idx, test_idx in cpcv.split(data):
        # より多くの分割パターンで評価
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class PurgedKFoldConfig:
    """PurgedKFold設定

    Attributes:
        n_splits: 分割数
        purge_gap: Purge期間（サンプル数）
        embargo_pct: Embargo比率（テストサイズに対する比率）
        embargo_samples: Embargo期間（サンプル数、embargo_pctより優先）
    """
    n_splits: int = 5
    purge_gap: int = 5
    embargo_pct: float = 0.01
    embargo_samples: int | None = None

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        if not 0 <= self.embargo_pct <= 1:
            raise ValueError("embargo_pct must be in [0, 1]")


@dataclass
class FoldInfo:
    """フォールド情報

    Attributes:
        fold_id: フォールドID
        train_indices: 学習データのインデックス
        test_indices: テストデータのインデックス
        purge_indices: Purge期間のインデックス
        embargo_indices: Embargo期間のインデックス
        train_size: 学習データサイズ
        test_size: テストデータサイズ
    """
    fold_id: int
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    purge_indices: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))
    embargo_indices: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))

    @property
    def train_size(self) -> int:
        """学習データサイズ"""
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        """テストデータサイズ"""
        return len(self.test_indices)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "fold_id": self.fold_id,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "purge_size": len(self.purge_indices),
            "embargo_size": len(self.embargo_indices),
        }


@dataclass
class CVSummary:
    """CV結果のサマリー

    Attributes:
        n_splits: 分割数
        n_samples: 総サンプル数
        folds: フォールド情報リスト
        avg_train_size: 平均学習データサイズ
        avg_test_size: 平均テストデータサイズ
    """
    n_splits: int
    n_samples: int
    folds: list[FoldInfo]

    @property
    def avg_train_size(self) -> float:
        """平均学習データサイズ"""
        return np.mean([f.train_size for f in self.folds])

    @property
    def avg_test_size(self) -> float:
        """平均テストデータサイズ"""
        return np.mean([f.test_size for f in self.folds])

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "n_splits": self.n_splits,
            "n_samples": self.n_samples,
            "avg_train_size": self.avg_train_size,
            "avg_test_size": self.avg_test_size,
            "folds": [f.to_dict() for f in self.folds],
        }


# =============================================================================
# Purged K-Fold
# =============================================================================

class PurgedKFold:
    """Purged K-Fold クロスバリデーション

    時系列データ用のK-Fold CV。データリークを防ぐため：
    - テスト前にpurge期間を設けて汚染データを除外
    - テスト後にembargo期間を設けて逆方向の漏れを防止

    アルゴリズム:
    1. データをn_splits個のfoldに分割
    2. 各foldをテストセットとして使用
    3. テスト前: train_before_end = test_start - purge_gap
    4. テスト後: train_after_start = test_end + embargo

    Usage:
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            # モデル学習と評価
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01,
        config: PurgedKFoldConfig | None = None,
    ) -> None:
        """初期化

        Args:
            n_splits: 分割数
            purge_gap: Purge期間（サンプル数）
            embargo_pct: Embargo比率
            config: 設定（指定時は他の引数は無視）
        """
        if config is not None:
            self.config = config
        else:
            self.config = PurgedKFoldConfig(
                n_splits=n_splits,
                purge_gap=purge_gap,
                embargo_pct=embargo_pct,
            )

        self._fold_infos: list[FoldInfo] = []

    @property
    def n_splits(self) -> int:
        """分割数"""
        return self.config.n_splits

    def split(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        groups: NDArray[np.int64] | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """データを分割

        Args:
            X: 特徴量データ
            y: ターゲット（未使用、sklearn互換のため）
            groups: グループ（未使用、sklearn互換のため）

        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)

        if n_samples < self.config.n_splits:
            raise ValueError(f"Cannot split {n_samples} samples into {self.config.n_splits} folds")

        # Fold境界を計算
        fold_sizes = np.full(self.config.n_splits, n_samples // self.config.n_splits, dtype=int)
        fold_sizes[:n_samples % self.config.n_splits] += 1

        fold_starts = np.cumsum(np.r_[0, fold_sizes])

        # Embargoサンプル数を計算
        if self.config.embargo_samples is not None:
            embargo = self.config.embargo_samples
        else:
            avg_test_size = n_samples // self.config.n_splits
            embargo = int(np.ceil(avg_test_size * self.config.embargo_pct))

        self._fold_infos = []

        for fold_id in range(self.config.n_splits):
            # テストセット
            test_start = fold_starts[fold_id]
            test_end = fold_starts[fold_id + 1]
            test_indices = np.arange(test_start, test_end)

            # Purge期間（テスト直前）
            purge_start = max(0, test_start - self.config.purge_gap)
            purge_end = test_start
            purge_indices = np.arange(purge_start, purge_end)

            # Embargo期間（テスト直後）
            embargo_start = test_end
            embargo_end = min(n_samples, test_end + embargo)
            embargo_indices = np.arange(embargo_start, embargo_end)

            # 学習セット（テスト、purge、embargoを除く）
            excluded = set(test_indices.tolist()) | set(purge_indices.tolist()) | set(embargo_indices.tolist())
            train_indices = np.array([i for i in range(n_samples) if i not in excluded], dtype=np.int64)

            # フォールド情報を保存
            fold_info = FoldInfo(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices,
            )
            self._fold_infos.append(fold_info)

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: pd.DataFrame | NDArray[np.float64] | None = None,
        y: NDArray[np.float64] | None = None,
        groups: NDArray[np.int64] | None = None,
    ) -> int:
        """分割数を取得（sklearn互換）

        Returns:
            分割数
        """
        return self.config.n_splits

    def get_fold_info(self, fold_id: int) -> FoldInfo | None:
        """フォールド情報を取得

        Args:
            fold_id: フォールドID

        Returns:
            FoldInfo or None
        """
        if 0 <= fold_id < len(self._fold_infos):
            return self._fold_infos[fold_id]
        return None

    def get_summary(self, n_samples: int | None = None) -> CVSummary:
        """サマリーを取得

        Args:
            n_samples: 総サンプル数

        Returns:
            CVSummary
        """
        return CVSummary(
            n_splits=self.config.n_splits,
            n_samples=n_samples or (sum(len(f.train_indices) + len(f.test_indices) for f in self._fold_infos) // self.config.n_splits if self._fold_infos else 0),
            folds=self._fold_infos.copy(),
        )


# =============================================================================
# Combinatorial Purged CV
# =============================================================================

class CombinatorialPurgedCV:
    """Combinatorial Purged クロスバリデーション

    複数の分割パターンを組み合わせたCV。
    n_splits個のfoldからn_test_splits個を選んでテストとする全組み合わせを生成。

    利点:
    - より多くのバックテストパスを生成
    - パラメータの過学習を検出しやすい
    - 統計的に堅牢な評価が可能

    アルゴリズム:
    1. データをn_splits個のfoldに分割
    2. itertools.combinationsでn_test_splits個のfoldをテストとする組み合わせを生成
    3. 隣接foldにはpurgeを適用

    Usage:
        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)

        for train_idx, test_idx in cpcv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            # モデル学習と評価

        print(f"Total combinations: {cpcv.n_combinations}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        """初期化

        Args:
            n_splits: 分割数
            n_test_splits: テストに使用するfold数
            purge_gap: Purge期間（サンプル数）
            embargo_pct: Embargo比率
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be in [1, n_splits)")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

        # 組み合わせ数
        self.n_combinations = len(list(combinations(range(n_splits), n_test_splits)))

        self._fold_infos: list[FoldInfo] = []

    def split(
        self,
        X: pd.DataFrame | NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        groups: NDArray[np.int64] | None = None,
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """データを分割

        Args:
            X: 特徴量データ
            y: ターゲット（未使用）
            groups: グループ（未使用）

        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)

        if n_samples < self.n_splits:
            raise ValueError(f"Cannot split {n_samples} samples into {self.n_splits} folds")

        # Fold境界を計算
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        fold_starts = np.cumsum(np.r_[0, fold_sizes])

        # Embargoサンプル数を計算
        avg_test_size = n_samples // self.n_splits
        embargo = int(np.ceil(avg_test_size * self.embargo_pct))

        self._fold_infos = []

        # 全組み合わせを生成
        fold_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        for combo_id, test_folds in enumerate(fold_combinations):
            test_folds_set = set(test_folds)
            train_folds = [i for i in range(self.n_splits) if i not in test_folds_set]

            # テストインデックス
            test_indices_list = []
            for fold_id in test_folds:
                start = fold_starts[fold_id]
                end = fold_starts[fold_id + 1]
                test_indices_list.extend(range(start, end))
            test_indices = np.array(test_indices_list, dtype=np.int64)

            # Purge/Embargoを考慮した学習インデックス
            purge_indices_set: set[int] = set()
            embargo_indices_set: set[int] = set()

            for test_fold in test_folds:
                test_start = fold_starts[test_fold]
                test_end = fold_starts[test_fold + 1]

                # Purge（テスト直前）
                purge_start = max(0, test_start - self.purge_gap)
                for i in range(purge_start, test_start):
                    purge_indices_set.add(i)

                # Embargo（テスト直後）
                embargo_end = min(n_samples, test_end + embargo)
                for i in range(test_end, embargo_end):
                    embargo_indices_set.add(i)

            # 学習インデックス（テスト、purge、embargoを除く）
            excluded = set(test_indices.tolist()) | purge_indices_set | embargo_indices_set
            train_indices = np.array([i for i in range(n_samples) if i not in excluded], dtype=np.int64)

            # フォールド情報を保存
            fold_info = FoldInfo(
                fold_id=combo_id,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=np.array(list(purge_indices_set), dtype=np.int64),
                embargo_indices=np.array(list(embargo_indices_set), dtype=np.int64),
            )
            self._fold_infos.append(fold_info)

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: pd.DataFrame | NDArray[np.float64] | None = None,
        y: NDArray[np.float64] | None = None,
        groups: NDArray[np.int64] | None = None,
    ) -> int:
        """分割数を取得（sklearn互換）

        Returns:
            組み合わせ数
        """
        return self.n_combinations

    def get_fold_info(self, fold_id: int) -> FoldInfo | None:
        """フォールド情報を取得

        Args:
            fold_id: フォールドID

        Returns:
            FoldInfo or None
        """
        if 0 <= fold_id < len(self._fold_infos):
            return self._fold_infos[fold_id]
        return None

    def get_summary(self, n_samples: int | None = None) -> CVSummary:
        """サマリーを取得

        Args:
            n_samples: 総サンプル数

        Returns:
            CVSummary
        """
        return CVSummary(
            n_splits=self.n_combinations,
            n_samples=n_samples or 0,
            folds=self._fold_infos.copy(),
        )


# =============================================================================
# 便利関数
# =============================================================================

def create_purged_kfold(
    n_splits: int = 5,
    purge_gap: int = 5,
    embargo_pct: float = 0.01,
) -> PurgedKFold:
    """PurgedKFoldを作成（ファクトリ関数）

    Args:
        n_splits: 分割数
        purge_gap: Purge期間
        embargo_pct: Embargo比率

    Returns:
        PurgedKFold
    """
    return PurgedKFold(
        n_splits=n_splits,
        purge_gap=purge_gap,
        embargo_pct=embargo_pct,
    )


def create_combinatorial_purged_cv(
    n_splits: int = 5,
    n_test_splits: int = 2,
    purge_gap: int = 5,
    embargo_pct: float = 0.01,
) -> CombinatorialPurgedCV:
    """CombinatorialPurgedCVを作成（ファクトリ関数）

    Args:
        n_splits: 分割数
        n_test_splits: テストに使用するfold数
        purge_gap: Purge期間
        embargo_pct: Embargo比率

    Returns:
        CombinatorialPurgedCV
    """
    return CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        purge_gap=purge_gap,
        embargo_pct=embargo_pct,
    )


def calculate_optimal_n_splits(
    n_samples: int,
    min_train_size: int = 100,
    min_test_size: int = 50,
    purge_gap: int = 5,
) -> int:
    """最適な分割数を計算

    Args:
        n_samples: 総サンプル数
        min_train_size: 最小学習データサイズ
        min_test_size: 最小テストデータサイズ
        purge_gap: Purge期間

    Returns:
        最適な分割数
    """
    # 最大分割数（テストサイズが最小値以上になる）
    max_splits = n_samples // min_test_size

    # 学習データサイズを確保できる分割数
    for n_splits in range(min(max_splits, 10), 1, -1):
        test_size = n_samples // n_splits
        train_size = n_samples - test_size - purge_gap

        if train_size >= min_train_size and test_size >= min_test_size:
            return n_splits

    return 2  # 最小値


def validate_no_leakage(
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
    purge_gap: int = 0,
) -> bool:
    """データリークがないことを検証

    Args:
        train_indices: 学習インデックス
        test_indices: テストインデックス
        purge_gap: 必要なpurge期間

    Returns:
        リークがなければTrue
    """
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())

    # 重複チェック
    if train_set & test_set:
        return False

    # Purgeチェック
    if purge_gap > 0:
        test_min = min(test_indices)
        test_max = max(test_indices)

        # テスト直前のpurge期間が学習に含まれていないか
        purge_before = set(range(test_min - purge_gap, test_min))
        if train_set & purge_before:
            return False

    return True
