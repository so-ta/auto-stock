"""
Meta-Validation for Optimization Level Selection.

バックテスト内で「最適化の最適化」を行うメタ検証システム。
年次でメタ検証を実行し、結果をキャッシュして効率化。

設計:
- 年1回のメタ検証で最適な最適化レベルを決定
- 結果はディスクキャッシュに保存（再計算回避）
- 日常のシグナル生成は決定済みレベルの軽量計算のみ

使用例:
    from src.signals.meta_validation import (
        MetaValidationCache,
        AdaptiveParameterCalculator,
        OptimizationLevel,
    )

    # バックテスト内での使用
    calculator = AdaptiveParameterCalculator()

    for rebalance_date in rebalance_dates:
        # 年末にメタ検証（キャッシュ活用）
        params = calculator.get_params(
            prices=prices.loc[:rebalance_date],
            current_date=rebalance_date,
        )

        # シグナル生成
        signal = MomentumSignal(**params)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptimizationLevel(IntEnum):
    """最適化レベル"""
    FIXED = 0           # 完全固定パラメータ
    STATISTICAL = 1     # 統計量ベース動的（std, percentile）
    CONSTRAINED = 2     # 制約付き最適化
    FULL = 3            # 完全最適化（非推奨）


@dataclass
class MetaValidationResult:
    """メタ検証結果"""
    level: OptimizationLevel
    oos_sharpe: float
    oos_sharpe_std: float
    is_sharpe: float
    overfitting_score: float
    stability_score: float
    validated_at: datetime
    data_end_date: datetime
    n_folds: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        import json
        return {
            "level": self.level.value,
            "level_name": self.level.name,
            "oos_sharpe": self.oos_sharpe,
            "oos_sharpe_std": self.oos_sharpe_std,
            "is_sharpe": self.is_sharpe,
            "overfitting_score": self.overfitting_score,
            "stability_score": self.stability_score,
            "validated_at": self.validated_at.isoformat(),
            "data_end_date": self.data_end_date.isoformat(),
            "n_folds": self.n_folds,
            "metadata_json": json.dumps(self.metadata),  # JSON string for Parquet compatibility
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetaValidationResult":
        """辞書から復元"""
        import json
        # Handle both old format (metadata) and new format (metadata_json)
        if "metadata_json" in data:
            metadata = json.loads(data["metadata_json"])
        else:
            metadata = data.get("metadata", {})

        return cls(
            level=OptimizationLevel(data["level"]),
            oos_sharpe=data["oos_sharpe"],
            oos_sharpe_std=data["oos_sharpe_std"],
            is_sharpe=data["is_sharpe"],
            overfitting_score=data["overfitting_score"],
            stability_score=data["stability_score"],
            validated_at=datetime.fromisoformat(data["validated_at"]),
            data_end_date=datetime.fromisoformat(data["data_end_date"]),
            n_folds=data["n_folds"],
            metadata=metadata,
        )


@dataclass
class LevelValidationResult:
    """単一レベルの検証結果"""
    level: OptimizationLevel
    oos_sharpes: list[float]
    is_sharpes: list[float]

    @property
    def oos_sharpe(self) -> float:
        return np.mean(self.oos_sharpes) if self.oos_sharpes else 0.0

    @property
    def oos_sharpe_std(self) -> float:
        return np.std(self.oos_sharpes) if len(self.oos_sharpes) > 1 else 0.0

    @property
    def is_sharpe(self) -> float:
        return np.mean(self.is_sharpes) if self.is_sharpes else 0.0

    @property
    def overfitting_score(self) -> float:
        """過学習スコア（IS - OOS）"""
        return max(0, self.is_sharpe - self.oos_sharpe)

    @property
    def stability_score(self) -> float:
        """安定性スコア（1 / std）"""
        std = self.oos_sharpe_std
        return 1.0 / (std + 0.01)

    @property
    def composite_score(self) -> float:
        """複合スコア（推奨レベル選択用）"""
        return (
            self.oos_sharpe * self.stability_score
            - 0.5 * self.overfitting_score
        )


class MetaValidationCache:
    """
    メタ検証結果のキャッシュ

    年単位でキャッシュキーを生成し、同じ年のデータでは
    再計算を回避する。
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        signal_type: str = "default",
    ):
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
            signal_type: シグナルタイプ（キャッシュキーに使用）
        """
        if cache_dir is None:
            from src.config.settings import get_cache_path
            cache_dir = get_cache_path("meta_validation")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._signal_type = signal_type
        self._memory_cache: dict[str, MetaValidationResult] = {}

    def _generate_cache_key(self, data_end_date: datetime) -> str:
        """
        キャッシュキーを生成

        年単位でキーを生成（同じ年なら同じキー）
        """
        year = data_end_date.year
        key_string = f"{self._signal_type}|{year}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, data_end_date: datetime) -> MetaValidationResult | None:
        """
        キャッシュから結果を取得

        Args:
            data_end_date: データ終了日

        Returns:
            キャッシュされた結果、または None
        """
        key = self._generate_cache_key(data_end_date)

        # メモリキャッシュを確認
        if key in self._memory_cache:
            logger.debug(f"Meta validation cache hit (memory): {key}")
            return self._memory_cache[key]

        # ディスクキャッシュを確認
        cache_file = self._cache_dir / f"{key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    result = MetaValidationResult.from_dict(df.iloc[0].to_dict())
                    self._memory_cache[key] = result
                    logger.debug(f"Meta validation cache hit (disk): {key}")
                    return result
            except Exception as e:
                logger.warning(f"Failed to read meta validation cache: {e}")

        return None

    def put(self, data_end_date: datetime, result: MetaValidationResult) -> None:
        """
        結果をキャッシュに保存

        Args:
            data_end_date: データ終了日
            result: 検証結果
        """
        key = self._generate_cache_key(data_end_date)

        # メモリキャッシュに保存
        self._memory_cache[key] = result

        # ディスクキャッシュに保存
        cache_file = self._cache_dir / f"{key}.parquet"
        try:
            df = pd.DataFrame([result.to_dict()])
            df.to_parquet(cache_file, compression="snappy")
            logger.debug(f"Meta validation cached: {key}")
        except Exception as e:
            logger.warning(f"Failed to write meta validation cache: {e}")

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.parquet"):
            try:
                f.unlink()
            except OSError:
                pass


class MetaValidator:
    """
    メタ検証器

    各最適化レベルをWalk-forward CVで評価し、
    最適なレベルを推奨する。
    """

    def __init__(
        self,
        n_folds: int = 3,
        min_train_days: int = 504,  # 2年
        test_days: int = 63,        # 3ヶ月
    ):
        """
        初期化

        Args:
            n_folds: CVフォールド数
            min_train_days: 最小訓練期間（日数）
            test_days: テスト期間（日数）
        """
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.test_days = test_days

    def validate_all_levels(
        self,
        prices: pd.DataFrame,
        signal_class: type | None = None,
    ) -> dict[OptimizationLevel, LevelValidationResult]:
        """
        全レベルを検証

        Args:
            prices: 価格データ（'close'列必須）
            signal_class: シグナルクラス（省略時は内蔵評価を使用）

        Returns:
            レベル別検証結果
        """
        results = {}

        for level in OptimizationLevel:
            result = self._validate_level(prices, level, signal_class)
            results[level] = result
            logger.debug(
                f"Level {level.name}: OOS={result.oos_sharpe:.3f}, "
                f"IS={result.is_sharpe:.3f}, composite={result.composite_score:.3f}"
            )

        return results

    def _validate_level(
        self,
        prices: pd.DataFrame,
        level: OptimizationLevel,
        signal_class: type | None,
    ) -> LevelValidationResult:
        """単一レベルを検証"""
        n = len(prices)
        fold_size = (n - self.min_train_days) // self.n_folds

        if fold_size < self.test_days:
            # データ不足時はデフォルト結果
            return LevelValidationResult(
                level=level,
                oos_sharpes=[0.0],
                is_sharpes=[0.0],
            )

        oos_sharpes = []
        is_sharpes = []

        for fold_idx in range(self.n_folds):
            # データ分割
            test_end = n - fold_idx * fold_size
            test_start = test_end - self.test_days
            train_end = test_start

            if train_end < self.min_train_days:
                continue

            train_data = prices.iloc[:train_end]
            test_data = prices.iloc[test_start:test_end]

            # パラメータ計算
            params = self._calculate_params(train_data, level)

            # 評価
            is_sharpe = self._evaluate(train_data, params)
            oos_sharpe = self._evaluate(test_data, params)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

        return LevelValidationResult(
            level=level,
            oos_sharpes=oos_sharpes,
            is_sharpes=is_sharpes,
        )

    def _calculate_params(
        self,
        data: pd.DataFrame,
        level: OptimizationLevel,
    ) -> dict[str, Any]:
        """レベルに応じたパラメータ計算"""
        returns = data["close"].pct_change().dropna()

        if level == OptimizationLevel.FIXED:
            return {"lookback": 20, "scale": 5.0}

        elif level == OptimizationLevel.STATISTICAL:
            # 統計量ベース
            std = returns.std()
            scale = 1.0 / (3.0 * std) if std > 0 else 5.0

            # ボラティリティに応じたlookback
            recent_vol = returns.tail(20).std()
            long_vol = std
            ratio = recent_vol / long_vol if long_vol > 0 else 1.0

            if ratio < 0.7:
                lookback = 40
            elif ratio > 1.5:
                lookback = 10
            else:
                lookback = 20

            return {"lookback": lookback, "scale": scale}

        elif level == OptimizationLevel.CONSTRAINED:
            # 制約付き最適化（グリッドサーチ）
            best_params = {"lookback": 20, "scale": 5.0}
            best_sharpe = -np.inf

            for lookback in [10, 20, 40]:
                for scale in [3.0, 5.0, 8.0]:
                    params = {"lookback": lookback, "scale": scale}
                    sharpe = self._evaluate(data.tail(252), params)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

            return best_params

        else:  # FULL
            # 完全最適化（より細かいグリッド）
            best_params = {"lookback": 20, "scale": 5.0}
            best_sharpe = -np.inf

            for lookback in range(5, 61, 5):
                for scale in np.arange(1.0, 15.0, 2.0):
                    params = {"lookback": lookback, "scale": float(scale)}
                    sharpe = self._evaluate(data.tail(252), params)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

            return best_params

    def _evaluate(self, data: pd.DataFrame, params: dict[str, Any]) -> float:
        """シンプルなモメンタム戦略で評価"""
        if len(data) < params["lookback"] + 20:
            return 0.0

        close = data["close"]
        returns = close.pct_change().dropna()

        # モメンタムシグナル
        lookback = params["lookback"]
        momentum = close.pct_change(periods=lookback)
        signal = np.tanh(momentum * params["scale"])

        # 戦略リターン
        strategy_returns = returns * signal.shift(1)
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) < 20:
            return 0.0

        # シャープ比
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()

        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = 0.0

        return float(sharpe)

    def recommend_level(
        self,
        results: dict[OptimizationLevel, LevelValidationResult],
    ) -> OptimizationLevel:
        """最適なレベルを推奨"""
        best_level = OptimizationLevel.STATISTICAL  # デフォルト
        best_score = -np.inf

        for level, result in results.items():
            if result.composite_score > best_score:
                best_score = result.composite_score
                best_level = level

        return best_level


class AdaptiveParameterCalculator:
    """
    適応的パラメータ計算器

    キャッシュを活用して効率的にパラメータを計算。
    年次でメタ検証を実行し、結果に基づいてパラメータを生成。
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        signal_type: str = "momentum",
        min_history_years: int = 2,
        validation_month: int = 12,  # 12月にメタ検証
    ):
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
            signal_type: シグナルタイプ
            min_history_years: メタ検証に必要な最小データ年数
            validation_month: メタ検証を実行する月
        """
        self._cache = MetaValidationCache(cache_dir, signal_type)
        self._validator = MetaValidator()
        self._signal_type = signal_type
        self._min_history_days = min_history_years * 252
        self._validation_month = validation_month

        # 現在のレベル（初期値）
        self._current_level = OptimizationLevel.STATISTICAL
        self._level_history: dict[datetime, OptimizationLevel] = {}

    def get_params(
        self,
        prices: pd.DataFrame,
        current_date: datetime | pd.Timestamp,
    ) -> dict[str, Any]:
        """
        パラメータを取得

        年末にはメタ検証を実行（キャッシュ活用）、
        それ以外は現在のレベルでパラメータを計算。

        Args:
            prices: 現在日までの価格データ
            current_date: 現在日

        Returns:
            シグナルパラメータ
        """
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.to_pydatetime()

        # メタ検証を実行すべきか確認
        if self._should_run_meta_validation(current_date, len(prices)):
            self._run_meta_validation(prices, current_date)

        # 現在のレベルでパラメータを計算
        return self._calculate_params(prices, self._current_level)

    def _should_run_meta_validation(
        self,
        current_date: datetime,
        data_length: int,
    ) -> bool:
        """メタ検証を実行すべきか判定"""
        # データ量チェック
        if data_length < self._min_history_days:
            return False

        # 年末かどうか
        if current_date.month != self._validation_month:
            return False

        # 月末かどうか（25日以降）
        if current_date.day < 25:
            return False

        # 今年のキャッシュが既にあるか
        cached = self._cache.get(current_date)
        if cached is not None and cached.data_end_date.year == current_date.year:
            # キャッシュがあるならレベルを復元して検証スキップ
            self._current_level = cached.level
            return False

        return True

    def _run_meta_validation(
        self,
        prices: pd.DataFrame,
        current_date: datetime,
    ) -> None:
        """メタ検証を実行"""
        logger.info(f"Running meta validation for {current_date.year}")

        # キャッシュ確認（念のため）
        cached = self._cache.get(current_date)
        if cached is not None:
            self._current_level = cached.level
            self._level_history[current_date] = cached.level
            logger.info(f"Meta validation loaded from cache: Level {cached.level.name}")
            return

        # メタ検証実行
        results = self._validator.validate_all_levels(prices)
        recommended = self._validator.recommend_level(results)

        # 結果を保存
        best_result = results[recommended]
        meta_result = MetaValidationResult(
            level=recommended,
            oos_sharpe=best_result.oos_sharpe,
            oos_sharpe_std=best_result.oos_sharpe_std,
            is_sharpe=best_result.is_sharpe,
            overfitting_score=best_result.overfitting_score,
            stability_score=best_result.stability_score,
            validated_at=datetime.now(),
            data_end_date=current_date,
            n_folds=self._validator.n_folds,
            metadata={
                "all_levels": {
                    level.name: {
                        "oos_sharpe": r.oos_sharpe,
                        "composite_score": r.composite_score,
                    }
                    for level, r in results.items()
                }
            },
        )

        self._cache.put(current_date, meta_result)
        self._current_level = recommended
        self._level_history[current_date] = recommended

        logger.info(
            f"Meta validation completed: Level {recommended.name} "
            f"(OOS Sharpe={best_result.oos_sharpe:.3f})"
        )

    def _calculate_params(
        self,
        prices: pd.DataFrame,
        level: OptimizationLevel,
    ) -> dict[str, Any]:
        """レベルに応じたパラメータ計算（軽量）"""
        returns = prices["close"].pct_change().dropna()

        if level == OptimizationLevel.FIXED:
            return {"lookback": 20, "scale": 5.0}

        elif level == OptimizationLevel.STATISTICAL:
            std = returns.std()
            scale = 1.0 / (3.0 * std) if std > 0 else 5.0

            recent_vol = returns.tail(20).std()
            long_vol = std
            ratio = recent_vol / long_vol if long_vol > 0 else 1.0

            if ratio < 0.7:
                lookback = 40
            elif ratio > 1.5:
                lookback = 10
            else:
                lookback = 20

            return {"lookback": lookback, "scale": float(scale)}

        elif level == OptimizationLevel.CONSTRAINED:
            # 制約付き最適化（毎回実行は重いので簡易版）
            std = returns.std()
            scale = 1.0 / (2.5 * std) if std > 0 else 6.0
            scale = np.clip(scale, 3.0, 10.0)

            recent_vol = returns.tail(20).std()
            ratio = recent_vol / std if std > 0 else 1.0

            if ratio < 0.6:
                lookback = 50
            elif ratio > 1.8:
                lookback = 8
            else:
                lookback = 25

            return {"lookback": lookback, "scale": float(scale)}

        else:  # FULL（非推奨、STATISTICALにフォールバック）
            return self._calculate_params(prices, OptimizationLevel.STATISTICAL)

    @property
    def current_level(self) -> OptimizationLevel:
        """現在の最適化レベル"""
        return self._current_level

    @property
    def level_history(self) -> dict[datetime, OptimizationLevel]:
        """レベル変更履歴"""
        return self._level_history.copy()

    def get_validation_summary(self) -> str:
        """検証サマリを生成"""
        lines = [
            "=" * 60,
            "  META VALIDATION SUMMARY",
            "=" * 60,
            f"  Current Level: {self._current_level.name}",
            f"  Signal Type: {self._signal_type}",
            "",
            "  Level History:",
        ]

        for date, level in sorted(self._level_history.items()):
            lines.append(f"    {date.strftime('%Y-%m-%d')}: {level.name}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ファクトリ関数
def create_adaptive_calculator(
    signal_type: str = "momentum",
    cache_dir: str | Path | None = None,
) -> AdaptiveParameterCalculator:
    """適応的パラメータ計算器を作成"""
    return AdaptiveParameterCalculator(
        cache_dir=cache_dir,
        signal_type=signal_type,
    )
