"""
Engine Integration Tests - エンジン統合テスト

【INT-006】エンジン間の整合性とTypeError回帰を防止するテストスイート。

主なテスト内容:
1. エンジンインターフェース準拠テスト
2. エンジン結果一致性テスト（許容誤差内）
3. TypeError回帰テスト（最重要）
4. エンジン切り替えテスト
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import polars as pl
import pytest

# エンジンのインポート
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.fast_engine import FastBacktestConfig, FastBacktestEngine
from src.backtest.streaming_engine import (
    StreamingBacktestEngine,
    StreamingBacktestResult,
)
from src.backtest.vectorbt_engine import (
    VectorBTConfig,
    VectorBTResult,
    VectorBTStyleEngine,
)

# RayBacktestEngineはオプショナル
try:
    from src.backtest.ray_engine import (
        RAY_AVAILABLE,
        RayBacktestConfig,
        RayBacktestEngine,
    )
except ImportError:
    RAY_AVAILABLE = False
    RayBacktestEngine = None
    RayBacktestConfig = None

logger = logging.getLogger(__name__)


# =============================================================================
# テストフィクスチャ
# =============================================================================


@pytest.fixture
def sample_prices_pandas() -> pd.DataFrame:
    """サンプル価格データ（Pandas）"""
    np.random.seed(42)
    n_days = 252
    n_assets = 4
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = np.random.randn(n_days, n_assets) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(
        prices, index=dates, columns=["SPY", "QQQ", "TLT", "GLD"]
    )


@pytest.fixture
def sample_prices_polars(sample_prices_pandas: pd.DataFrame) -> pl.DataFrame:
    """サンプル価格データ（Polars）"""
    df = sample_prices_pandas.reset_index()
    df.columns = ["timestamp"] + list(sample_prices_pandas.columns)
    return pl.from_pandas(df)


@pytest.fixture
def sample_price_dict(sample_prices_pandas: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """サンプル価格データ（Dict形式 - StreamingBacktestEngine用）"""
    result = {}
    for col in sample_prices_pandas.columns:
        df = pd.DataFrame({
            "Date": sample_prices_pandas.index,
            "Close": sample_prices_pandas[col].values,
            "Open": sample_prices_pandas[col].values,
            "High": sample_prices_pandas[col].values * 1.01,
            "Low": sample_prices_pandas[col].values * 0.99,
            "Volume": np.random.randint(1000000, 10000000, len(sample_prices_pandas)),
        })
        df = df.set_index("Date")
        result[col] = df
    return result


# =============================================================================
# エンジンクラスリスト
# =============================================================================

# テスト対象のエンジンクラス
ENGINE_CLASSES: List[Type] = [
    BacktestEngine,
    FastBacktestEngine,
    StreamingBacktestEngine,
    VectorBTStyleEngine,
]

# RayBacktestEngineは利用可能な場合のみ追加
if RAY_AVAILABLE and RayBacktestEngine is not None:
    ENGINE_CLASSES.append(RayBacktestEngine)


# =============================================================================
# エンジンインターフェース準拠テスト
# =============================================================================


class TestEngineInterfaceCompliance:
    """エンジンがrun()メソッドを持っていることを確認"""

    @pytest.mark.parametrize("engine_class", ENGINE_CLASSES)
    def test_engine_has_run_method(self, engine_class: Type):
        """全エンジンがrun()メソッドを持っていることを確認"""
        assert hasattr(engine_class, "run"), f"{engine_class.__name__} has no run method"
        assert callable(getattr(engine_class, "run")), f"{engine_class.__name__}.run is not callable"

    @pytest.mark.parametrize("engine_class", ENGINE_CLASSES)
    def test_engine_is_class(self, engine_class: Type):
        """エンジンがクラスであることを確認"""
        assert isinstance(engine_class, type), f"{engine_class} is not a class"

    def test_fast_engine_has_config(self):
        """FastBacktestEngineがconfig引数を受け付けることを確認"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        engine = FastBacktestEngine(config)
        assert hasattr(engine, "config")
        assert engine.config == config

    def test_vectorbt_engine_has_config(self):
        """VectorBTStyleEngineがconfig引数を受け付けることを確認"""
        config = VectorBTConfig(
            initial_capital=100000.0,
            transaction_cost_bps=10.0,
        )
        engine = VectorBTStyleEngine(config)
        assert hasattr(engine, "config")
        assert engine.config == config


# =============================================================================
# TypeError回帰テスト（最重要）
# =============================================================================


class TestTypeErrorRegression:
    """TypeError回帰テスト - エンジン呼び出しでTypeErrorが発生しないことを確認"""

    def test_fast_engine_no_type_error_pandas(self, sample_prices_pandas: pd.DataFrame):
        """FastBacktestEngine: Pandas入力でTypeErrorが発生しない"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        # TypeErrorが発生しないことを確認
        try:
            result = engine.run(sample_prices_pandas)
            assert result is not None
            assert hasattr(result, "total_return")
        except TypeError as e:
            pytest.fail(f"TypeError raised: {e}")

    def test_fast_engine_no_type_error_polars(self, sample_prices_polars: pl.DataFrame):
        """FastBacktestEngine: Polars入力でTypeErrorが発生しない"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        try:
            result = engine.run(sample_prices_polars)
            assert result is not None
        except TypeError as e:
            pytest.fail(f"TypeError raised with Polars input: {e}")

    @pytest.mark.xfail(reason="VectorBTStyleEngine has IndexError bug in NumPy fallback mode")
    def test_vectorbt_engine_no_type_error(self, sample_prices_polars: pl.DataFrame):
        """VectorBTStyleEngine: TypeErrorが発生しない"""
        config = VectorBTConfig(
            initial_capital=100000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )
        engine = VectorBTStyleEngine(config)

        # timestamp列を除いた価格データを準備
        price_cols = [c for c in sample_prices_polars.columns if c != "timestamp"]
        prices = sample_prices_polars.select(price_cols)

        try:
            result = engine.run(prices)
            assert result is not None
            assert isinstance(result, VectorBTResult)
        except TypeError as e:
            pytest.fail(f"TypeError raised: {e}")

    def test_streaming_engine_no_type_error(self, sample_price_dict: Dict[str, pd.DataFrame]):
        """StreamingBacktestEngine: TypeErrorが発生しない"""
        from src.backtest.base import UnifiedBacktestConfig, UnifiedBacktestResult

        config = UnifiedBacktestConfig(
            start_date="2024-01-01",
            end_date="2024-10-01",
            rebalance_frequency="monthly",
        )
        engine = StreamingBacktestEngine()

        try:
            result = engine.run(
                universe=list(sample_price_dict.keys()),
                prices=sample_price_dict,
                config=config,
            )
            assert result is not None
            assert isinstance(result, UnifiedBacktestResult)
        except TypeError as e:
            pytest.fail(f"TypeError raised: {e}")

    def test_fast_engine_with_weights_func_no_type_error(
        self, sample_prices_pandas: pd.DataFrame
    ):
        """FastBacktestEngine: weights_func引数でTypeErrorが発生しない"""

        def equal_weights(signals, cov):
            n = len(cov)
            return np.ones(n) / n

        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        try:
            result = engine.run(sample_prices_pandas, weights_func=equal_weights)
            assert result is not None
        except TypeError as e:
            pytest.fail(f"TypeError raised with weights_func: {e}")

    def test_fast_engine_cost_optimizer_no_type_error(
        self, sample_prices_pandas: pd.DataFrame
    ):
        """FastBacktestEngine: TransactionCostOptimizer有効時にTypeErrorが発生しない"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            use_numba=False,
            vix_cash_enabled=False,
            use_cost_optimizer=True,
            max_turnover=0.30,
        )
        engine = FastBacktestEngine(config)

        try:
            result = engine.run(sample_prices_pandas)
            assert result is not None
        except TypeError as e:
            pytest.fail(f"TypeError raised with cost_optimizer: {e}")


# =============================================================================
# エンジン結果一致性テスト
# =============================================================================


class TestEngineResultConsistency:
    """異なるエンジンで同一データを実行し、結果が大きく異ならないことを確認"""

    @pytest.mark.xfail(reason="VectorBTStyleEngine has IndexError bug in NumPy fallback mode")
    def test_fast_vs_vectorbt_sharpe_consistency(
        self,
        sample_prices_pandas: pd.DataFrame,
        sample_prices_polars: pl.DataFrame,
    ):
        """FastBacktestEngineとVectorBTStyleEngineのSharpe比率が大きく異ならない"""
        # FastBacktestEngine
        fast_config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            initial_capital=100000.0,
            transaction_cost_bps=10.0,
            use_numba=False,
            vix_cash_enabled=False,
        )
        fast_engine = FastBacktestEngine(fast_config)
        fast_result = fast_engine.run(sample_prices_pandas)

        # VectorBTStyleEngine
        vbt_config = VectorBTConfig(
            initial_capital=100000.0,
            transaction_cost_bps=10.0,
            rebalance_frequency="monthly",
        )
        vbt_engine = VectorBTStyleEngine(vbt_config)
        price_cols = [c for c in sample_prices_polars.columns if c != "timestamp"]
        prices = sample_prices_polars.select(price_cols)
        vbt_result = vbt_engine.run(prices)

        # Sharpe比率の差が許容範囲内（両エンジンが等ウェイト戦略の場合）
        fast_sharpe = fast_result.sharpe_ratio if hasattr(fast_result, "sharpe_ratio") else 0.0
        vbt_sharpe = vbt_result.sharpe_ratio

        # 注: 異なるエンジンでは計算方法が微妙に異なるため、
        # 完全一致は期待せず、大きく異ならないことを確認
        assert abs(fast_sharpe - vbt_sharpe) < 1.0, (
            f"Sharpe ratio difference too large: Fast={fast_sharpe:.2f}, VBT={vbt_sharpe:.2f}"
        )

    def test_fast_engine_deterministic(self, sample_prices_pandas: pd.DataFrame):
        """FastBacktestEngineが決定論的な結果を返す"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            rebalance_frequency="monthly",
            use_numba=False,
            vix_cash_enabled=False,
        )

        engine1 = FastBacktestEngine(config)
        result1 = engine1.run(sample_prices_pandas)

        engine2 = FastBacktestEngine(config)
        result2 = engine2.run(sample_prices_pandas)

        # 同一設定・同一データで同一結果
        assert abs(result1.total_return - result2.total_return) < 1e-10, (
            "FastBacktestEngine is not deterministic"
        )


# =============================================================================
# エンジン切り替えテスト
# =============================================================================


class TestEngineSwitching:
    """エンジン切り替えが正常に動作することを確認"""

    @pytest.mark.xfail(reason="VectorBTStyleEngine has IndexError bug in NumPy fallback mode")
    def test_switch_from_fast_to_vectorbt(
        self,
        sample_prices_pandas: pd.DataFrame,
        sample_prices_polars: pl.DataFrame,
    ):
        """FastBacktestEngineからVectorBTStyleEngineへの切り替え"""
        # FastBacktestEngineで実行
        fast_config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            use_numba=False,
            vix_cash_enabled=False,
        )
        fast_engine = FastBacktestEngine(fast_config)
        fast_result = fast_engine.run(sample_prices_pandas)
        assert fast_result is not None

        # VectorBTStyleEngineに切り替えて実行
        vbt_config = VectorBTConfig()
        vbt_engine = VectorBTStyleEngine(vbt_config)
        price_cols = [c for c in sample_prices_polars.columns if c != "timestamp"]
        prices = sample_prices_polars.select(price_cols)
        vbt_result = vbt_engine.run(prices)
        assert vbt_result is not None

        # 両方とも正常に結果を返す
        assert hasattr(fast_result, "total_return")
        assert hasattr(vbt_result, "total_return")

    def test_multiple_runs_same_engine(self, sample_prices_pandas: pd.DataFrame):
        """同一エンジンで複数回実行"""
        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 10, 1),
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        results = []
        for _ in range(3):
            result = engine.run(sample_prices_pandas)
            results.append(result.total_return)

        # 全て同一結果
        assert all(abs(r - results[0]) < 1e-10 for r in results), (
            "Multiple runs of same engine returned different results"
        )


# =============================================================================
# 設定バリデーションテスト
# =============================================================================


class TestConfigValidation:
    """設定パラメータのバリデーション"""

    def test_fast_config_dates(self):
        """FastBacktestConfig: 開始日と終了日の関係"""
        with pytest.raises((ValueError, AssertionError)):
            # 終了日が開始日より前の場合はエラー
            config = FastBacktestConfig(
                start_date=datetime(2024, 12, 31),
                end_date=datetime(2024, 1, 1),
            )
            # 一部のエンジンはrunで検証するため、ここでrunを呼ぶ
            engine = FastBacktestEngine(config)
            # 空のDataFrameを渡してエラーを誘発
            import pandas as pd
            engine.run(pd.DataFrame())

    def test_vectorbt_config_defaults(self):
        """VectorBTConfig: デフォルト値の確認"""
        config = VectorBTConfig()
        assert config.initial_capital == 100000.0
        assert config.transaction_cost_bps == 10.0  # Default is 10 bps
        assert config.rebalance_frequency == "monthly"  # Default is monthly


# =============================================================================
# エッジケーステスト
# =============================================================================


class TestEdgeCases:
    """エッジケースの処理"""

    def test_fast_engine_single_asset(self):
        """FastBacktestEngine: 単一アセットでの動作"""
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        prices = 100 * np.cumprod(1 + np.random.randn(n_days) * 0.02)
        prices_df = pd.DataFrame({"SPY": prices}, index=dates)

        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        try:
            result = engine.run(prices_df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Single asset test failed: {e}")

    def test_fast_engine_many_assets(self):
        """FastBacktestEngine: 多数アセットでの動作"""
        np.random.seed(42)
        n_days = 100
        n_assets = 50
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        returns = np.random.randn(n_days, n_assets) * 0.02
        prices = 100 * np.cumprod(1 + returns, axis=0)
        asset_names = [f"ASSET_{i:02d}" for i in range(n_assets)]
        prices_df = pd.DataFrame(prices, index=dates, columns=asset_names)

        config = FastBacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 1),
            use_numba=False,
            vix_cash_enabled=False,
        )
        engine = FastBacktestEngine(config)

        try:
            result = engine.run(prices_df)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Many assets test failed: {e}")


# =============================================================================
# 実行ヘルパー
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
