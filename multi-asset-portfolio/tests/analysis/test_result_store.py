"""
Tests for BacktestResultStore

バックテスト結果の保存・読み込み・検索機能のテスト。
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.result_store import BacktestResultStore
from src.analysis.backtest_archive import (
    BacktestArchive,
    generate_config_hash,
    generate_universe_hash,
)
from src.backtest.base import (
    UnifiedBacktestResult,
    UnifiedBacktestConfig,
    RebalanceRecord,
)


@pytest.fixture
def temp_results_dir():
    """一時ディレクトリを作成し、テスト後に削除"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_result():
    """テスト用のUnifiedBacktestResult"""
    config = UnifiedBacktestConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    # 時系列データ
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    n_days = len(dates)
    daily_returns = pd.Series(
        [0.001] * n_days,
        index=dates,
    )
    portfolio_values = pd.Series(
        [100000.0 * (1.001 ** i) for i in range(n_days)],
        index=dates,
    )

    # リバランス記録
    rebalances = [
        RebalanceRecord(
            date=datetime(2020, 1, 31),
            weights_before={"AAPL": 0.0, "MSFT": 0.0},
            weights_after={"AAPL": 0.5, "MSFT": 0.5},
            turnover=1.0,
            transaction_cost=50.0,
            portfolio_value=100000.0,
        ),
        RebalanceRecord(
            date=datetime(2020, 2, 28),
            weights_before={"AAPL": 0.5, "MSFT": 0.5},
            weights_after={"AAPL": 0.6, "MSFT": 0.4},
            turnover=0.1,
            transaction_cost=5.0,
            portfolio_value=101000.0,
        ),
    ]

    result = UnifiedBacktestResult(
        total_return=0.5,
        annual_return=0.12,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=-0.1,
        volatility=0.15,
        calmar_ratio=1.2,
        win_rate=0.55,
        profit_factor=1.5,
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        rebalances=rebalances,
        total_turnover=1.1,
        total_transaction_costs=55.0,
        config=config,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        engine_name="test_engine",
    )

    return result


class TestBacktestResultStore:
    """BacktestResultStoreのテスト"""

    def test_init_creates_directory(self, temp_results_dir):
        """初期化時にディレクトリが作成される"""
        store = BacktestResultStore(temp_results_dir)
        assert Path(temp_results_dir).exists()
        assert (Path(temp_results_dir) / "index.json").exists()

    def test_save_creates_archive(self, temp_results_dir, sample_result):
        """結果を保存するとアーカイブが作成される"""
        store = BacktestResultStore(temp_results_dir)

        archive_id = store.save(
            result=sample_result,
            name="Test Backtest",
            description="Test description",
            tags=["test", "monthly"],
            universe=["AAPL", "MSFT"],
        )

        # アーカイブIDが生成される
        assert archive_id.startswith("bt_")
        assert len(archive_id) > 20

        # ディレクトリが作成される
        archive_dir = Path(temp_results_dir) / archive_id
        assert archive_dir.exists()

        # 必要なファイルが作成される
        assert (archive_dir / "metadata.json").exists()
        assert (archive_dir / "config_snapshot.yaml").exists()
        assert (archive_dir / "universe.json").exists()
        assert (archive_dir / "timeseries.parquet").exists()
        assert (archive_dir / "rebalances.parquet").exists()

    def test_load_archive(self, temp_results_dir, sample_result):
        """保存したアーカイブを読み込める"""
        store = BacktestResultStore(temp_results_dir)

        archive_id = store.save(
            result=sample_result,
            name="Test Backtest",
            description="Test description",
            tags=["test"],
            universe=["AAPL", "MSFT"],
        )

        archive = store.load(archive_id)

        assert archive.archive_id == archive_id
        assert archive.name == "Test Backtest"
        assert archive.description == "Test description"
        assert "test" in archive.tags
        assert archive.universe == ["AAPL", "MSFT"]
        assert archive.metrics["total_return"] == pytest.approx(0.5)
        assert archive.metrics["sharpe_ratio"] == pytest.approx(1.5)

    def test_load_timeseries(self, temp_results_dir, sample_result):
        """時系列データを読み込める"""
        store = BacktestResultStore(temp_results_dir)

        archive_id = store.save(
            result=sample_result,
            name="Test",
            universe=["AAPL", "MSFT"],
        )

        ts = store.load_timeseries(archive_id)

        assert "portfolio_value" in ts.columns
        assert "daily_return" in ts.columns
        assert "cumulative_return" in ts.columns
        assert "drawdown" in ts.columns
        assert len(ts) > 0

    def test_load_rebalances(self, temp_results_dir, sample_result):
        """リバランスデータを読み込める"""
        store = BacktestResultStore(temp_results_dir)

        archive_id = store.save(
            result=sample_result,
            name="Test",
            universe=["AAPL", "MSFT"],
        )

        reb = store.load_rebalances(archive_id)

        assert "weights_before" in reb.columns
        assert "weights_after" in reb.columns
        assert "turnover" in reb.columns
        assert len(reb) == 2

    def test_list_archives(self, temp_results_dir, sample_result):
        """アーカイブ一覧を取得できる"""
        store = BacktestResultStore(temp_results_dir)

        # 複数保存
        id1 = store.save(sample_result, name="Test 1", tags=["monthly"])
        id2 = store.save(sample_result, name="Test 2", tags=["weekly"])

        # 全件取得
        all_archives = store.list_archives()
        assert len(all_archives) == 2

        # タグでフィルタ
        monthly = store.list_archives(tags=["monthly"])
        assert len(monthly) == 1
        assert monthly[0]["archive_id"] == id1

    def test_find_by_config_hash(self, temp_results_dir, sample_result):
        """設定ハッシュで検索できる"""
        store = BacktestResultStore(temp_results_dir)

        # 同一設定で2回保存
        id1 = store.save(sample_result, name="Test 1", universe=["AAPL", "MSFT"])
        id2 = store.save(sample_result, name="Test 2", universe=["AAPL", "MSFT"])

        # 設定ハッシュを取得
        archive1 = store.load(id1)
        config_hash = archive1.config_hash

        # 検索
        found = store.find_by_config_hash(config_hash)
        assert len(found) == 2
        assert id1 in found
        assert id2 in found

    def test_delete_archive(self, temp_results_dir, sample_result):
        """アーカイブを削除できる"""
        store = BacktestResultStore(temp_results_dir)

        archive_id = store.save(sample_result, name="Test")

        # 存在確認
        assert store.exists(archive_id)

        # 削除
        success = store.delete(archive_id)
        assert success

        # 存在しないことを確認
        assert not store.exists(archive_id)

        # インデックスからも削除される
        archives = store.list_archives()
        assert len(archives) == 0

    def test_get_stats(self, temp_results_dir, sample_result):
        """統計情報を取得できる"""
        store = BacktestResultStore(temp_results_dir)

        store.save(sample_result, name="Test 1", tags=["monthly"])
        store.save(sample_result, name="Test 2", tags=["weekly"])

        stats = store.get_stats()

        assert stats["total_archives"] == 2
        assert stats["total_size_mb"] > 0
        assert stats["unique_config_hashes"] >= 1
        assert "monthly" in stats["unique_tags"]
        assert "weekly" in stats["unique_tags"]


class TestHashFunctions:
    """ハッシュ関数のテスト"""

    def test_generate_config_hash_deterministic(self):
        """同一設定で同一ハッシュが生成される"""
        config = {
            "initial_capital": 100000,
            "transaction_cost_bps": 10,
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }
        universe = ["AAPL", "MSFT"]

        hash1 = generate_config_hash(config, universe)
        hash2 = generate_config_hash(config, universe)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_generate_config_hash_order_independent(self):
        """ユニバースの順序が違っても同一ハッシュ"""
        config = {"initial_capital": 100000}

        hash1 = generate_config_hash(config, ["AAPL", "MSFT"])
        hash2 = generate_config_hash(config, ["MSFT", "AAPL"])

        assert hash1 == hash2

    def test_generate_config_hash_different_configs(self):
        """設定が違えば異なるハッシュ"""
        config1 = {"initial_capital": 100000}
        config2 = {"initial_capital": 200000}
        universe = ["AAPL"]

        hash1 = generate_config_hash(config1, universe)
        hash2 = generate_config_hash(config2, universe)

        assert hash1 != hash2

    def test_generate_universe_hash_deterministic(self):
        """同一ユニバースで同一ハッシュ"""
        universe = ["AAPL", "MSFT", "GOOGL"]

        hash1 = generate_universe_hash(universe)
        hash2 = generate_universe_hash(universe)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_generate_universe_hash_order_independent(self):
        """ユニバースの順序が違っても同一ハッシュ"""
        hash1 = generate_universe_hash(["AAPL", "MSFT"])
        hash2 = generate_universe_hash(["MSFT", "AAPL"])

        assert hash1 == hash2
