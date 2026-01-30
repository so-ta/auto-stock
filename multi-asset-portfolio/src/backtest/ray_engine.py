"""
Ray Distributed Backtest Engine - Ray分散バックテストエンジン

PythonのGIL制約を回避し、真の並列処理で大規模バックテストを高速化。
800+銘柄のユニバースをCPUコア数に応じて分散処理。

Usage:
    from src.backtest.ray_engine import RayBacktestEngine

    engine = RayBacktestEngine(n_workers=4)
    result = engine.run_parallel(
        universe=['AAPL', 'MSFT', ...],
        price_data=price_dict,
        start_date='2010-01-01',
        end_date='2024-12-31',
    )
    print(f"Sharpe: {result['sharpe_ratio']:.3f}")

Note:
    Rayがインストールされていない場合は、シングルスレッド処理に
    自動フォールバックする。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import base class for unified interface
from src.backtest.base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
)

# Ray import with fallback
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


@dataclass
class RayBacktestConfig:
    """Ray分散バックテスト設定"""

    # ワーカー数（-1でCPUコア数-1）
    n_workers: int = -1

    # バックテスト設定
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"
    rebalance_freq: str = "weekly"

    # 戦略パラメータ
    top_n: int = 40
    lookback_days: int = 60
    momentum_weight: float = 0.6
    reversion_weight: float = 0.4

    # Rayオプション
    ray_address: Optional[str] = None  # 既存クラスタに接続する場合
    object_store_memory: Optional[int] = None  # オブジェクトストアサイズ


@dataclass
class BacktestChunkResult:
    """チャンク別バックテスト結果"""
    chunk_id: int
    tickers: List[str]
    returns: List[float]
    n_periods: int
    elapsed_seconds: float
    error: Optional[str] = None


def _run_backtest_chunk(
    chunk_id: int,
    tickers: List[str],
    price_data: Dict[str, pd.DataFrame],
    config: dict,
) -> BacktestChunkResult:
    """
    単一チャンクのバックテスト実行（ワーカー内で実行）

    Args:
        chunk_id: チャンクID
        tickers: 対象銘柄リスト
        price_data: 価格データ辞書
        config: 設定辞書

    Returns:
        BacktestChunkResult
    """
    start_time = time.perf_counter()

    try:
        start_date = config.get("start_date", "2010-01-01")
        end_date = config.get("end_date", "2024-12-31")
        lookback = config.get("lookback_days", 60)
        top_n = min(config.get("top_n", 40), len(tickers))

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # 価格マトリクス構築
        all_closes = {}
        for ticker in tickers:
            if ticker not in price_data:
                continue

            df = price_data[ticker]
            if isinstance(df, pd.DataFrame):
                # 既にDataFrame
                if "close" in df.columns:
                    closes = df["close"]
                elif len(df.columns) > 0:
                    closes = df.iloc[:, 0]
                else:
                    continue

                if hasattr(df, "index") and isinstance(df.index, pd.DatetimeIndex):
                    all_closes[ticker] = closes
                else:
                    # インデックスが日付でない場合
                    if "date" in df.columns:
                        temp_df = df.set_index("date")["close"] if "close" in df.columns else df.set_index("date").iloc[:, 0]
                        all_closes[ticker] = temp_df
            else:
                continue

        if not all_closes:
            return BacktestChunkResult(
                chunk_id=chunk_id,
                tickers=tickers,
                returns=[],
                n_periods=0,
                elapsed_seconds=time.perf_counter() - start_time,
                error="No valid price data",
            )

        price_df = pd.DataFrame(all_closes)
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        price_df = price_df[(price_df.index >= start_dt) & (price_df.index <= end_dt)]
        price_df = price_df.ffill().dropna(axis=1, how="all")

        if price_df.empty:
            return BacktestChunkResult(
                chunk_id=chunk_id,
                tickers=tickers,
                returns=[],
                n_periods=0,
                elapsed_seconds=time.perf_counter() - start_time,
                error="Empty price matrix after filtering",
            )

        # 週次リバランス
        rebalance_dates = price_df.resample("W-FRI").last().index
        rebalance_dates = [d for d in rebalance_dates if d in price_df.index]

        portfolio_returns = []

        for i in range(len(rebalance_dates) - 1):
            rebal_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            lookback_start = rebal_date - pd.Timedelta(days=lookback)
            hist = price_df[(price_df.index >= lookback_start) & (price_df.index <= rebal_date)]

            if len(hist) < lookback // 2:
                continue

            # モメンタムスコア計算
            scores = {}
            for ticker in price_df.columns:
                if ticker in hist.columns:
                    prices = hist[ticker].dropna()
                    if len(prices) >= 2 and prices.iloc[0] > 0:
                        ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                        vol = prices.pct_change().std()
                        if vol > 0 and not np.isnan(ret):
                            scores[ticker] = ret / vol

            if not scores:
                continue

            # トップN選択
            sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_tickers = [t[0] for t in sorted_tickers[:top_n] if t[1] > 0]

            if not top_tickers:
                continue

            weight = 1.0 / len(top_tickers)

            # 期間リターン計算
            period_data = price_df[(price_df.index > rebal_date) & (price_df.index <= next_date)]

            if len(period_data) == 0:
                continue

            period_return = 0.0
            for ticker in top_tickers:
                if ticker in period_data.columns:
                    tp = period_data[ticker].dropna()
                    if len(tp) >= 2:
                        ticker_ret = (tp.iloc[-1] / tp.iloc[0]) - 1
                        period_return += weight * ticker_ret

            portfolio_returns.append(period_return)

        elapsed = time.perf_counter() - start_time

        return BacktestChunkResult(
            chunk_id=chunk_id,
            tickers=list(price_df.columns),
            returns=portfolio_returns,
            n_periods=len(portfolio_returns),
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        return BacktestChunkResult(
            chunk_id=chunk_id,
            tickers=tickers,
            returns=[],
            n_periods=0,
            elapsed_seconds=time.perf_counter() - start_time,
            error=str(e),
        )


# Ray Remote Function（Rayがある場合のみ定義）
if RAY_AVAILABLE:
    @ray.remote
    def _ray_backtest_worker(
        chunk_id: int,
        tickers: List[str],
        price_data_ref,  # Ray object reference
        config: dict,
    ) -> BacktestChunkResult:
        """Rayリモートワーカー"""
        return _run_backtest_chunk(chunk_id, tickers, price_data_ref, config)


class RayBacktestEngine(BacktestEngineBase):
    """
    Ray分散バックテストエンジン

    CPUコアを活用して大規模ユニバースのバックテストを並列実行。
    Rayが利用できない場合はシングルスレッドにフォールバック。

    BacktestEngineBase準拠: 統一インターフェースで呼び出し可能。

    Attributes:
        ray_config: RayBacktestConfig設定
        n_workers: 実際のワーカー数
        ray_initialized: Rayが初期化されているか

    Usage with ResourceConfig:
        from src.config.resource_config import get_current_resource_config

        rc = get_current_resource_config()
        engine = RayBacktestEngine.from_unified_config(
            config,
            n_workers=rc.ray_workers,
        )
    """

    ENGINE_NAME: str = "ray"

    @classmethod
    def from_unified_config(
        cls,
        config: UnifiedBacktestConfig,
        n_workers: Optional[int] = None,
        auto_init_ray: bool = True,
        **kwargs,
    ) -> "RayBacktestEngine":
        """
        統一設定からRayBacktestEngineを生成

        BacktestEngineFactory.create() から呼び出される標準メソッド。

        Args:
            config: 統一バックテスト設定
            n_workers: ワーカー数（Noneで自動検出）
            auto_init_ray: Rayを自動初期化するか
            **kwargs: その他のRayBacktestConfig設定

        Returns:
            RayBacktestEngine: 生成されたエンジン

        Example:
            from src.config.resource_config import get_current_resource_config

            rc = get_current_resource_config()
            engine = RayBacktestEngine.from_unified_config(
                config,
                n_workers=rc.ray_workers,
            )
        """
        # UnifiedBacktestConfigからRayBacktestConfig用の値を抽出
        start_date = config.start_date
        end_date = config.end_date

        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        else:
            start_date = str(start_date)

        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")
        else:
            end_date = str(end_date)

        # RayBacktestConfig作成
        ray_config_kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "rebalance_freq": config.rebalance_frequency or "weekly",
        }

        # n_workers が指定されていれば追加
        if n_workers is not None:
            ray_config_kwargs["n_workers"] = n_workers

        # その他のkwargs（top_n, lookback_days等）をマージ
        ray_config_kwargs.update(kwargs)

        ray_config = RayBacktestConfig(**ray_config_kwargs)

        return cls(
            config=ray_config,
            unified_config=config,
            n_workers=n_workers,
            auto_init_ray=auto_init_ray,
        )

    def __init__(
        self,
        config: RayBacktestConfig | dict | None = None,
        unified_config: Optional[UnifiedBacktestConfig] = None,
        n_workers: Optional[int] = None,
        auto_init_ray: bool = True,
    ):
        """
        初期化

        Args:
            config: Ray固有設定（RayBacktestConfigまたは辞書）
            unified_config: 統一バックテスト設定
            n_workers: ワーカー数（Noneで自動検出、configより優先）
            auto_init_ray: Rayを自動初期化するか（デフォルト: True）
        """
        super().__init__(unified_config)

        if config is None:
            self.ray_config = RayBacktestConfig()
        elif isinstance(config, dict):
            self.ray_config = RayBacktestConfig(**config)
        else:
            self.ray_config = config

        # 後方互換性のため config プロパティも維持
        self.config = self.ray_config

        # ワーカー数決定（引数 > config > 自動検出の優先順位）
        if n_workers is not None:
            self.n_workers = n_workers
        elif self.ray_config.n_workers == -1:
            self.n_workers = max(1, (os.cpu_count() or 4) - 1)
        else:
            self.n_workers = self.ray_config.n_workers

        self.ray_initialized = False
        self.auto_init_ray = auto_init_ray

        if auto_init_ray:
            self._init_ray()

    # =========================================================================
    # BacktestEngineBase Interface Implementation
    # =========================================================================

    def run(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: Optional[UnifiedBacktestConfig] = None,
        weights_func: Optional[Callable] = None,
    ) -> UnifiedBacktestResult:
        """
        統一インターフェースでバックテスト実行（BacktestEngineBase準拠）

        Args:
            universe: ユニバース（銘柄リスト）
            prices: 価格データ（{symbol: DataFrame}）
            config: バックテスト設定
            weights_func: ウェイト計算関数（使用しない場合はNone）

        Returns:
            UnifiedBacktestResult: 統一バックテスト結果
        """
        cfg = config or self._config
        if cfg is None:
            cfg = UnifiedBacktestConfig(
                start_date=self.ray_config.start_date,
                end_date=self.ray_config.end_date,
                initial_capital=100000.0,
                rebalance_frequency="weekly",
            )

        # 入力検証
        self.validate_inputs(universe, prices, cfg)

        # 内部メソッドで実行
        result = self.run_parallel(
            universe=universe,
            price_data=prices,
            start_date=cfg.start_date.strftime("%Y-%m-%d") if isinstance(cfg.start_date, datetime) else str(cfg.start_date),
            end_date=cfg.end_date.strftime("%Y-%m-%d") if isinstance(cfg.end_date, datetime) else str(cfg.end_date),
        )

        # UnifiedBacktestResult に変換
        return self._convert_to_unified_result(result, cfg)

    def validate_inputs(
        self,
        universe: List[str],
        prices: Dict[str, pd.DataFrame],
        config: UnifiedBacktestConfig,
    ) -> bool:
        """
        入力を検証（BacktestEngineBase準拠）

        Args:
            universe: ユニバース
            prices: 価格データ
            config: 設定

        Returns:
            bool: 検証結果

        Raises:
            ValueError: 検証エラー
        """
        # 共通検証
        warnings = self._validate_common_inputs(universe, prices, config)
        for warning in warnings:
            print(f"Warning: {warning}")

        if len(universe) == 0:
            raise ValueError("Universe cannot be empty")

        return True

    def _convert_to_unified_result(
        self,
        result: Dict[str, Any],
        config: UnifiedBacktestConfig,
    ) -> UnifiedBacktestResult:
        """Ray結果を UnifiedBacktestResult に変換"""
        return UnifiedBacktestResult(
            total_return=result.get("total_return", 0.0),
            annual_return=result.get("annual_return", 0.0),
            sharpe_ratio=result.get("sharpe_ratio", 0.0),
            max_drawdown=result.get("max_drawdown", 0.0),
            volatility=result.get("volatility", 0.0),
            config=config,
            start_date=config.start_date if isinstance(config.start_date, datetime) else datetime.strptime(str(config.start_date), "%Y-%m-%d"),
            end_date=config.end_date if isinstance(config.end_date, datetime) else datetime.strptime(str(config.end_date), "%Y-%m-%d"),
            engine_name=self.ENGINE_NAME,
            engine_specific_results={
                "n_workers": result.get("n_workers", self.n_workers),
                "ray_used": result.get("ray_used", self.ray_initialized),
                "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                "n_periods": result.get("n_periods", 0),
            },
        )

    def _init_ray(self) -> None:
        """Ray初期化"""
        if not RAY_AVAILABLE:
            print("Ray not available, will use single-threaded fallback")
            return

        try:
            if self.config.ray_address:
                ray.init(address=self.config.ray_address, ignore_reinit_error=True)
            else:
                init_kwargs = {"ignore_reinit_error": True}
                if self.config.object_store_memory:
                    init_kwargs["object_store_memory"] = self.config.object_store_memory
                ray.init(**init_kwargs)

            self.ray_initialized = True
            print(f"Ray initialized with {self.n_workers} workers")

        except Exception as e:
            print(f"Ray initialization failed: {e}, using single-threaded fallback")
            self.ray_initialized = False

    def run_parallel(
        self,
        universe: List[str],
        price_data: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        並列バックテスト実行

        Args:
            universe: 銘柄リスト
            price_data: 価格データ辞書
            start_date: 開始日（省略時はconfig値）
            end_date: 終了日（省略時はconfig値）
            progress_callback: 進捗コールバック(completed, total)

        Returns:
            バックテスト結果辞書
        """
        start_time = time.perf_counter()

        config_dict = {
            "start_date": start_date or self.config.start_date,
            "end_date": end_date or self.config.end_date,
            "top_n": self.config.top_n,
            "lookback_days": self.config.lookback_days,
            "momentum_weight": self.config.momentum_weight,
            "reversion_weight": self.config.reversion_weight,
        }

        # ユニバースをチャンクに分割
        chunks = np.array_split(universe, self.n_workers)
        chunks = [list(c) for c in chunks if len(c) > 0]

        print(f"Running backtest: {len(universe)} tickers, {len(chunks)} chunks")

        if self.ray_initialized and RAY_AVAILABLE:
            results = self._run_with_ray(chunks, price_data, config_dict, progress_callback)
        else:
            results = self._run_single_threaded(chunks, price_data, config_dict, progress_callback)

        # 結果をマージ
        merged = self._merge_results(results)
        merged["elapsed_seconds"] = time.perf_counter() - start_time
        merged["n_workers"] = self.n_workers
        merged["ray_used"] = self.ray_initialized and RAY_AVAILABLE

        return merged

    def _run_with_ray(
        self,
        chunks: List[List[str]],
        price_data: Dict[str, Any],
        config: dict,
        progress_callback: Optional[Callable],
    ) -> List[BacktestChunkResult]:
        """Ray並列実行"""
        # 価格データをRayオブジェクトストアに格納
        price_data_ref = ray.put(price_data)

        # 全チャンクのタスクを投入
        futures = [
            _ray_backtest_worker.remote(i, chunk, price_data_ref, config)
            for i, chunk in enumerate(chunks)
        ]

        # 結果を収集
        results = []
        completed = 0
        total = len(futures)

        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            for future in done:
                result = ray.get(future)
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                print(f"  Chunk {result.chunk_id} completed: {len(result.tickers)} tickers, {result.n_periods} periods")

        return results

    def _run_single_threaded(
        self,
        chunks: List[List[str]],
        price_data: Dict[str, Any],
        config: dict,
        progress_callback: Optional[Callable],
    ) -> List[BacktestChunkResult]:
        """シングルスレッド実行（フォールバック）"""
        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            result = _run_backtest_chunk(i, chunk, price_data, config)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            print(f"  Chunk {i} completed: {len(result.tickers)} tickers, {result.n_periods} periods")

        return results

    def _merge_results(self, results: List[BacktestChunkResult]) -> Dict[str, Any]:
        """チャンク結果をマージ"""
        all_returns = []
        all_tickers = []
        total_periods = 0
        errors = []

        for r in results:
            if r.error:
                errors.append(f"Chunk {r.chunk_id}: {r.error}")
            else:
                all_returns.extend(r.returns)
                all_tickers.extend(r.tickers)
                total_periods = max(total_periods, r.n_periods)

        if not all_returns:
            return {
                "error": "No valid results",
                "errors": errors,
                "sharpe_ratio": 0,
                "annual_return": 0,
                "max_drawdown": 0,
            }

        returns_arr = np.array(all_returns)

        # メトリクス計算
        total_return = np.prod(1 + returns_arr) - 1
        n_periods = len(returns_arr)
        n_years = n_periods / 52

        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = np.std(returns_arr) * np.sqrt(52)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        # Max Drawdown
        cumulative = np.cumprod(1 + returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = float(np.min(drawdowns))

        # Sortino
        downside = returns_arr[returns_arr < 0]
        downside_vol = np.std(downside) * np.sqrt(52) if len(downside) > 0 else 0
        sortino = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "n_tickers": len(set(all_tickers)),
            "n_periods": n_periods,
            "n_chunks": len(results),
            "errors": errors if errors else None,
        }

    def shutdown(self) -> None:
        """Ray終了"""
        if self.ray_initialized and RAY_AVAILABLE:
            ray.shutdown()
            self.ray_initialized = False

    def __del__(self):
        """デストラクタ"""
        # Note: Rayはプロセス終了時に自動クリーンアップされるため、
        # 明示的なshutdownは通常不要
        pass


def run_benchmark(
    universe: List[str],
    price_data: Dict[str, Any],
    n_workers_list: List[int] = [1, 2, 4],
) -> Dict[str, Any]:
    """
    ベンチマーク実行

    Args:
        universe: 銘柄リスト
        price_data: 価格データ
        n_workers_list: テストするワーカー数リスト

    Returns:
        ベンチマーク結果
    """
    results = {}

    for n_workers in n_workers_list:
        print(f"\n=== Benchmark: {n_workers} workers ===")

        config = RayBacktestConfig(n_workers=n_workers)
        engine = RayBacktestEngine(config)

        start = time.perf_counter()
        result = engine.run_parallel(universe, price_data)
        elapsed = time.perf_counter() - start

        results[f"workers_{n_workers}"] = {
            "elapsed_seconds": elapsed,
            "sharpe_ratio": result.get("sharpe_ratio", 0),
            "ray_used": result.get("ray_used", False),
        }

        engine.shutdown()

        print(f"  Time: {elapsed:.2f}s, Sharpe: {result.get('sharpe_ratio', 0):.3f}")

    # スピードアップ計算
    if "workers_1" in results:
        baseline = results["workers_1"]["elapsed_seconds"]
        for key, val in results.items():
            if key != "workers_1":
                speedup = baseline / val["elapsed_seconds"] if val["elapsed_seconds"] > 0 else 0
                results[key]["speedup"] = speedup

    return results
