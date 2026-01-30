"""
Streaming Backtest Engine - メモリ効率的なストリーミングバックテストエンジン

大規模ユニバース（800+銘柄）のバックテストをメモリ効率的に実行するためのエンジン。
チャンク単位で処理し、中間結果をディスクに保存することで、長時間実行の中断・再開が可能。

主な機能:
- チャンク単位処理（メモリ節約）
- 中間結果のディスク保存（再開可能）
- 進捗表示
- FastBacktestEngine統合

使用例:
    from src.backtest.streaming_engine import StreamingBacktestEngine, StreamingBacktestResult

    engine = StreamingBacktestEngine(
        chunk_size=50,
        temp_dir="cache/streaming_temp",
        save_interval=5
    )

    result = engine.run(
        tickers=["SPY", "QQQ", ...],  # 800+銘柄
        price_data=price_dict,
        start_date="2010-01-01",
        end_date="2024-12-31",
        rebalance_freq="monthly",
        progress_callback=lambda i, total, tickers: print(f"{i}/{total}")
    )

    print(f"Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Annual Return: {result.annual_return:.2%}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Import base class for unified interface
from src.backtest.base import (
    BacktestEngineBase,
    UnifiedBacktestConfig,
    UnifiedBacktestResult,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingBacktestResult:
    """
    ストリーミングバックテスト結果

    Attributes:
        total_return: 累積リターン
        annual_return: 年率リターン
        sharpe_ratio: シャープレシオ
        max_drawdown: 最大ドローダウン
        volatility: ボラティリティ（年率）
        num_trades: 取引回数
        final_value: 最終ポートフォリオ価値
        chunk_results: チャンク別結果リスト
        run_id: 実行ID
        elapsed_seconds: 実行時間（秒）
    """

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    num_trades: int
    final_value: float
    chunk_results: List[dict] = field(default_factory=list)
    run_id: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "num_trades": self.num_trades,
            "final_value": self.final_value,
            "run_id": self.run_id,
            "elapsed_seconds": self.elapsed_seconds,
            "num_chunks": len(self.chunk_results),
            "success_chunks": sum(
                1 for cr in self.chunk_results if cr.get("status") == "success"
            ),
        }

    def summary(self) -> str:
        """サマリ文字列を生成"""
        return (
            f"StreamingBacktestResult:\n"
            f"  Run ID: {self.run_id}\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annual Return: {self.annual_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Volatility: {self.volatility:.2%}\n"
            f"  Trades: {self.num_trades}\n"
            f"  Final Value: ${self.final_value:,.2f}\n"
            f"  Elapsed: {self.elapsed_seconds:.1f}s\n"
            f"  Chunks: {len(self.chunk_results)} "
            f"({sum(1 for cr in self.chunk_results if cr.get('status') == 'success')} success)"
        )


class StreamingBacktestEngine(BacktestEngineBase):
    """
    メモリ効率的なストリーミングバックテストエンジン

    大規模ユニバースを効率的に処理するため、銘柄をチャンクに分割して
    順次処理する。中間結果はディスクに保存され、中断しても再開可能。

    BacktestEngineBase準拠: 統一インターフェースで呼び出し可能。

    Attributes:
        chunk_size: チャンクあたりの銘柄数
        temp_dir: 一時ファイル保存先
        save_interval: チェックポイント保存間隔（チャンク数）
    """

    ENGINE_NAME: str = "streaming"

    def __init__(
        self,
        chunk_size: int = 50,
        temp_dir: str = "cache/streaming_temp",
        save_interval: int = 5,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        config: Optional[UnifiedBacktestConfig] = None,
    ):
        """
        初期化

        Parameters
        ----------
        chunk_size : int
            チャンクあたりの銘柄数（デフォルト: 50）
        temp_dir : str
            一時ファイル保存先（デフォルト: cache/streaming_temp）
        save_interval : int
            チェックポイント保存間隔（デフォルト: 5チャンク）
        initial_capital : float
            初期資金（デフォルト: 100,000）
        transaction_cost_bps : float
            取引コスト（bps）（デフォルト: 10）
        config : Optional[UnifiedBacktestConfig]
            統一バックテスト設定
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps

        logger.info(
            f"StreamingBacktestEngine initialized: "
            f"chunk_size={chunk_size}, temp_dir={temp_dir}"
        )

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
        # 設定の決定
        cfg = config or self._config
        if cfg is None:
            cfg = UnifiedBacktestConfig(
                start_date="2010-01-01",
                end_date="2024-12-31",
                initial_capital=self.initial_capital,
                transaction_cost_bps=self.transaction_cost_bps,
                rebalance_frequency="monthly",
            )

        # 入力検証
        self.validate_inputs(universe, prices, cfg)

        # 内部メソッドで実行
        result = self.run_streaming(
            tickers=universe,
            price_data=prices,
            start_date=cfg.start_date.strftime("%Y-%m-%d") if isinstance(cfg.start_date, datetime) else cfg.start_date,
            end_date=cfg.end_date.strftime("%Y-%m-%d") if isinstance(cfg.end_date, datetime) else cfg.end_date,
            rebalance_freq=cfg.rebalance_frequency,
            config=cfg.engine_specific_config,
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
            logger.warning(warning)

        # ストリーミング固有の検証
        if len(universe) == 0:
            raise ValueError("Universe cannot be empty")

        return True

    def _convert_to_unified_result(
        self,
        result: StreamingBacktestResult,
        config: UnifiedBacktestConfig,
    ) -> UnifiedBacktestResult:
        """StreamingBacktestResult を UnifiedBacktestResult に変換"""
        # ポートフォリオ価値の推定（チャンク結果から）
        portfolio_values = pd.Series(dtype=float)
        daily_returns = pd.Series(dtype=float)

        # 統一結果を生成
        unified = UnifiedBacktestResult(
            total_return=result.total_return,
            annual_return=result.annual_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            volatility=result.volatility,
            config=config,
            start_date=config.start_date if isinstance(config.start_date, datetime) else datetime.strptime(config.start_date, "%Y-%m-%d"),
            end_date=config.end_date if isinstance(config.end_date, datetime) else datetime.strptime(config.end_date, "%Y-%m-%d"),
            engine_name=self.ENGINE_NAME,
            engine_specific_results={
                "run_id": result.run_id,
                "elapsed_seconds": result.elapsed_seconds,
                "chunk_results": result.chunk_results,
                "num_trades": result.num_trades,
                "final_value": result.final_value,
            },
        )

        return unified

    def _chunk_tickers(self, tickers: List[str]) -> Iterator[List[str]]:
        """
        銘柄をチャンクに分割

        Parameters
        ----------
        tickers : List[str]
            銘柄リスト

        Yields
        ------
        List[str]
            チャンク（銘柄リスト）
        """
        for i in range(0, len(tickers), self.chunk_size):
            yield tickers[i : i + self.chunk_size]

    def _save_checkpoint(self, checkpoint_id: str, data: dict) -> None:
        """
        チェックポイントを保存

        Parameters
        ----------
        checkpoint_id : str
            チェックポイントID
        data : dict
            保存するデータ
        """
        path = self.temp_dir / f"checkpoint_{checkpoint_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, default=str)
        logger.debug(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[dict]:
        """
        チェックポイントを読み込み

        Parameters
        ----------
        checkpoint_id : str
            チェックポイントID

        Returns
        -------
        Optional[dict]
            チェックポイントデータ（存在しない場合はNone）
        """
        path = self.temp_dir / f"checkpoint_{checkpoint_id}.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {path}")
            return data
        return None

    def _process_chunk(
        self,
        chunk_tickers: List[str],
        price_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        start_date: str,
        end_date: str,
        rebalance_freq: str,
        config: dict,
    ) -> dict:
        """
        チャンク単位でバックテスト実行

        Parameters
        ----------
        chunk_tickers : List[str]
            チャンク内の銘柄リスト
        price_data : Union[Dict[str, pd.DataFrame], pd.DataFrame]
            価格データ（銘柄->DataFrame の辞書、または銘柄列のDataFrame）
        start_date : str
            開始日
        end_date : str
            終了日
        rebalance_freq : str
            リバランス頻度
        config : dict
            追加設定

        Returns
        -------
        dict
            チャンク処理結果
        """
        from .fast_engine import FastBacktestConfig, FastBacktestEngine

        # 価格データを抽出
        if isinstance(price_data, dict):
            # 辞書形式: {ticker: DataFrame}
            chunk_prices = {}
            for t in chunk_tickers:
                if t in price_data:
                    chunk_prices[t] = price_data[t]
        else:
            # DataFrame形式: 列が銘柄
            available = [t for t in chunk_tickers if t in price_data.columns]
            if not available:
                return {
                    "tickers": chunk_tickers,
                    "returns": [],
                    "sharpe": 0.0,
                    "final_value": 0.0,
                    "status": "no_data",
                }
            chunk_prices = price_data[available]

        if isinstance(chunk_prices, dict) and len(chunk_prices) == 0:
            return {
                "tickers": chunk_tickers,
                "returns": [],
                "sharpe": 0.0,
                "final_value": 0.0,
                "status": "no_data",
            }
        if isinstance(chunk_prices, pd.DataFrame) and chunk_prices.empty:
            return {
                "tickers": chunk_tickers,
                "returns": [],
                "sharpe": 0.0,
                "final_value": 0.0,
                "status": "no_data",
            }

        # DataFrame に変換
        if isinstance(chunk_prices, dict):
            # 辞書からDataFrameに変換
            try:
                # 各銘柄のclose価格を結合
                close_data = {}
                for ticker, df in chunk_prices.items():
                    if isinstance(df, pd.DataFrame):
                        if "close" in df.columns:
                            close_data[ticker] = df["close"]
                        elif "Close" in df.columns:
                            close_data[ticker] = df["Close"]
                        else:
                            # 最初の列を使用
                            close_data[ticker] = df.iloc[:, 0]
                    else:
                        close_data[ticker] = df
                prices_df = pd.DataFrame(close_data)
            except Exception as e:
                logger.warning(f"Failed to convert price data: {e}")
                return {
                    "tickers": chunk_tickers,
                    "returns": [],
                    "sharpe": 0.0,
                    "final_value": 0.0,
                    "status": f"data_error: {str(e)}",
                }
        else:
            prices_df = chunk_prices

        # 日付でフィルタ
        if isinstance(prices_df.index, pd.DatetimeIndex):
            mask = (prices_df.index >= start_date) & (prices_df.index <= end_date)
            prices_df = prices_df.loc[mask]
        else:
            prices_df.index = pd.to_datetime(prices_df.index)
            mask = (prices_df.index >= start_date) & (prices_df.index <= end_date)
            prices_df = prices_df.loc[mask]

        if len(prices_df) == 0:
            return {
                "tickers": chunk_tickers,
                "returns": [],
                "sharpe": 0.0,
                "final_value": 0.0,
                "status": "no_data_in_period",
            }

        # FastBacktestEngine で実行
        try:
            fast_config = FastBacktestConfig(
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                rebalance_frequency=rebalance_freq,
                initial_capital=self.initial_capital,
                transaction_cost_bps=self.transaction_cost_bps,
            )
            engine = FastBacktestEngine(fast_config)

            result = engine.run(prices_df)

            # 結果を抽出
            returns = []
            if hasattr(result, "snapshots") and result.snapshots:
                returns = [s.daily_return for s in result.snapshots if s.daily_return != 0]

            sharpe = result.sharpe_ratio if hasattr(result, "sharpe_ratio") else 0.0
            final_value = (
                result.snapshots[-1].portfolio_value
                if hasattr(result, "snapshots") and result.snapshots
                else self.initial_capital
            )

            return {
                "tickers": list(prices_df.columns),
                "returns": returns,
                "sharpe": float(sharpe) if sharpe is not None else 0.0,
                "final_value": float(final_value),
                "total_return": float(result.total_return) if hasattr(result, "total_return") else 0.0,
                "status": "success",
            }

        except Exception as e:
            logger.warning(f"Chunk processing failed: {e}")
            return {
                "tickers": chunk_tickers,
                "returns": [],
                "sharpe": 0.0,
                "final_value": 0.0,
                "status": f"error: {str(e)}",
            }

    def run_streaming(
        self,
        tickers: List[str],
        price_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        start_date: str,
        end_date: str,
        rebalance_freq: str = "monthly",
        config: Optional[dict] = None,
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, List[str]], None]] = None,
    ) -> StreamingBacktestResult:
        """
        ストリーミングバックテスト実行（内部メソッド）

        銘柄をチャンク分割して処理、中間結果をディスクに保存。
        中断しても resume_from でIDを指定して再開可能。

        Note: 統一インターフェースは run() を使用。こちらは内部/後方互換用。

        Parameters
        ----------
        tickers : List[str]
            銘柄リスト
        price_data : Union[Dict[str, pd.DataFrame], pd.DataFrame]
            価格データ
        start_date : str
            開始日（YYYY-MM-DD）
        end_date : str
            終了日（YYYY-MM-DD）
        rebalance_freq : str
            リバランス頻度（daily/weekly/monthly/quarterly）
        config : Optional[dict]
            追加設定
        resume_from : Optional[str]
            再開するrun_id
        progress_callback : Optional[Callable]
            進捗コールバック関数 (current, total, tickers)

        Returns
        -------
        StreamingBacktestResult
            バックテスト結果
        """
        import time

        start_time = time.perf_counter()
        config = config or {}
        run_id = resume_from or datetime.now().strftime("%Y%m%d_%H%M%S")

        # チェックポイントから再開
        checkpoint = self._load_checkpoint(run_id) if resume_from else None
        processed_chunks = checkpoint.get("processed_chunks", 0) if checkpoint else 0
        chunk_results = checkpoint.get("chunk_results", []) if checkpoint else []

        chunks = list(self._chunk_tickers(tickers))
        total_chunks = len(chunks)

        logger.info(
            f"Starting streaming backtest: {len(tickers)} tickers, "
            f"{total_chunks} chunks, chunk_size={self.chunk_size}"
        )

        if processed_chunks > 0:
            logger.info(f"Resuming from chunk {processed_chunks}/{total_chunks}")

        for i, chunk_tickers in enumerate(chunks):
            if i < processed_chunks:
                continue  # 既処理スキップ

            # チャンク処理
            result = self._process_chunk(
                chunk_tickers,
                price_data,
                start_date,
                end_date,
                rebalance_freq,
                config,
            )
            chunk_results.append(result)

            # 進捗報告
            if progress_callback:
                progress_callback(i + 1, total_chunks, chunk_tickers)

            logger.info(
                f"Chunk {i + 1}/{total_chunks} completed: "
                f"{len(chunk_tickers)} tickers, status={result.get('status')}"
            )

            # チェックポイント保存
            if (i + 1) % self.save_interval == 0:
                self._save_checkpoint(
                    run_id,
                    {
                        "processed_chunks": i + 1,
                        "chunk_results": chunk_results,
                    },
                )

        # 最終チェックポイント保存
        self._save_checkpoint(
            run_id,
            {
                "processed_chunks": total_chunks,
                "chunk_results": chunk_results,
                "completed": True,
            },
        )

        elapsed = time.perf_counter() - start_time

        # 最終結果を集計
        final_result = self._aggregate_results(chunk_results)
        final_result.run_id = run_id
        final_result.elapsed_seconds = elapsed

        logger.info(
            f"Streaming backtest completed: "
            f"{final_result.sharpe_ratio:.2f} Sharpe, "
            f"{final_result.annual_return:.2%} annual return, "
            f"{elapsed:.1f}s elapsed"
        )

        return final_result

    def _aggregate_results(self, chunk_results: List[dict]) -> StreamingBacktestResult:
        """
        チャンク結果を集計

        Parameters
        ----------
        chunk_results : List[dict]
            チャンク処理結果のリスト

        Returns
        -------
        StreamingBacktestResult
            集計結果
        """
        all_returns = []
        total_value = 0.0
        success_count = 0

        for cr in chunk_results:
            if cr.get("status") == "success":
                all_returns.extend(cr.get("returns", []))
                total_value += cr.get("final_value", 0.0)
                success_count += 1

        if not all_returns:
            return StreamingBacktestResult(
                total_return=0.0,
                annual_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                num_trades=0,
                final_value=0.0,
                chunk_results=chunk_results,
            )

        returns = np.array(all_returns)

        # Total return
        total_return = float(np.prod(1 + returns) - 1)

        # Annual return
        n_days = len(returns)
        annual_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

        # Volatility
        volatility = float(np.std(returns) * np.sqrt(252))

        # Sharpe ratio
        sharpe = annual_return / volatility if volatility > 0 else 0.0

        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = float(np.min(drawdown))

        return StreamingBacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            num_trades=len(returns),
            final_value=total_value / max(success_count, 1),  # 平均
            chunk_results=chunk_results,
        )

    def clean_checkpoints(self, run_id: Optional[str] = None) -> int:
        """
        チェックポイントファイルを削除

        Parameters
        ----------
        run_id : Optional[str]
            削除対象のrun_id（Noneの場合は全て削除）

        Returns
        -------
        int
            削除したファイル数
        """
        deleted = 0
        if run_id:
            path = self.temp_dir / f"checkpoint_{run_id}.json"
            if path.exists():
                path.unlink()
                deleted = 1
        else:
            for path in self.temp_dir.glob("checkpoint_*.json"):
                path.unlink()
                deleted += 1
        logger.info(f"Deleted {deleted} checkpoint files")
        return deleted


# ショートカット関数
def run_streaming_backtest(
    tickers: List[str],
    price_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    start_date: str,
    end_date: str,
    chunk_size: int = 50,
    rebalance_freq: str = "monthly",
    progress_callback: Optional[Callable[[int, int, List[str]], None]] = None,
) -> StreamingBacktestResult:
    """
    ストリーミングバックテストのショートカット関数

    Parameters
    ----------
    tickers : List[str]
        銘柄リスト
    price_data : Union[Dict[str, pd.DataFrame], pd.DataFrame]
        価格データ
    start_date : str
        開始日
    end_date : str
        終了日
    chunk_size : int
        チャンクサイズ
    rebalance_freq : str
        リバランス頻度
    progress_callback : Optional[Callable]
        進捗コールバック

    Returns
    -------
    StreamingBacktestResult
        バックテスト結果
    """
    engine = StreamingBacktestEngine(chunk_size=chunk_size)
    return engine.run(
        tickers=tickers,
        price_data=price_data,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        progress_callback=progress_callback,
    )
