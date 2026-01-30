"""
DuckDB Unified Data Layer - 統合データレイヤー

複数のデータソース（CSV, Parquet）を統合し、
高速なSQLクエリアクセスを提供する。

Key Features:
- Parquetファイルの自動スキャン・統合
- SQLによる柔軟なクエリ
- 価格マトリクスの高速生成（PIVOT操作）
- Polars DataFrameとの連携
- メモリ/永続化の選択可能

Based on HI-003: DuckDB integrated data layer for performance.

Expected Effect:
- Data load: 5-10x faster
- Query flexibility: SQL support

Usage:
    layer = DuckDBDataLayer()

    # キャッシュディレクトリをスキャン
    layer.load_cache_directory("cache/price_data")

    # SQLクエリ
    df = layer.query("SELECT * FROM prices WHERE symbol = 'SPY' AND date > '2020-01-01'")

    # 価格マトリクス生成
    matrix = layer.get_price_matrix(['SPY', 'TLT', 'GLD'], '2020-01-01', '2024-01-01')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DuckDBLayerConfig:
    """DuckDBレイヤー設定

    Attributes:
        db_path: データベースパス（":memory:"でインメモリ）
        read_only: 読み取り専用モード
        threads: 並列スレッド数
        memory_limit: メモリ制限
        enable_progress: プログレスバー表示
    """
    db_path: str = ":memory:"
    read_only: bool = False
    threads: int | None = None
    memory_limit: str | None = None
    enable_progress: bool = False

    def to_duckdb_config(self) -> dict[str, Any]:
        """DuckDB接続設定に変換"""
        config = {}
        if self.threads is not None:
            config["threads"] = self.threads
        if self.memory_limit is not None:
            config["memory_limit"] = self.memory_limit
        return config


@dataclass
class TableInfo:
    """テーブル情報

    Attributes:
        name: テーブル名
        row_count: 行数
        column_count: カラム数
        columns: カラム名リスト
        source_path: 元ファイルパス
    """
    name: str
    row_count: int
    column_count: int
    columns: list[str]
    source_path: str | None = None


@dataclass
class QueryResult:
    """クエリ結果

    Attributes:
        data: Polars DataFrame
        execution_time_ms: 実行時間（ミリ秒）
        row_count: 行数
    """
    data: pl.DataFrame
    execution_time_ms: float
    row_count: int


class DuckDBDataLayer:
    """
    DuckDB統合データレイヤー

    複数のデータソースを統合し、SQLクエリによる
    高速なデータアクセスを提供する。

    Features:
    - Parquet/CSVファイルの自動読み込み
    - 複数テーブルの統合（UNION）
    - SQLクエリによる柔軟なフィルタリング
    - 価格マトリクスのPIVOT生成
    - Polars DataFrame出力

    Usage:
        # インメモリモード
        layer = DuckDBDataLayer()

        # 永続化モード
        layer = DuckDBDataLayer(db_path="data/prices.duckdb")

        # キャッシュ読み込み
        layer.load_cache_directory("cache/price_data")

        # クエリ
        df = layer.query("SELECT * FROM unified_prices WHERE symbol = 'SPY'")

        # 価格マトリクス
        matrix = layer.get_price_matrix(['SPY', 'TLT'], '2020-01-01', '2024-01-01')
    """

    UNIFIED_TABLE = "unified_prices"

    def __init__(
        self,
        config: DuckDBLayerConfig | None = None,
        db_path: str = ":memory:",
    ) -> None:
        """初期化

        Args:
            config: 設定（指定時はdb_pathより優先）
            db_path: データベースパス

        Raises:
            ImportError: DuckDBがインストールされていない場合
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError(
                "duckdb is required for DuckDBDataLayer. "
                "Install with: pip install duckdb"
            )

        if config is not None:
            self.config = config
        else:
            self.config = DuckDBLayerConfig(db_path=db_path)

        # DuckDB接続
        self._conn = duckdb.connect(
            self.config.db_path,
            read_only=self.config.read_only,
            config=self.config.to_duckdb_config(),
        )

        # プログレスバー設定
        if not self.config.enable_progress:
            self._conn.execute("SET enable_progress_bar = false")

        # テーブル情報キャッシュ
        self._tables: dict[str, TableInfo] = {}
        self._loaded_files: set[str] = set()

        logger.info(f"DuckDBDataLayer initialized: {self.config.db_path}")

    def __enter__(self) -> "DuckDBDataLayer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """接続を閉じる"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("DuckDB connection closed")

    def load_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        replace: bool = False,
    ) -> TableInfo:
        """Parquetファイルをテーブルとして読み込み

        Args:
            path: Parquetファイルパス
            table_name: テーブル名（None=ファイル名から生成）
            replace: 既存テーブルを置換

        Returns:
            TableInfo
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        if table_name is None:
            table_name = path.stem.replace("-", "_").replace(".", "_")

        # 既存テーブルチェック
        if table_name in self._tables and not replace:
            logger.debug(f"Table {table_name} already exists, skipping")
            return self._tables[table_name]

        # テーブル作成
        if replace:
            self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self._conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{path}')
        """)

        # テーブル情報取得
        info = self._get_table_info(table_name, str(path))
        self._tables[table_name] = info
        self._loaded_files.add(str(path))

        logger.info(f"Loaded parquet: {path} -> {table_name} ({info.row_count} rows)")
        return info

    def load_csv(
        self,
        path: str | Path,
        table_name: str | None = None,
        replace: bool = False,
        **csv_options,
    ) -> TableInfo:
        """CSVファイルをテーブルとして読み込み

        Args:
            path: CSVファイルパス
            table_name: テーブル名
            replace: 既存テーブルを置換
            **csv_options: CSVオプション（header, delimiter等）

        Returns:
            TableInfo
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if table_name is None:
            table_name = path.stem.replace("-", "_").replace(".", "_")

        if table_name in self._tables and not replace:
            logger.debug(f"Table {table_name} already exists, skipping")
            return self._tables[table_name]

        # CSVオプション構築
        options = []
        if "header" in csv_options:
            options.append(f"header={csv_options['header']}")
        if "delimiter" in csv_options:
            options.append(f"delim='{csv_options['delimiter']}'")

        options_str = ", ".join(options) if options else ""

        if replace:
            self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self._conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv('{path}'{', ' + options_str if options_str else ''})
        """)

        info = self._get_table_info(table_name, str(path))
        self._tables[table_name] = info
        self._loaded_files.add(str(path))

        logger.info(f"Loaded CSV: {path} -> {table_name} ({info.row_count} rows)")
        return info

    def load_cache_directory(
        self,
        cache_dir: str | Path,
        pattern: str = "*.parquet",
        create_unified: bool = True,
    ) -> list[TableInfo]:
        """キャッシュディレクトリを一括読み込み

        Args:
            cache_dir: キャッシュディレクトリ
            pattern: ファイルパターン
            create_unified: 統合テーブルを作成

        Returns:
            読み込んだテーブル情報のリスト
        """
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            logger.warning(f"Cache directory not found: {cache_dir}")
            return []

        files = list(cache_dir.glob(pattern))
        if not files:
            logger.warning(f"No files matching {pattern} in {cache_dir}")
            return []

        loaded = []
        for file_path in files:
            if str(file_path) in self._loaded_files:
                continue
            try:
                info = self.load_parquet(file_path)
                loaded.append(info)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        # 統合テーブル作成
        if create_unified and loaded:
            self._create_unified_table()

        logger.info(f"Loaded {len(loaded)} files from {cache_dir}")
        return loaded

    def _create_unified_table(self) -> None:
        """全テーブルを統合したビューを作成"""
        if not self._tables:
            return

        # 共通カラムを特定
        all_columns = set()
        for info in self._tables.values():
            all_columns.update(info.columns)

        # 価格データに必要なカラム
        required_columns = {"date", "open", "high", "low", "close", "volume"}
        common_columns = required_columns.intersection(all_columns)

        if len(common_columns) < 4:  # 最低限date, open, high, low, closeが必要
            logger.warning("Insufficient common columns for unified table")
            return

        # 各テーブルからUNION ALL
        unions = []
        for table_name, info in self._tables.items():
            # シンボル名を抽出（テーブル名から）
            symbol = table_name.upper()

            select_cols = []
            for col in ["date", "open", "high", "low", "close", "volume"]:
                if col in info.columns:
                    select_cols.append(col)
                else:
                    select_cols.append(f"NULL as {col}")

            unions.append(f"""
                SELECT '{symbol}' as symbol, {', '.join(select_cols)}
                FROM {table_name}
            """)

        union_sql = " UNION ALL ".join(unions)

        self._conn.execute(f"""
            CREATE OR REPLACE VIEW {self.UNIFIED_TABLE} AS
            {union_sql}
        """)

        logger.info(f"Created unified view: {self.UNIFIED_TABLE}")

    def _get_table_info(self, table_name: str, source_path: str | None = None) -> TableInfo:
        """テーブル情報を取得"""
        # カラム情報
        columns_result = self._conn.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).fetchall()
        columns = [row[0] for row in columns_result]

        # 行数
        count_result = self._conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        row_count = count_result[0] if count_result else 0

        return TableInfo(
            name=table_name,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            source_path=source_path,
        )

    def query(self, sql: str) -> pl.DataFrame:
        """SQLクエリを実行してPolars DataFrameを返す

        Args:
            sql: SQLクエリ

        Returns:
            Polars DataFrame
        """
        return self._conn.execute(sql).pl()

    def query_with_timing(self, sql: str) -> QueryResult:
        """タイミング付きでSQLクエリを実行

        Args:
            sql: SQLクエリ

        Returns:
            QueryResult
        """
        import time

        start = time.perf_counter()
        df = self._conn.execute(sql).pl()
        elapsed = (time.perf_counter() - start) * 1000

        return QueryResult(
            data=df,
            execution_time_ms=elapsed,
            row_count=len(df),
        )

    def get_price_matrix(
        self,
        tickers: list[str],
        start: str,
        end: str,
        price_col: str = "close",
        table: str | None = None,
    ) -> pl.DataFrame:
        """価格マトリクスを高速生成

        銘柄をカラム、日付を行とするマトリクス形式を返す。

        Args:
            tickers: 銘柄リスト
            start: 開始日（YYYY-MM-DD）
            end: 終了日（YYYY-MM-DD）
            price_col: 価格カラム（close, open, high, low）
            table: テーブル名（None=統合テーブル）

        Returns:
            Polars DataFrame (date, ticker1, ticker2, ...)
        """
        table = table or self.UNIFIED_TABLE

        # 銘柄リストをSQL用に整形
        tickers_sql = ", ".join([f"'{t.upper()}'" for t in tickers])

        # PIVOTクエリ
        sql = f"""
            PIVOT (
                SELECT date, symbol, {price_col}
                FROM {table}
                WHERE symbol IN ({tickers_sql})
                  AND date >= '{start}'
                  AND date <= '{end}'
            )
            ON symbol
            USING FIRST({price_col})
            ORDER BY date
        """

        try:
            df = self._conn.execute(sql).pl()
        except Exception as e:
            # PIVOTが使えない場合はPolarsで処理
            logger.debug(f"PIVOT failed, falling back to Polars: {e}")
            df = self._pivot_fallback(tickers, start, end, price_col, table)

        return df

    def _pivot_fallback(
        self,
        tickers: list[str],
        start: str,
        end: str,
        price_col: str,
        table: str,
    ) -> pl.DataFrame:
        """PIVOTのフォールバック実装"""
        tickers_sql = ", ".join([f"'{t.upper()}'" for t in tickers])

        sql = f"""
            SELECT date, symbol, {price_col}
            FROM {table}
            WHERE symbol IN ({tickers_sql})
              AND date >= '{start}'
              AND date <= '{end}'
            ORDER BY date, symbol
        """

        df = self._conn.execute(sql).pl()

        if df.is_empty():
            return pl.DataFrame()

        # Polarsでpivot
        return df.pivot(
            values=price_col,
            index="date",
            on="symbol",
        ).sort("date")

    def get_returns_matrix(
        self,
        tickers: list[str],
        start: str,
        end: str,
        price_col: str = "close",
    ) -> pl.DataFrame:
        """リターンマトリクスを計算

        Args:
            tickers: 銘柄リスト
            start: 開始日
            end: 終了日
            price_col: 価格カラム

        Returns:
            リターンマトリクス
        """
        prices = self.get_price_matrix(tickers, start, end, price_col)

        if prices.is_empty():
            return pl.DataFrame()

        # 日次リターン計算
        date_col = prices.columns[0]  # 最初のカラムがdate
        price_cols = prices.columns[1:]

        returns_exprs = [
            (pl.col(col) / pl.col(col).shift(1) - 1).alias(col)
            for col in price_cols
        ]

        returns = prices.select(
            pl.col(date_col),
            *returns_exprs
        ).slice(1)  # 最初の行はNaN

        return returns

    def get_symbols(self, table: str | None = None) -> list[str]:
        """利用可能な銘柄リストを取得

        Args:
            table: テーブル名

        Returns:
            銘柄リスト
        """
        table = table or self.UNIFIED_TABLE

        try:
            result = self._conn.execute(f"""
                SELECT DISTINCT symbol FROM {table} ORDER BY symbol
            """).fetchall()
            return [row[0] for row in result]
        except Exception:
            return list(self._tables.keys())

    def get_date_range(self, table: str | None = None) -> tuple[str, str] | None:
        """データの日付範囲を取得

        Args:
            table: テーブル名

        Returns:
            (開始日, 終了日) or None
        """
        table = table or self.UNIFIED_TABLE

        try:
            result = self._conn.execute(f"""
                SELECT MIN(date), MAX(date) FROM {table}
            """).fetchone()
            if result and result[0] is not None:
                return (str(result[0]), str(result[1]))
        except Exception:
            pass

        return None

    def list_tables(self) -> list[TableInfo]:
        """登録済みテーブル一覧を取得

        Returns:
            TableInfoのリスト
        """
        return list(self._tables.values())

    def execute(self, sql: str) -> Any:
        """生SQLを実行（DDL等用）

        Args:
            sql: SQLクエリ

        Returns:
            実行結果
        """
        return self._conn.execute(sql)

    def register_dataframe(
        self,
        df: pl.DataFrame,
        table_name: str,
        replace: bool = False,
    ) -> TableInfo:
        """Polars DataFrameをテーブルとして登録

        Args:
            df: Polars DataFrame
            table_name: テーブル名
            replace: 既存テーブルを置換

        Returns:
            TableInfo
        """
        if table_name in self._tables and not replace:
            logger.debug(f"Table {table_name} already exists, skipping")
            return self._tables[table_name]

        # DuckDBにDataFrameを登録
        if replace:
            self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self._conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        info = self._get_table_info(table_name)
        self._tables[table_name] = info

        logger.info(f"Registered DataFrame as {table_name} ({info.row_count} rows)")
        return info

    def get_summary(self) -> dict[str, Any]:
        """レイヤーのサマリーを取得"""
        total_rows = sum(t.row_count for t in self._tables.values())

        return {
            "db_path": self.config.db_path,
            "table_count": len(self._tables),
            "total_rows": total_rows,
            "loaded_files": len(self._loaded_files),
            "tables": [
                {"name": t.name, "rows": t.row_count, "columns": t.column_count}
                for t in self._tables.values()
            ],
        }


# =============================================================================
# 便利関数
# =============================================================================

def create_duckdb_layer(
    db_path: str = ":memory:",
    cache_dir: str | Path | None = None,
    threads: int | None = None,
) -> DuckDBDataLayer:
    """DuckDBDataLayerを作成（ファクトリ関数）

    Args:
        db_path: データベースパス
        cache_dir: 自動読み込みするキャッシュディレクトリ
        threads: 並列スレッド数

    Returns:
        DuckDBDataLayer
    """
    config = DuckDBLayerConfig(db_path=db_path, threads=threads)
    layer = DuckDBDataLayer(config=config)

    if cache_dir is not None:
        layer.load_cache_directory(cache_dir)

    return layer


def quick_price_matrix(
    cache_dir: str | Path,
    tickers: list[str],
    start: str,
    end: str,
) -> pl.DataFrame:
    """キャッシュから価格マトリクスを高速生成（ワンライナー）

    Args:
        cache_dir: キャッシュディレクトリ
        tickers: 銘柄リスト
        start: 開始日
        end: 終了日

    Returns:
        価格マトリクス
    """
    with DuckDBDataLayer() as layer:
        layer.load_cache_directory(cache_dir)
        return layer.get_price_matrix(tickers, start, end)


def quick_query(
    cache_dir: str | Path,
    sql: str,
) -> pl.DataFrame:
    """キャッシュに対してSQLクエリを実行（ワンライナー）

    Args:
        cache_dir: キャッシュディレクトリ
        sql: SQLクエリ

    Returns:
        クエリ結果
    """
    with DuckDBDataLayer() as layer:
        layer.load_cache_directory(cache_dir)
        return layer.query(sql)
