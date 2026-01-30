#!/usr/bin/env python3
"""
統一データキャッシュ構築スクリプト

全銘柄の2010-2024データを事前取得し、Parquet形式でキャッシュ。
バックテスト実行時はこのキャッシュを使用することで、
API呼び出しを削減し、実行時間を大幅に短縮する。

Usage:
    python scripts/build_standard_cache.py
    python scripts/build_standard_cache.py --verify  # キャッシュ検証のみ
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

# yfinance は遅延インポート（検証モードでは不要）
yf = None


def get_yfinance():
    """yfinanceを遅延インポート"""
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf
    return yf


# 設定
CACHE_DIR = Path("data/cache/standard_universe")
CONFIG_PATH = Path("config/universe_standard.yaml")
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
BATCH_SIZE = 50  # 一度に取得する銘柄数
SLEEP_BETWEEN_BATCHES = 2  # バッチ間の待機秒数
MAX_RETRIES = 3  # リトライ回数


def load_universe() -> list[str]:
    """universe_standard.yaml から全銘柄リストを読み込む"""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    tickers = []

    def extract_symbols(data: dict) -> list[str]:
        """ネストされた構造からシンボルを抽出"""
        symbols = []
        if "symbols" in data:
            symbols.extend(data["symbols"])
        else:
            # ネストされたサブカテゴリをチェック
            for key, value in data.items():
                if isinstance(value, dict) and "symbols" in value:
                    symbols.extend(value["symbols"])
        return symbols

    # universe_standard.yaml format: カテゴリごとに symbols リスト
    for category in ["us_stocks", "japan_stocks", "etfs", "forex"]:
        data = config.get(category, {})
        if isinstance(data, dict):
            tickers.extend(extract_symbols(data))

    # yfinance 用に forex シンボルを変換 (EURUSD_X -> EURUSD=X)
    tickers = [t.replace("_X", "=X") if t.endswith("_X") else t for t in tickers]

    return tickers


def sanitize_filename(symbol: str) -> str:
    """ファイル名として安全な形式に変換"""
    return symbol.replace("/", "_").replace("=", "_")


def fetch_single_ticker(symbol: str, retries: int = MAX_RETRIES) -> pl.DataFrame | None:
    """単一銘柄のデータを取得"""
    yfinance = get_yfinance()

    for attempt in range(retries):
        try:
            data = yfinance.download(
                symbol,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                print(f"  [WARN] {symbol}: No data available")
                return None

            # pandas -> polars 変換
            data = data.reset_index()
            df = pl.from_pandas(data)
            return df

        except Exception as e:
            if attempt < retries - 1:
                print(f"  [RETRY] {symbol}: {e} (attempt {attempt + 1}/{retries})")
                time.sleep(1)
            else:
                print(f"  [FAIL] {symbol}: {e}")
                return None

    return None


def build_cache(symbols: list[str], force: bool = False) -> dict:
    """全銘柄のデータを取得しキャッシュ"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }

    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        filename = sanitize_filename(symbol) + ".parquet"
        filepath = CACHE_DIR / filename

        # 既存ファイルをスキップ（forceでない場合）
        if not force and filepath.exists():
            results["skipped"].append(symbol)
            continue

        print(f"[{i}/{total}] Fetching {symbol}...")
        df = fetch_single_ticker(symbol)

        if df is not None:
            df.write_parquet(filepath)
            results["success"].append(symbol)
            print(f"  [OK] {symbol}: {len(df)} rows")
        else:
            results["failed"].append(symbol)

        # レート制限対策
        if i % BATCH_SIZE == 0:
            print(f"  ... sleeping {SLEEP_BETWEEN_BATCHES}s ...")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return results


def load_cache(symbol: str) -> pl.DataFrame | None:
    """キャッシュからデータ読み込み"""
    filename = sanitize_filename(symbol) + ".parquet"
    path = CACHE_DIR / filename

    if not path.exists():
        return None

    return pl.read_parquet(path)


def verify_cache(symbols: list[str]) -> dict:
    """キャッシュの検証"""
    results = {
        "found": [],
        "missing": [],
        "invalid": [],
        "stats": {},
    }

    total_rows = 0
    total_size = 0
    min_date = None
    max_date = None

    for symbol in symbols:
        filename = sanitize_filename(symbol) + ".parquet"
        filepath = CACHE_DIR / filename

        if not filepath.exists():
            results["missing"].append(symbol)
            continue

        try:
            df = pl.read_parquet(filepath)
            rows = len(df)
            size = filepath.stat().st_size

            if rows == 0:
                results["invalid"].append({"symbol": symbol, "reason": "empty"})
                continue

            # 日付範囲確認
            if "Date" in df.columns:
                dates = df["Date"]
                file_min = dates.min()
                file_max = dates.max()

                if min_date is None or file_min < min_date:
                    min_date = file_min
                if max_date is None or file_max > max_date:
                    max_date = file_max

            results["found"].append({
                "symbol": symbol,
                "rows": rows,
                "size_kb": round(size / 1024, 2),
            })

            total_rows += rows
            total_size += size

        except Exception as e:
            results["invalid"].append({"symbol": symbol, "reason": str(e)})

    results["stats"] = {
        "total_symbols": len(symbols),
        "found_count": len(results["found"]),
        "missing_count": len(results["missing"]),
        "invalid_count": len(results["invalid"]),
        "total_rows": total_rows,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "date_range": {
            "min": str(min_date) if min_date else None,
            "max": str(max_date) if max_date else None,
        },
    }

    return results


def save_metadata(build_results: dict, verify_results: dict):
    """メタデータファイルを保存"""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "source": str(CONFIG_PATH),
        },
        "build": {
            "success_count": len(build_results.get("success", [])),
            "failed_count": len(build_results.get("failed", [])),
            "skipped_count": len(build_results.get("skipped", [])),
            "failed_symbols": build_results.get("failed", []),
        },
        "verification": verify_results.get("stats", {}),
    }

    metadata_path = CACHE_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nMetadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Build standard universe data cache")
    parser.add_argument("--verify", action="store_true", help="Verify cache only")
    parser.add_argument("--force", action="store_true", help="Force rebuild all")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    args = parser.parse_args()

    print("=" * 60)
    print("Standard Universe Data Cache Builder")
    print("=" * 60)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    # 銘柄リスト読み込み
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = load_universe()

    print(f"Total symbols: {len(symbols)}")
    print()

    if args.verify:
        # 検証のみ
        print("Verifying cache...")
        verify_results = verify_cache(symbols)

        stats = verify_results["stats"]
        print(f"\n{'=' * 40}")
        print("Verification Results:")
        print(f"  Found: {stats['found_count']}/{stats['total_symbols']}")
        print(f"  Missing: {stats['missing_count']}")
        print(f"  Invalid: {stats['invalid_count']}")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        print(f"  Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")

        if verify_results["missing"]:
            print(f"\nMissing symbols ({len(verify_results['missing'])}):")
            for s in verify_results["missing"][:10]:
                print(f"  - {s}")
            if len(verify_results["missing"]) > 10:
                print(f"  ... and {len(verify_results['missing']) - 10} more")

    else:
        # キャッシュ構築
        print("Building cache...")
        build_results = build_cache(symbols, force=args.force)

        print(f"\n{'=' * 40}")
        print("Build Results:")
        print(f"  Success: {len(build_results['success'])}")
        print(f"  Failed: {len(build_results['failed'])}")
        print(f"  Skipped: {len(build_results['skipped'])}")

        if build_results["failed"]:
            print(f"\nFailed symbols ({len(build_results['failed'])}):")
            for s in build_results["failed"]:
                print(f"  - {s}")

        # 検証実行
        print("\nVerifying cache...")
        verify_results = verify_cache(symbols)

        # メタデータ保存
        save_metadata(build_results, verify_results)

        stats = verify_results["stats"]
        print(f"\nFinal Stats:")
        print(f"  Cached: {stats['found_count']}/{stats['total_symbols']}")
        print(f"  Total size: {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
