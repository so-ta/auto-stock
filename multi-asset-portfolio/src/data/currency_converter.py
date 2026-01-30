"""
Currency Converter Module - 為替レート変換

価格データを USD に変換するためのユーティリティ。
yfinance 経由で為替レートを取得し、日次でキャッシュする。

機能:
1. convert_to_usd: 価格シリーズを USD に変換
2. get_fx_rate: 特定日の為替レートを取得
3. get_fx_history: 期間の為替レート履歴を取得

対応通貨ペア:
- JPY→USD: USDJPY=X の逆数
- EUR→USD: EURUSD=X
- GBP→USD: GBPUSD=X
- AUD→USD: AUDUSD=X
- その他: {CCY}USD=X または USD{CCY}=X の逆数

キャッシュ:
- data/cache/fx_rates/ に Parquet 形式で保存
- メタデータは JSON で保存
- 有効期限: 1日
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

logger = logging.getLogger(__name__)


# 通貨ペアのマッピング（CCY→USD変換用）
# key: 通貨コード, value: (Yahoo Finance シンボル, 逆数かどうか)
CURRENCY_PAIR_MAP: dict[str, tuple[str, bool]] = {
    "USD": ("", False),  # USD→USD は変換不要
    "JPY": ("USDJPY=X", True),  # USD/JPY の逆数
    "EUR": ("EURUSD=X", False),  # EUR/USD
    "GBP": ("GBPUSD=X", False),  # GBP/USD
    "AUD": ("AUDUSD=X", False),  # AUD/USD
    "CAD": ("USDCAD=X", True),  # USD/CAD の逆数
    "CHF": ("USDCHF=X", True),  # USD/CHF の逆数
    "NZD": ("NZDUSD=X", False),  # NZD/USD
    "CNY": ("USDCNY=X", True),  # USD/CNY の逆数
    "HKD": ("USDHKD=X", True),  # USD/HKD の逆数
    "KRW": ("USDKRW=X", True),  # USD/KRW の逆数
    "SGD": ("USDSGD=X", True),  # USD/SGD の逆数
    "INR": ("USDINR=X", True),  # USD/INR の逆数
}


class CurrencyConverter:
    """
    為替レート変換クラス

    価格データを USD に変換するためのユーティリティ。
    yfinance 経由で為替レートを取得し、ローカルにキャッシュする。

    使用例:
        converter = CurrencyConverter()

        # 価格シリーズを変換
        jpy_prices = pd.Series([1500, 1510, 1520], index=dates)
        usd_prices = converter.convert_to_usd(jpy_prices, "JPY")

        # 特定日のレートを取得
        rate = converter.get_fx_rate("USDJPY", datetime(2024, 1, 15))

        # 期間の履歴を取得
        rates = converter.get_fx_history("EURUSD", date(2024, 1, 1), date(2024, 1, 31))
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/fx_rates",
        cache_max_age_days: int = 1,
    ) -> None:
        """
        初期化

        Args:
            cache_dir: キャッシュディレクトリ
            cache_max_age_days: キャッシュ有効期限（日）

        Raises:
            ImportError: yfinance がインストールされていない場合
        """
        if yf is None:
            raise ImportError(
                "yfinance is required for CurrencyConverter. "
                "Install with: pip install yfinance"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_days = cache_max_age_days

        # メモリキャッシュ（セッション中の高速アクセス用）
        self._memory_cache: dict[str, pd.Series] = {}

    def convert_to_usd(
        self,
        prices: pd.Series,
        from_currency: str,
    ) -> pd.Series:
        """
        価格シリーズを USD に変換

        Args:
            prices: 価格シリーズ（DatetimeIndex を持つこと）
            from_currency: 元の通貨コード（JPY, EUR, GBP 等）

        Returns:
            USD 建ての価格シリーズ

        Raises:
            ValueError: サポートされていない通貨の場合
        """
        from_currency = from_currency.upper()

        # USD→USD は変換不要
        if from_currency == "USD":
            return prices.copy()

        if prices.empty:
            return prices.copy()

        # インデックスから日付範囲を取得
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must have DatetimeIndex")

        start_date = prices.index.min().date()
        end_date = prices.index.max().date()

        # 為替レート履歴を取得
        pair = f"{from_currency}USD"
        fx_rates = self.get_fx_history(pair, start_date, end_date)

        if fx_rates.empty:
            logger.warning(
                f"No FX rates available for {pair}, returning original prices"
            )
            return prices.copy()

        # 日付でマージ（価格の日付に合わせる）
        prices_df = pd.DataFrame({"price": prices})
        prices_df["date"] = prices_df.index.date

        fx_df = pd.DataFrame({"fx_rate": fx_rates})
        fx_df["date"] = fx_df.index.date

        # 日付でマージ（前方補完で欠損日を埋める）
        merged = prices_df.merge(fx_df, on="date", how="left")
        merged["fx_rate"] = merged["fx_rate"].ffill().bfill()

        # 変換
        converted = merged["price"] * merged["fx_rate"]
        converted.index = prices.index
        converted.name = prices.name

        return converted

    def get_fx_rate(
        self,
        pair: str,
        date: datetime | date,
    ) -> float:
        """
        特定日の為替レートを取得

        Args:
            pair: 通貨ペア（USDJPY, EURUSD 等）
            date: 日付

        Returns:
            為替レート

        Raises:
            ValueError: レートが取得できない場合
        """
        if isinstance(date, datetime):
            target_date = date.date()
        else:
            target_date = date

        # 週末の場合は直前の営業日を探す
        search_start = target_date - timedelta(days=7)
        search_end = target_date + timedelta(days=1)

        rates = self.get_fx_history(pair, search_start, search_end)

        if rates.empty:
            raise ValueError(f"No FX rate available for {pair} around {target_date}")

        # target_date 以前の最新レートを返す
        valid_rates = rates[rates.index.date <= target_date]
        if valid_rates.empty:
            # target_date 以降の最初のレートを返す
            return float(rates.iloc[0])

        return float(valid_rates.iloc[-1])

    def get_fx_history(
        self,
        pair: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        期間の為替レート履歴を取得

        Args:
            pair: 通貨ペア（USDJPY, EURUSD, JPYUSD 等）
            start: 開始日
            end: 終了日

        Returns:
            為替レートのシリーズ（DatetimeIndex）
        """
        pair = pair.upper().replace("/", "").replace("-", "").replace("=X", "")

        # CCY→USD 形式に正規化
        yahoo_symbol, is_inverse = self._resolve_pair(pair)

        if yahoo_symbol == "":
            # USD→USD の場合は 1.0 を返す
            idx = pd.date_range(start, end, freq="D")
            return pd.Series(1.0, index=idx, name=pair)

        # キャッシュを確認
        cached = self._load_cache(yahoo_symbol, start, end)
        if cached is not None:
            rates = cached
        else:
            # API から取得
            rates = self._fetch_fx_data(yahoo_symbol, start, end)
            if not rates.empty:
                self._save_cache(yahoo_symbol, rates)

        # 逆数が必要な場合は変換
        if is_inverse and not rates.empty:
            rates = 1.0 / rates

        rates.name = pair
        return rates

    def _resolve_pair(self, pair: str) -> tuple[str, bool]:
        """
        通貨ペアを Yahoo Finance シンボルに解決

        Args:
            pair: 通貨ペア（6文字、例: USDJPY, JPYUSD）

        Returns:
            (Yahoo Finance シンボル, 逆数かどうか)
        """
        if len(pair) != 6:
            raise ValueError(f"Invalid currency pair: {pair}. Expected 6 characters.")

        base = pair[:3]
        quote = pair[3:]

        # CCY→USD の形式かチェック
        if quote == "USD":
            if base in CURRENCY_PAIR_MAP:
                symbol, is_inverse = CURRENCY_PAIR_MAP[base]
                return symbol, is_inverse
            # マップにない場合は直接ペアを試す
            return f"{base}USD=X", False

        # USD→CCY の形式かチェック
        if base == "USD":
            if quote in CURRENCY_PAIR_MAP:
                symbol, is_inverse = CURRENCY_PAIR_MAP[quote]
                # 逆方向なので is_inverse を反転
                return symbol, not is_inverse
            return f"USD{quote}=X", False

        # その他のクロスペア（CCY1→CCY2）
        # まず CCY1→USD、次に USD→CCY2 で計算が必要だが、
        # 簡易実装として直接ペアを試す
        return f"{pair}=X", False

    def _fetch_fx_data(
        self,
        yahoo_symbol: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        yfinance から為替データを取得

        Args:
            yahoo_symbol: Yahoo Finance シンボル（例: USDJPY=X）
            start: 開始日
            end: 終了日

        Returns:
            終値のシリーズ
        """
        try:
            ticker = yf.Ticker(yahoo_symbol)

            # 終了日を1日後にして当日を含める
            end_plus = end + timedelta(days=1)

            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end_plus.strftime("%Y-%m-%d"),
                interval="1d",
                actions=False,
            )

            if df.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return pd.Series(dtype=float)

            # 終値を抽出
            rates = df["Close"].copy()
            rates.name = yahoo_symbol

            logger.info(
                f"Fetched {len(rates)} FX rates for {yahoo_symbol} "
                f"({start} to {end})"
            )

            return rates

        except Exception as e:
            logger.error(f"Failed to fetch FX data for {yahoo_symbol}: {e}")
            return pd.Series(dtype=float)

    def _get_cache_path(self, yahoo_symbol: str) -> Path:
        """キャッシュファイルのパスを取得"""
        safe_name = yahoo_symbol.replace("=", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def _get_meta_path(self, yahoo_symbol: str) -> Path:
        """メタデータファイルのパスを取得"""
        safe_name = yahoo_symbol.replace("=", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}.meta.json"

    def _load_cache(
        self,
        yahoo_symbol: str,
        start: date,
        end: date,
    ) -> Optional[pd.Series]:
        """
        キャッシュからデータを読み込み

        Args:
            yahoo_symbol: Yahoo Finance シンボル
            start: 開始日
            end: 終了日

        Returns:
            キャッシュされたデータ（なければ None）
        """
        # メモリキャッシュを確認
        cache_key = f"{yahoo_symbol}:{start}:{end}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        cache_path = self._get_cache_path(yahoo_symbol)
        meta_path = self._get_meta_path(yahoo_symbol)

        if not cache_path.exists() or not meta_path.exists():
            return None

        # メタデータを確認
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read cache metadata: {e}")
            return None

        # 有効期限を確認
        fetched_at = datetime.fromisoformat(metadata["fetched_at"])
        age = datetime.now(timezone.utc) - fetched_at
        if age.days > self.cache_max_age_days:
            logger.debug(f"Cache expired for {yahoo_symbol}")
            return None

        # キャッシュされた期間を確認
        cached_start = date.fromisoformat(metadata["start_date"])
        cached_end = date.fromisoformat(metadata["end_date"])

        if start < cached_start or end > cached_end:
            # 要求された期間がキャッシュをカバーしていない
            logger.debug(
                f"Cache range mismatch for {yahoo_symbol}: "
                f"cached {cached_start}-{cached_end}, "
                f"requested {start}-{end}"
            )
            return None

        # Parquet を読み込み
        try:
            df = pd.read_parquet(cache_path)
            rates = df["rate"]
            rates.index = pd.to_datetime(df["date"])
            rates.name = yahoo_symbol

            # 要求された期間でフィルタ
            mask = (rates.index.date >= start) & (rates.index.date <= end)
            result = rates[mask]

            # メモリキャッシュに保存
            self._memory_cache[cache_key] = result

            logger.debug(f"Cache hit for {yahoo_symbol}")
            return result

        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def _save_cache(
        self,
        yahoo_symbol: str,
        rates: pd.Series,
    ) -> None:
        """
        データをキャッシュに保存

        Args:
            yahoo_symbol: Yahoo Finance シンボル
            rates: 為替レートのシリーズ
        """
        if rates.empty:
            return

        cache_path = self._get_cache_path(yahoo_symbol)
        meta_path = self._get_meta_path(yahoo_symbol)

        try:
            # Parquet に保存
            df = pd.DataFrame({
                "date": rates.index,
                "rate": rates.values,
            })
            df.to_parquet(cache_path, index=False)

            # メタデータを保存
            metadata = {
                "symbol": yahoo_symbol,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "start_date": rates.index.min().date().isoformat(),
                "end_date": rates.index.max().date().isoformat(),
                "row_count": len(rates),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Cached {len(rates)} rates for {yahoo_symbol}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        キャッシュをクリア

        Args:
            symbol: 特定シンボルのみクリア（None で全クリア）

        Returns:
            削除したファイル数
        """
        count = 0

        if symbol:
            # 特定シンボルのみ
            safe_name = symbol.replace("=", "_").replace("/", "_")
            for suffix in [".parquet", ".meta.json"]:
                path = self.cache_dir / f"{safe_name}{suffix}"
                if path.exists():
                    path.unlink()
                    count += 1
        else:
            # 全クリア
            for path in self.cache_dir.glob("*.parquet"):
                path.unlink()
                count += 1
            for path in self.cache_dir.glob("*.meta.json"):
                path.unlink()
                count += 1

        # メモリキャッシュもクリア
        self._memory_cache.clear()

        return count

    def get_supported_currencies(self) -> list[str]:
        """
        サポートされている通貨コードのリストを取得

        Returns:
            通貨コードのリスト
        """
        return list(CURRENCY_PAIR_MAP.keys())

    def get_cache_stats(self) -> dict:
        """
        キャッシュ統計を取得

        Returns:
            統計情報の辞書
        """
        parquet_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)

        return {
            "cache_dir": str(self.cache_dir),
            "entry_count": len(parquet_files),
            "total_size_mb": round(total_size / (1024 * 1024), 4),
            "max_age_days": self.cache_max_age_days,
            "memory_cache_entries": len(self._memory_cache),
        }
