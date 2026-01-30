"""
Dividend Handler - 配当・トータルリターン計算モジュール

配当込みトータルリターンの計算と株式分割調整を提供。
fast_engine.py や pipeline.py から利用される。

【SYNC-006】fast_engine.py統合用として作成。

使用例:
    from src.data.dividend_handler import DividendHandler, DividendConfig

    handler = DividendHandler()
    total_returns = handler.calculate_total_returns(prices, dividends)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class DividendConfig:
    """配当処理設定

    Attributes:
        reinvest_dividends: 配当を再投資するかどうか
        withholding_tax_rate: 源泉徴収税率（デフォルト0%）
        use_ex_date: 配落日を使用するか（False=支払日）
        handle_missing: 配当データ欠損時の処理方法
    """
    reinvest_dividends: bool = True
    withholding_tax_rate: float = 0.0
    use_ex_date: bool = True
    handle_missing: str = "ignore"  # "ignore", "zero", "interpolate"


@dataclass
class DividendData:
    """配当データコンテナ

    Attributes:
        symbol: シンボル
        ex_dates: 配落日リスト
        amounts: 配当額リスト
        pay_dates: 支払日リスト（オプション）
    """
    symbol: str
    ex_dates: List[datetime]
    amounts: List[float]
    pay_dates: Optional[List[datetime]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrameに変換"""
        data = {
            "ex_date": self.ex_dates,
            "amount": self.amounts,
        }
        if self.pay_dates:
            data["pay_date"] = self.pay_dates
        return pd.DataFrame(data)


@dataclass
class TotalReturnResult:
    """トータルリターン計算結果

    Attributes:
        price_returns: 価格リターン
        dividend_returns: 配当リターン
        total_returns: トータルリターン（価格＋配当）
        cumulative_returns: 累積リターン
        dividend_contribution: 配当寄与率
    """
    price_returns: np.ndarray
    dividend_returns: np.ndarray
    total_returns: np.ndarray
    cumulative_returns: np.ndarray
    dividend_contribution: float


class DividendHandler:
    """
    配当ハンドラ

    配当込みトータルリターンの計算と株式分割調整を提供。

    主な機能:
    - calculate_total_returns(): 配当込みリターン計算
    - adjust_for_splits(): 株式分割調整
    - get_dividend_yield(): 配当利回り計算
    - merge_dividend_data(): 配当データを価格データにマージ
    """

    def __init__(self, config: Optional[DividendConfig] = None):
        """
        初期化

        Parameters
        ----------
        config : DividendConfig, optional
            配当処理設定
        """
        self.config = config or DividendConfig()
        self._dividend_cache: Dict[str, DividendData] = {}

    def calculate_total_returns(
        self,
        prices: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        dividends: Optional[Union[pd.DataFrame, pl.DataFrame, Dict[str, DividendData]]] = None,
        dates: Optional[List[datetime]] = None,
    ) -> TotalReturnResult:
        """
        配当込みトータルリターンを計算

        Parameters
        ----------
        prices : pd.DataFrame | pl.DataFrame | np.ndarray
            価格データ。DataFrameの場合は列がアセット、インデックスが日付。
        dividends : pd.DataFrame | pl.DataFrame | Dict, optional
            配当データ。Noneの場合は価格リターンのみ返す。
        dates : List[datetime], optional
            日付リスト（pricesがnp.ndarrayの場合に必要）

        Returns
        -------
        TotalReturnResult
            トータルリターン計算結果
        """
        # 価格をNumPy配列に変換
        if isinstance(prices, pl.DataFrame):
            if "timestamp" in prices.columns:
                prices_df = prices.to_pandas().set_index("timestamp")
            else:
                prices_df = prices.to_pandas()
            price_array = prices_df.values
            if dates is None:
                dates = list(prices_df.index)
        elif isinstance(prices, pd.DataFrame):
            price_array = prices.values
            if dates is None:
                dates = list(prices.index)
        else:
            price_array = prices

        n_days, n_assets = price_array.shape

        # 価格リターンを計算
        price_returns = np.zeros((n_days, n_assets))
        price_returns[1:] = price_array[1:] / price_array[:-1] - 1

        # 配当リターンを計算
        dividend_returns = np.zeros((n_days, n_assets))

        if dividends is not None and dates is not None:
            dividend_returns = self._calculate_dividend_returns(
                price_array, dividends, dates
            )

        # トータルリターン
        total_returns = price_returns + dividend_returns

        # 累積リターン
        cumulative_returns = np.cumprod(1 + total_returns, axis=0) - 1

        # 配当寄与率（全期間）
        total_price_return = np.prod(1 + price_returns, axis=0) - 1
        total_total_return = np.prod(1 + total_returns, axis=0) - 1

        # 配当寄与率（平均）
        if np.mean(np.abs(total_total_return)) > 1e-10:
            dividend_contribution = np.mean(
                (total_total_return - total_price_return) /
                np.maximum(np.abs(total_total_return), 1e-10)
            )
        else:
            dividend_contribution = 0.0

        return TotalReturnResult(
            price_returns=price_returns,
            dividend_returns=dividend_returns,
            total_returns=total_returns,
            cumulative_returns=cumulative_returns,
            dividend_contribution=float(dividend_contribution),
        )

    def _calculate_dividend_returns(
        self,
        prices: np.ndarray,
        dividends: Union[pd.DataFrame, pl.DataFrame, Dict[str, DividendData]],
        dates: List[datetime],
    ) -> np.ndarray:
        """
        配当リターンを計算

        Parameters
        ----------
        prices : np.ndarray
            価格配列 (n_days, n_assets)
        dividends : pd.DataFrame | pl.DataFrame | Dict
            配当データ
        dates : List[datetime]
            日付リスト

        Returns
        -------
        np.ndarray
            配当リターン配列 (n_days, n_assets)
        """
        n_days, n_assets = prices.shape
        dividend_returns = np.zeros((n_days, n_assets))

        # 配当データを処理
        if isinstance(dividends, dict):
            # Dict[str, DividendData]形式
            for asset_idx, (symbol, div_data) in enumerate(dividends.items()):
                if asset_idx >= n_assets:
                    break
                for ex_date, amount in zip(div_data.ex_dates, div_data.amounts):
                    day_idx = self._find_date_index(dates, ex_date)
                    if day_idx is not None and day_idx > 0:
                        prev_price = prices[day_idx - 1, asset_idx]
                        if prev_price > 0:
                            # 税引後配当
                            net_amount = amount * (1 - self.config.withholding_tax_rate)
                            dividend_returns[day_idx, asset_idx] = net_amount / prev_price

        elif isinstance(dividends, (pd.DataFrame, pl.DataFrame)):
            # DataFrame形式
            if isinstance(dividends, pl.DataFrame):
                div_df = dividends.to_pandas()
            else:
                div_df = dividends

            # 日付列と配当額列を特定
            date_col = None
            amount_col = None

            for col in ["ex_date", "date", "Date", "timestamp"]:
                if col in div_df.columns:
                    date_col = col
                    break

            for col in ["amount", "dividend_amount", "dividend", "Amount"]:
                if col in div_df.columns:
                    amount_col = col
                    break

            if date_col and amount_col:
                div_df[date_col] = pd.to_datetime(div_df[date_col])

                # アセット列を特定（symbol列があれば使う）
                if "symbol" in div_df.columns:
                    for asset_idx in range(n_assets):
                        for _, row in div_df.iterrows():
                            ex_date = row[date_col].to_pydatetime() if hasattr(row[date_col], 'to_pydatetime') else row[date_col]
                            day_idx = self._find_date_index(dates, ex_date)
                            if day_idx is not None and day_idx > 0:
                                prev_price = prices[day_idx - 1, asset_idx]
                                if prev_price > 0:
                                    net_amount = row[amount_col] * (1 - self.config.withholding_tax_rate)
                                    dividend_returns[day_idx, asset_idx] = net_amount / prev_price
                else:
                    # 単一アセットの配当データ
                    for _, row in div_df.iterrows():
                        ex_date = row[date_col].to_pydatetime() if hasattr(row[date_col], 'to_pydatetime') else row[date_col]
                        day_idx = self._find_date_index(dates, ex_date)
                        if day_idx is not None and day_idx > 0:
                            for asset_idx in range(n_assets):
                                prev_price = prices[day_idx - 1, asset_idx]
                                if prev_price > 0:
                                    net_amount = row[amount_col] * (1 - self.config.withholding_tax_rate)
                                    dividend_returns[day_idx, asset_idx] = net_amount / prev_price

        return dividend_returns

    def _find_date_index(
        self,
        dates: List[datetime],
        target: datetime,
    ) -> Optional[int]:
        """日付のインデックスを見つける"""
        # datetimeの正規化
        if hasattr(target, 'to_pydatetime'):
            target = target.to_pydatetime()
        if hasattr(target, 'date'):
            target_date = target.date()
        else:
            target_date = target

        for i, d in enumerate(dates):
            if hasattr(d, 'to_pydatetime'):
                d = d.to_pydatetime()
            if hasattr(d, 'date'):
                d_date = d.date()
            else:
                d_date = d

            if d_date == target_date:
                return i

        return None

    def adjust_for_splits(
        self,
        prices: Union[pd.DataFrame, pl.DataFrame, np.ndarray],
        splits: Union[pd.DataFrame, pl.DataFrame, Dict[str, List[Tuple[datetime, float]]]],
        dates: Optional[List[datetime]] = None,
    ) -> np.ndarray:
        """
        株式分割を調整

        Parameters
        ----------
        prices : pd.DataFrame | pl.DataFrame | np.ndarray
            価格データ
        splits : pd.DataFrame | pl.DataFrame | Dict
            分割データ。Dict形式: {symbol: [(date, ratio), ...]}
        dates : List[datetime], optional
            日付リスト

        Returns
        -------
        np.ndarray
            分割調整済み価格
        """
        # 価格をNumPy配列に変換
        if isinstance(prices, pl.DataFrame):
            prices_df = prices.to_pandas()
            if "timestamp" in prices.columns:
                prices_df = prices_df.set_index("timestamp")
            price_array = prices_df.values.copy()
            if dates is None:
                dates = list(prices_df.index)
        elif isinstance(prices, pd.DataFrame):
            price_array = prices.values.copy()
            if dates is None:
                dates = list(prices.index)
        else:
            price_array = prices.copy()

        n_days, n_assets = price_array.shape

        if isinstance(splits, dict):
            # Dict形式
            for asset_idx, (symbol, split_list) in enumerate(splits.items()):
                if asset_idx >= n_assets:
                    break
                for split_date, ratio in split_list:
                    day_idx = self._find_date_index(dates, split_date)
                    if day_idx is not None:
                        # 分割日より前の価格を調整
                        price_array[:day_idx, asset_idx] /= ratio

        elif isinstance(splits, (pd.DataFrame, pl.DataFrame)):
            if isinstance(splits, pl.DataFrame):
                split_df = splits.to_pandas()
            else:
                split_df = splits

            # 日付列と分割比率列を特定
            date_col = None
            ratio_col = None

            for col in ["date", "Date", "timestamp", "split_date"]:
                if col in split_df.columns:
                    date_col = col
                    break

            for col in ["ratio", "split_ratio", "value", "Value"]:
                if col in split_df.columns:
                    ratio_col = col
                    break

            if date_col and ratio_col:
                split_df[date_col] = pd.to_datetime(split_df[date_col])

                for _, row in split_df.iterrows():
                    split_date = row[date_col]
                    if hasattr(split_date, 'to_pydatetime'):
                        split_date = split_date.to_pydatetime()
                    ratio = row[ratio_col]

                    day_idx = self._find_date_index(dates, split_date)
                    if day_idx is not None:
                        # 全アセットに適用（symbol列がない場合）
                        if "symbol" not in split_df.columns:
                            for asset_idx in range(n_assets):
                                price_array[:day_idx, asset_idx] /= ratio

        return price_array

    def get_dividend_yield(
        self,
        prices: Union[pd.DataFrame, np.ndarray],
        dividends: Union[pd.DataFrame, Dict[str, DividendData]],
        lookback_days: int = 252,
    ) -> np.ndarray:
        """
        配当利回りを計算

        Parameters
        ----------
        prices : pd.DataFrame | np.ndarray
            価格データ
        dividends : pd.DataFrame | Dict
            配当データ
        lookback_days : int
            配当利回り計算のルックバック期間（日）

        Returns
        -------
        np.ndarray
            配当利回り（年率）
        """
        if isinstance(prices, pd.DataFrame):
            price_array = prices.values
            dates = list(prices.index)
        else:
            price_array = prices
            dates = None

        n_days, n_assets = price_array.shape
        yields = np.zeros(n_assets)

        # 直近のルックバック期間内の配当を集計
        if isinstance(dividends, dict):
            for asset_idx, (symbol, div_data) in enumerate(dividends.items()):
                if asset_idx >= n_assets:
                    break
                total_div = sum(div_data.amounts[-lookback_days:]) if div_data.amounts else 0
                current_price = price_array[-1, asset_idx]
                if current_price > 0:
                    yields[asset_idx] = total_div / current_price

        return yields

    def merge_dividend_data(
        self,
        prices: pd.DataFrame,
        dividends: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        配当データを価格データにマージ

        Parameters
        ----------
        prices : pd.DataFrame
            価格データ（インデックスは日付）
        dividends : pd.DataFrame
            配当データ（date, amount列を含む）

        Returns
        -------
        pd.DataFrame
            配当列を追加した価格データ
        """
        result = prices.copy()

        # 配当列を追加（デフォルト0）
        for col in prices.columns:
            result[f"{col}_dividend"] = 0.0

        # 配当データをマージ
        if "date" in dividends.columns and "amount" in dividends.columns:
            div_df = dividends.set_index("date")

            for date in div_df.index:
                if date in result.index:
                    for col in prices.columns:
                        result.loc[date, f"{col}_dividend"] = div_df.loc[date, "amount"]

        return result

    def calculate_reinvested_value(
        self,
        prices: np.ndarray,
        dividends: np.ndarray,
        initial_shares: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        配当再投資後の価値と株数を計算

        Parameters
        ----------
        prices : np.ndarray
            価格配列 (n_days,) または (n_days, n_assets)
        dividends : np.ndarray
            配当配列（1株あたり）
        initial_shares : float
            初期株数

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (再投資後価値, 株数履歴)
        """
        if prices.ndim == 1:
            n_days = len(prices)
            shares = np.zeros(n_days)
            values = np.zeros(n_days)

            shares[0] = initial_shares
            values[0] = initial_shares * prices[0]

            for i in range(1, n_days):
                # 配当で追加購入
                if dividends[i] > 0 and prices[i] > 0:
                    additional_shares = (shares[i-1] * dividends[i]) / prices[i]
                    shares[i] = shares[i-1] + additional_shares
                else:
                    shares[i] = shares[i-1]

                values[i] = shares[i] * prices[i]

            return values, shares
        else:
            # 複数アセット
            n_days, n_assets = prices.shape
            shares = np.zeros((n_days, n_assets))
            values = np.zeros((n_days, n_assets))

            shares[0] = initial_shares
            values[0] = initial_shares * prices[0]

            for i in range(1, n_days):
                for j in range(n_assets):
                    if dividends[i, j] > 0 and prices[i, j] > 0:
                        additional = (shares[i-1, j] * dividends[i, j]) / prices[i, j]
                        shares[i, j] = shares[i-1, j] + additional
                    else:
                        shares[i, j] = shares[i-1, j]

                    values[i, j] = shares[i, j] * prices[i, j]

            return values, shares


# ユーティリティ関数
def calculate_total_return(
    prices: Union[pd.DataFrame, np.ndarray],
    dividends: Optional[Union[pd.DataFrame, Dict]] = None,
    config: Optional[DividendConfig] = None,
) -> TotalReturnResult:
    """
    トータルリターンを計算するショートカット関数

    Parameters
    ----------
    prices : pd.DataFrame | np.ndarray
        価格データ
    dividends : pd.DataFrame | Dict, optional
        配当データ
    config : DividendConfig, optional
        設定

    Returns
    -------
    TotalReturnResult
        計算結果
    """
    handler = DividendHandler(config)
    return handler.calculate_total_returns(prices, dividends)


def adjust_prices_for_splits(
    prices: Union[pd.DataFrame, np.ndarray],
    splits: Union[pd.DataFrame, Dict],
) -> np.ndarray:
    """
    株式分割調整のショートカット関数
    """
    handler = DividendHandler()
    return handler.adjust_for_splits(prices, splits)
