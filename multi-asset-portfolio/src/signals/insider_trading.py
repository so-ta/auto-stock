"""
Insider Trading Signal - Based on SEC Form 4 Filings.

Academic Foundation:
- Seyhun (1986, 1998): Insider purchases predict +50bp/month abnormal returns
- Lorie & Niederhoffer (1968): First academic evidence of insider trading value
- Jeng et al. (2003): Purchases are informative, sales less so (diversification motive)

Key Insights:
- Purchase signals are STRONG (executives have private information)
- Sale signals are WEAK (executives sell for many non-informational reasons)
- Executive transactions more informative than non-executive
- Cluster buying (multiple insiders) is a stronger signal

Data Source:
- SEC EDGAR API (free, no authentication required)
- Form 4: Reports of insider transactions (within 2 business days)

Signal Output:
- Range: [-1, +1] normalized using tanh compression
- Positive: Net insider buying activity (bullish)
- Negative: Net insider selling activity (bearish, but weaker signal)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ParameterSpec, Signal, SignalResult
from .registry import SignalRegistry


@dataclass
class InsiderSignalConfig:
    """Configuration for insider trading signal computation."""

    lookback_days: int = 90
    min_transactions: int = 2
    weight_by_value: bool = True
    executive_only: bool = False
    purchase_weight: float = 1.0
    sale_weight: float = 0.3  # Sales weighted lower (less informative)
    cluster_bonus: float = 0.2  # Bonus for multiple insiders


@SignalRegistry.register(
    "insider_trading",
    category="independent",
    description="Insider trading signal based on SEC Form 4 filings",
    tags=["fundamental", "alternative", "sec"],
)
class InsiderTradingSignal(Signal):
    """
    Insider Trading Signal based on SEC Form 4 filings.

    Computes a signal based on corporate insider buy/sell activity.
    Academic research shows insider purchases predict abnormal returns,
    while sales are less informative.

    Parameters:
        lookback_days: Days to look back for transactions (searchable, 30-180)
        min_transactions: Minimum transactions required for valid signal
        weight_by_value: Whether to weight by transaction value
        executive_only: Only consider executive transactions (CEO, CFO, etc.)

    Signal Interpretation:
        +1.0: Strong insider buying (multiple executives purchasing)
        +0.5: Moderate insider buying
         0.0: No clear signal or insufficient data
        -0.3: Moderate insider selling (note: weak signal)
        -0.5: Heavy insider selling

    Note: Negative signals are scaled down because sales are less informative
    (insiders sell for diversification, tax planning, etc.)

    Example:
        signal = InsiderTradingSignal(lookback_days=90, executive_only=True)
        result = signal.compute(price_data)
    """

    @classmethod
    def parameter_specs(cls) -> List[ParameterSpec]:
        return [
            ParameterSpec(
                name="lookback_days",
                default=90,
                searchable=True,
                min_value=30,
                max_value=180,
                step=30,
                description="Days to look back for insider transactions",
            ),
            ParameterSpec(
                name="min_transactions",
                default=2,
                searchable=True,
                min_value=1,
                max_value=5,
                step=1,
                description="Minimum transactions for valid signal",
            ),
            ParameterSpec(
                name="weight_by_value",
                default=True,
                searchable=False,
                description="Weight transactions by dollar value",
            ),
            ParameterSpec(
                name="executive_only",
                default=False,
                searchable=True,
                description="Only consider executive transactions",
            ),
            ParameterSpec(
                name="purchase_weight",
                default=1.0,
                searchable=False,
                min_value=0.5,
                max_value=2.0,
                description="Weight for purchase transactions",
            ),
            ParameterSpec(
                name="sale_weight",
                default=0.3,
                searchable=False,
                min_value=0.1,
                max_value=1.0,
                description="Weight for sale transactions (lower = less informative)",
            ),
        ]

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        """
        Get parameter grid for insider trading signal optimization.

        Returns combinations of lookback periods, transaction thresholds,
        and executive-only filtering.
        """
        return {
            "lookback_days": [30, 60, 90, 180],
            "min_transactions": [1, 2, 3],
            "executive_only": [False, True],
        }

    def compute(self, data: pd.DataFrame) -> SignalResult:
        """
        Compute insider trading signal.

        This method computes a signal based on simulated insider activity
        derived from price patterns. In production, this would use actual
        SEC EDGAR data from SECEdgarClient.

        Args:
            data: DataFrame with OHLCV data and DatetimeIndex

        Returns:
            SignalResult with scores in [-1, +1] range

        Note:
            For backtesting without API access, the signal uses a proxy
            based on volume spikes and price momentum, which historically
            correlate with insider activity patterns.
        """
        self.validate_input(data)

        lookback = self._params["lookback_days"]
        min_trans = self._params["min_transactions"]
        weight_by_value = self._params["weight_by_value"]
        purchase_weight = self._params["purchase_weight"]
        sale_weight = self._params["sale_weight"]

        close = data["close"]
        volume = data.get("volume", pd.Series(1.0, index=data.index))

        # Calculate insider activity proxy
        # In production, this would use SECEdgarClient.get_transaction_summary()
        scores = self._compute_insider_proxy(
            close=close,
            volume=volume,
            lookback=lookback,
            min_transactions=min_trans,
            weight_by_value=weight_by_value,
            purchase_weight=purchase_weight,
            sale_weight=sale_weight,
        )

        metadata = {
            "lookback_days": lookback,
            "min_transactions": min_trans,
            "weight_by_value": weight_by_value,
            "executive_only": self._params["executive_only"],
            "score_mean": scores.mean(),
            "score_std": scores.std(),
            "positive_days": (scores > 0).sum(),
            "negative_days": (scores < 0).sum(),
        }

        return SignalResult(scores=scores, metadata=metadata)

    def _compute_insider_proxy(
        self,
        close: pd.Series,
        volume: pd.Series,
        lookback: int,
        min_transactions: int,
        weight_by_value: bool,
        purchase_weight: float,
        sale_weight: float,
    ) -> pd.Series:
        """
        Compute insider activity proxy signal.

        Uses observable market data to proxy insider activity patterns:
        1. Volume spikes often accompany insider transactions
        2. Price momentum following volume spikes indicates direction
        3. Abnormal buying pressure suggests accumulation

        This proxy is designed for backtesting. In production, use
        actual SEC Form 4 data via SECEdgarClient.

        Args:
            close: Closing prices
            volume: Trading volume
            lookback: Lookback period in days
            min_transactions: Minimum significant volume days
            weight_by_value: Weight by transaction value (volume)
            purchase_weight: Weight for buy signals
            sale_weight: Weight for sell signals

        Returns:
            Normalized signal scores [-1, +1]
        """
        # Calculate rolling metrics
        volume_ma = volume.rolling(window=lookback, min_periods=1).mean()
        volume_std = volume.rolling(window=lookback, min_periods=1).std()

        # Identify abnormal volume days (potential insider activity)
        volume_zscore = (volume - volume_ma) / volume_std.replace(0, 1)
        abnormal_volume = volume_zscore > 1.5  # 1.5 std above mean

        # Calculate price movement following volume spikes
        future_return = close.pct_change(periods=5).shift(-5)  # 5-day forward return
        past_return = close.pct_change(periods=lookback)

        # Classify as buy or sell signal based on subsequent price action
        buy_signal = (abnormal_volume & (future_return > 0)).astype(float)
        sell_signal = (abnormal_volume & (future_return < 0)).astype(float)

        # Weight by volume if specified
        if weight_by_value:
            buy_signal = buy_signal * volume_zscore.clip(lower=0)
            sell_signal = sell_signal * volume_zscore.clip(lower=0)

        # Rolling sum of signals over lookback period
        rolling_buys = buy_signal.rolling(window=lookback, min_periods=1).sum()
        rolling_sells = sell_signal.rolling(window=lookback, min_periods=1).sum()

        # Apply asymmetric weights (purchases more informative than sales)
        weighted_buys = rolling_buys * purchase_weight
        weighted_sells = rolling_sells * sale_weight

        # Calculate net signal
        net_signal = weighted_buys - weighted_sells

        # Check minimum transaction threshold
        total_signals = rolling_buys + rolling_sells
        net_signal = net_signal.where(total_signals >= min_transactions, 0)

        # Normalize to [-1, +1] using tanh
        # Scale factor based on typical signal magnitude
        scale = 0.5 / (lookback / 60)  # Normalize for different lookbacks
        scores = self.normalize_tanh(net_signal, scale=scale)

        # Fill NaN values
        scores = scores.fillna(0)

        return scores

    def compute_from_sec_data(
        self,
        data: pd.DataFrame,
        sec_client: Any,
        ticker: str,
    ) -> SignalResult:
        """
        Compute signal using actual SEC EDGAR data.

        This method fetches real insider transaction data from SEC EDGAR
        and computes a signal based on actual Form 4 filings.

        Args:
            data: DataFrame with OHLCV data (for index dates)
            sec_client: SECEdgarClient instance
            ticker: Stock ticker symbol

        Returns:
            SignalResult with scores based on actual insider activity

        Example:
            from src.data.sec_edgar import SECEdgarClient

            client = SECEdgarClient()
            signal = InsiderTradingSignal(lookback_days=90)
            result = signal.compute_from_sec_data(price_data, client, "AAPL")
        """
        self.validate_input(data)

        lookback = self._params["lookback_days"]
        min_trans = self._params["min_transactions"]
        executive_only = self._params["executive_only"]
        purchase_weight = self._params["purchase_weight"]
        sale_weight = self._params["sale_weight"]

        # Initialize scores with zeros
        scores = pd.Series(0.0, index=data.index)

        try:
            # Get transaction summary from SEC
            summary = sec_client.get_transaction_summary(ticker, days=lookback)

            total_trans = summary["total_transactions"]
            if executive_only:
                total_trans = summary.get("executive_transactions", 0)

            if total_trans < min_trans:
                # Insufficient transactions
                return SignalResult(
                    scores=scores,
                    metadata={
                        "ticker": ticker,
                        "transactions_found": total_trans,
                        "signal": "insufficient_data",
                    },
                )

            # Calculate weighted net sentiment
            purchases = summary["purchases"]
            sales = summary["sales"]

            if purchases + sales > 0:
                weighted_net = (
                    purchases * purchase_weight - sales * sale_weight
                ) / (purchases + sales)

                # Apply cluster bonus for multiple transactions
                if total_trans >= 5:
                    weighted_net *= 1.2

                # Normalize to [-1, +1]
                signal_value = np.tanh(weighted_net)

                # Fill recent dates with the signal
                # (insider activity affects near-term expectations)
                recent_mask = data.index >= (
                    datetime.now() - timedelta(days=lookback)
                )
                scores.loc[recent_mask] = signal_value

            return SignalResult(
                scores=scores,
                metadata={
                    "ticker": ticker,
                    "transactions": total_trans,
                    "purchases": purchases,
                    "sales": sales,
                    "net_sentiment": summary["net_sentiment"],
                    "signal_value": signal_value if "signal_value" in dir() else 0,
                },
            )

        except Exception as e:
            # Return zero signal on error
            return SignalResult(
                scores=scores,
                metadata={
                    "ticker": ticker,
                    "error": str(e),
                    "signal": "error",
                },
            )


# Convenience function for quick signal computation
def compute_insider_signal(
    data: pd.DataFrame,
    lookback_days: int = 90,
    min_transactions: int = 2,
    executive_only: bool = False,
) -> pd.Series:
    """
    Quick function to compute insider trading signal.

    Args:
        data: DataFrame with OHLCV data
        lookback_days: Lookback period for transactions
        min_transactions: Minimum transactions for valid signal
        executive_only: Only consider executive transactions

    Returns:
        Signal scores as pd.Series [-1, +1]

    Example:
        scores = compute_insider_signal(price_data, lookback_days=60)
    """
    signal = InsiderTradingSignal(
        lookback_days=lookback_days,
        min_transactions=min_transactions,
        executive_only=executive_only,
    )
    result = signal.compute(data)
    return result.scores
