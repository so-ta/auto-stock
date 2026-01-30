"""
SEC EDGAR API Client - Insider Trading Data Retrieval.

Provides access to SEC EDGAR filings, specifically Form 4 (insider transactions).
Fully free, no authentication required.

API Documentation: https://www.sec.gov/developer

Rate Limits:
- Recommended: 10 requests/second maximum
- User-Agent header required

Example:
    client = SECEdgarClient()
    transactions = client.get_insider_transactions("AAPL")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# CIK to Ticker mapping cache (loaded from SEC API)
_CIK_TICKER_MAP: Dict[str, str] = {}
_TICKER_CIK_MAP: Dict[str, str] = {}


@dataclass
class InsiderTransaction:
    """Represents a single insider transaction from Form 4."""

    filing_date: datetime
    transaction_date: Optional[datetime]
    insider_name: str
    insider_title: str
    transaction_type: str  # "P" = Purchase, "S" = Sale, "A" = Award, etc.
    shares: float
    price_per_share: Optional[float]
    total_value: Optional[float]
    shares_owned_after: Optional[float]
    is_direct: bool = True  # Direct vs Indirect ownership
    form_type: str = "4"

    @property
    def is_purchase(self) -> bool:
        """Check if this is a purchase transaction."""
        return self.transaction_type in ("P", "M")  # P = Purchase, M = Exercise

    @property
    def is_sale(self) -> bool:
        """Check if this is a sale transaction."""
        return self.transaction_type in ("S", "F")  # S = Sale, F = Tax payment

    @property
    def is_executive(self) -> bool:
        """Check if insider is an executive (CEO, CFO, COO, etc.)."""
        import re

        title_upper = self.insider_title.upper()
        # Use word boundaries to avoid partial matches (e.g., "DIRECTOR" matching "CTO")
        executive_patterns = [
            r"\bCEO\b",
            r"\bCFO\b",
            r"\bCOO\b",
            r"\bCTO\b",
            r"\bCIO\b",
            r"\bPRESIDENT\b",
            r"\bCHAIRMAN\b",
            r"\bCHIEF\s+EXECUTIVE",
            r"\bCHIEF\s+FINANCIAL",
            r"\bCHIEF\s+OPERATING",
        ]
        return any(re.search(pattern, title_upper) for pattern in executive_patterns)


@dataclass
class SECEdgarClientConfig:
    """Configuration for SEC EDGAR API client."""

    user_agent: str = "MultiAssetPortfolio/1.0 (contact@example.com)"
    rate_limit_delay: float = 0.1  # Seconds between requests (10 req/sec)
    timeout: int = 30
    max_retries: int = 3
    cache_cik_mapping: bool = True


class SECEdgarError(Exception):
    """Exception raised for SEC EDGAR API errors."""

    pass


class SECEdgarClient:
    """
    SEC EDGAR API client for insider trading data.

    Retrieves Form 4 filings (insider transactions) from SEC EDGAR.
    All data is free and publicly available.

    Example:
        client = SECEdgarClient()
        transactions = client.get_insider_transactions("AAPL", days=90)

        # Process transactions
        purchases = [t for t in transactions if t.is_purchase]
        total_purchase_value = sum(t.total_value or 0 for t in purchases)
    """

    BASE_URL = "https://data.sec.gov"
    COMPANY_TICKERS_URL = f"{BASE_URL}/files/company_tickers.json"

    def __init__(self, config: Optional[SECEdgarClientConfig] = None):
        """
        Initialize SEC EDGAR client.

        Args:
            config: Client configuration. If None, uses defaults.
        """
        self.config = config or SECEdgarClientConfig()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.config.user_agent
        self.session.headers["Accept"] = "application/json"
        self._last_request_time: Optional[float] = None

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.config.rate_limit_delay:
                time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str) -> Dict[str, Any]:
        """
        Make a rate-limited request to SEC EDGAR API.

        Args:
            url: URL to request

        Returns:
            JSON response as dictionary

        Raises:
            SECEdgarError: If request fails after retries
        """
        self._rate_limit()

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=self.config.timeout)

                if response.status_code == 404:
                    raise SECEdgarError(f"Resource not found: {url}")

                if response.status_code == 429:
                    # Rate limited - back off
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limited by SEC, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt == self.config.max_retries - 1:
                    raise SECEdgarError(f"Request timeout after {self.config.max_retries} attempts")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise SECEdgarError(f"Request failed: {e}")

            time.sleep(1)

        raise SECEdgarError("Request failed after all retries")

    def get_cik(self, ticker: str) -> str:
        """
        Get CIK (Central Index Key) for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            CIK as zero-padded 10-digit string

        Raises:
            SECEdgarError: If ticker not found
        """
        global _TICKER_CIK_MAP

        ticker_upper = ticker.upper()

        # Check cache first
        if ticker_upper in _TICKER_CIK_MAP:
            return _TICKER_CIK_MAP[ticker_upper]

        # Load company tickers mapping
        if not _TICKER_CIK_MAP and self.config.cache_cik_mapping:
            self._load_ticker_cik_mapping()

        if ticker_upper in _TICKER_CIK_MAP:
            return _TICKER_CIK_MAP[ticker_upper]

        raise SECEdgarError(f"Ticker '{ticker}' not found in SEC database")

    def _load_ticker_cik_mapping(self) -> None:
        """Load ticker to CIK mapping from SEC API."""
        global _TICKER_CIK_MAP, _CIK_TICKER_MAP

        try:
            data = self._request(self.COMPANY_TICKERS_URL)

            for key, company in data.items():
                ticker = company.get("ticker", "").upper()
                cik = str(company.get("cik_str", "")).zfill(10)

                if ticker and cik:
                    _TICKER_CIK_MAP[ticker] = cik
                    _CIK_TICKER_MAP[cik] = ticker

            logger.info(f"Loaded {len(_TICKER_CIK_MAP)} ticker-CIK mappings")

        except Exception as e:
            logger.error(f"Failed to load ticker-CIK mapping: {e}")
            raise SECEdgarError(f"Failed to load ticker-CIK mapping: {e}")

    def get_company_filings(self, ticker: str) -> Dict[str, Any]:
        """
        Get all filings for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company filings data from SEC EDGAR
        """
        cik = self.get_cik(ticker)
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        return self._request(url)

    def get_insider_transactions(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        days: int = 90,
    ) -> List[InsiderTransaction]:
        """
        Get insider transactions from Form 4 filings.

        Args:
            ticker: Stock ticker symbol
            start_date: Only include transactions after this date.
                       If None, uses (today - days).
            days: Number of days to look back if start_date is None.

        Returns:
            List of InsiderTransaction objects

        Example:
            transactions = client.get_insider_transactions("AAPL", days=180)
            exec_purchases = [t for t in transactions if t.is_purchase and t.is_executive]
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        try:
            filings_data = self.get_company_filings(ticker)
        except SECEdgarError:
            logger.warning(f"No filings found for {ticker}")
            return []

        transactions: List[InsiderTransaction] = []

        # Process recent filings
        recent = filings_data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        for i, form_type in enumerate(forms):
            if form_type not in ("4", "4/A"):  # Form 4 and amendments
                continue

            try:
                filing_date_str = filing_dates[i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")

                if filing_date < start_date:
                    continue

                # Get detailed filing data
                accession = accession_numbers[i].replace("-", "")
                cik = self.get_cik(ticker)
                filing_url = f"{self.BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/primary_doc.xml"

                # Parse Form 4 - simplified approach using JSON summary
                # Full XML parsing would require additional logic
                transaction = self._parse_form4_summary(
                    filings_data, i, filing_date
                )
                if transaction:
                    transactions.append(transaction)

            except (IndexError, ValueError, KeyError) as e:
                logger.debug(f"Error parsing filing {i}: {e}")
                continue

        return transactions

    def _parse_form4_summary(
        self,
        filings_data: Dict[str, Any],
        index: int,
        filing_date: datetime,
    ) -> Optional[InsiderTransaction]:
        """
        Parse Form 4 summary from filings data.

        Note: This is a simplified parser using available metadata.
        Full transaction details require parsing the XML filing directly.

        Args:
            filings_data: Company filings data from SEC API
            index: Index of the filing in the recent filings list
            filing_date: Date of the filing

        Returns:
            InsiderTransaction or None if parsing fails
        """
        try:
            recent = filings_data.get("filings", {}).get("recent", {})

            # Extract what we can from the summary
            # Note: Full details require XML parsing
            primary_doc = recent.get("primaryDocument", [])
            doc_description = recent.get("primaryDocDescription", [])

            # For now, create a minimal transaction record
            # Real implementation would parse the XML for full details
            return InsiderTransaction(
                filing_date=filing_date,
                transaction_date=filing_date,  # Approximate
                insider_name="Unknown",  # Would come from XML
                insider_title="",  # Would come from XML
                transaction_type="U",  # Unknown - would come from XML
                shares=0.0,  # Would come from XML
                price_per_share=None,
                total_value=None,
                shares_owned_after=None,
                is_direct=True,
                form_type="4",
            )

        except Exception as e:
            logger.debug(f"Error parsing Form 4 summary: {e}")
            return None

    def get_transaction_summary(
        self,
        ticker: str,
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Get summarized insider transaction statistics.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            Dictionary with transaction summary statistics

        Example:
            summary = client.get_transaction_summary("AAPL", days=90)
            print(f"Net insider sentiment: {summary['net_sentiment']}")
        """
        transactions = self.get_insider_transactions(ticker, days=days)

        if not transactions:
            return {
                "ticker": ticker,
                "period_days": days,
                "total_transactions": 0,
                "purchases": 0,
                "sales": 0,
                "purchase_value": 0.0,
                "sale_value": 0.0,
                "net_value": 0.0,
                "net_sentiment": 0.0,
                "executive_transactions": 0,
            }

        purchases = [t for t in transactions if t.is_purchase]
        sales = [t for t in transactions if t.is_sale]
        exec_trans = [t for t in transactions if t.is_executive]

        purchase_value = sum(t.total_value or 0 for t in purchases)
        sale_value = sum(t.total_value or 0 for t in sales)
        net_value = purchase_value - sale_value

        # Calculate net sentiment (-1 to +1)
        total_value = purchase_value + sale_value
        if total_value > 0:
            net_sentiment = net_value / total_value
        else:
            # Use transaction count if no value data
            if len(purchases) + len(sales) > 0:
                net_sentiment = (len(purchases) - len(sales)) / (
                    len(purchases) + len(sales)
                )
            else:
                net_sentiment = 0.0

        return {
            "ticker": ticker,
            "period_days": days,
            "total_transactions": len(transactions),
            "purchases": len(purchases),
            "sales": len(sales),
            "purchase_value": purchase_value,
            "sale_value": sale_value,
            "net_value": net_value,
            "net_sentiment": net_sentiment,
            "executive_transactions": len(exec_trans),
        }


def create_sec_client(
    user_agent: Optional[str] = None,
    rate_limit_delay: float = 0.1,
) -> SECEdgarClient:
    """
    Create a configured SEC EDGAR client.

    Args:
        user_agent: Custom User-Agent string (required by SEC)
        rate_limit_delay: Delay between requests in seconds

    Returns:
        Configured SECEdgarClient instance
    """
    config = SECEdgarClientConfig(
        rate_limit_delay=rate_limit_delay,
    )
    if user_agent:
        config.user_agent = user_agent

    return SECEdgarClient(config)
