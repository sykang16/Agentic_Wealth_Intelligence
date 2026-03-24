"""SEC EDGAR API collector for company filings."""

import logging
import os
import re
import uuid
from datetime import datetime, timedelta

import httpx

from ..rag.schemas import Document, DocumentMetadata, DocumentType
from .base import BaseCollector, CollectorResult, CollectorStatus

logger = logging.getLogger(__name__)


class SECEdgarCollector(BaseCollector):
    """Collector for SEC EDGAR filings.

    Provides:
    - 10-K annual reports
    - 10-Q quarterly reports
    - 8-K current reports
    - Company facts and financials
    """

    name = "sec_edgar"
    description = "Company filings (10-K, 10-Q, 8-K) from SEC EDGAR"

    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    COMPANY_FACTS_URL = f"{BASE_URL}/api/xbrl/companyfacts"

    # CIK numbers for popular companies (Central Index Key)
    DEFAULT_CIKS = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "META": "0001326801",
        "NVDA": "0001045810",
        "JPM": "0000019617",
        "V": "0001403161",
    }

    def __init__(self, user_agent: str | None = None):
        """Initialize SEC EDGAR collector.

        Args:
            user_agent: User agent string required by SEC.
                       Format: "Name email@domain.com"
        """
        super().__init__()
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        self._client = None

    def _get_client(self) -> httpx.Client:
        """Get HTTP client with proper headers."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=30.0,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if user agent is configured."""
        if not self.user_agent:
            return False
        # Check for placeholder value
        if "your" in self.user_agent.lower() or "@" not in self.user_agent:
            return False
        return True

    def collect(
        self,
        tickers: list[str] | None = None,
        filing_types: list[str] | None = None,
        max_filings_per_company: int = 3,
        **kwargs,
    ) -> CollectorResult:
        """Collect SEC filings for specified companies.

        Args:
            tickers: Company ticker symbols. Uses defaults if not provided.
            filing_types: Types of filings (10-K, 10-Q, 8-K). Defaults to 10-K.
            max_filings_per_company: Maximum filings to collect per company.

        Returns:
            CollectorResult with collected documents.
        """
        if not self.is_configured():
            return self._create_result(
                CollectorStatus.NO_API_KEY,
                errors=["SEC_USER_AGENT not configured. Set to 'YourName your@email.com'"],
            )

        documents = []
        errors = []

        tickers = tickers or list(self.DEFAULT_CIKS.keys())[:4]  # Default to top 4
        filing_types = filing_types or ["10-K"]

        for ticker in tickers:
            try:
                # Get CIK for ticker
                cik = self._get_cik(ticker)
                if not cik:
                    errors.append(f"Could not find CIK for {ticker}")
                    continue

                # Get company filings
                filings = self._get_recent_filings(cik, filing_types, max_filings_per_company)

                for filing in filings:
                    doc = self._create_filing_document(ticker, cik, filing)
                    if doc:
                        documents.append(doc)

                logger.info(f"Collected {len(filings)} filings for {ticker}")

            except Exception as e:
                error_msg = f"Failed to collect filings for {ticker}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Determine status
        if not documents and errors:
            status = CollectorStatus.FAILED
        elif documents and errors:
            status = CollectorStatus.PARTIAL
        elif documents:
            status = CollectorStatus.SUCCESS
        else:
            status = CollectorStatus.FAILED
            errors.append("No documents collected")

        return self._create_result(status, documents, errors)

    def _get_cik(self, ticker: str) -> str | None:
        """Get CIK number for a ticker symbol.

        Args:
            ticker: Company ticker symbol.

        Returns:
            CIK number or None.
        """
        # Check known CIKs first
        if ticker.upper() in self.DEFAULT_CIKS:
            return self.DEFAULT_CIKS[ticker.upper()]

        # Search SEC for CIK
        try:
            client = self._get_client()
            response = client.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={
                    "action": "getcompany",
                    "company": ticker,
                    "type": "",
                    "dateb": "",
                    "owner": "include",
                    "count": "1",
                    "output": "atom",
                },
            )

            # Parse CIK from response (simplified)
            text = response.text
            match = re.search(r"CIK=(\d+)", text)
            if match:
                return match.group(1).zfill(10)

        except Exception as e:
            logger.warning(f"Could not look up CIK for {ticker}: {e}")

        return None

    def _get_recent_filings(
        self,
        cik: str,
        filing_types: list[str],
        max_filings: int,
    ) -> list[dict]:
        """Get recent filings for a company.

        Args:
            cik: Company CIK number.
            filing_types: Types of filings to retrieve.
            max_filings: Maximum number of filings.

        Returns:
            List of filing metadata dictionaries.
        """
        client = self._get_client()

        # Get company submissions
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

        filings = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form in filing_types and len(filings) < max_filings:
                filings.append({
                    "form": form,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                    "accession_number": accession_numbers[i] if i < len(accession_numbers) else None,
                    "primary_document": primary_docs[i] if i < len(primary_docs) else None,
                    "company_name": data.get("name", "Unknown"),
                })

        return filings

    def _create_filing_document(
        self,
        ticker: str,
        cik: str,
        filing: dict,
    ) -> Document | None:
        """Create a Document from filing metadata.

        Args:
            ticker: Company ticker.
            cik: Company CIK.
            filing: Filing metadata.

        Returns:
            Document instance or None.
        """
        try:
            form = filing.get("form", "Unknown")
            company_name = filing.get("company_name", ticker)
            filing_date_str = filing.get("filing_date", "")
            accession = filing.get("accession_number", "")

            # Parse filing date
            if filing_date_str:
                try:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                except ValueError:
                    filing_date = datetime.now()
            else:
                filing_date = datetime.now()

            # Build SEC URL
            accession_clean = accession.replace("-", "")
            sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}"

            # Determine document type and description
            form_descriptions = {
                "10-K": "Annual Report - comprehensive overview of business and financial condition",
                "10-Q": "Quarterly Report - unaudited financial statements and business updates",
                "8-K": "Current Report - material events and corporate changes",
            }
            form_desc = form_descriptions.get(form, f"{form} Filing")

            content = f"""# {company_name} ({ticker}) - {form} Filing

## Filing Information
- **Form Type:** {form}
- **Filing Date:** {filing_date.strftime("%Y-%m-%d")}
- **Company:** {company_name}
- **Ticker:** {ticker}
- **CIK:** {cik}

## Description
{form_desc}

## Key Points
This {form} filing provides {'annual' if form == '10-K' else 'quarterly' if form == '10-Q' else 'current'} information about {company_name}.

### What to Look For in a {form}:
{"- Business overview and risk factors" if form == "10-K" else ""}
{"- Management discussion and analysis (MD&A)" if form in ["10-K", "10-Q"] else ""}
{"- Financial statements and notes" if form in ["10-K", "10-Q"] else ""}
{"- Material events and their impact" if form == "8-K" else ""}

## Source
[View Full Filing on SEC EDGAR]({sec_url})

*Note: This is a summary. Please refer to the official SEC filing for complete information.*
"""

            # Determine expiry based on form type
            if form == "10-K":
                expiry_days = 365  # Annual reports relevant for a year
            elif form == "10-Q":
                expiry_days = 90  # Quarterly reports
            else:
                expiry_days = 30  # Current reports

            metadata = DocumentMetadata(
                source="SEC EDGAR",
                document_type=DocumentType.RESEARCH_REPORT,
                publish_date=filing_date,
                expiry_date=filing_date + timedelta(days=expiry_days),
                tags=["sec", form.lower(), "filing", ticker.lower()],
                author="SEC EDGAR",
                url=sec_url,
                ticker=ticker,
            )

            return Document(
                doc_id=f"sec-{ticker.lower()}-{form.lower()}-{uuid.uuid4().hex[:8]}",
                title=f"{company_name} ({ticker}) - {form} ({filing_date.strftime('%Y-%m-%d')})",
                content=content,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to create document for {ticker} {filing.get('form')}: {e}")
            return None

    def collect_annual_reports(self, tickers: list[str] | None = None) -> CollectorResult:
        """Convenience method to collect only 10-K annual reports.

        Args:
            tickers: Company tickers.

        Returns:
            CollectorResult with 10-K documents.
        """
        return self.collect(tickers=tickers, filing_types=["10-K"])

    def collect_quarterly_reports(self, tickers: list[str] | None = None) -> CollectorResult:
        """Convenience method to collect only 10-Q quarterly reports.

        Args:
            tickers: Company tickers.

        Returns:
            CollectorResult with 10-Q documents.
        """
        return self.collect(tickers=tickers, filing_types=["10-Q"])
