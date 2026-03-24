"""Alpha Vantage API collector for financial news and market data."""

import logging
import os
import time
import uuid
from datetime import datetime, timedelta

import httpx

from ..rag.schemas import Document, DocumentMetadata, DocumentType
from .base import BaseCollector, CollectorResult, CollectorStatus

logger = logging.getLogger(__name__)


class AlphaVantageCollector(BaseCollector):
    """Collector for Alpha Vantage financial data API.

    Provides:
    - Market news and sentiment
    - Company/ETF overview data
    - Top gainers/losers market data
    """

    name = "alpha_vantage"
    description = "Financial news, ETF data, and market sentiment from Alpha Vantage"

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        """Initialize Alpha Vantage collector.

        Args:
            api_key: Alpha Vantage API key. If not provided, reads from env.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self._client = httpx.Client(timeout=30.0)
        self._last_request_time: float = 0
        self._min_request_interval: float = 1.5  # seconds between requests (free tier: 1/sec)

    def _rate_limited_get(self, **kwargs) -> httpx.Response:
        """Make a GET request with rate limiting to respect API limits."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.monotonic()
        response = self._client.get(self.BASE_URL, **kwargs)
        response.raise_for_status()
        return response

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != "your-alpha-vantage-key-here")

    def collect(
        self,
        collect_news: bool = True,
        collect_etf_data: bool = True,
        news_topics: list[str] | None = None,
        etf_symbols: list[str] | None = None,
        **kwargs,
    ) -> CollectorResult:
        """Collect financial data from Alpha Vantage.

        Args:
            collect_news: Whether to collect market news.
            collect_etf_data: Whether to collect ETF overview data.
            news_topics: Topics for news (e.g., ["technology", "finance"]).
            etf_symbols: ETF symbols to get data for.

        Returns:
            CollectorResult with collected documents.
        """
        if not self.is_configured():
            return self._create_result(
                CollectorStatus.NO_API_KEY,
                errors=["Alpha Vantage API key not configured"],
            )

        documents = []
        errors = []

        # Collect news
        if collect_news:
            try:
                news_docs = self._collect_news(news_topics or ["financial_markets"])
                documents.extend(news_docs)
                logger.info(f"Collected {len(news_docs)} news articles")
            except Exception as e:
                error_msg = f"Failed to collect news: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Collect ETF data (via yfinance — no API key required)
        if collect_etf_data:
            symbols = etf_symbols or ["SPY", "QQQ", "VTI", "BND", "GLD"]
            for symbol in symbols:
                try:
                    etf_doc = self._collect_etf_overview_yf(symbol)
                    if etf_doc:
                        documents.append(etf_doc)
                except Exception as e:
                    error_msg = f"Failed to collect ETF data for {symbol}: {e}"
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

    def _collect_news(self, topics: list[str]) -> list[Document]:
        """Collect news articles for given topics.

        Args:
            topics: List of topics to search for.

        Returns:
            List of Document instances.
        """
        documents = []

        for topic in topics:
            params = {
                "function": "NEWS_SENTIMENT",
                "topics": topic,
                "apikey": self.api_key,
                "limit": 10,  # Free tier limit
            }

            response = self._rate_limited_get(params=params)
            data = response.json()

            # Check for API error / rate-limit
            if "Note" in data:
                raise Exception(f"Alpha Vantage rate limit reached: {data['Note']}")
            if "Error Message" in data:
                raise Exception(f"Alpha Vantage API error: {data['Error Message']}")

            # Parse news feed
            feed = data.get("feed", [])
            for article in feed:
                doc = self._parse_news_article(article, topic)
                if doc:
                    documents.append(doc)

        return documents

    def _parse_news_article(self, article: dict, topic: str) -> Document | None:
        """Parse a news article into a Document.

        Args:
            article: Raw article data from API.
            topic: The topic this article was found under.

        Returns:
            Document instance or None if parsing fails.
        """
        try:
            title = article.get("title", "Untitled")
            summary = article.get("summary", "")
            source = article.get("source", "Unknown")
            url = article.get("url", "")
            time_published = article.get("time_published", "")

            # Parse publish time
            if time_published:
                try:
                    publish_date = datetime.strptime(time_published[:8], "%Y%m%d")
                except ValueError:
                    publish_date = datetime.now()
            else:
                publish_date = datetime.now()

            # Get sentiment
            sentiment = article.get("overall_sentiment_label", "Neutral")
            sentiment_score = article.get("overall_sentiment_score", 0)

            # Get related tickers
            tickers = [t.get("ticker", "") for t in article.get("ticker_sentiment", [])]
            ticker_str = ", ".join(tickers[:5]) if tickers else "General Market"

            # Build content
            content = f"""# {title}

**Source:** {source}
**Published:** {publish_date.strftime("%Y-%m-%d")}
**Sentiment:** {sentiment} (Score: {sentiment_score:.2f})
**Related Tickers:** {ticker_str}

## Summary
{summary}

## Analysis
This article discusses {topic.replace('_', ' ')} with an overall {sentiment.lower()} sentiment.
The sentiment score of {sentiment_score:.2f} indicates {'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'} market perception.
"""

            # Create metadata
            metadata = DocumentMetadata(
                source=source,
                document_type=DocumentType.NEWS_ARTICLE,
                publish_date=publish_date,
                expiry_date=publish_date + timedelta(days=7),  # News expires in 7 days
                tags=["news", topic, sentiment.lower()],
                author=source,
                url=url,
                ticker=tickers[0] if tickers else None,
            )

            return Document(
                doc_id=f"news-av-{uuid.uuid4().hex[:8]}",
                title=title,
                content=content,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse news article: {e}")
            return None

    def _collect_etf_overview_yf(self, symbol: str) -> Document | None:
        """Collect ETF/stock overview data via yfinance (no API key required).

        Args:
            symbol: ETF or stock ticker symbol.

        Returns:
            Document instance or None if collection fails.
        """
        import yfinance as yf

        info = yf.Ticker(symbol).info
        if not info or info.get("trailingPegRatio") is None and not info.get("longName"):
            logger.warning("No yfinance data found for %s", symbol)
            return None

        name = info.get("longName") or info.get("shortName") or symbol
        description = info.get("longBusinessSummary") or "No description available."
        sector = info.get("sector") or info.get("category") or "Diversified"
        asset_type = info.get("quoteType") or "ETF"
        exchange = info.get("exchange") or "Unknown"

        pe_ratio = info.get("trailingPE") or info.get("forwardPE") or "N/A"
        dividend_yield = info.get("dividendYield")
        dividend_yield_str = f"{dividend_yield:.2%}" if dividend_yield else "N/A"
        market_cap = info.get("marketCap")
        market_cap_str = f"${market_cap:,}" if market_cap else "N/A"
        week_52_high = info.get("fiftyTwoWeekHigh") or "N/A"
        week_52_low = info.get("fiftyTwoWeekLow") or "N/A"

        content = f"""# {name} ({symbol})

## Overview
{description}

## Key Information
- **Asset Type:** {asset_type}
- **Exchange:** {exchange}
- **Sector:** {sector}

## Financial Metrics
- **P/E Ratio:** {pe_ratio}
- **Dividend Yield:** {dividend_yield_str}
- **Market Cap:** {market_cap_str}
- **52-Week High:** {week_52_high}
- **52-Week Low:** {week_52_low}

## Investment Considerations
This {asset_type.lower()} trades on {exchange} in the {sector} sector.
Investors should consider the current valuation metrics and market conditions before investing.
"""

        metadata = DocumentMetadata(
            source="Yahoo Finance",
            document_type=DocumentType.ETF_FACTSHEET,
            publish_date=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=30),
            tags=["etf", sector.lower(), "market_data"],
            ticker=symbol,
            sector=sector,
        )

        return Document(
            doc_id=f"etf-yf-{symbol.lower()}-{uuid.uuid4().hex[:8]}",
            title=f"{name} ({symbol}) - Market Data",
            content=content,
            metadata=metadata,
        )

    def _parse_etf_data(self, data: dict, symbol: str) -> Document | None:
        """Parse ETF/stock data into a Document.

        Args:
            data: Raw data from API.
            symbol: Ticker symbol.

        Returns:
            Document instance or None.
        """
        try:
            name = data.get("Name", symbol)
            description = data.get("Description", "No description available.")
            sector = data.get("Sector", "Diversified")
            asset_type = data.get("AssetType", "ETF")
            exchange = data.get("Exchange", "Unknown")

            # Financial metrics
            pe_ratio = data.get("PERatio", "N/A")
            dividend_yield = data.get("DividendYield", "N/A")
            market_cap = data.get("MarketCapitalization", "N/A")
            week_52_high = data.get("52WeekHigh", "N/A")
            week_52_low = data.get("52WeekLow", "N/A")

            content = f"""# {name} ({symbol})

## Overview
{description}

## Key Information
- **Asset Type:** {asset_type}
- **Exchange:** {exchange}
- **Sector:** {sector}

## Financial Metrics
- **P/E Ratio:** {pe_ratio}
- **Dividend Yield:** {dividend_yield}
- **Market Cap:** ${int(market_cap):,} if market_cap != "N/A" else "N/A"
- **52-Week High:** ${week_52_high}
- **52-Week Low:** ${week_52_low}

## Investment Considerations
This {asset_type.lower()} trades on {exchange} in the {sector} sector.
Investors should consider the current valuation metrics and market conditions before investing.
"""

            metadata = DocumentMetadata(
                source="Alpha Vantage",
                document_type=DocumentType.ETF_FACTSHEET,
                publish_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),  # Refresh monthly
                tags=["etf", sector.lower(), "market_data"],
                ticker=symbol,
                sector=sector,
            )

            return Document(
                doc_id=f"etf-av-{symbol.lower()}-{uuid.uuid4().hex[:8]}",
                title=f"{name} ({symbol}) - Market Data",
                content=content,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse ETF data for {symbol}: {e}")
            return None

    def collect_news_only(self, topics: list[str] | None = None) -> CollectorResult:
        """Convenience method to collect only news.

        Args:
            topics: News topics to collect.

        Returns:
            CollectorResult with news documents.
        """
        return self.collect(
            collect_news=True,
            collect_etf_data=False,
            news_topics=topics,
        )

    def collect_etf_only(self, symbols: list[str] | None = None) -> CollectorResult:
        """Convenience method to collect only ETF data.

        Args:
            symbols: ETF symbols to collect.

        Returns:
            CollectorResult with ETF documents.
        """
        return self.collect(
            collect_news=False,
            collect_etf_data=True,
            etf_symbols=symbols,
        )
