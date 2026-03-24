"""News data MCP server.

Wraps existing collectors to provide news data via MCP tools and resources.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

from fastmcp.exceptions import ToolError

from backend.src.recommendation.collectors.alpha_vantage import AlphaVantageCollector
from backend.src.recommendation.collectors.news_api import NewsAPICollector

from ..schemas import (
    MCPDataSource,
    MCPToolStatus,
    MarketSentiment,
    NewsArticle,
    NewsFeed,
)
from .base import BaseMCPServer

logger = logging.getLogger(__name__)


class NewsMCPServer(BaseMCPServer):
    """MCP server for news data.

    Provides tools for:
    - Getting market news with sentiment analysis
    - Getting ticker-specific sentiment
    - Searching business news

    Resources:
    - news://sources - List available news sources
    - news://status - Collector configuration status
    """

    name = "news"
    description = "Financial news and sentiment analysis"

    def __init__(
        self,
        alpha_vantage_key: str | None = None,
        news_api_key: str | None = None,
    ):
        """Initialize news MCP server.

        Args:
            alpha_vantage_key: Alpha Vantage API key.
            news_api_key: News API key.
        """
        # Initialize collectors
        self._av_collector = AlphaVantageCollector(
            api_key=alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        )
        self._news_collector = NewsAPICollector(
            api_key=news_api_key or os.getenv("NEWS_API_KEY")
        )

        # Thread pool for running sync collectors in async context
        self._executor = ThreadPoolExecutor(max_workers=2)

        # News cache (5 min TTL)
        self._cache: dict[str, tuple[datetime, object]] = {}
        self._cache_ttl = 300  # 5 minutes

        # Public references to tool functions (set during _register_tools)
        self.get_market_news = None
        self.get_ticker_sentiment = None
        self.search_business_news = None

        super().__init__(data_source=MCPDataSource.NEWS_API)

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return (datetime.now() - cached_time).total_seconds() < self._cache_ttl

    def _get_cached(self, key: str) -> object | None:
        """Get a cached value if valid."""
        if self._is_cache_valid(key):
            return self._cache[key][1]
        return None

    def _set_cached(self, key: str, value: object) -> None:
        """Cache a value."""
        self._cache[key] = (datetime.now(), value)

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(func, *args, **kwargs),
        )

    def _register_tools(self) -> None:
        """Register MCP tools for news data."""

        @self._mcp.tool()
        async def get_market_news(
            topics: list[str] | None = None,
            limit: int = 10,
        ) -> dict:
            """Get latest market news with sentiment analysis.

            Args:
                topics: News topics to filter by (e.g., ["technology", "finance"]).
                        Defaults to ["financial_markets"].
                limit: Maximum number of articles to return.

            Returns:
                NewsFeed data with articles and sentiment.
            """
            self.log_tool_call("get_market_news", topics=topics, limit=limit)

            topics = topics or ["financial_markets"]
            cache_key = f"news:{','.join(sorted(topics))}:{limit}"

            # Check cache
            cached = self._get_cached(cache_key)
            if cached:
                return {"cached": True, **cached}

            # Check configuration
            if not self._av_collector.is_configured():
                raise ToolError(
                    "Alpha Vantage API not configured. Set ALPHA_VANTAGE_API_KEY."
                )

            # Collect news using Alpha Vantage
            result = await self._run_sync(
                self._av_collector.collect_news_only,
                topics=topics,
            )

            if not result.is_success:
                # Distinguish real errors (rate limit, auth) from simply no articles
                error_msg = result.errors[0] if result.errors else "No news data"
                if any(k in error_msg.lower() for k in ("rate limit", "api key", "not configured", "error message")):
                    raise ToolError(f"Failed to collect news: {error_msg}")
                # No articles found — return empty feed gracefully
                logger.warning(f"News collector returned no articles: {error_msg}")
                feed = NewsFeed(articles=[], total_count=0, topics=topics)
                return feed.model_dump(mode="json")

            # Parse documents into NewsArticle format
            articles = []
            for doc in result.documents[:limit]:
                article = NewsArticle(
                    title=doc.title,
                    summary=doc.content[:500] if doc.content else None,
                    source=doc.metadata.source,
                    url=doc.metadata.url,
                    published_at=doc.metadata.publish_date,
                    sentiment_label=next(
                        (t for t in doc.metadata.tags if t in ["bullish", "bearish", "neutral"]),
                        None,
                    ),
                    sentiment_score=None,  # Would need to parse from content
                    tickers=[doc.metadata.ticker] if doc.metadata.ticker else [],
                    topics=topics,
                )
                articles.append(article)

            feed = NewsFeed(
                articles=articles,
                total_count=len(articles),
                topics=topics,
            )

            feed_dict = feed.model_dump(mode="json")
            self._set_cached(cache_key, feed_dict)
            return feed_dict

        @self._mcp.tool()
        async def get_ticker_sentiment(ticker: str) -> dict:
            """Get sentiment analysis for a specific ticker.

            Args:
                ticker: Stock ticker symbol (e.g., "AAPL").

            Returns:
                MarketSentiment data for the ticker.
            """
            self.log_tool_call("get_ticker_sentiment", ticker=ticker)

            ticker = ticker.upper()
            cache_key = f"sentiment:{ticker}"

            # Check cache
            cached = self._get_cached(cache_key)
            if cached:
                return {"cached": True, **cached}

            if not self._av_collector.is_configured():
                raise ToolError(
                    "Alpha Vantage API not configured. Set ALPHA_VANTAGE_API_KEY."
                )

            # Collect news for the ticker
            result = await self._run_sync(
                self._av_collector.collect_news_only,
                topics=[ticker],  # Use ticker as topic to filter
            )

            if not result.is_success or not result.documents:
                # Return neutral sentiment if no news found
                sentiment = MarketSentiment(
                    ticker=ticker,
                    sentiment_label="Neutral",
                    sentiment_score=0.0,
                    article_count=0,
                    sources=[],
                )
                return sentiment.model_dump(mode="json")

            # Aggregate sentiment from articles
            sentiment_scores = []
            sources = set()

            for doc in result.documents:
                sources.add(doc.metadata.source)
                # Try to extract sentiment from tags
                if "bullish" in doc.metadata.tags:
                    sentiment_scores.append(0.5)
                elif "bearish" in doc.metadata.tags:
                    sentiment_scores.append(-0.5)
                else:
                    sentiment_scores.append(0.0)

            avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            if avg_score > 0.25:
                label = "Bullish"
            elif avg_score < -0.25:
                label = "Bearish"
            else:
                label = "Neutral"

            sentiment = MarketSentiment(
                ticker=ticker,
                sentiment_label=label,
                sentiment_score=avg_score,
                article_count=len(result.documents),
                sources=list(sources),
            )

            sentiment_dict = sentiment.model_dump(mode="json")
            self._set_cached(cache_key, sentiment_dict)
            return sentiment_dict

        @self._mcp.tool()
        async def search_business_news(
            query: str,
            max_articles: int = 10,
        ) -> dict:
            """Search for business news articles.

            Args:
                query: Search query (e.g., "federal reserve interest rates").
                max_articles: Maximum number of articles to return.

            Returns:
                NewsFeed data with matching articles.
            """
            self.log_tool_call("search_business_news", query=query, max_articles=max_articles)

            cache_key = f"search:{query}:{max_articles}"

            # Check cache
            cached = self._get_cached(cache_key)
            if cached:
                return {"cached": True, **cached}

            # Try News API first (better for search)
            if self._news_collector.is_configured():
                result = await self._run_sync(
                    self._news_collector.collect,
                    query=query,
                    max_articles=max_articles,
                )

                if result.is_success and result.documents:
                    articles = []
                    for doc in result.documents[:max_articles]:
                        article = NewsArticle(
                            title=doc.title,
                            summary=doc.content[:500] if doc.content else None,
                            source=doc.metadata.source,
                            url=doc.metadata.url,
                            published_at=doc.metadata.publish_date,
                        )
                        articles.append(article)

                    feed = NewsFeed(
                        articles=articles,
                        total_count=len(articles),
                        query=query,
                    )

                    feed_dict = feed.model_dump(mode="json")
                    self._set_cached(cache_key, feed_dict)
                    return feed_dict

            # Fallback to Alpha Vantage if News API not configured or failed
            if self._av_collector.is_configured():
                result = await self._run_sync(
                    self._av_collector.collect_news_only,
                    topics=[query.replace(" ", "_")],
                )

                if result.is_success and result.documents:
                    articles = []
                    for doc in result.documents[:max_articles]:
                        article = NewsArticle(
                            title=doc.title,
                            summary=doc.content[:500] if doc.content else None,
                            source=doc.metadata.source,
                            url=doc.metadata.url,
                            published_at=doc.metadata.publish_date,
                        )
                        articles.append(article)

                    feed = NewsFeed(
                        articles=articles,
                        total_count=len(articles),
                        query=query,
                    )

                    feed_dict = feed.model_dump(mode="json")
                    self._set_cached(cache_key, feed_dict)
                    return feed_dict

            raise ToolError(
                "No news sources configured. Set ALPHA_VANTAGE_API_KEY or NEWS_API_KEY."
            )

        # Save references for direct invocation
        self.get_market_news = get_market_news.fn
        self.get_ticker_sentiment = get_ticker_sentiment.fn
        self.search_business_news = search_business_news.fn

    def _register_resources(self) -> None:
        """Register MCP resources for news data."""

        @self._mcp.resource("news://sources")
        def list_news_sources() -> str:
            """List available news data sources."""
            sources = []

            if self._av_collector.is_configured():
                sources.append({
                    "name": "Alpha Vantage",
                    "type": "financial_news",
                    "features": ["sentiment", "ticker_sentiment", "market_news"],
                    "configured": True,
                })
            else:
                sources.append({
                    "name": "Alpha Vantage",
                    "configured": False,
                    "env_var": "ALPHA_VANTAGE_API_KEY",
                })

            if self._news_collector.is_configured():
                sources.append({
                    "name": "News API",
                    "type": "business_news",
                    "features": ["search", "headlines", "business_news"],
                    "configured": True,
                })
            else:
                sources.append({
                    "name": "News API",
                    "configured": False,
                    "env_var": "NEWS_API_KEY",
                })

            import json
            return json.dumps(sources, indent=2)

        @self._mcp.resource("news://status")
        def get_status() -> str:
            """Get collector configuration status."""
            status = {
                "alpha_vantage": self._av_collector.get_status(),
                "news_api": self._news_collector.get_status(),
                "cache_entries": len(self._cache),
                "cache_ttl_seconds": self._cache_ttl,
            }
            import json
            return json.dumps(status, indent=2)

        # Save references for direct invocation
        self.list_news_sources = list_news_sources.fn
        self.get_status = get_status.fn


def main():
    """Run the news MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = NewsMCPServer()
    server.run()


if __name__ == "__main__":
    main()
