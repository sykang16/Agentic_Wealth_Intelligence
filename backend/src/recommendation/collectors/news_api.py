"""News API collector for general financial news."""

import logging
import os
import uuid
from datetime import datetime, timedelta

import httpx

from ..rag.schemas import Document, DocumentMetadata, DocumentType
from .base import BaseCollector, CollectorResult, CollectorStatus

logger = logging.getLogger(__name__)


class NewsAPICollector(BaseCollector):
    """Collector for News API (newsapi.org).

    Provides:
    - Business and financial news
    - Market-related headlines
    - Economic news
    """

    name = "news_api"
    description = "Business and financial news from News API"

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str | None = None):
        """Initialize News API collector.

        Args:
            api_key: News API key. If not provided, reads from env.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self._client = httpx.Client(timeout=30.0)

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        if not self.api_key:
            return False
        # Check for placeholder
        if "your" in self.api_key.lower() or len(self.api_key) < 10:
            return False
        return True

    def collect(
        self,
        query: str | None = None,
        category: str = "business",
        max_articles: int = 10,
        days_back: int = 7,
        **kwargs,
    ) -> CollectorResult:
        """Collect news articles.

        Args:
            query: Search query (e.g., "stock market", "federal reserve").
            category: News category (business, technology, etc.).
            max_articles: Maximum articles to collect.
            days_back: How many days back to search.

        Returns:
            CollectorResult with collected documents.
        """
        if not self.is_configured():
            return self._create_result(
                CollectorStatus.NO_API_KEY,
                errors=["NEWS_API_KEY not configured"],
            )

        documents = []
        errors = []

        try:
            if query:
                # Search for specific query
                articles = self._search_news(query, max_articles, days_back)
            else:
                # Get top headlines for category
                articles = self._get_top_headlines(category, max_articles)

            for article in articles:
                doc = self._parse_article(article)
                if doc:
                    documents.append(doc)

            logger.info(f"Collected {len(documents)} news articles")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return self._create_result(
                    CollectorStatus.NO_API_KEY,
                    errors=["Invalid News API key"],
                )
            elif e.response.status_code == 429:
                return self._create_result(
                    CollectorStatus.RATE_LIMITED,
                    errors=["News API rate limit exceeded"],
                )
            else:
                errors.append(f"HTTP error: {e}")
        except Exception as e:
            errors.append(f"Failed to collect news: {e}")

        # Determine status
        if not documents and errors:
            status = CollectorStatus.FAILED
        elif documents and errors:
            status = CollectorStatus.PARTIAL
        elif documents:
            status = CollectorStatus.SUCCESS
        else:
            status = CollectorStatus.FAILED
            errors.append("No articles collected")

        return self._create_result(status, documents, errors)

    def _search_news(self, query: str, max_articles: int, days_back: int) -> list[dict]:
        """Search for news articles.

        Args:
            query: Search query.
            max_articles: Maximum results.
            days_back: Days to search back.

        Returns:
            List of article dictionaries.
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": min(max_articles, 100),
            "apiKey": self.api_key,
        }

        response = self._client.get(f"{self.BASE_URL}/everything", params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            raise Exception(data.get("message", "Unknown error"))

        return data.get("articles", [])

    def _get_top_headlines(self, category: str, max_articles: int) -> list[dict]:
        """Get top headlines for a category.

        Args:
            category: News category.
            max_articles: Maximum results.

        Returns:
            List of article dictionaries.
        """
        params = {
            "category": category,
            "country": "us",
            "pageSize": min(max_articles, 100),
            "apiKey": self.api_key,
        }

        response = self._client.get(f"{self.BASE_URL}/top-headlines", params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            raise Exception(data.get("message", "Unknown error"))

        return data.get("articles", [])

    def _parse_article(self, article: dict) -> Document | None:
        """Parse a news article into a Document.

        Args:
            article: Raw article data.

        Returns:
            Document instance or None.
        """
        try:
            title = article.get("title", "")
            if not title or title == "[Removed]":
                return None

            description = article.get("description", "") or ""
            content = article.get("content", "") or ""
            source_name = article.get("source", {}).get("name", "Unknown")
            author = article.get("author", "Unknown")
            url = article.get("url", "")
            published_at = article.get("publishedAt", "")

            # Parse publish date
            if published_at:
                try:
                    publish_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    publish_date = publish_date.replace(tzinfo=None)
                except ValueError:
                    publish_date = datetime.now()
            else:
                publish_date = datetime.now()

            # Build full content
            full_content = f"""# {title}

**Source:** {source_name}
**Author:** {author}
**Published:** {publish_date.strftime("%Y-%m-%d %H:%M")}

## Summary
{description}

## Content
{content if content else description}

## Source
[Read Full Article]({url})
"""

            metadata = DocumentMetadata(
                source=source_name,
                document_type=DocumentType.NEWS_ARTICLE,
                publish_date=publish_date,
                expiry_date=publish_date + timedelta(days=7),  # News expires in 7 days
                tags=["news", "business", source_name.lower().replace(" ", "_")],
                author=author,
                url=url,
            )

            return Document(
                doc_id=f"news-{uuid.uuid4().hex[:8]}",
                title=title,
                content=full_content,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None

    def collect_financial_news(self, max_articles: int = 10) -> CollectorResult:
        """Convenience method to collect financial news.

        Args:
            max_articles: Maximum articles.

        Returns:
            CollectorResult with financial news.
        """
        queries = ["stock market", "federal reserve", "economy"]
        all_documents = []
        all_errors = []

        for query in queries:
            result = self.collect(
                query=query,
                max_articles=max_articles // len(queries),
            )
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        if not all_documents and all_errors:
            status = CollectorStatus.FAILED
        elif all_documents:
            status = CollectorStatus.SUCCESS
        else:
            status = CollectorStatus.FAILED

        return self._create_result(status, all_documents, all_errors)
