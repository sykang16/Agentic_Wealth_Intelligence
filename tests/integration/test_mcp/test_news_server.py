"""Integration tests for News MCP server."""

import os

import pytest

from backend.src.mcp.servers.news_server import NewsMCPServer


@pytest.fixture
def news_server():
    """Create a news MCP server for testing."""
    return NewsMCPServer()


class TestNewsMCPServerConfiguration:
    """Tests for News MCP server configuration."""

    def test_server_initialization(self, news_server):
        """Test that server initializes correctly."""
        assert news_server.name == "news"
        assert news_server.description == "Financial news and sentiment analysis"

    def test_collector_status(self, news_server):
        """Test getting collector status."""
        av_status = news_server._av_collector.get_status()
        news_status = news_server._news_collector.get_status()

        assert "name" in av_status
        assert "configured" in av_status
        assert "name" in news_status
        assert "configured" in news_status


class TestNewsMCPServerTools:
    """Tests for News MCP server tools."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_market_news(self, news_server):
        """Test getting market news (requires API key)."""
        result = await news_server.get_market_news(topics=["financial_markets"], limit=5)

        assert "articles" in result or "cached" in result
        if "articles" in result:
            assert isinstance(result["articles"], list)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_ticker_sentiment(self, news_server):
        """Test getting ticker sentiment (requires API key)."""
        result = await news_server.get_ticker_sentiment(ticker="AAPL")

        assert "ticker" in result or "cached" in result
        if "ticker" in result:
            assert result["ticker"] == "AAPL"
            assert "sentiment_label" in result
            assert "sentiment_score" in result

    @pytest.mark.asyncio
    async def test_search_business_news_no_config(self, news_server):
        """Test that search fails gracefully without API keys."""
        # Clear any existing API keys for this test
        news_server._av_collector.api_key = None
        news_server._news_collector.api_key = None

        # Should raise ToolError when no APIs are configured
        with pytest.raises(Exception) as exc_info:
            await news_server.search_business_news(query="stock market", max_articles=5)

        assert "configured" in str(exc_info.value).lower()


class TestNewsMCPServerResources:
    """Tests for News MCP server resources."""

    def test_list_news_sources_resource(self, news_server):
        """Test the news sources resource."""
        result = news_server.list_news_sources()

        import json
        sources = json.loads(result)

        assert isinstance(sources, list)
        assert len(sources) >= 2  # Alpha Vantage and News API

    def test_get_status_resource(self, news_server):
        """Test the status resource."""
        result = news_server.get_status()

        import json
        status = json.loads(result)

        assert "alpha_vantage" in status
        assert "news_api" in status
        assert "cache_entries" in status


class TestNewsMCPServerCaching:
    """Tests for News MCP server caching."""

    def test_cache_validation(self, news_server):
        """Test cache validation logic."""
        # Initially cache should be empty
        assert not news_server._is_cache_valid("test_key")

        # Add to cache
        news_server._set_cached("test_key", {"data": "test"})

        # Should be valid now
        assert news_server._is_cache_valid("test_key")

        # Should be able to retrieve
        cached = news_server._get_cached("test_key")
        assert cached == {"data": "test"}

    def test_cache_expiry(self, news_server):
        """Test that cache entries expire."""
        from datetime import datetime, timedelta

        # Set a cache entry
        news_server._set_cached("expire_test", {"data": "test"})

        # Manually set the cache time to be old
        old_time = datetime.now() - timedelta(seconds=news_server._cache_ttl + 60)
        news_server._cache["expire_test"] = (old_time, {"data": "test"})

        # Should now be invalid
        assert not news_server._is_cache_valid("expire_test")
        assert news_server._get_cached("expire_test") is None
