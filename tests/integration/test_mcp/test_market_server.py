"""Integration tests for Market MCP server."""

import os

import pytest

from backend.src.mcp.servers.market_server import MarketMCPServer


@pytest.fixture
def market_server():
    """Create a market MCP server for testing."""
    return MarketMCPServer()


class TestMarketMCPServerConfiguration:
    """Tests for Market MCP server configuration."""

    def test_server_initialization(self, market_server):
        """Test that server initializes correctly."""
        assert market_server.name == "market"
        assert market_server.description == "Real-time market data and quotes"

    def test_popular_symbols(self, market_server):
        """Test that popular symbols are defined."""
        assert len(market_server.POPULAR_SYMBOLS) > 0
        assert "SPY" in market_server.POPULAR_SYMBOLS
        assert "AAPL" in market_server.POPULAR_SYMBOLS


class TestMarketMCPServerTools:
    """Tests for Market MCP server tools."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_stock_quote(self, market_server):
        """Test getting a stock quote (requires API key)."""
        result = await market_server.get_stock_quote(symbol="SPY")

        assert "symbol" in result or "cached" in result
        if "symbol" in result:
            assert result["symbol"] == "SPY"
            assert "price" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_etf_profile(self, market_server):
        """Test getting an ETF profile (requires API key)."""
        result = await market_server.get_etf_profile(symbol="VOO")

        assert "symbol" in result or "cached" in result
        if "symbol" in result:
            assert result["symbol"] == "VOO"
            assert "name" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_market_overview(self, market_server):
        """Test getting market overview (requires API key)."""
        result = await market_server.get_market_overview()

        assert "top_gainers" in result or "cached" in result
        if "top_gainers" in result:
            assert isinstance(result["top_gainers"], list)
            assert isinstance(result["top_losers"], list)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_multiple_quotes(self, market_server):
        """Test getting multiple quotes (requires API key)."""
        result = await market_server.get_multiple_quotes(symbols=["AAPL", "MSFT"])

        assert "quotes" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_get_multiple_quotes_limit(self, market_server):
        """Test that multiple quotes enforces limit."""
        # Should raise error for more than 5 symbols
        with pytest.raises(Exception) as exc_info:
            await market_server.get_multiple_quotes(symbols=["A", "B", "C", "D", "E", "F"])

        assert "Maximum 5 symbols" in str(exc_info.value)


class TestMarketMCPServerResources:
    """Tests for Market MCP server resources."""

    def test_popular_symbols_resource(self, market_server):
        """Test the popular symbols resource."""
        result = market_server.list_popular_symbols()

        import json
        symbols = json.loads(result)

        assert "major_etfs" in symbols
        assert "tech_stocks" in symbols
        assert "SPY" in symbols["major_etfs"]


class TestMarketMCPServerRateLimiting:
    """Tests for Market MCP server rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_tracking(self, market_server):
        """Test that rate limiting tracks calls."""
        initial_calls = len(market_server._call_times)

        # Check rate limit (this adds to call times)
        await market_server._check_rate_limit()

        assert len(market_server._call_times) == initial_calls + 1

    def test_rate_limit_config(self, market_server):
        """Test rate limit configuration."""
        assert market_server._rate_limit > 0


class TestMarketMCPServerCaching:
    """Tests for Market MCP server caching."""

    def test_quote_cache(self, market_server):
        """Test quote caching."""
        # Add to cache
        market_server._set_cached("quote:TEST", {"symbol": "TEST", "price": 100})

        # Should be valid
        assert market_server._is_cache_valid("quote:TEST")

        # Should retrieve correctly
        cached = market_server._get_cached("quote:TEST")
        assert cached["symbol"] == "TEST"
        assert cached["price"] == 100
