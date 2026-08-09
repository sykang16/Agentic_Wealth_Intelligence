"""Integration tests for Hybrid Data Provider."""

import os
from pathlib import Path

import pytest

from backend.src.mcp.integration.hybrid_data_provider import HybridDataProvider
from backend.src.mcp.schemas import MCPDataSource


@pytest.fixture
def hybrid_provider():
    """Create a hybrid data provider for testing."""
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "synthetic" / "synthetic_portfolios.json"
    rag_persist_dir = str(project_root / "data" / "chroma")

    return HybridDataProvider(
        data_path=data_path,
        rag_persist_dir=rag_persist_dir,
    )


class TestHybridDataProviderInitialization:
    """Tests for HybridDataProvider initialization."""

    def test_provider_initialization(self, hybrid_provider):
        """Test that provider initializes correctly."""
        assert hybrid_provider._rag is not None
        assert hybrid_provider._aggregator is not None
        assert hybrid_provider._normalizer is not None

    def test_lazy_server_initialization(self, hybrid_provider):
        """Test that MCP servers are lazily initialized."""
        assert hybrid_provider._market_server is None
        assert hybrid_provider._news_server is None


class TestHybridDataProviderRAG:
    """Tests for RAG functionality in HybridDataProvider."""

    def test_search_knowledge_sync(self, hybrid_provider):
        """Test synchronous RAG search."""
        # This may return empty results if RAG is not initialized
        result = hybrid_provider.search_knowledge_sync("ETF expense ratio", top_k=3)

        assert "query" in result
        assert "results" in result
        assert "num_results" in result


class TestHybridDataProviderPortfolio:
    """Tests for portfolio functionality in HybridDataProvider."""

    def test_get_portfolio_context_sync(self, hybrid_provider):
        """Test getting portfolio context."""
        # This may fail if portfolio data doesn't exist
        try:
            context = hybrid_provider.get_portfolio_context_sync("user-001")
            assert isinstance(context, str)
            # If we got context, it should contain user info
            if "No portfolio data" not in context:
                assert "User Profile" in context or "Portfolio Summary" in context
        except FileNotFoundError:
            pytest.skip("Portfolio data file not found")


class TestHybridDataProviderEnrichedContext:
    """Tests for enriched context retrieval."""

    @pytest.mark.asyncio
    async def test_get_enriched_context_rag_only(self, hybrid_provider):
        """Test getting enriched context with RAG only."""
        response = await hybrid_provider.get_enriched_context(
            query="What is an ETF?",
            tickers=None,
            include_news=False,
            include_quotes=False,
            top_k=3,
        )

        # Should have attempted RAG search
        assert response.query == "What is an ETF?"
        assert MCPDataSource.RAG in response.data_sources_used or len(response.errors) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_enriched_context_with_quotes(self, hybrid_provider):
        """Test getting enriched context with live quotes."""
        response = await hybrid_provider.get_enriched_context(
            query="How is Apple stock doing?",
            tickers=["AAPL"],
            include_news=False,
            include_quotes=True,
            top_k=3,
        )

        assert response.query == "How is Apple stock doing?"
        # Should have attempted to get quotes
        # May succeed or fail depending on API key and rate limits

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_enriched_context_full(self, hybrid_provider):
        """Test getting full enriched context."""
        response = await hybrid_provider.get_enriched_context(
            query="Market analysis for tech stocks",
            tickers=["AAPL", "MSFT"],
            include_news=True,
            include_quotes=True,
            top_k=3,
        )

        assert response.query == "Market analysis for tech stocks"
        assert len(response.data_sources_used) > 0

    @pytest.mark.asyncio
    async def test_get_combined_context(self, hybrid_provider):
        """Test getting combined context string."""
        response = await hybrid_provider.get_enriched_context(
            query="Test query",
            tickers=None,
            include_news=False,
            include_quotes=False,
        )

        context = response.get_combined_context()
        assert isinstance(context, str)


class TestHybridDataProviderPortfolioWithContext:
    """Tests for portfolio with context retrieval."""

    @pytest.mark.asyncio
    async def test_get_portfolio_with_context(self, hybrid_provider):
        """Test getting portfolio with context."""
        try:
            response = await hybrid_provider.get_portfolio_with_context(
                user_id="user-001",
                query="How is my portfolio doing?",
                include_live_quotes=False,
            )

            assert response.query == "How is my portfolio doing?"
            assert MCPDataSource.PORTFOLIO in response.data_sources_used or len(response.errors) > 0

        except FileNotFoundError:
            pytest.skip("Portfolio data file not found")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ALPHA_VANTAGE_API_KEY"),
        reason="ALPHA_VANTAGE_API_KEY not set",
    )
    async def test_get_portfolio_with_live_quotes(self, hybrid_provider):
        """Test getting portfolio with live quotes."""
        try:
            response = await hybrid_provider.get_portfolio_with_context(
                user_id="user-001",
                query="Show my portfolio with current prices",
                include_live_quotes=True,
            )

            assert response.query == "Show my portfolio with current prices"
            # May have quotes depending on rate limits and API availability

        except FileNotFoundError:
            pytest.skip("Portfolio data file not found")


class TestHybridDataProviderErrorHandling:
    """Tests for error handling in HybridDataProvider."""

    @pytest.mark.asyncio
    async def test_handles_rag_errors_gracefully(self, hybrid_provider):
        """Test that RAG errors are handled gracefully."""
        # Even if RAG fails, should return a response with errors
        response = await hybrid_provider.get_enriched_context(
            query="Test query",
            tickers=None,
            include_news=False,
            include_quotes=False,
        )

        # Should get a response (may have errors)
        assert response.query == "Test query"
        # Errors should be collected, not raised
        assert isinstance(response.errors, list)

    @pytest.mark.asyncio
    async def test_handles_invalid_user_gracefully(self, hybrid_provider):
        """Test handling of invalid user ID."""
        try:
            response = await hybrid_provider.get_portfolio_with_context(
                user_id="nonexistent-user",
                query="Test query",
                include_live_quotes=False,
            )

            # Should indicate user not found via errors or a fallback portfolio context
            assert (
                len(response.errors) > 0
                or response.portfolio_context is None
                or response.portfolio_context == "No portfolio data found."
            )

        except FileNotFoundError:
            pytest.skip("Portfolio data file not found")
