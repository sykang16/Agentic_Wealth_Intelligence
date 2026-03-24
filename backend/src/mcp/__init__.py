"""MCP (Model Context Protocol) integration module.

This module provides MCP servers and clients for real-time financial data access,
combining static RAG knowledge with live market data.
"""

from .schemas import (
    ETFProfile,
    HybridDataResponse,
    MarketOverview,
    MarketSentiment,
    MCPDataSource,
    MCPToolResult,
    MCPToolStatus,
    NewsArticle,
    NewsFeed,
    PortfolioAnalysisResult,
    PortfolioHolding,
    RAGSearchResult,
    StockQuote,
)

__all__ = [
    # Enums
    "MCPDataSource",
    "MCPToolStatus",
    # Core schemas
    "MCPToolResult",
    "StockQuote",
    "MarketSentiment",
    "NewsArticle",
    "NewsFeed",
    "ETFProfile",
    "MarketOverview",
    "PortfolioHolding",
    "PortfolioAnalysisResult",
    "RAGSearchResult",
    "HybridDataResponse",
]


def get_hybrid_provider():
    """Get a configured HybridDataProvider instance.

    Returns:
        HybridDataProvider instance.
    """
    from .integration import HybridDataProvider
    return HybridDataProvider()


def get_mcp_client():
    """Get a configured MCPClient instance.

    Returns:
        MCPClient instance.
    """
    from .client import create_default_client
    return create_default_client()
