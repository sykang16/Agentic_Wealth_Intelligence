"""Hybrid data provider combining RAG and MCP live data."""

import asyncio
import concurrent.futures
import logging
from decimal import Decimal
from pathlib import Path

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.recommendation.rag.initializer import RAGInitializer

from ..schemas import (
    HybridDataResponse,
    MCPDataSource,
    NewsArticle,
    StockQuote,
)
from .data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class HybridDataProvider:
    """Combines RAG static knowledge with live MCP data.

    Live market data (quotes + news) is fetched via MarketMCPServer,
    which is backed by yfinance — no API key required.
    """

    def __init__(
        self,
        rag_initializer: RAGInitializer | None = None,
        portfolio_aggregator: PortfolioAggregator | None = None,
        data_path: str | Path | None = None,
        rag_persist_dir: str | None = None,
    ):
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Initialize RAG
        if rag_initializer:
            self._rag = rag_initializer
        else:
            persist_dir = rag_persist_dir or str(project_root / "data" / "chroma")
            self._rag = RAGInitializer(persist_directory=persist_dir)

        # Initialize portfolio aggregator
        if portfolio_aggregator:
            self._aggregator = portfolio_aggregator
        else:
            path = Path(data_path) if data_path else project_root / "data" / "synthetic" / "synthetic_portfolios.json"
            self._aggregator = PortfolioAggregator(path)
            if path.exists():
                self._aggregator.load_data()

        self._normalizer = DataNormalizer()
        self._market_server = None  # lazy-loaded

    def _get_market_server(self):
        """Get or create the market MCP server (lazy init)."""
        if self._market_server is None:
            from ..servers.market_server import MarketMCPServer
            self._market_server = MarketMCPServer()
        return self._market_server

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def get_enriched_context(
        self,
        query: str,
        tickers: list[str] | None = None,
        include_news: bool = True,
        include_quotes: bool = True,
        top_k: int = 5,
    ) -> HybridDataResponse:
        """Get enriched context combining RAG and live market data.

        Args:
            query: Natural language query for RAG search.
            tickers: Portfolio ticker symbols to fetch live data for.
            include_news: Whether to include per-ticker news headlines.
            include_quotes: Whether to include live price quotes.
            top_k: Number of RAG results to include.

        Returns:
            HybridDataResponse with combined data.
        """
        response = HybridDataResponse(query=query)
        data_sources_used = []

        # 1. RAG knowledge context
        try:
            rag_result = self._rag.search(query, top_k=top_k)
            response.rag_context = rag_result.get("context")
            response.rag_results = rag_result.get("results", [])
            data_sources_used.append(MCPDataSource.RAG)
        except Exception as e:
            logger.error("RAG search failed: %s", e)
            response.errors.append(f"RAG search failed: {e}")

        if not tickers:
            response.data_sources_used = data_sources_used
            return response

        market_server = self._get_market_server()

        # 2. Live quotes
        if include_quotes:
            quotes = []
            for ticker in tickers[:10]:
                try:
                    quote_data = await market_server.get_stock_quote(symbol=ticker)
                    normalized = self._normalizer.normalize_quote(quote_data)
                    quote = StockQuote(
                        symbol=normalized.get("symbol", ticker),
                        price=normalized.get("price") or Decimal("0"),
                        previous_close=normalized.get("previous_close"),
                        change=normalized.get("change"),
                        change_percent=normalized.get("change_percent"),
                        volume=normalized.get("volume"),
                    )
                    quotes.append(quote)
                except Exception as e:
                    logger.warning("Failed to get quote for %s: %s", ticker, e)

            response.live_quotes = quotes
            if quotes:
                data_sources_used.append(MCPDataSource.YFINANCE)

        # 3. Per-ticker news headlines
        if include_news:
            articles = []
            for ticker in tickers[:5]:
                try:
                    news_data = await market_server.get_ticker_news(symbol=ticker, limit=2)
                    for item in news_data.get("articles", []):
                        normalized = self._normalizer.normalize_article(item)
                        article = NewsArticle(
                            title=normalized.get("title", ""),
                            summary=normalized.get("summary"),
                            source=normalized.get("source", "Unknown"),
                            url=normalized.get("url"),
                            published_at=normalized.get("published_at"),
                            tickers=normalized.get("tickers", [ticker]),
                        )
                        articles.append(article)
                except Exception as e:
                    logger.warning("Failed to get news for %s: %s", ticker, e)

            response.live_news = articles
            if articles:
                data_sources_used.append(MCPDataSource.YFINANCE)

        response.data_sources_used = data_sources_used
        return response

    async def get_portfolio_with_context(
        self,
        user_id: str,
        query: str,
        include_live_quotes: bool = True,
    ) -> HybridDataResponse:
        """Get portfolio data enriched with RAG context and live quotes.

        Args:
            user_id: User ID.
            query: Natural language query for RAG search.
            include_live_quotes: Whether to fetch live quotes for holdings.

        Returns:
            HybridDataResponse with portfolio context.
        """
        response = HybridDataResponse(query=query)
        data_sources_used = [MCPDataSource.PORTFOLIO]

        # Portfolio context
        try:
            response.portfolio_context = self._aggregator.format_portfolio_context(user_id)
            portfolio = self._aggregator.get_portfolio(user_id)
            tickers = [h.symbol for h in portfolio.holdings[:10]] if portfolio else []
        except Exception as e:
            logger.error("Failed to get portfolio: %s", e)
            response.errors.append(f"Portfolio data fetch failed: {e}")
            tickers = []

        # RAG context
        try:
            rag_result = self._rag.search(query, top_k=3)
            response.rag_context = rag_result.get("context")
            response.rag_results = rag_result.get("results", [])
            data_sources_used.append(MCPDataSource.RAG)
        except Exception as e:
            logger.error("RAG search failed: %s", e)
            response.errors.append(f"RAG search failed: {e}")

        # Live quotes for holdings
        if include_live_quotes and tickers:
            market_server = self._get_market_server()
            quotes = []
            for ticker in tickers[:5]:
                try:
                    quote_data = await market_server.get_stock_quote(symbol=ticker)
                    normalized = self._normalizer.normalize_quote(quote_data)
                    quote = StockQuote(
                        symbol=normalized.get("symbol", ticker),
                        price=normalized.get("price") or Decimal("0"),
                        change=normalized.get("change"),
                        change_percent=normalized.get("change_percent"),
                    )
                    quotes.append(quote)
                except Exception as e:
                    logger.warning("Failed to get quote for %s: %s", ticker, e)

            response.live_quotes = quotes
            if quotes:
                data_sources_used.append(MCPDataSource.YFINANCE)

        response.data_sources_used = data_sources_used
        return response

    # ------------------------------------------------------------------
    # Sync wrapper (safe to call from Streamlit / non-async code)
    # ------------------------------------------------------------------

    def get_enriched_context_sync(
        self,
        query: str,
        tickers: list[str] | None = None,
        include_news: bool = True,
        include_quotes: bool = True,
        top_k: int = 5,
        timeout: float = 30.0,
    ) -> HybridDataResponse:
        """Synchronous wrapper around get_enriched_context.

        Runs the async method in a dedicated thread so it is safe to call
        from Streamlit or any other synchronous context (no existing event loop
        in the thread means asyncio.run() works without conflict).

        Args:
            query: Natural language query.
            tickers: Ticker symbols for live data.
            include_news: Whether to include ticker news.
            include_quotes: Whether to include live quotes.
            top_k: Number of RAG results.
            timeout: Maximum seconds to wait.

        Returns:
            HybridDataResponse with combined data.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.get_enriched_context(query, tickers, include_news, include_quotes, top_k),
            )
            return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Convenience sync helpers (unchanged)
    # ------------------------------------------------------------------

    def get_portfolio_context_sync(self, user_id: str) -> str:
        """Get portfolio context synchronously."""
        return self._aggregator.format_portfolio_context(user_id)

    def search_knowledge_sync(self, query: str, top_k: int = 5) -> dict:
        """Search RAG knowledge base synchronously."""
        return self._rag.search(query, top_k=top_k)
