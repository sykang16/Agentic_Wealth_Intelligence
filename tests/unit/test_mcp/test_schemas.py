"""Unit tests for MCP schemas."""

from datetime import datetime
from decimal import Decimal

import pytest

from backend.src.mcp.schemas import (
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


class TestMCPToolResult:
    """Tests for MCPToolResult schema."""

    def test_create_success_result(self):
        """Test creating a success result."""
        result = MCPToolResult(
            status=MCPToolStatus.SUCCESS,
            data={"price": 150.25},
            message="Quote retrieved successfully",
            source=MCPDataSource.ALPHA_VANTAGE,
        )

        assert result.status == MCPToolStatus.SUCCESS
        assert result.is_success is True
        assert result.data == {"price": 150.25}
        assert result.source == MCPDataSource.ALPHA_VANTAGE

    def test_create_error_result(self):
        """Test creating an error result."""
        result = MCPToolResult(
            status=MCPToolStatus.ERROR,
            message="API rate limit exceeded",
        )

        assert result.status == MCPToolStatus.ERROR
        assert result.is_success is False
        assert result.message == "API rate limit exceeded"

    def test_partial_status_is_success(self):
        """Test that PARTIAL status counts as success."""
        result = MCPToolResult(
            status=MCPToolStatus.PARTIAL,
            data={"partial": "data"},
        )

        assert result.is_success is True

    def test_cached_flag(self):
        """Test cached flag."""
        result = MCPToolResult(
            status=MCPToolStatus.SUCCESS,
            data={},
            cached=True,
        )

        assert result.cached is True


class TestStockQuote:
    """Tests for StockQuote schema."""

    def test_create_quote(self):
        """Test creating a stock quote."""
        quote = StockQuote(
            symbol="AAPL",
            price=Decimal("175.50"),
            open=Decimal("174.00"),
            high=Decimal("176.00"),
            low=Decimal("173.50"),
            volume=50000000,
            previous_close=Decimal("174.25"),
            change=Decimal("1.25"),
            change_percent=0.72,
        )

        assert quote.symbol == "AAPL"
        assert quote.price == Decimal("175.50")
        assert quote.is_up is True

    def test_quote_is_down(self):
        """Test quote is_up property when stock is down."""
        quote = StockQuote(
            symbol="AAPL",
            price=Decimal("170.00"),
            change=Decimal("-5.00"),
        )

        assert quote.is_up is False

    def test_quote_with_minimal_data(self):
        """Test quote with only required fields."""
        quote = StockQuote(
            symbol="SPY",
            price=Decimal("450.00"),
        )

        assert quote.symbol == "SPY"
        assert quote.price == Decimal("450.00")
        assert quote.volume is None


class TestMarketSentiment:
    """Tests for MarketSentiment schema."""

    def test_create_bullish_sentiment(self):
        """Test creating bullish sentiment."""
        sentiment = MarketSentiment(
            ticker="AAPL",
            sentiment_label="Bullish",
            sentiment_score=0.65,
            article_count=10,
            sources=["Reuters", "Bloomberg"],
        )

        assert sentiment.is_bullish is True
        assert sentiment.is_bearish is False

    def test_create_bearish_sentiment(self):
        """Test creating bearish sentiment."""
        sentiment = MarketSentiment(
            ticker="META",
            sentiment_label="Bearish",
            sentiment_score=-0.45,
            article_count=5,
        )

        assert sentiment.is_bearish is True
        assert sentiment.is_bullish is False

    def test_neutral_sentiment(self):
        """Test neutral sentiment."""
        sentiment = MarketSentiment(
            topic="technology",
            sentiment_label="Neutral",
            sentiment_score=0.0,
        )

        assert sentiment.is_bullish is False
        assert sentiment.is_bearish is False


class TestNewsFeed:
    """Tests for NewsFeed schema."""

    def test_create_news_feed(self):
        """Test creating a news feed."""
        articles = [
            NewsArticle(
                title="Market Rally Continues",
                source="Reuters",
                sentiment_score=0.5,
            ),
            NewsArticle(
                title="Tech Stocks Rise",
                source="Bloomberg",
                sentiment_score=0.3,
            ),
        ]

        feed = NewsFeed(
            articles=articles,
            total_count=2,
            topics=["financial_markets"],
        )

        assert len(feed.articles) == 2
        assert feed.total_count == 2
        assert feed.average_sentiment == 0.4

    def test_empty_feed_average_sentiment(self):
        """Test average sentiment with no scored articles."""
        feed = NewsFeed(
            articles=[
                NewsArticle(title="Test", source="Test"),
            ],
            total_count=1,
        )

        assert feed.average_sentiment == 0.0


class TestPortfolioHolding:
    """Tests for PortfolioHolding schema."""

    def test_create_holding(self):
        """Test creating a portfolio holding."""
        holding = PortfolioHolding(
            symbol="VTI",
            name="Vanguard Total Stock Market",
            quantity=Decimal("100"),
            average_cost=Decimal("200.00"),
            current_price=Decimal("220.00"),
            market_value=Decimal("22000.00"),
            unrealized_gain_loss=Decimal("2000.00"),
            unrealized_gain_loss_percent=10.0,
            sector="Diversified",
            asset_type="ETF",
        )

        assert holding.symbol == "VTI"
        assert holding.market_value == Decimal("22000.00")


class TestPortfolioAnalysisResult:
    """Tests for PortfolioAnalysisResult schema."""

    def test_create_analysis_result(self):
        """Test creating a portfolio analysis result."""
        result = PortfolioAnalysisResult(
            user_id="user-001",
            net_worth=Decimal("500000.00"),
            total_assets=Decimal("550000.00"),
            total_liabilities=Decimal("50000.00"),
            liquidity_ratio=0.15,
            debt_to_asset_ratio=0.09,
            allocation_by_asset_type={"stocks": 60.0, "bonds": 30.0, "cash": 10.0},
            risk_score=45.0,
            risk_factors=["High concentration in tech sector"],
        )

        assert result.user_id == "user-001"
        assert result.net_worth == Decimal("500000.00")
        assert result.risk_score == 45.0


class TestHybridDataResponse:
    """Tests for HybridDataResponse schema."""

    def test_create_hybrid_response(self):
        """Test creating a hybrid data response."""
        response = HybridDataResponse(
            query="What is my portfolio performance?",
            rag_context="Your portfolio consists of...",
            live_quotes=[
                StockQuote(symbol="AAPL", price=Decimal("175.00"), change_percent=1.5),
            ],
            data_sources_used=[MCPDataSource.RAG, MCPDataSource.ALPHA_VANTAGE],
        )

        assert response.query == "What is my portfolio performance?"
        assert len(response.live_quotes) == 1
        assert MCPDataSource.RAG in response.data_sources_used

    def test_get_combined_context(self):
        """Test getting combined context."""
        response = HybridDataResponse(
            query="Market analysis",
            rag_context="The market has been volatile...",
            live_quotes=[
                StockQuote(symbol="SPY", price=Decimal("450.00"), change_percent=0.5),
            ],
            sentiment=[
                MarketSentiment(
                    ticker="SPY",
                    sentiment_label="Bullish",
                    sentiment_score=0.4,
                ),
            ],
            portfolio_context="Net worth: $500,000",
        )

        context = response.get_combined_context()

        assert "Knowledge Base Context" in context
        assert "Live Market Data" in context
        assert "SPY: $450.00" in context
        assert "Market Sentiment" in context
        assert "Portfolio Context" in context


class TestRAGSearchResult:
    """Tests for RAGSearchResult schema."""

    def test_create_search_result(self):
        """Test creating a RAG search result."""
        result = RAGSearchResult(
            query="ETF expense ratios",
            results=[
                {"content": "VOO has an expense ratio of 0.03%", "score": 0.95},
            ],
            num_results=1,
            context="VOO has an expense ratio of 0.03%...",
            sources=["Alpha Vantage"],
        )

        assert result.query == "ETF expense ratios"
        assert result.num_results == 1
        assert len(result.sources) == 1


class TestETFProfile:
    """Tests for ETFProfile schema."""

    def test_create_etf_profile(self):
        """Test creating an ETF profile."""
        profile = ETFProfile(
            symbol="VOO",
            name="Vanguard S&P 500 ETF",
            description="Tracks the S&P 500 index",
            asset_type="ETF",
            exchange="NYSE",
            expense_ratio=0.03,
            dividend_yield=1.5,
            market_cap=Decimal("350000000000"),
        )

        assert profile.symbol == "VOO"
        assert profile.expense_ratio == 0.03


class TestMarketOverview:
    """Tests for MarketOverview schema."""

    def test_create_market_overview(self):
        """Test creating a market overview."""
        overview = MarketOverview(
            top_gainers=[
                {"symbol": "AAPL", "change_percent": 5.0},
            ],
            top_losers=[
                {"symbol": "META", "change_percent": -3.0},
            ],
            most_active=[
                {"symbol": "NVDA", "volume": 100000000},
            ],
        )

        assert len(overview.top_gainers) == 1
        assert len(overview.top_losers) == 1
        assert len(overview.most_active) == 1
