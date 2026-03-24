"""Pydantic models for MCP data schemas."""

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class MCPDataSource(str, Enum):
    """Available data sources for MCP integration."""

    ALPHA_VANTAGE = "alpha_vantage"
    YFINANCE = "yfinance"
    SEC_EDGAR = "sec_edgar"
    NEWS_API = "news_api"
    PORTFOLIO = "portfolio"
    RAG = "rag"


class MCPToolStatus(str, Enum):
    """Status of an MCP tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    RATE_LIMITED = "rate_limited"
    NOT_CONFIGURED = "not_configured"


class MCPToolResult(BaseModel):
    """Standard result wrapper for MCP tool responses."""

    status: MCPToolStatus
    data: dict | list | None = None
    message: str | None = None
    source: MCPDataSource | None = None
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.status in (MCPToolStatus.SUCCESS, MCPToolStatus.PARTIAL)


class StockQuote(BaseModel):
    """Real-time stock quote data."""

    symbol: str
    price: Decimal
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    volume: int | None = None
    previous_close: Decimal | None = None
    change: Decimal | None = None
    change_percent: float | None = None
    latest_trading_day: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def is_up(self) -> bool:
        """Check if the stock is up from previous close."""
        if self.change is not None:
            return self.change > 0
        return False


class MarketSentiment(BaseModel):
    """Sentiment analysis result for a ticker or topic."""

    ticker: str | None = None
    topic: str | None = None
    sentiment_label: str  # Bullish, Bearish, Neutral
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float | None = None
    article_count: int = 0
    sources: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment_score > 0.25

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment_score < -0.25


class NewsArticle(BaseModel):
    """Individual news article data."""

    title: str
    summary: str | None = None
    source: str
    url: str | None = None
    published_at: datetime | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    tickers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class NewsFeed(BaseModel):
    """Collection of news articles."""

    articles: list[NewsArticle]
    total_count: int
    query: str | None = None
    topics: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def average_sentiment(self) -> float:
        """Calculate average sentiment score across articles."""
        scores = [a.sentiment_score for a in self.articles if a.sentiment_score is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


class ETFProfile(BaseModel):
    """ETF profile and metrics."""

    symbol: str
    name: str
    description: str | None = None
    asset_type: str = "ETF"
    exchange: str | None = None
    sector: str | None = None
    expense_ratio: float | None = None
    dividend_yield: float | None = None
    pe_ratio: float | None = None
    market_cap: Decimal | None = None
    week_52_high: Decimal | None = None
    week_52_low: Decimal | None = None
    holdings_count: int | None = None
    top_holdings: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class MarketOverview(BaseModel):
    """Market overview with top movers."""

    top_gainers: list[dict] = Field(default_factory=list)
    top_losers: list[dict] = Field(default_factory=list)
    most_active: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class PortfolioHolding(BaseModel):
    """Portfolio holding with current market data."""

    symbol: str
    name: str
    quantity: Decimal
    average_cost: Decimal
    current_price: Decimal | None = None
    market_value: Decimal | None = None
    unrealized_gain_loss: Decimal | None = None
    unrealized_gain_loss_percent: float | None = None
    sector: str | None = None
    asset_type: str | None = None


class PortfolioAnalysisResult(BaseModel):
    """Result of portfolio analysis."""

    user_id: str
    net_worth: Decimal | None = None
    total_assets: Decimal | None = None
    total_liabilities: Decimal | None = None
    liquidity_ratio: float | None = None
    debt_to_asset_ratio: float | None = None
    allocation_by_asset_type: dict[str, float] = Field(default_factory=dict)
    allocation_by_sector: dict[str, float] = Field(default_factory=dict)
    top_holdings: list[PortfolioHolding] = Field(default_factory=list)
    risk_score: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class RAGSearchResult(BaseModel):
    """Result from RAG knowledge search."""

    query: str
    results: list[dict] = Field(default_factory=list)
    num_results: int = 0
    context: str | None = None
    sources: list[str] = Field(default_factory=list)


class HybridDataResponse(BaseModel):
    """Combined response from RAG and MCP live data sources."""

    query: str
    rag_context: str | None = None
    rag_results: list[dict] = Field(default_factory=list)
    live_quotes: list[StockQuote] = Field(default_factory=list)
    live_news: list[NewsArticle] = Field(default_factory=list)
    sentiment: list[MarketSentiment] = Field(default_factory=list)
    portfolio_context: str | None = None
    data_sources_used: list[MCPDataSource] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_combined_context(self, max_length: int = 4000) -> str:
        """Get combined context string for LLM consumption.

        Args:
            max_length: Maximum length of the combined context.

        Returns:
            Combined context string.
        """
        parts = []

        # Add RAG context
        if self.rag_context:
            parts.append("## Knowledge Base Context")
            parts.append(self.rag_context[:max_length // 3])

        # Add live quotes
        if self.live_quotes:
            parts.append("\n## Live Market Data")
            for quote in self.live_quotes[:5]:
                change_str = ""
                if quote.change_percent is not None:
                    sign = "+" if quote.change_percent >= 0 else ""
                    change_str = f" ({sign}{quote.change_percent:.2f}%)"
                parts.append(f"- {quote.symbol}: ${quote.price:.2f}{change_str}")

        # Add sentiment summary
        if self.sentiment:
            parts.append("\n## Market Sentiment")
            for s in self.sentiment[:3]:
                target = s.ticker or s.topic or "Market"
                parts.append(f"- {target}: {s.sentiment_label} (score: {s.sentiment_score:.2f})")

        # Add portfolio context
        if self.portfolio_context:
            parts.append("\n## Portfolio Context")
            parts.append(self.portfolio_context[:max_length // 3])

        combined = "\n".join(parts)
        return combined[:max_length]
