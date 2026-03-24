"""Pydantic schemas for the recommendation engine."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class RecommendationCategory(str, Enum):
    """Categories of investment recommendations."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"
    DIVERSIFY = "diversify"
    RISK_REDUCTION = "risk_reduction"
    TAX_OPTIMIZATION = "tax_optimization"
    INCOME_GENERATION = "income_generation"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(str, Enum):
    """Confidence levels for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    """Risk levels for recommendations."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DataSource(BaseModel):
    """A data source that contributed to a recommendation."""

    source_type: str = Field(
        ...,
        description="Type: 'rag', 'mcp_quote', 'mcp_news', 'mcp_sentiment', 'portfolio'",
    )
    source_name: str = Field(..., description="Human-readable name of the source")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    snippet: str | None = Field(default=None, description="Brief excerpt from this source")


class Recommendation(BaseModel):
    """A single investment recommendation with explanation."""

    id: str = Field(..., description="Unique recommendation ID")
    category: RecommendationCategory
    title: str = Field(..., description="Short title, e.g. 'Increase Bond Allocation'")
    summary: str = Field(..., description="1-2 sentence summary")
    detailed_rationale: str = Field(..., description="Full explanation with reasoning")

    # Actionable details
    tickers: list[str] = Field(default_factory=list, description="Related ticker symbols")
    suggested_action: str = Field(..., description="Concrete action to take")
    suggested_allocation_pct: float | None = Field(
        default=None, description="Suggested portfolio allocation %"
    )

    # Risk and return
    risk_level: RiskLevel = Field(default=RiskLevel.MODERATE)
    expected_return_range: str | None = Field(
        default=None, description="e.g. '5-8% annually'"
    )
    time_horizon: str | None = Field(default=None, description="e.g. '6-12 months'")

    # Scoring
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How relevant to user query"
    )
    risk_alignment_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How well aligned with user risk profile"
    )
    diversification_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How much it improves diversification"
    )

    # Metadata
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    priority: RecommendationPriority = Field(default=RecommendationPriority.MEDIUM)
    data_sources: list[DataSource] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def composite_score(self) -> float:
        """Weighted composite score for ranking."""
        return (
            self.relevance_score * 0.4
            + self.risk_alignment_score * 0.35
            + self.diversification_score * 0.25
        )


class RecommendationRequest(BaseModel):
    """Request for generating recommendations."""

    user_id: str = Field(..., description="User ID")
    query: str = Field(
        default="", description="Optional user query for targeted recommendations"
    )
    max_recommendations: int = Field(default=5, ge=1, le=10)
    include_live_data: bool = Field(
        default=True, description="Whether to fetch live market data"
    )
    categories: list[RecommendationCategory] | None = Field(
        default=None, description="Filter to specific categories"
    )
    min_composite_score: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum composite score (0-1) a recommendation must reach to be returned. "
            "In financial systems, no recommendation is better than a bad one. "
            "Default 0.55 blocks recs with simultaneously low relevance, adjacent-only "
            "risk alignment, and poor diversification benefit. Set to 0.0 to disable."
        ),
    )


class AggregatedContext(BaseModel):
    """Aggregated context from all data sources for LLM consumption."""

    portfolio_context: str = Field(default="", description="Formatted portfolio data")
    investment_profile_context: str = Field(
        default="", description="Formatted investment profile"
    )
    rag_context: str = Field(default="", description="RAG knowledge base results")
    live_market_context: str = Field(default="", description="Live quotes and news")
    sentiment_context: str = Field(default="", description="Market sentiment data")

    # Raw data for ranking
    user_risk_tolerance: str | None = None
    user_investment_horizon: str | None = None
    user_experience_level: str | None = None
    current_sector_allocation: dict[str, float] = Field(default_factory=dict)
    current_asset_allocation: dict[str, float] = Field(default_factory=dict)
    portfolio_tickers: list[str] = Field(default_factory=list)
    excluded_sectors: list[str] = Field(default_factory=list)
    data_sources_used: list[str] = Field(default_factory=list)

    def get_full_context(self, max_length: int = 8000) -> str:
        """Combine all context into a single string for LLM."""
        parts = []
        if self.portfolio_context:
            parts.append(self.portfolio_context)
        if self.investment_profile_context:
            parts.append(self.investment_profile_context)
        if self.rag_context:
            parts.append(f"## Relevant Knowledge\n{self.rag_context}")
        if self.live_market_context:
            parts.append(f"## Live Market Data\n{self.live_market_context}")
        if self.sentiment_context:
            parts.append(f"## Market Sentiment\n{self.sentiment_context}")
        combined = "\n\n".join(parts)
        return combined[:max_length]


class RecommendationResponse(BaseModel):
    """Response from the recommendation engine."""

    user_id: str
    query: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    summary: str = Field(default="", description="Overall summary of recommendations")
    data_sources_used: list[str] = Field(default_factory=list)
    profile_completeness: float = Field(
        default=0.0, description="0-1, how complete the user profile is"
    )
    warnings: list[str] = Field(default_factory=list, description="Caveats or warnings")
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(default=0.0)
    success: bool = True
    error: str | None = None

    @computed_field
    @property
    def has_recommendations(self) -> bool:
        """Whether any recommendations were generated."""
        return len(self.recommendations) > 0
