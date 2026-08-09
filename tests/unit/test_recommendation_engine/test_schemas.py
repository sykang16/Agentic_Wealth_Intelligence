"""Unit tests for recommendation engine schemas."""

from datetime import datetime

import pytest

from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    DataSource,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationRequest,
    RecommendationResponse,
    RiskLevel,
)


class TestRecommendationCategory:
    """Tests for RecommendationCategory enum."""

    def test_all_categories(self):
        """Test all category values."""
        assert RecommendationCategory.BUY == "buy"
        assert RecommendationCategory.SELL == "sell"
        assert RecommendationCategory.HOLD == "hold"
        assert RecommendationCategory.REBALANCE == "rebalance"
        assert RecommendationCategory.DIVERSIFY == "diversify"
        assert RecommendationCategory.RISK_REDUCTION == "risk_reduction"
        assert RecommendationCategory.TAX_OPTIMIZATION == "tax_optimization"
        assert RecommendationCategory.INCOME_GENERATION == "income_generation"


class TestDataSource:
    """Tests for DataSource model."""

    def test_create_data_source(self):
        """Test creating a data source."""
        ds = DataSource(
            source_type="rag",
            source_name="Knowledge Base",
            relevance_score=0.8,
        )
        assert ds.source_type == "rag"
        assert ds.relevance_score == 0.8
        assert ds.snippet is None

    def test_defaults(self):
        """Test default values."""
        ds = DataSource(source_type="portfolio", source_name="Portfolio")
        assert ds.relevance_score == 0.0
        assert ds.snippet is None


class TestRecommendation:
    """Tests for Recommendation model."""

    @pytest.fixture
    def sample_recommendation(self) -> Recommendation:
        """Create a sample recommendation for testing."""
        return Recommendation(
            id="rec_test001",
            category=RecommendationCategory.BUY,
            title="Increase Bond Allocation",
            summary="Add bonds to reduce portfolio risk.",
            detailed_rationale="Your portfolio is equity-heavy at 80%.",
            tickers=["BND", "AGG"],
            suggested_action="Buy BND to bring bond allocation to 20%.",
            risk_level=RiskLevel.LOW,
            relevance_score=0.8,
            risk_alignment_score=0.9,
            diversification_score=0.7,
            confidence=ConfidenceLevel.HIGH,
            priority=RecommendationPriority.HIGH,
        )

    def test_create_recommendation(self, sample_recommendation: Recommendation):
        """Test creating a recommendation."""
        assert sample_recommendation.id == "rec_test001"
        assert sample_recommendation.category == RecommendationCategory.BUY
        assert sample_recommendation.tickers == ["BND", "AGG"]

    def test_composite_score(self, sample_recommendation: Recommendation):
        """Test composite score computation."""
        # 0.8 * 0.4 + 0.9 * 0.35 + 0.7 * 0.25 = 0.32 + 0.315 + 0.175 = 0.81
        expected = 0.8 * 0.4 + 0.9 * 0.35 + 0.7 * 0.25
        assert abs(sample_recommendation.composite_score - expected) < 0.001

    def test_composite_score_all_zeros(self):
        """Test composite score when all scores are zero."""
        rec = Recommendation(
            id="rec_zero",
            category=RecommendationCategory.HOLD,
            title="Hold",
            summary="Hold.",
            detailed_rationale="Rationale.",
            suggested_action="Do nothing.",
        )
        assert rec.composite_score == 0.0

    def test_defaults(self):
        """Test default values."""
        rec = Recommendation(
            id="rec_defaults",
            category=RecommendationCategory.HOLD,
            title="Test",
            summary="Summary",
            detailed_rationale="Rationale",
            suggested_action="Action",
        )
        assert rec.risk_level == RiskLevel.MODERATE
        assert rec.confidence == ConfidenceLevel.MEDIUM
        assert rec.priority == RecommendationPriority.MEDIUM
        assert rec.tickers == []
        assert rec.data_sources == []
        assert rec.suggested_allocation_pct is None
        assert rec.expected_return_range is None
        assert rec.time_horizon is None


class TestRecommendationRequest:
    """Tests for RecommendationRequest model."""

    def test_minimal_request(self):
        """Test request with minimal fields."""
        req = RecommendationRequest(user_id="user_001")
        assert req.user_id == "user_001"
        assert req.query == ""
        assert req.max_recommendations == 5
        assert req.include_live_data is True
        assert req.categories is None

    def test_full_request(self):
        """Test request with all fields."""
        req = RecommendationRequest(
            user_id="user_002",
            query="How should I diversify?",
            max_recommendations=3,
            include_live_data=False,
            categories=[RecommendationCategory.DIVERSIFY, RecommendationCategory.BUY],
        )
        assert req.max_recommendations == 3
        assert not req.include_live_data
        assert len(req.categories) == 2


class TestAggregatedContext:
    """Tests for AggregatedContext model."""

    def test_empty_context(self):
        """Test empty context defaults."""
        ctx = AggregatedContext()
        assert ctx.portfolio_context == ""
        assert ctx.rag_context == ""
        assert ctx.user_risk_tolerance is None
        assert ctx.portfolio_tickers == []
        assert ctx.excluded_sectors == []

    def test_get_full_context(self):
        """Test combining context into a single string."""
        ctx = AggregatedContext(
            portfolio_context="## Portfolio\n- Net Worth: $1M",
            investment_profile_context="## Profile\n- Risk: moderate",
            rag_context="Bond ETFs provide stability.",
            live_market_context="- SPY: $450.00 (+0.5%)",
            sentiment_context="- SPY: Bullish (score: 0.8)",
        )
        full = ctx.get_full_context()
        assert "Portfolio" in full
        assert "Profile" in full
        assert "Relevant Knowledge" in full
        assert "Live Market Data" in full
        assert "Market Sentiment" in full

    def test_get_full_context_truncation(self):
        """Test context truncation at max_length."""
        ctx = AggregatedContext(
            portfolio_context="A" * 5000,
            rag_context="B" * 5000,
        )
        full = ctx.get_full_context(max_length=100)
        assert len(full) == 100

    def test_get_full_context_empty_parts_skipped(self):
        """Test that empty parts are skipped."""
        ctx = AggregatedContext(
            portfolio_context="Portfolio data here",
        )
        full = ctx.get_full_context()
        assert "Relevant Knowledge" not in full
        assert "Live Market Data" not in full


class TestRecommendationResponse:
    """Tests for RecommendationResponse model."""

    def test_empty_response(self):
        """Test response with no recommendations."""
        resp = RecommendationResponse(user_id="user_001", query="test")
        assert not resp.has_recommendations
        assert resp.success is True
        assert resp.error is None
        assert resp.recommendations == []

    def test_response_with_recommendations(self):
        """Test response with recommendations."""
        rec = Recommendation(
            id="rec_1",
            category=RecommendationCategory.BUY,
            title="Buy BND",
            summary="Summary",
            detailed_rationale="Rationale",
            suggested_action="Action",
        )
        resp = RecommendationResponse(
            user_id="user_001",
            query="diversify",
            recommendations=[rec],
            summary="1 recommendation generated.",
            data_sources_used=["portfolio", "rag_knowledge_base"],
            profile_completeness=0.75,
        )
        assert resp.has_recommendations
        assert len(resp.recommendations) == 1
        assert resp.profile_completeness == 0.75

    def test_error_response(self):
        """Test error response."""
        resp = RecommendationResponse(
            user_id="user_001",
            query="test",
            success=False,
            error="User not found",
        )
        assert not resp.success
        assert resp.error == "User not found"
        assert not resp.has_recommendations
