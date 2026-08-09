"""Unit tests for the main recommendation engine."""

import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.common.models import (
    InvestmentProfile,
    PortfolioSummary,
    RiskTolerance,
    User,
    UserPortfolio,
)
from backend.src.recommendation.engine.engine import RecommendationEngine
from backend.src.recommendation.engine.schemas import (
    RecommendationCategory,
    RecommendationRequest,
)


SAMPLE_LLM_OUTPUT = json.dumps([
    {
        "category": "buy",
        "title": "Buy Bond ETF",
        "summary": "Add bonds to reduce risk.",
        "detailed_rationale": "Portfolio is equity-heavy.",
        "tickers": ["BND"],
        "suggested_action": "Buy BND.",
        "risk_level": "low",
        "confidence": "high",
        "priority": "high",
    },
    {
        "category": "diversify",
        "title": "International Exposure",
        "summary": "Add international stocks.",
        "detailed_rationale": "100% domestic is risky.",
        "tickers": ["VXUS"],
        "suggested_action": "Buy VXUS.",
        "risk_level": "moderate",
        "confidence": "medium",
        "priority": "medium",
    },
    {
        "category": "rebalance",
        "title": "Reduce Tech",
        "summary": "Tech is overweight.",
        "detailed_rationale": "40% in tech is concentrated.",
        "tickers": [],
        "suggested_action": "Sell some tech holdings.",
        "risk_level": "low",
        "confidence": "high",
        "priority": "high",
    },
])


def _make_portfolio(user_id: str = "user_001") -> UserPortfolio:
    """Create a test portfolio."""
    user = User(
        user_id=user_id,
        name="Test User",
        email="test@test.com",
        age=35,
        occupation="Engineer",
        annual_income=Decimal("120000"),
        monthly_expenses=Decimal("5000"),
    )
    profile = InvestmentProfile(
        user_id=user_id,
        risk_tolerance=RiskTolerance.MODERATE,
    )
    summary = PortfolioSummary(
        user_id=user_id,
        total_assets=Decimal("500000"),
        total_liabilities=Decimal("100000"),
        total_net_worth=Decimal("400000"),
        allocation_by_sector={"Technology": 40, "Healthcare": 20},
        allocation_by_asset_type={"stock": 60, "etf": 25, "bond": 15},
    )
    return UserPortfolio(
        user=user,
        investment_profile=profile,
        summary=summary,
    )


@pytest.fixture
def mock_aggregator() -> MagicMock:
    """Mock portfolio aggregator."""
    mock = MagicMock()
    mock.get_portfolio.return_value = _make_portfolio()
    mock.format_portfolio_context.return_value = "## Portfolio\n- Net Worth: $400k"
    return mock


@pytest.fixture
def mock_rag() -> MagicMock:
    """Mock RAG initializer."""
    mock = MagicMock()
    mock.search.return_value = {
        "query": "test",
        "num_results": 1,
        "results": [{"content": "Bond ETFs are good for stability."}],
        "context": "Bond ETFs are good for stability.",
    }
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM client."""
    mock = MagicMock()
    mock.chat.return_value = LLMResponse(
        content=SAMPLE_LLM_OUTPUT,
        model="test-model",
        provider="test",
    )
    return mock


class TestRecommendationEngine:
    """Tests for the main recommendation engine."""

    def test_generate_recommendations_success(
        self,
        mock_aggregator: MagicMock,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ):
        """Test successful recommendation generation."""
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            query="How should I diversify?",
            max_recommendations=5,
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        assert response.has_recommendations
        assert len(response.recommendations) <= 5
        assert response.user_id == "user_001"
        assert response.processing_time_ms >= 0
        assert len(response.data_sources_used) > 0
        assert response.summary != ""

    def test_generate_recommendations_user_not_found(
        self,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ):
        """Test error when user is not found."""
        mock_agg = MagicMock()
        mock_agg.get_portfolio.return_value = None

        engine = RecommendationEngine(
            portfolio_aggregator=mock_agg,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(user_id="nonexistent")
        response = engine.generate_recommendations(request)

        assert not response.success
        assert "No portfolio data found" in response.error

    def test_generate_recommendations_with_category_filter(
        self,
        mock_aggregator: MagicMock,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ):
        """Test filtering by category."""
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            categories=[RecommendationCategory.BUY],
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        for rec in response.recommendations:
            assert rec.category == RecommendationCategory.BUY

    def test_warnings_for_incomplete_profile(
        self,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ):
        """Test warnings when profile is incomplete."""
        mock_agg = MagicMock()
        portfolio = _make_portfolio()
        # Profile with only risk_tolerance (low completeness)
        portfolio.investment_profile = InvestmentProfile(
            user_id="user_001",
            risk_tolerance=RiskTolerance.MODERATE,
        )
        mock_agg.get_portfolio.return_value = portfolio
        mock_agg.format_portfolio_context.return_value = "Portfolio data"

        engine = RecommendationEngine(
            portfolio_aggregator=mock_agg,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        # Should have profile warning since completeness < 0.5
        profile_warnings = [
            w for w in response.warnings if "profile" in w.lower()
        ]
        assert len(profile_warnings) >= 1

    def test_disclaimer_always_present(
        self,
        mock_aggregator: MagicMock,
        mock_rag: MagicMock,
        mock_llm: MagicMock,
    ):
        """Test that disclaimer warning is always included."""
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        disclaimer_warnings = [
            w for w in response.warnings if "financial advice" in w.lower()
        ]
        assert len(disclaimer_warnings) == 1

    def test_llm_failure_returns_error(
        self,
        mock_aggregator: MagicMock,
        mock_rag: MagicMock,
    ):
        """Test that LLM failure is handled gracefully."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API timeout")

        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        # Should succeed but with no recommendations
        assert response.success
        assert not response.has_recommendations

    def test_build_summary_no_recommendations(
        self,
        mock_aggregator: MagicMock,
        mock_rag: MagicMock,
    ):
        """Test summary when no recommendations are generated."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="[]",
            model="test",
            provider="test",
        )
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            llm_client=mock_llm,
        )
        request = RecommendationRequest(
            user_id="user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)
        assert "No recommendations" in response.summary
