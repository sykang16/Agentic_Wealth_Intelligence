"""Unit tests for context builder."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from backend.src.common.models import (
    ExperienceLevel,
    InvestmentHorizon,
    InvestmentProfile,
    PortfolioSummary,
    RiskTolerance,
    User,
    UserPortfolio,
)
from backend.src.recommendation.engine.context_builder import ContextBuilder


def _make_portfolio(
    user_id: str = "user_001",
    risk_tolerance: RiskTolerance | None = RiskTolerance.MODERATE,
    investment_horizon: InvestmentHorizon | None = InvestmentHorizon.LONG,
    excluded_sectors: list[str] | None = None,
) -> UserPortfolio:
    """Create a mock portfolio for testing."""
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
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        investment_experience=ExperienceLevel.INTERMEDIATE,
        excluded_sectors=excluded_sectors or [],
    )
    summary = PortfolioSummary(
        user_id=user_id,
        total_assets=Decimal("500000"),
        total_liabilities=Decimal("100000"),
        total_net_worth=Decimal("400000"),
        allocation_by_sector={"Technology": 40, "Healthcare": 20, "Finance": 15},
        allocation_by_asset_type={"stock": 60, "etf": 25, "bond": 15},
    )
    return UserPortfolio(
        user=user,
        investment_profile=profile,
        summary=summary,
    )


@pytest.fixture
def mock_aggregator() -> MagicMock:
    """Create a mock portfolio aggregator."""
    mock = MagicMock()
    portfolio = _make_portfolio()
    mock.get_portfolio.return_value = portfolio
    mock.format_portfolio_context.return_value = "## Portfolio\n- Net Worth: $400,000"
    return mock


@pytest.fixture
def mock_rag() -> MagicMock:
    """Create a mock RAG initializer."""
    mock = MagicMock()
    mock.search.return_value = {
        "query": "test",
        "num_results": 1,
        "results": [{"content": "Bond ETFs provide stability."}],
        "context": "Bond ETFs provide stability in a diversified portfolio.",
    }
    return mock


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_context_adds_portfolio(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that portfolio context is added."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert "Portfolio" in ctx.portfolio_context
        assert "portfolio" in ctx.data_sources_used

    def test_build_context_adds_profile(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that investment profile is extracted."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.user_risk_tolerance == "moderate"
        assert ctx.user_investment_horizon == "long"
        assert ctx.user_experience_level == "intermediate"
        assert "investment_profile" in ctx.data_sources_used

    def test_build_context_adds_rag(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that RAG context is added."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context(
            "user_001", query="bond allocation", include_live_data=False
        )
        assert "Bond ETFs" in ctx.rag_context
        assert "rag_knowledge_base" in ctx.data_sources_used
        mock_rag.search.assert_called_once_with("bond allocation", top_k=5)

    def test_build_context_no_profile(
        self, mock_rag: MagicMock
    ):
        """Test fallback when no investment profile exists."""
        mock_agg = MagicMock()
        portfolio = _make_portfolio()
        portfolio.investment_profile = None
        mock_agg.get_portfolio.return_value = portfolio
        mock_agg.format_portfolio_context.return_value = "Portfolio data"

        builder = ContextBuilder(
            portfolio_aggregator=mock_agg,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.user_risk_tolerance is None
        assert "No investment profile available" in ctx.investment_profile_context
        assert "investment_profile" not in ctx.data_sources_used

    def test_build_context_rag_failure(self, mock_aggregator: MagicMock):
        """Test graceful handling when RAG search fails."""
        mock_rag = MagicMock()
        mock_rag.search.side_effect = Exception("ChromaDB error")

        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.rag_context == ""
        assert "rag_knowledge_base" not in ctx.data_sources_used

    def test_build_context_portfolio_failure(self, mock_rag: MagicMock):
        """Test graceful handling when portfolio load fails."""
        mock_agg = MagicMock()
        mock_agg.format_portfolio_context.side_effect = RuntimeError("Not loaded")
        mock_agg.get_portfolio.side_effect = RuntimeError("Not loaded")

        builder = ContextBuilder(
            portfolio_aggregator=mock_agg,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.portfolio_context == ""
        assert ctx.portfolio_tickers == []

    def test_build_context_extracts_allocations(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that sector and asset allocations are extracted."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.current_sector_allocation == {
            "Technology": 40,
            "Healthcare": 20,
            "Finance": 15,
        }
        assert ctx.current_asset_allocation == {
            "stock": 60,
            "etf": 25,
            "bond": 15,
        }

    def test_build_context_excluded_sectors(self, mock_rag: MagicMock):
        """Test that excluded sectors are extracted from profile."""
        mock_agg = MagicMock()
        portfolio = _make_portfolio(excluded_sectors=["Gambling", "Tobacco"])
        mock_agg.get_portfolio.return_value = portfolio
        mock_agg.format_portfolio_context.return_value = "Portfolio data"

        builder = ContextBuilder(
            portfolio_aggregator=mock_agg,
            rag_initializer=mock_rag,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        assert ctx.excluded_sectors == ["Gambling", "Tobacco"]

    def test_build_rag_query_from_profile(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that RAG query is built from profile when no query provided."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        # No query provided -> should build from profile
        ctx = builder.build_context("user_001", query="", include_live_data=False)
        # Verify RAG was called with auto-generated query
        call_args = mock_rag.search.call_args[0][0]
        assert "moderate" in call_args
        assert "long" in call_args

    def test_no_live_data_when_disabled(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test that live data is skipped when disabled."""
        mock_hybrid = MagicMock()
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
            hybrid_data_provider=mock_hybrid,
        )
        ctx = builder.build_context("user_001", include_live_data=False)
        mock_hybrid.get_enriched_context.assert_not_called()
        assert ctx.live_market_context == ""


class TestFormatProfile:
    """Tests for profile formatting."""

    def test_format_profile_all_fields(
        self, mock_aggregator: MagicMock, mock_rag: MagicMock
    ):
        """Test formatting a complete profile."""
        builder = ContextBuilder(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=mock_rag,
        )
        profile = InvestmentProfile(
            user_id="user_001",
            risk_tolerance=RiskTolerance.MODERATE,
            investment_horizon=InvestmentHorizon.LONG,
            investment_experience=ExperienceLevel.ADVANCED,
            preferred_sectors=["Technology", "Healthcare"],
            excluded_sectors=["Energy"],
            esg_preference=True,
        )
        result = builder._format_profile(profile)
        assert "Risk Tolerance: moderate" in result
        assert "Investment Horizon: long" in result
        assert "Experience Level: advanced" in result
        assert "Technology" in result
        assert "Energy" in result
        assert "ESG Preference: Yes" in result
