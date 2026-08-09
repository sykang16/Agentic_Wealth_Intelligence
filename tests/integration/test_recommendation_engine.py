"""Integration tests for the recommendation engine pipeline."""

import json
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.common.models import (
    Holding,
    InvestmentProfile,
    InvestmentHorizon,
    PortfolioSummary,
    RiskTolerance,
    User,
    UserPortfolio,
    AssetType,
)
from backend.src.recommendation.engine.engine import RecommendationEngine
from backend.src.recommendation.engine.schemas import (
    RecommendationRequest,
    RecommendationCategory,
)
from backend.src.recommendation.rag.initializer import RAGInitializer


SAMPLE_LLM_OUTPUT = json.dumps([
    {
        "category": "diversify",
        "title": "Add International Exposure",
        "summary": "Your portfolio is 100% domestic. Add international ETFs.",
        "detailed_rationale": (
            "International diversification reduces country-specific risk. "
            "VXUS provides broad international coverage with low fees."
        ),
        "tickers": ["VXUS", "IXUS"],
        "suggested_action": "Allocate 15% of portfolio to VXUS.",
        "suggested_allocation_pct": 15.0,
        "risk_level": "moderate",
        "expected_return_range": "5-8% annually",
        "time_horizon": "3-5 years",
        "confidence": "high",
        "priority": "high",
    },
    {
        "category": "buy",
        "title": "Increase Bond Allocation",
        "summary": "Bonds are underweight for moderate risk tolerance.",
        "detailed_rationale": (
            "With moderate risk tolerance and long horizon, "
            "a 20-30% bond allocation provides stability."
        ),
        "tickers": ["BND", "AGG"],
        "suggested_action": "Buy BND to increase bond allocation to 25%.",
        "suggested_allocation_pct": 25.0,
        "risk_level": "low",
        "expected_return_range": "3-5% annually",
        "time_horizon": "1-3 years",
        "confidence": "high",
        "priority": "medium",
    },
    {
        "category": "rebalance",
        "title": "Reduce Technology Concentration",
        "summary": "Technology sector is overweight at 45%.",
        "detailed_rationale": (
            "Sector concentration above 30% increases risk. "
            "Rebalance by trimming tech and adding healthcare/consumer."
        ),
        "tickers": ["XLV", "XLP"],
        "suggested_action": "Sell 10% of tech holdings and buy XLV/XLP.",
        "risk_level": "low",
        "confidence": "medium",
        "priority": "medium",
    },
])


def _create_test_portfolio() -> UserPortfolio:
    """Create a realistic test portfolio."""
    user = User(
        user_id="test_user_001",
        name="Jane Doe",
        email="jane@example.com",
        age=35,
        occupation="Software Engineer",
        annual_income=Decimal("150000"),
        monthly_expenses=Decimal("6000"),
    )
    profile = InvestmentProfile(
        user_id="test_user_001",
        risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon=InvestmentHorizon.LONG,
        monthly_income=Decimal("12500"),
        monthly_expenses=Decimal("6000"),
    )
    holdings = [
        Holding(
            holding_id="h1",
            account_id="acc1",
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=Decimal("50"),
            average_cost=Decimal("150"),
            current_price=Decimal("180"),
            sector="Technology",
        ),
        Holding(
            holding_id="h2",
            account_id="acc1",
            asset_type=AssetType.ETF,
            symbol="SPY",
            name="SPDR S&P 500",
            quantity=Decimal("100"),
            average_cost=Decimal("400"),
            current_price=Decimal("450"),
            sector="Broad Market",
        ),
        Holding(
            holding_id="h3",
            account_id="acc1",
            asset_type=AssetType.BOND,
            symbol="BND",
            name="Vanguard Total Bond",
            quantity=Decimal("50"),
            average_cost=Decimal("80"),
            current_price=Decimal("75"),
            sector="Fixed Income",
        ),
    ]
    summary = PortfolioSummary(
        user_id="test_user_001",
        total_assets=Decimal("600000"),
        total_liabilities=Decimal("200000"),
        total_net_worth=Decimal("400000"),
        allocation_by_sector={
            "Technology": 45,
            "Broad Market": 35,
            "Fixed Income": 10,
            "Other": 10,
        },
        allocation_by_asset_type={
            "stock": 40,
            "etf": 35,
            "bond": 10,
            "cash": 15,
        },
        liquidity_ratio=0.15,
        debt_to_asset_ratio=0.33,
    )
    return UserPortfolio(
        user=user,
        holdings=holdings,
        investment_profile=profile,
        summary=summary,
    )


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM that returns realistic recommendations."""
    mock = MagicMock()
    mock.chat.return_value = LLMResponse(
        content=SAMPLE_LLM_OUTPUT,
        model="test-model",
        provider="test",
    )
    return mock


@pytest.fixture
def mock_aggregator() -> MagicMock:
    """Create a mock aggregator with realistic portfolio."""
    portfolio = _create_test_portfolio()
    mock = MagicMock()
    mock.get_portfolio.return_value = portfolio
    mock.format_portfolio_context.return_value = (
        "## User Profile\n"
        "- Name: Jane Doe\n"
        "- Age: 35\n"
        "- Annual Income: $150,000\n\n"
        "## Portfolio Summary\n"
        "- Net Worth: $400,000\n"
        "- Total Assets: $600,000\n\n"
        "## Asset Allocation\n"
        "- Stock: 40%\n"
        "- ETF: 35%\n"
        "- Bond: 10%\n"
        "- Cash: 15%\n\n"
        "## Top Holdings\n"
        "- AAPL (Apple Inc.): $9,000 (+20.0%)\n"
        "- SPY (SPDR S&P 500): $45,000 (+12.5%)\n"
        "- BND (Vanguard Total Bond): $3,750 (-6.2%)\n"
    )
    return mock


class TestRecommendationEngineIntegration:
    """Integration tests for the full recommendation pipeline."""

    def test_full_pipeline_with_mocked_llm(
        self,
        mock_aggregator: MagicMock,
        mock_llm: MagicMock,
        temp_chroma_dir: str,
    ):
        """Test the full pipeline: context -> generate -> rank -> response."""
        rag = RAGInitializer(persist_directory=temp_chroma_dir)

        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=rag,
            llm_client=mock_llm,
        )

        request = RecommendationRequest(
            user_id="test_user_001",
            query="How should I diversify my portfolio?",
            max_recommendations=5,
            include_live_data=False,
        )

        response = engine.generate_recommendations(request)

        # Verify success
        assert response.success
        assert response.has_recommendations
        assert response.user_id == "test_user_001"
        assert response.processing_time_ms > 0

        # Verify recommendations are scored
        for rec in response.recommendations:
            assert rec.composite_score >= 0.0
            assert rec.risk_alignment_score >= 0.0
            assert rec.diversification_score >= 0.0
            assert rec.relevance_score >= 0.0
            assert rec.id.startswith("rec_")

        # Verify ranked (descending composite score)
        scores = [r.composite_score for r in response.recommendations]
        assert scores == sorted(scores, reverse=True)

        # Verify data sources tracked
        assert "portfolio" in response.data_sources_used

        # Verify warnings include disclaimer
        assert any("financial advice" in w.lower() for w in response.warnings)

    def test_pipeline_respects_risk_tolerance(
        self,
        mock_aggregator: MagicMock,
        mock_llm: MagicMock,
        temp_chroma_dir: str,
    ):
        """Test that risk filtering works end-to-end."""
        # Set user to conservative
        portfolio = mock_aggregator.get_portfolio.return_value
        portfolio.investment_profile.risk_tolerance = RiskTolerance.CONSERVATIVE

        rag = RAGInitializer(persist_directory=temp_chroma_dir)
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=rag,
            llm_client=mock_llm,
        )

        request = RecommendationRequest(
            user_id="test_user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        # Conservative should only have low-risk recommendations
        for rec in response.recommendations:
            assert rec.risk_level.value == "low"

    def test_pipeline_with_category_filter(
        self,
        mock_aggregator: MagicMock,
        mock_llm: MagicMock,
        temp_chroma_dir: str,
    ):
        """Test category filtering end-to-end."""
        rag = RAGInitializer(persist_directory=temp_chroma_dir)
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=rag,
            llm_client=mock_llm,
        )

        request = RecommendationRequest(
            user_id="test_user_001",
            categories=[RecommendationCategory.DIVERSIFY],
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        for rec in response.recommendations:
            assert rec.category == RecommendationCategory.DIVERSIFY

    def test_pipeline_with_empty_llm_response(
        self,
        mock_aggregator: MagicMock,
        temp_chroma_dir: str,
    ):
        """Test pipeline when LLM returns empty recommendations."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="[]",
            model="test",
            provider="test",
        )

        rag = RAGInitializer(persist_directory=temp_chroma_dir)
        engine = RecommendationEngine(
            portfolio_aggregator=mock_aggregator,
            rag_initializer=rag,
            llm_client=mock_llm,
        )

        request = RecommendationRequest(
            user_id="test_user_001",
            include_live_data=False,
        )
        response = engine.generate_recommendations(request)

        assert response.success
        assert not response.has_recommendations
        assert "No recommendations" in response.summary
