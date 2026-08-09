"""Unit tests for recommendation generator."""

import json
from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.recommendation.engine.generator import RecommendationGenerator
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    RecommendationCategory,
    RiskLevel,
)


SAMPLE_LLM_JSON = json.dumps([
    {
        "category": "buy",
        "title": "Buy Bond ETF",
        "summary": "Add bonds for stability.",
        "detailed_rationale": "Your portfolio is equity-heavy.",
        "tickers": ["BND", "AGG"],
        "suggested_action": "Buy BND for 20% allocation.",
        "suggested_allocation_pct": 20.0,
        "risk_level": "low",
        "expected_return_range": "3-5% annually",
        "time_horizon": "12-24 months",
        "confidence": "high",
        "priority": "high",
    },
    {
        "category": "diversify",
        "title": "Add International Exposure",
        "summary": "Diversify with international ETFs.",
        "detailed_rationale": "Currently 100% domestic.",
        "tickers": ["VXUS"],
        "suggested_action": "Buy VXUS for 15% allocation.",
        "suggested_allocation_pct": 15.0,
        "risk_level": "moderate",
        "expected_return_range": "5-8% annually",
        "time_horizon": "3-5 years",
        "confidence": "medium",
        "priority": "medium",
    },
])


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.chat.return_value = LLMResponse(
        content=SAMPLE_LLM_JSON,
        model="test-model",
        provider="test",
    )
    return mock


@pytest.fixture
def sample_context() -> AggregatedContext:
    """Create a sample context."""
    return AggregatedContext(
        portfolio_context="## Portfolio\n- Net Worth: $500,000",
        investment_profile_context="## Profile\n- Risk: moderate",
        data_sources_used=["portfolio", "rag_knowledge_base"],
    )


class TestRecommendationGenerator:
    """Tests for RecommendationGenerator."""

    def test_generate_returns_recommendations(
        self, mock_llm: MagicMock, sample_context: AggregatedContext
    ):
        """Test that generate returns parsed recommendations."""
        gen = RecommendationGenerator(llm_client=mock_llm)
        recs = gen.generate(sample_context, query="How should I diversify?")
        assert len(recs) == 2
        assert recs[0].category == RecommendationCategory.BUY
        assert recs[0].title == "Buy Bond ETF"
        assert recs[0].tickers == ["BND", "AGG"]
        assert recs[1].category == RecommendationCategory.DIVERSIFY

    def test_generate_respects_max_recommendations(
        self, mock_llm: MagicMock, sample_context: AggregatedContext
    ):
        """Test that max_recommendations limits output."""
        gen = RecommendationGenerator(llm_client=mock_llm)
        recs = gen.generate(sample_context, max_recommendations=1)
        assert len(recs) == 1

    def test_generate_uses_default_query(
        self, mock_llm: MagicMock, sample_context: AggregatedContext
    ):
        """Test that empty query uses default."""
        gen = RecommendationGenerator(llm_client=mock_llm)
        gen.generate(sample_context, query="")
        # Verify LLM was called
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        assert "investment actions" in call_args.kwargs["user_message"]

    def test_generate_handles_llm_error(self, sample_context: AggregatedContext):
        """Test graceful handling of LLM errors."""
        mock = MagicMock()
        mock.chat.side_effect = Exception("API error")
        gen = RecommendationGenerator(llm_client=mock)
        recs = gen.generate(sample_context)
        assert recs == []


class TestParseRecommendations:
    """Tests for JSON parsing of LLM output."""

    @pytest.fixture
    def generator(self) -> RecommendationGenerator:
        return RecommendationGenerator(llm_client=MagicMock())

    @pytest.fixture
    def context(self) -> AggregatedContext:
        return AggregatedContext(data_sources_used=["portfolio"])

    def test_parse_valid_json(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test parsing valid JSON array."""
        recs = generator._parse_recommendations(SAMPLE_LLM_JSON, context)
        assert len(recs) == 2

    def test_parse_json_with_code_fences(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test parsing JSON wrapped in markdown code fences."""
        wrapped = f"```json\n{SAMPLE_LLM_JSON}\n```"
        recs = generator._parse_recommendations(wrapped, context)
        assert len(recs) == 2

    def test_parse_single_object(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test parsing a single JSON object (not array)."""
        single = json.dumps({
            "category": "hold",
            "title": "Hold Current",
            "summary": "Keep positions.",
            "detailed_rationale": "Market is stable.",
            "tickers": [],
            "suggested_action": "No action needed.",
            "risk_level": "low",
            "confidence": "medium",
            "priority": "low",
        })
        recs = generator._parse_recommendations(single, context)
        assert len(recs) == 1
        assert recs[0].category == RecommendationCategory.HOLD

    def test_parse_malformed_json(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test graceful handling of malformed JSON."""
        recs = generator._parse_recommendations("not valid json", context)
        assert recs == []

    def test_parse_with_missing_fields(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test parsing with missing optional fields uses defaults."""
        minimal = json.dumps([{
            "category": "buy",
            "title": "Minimal Rec",
            "summary": "Minimal.",
            "detailed_rationale": "Rationale.",
            "suggested_action": "Buy something.",
        }])
        recs = generator._parse_recommendations(minimal, context)
        assert len(recs) == 1
        assert recs[0].risk_level == RiskLevel.MODERATE
        assert recs[0].tickers == []

    def test_parse_skips_invalid_items(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test that invalid items are skipped, valid ones kept."""
        mixed = json.dumps([
            {
                "category": "INVALID_CATEGORY",
                "title": "Bad",
                "summary": "Bad",
                "detailed_rationale": "Bad",
                "suggested_action": "Bad",
            },
            {
                "category": "buy",
                "title": "Good",
                "summary": "Good",
                "detailed_rationale": "Good",
                "suggested_action": "Good",
            },
        ])
        recs = generator._parse_recommendations(mixed, context)
        assert len(recs) == 1
        assert recs[0].title == "Good"

    def test_data_sources_attached(
        self, generator: RecommendationGenerator, context: AggregatedContext
    ):
        """Test that data sources from context are attached to recommendations."""
        recs = generator._parse_recommendations(SAMPLE_LLM_JSON, context)
        assert len(recs[0].data_sources) == 1
        assert recs[0].data_sources[0].source_type == "portfolio"


class TestEnhanceExplanation:
    """Tests for explanation enhancement."""

    def test_enhance_returns_explanation(self):
        """Test that enhance_explanation returns LLM response."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="This recommendation makes sense because...",
            model="test",
            provider="test",
        )
        gen = RecommendationGenerator(llm_client=mock_llm)

        from backend.src.recommendation.engine.schemas import Recommendation, RecommendationCategory

        rec = Recommendation(
            id="rec_1",
            category=RecommendationCategory.BUY,
            title="Buy BND",
            summary="Summary",
            detailed_rationale="Original rationale",
            suggested_action="Buy BND",
        )
        ctx = AggregatedContext(
            portfolio_context="Portfolio data",
            rag_context="RAG data",
        )
        result = gen.enhance_explanation(rec, ctx, experience_level="beginner")
        assert result == "This recommendation makes sense because..."

    def test_enhance_falls_back_on_error(self):
        """Test fallback to original rationale on error."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API error")
        gen = RecommendationGenerator(llm_client=mock_llm)

        from backend.src.recommendation.engine.schemas import Recommendation, RecommendationCategory

        rec = Recommendation(
            id="rec_1",
            category=RecommendationCategory.BUY,
            title="Buy BND",
            summary="Summary",
            detailed_rationale="Original rationale",
            suggested_action="Buy BND",
        )
        ctx = AggregatedContext()
        result = gen.enhance_explanation(rec, ctx)
        assert result == "Original rationale"
