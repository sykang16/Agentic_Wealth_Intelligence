"""Unit tests for recommendation ranker."""

import pytest

from backend.src.recommendation.engine.ranker import RecommendationRanker
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)


def _make_rec(
    id: str = "rec_1",
    category: RecommendationCategory = RecommendationCategory.BUY,
    title: str = "Test",
    risk_level: RiskLevel = RiskLevel.MODERATE,
    priority: RecommendationPriority = RecommendationPriority.MEDIUM,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    tickers: list[str] | None = None,
) -> Recommendation:
    """Helper to create a recommendation for testing."""
    return Recommendation(
        id=id,
        category=category,
        title=title,
        summary=f"Summary for {title}",
        detailed_rationale=f"Rationale for {title}",
        suggested_action=f"Action for {title}",
        risk_level=risk_level,
        priority=priority,
        confidence=confidence,
        tickers=tickers or [],
    )


class TestRiskFiltering:
    """Tests for risk-based filtering."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_conservative_filters_high_risk(self, ranker: RecommendationRanker):
        """Conservative user should only see low-risk recommendations."""
        recs = [
            _make_rec(id="low", risk_level=RiskLevel.LOW),
            _make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            _make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = AggregatedContext(user_risk_tolerance="conservative")
        result = ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "low"

    def test_moderate_filters_high_risk(self, ranker: RecommendationRanker):
        """Moderate user should see low and moderate risk."""
        recs = [
            _make_rec(id="low", risk_level=RiskLevel.LOW),
            _make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            _make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = AggregatedContext(user_risk_tolerance="moderate")
        result = ranker.rank(recs, ctx)
        assert len(result) == 2
        result_ids = {r.id for r in result}
        assert "low" in result_ids
        assert "mod" in result_ids

    def test_aggressive_keeps_all(self, ranker: RecommendationRanker):
        """Aggressive user should see all risk levels."""
        recs = [
            _make_rec(id="low", risk_level=RiskLevel.LOW),
            _make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            _make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = AggregatedContext(user_risk_tolerance="aggressive")
        result = ranker.rank(recs, ctx)
        assert len(result) == 3

    def test_no_profile_keeps_all(self, ranker: RecommendationRanker):
        """No risk tolerance set should keep all recommendations."""
        recs = [
            _make_rec(id="low", risk_level=RiskLevel.LOW),
            _make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = AggregatedContext()  # No risk tolerance
        result = ranker.rank(recs, ctx)
        assert len(result) == 2


class TestExcludedSectorFiltering:
    """Tests for excluded sector filtering."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_excludes_matching_sectors(self, ranker: RecommendationRanker):
        """Recommendations mentioning excluded sectors should be removed."""
        recs = [
            _make_rec(id="tech", title="Buy Tech ETF"),
            _make_rec(id="bonds", title="Buy Bond Fund"),
        ]
        # The detailed_rationale contains "Buy Tech ETF" in title
        ctx = AggregatedContext(excluded_sectors=["tech"])
        result = ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "bonds"

    def test_no_exclusions_keeps_all(self, ranker: RecommendationRanker):
        """No excluded sectors should keep all recommendations."""
        recs = [
            _make_rec(id="1"),
            _make_rec(id="2"),
        ]
        ctx = AggregatedContext()
        result = ranker.rank(recs, ctx)
        assert len(result) == 2


class TestCategoryFiltering:
    """Tests for category filtering."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_filter_by_category(self, ranker: RecommendationRanker):
        """Only matching categories should pass through."""
        recs = [
            _make_rec(id="buy", category=RecommendationCategory.BUY),
            _make_rec(id="sell", category=RecommendationCategory.SELL),
            _make_rec(id="div", category=RecommendationCategory.DIVERSIFY),
        ]
        ctx = AggregatedContext()
        result = ranker.rank(recs, ctx, categories=[RecommendationCategory.BUY])
        assert len(result) == 1
        assert result[0].id == "buy"


class TestRiskAlignmentScoring:
    """Tests for risk alignment score computation."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_perfect_match_conservative(self, ranker: RecommendationRanker):
        """Conservative user + low risk = 1.0 score."""
        rec = _make_rec(risk_level=RiskLevel.LOW)
        ctx = AggregatedContext(user_risk_tolerance="conservative")
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == 1.0

    def test_perfect_match_aggressive(self, ranker: RecommendationRanker):
        """Aggressive user + high risk = 1.0 score."""
        rec = _make_rec(risk_level=RiskLevel.HIGH)
        ctx = AggregatedContext(user_risk_tolerance="aggressive")
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == 1.0

    def test_adjacent_match(self, ranker: RecommendationRanker):
        """Moderate user + low risk = 0.7 score."""
        rec = _make_rec(risk_level=RiskLevel.LOW)
        ctx = AggregatedContext(user_risk_tolerance="moderate")
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == 0.7

    def test_mismatch(self, ranker: RecommendationRanker):
        """Conservative user + high risk = 0.3 score."""
        rec = _make_rec(risk_level=RiskLevel.HIGH)
        ctx = AggregatedContext(user_risk_tolerance="conservative")
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == 0.3

    def test_no_profile(self, ranker: RecommendationRanker):
        """No risk tolerance = 0.5 neutral score."""
        rec = _make_rec(risk_level=RiskLevel.MODERATE)
        ctx = AggregatedContext()
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == 0.5


class TestDiversificationScoring:
    """Tests for diversification score computation."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_diversify_category_boost(self, ranker: RecommendationRanker):
        """Diversify category should get +0.2 boost."""
        rec = _make_rec(category=RecommendationCategory.DIVERSIFY)
        ctx = AggregatedContext(current_sector_allocation={"Technology": 50})
        score = ranker._score_diversification(rec, ctx)
        assert score >= 0.7  # 0.5 base + 0.2 category boost

    def test_new_tickers_boost(self, ranker: RecommendationRanker):
        """New tickers not in portfolio should get boost."""
        rec = _make_rec(tickers=["XYZ"])
        ctx = AggregatedContext(
            current_sector_allocation={"Technology": 20},
            portfolio_tickers=["AAPL", "MSFT"],
        )
        score = ranker._score_diversification(rec, ctx)
        assert score >= 0.6  # 0.5 base + 0.1 new ticker

    def test_no_allocation_data(self, ranker: RecommendationRanker):
        """No allocation data should return 0.5 neutral."""
        rec = _make_rec()
        ctx = AggregatedContext()
        score = ranker._score_diversification(rec, ctx)
        assert score == 0.5


class TestRelevanceScoring:
    """Tests for relevance score computation."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_high_priority_high_confidence(self, ranker: RecommendationRanker):
        """High priority + high confidence = maximum relevance."""
        rec = _make_rec(
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        score = ranker._score_relevance(rec)
        assert score == 1.0

    def test_low_priority_low_confidence(self, ranker: RecommendationRanker):
        """Low priority + low confidence = base score."""
        rec = _make_rec(
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        score = ranker._score_relevance(rec)
        assert score == 0.5


class TestRanking:
    """Tests for overall ranking behavior."""

    @pytest.fixture
    def ranker(self) -> RecommendationRanker:
        return RecommendationRanker()

    def test_empty_list(self, ranker: RecommendationRanker):
        """Empty list should return empty."""
        result = ranker.rank([], AggregatedContext())
        assert result == []

    def test_sorted_by_composite_score(self, ranker: RecommendationRanker):
        """Results should be sorted by composite score descending."""
        recs = [
            _make_rec(
                id="low",
                priority=RecommendationPriority.LOW,
                confidence=ConfidenceLevel.LOW,
            ),
            _make_rec(
                id="high",
                priority=RecommendationPriority.HIGH,
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        ctx = AggregatedContext()
        result = ranker.rank(recs, ctx)
        assert result[0].id == "high"
        assert result[-1].id == "low"
