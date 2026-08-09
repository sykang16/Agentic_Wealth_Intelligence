"""Evaluation tests for RecommendationEngine quality.

Tests (all offline - no LLM calls):
  1. RISK_COMPATIBILITY table structure
  2. Profile alignment: risk filter respects tolerance
  3. Excluded sector filtering
  4. Ranking order by composite score
  5. Composite score formula (0.40/0.35/0.25 weights)
  6. Warning generation logic
  7. Summary building
  8. AggregatedContext schema and get_full_context
  9. RecommendationResponse schema completeness
"""

import pytest
from unittest.mock import MagicMock

from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationResponse,
    RiskLevel,
)
from backend.src.recommendation.engine.ranker import (
    RecommendationRanker,
    RISK_COMPATIBILITY,
)
from backend.src.recommendation.engine.engine import RecommendationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rec(
    id: str = "rec_001",
    category: RecommendationCategory = RecommendationCategory.BUY,
    title: str = "Test Recommendation",
    summary: str = "A test recommendation summary.",
    detailed_rationale: str = "Detailed rationale for testing purposes.",
    suggested_action: str = "Take this action.",
    risk_level: RiskLevel = RiskLevel.MODERATE,
    priority: RecommendationPriority = RecommendationPriority.MEDIUM,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    tickers: list | None = None,
    relevance_score: float = 0.5,
    risk_alignment_score: float = 0.5,
    diversification_score: float = 0.5,
) -> Recommendation:
    return Recommendation(
        id=id,
        category=category,
        title=title,
        summary=summary,
        detailed_rationale=detailed_rationale,
        suggested_action=suggested_action,
        risk_level=risk_level,
        priority=priority,
        confidence=confidence,
        tickers=tickers or [],
        relevance_score=relevance_score,
        risk_alignment_score=risk_alignment_score,
        diversification_score=diversification_score,
    )


def _make_context(
    user_risk_tolerance: str | None = None,
    excluded_sectors: list | None = None,
    current_sector_allocation: dict | None = None,
    portfolio_tickers: list | None = None,
    rag_context: str = "",
    data_sources_used: list | None = None,
) -> AggregatedContext:
    return AggregatedContext(
        user_risk_tolerance=user_risk_tolerance,
        excluded_sectors=excluded_sectors or [],
        current_sector_allocation=current_sector_allocation or {},
        portfolio_tickers=portfolio_tickers or [],
        rag_context=rag_context,
        data_sources_used=data_sources_used or [],
    )


def _make_engine() -> RecommendationEngine:
    """Build an engine with all external dependencies mocked out."""
    return RecommendationEngine(
        portfolio_aggregator=MagicMock(),
        rag_initializer=MagicMock(),
        llm_client=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. RISK_COMPATIBILITY table
# ---------------------------------------------------------------------------


class TestRiskCompatibilityTable:
    def test_conservative_only_low(self):
        assert RISK_COMPATIBILITY["conservative"] == {RiskLevel.LOW}

    def test_moderate_low_and_moderate(self):
        assert RISK_COMPATIBILITY["moderate"] == {RiskLevel.LOW, RiskLevel.MODERATE}

    def test_aggressive_all_levels(self):
        assert RISK_COMPATIBILITY["aggressive"] == {
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
        }

    def test_all_three_tolerances_defined(self):
        assert set(RISK_COMPATIBILITY.keys()) == {
            "conservative",
            "moderate",
            "aggressive",
        }


# ---------------------------------------------------------------------------
# 2. Profile alignment: _filter_by_risk
# ---------------------------------------------------------------------------


class TestRiskFiltering:
    def setup_method(self):
        self.ranker = RecommendationRanker()

    def test_conservative_filters_high_risk(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.LOW),
            _make_rec("r2", risk_level=RiskLevel.HIGH),
        ]
        ctx = _make_context(user_risk_tolerance="conservative")
        result = self.ranker.rank(recs, ctx)
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r2" not in ids

    def test_conservative_filters_moderate_risk(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.LOW),
            _make_rec("r2", risk_level=RiskLevel.MODERATE),
        ]
        ctx = _make_context(user_risk_tolerance="conservative")
        result = self.ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "r1"

    def test_moderate_allows_low_and_moderate(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.LOW),
            _make_rec("r2", risk_level=RiskLevel.MODERATE),
            _make_rec("r3", risk_level=RiskLevel.HIGH),
        ]
        ctx = _make_context(user_risk_tolerance="moderate")
        result = self.ranker.rank(recs, ctx)
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r2" in ids
        assert "r3" not in ids

    def test_aggressive_allows_all_levels(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.LOW),
            _make_rec("r2", risk_level=RiskLevel.MODERATE),
            _make_rec("r3", risk_level=RiskLevel.HIGH),
        ]
        ctx = _make_context(user_risk_tolerance="aggressive")
        # min_composite_score=0.0 to isolate risk-filter logic from score threshold.
        # A LOW-risk rec for an aggressive user scores risk_align=0.3 (distance 2),
        # which drops composite below 0.55. This test is about the risk filter only.
        result = self.ranker.rank(recs, ctx, min_composite_score=0.0)
        assert len(result) == 3

    def test_no_profile_keeps_all_recs(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.LOW),
            _make_rec("r2", risk_level=RiskLevel.HIGH),
        ]
        ctx = _make_context(user_risk_tolerance=None)
        result = self.ranker.rank(recs, ctx)
        assert len(result) == 2

    def test_all_high_risk_filtered_for_conservative(self):
        recs = [
            _make_rec("r1", risk_level=RiskLevel.HIGH),
            _make_rec("r2", risk_level=RiskLevel.HIGH),
        ]
        ctx = _make_context(user_risk_tolerance="conservative")
        result = self.ranker.rank(recs, ctx)
        assert result == []


# ---------------------------------------------------------------------------
# 3. Excluded sector filtering
# ---------------------------------------------------------------------------


class TestExcludedSectorFiltering:
    def setup_method(self):
        self.ranker = RecommendationRanker()

    def test_excluded_sector_in_title_filtered(self):
        recs = [
            _make_rec("r1", title="Invest in Tobacco Companies"),
            _make_rec("r2", title="Buy Technology ETF"),
        ]
        ctx = _make_context(excluded_sectors=["tobacco"])
        result = self.ranker.rank(recs, ctx)
        ids = [r.id for r in result]
        assert "r1" not in ids
        assert "r2" in ids

    def test_excluded_sector_in_summary_filtered(self):
        recs = [
            _make_rec("r1", summary="Exposure to gambling industry recommended."),
            _make_rec("r2", summary="Diversified bond fund for stability."),
        ]
        ctx = _make_context(excluded_sectors=["gambling"])
        result = self.ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "r2"

    def test_excluded_sector_in_rationale_filtered(self):
        recs = [
            _make_rec(
                "r1",
                detailed_rationale="Arms manufacturing sector shows strong returns.",
            ),
            _make_rec("r2", detailed_rationale="Healthcare sector fundamentals strong."),
        ]
        ctx = _make_context(excluded_sectors=["arms"])
        result = self.ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "r2"

    def test_no_excluded_sectors_keeps_all(self):
        recs = [_make_rec("r1"), _make_rec("r2")]
        ctx = _make_context(excluded_sectors=[])
        result = self.ranker.rank(recs, ctx)
        assert len(result) == 2

    def test_case_insensitive_sector_filter(self):
        recs = [_make_rec("r1", title="TOBACCO stocks to buy")]
        ctx = _make_context(excluded_sectors=["Tobacco"])
        result = self.ranker.rank(recs, ctx)
        assert result == []


# ---------------------------------------------------------------------------
# 4. Ranking order
# ---------------------------------------------------------------------------


class TestRankingOrder:
    def setup_method(self):
        self.ranker = RecommendationRanker()

    def test_high_priority_high_confidence_ranks_first(self):
        rec_high = _make_rec(
            "high_prio",
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        rec_low = _make_rec(
            "low_prio",
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = _make_context()
        result = self.ranker.rank([rec_high, rec_low], ctx)
        assert result[0].id == "high_prio"

    def test_medium_priority_ranks_between_high_and_low(self):
        rec_high = _make_rec(
            "high",
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        rec_med = _make_rec(
            "med",
            priority=RecommendationPriority.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
        )
        rec_low = _make_rec(
            "low",
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = _make_context()
        # min_composite_score=0.0 to isolate ranking order from score threshold.
        # A LOW/LOW rec scores relevance=0.50, composite≈0.50 < 0.55, which would
        # correctly suppress it in production. Here we verify ordering, not filtering.
        result = self.ranker.rank([rec_low, rec_high, rec_med], ctx, min_composite_score=0.0)
        assert result[0].id == "high"
        assert result[2].id == "low"

    def test_empty_input_returns_empty(self):
        ctx = _make_context()
        result = self.ranker.rank([], ctx)
        assert result == []

    def test_category_filter_applied(self):
        recs = [
            _make_rec("r1", category=RecommendationCategory.BUY),
            _make_rec("r2", category=RecommendationCategory.SELL),
            _make_rec("r3", category=RecommendationCategory.HOLD),
        ]
        ctx = _make_context()
        result = self.ranker.rank(recs, ctx, categories=[RecommendationCategory.BUY])
        assert len(result) == 1
        assert result[0].id == "r1"

    def test_category_filter_multiple_categories(self):
        recs = [
            _make_rec("r1", category=RecommendationCategory.BUY),
            _make_rec("r2", category=RecommendationCategory.SELL),
            _make_rec("r3", category=RecommendationCategory.HOLD),
        ]
        ctx = _make_context()
        result = self.ranker.rank(
            recs,
            ctx,
            categories=[RecommendationCategory.BUY, RecommendationCategory.HOLD],
        )
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r3" in ids
        assert "r2" not in ids


# ---------------------------------------------------------------------------
# 5. Composite score formula
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_perfect_scores_give_1_0(self):
        rec = _make_rec(
            relevance_score=1.0,
            risk_alignment_score=1.0,
            diversification_score=1.0,
        )
        assert rec.composite_score == pytest.approx(1.0)

    def test_zero_scores_give_0_0(self):
        rec = _make_rec(
            relevance_score=0.0,
            risk_alignment_score=0.0,
            diversification_score=0.0,
        )
        assert rec.composite_score == pytest.approx(0.0)

    def test_formula_weights_applied(self):
        # 0.40*0.8 + 0.35*0.6 + 0.25*0.4 = 0.32 + 0.21 + 0.10 = 0.63
        rec = _make_rec(
            relevance_score=0.8,
            risk_alignment_score=0.6,
            diversification_score=0.4,
        )
        expected = 0.40 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4
        assert rec.composite_score == pytest.approx(expected, abs=1e-9)

    def test_weights_sum_to_one(self):
        assert (0.40 + 0.35 + 0.25) == pytest.approx(1.0)

    def test_relevance_has_highest_weight(self):
        rec_relevance = _make_rec(
            relevance_score=1.0,
            risk_alignment_score=0.0,
            diversification_score=0.0,
        )
        rec_risk = _make_rec(
            relevance_score=0.0,
            risk_alignment_score=1.0,
            diversification_score=0.0,
        )
        rec_divers = _make_rec(
            relevance_score=0.0,
            risk_alignment_score=0.0,
            diversification_score=1.0,
        )
        assert (
            rec_relevance.composite_score
            > rec_risk.composite_score
            > rec_divers.composite_score
        )


# ---------------------------------------------------------------------------
# 6. Warning generation
# ---------------------------------------------------------------------------


class TestWarningGeneration:
    def setup_method(self):
        self.engine = _make_engine()

    def test_disclaimer_always_present(self):
        ctx = _make_context(rag_context="some content", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.9)
        assert any("financial advice" in w for w in warnings)

    def test_incomplete_profile_triggers_warning(self):
        ctx = _make_context(rag_context="some content", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.3)
        assert any("50%" in w and "Investment Profile" in w for w in warnings)

    def test_complete_profile_no_completeness_warning(self):
        ctx = _make_context(rag_context="some content", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.5)
        completeness_warnings = [w for w in warnings if "50%" in w]
        assert completeness_warnings == []

    def test_no_live_quotes_triggers_warning(self):
        ctx = _make_context(rag_context="some content", data_sources_used=["portfolio"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.8)
        assert any("Live market data" in w for w in warnings)

    def test_with_live_quotes_no_market_warning(self):
        ctx = _make_context(rag_context="some content", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.8)
        live_warnings = [w for w in warnings if "Live market data" in w]
        assert live_warnings == []

    def test_no_rag_context_triggers_warning(self):
        ctx = _make_context(rag_context="", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.8)
        assert any("knowledge base" in w for w in warnings)

    def test_with_rag_context_no_knowledge_warning(self):
        ctx = _make_context(
            rag_context="Relevant financial knowledge.",
            data_sources_used=["live_quotes"],
        )
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.8)
        rag_warnings = [w for w in warnings if "knowledge base" in w]
        assert rag_warnings == []

    def test_all_conditions_bad_four_warnings(self):
        # incomplete profile + no live_quotes + no rag + disclaimer = 4
        ctx = _make_context(rag_context="", data_sources_used=[])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.1)
        assert len(warnings) == 4

    def test_all_conditions_good_only_disclaimer(self):
        ctx = _make_context(
            rag_context="Some knowledge.",
            data_sources_used=["live_quotes"],
        )
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.9)
        assert len(warnings) == 1
        assert "financial advice" in warnings[0]

    def test_completeness_boundary_exactly_05_no_warning(self):
        # 0.5 is NOT < 0.5, so no completeness warning
        ctx = _make_context(rag_context="x", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.5)
        completeness_warnings = [w for w in warnings if "50%" in w]
        assert completeness_warnings == []

    def test_completeness_just_below_05_triggers_warning(self):
        ctx = _make_context(rag_context="x", data_sources_used=["live_quotes"])
        warnings = self.engine._build_warnings(ctx, profile_completeness=0.499)
        completeness_warnings = [w for w in warnings if "50%" in w]
        assert len(completeness_warnings) == 1


# ---------------------------------------------------------------------------
# 7. Summary building
# ---------------------------------------------------------------------------


class TestSummaryBuilding:
    def setup_method(self):
        self.engine = _make_engine()

    def test_empty_recs_returns_no_data_message(self):
        summary = self.engine._build_summary([])
        assert "No recommendations" in summary

    def test_count_reflected_in_summary(self):
        recs = [_make_rec("r1"), _make_rec("r2"), _make_rec("r3")]
        summary = self.engine._build_summary(recs)
        assert "3" in summary

    def test_single_rec_count(self):
        recs = [_make_rec("r1")]
        summary = self.engine._build_summary(recs)
        assert "1" in summary

    def test_categories_included_in_summary(self):
        recs = [
            _make_rec("r1", category=RecommendationCategory.BUY),
            _make_rec("r2", category=RecommendationCategory.SELL),
        ]
        summary = self.engine._build_summary(recs)
        assert "buy" in summary
        assert "sell" in summary

    def test_categories_sorted_alphabetically(self):
        recs = [
            _make_rec("r1", category=RecommendationCategory.SELL),
            _make_rec("r2", category=RecommendationCategory.BUY),
        ]
        summary = self.engine._build_summary(recs)
        buy_pos = summary.index("buy")
        sell_pos = summary.index("sell")
        assert buy_pos < sell_pos

    def test_high_priority_count_included(self):
        recs = [
            _make_rec("r1", priority=RecommendationPriority.HIGH),
            _make_rec("r2", priority=RecommendationPriority.MEDIUM),
            _make_rec("r3", priority=RecommendationPriority.HIGH),
        ]
        summary = self.engine._build_summary(recs)
        assert "2" in summary
        assert "high-priority" in summary

    def test_no_high_priority_no_mention(self):
        recs = [
            _make_rec("r1", priority=RecommendationPriority.MEDIUM),
            _make_rec("r2", priority=RecommendationPriority.LOW),
        ]
        summary = self.engine._build_summary(recs)
        assert "high-priority" not in summary

    def test_duplicate_categories_deduped(self):
        recs = [
            _make_rec("r1", category=RecommendationCategory.BUY),
            _make_rec("r2", category=RecommendationCategory.BUY),
            _make_rec("r3", category=RecommendationCategory.BUY),
        ]
        summary = self.engine._build_summary(recs)
        # "buy" appears exactly once in the category listing
        assert summary.count("buy") == 1


# ---------------------------------------------------------------------------
# 8. AggregatedContext schema
# ---------------------------------------------------------------------------


class TestAggregatedContextSchema:
    def test_default_risk_tolerance_is_none(self):
        ctx = AggregatedContext()
        assert ctx.user_risk_tolerance is None

    def test_default_excluded_sectors_empty(self):
        ctx = AggregatedContext()
        assert ctx.excluded_sectors == []

    def test_default_portfolio_tickers_empty(self):
        ctx = AggregatedContext()
        assert ctx.portfolio_tickers == []

    def test_default_rag_context_empty_string(self):
        ctx = AggregatedContext()
        assert ctx.rag_context == ""

    def test_default_data_sources_used_empty(self):
        ctx = AggregatedContext()
        assert ctx.data_sources_used == []

    def test_get_full_context_combines_parts(self):
        ctx = AggregatedContext(
            portfolio_context="Portfolio data",
            rag_context="RAG knowledge",
        )
        full = ctx.get_full_context()
        assert "Portfolio data" in full
        assert "RAG knowledge" in full

    def test_get_full_context_excludes_empty_parts(self):
        ctx = AggregatedContext(portfolio_context="Portfolio data")
        full = ctx.get_full_context()
        assert "Portfolio data" in full
        assert "## Relevant Knowledge" not in full

    def test_get_full_context_respects_max_length(self):
        long_text = "A" * 10000
        ctx = AggregatedContext(portfolio_context=long_text)
        full = ctx.get_full_context(max_length=100)
        assert len(full) <= 100


# ---------------------------------------------------------------------------
# 9. RecommendationResponse schema
# ---------------------------------------------------------------------------


class TestRecommendationResponseSchema:
    def test_has_recommendations_false_when_empty(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert resp.has_recommendations is False

    def test_has_recommendations_true_when_populated(self):
        resp = RecommendationResponse(
            user_id="u1",
            query="test",
            recommendations=[_make_rec()],
        )
        assert resp.has_recommendations is True

    def test_default_success_is_true(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert resp.success is True

    def test_default_error_is_none(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert resp.error is None

    def test_default_profile_completeness_is_zero(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert resp.profile_completeness == 0.0

    def test_failed_response_carries_error(self):
        resp = RecommendationResponse(
            user_id="u1",
            query="test",
            success=False,
            error="No portfolio data found for user u1",
        )
        assert resp.success is False
        assert resp.error is not None
        assert "u1" in resp.error

    def test_warnings_field_is_list(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert isinstance(resp.warnings, list)

    def test_data_sources_used_is_list(self):
        resp = RecommendationResponse(user_id="u1", query="test")
        assert isinstance(resp.data_sources_used, list)
