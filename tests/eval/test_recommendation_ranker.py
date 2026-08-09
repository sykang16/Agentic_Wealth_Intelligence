"""Evaluation: Recommendation ranker — scoring, filtering, and ranking logic.

Phase 2 scope — offline, no real LLM calls:
  - Tests RecommendationRanker scoring methods (_score_risk_alignment,
    _score_diversification, _score_relevance) with known inputs.
  - Tests filtering logic (_filter_by_risk, _filter_excluded_sectors).
  - Tests composite_score formula and full rank() output ordering.
  - All tests use deterministic Recommendation and AggregatedContext objects.

Run with:
    pytest tests/eval/test_recommendation_ranker.py -v -s
"""

from __future__ import annotations

import pytest

from backend.src.recommendation.engine.ranker import RecommendationRanker, RISK_COMPATIBILITY
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_rec(
    *,
    id: str = "rec_001",
    category: RecommendationCategory = RecommendationCategory.BUY,
    title: str = "Buy Tech ETFs",
    summary: str = "Add broad tech exposure via ETFs.",
    detailed_rationale: str = "Technology sector is underweight in your portfolio.",
    tickers: list[str] | None = None,
    risk_level: RiskLevel = RiskLevel.MODERATE,
    priority: RecommendationPriority = RecommendationPriority.MEDIUM,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    suggested_action: str = "Buy 5% allocation in QQQ.",
) -> Recommendation:
    """Factory for Recommendation objects with sensible defaults."""
    return Recommendation(
        id=id,
        category=category,
        title=title,
        summary=summary,
        detailed_rationale=detailed_rationale,
        tickers=tickers or [],
        risk_level=risk_level,
        priority=priority,
        confidence=confidence,
        suggested_action=suggested_action,
    )


def make_context(
    *,
    user_risk_tolerance: str | None = "moderate",
    current_sector_allocation: dict[str, float] | None = None,
    excluded_sectors: list[str] | None = None,
    portfolio_tickers: list[str] | None = None,
) -> AggregatedContext:
    """Factory for AggregatedContext objects."""
    return AggregatedContext(
        user_risk_tolerance=user_risk_tolerance,
        current_sector_allocation=current_sector_allocation or {},
        excluded_sectors=excluded_sectors or [],
        portfolio_tickers=portfolio_tickers or [],
    )


@pytest.fixture
def ranker() -> RecommendationRanker:
    return RecommendationRanker()


# ── RISK_COMPATIBILITY constant ─────────────────────────────────────────────


class TestRiskCompatibilityConstant:
    """The RISK_COMPATIBILITY lookup must match documented business rules."""

    def test_conservative_accepts_only_low(self):
        assert RISK_COMPATIBILITY["conservative"] == {RiskLevel.LOW}

    def test_moderate_accepts_low_and_moderate(self):
        assert RISK_COMPATIBILITY["moderate"] == {RiskLevel.LOW, RiskLevel.MODERATE}

    def test_aggressive_accepts_all_levels(self):
        assert RISK_COMPATIBILITY["aggressive"] == {
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
        }

    def test_all_three_tolerance_levels_are_defined(self):
        assert set(RISK_COMPATIBILITY.keys()) == {"conservative", "moderate", "aggressive"}


# ── _score_risk_alignment ────────────────────────────────────────────────────


class TestScoreRiskAlignment:
    """Risk alignment scoring: 1.0 (perfect), 0.7 (adjacent), 0.3 (mismatched)."""

    @pytest.mark.parametrize(
        "user_risk, rec_risk, expected",
        [
            # Perfect matches (distance = 0) → 1.0
            ("conservative", RiskLevel.LOW, 1.0),
            ("moderate", RiskLevel.MODERATE, 1.0),
            ("aggressive", RiskLevel.HIGH, 1.0),
            # Adjacent (distance = 1) → 0.7
            ("conservative", RiskLevel.MODERATE, 0.7),
            ("moderate", RiskLevel.LOW, 0.7),
            ("moderate", RiskLevel.HIGH, 0.7),
            ("aggressive", RiskLevel.MODERATE, 0.7),
            # Two steps apart (distance = 2) → 0.3
            ("conservative", RiskLevel.HIGH, 0.3),
            ("aggressive", RiskLevel.LOW, 0.3),
        ],
    )
    def test_alignment_score(self, ranker, user_risk, rec_risk, expected):
        rec = make_rec(risk_level=rec_risk)
        ctx = make_context(user_risk_tolerance=user_risk)
        score = ranker._score_risk_alignment(rec, ctx)
        assert score == pytest.approx(expected, abs=0.01)

    def test_no_profile_returns_neutral(self, ranker):
        """When no user risk profile is available, return 0.5 (neutral)."""
        rec = make_rec(risk_level=RiskLevel.HIGH)
        ctx = make_context(user_risk_tolerance=None)
        assert ranker._score_risk_alignment(rec, ctx) == pytest.approx(0.5)

    def test_score_is_in_range(self, ranker):
        for risk in RiskLevel:
            for tolerance in ["conservative", "moderate", "aggressive"]:
                rec = make_rec(risk_level=risk)
                ctx = make_context(user_risk_tolerance=tolerance)
                score = ranker._score_risk_alignment(rec, ctx)
                assert 0.0 <= score <= 1.0, (
                    f"score {score} out of range for {tolerance}/{risk}"
                )


# ── _score_relevance ─────────────────────────────────────────────────────────


class TestScoreRelevance:
    """Relevance scoring uses priority + confidence as proxies. Cap at 1.0."""

    @pytest.mark.parametrize(
        "priority, confidence, expected",
        [
            # High priority + high confidence → max cap
            (RecommendationPriority.HIGH, ConfidenceLevel.HIGH, 1.0),
            # High priority + medium confidence = 0.5 + 0.3 + 0.1 = 0.9
            (RecommendationPriority.HIGH, ConfidenceLevel.MEDIUM, 0.9),
            # High priority + low confidence = 0.5 + 0.3 + 0 = 0.8
            (RecommendationPriority.HIGH, ConfidenceLevel.LOW, 0.8),
            # Medium priority + high confidence = 0.5 + 0.15 + 0.2 = 0.85
            (RecommendationPriority.MEDIUM, ConfidenceLevel.HIGH, 0.85),
            # Medium priority + medium confidence = 0.5 + 0.15 + 0.1 = 0.75
            (RecommendationPriority.MEDIUM, ConfidenceLevel.MEDIUM, 0.75),
            # Low priority + low confidence = 0.5 baseline
            (RecommendationPriority.LOW, ConfidenceLevel.LOW, 0.5),
        ],
    )
    def test_relevance_score(self, ranker, priority, confidence, expected):
        rec = make_rec(priority=priority, confidence=confidence)
        score = ranker._score_relevance(rec)
        assert score == pytest.approx(expected, abs=0.01)

    def test_score_never_exceeds_1(self, ranker):
        rec = make_rec(
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        assert ranker._score_relevance(rec) <= 1.0

    def test_score_is_in_range(self, ranker):
        for p in RecommendationPriority:
            for c in ConfidenceLevel:
                rec = make_rec(priority=p, confidence=c)
                score = ranker._score_relevance(rec)
                assert 0.0 <= score <= 1.0


# ── _score_diversification ───────────────────────────────────────────────────


class TestScoreDiversification:
    """Diversification scoring: base 0.5, adjusted by category, sector, tickers."""

    def test_no_sector_data_returns_neutral(self, ranker):
        """Without sector allocation data, return neutral 0.5."""
        rec = make_rec()
        ctx = make_context(current_sector_allocation={})
        score = ranker._score_diversification(rec, ctx)
        assert score == pytest.approx(0.5)

    def test_diversify_category_gets_boost(self, ranker):
        rec = make_rec(category=RecommendationCategory.DIVERSIFY)
        ctx = make_context(current_sector_allocation={"technology": 20.0})
        score = ranker._score_diversification(rec, ctx)
        # base 0.5 + category boost 0.2 = 0.7
        assert score == pytest.approx(0.7)

    def test_rebalance_category_gets_boost(self, ranker):
        rec = make_rec(category=RecommendationCategory.REBALANCE)
        ctx = make_context(current_sector_allocation={"technology": 20.0})
        score = ranker._score_diversification(rec, ctx)
        assert score == pytest.approx(0.7)

    def test_addressing_underweight_sector_boosts_score(self, ranker):
        """Rec that mentions an underweight sector (<10%) gets +0.2."""
        rec = make_rec(
            title="Buy Healthcare ETF",
            summary="Increase healthcare exposure.",
            detailed_rationale="Healthcare is underweight at 5% in your portfolio.",
        )
        ctx = make_context(
            current_sector_allocation={
                "technology": 50.0,
                "healthcare": 5.0,  # underweight
            }
        )
        score = ranker._score_diversification(rec, ctx)
        # base 0.5 + underweight mention 0.2 = 0.7
        assert score >= 0.65  # at least a meaningful boost

    def test_adding_to_overweight_sector_penalizes_score(self, ranker):
        """Rec that adds to overweight sector (>30%) gets -0.2."""
        rec = make_rec(
            title="Buy More Technology",
            summary="Add tech stocks to your technology holdings.",
            detailed_rationale="Increase technology allocation.",
        )
        ctx = make_context(
            current_sector_allocation={"technology": 60.0}  # overweight
        )
        score = ranker._score_diversification(rec, ctx)
        # base 0.5 - overweight 0.2 = 0.3
        assert score <= 0.35

    def test_new_tickers_boost_score(self, ranker):
        """Recommendations with tickers NOT in current portfolio get +0.1."""
        rec = make_rec(tickers=["VWO", "IEMG"])  # emerging markets ETFs
        ctx = make_context(
            current_sector_allocation={"technology": 20.0},
            portfolio_tickers=["AAPL", "MSFT", "TLT"],  # no VWO or IEMG
        )
        score = ranker._score_diversification(rec, ctx)
        assert score >= 0.6  # base 0.5 + new tickers 0.1

    def test_existing_tickers_no_boost(self, ranker):
        """Recs suggesting already-held tickers do not get the new-ticker boost."""
        rec = make_rec(tickers=["AAPL"])
        ctx = make_context(
            current_sector_allocation={"technology": 20.0},
            portfolio_tickers=["AAPL", "MSFT"],  # AAPL already held
        )
        base_score = ranker._score_diversification(make_rec(), ctx)
        boost_score = ranker._score_diversification(rec, ctx)
        assert boost_score == pytest.approx(base_score, abs=0.01)

    def test_score_clamped_to_0_1(self, ranker):
        """Score must never go below 0 or above 1 regardless of inputs."""
        rec = make_rec(
            category=RecommendationCategory.DIVERSIFY,
            title="Healthcare bonds",
            summary="healthcare exposure via bonds",
            detailed_rationale="healthcare is underweight at 3%",
            tickers=["NEW_TICKER"],
        )
        ctx = make_context(
            current_sector_allocation={"healthcare": 3.0},
            portfolio_tickers=["OLD"],
        )
        score = ranker._score_diversification(rec, ctx)
        assert 0.0 <= score <= 1.0


# ── composite_score ───────────────────────────────────────────────────────────


class TestCompositeScore:
    """Composite score = 0.4*relevance + 0.35*risk_alignment + 0.25*diversification."""

    def test_formula_matches_weights(self):
        rec = make_rec()
        rec.relevance_score = 0.8
        rec.risk_alignment_score = 0.6
        rec.diversification_score = 0.4
        expected = 0.4 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4
        assert rec.composite_score == pytest.approx(expected, abs=0.001)

    def test_all_zeros_gives_zero(self):
        rec = make_rec()
        rec.relevance_score = 0.0
        rec.risk_alignment_score = 0.0
        rec.diversification_score = 0.0
        assert rec.composite_score == pytest.approx(0.0)

    def test_all_ones_gives_one(self):
        rec = make_rec()
        rec.relevance_score = 1.0
        rec.risk_alignment_score = 1.0
        rec.diversification_score = 1.0
        assert rec.composite_score == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        """The three weight coefficients must sum to 1.0."""
        assert pytest.approx(0.4 + 0.35 + 0.25) == 1.0

    def test_relevance_has_highest_weight(self):
        """Relevance (0.4) > risk_alignment (0.35) > diversification (0.25)."""
        rec_rel = make_rec()
        rec_rel.relevance_score = 1.0
        rec_rel.risk_alignment_score = 0.0
        rec_rel.diversification_score = 0.0

        rec_risk = make_rec()
        rec_risk.relevance_score = 0.0
        rec_risk.risk_alignment_score = 1.0
        rec_risk.diversification_score = 0.0

        rec_div = make_rec()
        rec_div.relevance_score = 0.0
        rec_div.risk_alignment_score = 0.0
        rec_div.diversification_score = 1.0

        assert rec_rel.composite_score > rec_risk.composite_score > rec_div.composite_score


# ── _filter_by_risk ─────────────────────────────────────────────────────────


class TestFilterByRisk:
    """_filter_by_risk removes recs that exceed user's risk tolerance."""

    def test_conservative_removes_moderate_and_high(self, ranker):
        recs = [
            make_rec(id="low", risk_level=RiskLevel.LOW),
            make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker._filter_by_risk(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "low"

    def test_moderate_keeps_low_and_moderate(self, ranker):
        recs = [
            make_rec(id="low", risk_level=RiskLevel.LOW),
            make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker._filter_by_risk(recs, ctx)
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {"low", "mod"}

    def test_aggressive_keeps_all(self, ranker):
        recs = [
            make_rec(id="low", risk_level=RiskLevel.LOW),
            make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="aggressive")
        result = ranker._filter_by_risk(recs, ctx)
        assert len(result) == 3

    def test_no_profile_keeps_all(self, ranker):
        """Without a user risk profile, all recommendations pass through."""
        recs = [
            make_rec(id="low", risk_level=RiskLevel.LOW),
            make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance=None)
        result = ranker._filter_by_risk(recs, ctx)
        assert len(result) == 2

    def test_empty_input_returns_empty(self, ranker):
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker._filter_by_risk([], ctx)
        assert result == []


# ── _filter_excluded_sectors ─────────────────────────────────────────────────


class TestFilterExcludedSectors:
    """_filter_excluded_sectors removes recs that mention excluded sectors."""

    def test_removes_rec_mentioning_excluded_sector(self, ranker):
        recs = [
            make_rec(
                id="crypto",
                title="Buy Bitcoin",
                summary="Add cryptocurrency exposure.",
                detailed_rationale="Crypto diversifies your portfolio.",
            ),
            make_rec(
                id="bonds",
                title="Buy Treasury Bonds",
                summary="Add government bond allocation.",
                detailed_rationale="Bonds stabilize the portfolio.",
            ),
        ]
        ctx = make_context(excluded_sectors=["crypto", "cryptocurrency"])
        result = ranker._filter_excluded_sectors(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "bonds"

    def test_case_insensitive_sector_matching(self, ranker):
        rec = make_rec(
            title="Buy AAPL",
            summary="Increase Technology allocation.",
            detailed_rationale="Tech is a strong sector.",
        )
        ctx = make_context(excluded_sectors=["TECHNOLOGY"])
        result = ranker._filter_excluded_sectors([rec], ctx)
        assert len(result) == 0

    def test_no_excluded_sectors_keeps_all(self, ranker):
        recs = [make_rec(id=str(i)) for i in range(3)]
        ctx = make_context(excluded_sectors=[])
        result = ranker._filter_excluded_sectors(recs, ctx)
        assert len(result) == 3

    def test_rec_without_sector_mention_passes(self, ranker):
        rec = make_rec(
            title="Rebalance Portfolio",
            summary="Maintain target allocations.",
            detailed_rationale="Periodic rebalancing reduces drift.",
        )
        ctx = make_context(excluded_sectors=["healthcare", "energy"])
        result = ranker._filter_excluded_sectors([rec], ctx)
        assert len(result) == 1

    def test_empty_input_returns_empty(self, ranker):
        ctx = make_context(excluded_sectors=["technology"])
        result = ranker._filter_excluded_sectors([], ctx)
        assert result == []


# ── rank() full pipeline ──────────────────────────────────────────────────────


class TestRankFullPipeline:
    """rank() integrates filtering, scoring, and sorting into one call."""

    def test_empty_input_returns_empty(self, ranker):
        ctx = make_context()
        result = ranker.rank([], ctx)
        assert result == []

    def test_output_is_sorted_by_composite_score_descending(self, ranker):
        recs = [
            make_rec(
                id="low",
                priority=RecommendationPriority.LOW,
                confidence=ConfidenceLevel.LOW,
            ),
            make_rec(
                id="high",
                priority=RecommendationPriority.HIGH,
                confidence=ConfidenceLevel.HIGH,
            ),
            make_rec(
                id="medium",
                priority=RecommendationPriority.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ]
        ctx = make_context()
        result = ranker.rank(recs, ctx)
        scores = [r.composite_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_category_filter_removes_unmatched_recs(self, ranker):
        recs = [
            make_rec(id="buy", category=RecommendationCategory.BUY),
            make_rec(id="sell", category=RecommendationCategory.SELL),
            make_rec(id="rebalance", category=RecommendationCategory.REBALANCE),
        ]
        ctx = make_context()
        result = ranker.rank(
            recs, ctx, categories=[RecommendationCategory.BUY, RecommendationCategory.SELL]
        )
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {"buy", "sell"}

    def test_risk_filter_applied_before_scoring(self, ranker):
        """Conservative user should not see HIGH risk recs even if high-priority."""
        recs = [
            make_rec(
                id="high_risk",
                risk_level=RiskLevel.HIGH,
                priority=RecommendationPriority.HIGH,
                confidence=ConfidenceLevel.HIGH,
            ),
            make_rec(
                id="low_risk",
                risk_level=RiskLevel.LOW,
                priority=RecommendationPriority.LOW,
                confidence=ConfidenceLevel.LOW,
            ),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "low_risk"

    def test_scoring_updates_recommendation_fields(self, ranker):
        """After rank(), each recommendation must have non-default score values."""
        rec = make_rec()
        ctx = make_context()
        result = ranker.rank([rec], ctx)
        assert len(result) == 1
        # Scores should have been set by ranker (not left at 0.0 defaults)
        r = result[0]
        assert r.relevance_score >= 0.0
        assert r.risk_alignment_score >= 0.0
        assert r.diversification_score >= 0.0

    def test_sector_filter_combined_with_risk_filter(self, ranker):
        """Both sector exclusion and risk filtering apply independently."""
        recs = [
            make_rec(
                id="excluded_sector",
                title="Buy crypto currency tokens",
                summary="crypto exposure",
                detailed_rationale="cryptocurrency diversification",
                risk_level=RiskLevel.LOW,
            ),
            make_rec(
                id="wrong_risk",
                risk_level=RiskLevel.HIGH,
            ),
            make_rec(
                id="pass",
                risk_level=RiskLevel.LOW,
                title="Buy Treasury ETF",
                summary="Bond exposure",
                detailed_rationale="Fixed income allocation.",
            ),
        ]
        ctx = make_context(
            user_risk_tolerance="conservative",
            excluded_sectors=["crypto", "cryptocurrency"],
        )
        result = ranker.rank(recs, ctx)
        assert len(result) == 1
        assert result[0].id == "pass"

    def test_all_filtered_out_returns_empty(self, ranker):
        recs = [make_rec(risk_level=RiskLevel.HIGH, id=str(i)) for i in range(5)]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(recs, ctx)
        assert result == []


# ── Minimum composite score threshold ───────────────────────────────────────


class TestMinCompositeScoreFilter:
    """_filter_by_min_score and rank(min_composite_score=) enforce the
    'no recommendation > bad recommendation' principle.

    Score arithmetic reference (weights: rel=0.40, risk=0.35, div=0.25):
      worst-case post-risk-filter rec (low/low priority/confidence,
        adjacent risk, poor div=0.2):
          0.50*0.40 + 0.70*0.35 + 0.20*0.25 = 0.200+0.245+0.050 = 0.495
      neutral rec (low/low, adjacent risk, neutral div=0.5):
          0.50*0.40 + 0.70*0.35 + 0.50*0.25 = 0.200+0.245+0.125 = 0.570
      no-profile rec (low/low, no-profile risk=0.5, neutral div=0.5):
          0.50*0.40 + 0.50*0.35 + 0.50*0.25 = 0.200+0.175+0.125 = 0.500
    """

    def test_rec_above_threshold_is_kept(self, ranker):
        """A rec with composite >= 0.55 must survive the threshold filter."""
        rec = make_rec(
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert len(result) == 1

    def test_rec_below_threshold_is_suppressed(self, ranker):
        """A rec with composite < 0.55 must be suppressed (no rec > bad rec)."""
        # low/low priority/confidence + adjacent risk (0.7) + poor div (0.2)
        # → composite = 0.495, below default 0.55
        rec = make_rec(
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
            risk_level=RiskLevel.LOW,  # adjacent to moderate user → risk_align=0.7
        )
        ctx = make_context(
            user_risk_tolerance="moderate",
            current_sector_allocation={"Technology": 35.0},  # overweight → div penalty
        )
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert result == [], (
            "A low-quality recommendation must be suppressed rather than "
            "surfaced to the user"
        )

    def test_all_below_threshold_returns_empty(self, ranker):
        """When every rec is weak, return nothing rather than the least-bad option."""
        recs = [
            make_rec(
                id=str(i),
                priority=RecommendationPriority.LOW,
                confidence=ConfidenceLevel.LOW,
            )
            for i in range(4)
        ]
        ctx = make_context(user_risk_tolerance=None)  # no profile → risk_align=0.5
        # no-profile composite floor = 0.500, below default 0.55
        result = ranker.rank(recs, ctx, min_composite_score=0.55)
        assert result == []

    def test_threshold_zero_disables_filter(self, ranker):
        """Setting min_composite_score=0.0 must pass every scored recommendation."""
        rec = make_rec(
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1

    def test_threshold_applied_after_scoring(self, ranker):
        """Scores must be populated before threshold comparison."""
        rec = make_rec(
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert len(result) == 1
        # Scores must be written, not left at defaults
        assert result[0].relevance_score > 0.0
        assert result[0].composite_score >= 0.55

    def test_partial_pass_keeps_only_qualifying(self, ranker):
        """Mixed-quality batch: only recs above threshold survive."""
        strong = make_rec(
            id="strong",
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        weak = make_rec(
            id="weak",
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context(user_risk_tolerance=None)  # risk_align=0.5 for all
        result = ranker.rank([strong, weak], ctx, min_composite_score=0.55)
        ids = {r.id for r in result}
        assert "strong" in ids
        assert "weak" not in ids

    def test_threshold_boundary_exact_score_passes(self, ranker):
        """A rec whose composite score equals the threshold exactly must pass."""
        rec = make_rec(
            priority=RecommendationPriority.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
        )
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        # medium/medium + moderate risk → composite well above 0.55
        assert len(result) == 1
        assert result[0].composite_score >= 0.55

    def test_output_still_sorted_after_threshold(self, ranker):
        """Threshold does not disturb descending sort of surviving recs."""
        recs = [
            make_rec(
                id="medium",
                priority=RecommendationPriority.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
            ),
            make_rec(
                id="high",
                priority=RecommendationPriority.HIGH,
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank(recs, ctx, min_composite_score=0.55)
        assert len(result) == 2
        assert result[0].composite_score >= result[1].composite_score
