"""Phase 3 Adversarial & Stress Tests — Financial Recommendation Safety.

Core principle: In financial systems, "No Recommendation" is far more valuable
than a "Bad Recommendation."  Every test here verifies the system prefers
silence or a refusal over an inappropriate suggestion.

Three threat categories
-----------------------
1. ConflictingRiskRequests
   User's stated risk profile directly conflicts with the candidate products.
   Catches: risk-filter bypass, structural-field spoofing via text, poisoned datasets.

2. AmbiguousQueryRelevance
   Vague, under-specified queries that should produce low-confidence signals,
   not hallucinated high-confidence recommendations.
   Catches: relevance inflation, confidence fabrication, threshold evasion.

3. DataAbsenceHandling
   No eligible products exist for the requested category / sector / profile.
   Catches: silent substitution, fallback hallucination, error-vs-empty confusion.

All tests are offline — no LLM, no network, no external state.

Run with:
    pytest tests/eval/test_adversarial_safety.py -v -s
"""

from __future__ import annotations

import pytest

from backend.src.recommendation.engine.ranker import RISK_COMPATIBILITY, RecommendationRanker
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_rec(
    *,
    id: str = "rec_test",
    category: RecommendationCategory = RecommendationCategory.BUY,
    title: str = "Generic Fund",
    summary: str = "A generic investment fund.",
    detailed_rationale: str = "Standard investment rationale.",
    tickers: list[str] | None = None,
    risk_level: RiskLevel = RiskLevel.MODERATE,
    priority: RecommendationPriority = RecommendationPriority.MEDIUM,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    suggested_action: str = "Allocate 5% of portfolio.",
) -> Recommendation:
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
    return AggregatedContext(
        user_risk_tolerance=user_risk_tolerance,
        current_sector_allocation=current_sector_allocation or {},
        excluded_sectors=excluded_sectors or [],
        portfolio_tickers=portfolio_tickers or [],
    )


@pytest.fixture
def ranker() -> RecommendationRanker:
    return RecommendationRanker()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFLICTING RISK REQUESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictingRiskRequests:
    """Validates that the risk filter acts as an unconditional gate.

    A conservative or risk-averse user must NEVER receive a MODERATE or HIGH
    risk recommendation, regardless of how the product is described in text.
    The filter operates exclusively on the structured `risk_level` field —
    not on titles, summaries, or rationale prose.

    Failure mode caught: A poorly-configured pipeline that trusts text labels
    instead of the structured field, allowing "Bitcoin Fund (very safe!)" to
    slip through to a retiree.
    """

    # ── Scenario A: explicit profile vs explicit product risk ─────────────

    def test_conservative_user_blocked_from_high_risk_crypto(self, ranker):
        """Conservative investor requesting high-yield Bitcoin: must be blocked.

        Real-world analogue: a user self-declares 'conservative' then asks for
        'high-yield Bitcoin recommendations'. The system should return nothing.
        """
        bitcoin_fund = make_rec(
            id="btc_high_yield",
            title="Bitcoin High-Yield Growth Fund",
            summary="Aggressive crypto exposure for maximum returns.",
            detailed_rationale="Invest 100% in Bitcoin futures for high yield.",
            tickers=["BITO", "GBTC"],
            risk_level=RiskLevel.HIGH,
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([bitcoin_fund], ctx)
        assert result == [], (
            "A HIGH-risk crypto product must be blocked for a conservative user. "
            "Returning it would expose the user to inappropriate volatility risk."
        )

    def test_conservative_user_blocked_from_moderate_leveraged_etf(self, ranker):
        """Risk-averse retiree requesting leveraged ETFs: must be blocked.

        Leveraged ETFs are MODERATE-to-HIGH risk by nature. A conservative user
        (retiree profile) should never see them.
        """
        leveraged_etf = make_rec(
            id="leveraged_qqq",
            title="3x NASDAQ Leveraged ETF",
            summary="Triple leveraged exposure to NASDAQ-100.",
            detailed_rationale="Use 3x leverage for amplified tech sector returns.",
            tickers=["TQQQ", "SQQQ"],
            risk_level=RiskLevel.MODERATE,
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([leveraged_etf], ctx)
        assert result == [], (
            "A MODERATE-risk leveraged ETF must be blocked for a conservative user. "
            "The risk filter must include MODERATE as a blocked level for conservative."
        )

    def test_conservative_user_receives_only_low_risk_products(self, ranker):
        """Mixed portfolio: conservative user gets only LOW risk recs.

        Given a realistic product catalogue mixing LOW, MODERATE, and HIGH risk
        products, a conservative user must receive only the LOW-risk subset.
        """
        products = [
            make_rec(id="us_treasury", title="US Treasury Bond ETF", risk_level=RiskLevel.LOW,
                     summary="Government bonds, capital preservation focus."),
            make_rec(id="sp500_etf", title="S&P 500 Index ETF", risk_level=RiskLevel.MODERATE,
                     summary="Broad market equity exposure."),
            make_rec(id="growth_stock", title="Tech Growth Portfolio", risk_level=RiskLevel.HIGH,
                     summary="Concentrated tech equities for maximum growth."),
            make_rec(id="muni_bond", title="Municipal Bond Fund", risk_level=RiskLevel.LOW,
                     summary="Tax-advantaged local government bonds."),
            make_rec(id="options_fund", title="Options Income Strategy", risk_level=RiskLevel.HIGH,
                     summary="Generates income via covered call writing."),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(products, ctx)

        assert len(result) == 2, "Only the 2 LOW-risk products should survive."
        returned_ids = {r.id for r in result}
        assert returned_ids == {"us_treasury", "muni_bond"}
        assert all(r.risk_level == RiskLevel.LOW for r in result), (
            "Every returned recommendation must be LOW risk for a conservative user."
        )

    def test_moderate_user_blocked_from_high_risk_only(self, ranker):
        """Moderate-risk user: LOW and MODERATE pass, HIGH is blocked."""
        products = [
            make_rec(id="bond", risk_level=RiskLevel.LOW),
            make_rec(id="etf", risk_level=RiskLevel.MODERATE),
            make_rec(id="crypto", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank(products, ctx)
        assert len(result) == 2
        assert all(r.risk_level in {RiskLevel.LOW, RiskLevel.MODERATE} for r in result)
        assert not any(r.id == "crypto" for r in result)

    def test_aggressive_user_receives_all_risk_levels(self, ranker):
        """Aggressive investor: all risk levels are acceptable — none blocked by risk filter.

        Uses min_composite_score=0.0 to isolate the risk-filter behaviour from
        the composite-score threshold. A LOW-risk product scores risk_align=0.3
        for an aggressive user (distance 2), which can drop composite below 0.55.
        This test is specifically about the risk filter, not score quality.
        """
        products = [
            make_rec(id="bond", risk_level=RiskLevel.LOW),
            make_rec(id="etf", risk_level=RiskLevel.MODERATE),
            make_rec(id="crypto", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="aggressive")
        result = ranker.rank(products, ctx, min_composite_score=0.0)
        assert len(result) == 3, "Aggressive users should receive all risk levels."

    # ── Scenario B: adversarial data injection ────────────────────────────

    def test_adversarial_dataset_violation_rate_is_zero(self, ranker):
        """Poisoned dataset: HIGH-risk products disguised with conservative text.

        This is the core adversarial injection test. The dataset has been
        deliberately constructed so that every HIGH-risk product carries
        conservative-sounding titles and summaries intended to confuse a
        text-based filter. The structural `risk_level` field is the ground
        truth — the ranker must use it, not the prose.

        Failure mode: a system that trusts text labels would pass some or all
        of these through, causing a conservative user to receive toxic products.
        """
        # 20 HIGH-risk products masquerading as conservative instruments
        poisoned_products = [
            make_rec(
                id="poison_00",
                title="Conservative Capital Preservation Fund",
                summary="Designed specifically for conservative, risk-averse investors seeking stable returns.",
                detailed_rationale="Ultra-safe strategy with minimal volatility.",
                risk_level=RiskLevel.HIGH,  # structural truth: HIGH risk
            ),
            make_rec(
                id="poison_01",
                title="Retiree Safe-Income Bond Equivalent",
                summary="Income-focused product approved for conservative portfolios.",
                detailed_rationale="Suitable for retirement accounts, low-risk exposure.",
                risk_level=RiskLevel.HIGH,
            ),
            make_rec(
                id="poison_02",
                title="Low-Volatility Guaranteed Return Note",
                summary="Capital guaranteed. Conservative allocation. Risk-free equivalent.",
                detailed_rationale="Zero downside, conservative risk profile label applied.",
                risk_level=RiskLevel.HIGH,
            ),
            make_rec(
                id="poison_03",
                title="Government-Backed Conservative Growth",
                summary="Government-backed, conservative, low-risk, safe for all investors.",
                detailed_rationale="Conservative label applied to pass screening filters.",
                risk_level=RiskLevel.HIGH,
            ),
            make_rec(
                id="poison_04",
                title="Capital Protection Fund — Conservative",
                summary="Named 'conservative' to target risk-averse demographics.",
                detailed_rationale="conservative risk tolerance safe retiree low-risk bond equivalent.",
                risk_level=RiskLevel.HIGH,
            ),
            make_rec(id="poison_05", title="Safe Harbour Bond Fund", risk_level=RiskLevel.HIGH,
                     summary="conservative low-risk safe stable income."),
            make_rec(id="poison_06", title="Defensive Income Portfolio", risk_level=RiskLevel.HIGH,
                     summary="conservative capital preservation minimal risk."),
            make_rec(id="poison_07", title="Stable Value Conservative Pool", risk_level=RiskLevel.HIGH,
                     summary="suitable for conservative risk-averse investors."),
            make_rec(id="poison_08", title="Principal Protection Conservative Note", risk_level=RiskLevel.HIGH,
                     summary="conservative safe no-risk principal guaranteed."),
            make_rec(id="poison_09", title="Ultra-Low Risk Conservative Blend", risk_level=RiskLevel.HIGH,
                     summary="ultra low risk conservative approved safe harbour."),
            make_rec(id="poison_10", title="Conservative Anchor Portfolio", risk_level=RiskLevel.HIGH,
                     summary="conservative anchor strategy minimal drawdown."),
            make_rec(id="poison_11", title="Risk-Free Conservative ETF", risk_level=RiskLevel.HIGH,
                     summary="conservative label, risk-free equivalent in name."),
            make_rec(id="poison_12", title="Safety Net Conservative Fund", risk_level=RiskLevel.HIGH,
                     summary="conservative safety net stable-value product."),
            make_rec(id="poison_13", title="Low-Beta Conservative Equity", risk_level=RiskLevel.HIGH,
                     summary="low-beta conservative equity for retirement."),
            make_rec(id="poison_14", title="Balanced Conservative Income", risk_level=RiskLevel.HIGH,
                     summary="conservative balanced income strategy."),
            make_rec(id="poison_15", title="Capital Guaranteed Conservative Note", risk_level=RiskLevel.HIGH,
                     summary="conservative capital guaranteed product."),
            make_rec(id="poison_16", title="Safe Conservative Growth Bond", risk_level=RiskLevel.HIGH,
                     summary="conservative bond-equivalent safe growth."),
            make_rec(id="poison_17", title="Retiree Conservative Allocation", risk_level=RiskLevel.HIGH,
                     summary="built for retirees, conservative, low risk."),
            make_rec(id="poison_18", title="All-Weather Conservative Strategy", risk_level=RiskLevel.HIGH,
                     summary="all-weather conservative minimal risk inflation hedge."),
            make_rec(id="poison_19", title="Conservative Fixed Income Proxy", risk_level=RiskLevel.HIGH,
                     summary="conservative fixed-income proxy strategy."),
        ]

        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(poisoned_products, ctx)

        # ── Violation rate computation ────────────────────────────────────
        acceptable = RISK_COMPATIBILITY["conservative"]  # {RiskLevel.LOW}
        violations = [r for r in result if r.risk_level not in acceptable]
        violation_rate = len(violations) / len(poisoned_products) * 100

        print(f"\n  [ADVERSARIAL] Dataset size:   {len(poisoned_products)}")
        print(f"  [ADVERSARIAL] Results:        {len(result)}")
        print(f"  [ADVERSARIAL] Violations:     {len(violations)}")
        print(f"  [ADVERSARIAL] Violation Rate: {violation_rate:.1f}%")

        assert violation_rate == 0.0, (
            f"SAFETY VIOLATION: {len(violations)} HIGH-risk product(s) reached a "
            f"conservative user. Violation rate: {violation_rate:.1f}%. "
            "The risk filter must operate on the structural `risk_level` field, "
            "not on text content."
        )
        assert result == [], (
            "The poisoned dataset contains only HIGH-risk products. "
            "Zero items should reach a conservative user."
        )

    def test_adversarial_mixed_dataset_only_safe_products_pass(self, ranker):
        """Poisoned mix: HIGH/MODERATE disguised + genuine LOW-risk products.

        A realistic attack: an adversary injects disguised products alongside
        legitimate conservative-safe ones. Only the genuine LOW-risk products
        must pass. The disguised HIGH/MODERATE ones must be blocked.
        """
        legitimate = [
            make_rec(id="legit_treasury", title="US Treasury ETF", risk_level=RiskLevel.LOW,
                     summary="Government-backed bonds for capital preservation."),
            make_rec(id="legit_muni", title="Municipal Bond Fund", risk_level=RiskLevel.LOW,
                     summary="Tax-advantaged government bonds."),
        ]
        injected = [
            make_rec(id="inject_high_a", title="Safe Conservative Bond Equivalent",
                     risk_level=RiskLevel.HIGH, summary="conservative safe bond."),
            make_rec(id="inject_mod_b", title="Moderate Conservative Balanced Fund",
                     risk_level=RiskLevel.MODERATE, summary="conservative balanced safe."),
            make_rec(id="inject_high_c", title="Conservative Capital Preservation Notes",
                     risk_level=RiskLevel.HIGH, summary="capital guaranteed conservative."),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(legitimate + injected, ctx)

        returned_ids = {r.id for r in result}
        assert returned_ids == {"legit_treasury", "legit_muni"}, (
            "Only the 2 genuine LOW-risk products must pass. "
            "All 3 injected HIGH/MODERATE products must be blocked."
        )
        assert all(r.risk_level == RiskLevel.LOW for r in result)

    def test_stress_all_high_risk_large_dataset_blocked(self, ranker):
        """Stress test: 100 HIGH-risk products against a conservative user.

        Verifies the filter scales without any false negatives under load.
        """
        large_poisoned = [
            make_rec(id=f"stress_{i:03d}", risk_level=RiskLevel.HIGH,
                     title=f"High-Yield Aggressive Fund {i}",
                     summary="High risk, high return crypto-derivatives strategy.")
            for i in range(100)
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(large_poisoned, ctx)

        violations = [r for r in result if r.risk_level not in {RiskLevel.LOW}]
        violation_rate = len(violations) / len(large_poisoned) * 100

        print(f"\n  [STRESS] Dataset size:   {len(large_poisoned)}")
        print(f"  [STRESS] Violation Rate: {violation_rate:.1f}%")

        assert result == [], f"Expected 0 results, got {len(result)}"
        assert violation_rate == 0.0, f"Violation rate {violation_rate:.1f}% must be 0%"

    def test_single_safe_product_survives_poisoned_dataset(self, ranker):
        """1 legitimate product among 49 adversarial ones: exactly 1 must pass."""
        needle = make_rec(
            id="safe_needle",
            title="Short-Duration Treasury Bill Fund",
            risk_level=RiskLevel.LOW,
            summary="Capital preservation via T-bills.",
        )
        haystack = [
            make_rec(id=f"hay_{i}", risk_level=RiskLevel.HIGH,
                     title=f"Conservative Safe Fund {i}")
            for i in range(49)
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([needle] + haystack, ctx)

        assert len(result) == 1, "Exactly one LOW-risk product should survive."
        assert result[0].id == "safe_needle"

    def test_no_profile_user_is_not_over_protected(self, ranker):
        """No profile present: risk filter is bypassed entirely.

        When no risk profile exists the system cannot make a tolerance judgment.
        All products pass the risk filter (the caller must prompt for profiling).
        This is the correct safe default — withholding would be paternalistic
        without information.
        """
        products = [
            make_rec(id="low", risk_level=RiskLevel.LOW),
            make_rec(id="mod", risk_level=RiskLevel.MODERATE),
            make_rec(id="high", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance=None)
        # Pass min_composite_score=0.0 to isolate risk filter behaviour only
        result = ranker.rank(products, ctx, min_composite_score=0.0)
        assert len(result) == 3, (
            "Without a risk profile the filter must be bypassed. "
            "All 3 products should pass."
        )

    def test_conservative_all_moderate_batch_blocked(self, ranker):
        """Batch of 5 MODERATE-risk products: all blocked for conservative user."""
        products = [
            make_rec(id=f"mod_{i}", risk_level=RiskLevel.MODERATE,
                     title=f"Balanced Fund {i}", summary="Balanced equity-bond blend.")
            for i in range(5)
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(products, ctx)
        assert result == [], "All MODERATE products must be blocked for conservative users."

    def test_moderate_user_receives_low_and_moderate_exact_count(self, ranker):
        """Moderate user: exact subset of LOW + MODERATE products pass, HIGH blocked."""
        products = [
            make_rec(id="t1", risk_level=RiskLevel.LOW),
            make_rec(id="t2", risk_level=RiskLevel.LOW),
            make_rec(id="t3", risk_level=RiskLevel.MODERATE),
            make_rec(id="t4", risk_level=RiskLevel.MODERATE),
            make_rec(id="t5", risk_level=RiskLevel.HIGH),
            make_rec(id="t6", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank(products, ctx)
        assert len(result) == 4
        assert {r.id for r in result} == {"t1", "t2", "t3", "t4"}

    def test_risk_filter_not_bypassed_by_conservative_ticker_names(self, ranker):
        """Tickers named 'SAFE', 'BOND' do not bypass the structural risk_level filter."""
        rec = make_rec(
            id="sneaky",
            tickers=["SAFE", "BOND", "CONSERVATIVE"],
            risk_level=RiskLevel.HIGH,
            title="Aggressive Crypto Fund",
            summary="High-volatility cryptocurrency derivatives strategy.",
        )
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([rec], ctx)
        assert result == [], "Conservative-sounding tickers must not bypass the risk filter."

    def test_aggressive_user_high_risk_has_perfect_risk_alignment(self, ranker):
        """Aggressive user + HIGH risk rec → risk_alignment_score = 1.0 (distance 0)."""
        rec = make_rec(risk_level=RiskLevel.HIGH,
                       priority=RecommendationPriority.HIGH,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="aggressive")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        assert result[0].risk_alignment_score == pytest.approx(1.0)

    def test_conservative_low_risk_has_perfect_alignment(self, ranker):
        """Conservative user + LOW risk rec → risk_alignment_score = 1.0 (distance 0)."""
        rec = make_rec(risk_level=RiskLevel.LOW,
                       priority=RecommendationPriority.HIGH,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        assert result[0].risk_alignment_score == pytest.approx(1.0)

    def test_moderate_moderate_risk_has_perfect_alignment(self, ranker):
        """Moderate user + MODERATE risk rec → risk_alignment_score = 1.0 (distance 0)."""
        rec = make_rec(risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.HIGH,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        assert result[0].risk_alignment_score == pytest.approx(1.0)

    def test_poisoned_moderate_products_blocked_for_conservative(self, ranker):
        """MODERATE products with conservative-sounding labels are still blocked."""
        poisoned = [
            make_rec(id=f"pm_{i}", risk_level=RiskLevel.MODERATE,
                     title=f"Stable Low-Risk Bond-Like Fund {i}",
                     summary="conservative safe low-risk capital preservation fund.")
            for i in range(8)
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(poisoned, ctx)
        assert result == [], "MODERATE products must be blocked regardless of prose labels."

    def test_stress_300_products_conservative_risk_filter(self, ranker):
        """300 mixed-risk products: conservative user receives only LOW-risk subset."""
        low_recs = [make_rec(id=f"low_{i}", risk_level=RiskLevel.LOW) for i in range(100)]
        mod_recs = [make_rec(id=f"mod_{i}", risk_level=RiskLevel.MODERATE) for i in range(100)]
        high_recs = [make_rec(id=f"high_{i}", risk_level=RiskLevel.HIGH) for i in range(100)]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(low_recs + mod_recs + high_recs, ctx)
        violations = [r for r in result if r.risk_level != RiskLevel.LOW]
        assert violations == [], f"{len(violations)} non-LOW risk products leaked through."
        assert all(r.id.startswith("low_") for r in result)

    def test_ten_high_risk_recs_pass_for_aggressive(self, ranker):
        """Ten HIGH-risk products: all pass for an aggressive user."""
        products = [
            make_rec(id=f"hr_{i}", risk_level=RiskLevel.HIGH,
                     priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.HIGH)
            for i in range(10)
        ]
        ctx = make_context(user_risk_tolerance="aggressive")
        result = ranker.rank(products, ctx)
        assert len(result) == 10, "All 10 HIGH-risk products must pass for an aggressive user."

    def test_risk_filter_applied_regardless_of_priority_level(self, ranker):
        """HIGH priority does not grant a bypass of the risk filter."""
        rec = make_rec(
            id="urgent_high_risk",
            risk_level=RiskLevel.HIGH,
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
            title="Urgent: Exclusive Crypto Opportunity",
        )
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([rec], ctx)
        assert result == [], "HIGH priority must not bypass the risk filter."

    def test_risk_filter_applied_regardless_of_confidence_level(self, ranker):
        """HIGH confidence does not grant a bypass of the risk filter."""
        rec = make_rec(risk_level=RiskLevel.HIGH,
                       priority=RecommendationPriority.MEDIUM,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([rec], ctx)
        assert result == [], "HIGH confidence must not bypass the risk filter."

    def test_risk_filter_deterministic_same_result_twice(self, ranker):
        """Risk filter is stateless: identical inputs always produce identical output."""
        products = [
            make_rec(id="a", risk_level=RiskLevel.LOW),
            make_rec(id="b", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result_1 = ranker.rank(products, ctx)
        result_2 = ranker.rank(products, ctx)
        assert [r.id for r in result_1] == [r.id for r in result_2]

    def test_risk_compatibility_conservative_contains_only_low(self):
        """RISK_COMPATIBILITY table: conservative maps to exactly {LOW}."""
        assert RISK_COMPATIBILITY["conservative"] == {RiskLevel.LOW}

    def test_risk_compatibility_moderate_contains_low_and_moderate(self):
        """RISK_COMPATIBILITY table: moderate maps to exactly {LOW, MODERATE}."""
        assert RISK_COMPATIBILITY["moderate"] == {RiskLevel.LOW, RiskLevel.MODERATE}

    def test_risk_compatibility_aggressive_contains_all_three(self):
        """RISK_COMPATIBILITY table: aggressive maps to all three risk levels."""
        assert RISK_COMPATIBILITY["aggressive"] == {
            RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. AMBIGUOUS QUERY RELEVANCE
# ─────────────────────────────────────────────────────────────────────────────

class TestAmbiguousQueryRelevance:
    """Validates that vague queries produce low-confidence signals, never high.

    The ranker uses `priority` and `confidence` as proxies for how well a
    recommendation addresses the user's query.  An ambiguous user query
    ("make me money", "good investment") would cause the LLM generator to
    produce LOW priority / LOW confidence outputs.  These tests verify that
    such signals correctly propagate into low relevance scores and, when
    combined with a missing or neutral risk profile, into suppression by
    the minimum composite score threshold.

    Failure mode caught: A system that pads or inflates confidence for vague
    queries, producing a spuriously authoritative-sounding recommendation
    when the generator actually has little signal to work with.
    """

    # ── Relevance score arithmetic ─────────────────────────────────────────
    # _score_relevance:  base=0.5, priority.HIGH+0.3/MED+0.15, conf.HIGH+0.2/MED+0.1
    # LOW/LOW  → 0.50
    # LOW/MED  → 0.60
    # MED/LOW  → 0.65
    # MED/MED  → 0.75
    # HIGH/HIGH → 1.00 (capped)

    AMBIGUOUS_QUERIES = [
        "I want to make a lot of money",
        "Give me a good investment",
        "Something profitable",
        "What should I invest in?",
        "Help me grow my wealth",
    ]

    def test_low_confidence_low_priority_yields_minimum_relevance(self, ranker):
        """LOW/LOW priority+confidence produces minimum relevance score (0.50).

        A vague query such as 'make me money' should never yield a
        high-confidence recommendation.  The floor of relevance scoring is 0.50.
        """
        rec = make_rec(
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        assert result[0].relevance_score == 0.50, (
            "LOW priority + LOW confidence must yield the minimum relevance "
            "score of 0.50 — no fabricated confidence boost."
        )

    def test_medium_priority_medium_confidence_yields_expected_relevance(self, ranker):
        """MED/MED yields 0.75 relevance — confirms scoring is not inflated."""
        rec = make_rec(
            priority=RecommendationPriority.MEDIUM,
            confidence=ConfidenceLevel.MEDIUM,
        )
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.75, abs=1e-9)

    def test_ambiguous_query_rec_cannot_achieve_max_relevance_with_low_confidence(self, ranker):
        """LOW confidence caps relevance — system cannot fake certainty.

        Even if priority is HIGH (the generator tried hard), LOW confidence
        means the output is uncertain.  Maximum achievable relevance with
        LOW confidence is 0.80 (HIGH priority + LOW confidence).
        """
        rec = make_rec(
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score < 1.0, (
            "LOW confidence must prevent the system from claiming maximum relevance."
        )
        assert result[0].relevance_score == pytest.approx(0.80, abs=1e-9)

    def test_ambiguous_rec_suppressed_when_no_profile(self, ranker):
        """No profile + LOW/LOW ambiguous rec: composite 0.500 < threshold 0.55.

        When neither the user's risk profile nor the query provides useful
        signal, the system should stay silent rather than speculate.

        Score breakdown:
          relevance=0.50 × 0.40 = 0.200
          risk_align=0.50 × 0.35 = 0.175  (no profile → neutral 0.5)
          diversif=0.50  × 0.25 = 0.125  (no allocation data → neutral 0.5)
          composite = 0.500 < 0.55 threshold → SUPPRESSED
        """
        rec = make_rec(
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert result == [], (
            "An ambiguous recommendation for a user with no risk profile must be "
            "suppressed. composite=0.500 is below the 0.55 safety threshold."
        )

    def test_five_ambiguous_recs_all_suppressed_no_profile(self, ranker):
        """Five 'make-money' type recs: all suppressed for no-profile user.

        Simulates a batch of vague-query outputs from the LLM generator.
        None should reach the user without a risk profile providing alignment.
        """
        recs = [
            make_rec(
                id=f"vague_{i}",
                title=q[:40],
                summary=q,
                priority=RecommendationPriority.LOW,
                confidence=ConfidenceLevel.LOW,
            )
            for i, q in enumerate(self.AMBIGUOUS_QUERIES)
        ]
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank(recs, ctx, min_composite_score=0.55)

        print(f"\n  [AMBIGUOUS] Candidates:  {len(recs)}")
        print(f"  [AMBIGUOUS] Suppressed:  {len(recs) - len(result)}")
        print(f"  [AMBIGUOUS] Passed:      {len(result)}")

        assert result == [], (
            "All 5 vague-query recommendations must be suppressed for a user "
            "with no investment profile. Composite scores are all 0.500 < 0.55."
        )

    def test_relevance_score_monotonically_increases_with_confidence(self, ranker):
        """Relevance is a strict function of priority+confidence, never random.

        Given equal priority, HIGH confidence must outscore MEDIUM, which must
        outscore LOW. This ensures the scoring is deterministic and cannot
        accidentally surface a low-confidence rec above a high-confidence one.
        """
        low_conf = make_rec(id="low", priority=RecommendationPriority.MEDIUM,
                             confidence=ConfidenceLevel.LOW)
        med_conf = make_rec(id="med", priority=RecommendationPriority.MEDIUM,
                             confidence=ConfidenceLevel.MEDIUM)
        high_conf = make_rec(id="high", priority=RecommendationPriority.MEDIUM,
                              confidence=ConfidenceLevel.HIGH)

        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([low_conf, med_conf, high_conf], ctx, min_composite_score=0.0)

        scores = {r.id: r.relevance_score for r in result}
        assert scores["low"] < scores["med"] < scores["high"], (
            "Relevance scores must be strictly monotonic with confidence level."
        )

    def test_ambiguous_rec_with_profile_passes_but_has_low_relevance(self, ranker):
        """Aligned risk profile can lift ambiguous rec above threshold.

        A moderate user with a perfect-match rec (risk_align=1.0) may receive
        a LOW/LOW rec because risk alignment compensates:
          composite = 0.50*0.40 + 1.00*0.35 + 0.50*0.25 = 0.675 > 0.55

        This is intentional: a conservative, low-confidence suggestion that
        perfectly matches the user's profile is better than silence.
        However: relevance_score must still be 0.50 (no inflation).
        """
        rec = make_rec(
            risk_level=RiskLevel.MODERATE,   # matches moderate user → risk_align=1.0
            priority=RecommendationPriority.LOW,
            confidence=ConfidenceLevel.LOW,
        )
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.55)

        assert len(result) == 1, (
            "Risk-aligned LOW/LOW rec should pass for a moderate user "
            "(composite 0.675 > 0.55)."
        )
        assert result[0].relevance_score == pytest.approx(0.50, abs=1e-9), (
            "Relevance score must remain 0.50 (not inflated) even when the "
            "rec passes the threshold via risk alignment."
        )

    def test_composite_score_never_exceeds_1_for_any_valid_input(self, ranker):
        """Composite score is bounded [0, 1] — no overflow on best-case inputs."""
        best_rec = make_rec(
            risk_level=RiskLevel.MODERATE,
            priority=RecommendationPriority.HIGH,
            confidence=ConfidenceLevel.HIGH,
        )
        ctx = make_context(
            user_risk_tolerance="moderate",
            current_sector_allocation={"Technology": 5.0},   # underweight → +div
        )
        result = ranker.rank([best_rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        score = result[0].composite_score
        assert 0.0 <= score <= 1.0, f"Composite score {score} out of [0, 1] bounds."

    def test_threshold_prevents_low_quality_batch_from_polluting_output(self, ranker):
        """Mixed batch: 3 strong recs + 5 weak ambiguous recs.

        Only the 3 strong recs (medium+ quality) should survive the 0.55
        threshold. The 5 weak ones must be suppressed silently.
        """
        strong = [
            make_rec(id=f"strong_{i}", risk_level=RiskLevel.MODERATE,
                     priority=RecommendationPriority.HIGH,
                     confidence=ConfidenceLevel.HIGH)
            for i in range(3)
        ]
        weak = [
            make_rec(id=f"weak_{i}", risk_level=RiskLevel.MODERATE,
                     priority=RecommendationPriority.LOW,
                     confidence=ConfidenceLevel.LOW)
            for i in range(5)
        ]
        ctx = make_context(user_risk_tolerance=None)  # no profile → risk neutral
        result = ranker.rank(strong + weak, ctx, min_composite_score=0.55)

        # With no profile, risk_align=0.5:
        # Strong: rel=1.0*0.4 + risk=0.5*0.35 + div=0.5*0.25 = 0.400+0.175+0.125 = 0.700 → PASS
        # Weak:   rel=0.5*0.4 + risk=0.5*0.35 + div=0.5*0.25 = 0.200+0.175+0.125 = 0.500 → BLOCK
        passed_ids = {r.id for r in result}
        assert all(pid.startswith("strong_") for pid in passed_ids), (
            "Weak ambiguous recs must not pollute the output alongside strong ones."
        )
        assert len(result) == 3

    # ── Additional relevance scoring tests ────────────────────────────────

    def test_high_priority_low_confidence_yields_0_80_relevance(self, ranker):
        """HIGH priority + LOW confidence = 0.50 + 0.30 + 0.00 = 0.80."""
        rec = make_rec(priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.LOW)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.80, abs=1e-9)

    def test_low_priority_high_confidence_yields_0_70_relevance(self, ranker):
        """LOW priority + HIGH confidence = 0.50 + 0.00 + 0.20 = 0.70."""
        rec = make_rec(priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.HIGH)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.70, abs=1e-9)

    def test_medium_priority_low_confidence_yields_0_65_relevance(self, ranker):
        """MEDIUM priority + LOW confidence = 0.50 + 0.15 + 0.00 = 0.65."""
        rec = make_rec(priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.LOW)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.65, abs=1e-9)

    def test_low_priority_medium_confidence_yields_0_60_relevance(self, ranker):
        """LOW priority + MEDIUM confidence = 0.50 + 0.00 + 0.10 = 0.60."""
        rec = make_rec(priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.60, abs=1e-9)

    def test_high_priority_medium_confidence_yields_0_90_relevance(self, ranker):
        """HIGH priority + MEDIUM confidence = 0.50 + 0.30 + 0.10 = 0.90."""
        rec = make_rec(priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.90, abs=1e-9)

    def test_medium_priority_high_confidence_yields_0_85_relevance(self, ranker):
        """MEDIUM priority + HIGH confidence = 0.50 + 0.15 + 0.20 = 0.85."""
        rec = make_rec(priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.HIGH)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].relevance_score == pytest.approx(0.85, abs=1e-9)

    def test_relevance_monotonic_with_priority_fixed_high_confidence(self, ranker):
        """At HIGH confidence: LOW priority < MEDIUM priority < HIGH priority in relevance."""
        low_p = make_rec(id="lp", priority=RecommendationPriority.LOW,
                          confidence=ConfidenceLevel.HIGH)
        med_p = make_rec(id="mp", priority=RecommendationPriority.MEDIUM,
                          confidence=ConfidenceLevel.HIGH)
        high_p = make_rec(id="hp", priority=RecommendationPriority.HIGH,
                           confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([low_p, med_p, high_p], ctx, min_composite_score=0.0)
        scores = {r.id: r.relevance_score for r in result}
        assert scores["lp"] < scores["mp"] < scores["hp"]

    def test_ten_vague_recs_no_profile_all_suppressed(self, ranker):
        """Ten LOW/LOW recs with no risk profile: all suppressed (composite = 0.500 < 0.55)."""
        vague = [
            "make me money", "good investment idea", "profitable stock tip",
            "what is best to buy", "help me get rich", "easy returns",
            "safe guaranteed profit", "double my money", "best opportunity",
            "grow my savings fast",
        ]
        recs = [
            make_rec(id=f"vague_{i}", title=q[:40], summary=q,
                     priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW)
            for i, q in enumerate(vague)
        ]
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank(recs, ctx, min_composite_score=0.55)
        assert result == [], "All 10 LOW/LOW recs must be suppressed with no risk profile."

    def test_high_high_rec_composite_exact_arithmetic(self, ranker):
        """HIGH/HIGH MODERATE rec for moderate user: composite = 0.875 exactly.

        relevance = 1.00, risk_align = 1.00, div = 0.50 (BUY, no allocation data)
        composite = 0.40*1.00 + 0.35*1.00 + 0.25*0.50 = 0.400 + 0.350 + 0.125 = 0.875
        """
        rec = make_rec(risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.HIGH,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        assert result[0].composite_score == pytest.approx(0.875, abs=1e-9)

    def test_sort_descending_by_composite_three_recs(self, ranker):
        """Three distinct score profiles: result is sorted highest composite first."""
        best = make_rec(id="best", risk_level=RiskLevel.MODERATE,
                        priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.HIGH)
        mid = make_rec(id="mid", risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.MEDIUM)
        worst = make_rec(id="worst", risk_level=RiskLevel.LOW,
                         priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW)
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([worst, mid, best], ctx, min_composite_score=0.0)
        composites = [r.composite_score for r in result]
        assert composites == sorted(composites, reverse=True)
        assert result[0].id == "best"

    def test_no_profile_yields_neutral_risk_alignment_0_5(self, ranker):
        """Without a user profile, risk_alignment_score = 0.5 for any risk level."""
        for rl in [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH]:
            rec = make_rec(risk_level=rl)
            ctx = make_context(user_risk_tolerance=None)
            result = ranker.rank([rec], ctx, min_composite_score=0.0)
            assert result[0].risk_alignment_score == pytest.approx(0.5), (
                f"No-profile risk_align must be 0.5 regardless of rec risk level ({rl})."
            )

    def test_confidence_enum_not_mutated_by_ranker(self, ranker):
        """The confidence field on a recommendation is not modified by the ranker."""
        rec = make_rec(confidence=ConfidenceLevel.LOW)
        ctx = make_context()
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].confidence == ConfidenceLevel.LOW

    def test_all_priority_confidence_combos_bounded_0_to_1(self, ranker):
        """All 9 priority x confidence combinations produce relevance in [0, 1]."""
        ctx = make_context(user_risk_tolerance=None)
        for priority in RecommendationPriority:
            for confidence in ConfidenceLevel:
                rec = make_rec(priority=priority, confidence=confidence)
                result = ranker.rank([rec], ctx, min_composite_score=0.0)
                score = result[0].relevance_score
                assert 0.0 <= score <= 1.0, (
                    f"relevance={score} out of [0,1] for {priority}/{confidence}"
                )

    def test_low_low_recs_with_moderate_profile_pass_via_risk_alignment(self, ranker):
        """LOW/LOW recs perfectly matched to moderate user clear the threshold via risk_align.

        composite = 0.40*0.50 + 0.35*1.00 + 0.25*0.50 = 0.200 + 0.350 + 0.125 = 0.675 > 0.55
        """
        recs = [
            make_rec(id=f"ll_{i}", risk_level=RiskLevel.MODERATE,
                     priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW)
            for i in range(3)
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank(recs, ctx, min_composite_score=0.55)
        assert len(result) == 3, "3 LOW/LOW recs with perfect risk alignment must pass."

    def test_weak_rec_composite_0_500_suppressed_by_default_threshold(self, ranker):
        """BUY + LOW/LOW + no profile: composite = 0.500 < 0.55, suppressed."""
        rec = make_rec(category=RecommendationCategory.BUY,
                       priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW,
                       risk_level=RiskLevel.MODERATE)
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert result == [], "composite=0.500 must be suppressed by the 0.55 threshold."


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA ABSENCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestDataAbsenceHandling:
    """Validates graceful empty-result behaviour — no fallback hallucination.

    When no eligible products exist for the user's request (wrong sector,
    wrong category, no safe products after risk filter), the system must
    return an empty list. It must NOT:
      - Silently substitute a product from a different category/sector
      - Raise an exception or error
      - Return a product that failed filtering as a "best effort" fallback

    Failure mode caught: Systems that swap in a nearest-match product when
    the requested product type is unavailable, creating the illusion that
    the user's specific request was fulfilled.
    """

    def test_empty_candidate_list_returns_empty_not_error(self, ranker):
        """No candidates at all: ranker must return [] cleanly.

        The most fundamental absence test: an empty input catalogue.
        """
        ctx = make_context()
        result = ranker.rank([], ctx)
        assert result == [], (
            "An empty candidate list must produce an empty result, not an exception."
        )
        assert isinstance(result, list)

    def test_no_esg_bonds_in_dataset_returns_empty(self, ranker):
        """ESG bonds requested but only conventional equities in dataset.

        User requests ESG-screened bonds; only conventional equity products
        exist. The ranker filters by category — the caller passes only the
        ESG bond category. No silent substitution with equity products.
        """
        available_products = [
            make_rec(id="spy", title="S&P 500 ETF", category=RecommendationCategory.BUY,
                     risk_level=RiskLevel.MODERATE),
            make_rec(id="aapl", title="Apple Stock", category=RecommendationCategory.BUY,
                     risk_level=RiskLevel.MODERATE),
            make_rec(id="msft", title="Microsoft Stock", category=RecommendationCategory.HOLD,
                     risk_level=RiskLevel.MODERATE),
        ]
        ctx = make_context()
        # Caller requests only DIVERSIFY recs (ESG bond category proxy)
        result = ranker.rank(
            available_products,
            ctx,
            categories=[RecommendationCategory.DIVERSIFY],
        )
        assert result == [], (
            "When no products match the requested category (ESG bonds / DIVERSIFY), "
            "the system must return nothing — not substitute conventional equities."
        )

    def test_no_products_in_requested_geographic_market(self, ranker):
        """Specific geographic market (Emerging Markets) not in dataset.

        User requests Emerging Market products; dataset only has US/EU products.
        Modelled via excluded_sectors — EM products are excluded, none remain.
        """
        all_us_products = [
            make_rec(id="us_bond", title="US Government Bond", risk_level=RiskLevel.LOW,
                     summary="United States Treasury bonds allocation strategy."),
            make_rec(id="eu_bond", title="European Corporate Bond", risk_level=RiskLevel.LOW,
                     summary="European Union investment grade corporate bonds."),
        ]
        # User explicitly excluded US and European markets (wants EM only)
        ctx = make_context(
            user_risk_tolerance="moderate",
            excluded_sectors=["united states", "european"],
        )
        result = ranker.rank(all_us_products, ctx)
        assert result == [], (
            "When all available products belong to excluded markets, "
            "the system must return nothing — not silently serve excluded products."
        )

    def test_all_risk_incompatible_returns_empty_not_nearest_match(self, ranker):
        """Conservative user, only HIGH/MODERATE risk products available.

        The system must return nothing.  It must NOT pick the "least bad"
        option (e.g., MODERATE for a conservative user) as a fallback.
        """
        only_risky = [
            make_rec(id="mod", title="Balanced Growth ETF", risk_level=RiskLevel.MODERATE),
            make_rec(id="high", title="Aggressive Tech Portfolio", risk_level=RiskLevel.HIGH),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(only_risky, ctx)
        assert result == [], (
            "No products are appropriate for a conservative user here. "
            "Returning the MODERATE product as a 'nearest match' would be a "
            "safety violation — conservative means LOW risk only."
        )

    def test_below_threshold_products_not_returned_as_fallback(self, ranker):
        """All products score below the minimum composite threshold.

        The system must return nothing rather than serve low-quality
        recommendations just to have something to show.
        """
        low_quality_products = [
            make_rec(id=f"low_{i}", priority=RecommendationPriority.LOW,
                     confidence=ConfidenceLevel.LOW)
            for i in range(5)
        ]
        ctx = make_context(user_risk_tolerance=None)  # composite floor = 0.500
        result = ranker.rank(low_quality_products, ctx, min_composite_score=0.55)
        assert result == [], (
            "Products below the quality threshold must not be returned as fallback. "
            "Silence is safer than a low-confidence suggestion."
        )

    def test_sector_exclusion_removes_all_products_in_sector(self, ranker):
        """User excludes the only available sector — result must be empty."""
        crypto_products = [
            make_rec(id=f"crypto_{i}", title=f"Bitcoin Fund {i}",
                     summary="cryptocurrency bitcoin blockchain exposure",
                     risk_level=RiskLevel.HIGH)
            for i in range(5)
        ]
        ctx = make_context(
            user_risk_tolerance="aggressive",  # risk not the issue here
            excluded_sectors=["bitcoin", "cryptocurrency"],
        )
        result = ranker.rank(crypto_products, ctx)
        assert result == [], (
            "All products in an excluded sector must be removed. "
            "The system must not substitute products from that sector."
        )

    def test_combined_risk_and_sector_exclusion_returns_empty(self, ranker):
        """Risk filter + sector exclusion together eliminate all candidates."""
        products = [
            make_rec(id="high_crypto", risk_level=RiskLevel.HIGH,
                     title="Bitcoin Strategy", summary="bitcoin cryptocurrency"),
            make_rec(id="mod_equity", risk_level=RiskLevel.MODERATE,
                     title="Tech Growth", summary="technology sector equity growth"),
        ]
        ctx = make_context(
            user_risk_tolerance="conservative",
            excluded_sectors=["technology"],
        )
        result = ranker.rank(products, ctx)
        assert result == [], (
            "HIGH risk blocked by risk filter; MODERATE also blocked (conservative). "
            "Combined filters must produce empty, not a fallback."
        )

    def test_empty_result_is_list_type_not_none(self, ranker):
        """An empty result must be an empty list, never None or an exception.

        Callers rely on `len(result) == 0` and iteration. A None return would
        cause AttributeError downstream.
        """
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(
            [make_rec(risk_level=RiskLevel.HIGH)],
            ctx,
        )
        assert result is not None, "ranker.rank() must never return None."
        assert isinstance(result, list), "ranker.rank() must always return a list."
        assert len(result) == 0

    def test_max_recommendations_zero_eligible_after_all_filters(self, ranker):
        """After all 4 filter stages, zero items remain — max_recs has no effect."""
        products = [
            make_rec(id="a", risk_level=RiskLevel.HIGH,
                     title="Tech ETF", summary="technology sector"),
            make_rec(id="b", risk_level=RiskLevel.HIGH,
                     title="Crypto Fund", summary="bitcoin cryptocurrency"),
        ]
        ctx = make_context(
            user_risk_tolerance="conservative",
            excluded_sectors=["technology", "cryptocurrency"],
        )
        # Even requesting up to 10 recommendations, empty is correct
        result = ranker.rank(products, ctx)
        assert result == []

    def test_category_filter_and_risk_filter_both_eliminate_no_fallback(self, ranker):
        """Category filter removes half, risk filter removes the rest — no fallback."""
        products = [
            # Passes category filter (BUY) but fails risk filter (HIGH for conservative)
            make_rec(id="buy_high", category=RecommendationCategory.BUY,
                     risk_level=RiskLevel.HIGH),
            # Passes risk filter (LOW) but fails category filter (SELL, not in requested)
            make_rec(id="sell_low", category=RecommendationCategory.SELL,
                     risk_level=RiskLevel.LOW),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(
            products,
            ctx,
            categories=[RecommendationCategory.BUY],
        )
        assert result == [], (
            "No single product survives both filters simultaneously. "
            "The system must return nothing, not fall back to serving a "
            "partially-matching product."
        )

    # ── Additional data-absence and scoring-context tests ─────────────────

    def test_sell_category_excluded_by_buy_only_request(self, ranker):
        """A SELL rec is filtered when only BUY category is requested."""
        sell_rec = make_rec(id="sell", category=RecommendationCategory.SELL,
                            risk_level=RiskLevel.LOW)
        buy_rec = make_rec(id="buy", category=RecommendationCategory.BUY,
                           risk_level=RiskLevel.LOW)
        ctx = make_context()
        result = ranker.rank([sell_rec, buy_rec], ctx, categories=[RecommendationCategory.BUY])
        assert len(result) == 1
        assert result[0].id == "buy"

    def test_sector_exclusion_case_insensitive_uppercase(self, ranker):
        """Sector exclusion matching is case-insensitive."""
        rec = make_rec(id="tech_rec",
                       title="TECHNOLOGY Sector ETF",
                       summary="TECHNOLOGY growth stocks concentrated.",
                       risk_level=RiskLevel.LOW)
        ctx = make_context(user_risk_tolerance="conservative",
                           excluded_sectors=["technology"])
        result = ranker.rank([rec], ctx)
        assert result == [], "Sector exclusion must be case-insensitive."

    def test_new_ticker_raises_diversification_score(self, ranker):
        """A ticker not in portfolio adds +0.1 to the diversification score.

        Sector allocation must be non-empty (neutral: no under/overweight) to
        bypass the early-return path in _score_diversification.
        """
        rec = make_rec(tickers=["NVDA"], risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM,
                       confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate", portfolio_tickers=[],
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.6, abs=1e-9), (
            "NVDA is new: base 0.5 + new_ticker 0.1 = 0.6."
        )

    def test_existing_ticker_no_diversification_boost(self, ranker):
        """A ticker already held in portfolio does not trigger the +0.1 new-ticker boost."""
        rec = make_rec(tickers=["AAPL"], risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM,
                       confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate", portfolio_tickers=["AAPL"],
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.5, abs=1e-9), (
            "AAPL already in portfolio — no boost. Div stays at 0.5."
        )

    def test_rebalance_category_boosts_diversification_score(self, ranker):
        """REBALANCE category: base 0.5 + category bonus 0.2 = 0.7.

        Requires non-empty sector allocation to bypass early-return in _score_diversification.
        """
        rec = make_rec(category=RecommendationCategory.REBALANCE, risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate",
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.7, abs=1e-9)

    def test_diversify_category_boosts_diversification_score(self, ranker):
        """DIVERSIFY category: base 0.5 + category bonus 0.2 = 0.7.

        Requires non-empty sector allocation to bypass early-return in _score_diversification.
        """
        rec = make_rec(category=RecommendationCategory.DIVERSIFY, risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate",
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.7, abs=1e-9)

    def test_diversify_rec_at_exact_threshold_0_55_passes(self, ranker):
        """DIVERSIFY + LOW/LOW + no profile: composite = 0.550 exactly, passes.

        relevance=0.50, risk_align=0.50, div=0.70 (DIVERSIFY +0.2 from base 0.5)
        composite = 0.40*0.50 + 0.35*0.50 + 0.25*0.70 = 0.200 + 0.175 + 0.175 = 0.550

        Note: sector allocation must be non-empty so _score_diversification
        applies the category bonus rather than returning the early-return 0.5.
        """
        rec = make_rec(category=RecommendationCategory.DIVERSIFY,
                       priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW,
                       risk_level=RiskLevel.MODERATE)
        ctx = make_context(user_risk_tolerance=None,
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert len(result) == 1, "composite=0.550 >= 0.55 threshold — must pass."

    def test_buy_rec_below_threshold_0_50_suppressed(self, ranker):
        """BUY + LOW/LOW + no profile: composite = 0.500, suppressed by 0.55 threshold."""
        rec = make_rec(category=RecommendationCategory.BUY,
                       priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW,
                       risk_level=RiskLevel.MODERATE)
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([rec], ctx, min_composite_score=0.55)
        assert result == [], "composite=0.500 < 0.55 threshold — must be suppressed."

    def test_categories_none_returns_all_eligible(self, ranker):
        """categories=None applies no category filter — all eligible products return."""
        products = [
            make_rec(id="buy", category=RecommendationCategory.BUY, risk_level=RiskLevel.LOW),
            make_rec(id="sell", category=RecommendationCategory.SELL, risk_level=RiskLevel.LOW),
            make_rec(id="hold", category=RecommendationCategory.HOLD, risk_level=RiskLevel.LOW),
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(products, ctx, categories=None, min_composite_score=0.0)
        assert len(result) == 3, "categories=None must not apply any category filter."

    def test_multiple_excluded_sectors_removes_all_matching(self, ranker):
        """Two excluded sectors: each matching product removed, unmatched survives."""
        products = [
            make_rec(id="tech", title="Tech Growth ETF",
                     summary="technology sector growth stocks.", risk_level=RiskLevel.LOW),
            make_rec(id="crypto", title="Crypto Fund",
                     summary="bitcoin cryptocurrency blockchain.", risk_level=RiskLevel.LOW),
            make_rec(id="safe", title="Treasury Bond",
                     summary="government bonds safe haven.", risk_level=RiskLevel.LOW),
        ]
        ctx = make_context(user_risk_tolerance="conservative",
                           excluded_sectors=["technology", "bitcoin"])
        result = ranker.rank(products, ctx)
        assert len(result) == 1
        assert result[0].id == "safe"

    def test_large_safe_dataset_all_pass_for_conservative(self, ranker):
        """30 LOW-risk products: all pass the risk filter for a conservative user."""
        products = [
            make_rec(id=f"safe_{i}", risk_level=RiskLevel.LOW,
                     priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.MEDIUM)
            for i in range(30)
        ]
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank(products, ctx)
        assert len(result) == 30
        assert all(r.risk_level == RiskLevel.LOW for r in result)

    def test_no_portfolio_tickers_new_ticker_boost_applies(self, ranker):
        """Empty portfolio: any tickers in the rec are all 'new' and get the +0.1 boost."""
        rec = make_rec(tickers=["MSFT", "GOOGL"], risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.MEDIUM, confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(portfolio_tickers=[],
                           current_sector_allocation={"Bonds": 20.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.6, abs=1e-9)

    def test_overweight_sector_reduces_diversification_score(self, ranker):
        """Rec targeting an overweight sector (>30%): div = base 0.5 - penalty 0.2 = 0.3."""
        rec = make_rec(title="Technology Growth Fund",
                       summary="Increases technology sector stock exposure.",
                       risk_level=RiskLevel.MODERATE, priority=RecommendationPriority.MEDIUM,
                       confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate",
                           current_sector_allocation={"Technology": 40.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.3, abs=1e-9)

    def test_underweight_sector_raises_diversification_score(self, ranker):
        """Rec targeting an underweight sector (<10%): div = base 0.5 + boost 0.2 = 0.7."""
        rec = make_rec(title="Healthcare Sector ETF",
                       summary="Adds healthcare sector diversification exposure.",
                       risk_level=RiskLevel.MODERATE, priority=RecommendationPriority.MEDIUM,
                       confidence=ConfidenceLevel.MEDIUM)
        ctx = make_context(user_risk_tolerance="moderate",
                           current_sector_allocation={"Healthcare": 5.0})
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].diversification_score == pytest.approx(0.7, abs=1e-9)

    def test_result_sorted_descending_by_composite_score(self, ranker):
        """Final result list is always sorted descending by composite_score."""
        products = [
            make_rec(id="lo", priority=RecommendationPriority.LOW,
                     confidence=ConfidenceLevel.LOW, risk_level=RiskLevel.MODERATE),
            make_rec(id="md", priority=RecommendationPriority.MEDIUM,
                     confidence=ConfidenceLevel.MEDIUM, risk_level=RiskLevel.MODERATE),
            make_rec(id="hi", priority=RecommendationPriority.HIGH,
                     confidence=ConfidenceLevel.HIGH, risk_level=RiskLevel.MODERATE),
        ]
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank(products, ctx, min_composite_score=0.0)
        composites = [r.composite_score for r in result]
        assert composites == sorted(composites, reverse=True)
        assert result[0].id == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DIVERSIFICATION SCORING
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversificationScoring:
    """Validates _score_diversification arithmetic across portfolio contexts.

    The diversification sub-score (weight 0.25 in composite) rewards recommendations
    that improve portfolio spread. Tests verify all adjustments: category bonus,
    underweight/overweight sector matching, and new-ticker boost.

    All tests access div score via rank([rec], ctx, min_composite_score=0.0)
    to avoid threshold interference.
    """

    @pytest.fixture
    def ranker(self):
        return RecommendationRanker()

    def _div(self, ranker, rec, ctx):
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1
        return result[0].diversification_score

    def test_no_allocation_data_yields_neutral_0_5(self, ranker):
        """No sector allocation data → diversification score = 0.5 (neutral)."""
        rec = make_rec(category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation=None)
        assert self._div(ranker, rec, ctx) == pytest.approx(0.5, abs=1e-9)

    def test_underweight_sector_in_title_boosts_div(self, ranker):
        """Rec targeting an underweight sector (<10%) earns +0.2: 0.5 + 0.2 = 0.7."""
        rec = make_rec(title="Healthcare Sector Fund",
                       summary="Invests in healthcare companies.",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Healthcare": 3.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_overweight_sector_in_title_reduces_div(self, ranker):
        """Rec targeting an overweight sector (>30%) earns -0.2: 0.5 - 0.2 = 0.3."""
        rec = make_rec(title="Technology Mega-Cap ETF",
                       summary="Concentrated technology equity exposure.",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Technology": 50.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.3, abs=1e-9)

    def test_diversify_category_adds_0_2_to_base(self, ranker):
        """DIVERSIFY category: 0.5 + 0.2 = 0.7. Requires non-empty allocation to skip early-return."""
        rec = make_rec(category=RecommendationCategory.DIVERSIFY)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_rebalance_category_adds_0_2_to_base(self, ranker):
        """REBALANCE category: 0.5 + 0.2 = 0.7. Requires non-empty allocation to skip early-return."""
        rec = make_rec(category=RecommendationCategory.REBALANCE)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_buy_category_no_base_boost(self, ranker):
        """BUY category: no category bonus, stays at 0.5. Uses neutral allocation."""
        rec = make_rec(category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.5, abs=1e-9)

    def test_new_ticker_in_rec_adds_0_1_boost(self, ranker):
        """Ticker not in portfolio: +0.1 boost. 0.5 + 0.1 = 0.6. Requires non-empty allocation."""
        rec = make_rec(tickers=["TSLA"], category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0},
                           portfolio_tickers=["AAPL", "MSFT"])
        assert self._div(ranker, rec, ctx) == pytest.approx(0.6, abs=1e-9)

    def test_existing_ticker_no_boost(self, ranker):
        """Ticker already held: no boost. Score stays at 0.5. Requires non-empty allocation."""
        rec = make_rec(tickers=["AAPL"], category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0},
                           portfolio_tickers=["AAPL", "MSFT"])
        assert self._div(ranker, rec, ctx) == pytest.approx(0.5, abs=1e-9)

    def test_underweight_plus_diversify_double_boost(self, ranker):
        """DIVERSIFY + underweight sector: 0.5 + 0.2 + 0.2 = 0.9."""
        rec = make_rec(title="Healthcare Diversification Fund",
                       summary="Improves healthcare sector allocation.",
                       category=RecommendationCategory.DIVERSIFY)
        ctx = make_context(current_sector_allocation={"Healthcare": 2.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.9, abs=1e-9)

    def test_overweight_plus_diversify_category_net_neutral(self, ranker):
        """DIVERSIFY + overweight sector: 0.5 + 0.2 - 0.2 = 0.5 (net neutral)."""
        rec = make_rec(title="Technology Diversification Fund",
                       summary="Adds technology diversification exposure.",
                       category=RecommendationCategory.DIVERSIFY)
        ctx = make_context(current_sector_allocation={"Technology": 45.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.5, abs=1e-9)

    def test_both_underweight_and_overweight_rec_targets_underweight(self, ranker):
        """Portfolio with over/underweight sectors; rec only mentions underweight one."""
        rec = make_rec(title="Healthcare Sector Fund",
                       summary="Invests in healthcare sector.",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={
            "Technology": 45.0,   # overweight — not in rec text
            "Healthcare": 4.0,    # underweight — in rec text
        })
        # addresses underweight → +0.2; does NOT add to technology → no penalty
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_sector_matching_in_summary_field(self, ranker):
        """Sector matching checks the summary, not just the title."""
        rec = make_rec(title="Diversified Growth Fund",
                       summary="healthcare sector exposure for underweight portfolios",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Healthcare": 3.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_sector_matching_in_detailed_rationale(self, ranker):
        """Sector matching also checks the detailed_rationale field."""
        rec = make_rec(title="Growth Fund", summary="Broad exposure.",
                       detailed_rationale="Targets healthcare companies specifically.",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Healthcare": 3.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.7, abs=1e-9)

    def test_multiple_new_tickers_still_single_0_1_boost(self, ranker):
        """Multiple new tickers give a flat +0.1, not per-ticker. 0.5 + 0.1 = 0.6."""
        rec = make_rec(tickers=["TSLA", "NVDA", "AMD", "INTC"],
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Bonds": 20.0}, portfolio_tickers=[])
        assert self._div(ranker, rec, ctx) == pytest.approx(0.6, abs=1e-9)

    def test_div_score_clamped_at_maximum_1_0(self, ranker):
        """All bonuses stacked: 0.5 + 0.2 + 0.2 + 0.1 = 1.0, clamped at 1.0."""
        rec = make_rec(title="Healthcare Diversification Fund",
                       summary="healthcare rebalancing for underweight allocation.",
                       tickers=["HCA"],
                       category=RecommendationCategory.DIVERSIFY)
        ctx = make_context(portfolio_tickers=[],
                           current_sector_allocation={"Healthcare": 2.0})
        score = self._div(ranker, rec, ctx)
        assert score <= 1.0
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_div_score_minimum_with_overweight_only(self, ranker):
        """Overweight penalty alone: 0.5 - 0.2 = 0.3 (no further clamping needed)."""
        rec = make_rec(title="Technology Mega ETF",
                       summary="technology sector concentrated exposure.",
                       category=RecommendationCategory.BUY)
        ctx = make_context(current_sector_allocation={"Technology": 50.0})
        assert self._div(ranker, rec, ctx) == pytest.approx(0.3, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCORING BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringBoundaryConditions:
    """Validates the composite score formula and boundary conditions.

    Tests directly inspect the Pydantic computed_field `composite_score`
    and the risk_alignment_score output from the ranker, verifying the
    formula and distance-based scoring are arithmetically correct.
    """

    @pytest.fixture
    def ranker(self):
        return RecommendationRanker()

    def _scored_rec(self, rel=0.0, risk=0.0, div=0.0) -> Recommendation:
        rec = make_rec()
        rec.relevance_score = rel
        rec.risk_alignment_score = risk
        rec.diversification_score = div
        return rec

    def test_composite_exact_formula_weights(self):
        """composite = 0.40*rel + 0.35*risk + 0.25*div. Verify with (0.8, 0.6, 0.4)."""
        rec = self._scored_rec(rel=0.8, risk=0.6, div=0.4)
        expected = 0.40 * 0.8 + 0.35 * 0.6 + 0.25 * 0.4  # 0.32 + 0.21 + 0.10 = 0.63
        assert rec.composite_score == pytest.approx(expected, abs=1e-9)
        assert rec.composite_score == pytest.approx(0.63, abs=1e-9)

    def test_composite_all_zeros_returns_zero(self):
        """composite(0, 0, 0) = 0.0."""
        rec = self._scored_rec(rel=0.0, risk=0.0, div=0.0)
        assert rec.composite_score == pytest.approx(0.0, abs=1e-9)

    def test_composite_all_ones_returns_one(self):
        """composite(1, 1, 1) = 1.0."""
        rec = self._scored_rec(rel=1.0, risk=1.0, div=1.0)
        assert rec.composite_score == pytest.approx(1.0, abs=1e-9)

    def test_composite_weights_sum_to_one(self):
        """The three weights 0.40 + 0.35 + 0.25 must equal exactly 1.0."""
        assert 0.40 + 0.35 + 0.25 == pytest.approx(1.0, abs=1e-9)

    def test_risk_alignment_perfect_match_returns_1_0(self, ranker):
        """Distance-0 match (conservative+LOW): risk_alignment_score = 1.0."""
        rec = make_rec(risk_level=RiskLevel.LOW,
                       priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="conservative")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].risk_alignment_score == pytest.approx(1.0)

    def test_risk_alignment_adjacent_match_returns_0_7(self, ranker):
        """Distance-1 match (moderate user, LOW risk rec): risk_alignment_score = 0.7."""
        rec = make_rec(risk_level=RiskLevel.LOW,
                       priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].risk_alignment_score == pytest.approx(0.7)

    def test_risk_alignment_two_apart_returns_0_3(self, ranker):
        """Distance-2 match (aggressive user, LOW risk rec): risk_alignment_score = 0.3."""
        rec = make_rec(risk_level=RiskLevel.LOW,
                       priority=RecommendationPriority.HIGH, confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="aggressive")
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert result[0].risk_alignment_score == pytest.approx(0.3)

    def test_risk_alignment_no_profile_returns_0_5(self, ranker):
        """No profile: risk_alignment_score = 0.5 (neutral) for all risk levels."""
        for rl in [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH]:
            rec = make_rec(risk_level=rl)
            ctx = make_context(user_risk_tolerance=None)
            result = ranker.rank([rec], ctx, min_composite_score=0.0)
            assert result[0].risk_alignment_score == pytest.approx(0.5)

    def test_min_composite_zero_disables_filter(self, ranker):
        """min_composite_score=0.0 disables the threshold: even low-quality recs pass."""
        rec = make_rec(priority=RecommendationPriority.LOW, confidence=ConfidenceLevel.LOW)
        ctx = make_context(user_risk_tolerance=None)
        result = ranker.rank([rec], ctx, min_composite_score=0.0)
        assert len(result) == 1

    def test_min_composite_one_blocks_all_non_perfect_recs(self, ranker):
        """min_composite_score=1.0 blocks all recs with composite < 1.0.

        HIGH/HIGH MODERATE rec for moderate user:
          composite = 0.40*1.0 + 0.35*1.0 + 0.25*0.5 = 0.875 < 1.0 → blocked.
        """
        rec = make_rec(risk_level=RiskLevel.MODERATE,
                       priority=RecommendationPriority.HIGH,
                       confidence=ConfidenceLevel.HIGH)
        ctx = make_context(user_risk_tolerance="moderate")
        result = ranker.rank([rec], ctx, min_composite_score=1.0)
        assert result == [], "composite=0.875 < 1.0 — blocked by max threshold."
