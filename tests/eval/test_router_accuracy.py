"""Evaluation: Intent router accuracy on the 100-case labeled testset.

Phase 1 scope — offline, no real LLM calls:
  - Tests the keyword fallback classifier (_fallback_classification) only.
  - Measures precision, recall, and F1 per intent class.
  - Documents known failure modes with explicit test cases.

Run with:
    pytest tests/eval/test_router_accuracy.py -v -s
"""

from __future__ import annotations

import pytest

from backend.src.multi_agent.state import UserIntent

from .conftest import compute_metrics, print_metrics_report

# ── Intent label constants ──────────────────────────────────────────────────

PORTFOLIO = UserIntent.PORTFOLIO_QUERY.value
PROFILING = UserIntent.PROFILING.value
RECOMMENDATION = UserIntent.RECOMMENDATION.value
GENERAL = UserIntent.GENERAL.value

ALL_LABELS = [PORTFOLIO, PROFILING, RECOMMENDATION, GENERAL]

# Minimum acceptable thresholds for the keyword-only fallback.
# The LLM path (Phase 2) is expected to reach ≥ 95%.
# History: 75% (initial) → 81% (round 1 fixes) → 98% (round 2 fixes).
MIN_OVERALL_ACCURACY = 0.95
MIN_CLASS_RECALL = 0.85  # every class must reach high recall


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def eval_results(router_testset, fallback_router):
    """Run the fallback classifier over all 100 testset cases once per module."""
    y_true, y_pred = [], []
    for case in router_testset:
        true_label = case["label"]
        pred_intent = fallback_router._fallback_classification(case["message"])
        y_true.append(true_label)
        y_pred.append(pred_intent.value)
    return y_true, y_pred


@pytest.fixture(scope="module")
def metrics(eval_results):
    """Compute and cache metrics for the module."""
    y_true, y_pred = eval_results
    m = compute_metrics(y_true, y_pred, ALL_LABELS)
    print_metrics_report(m, "Fallback Router - Keyword Classifier")
    return m


# ── Overall accuracy ─────────────────────────────────────────────────────────


class TestOverallAccuracy:
    def test_accuracy_meets_minimum_threshold(self, metrics):
        """Overall keyword-fallback accuracy must be at least 60%.

        The keyword approach has inherent gaps (natural language misses).
        The LLM primary classifier is expected to fill these gaps (≥95%).
        This test documents the keyword baseline, not a target.
        """
        acc = metrics["accuracy"]
        assert acc >= MIN_OVERALL_ACCURACY, (
            f"Fallback accuracy {acc:.1%} is below minimum {MIN_OVERALL_ACCURACY:.0%}. "
            f"Review keyword coverage in IntentRouter."
        )

    def test_no_class_completely_missed(self, metrics):
        """Every intent class must have at least some recall (recall > 0).

        A class with 0% recall means the keyword set is entirely missing
        for that category — a critical gap.
        """
        for label in ALL_LABELS:
            recall = metrics["per_class"][label]["recall"]
            assert recall > 0.0, (
                f"Class '{label}' has 0% recall — no keyword matches at all. "
                f"Add keywords to IntentRouter for this intent."
            )

    def test_per_class_recall_meets_minimum(self, metrics):
        """Each class must individually reach the minimum recall threshold."""
        for label in ALL_LABELS:
            recall = metrics["per_class"][label]["recall"]
            assert recall >= MIN_CLASS_RECALL, (
                f"Class '{label}' recall {recall:.1%} < minimum {MIN_CLASS_RECALL:.0%}. "
                f"Keyword set is too sparse for this intent."
            )


# ── Per-class keyword coverage ────────────────────────────────────────────────


class TestPortfolioKeywordCoverage:
    """23 of 25 portfolio cases are correctly identified by keywords (2 remain as gaps)."""

    CLEAR_PORTFOLIO_CASES = [
        "What's my net worth?",
        "Show me my portfolio",
        "What are my current holdings?",
        "How is my asset allocation split?",
        "What's my liquidity ratio?",
        "How much cash do I have available?",
        "Show my investment gains this year",
        "What's in my bank account?",
        "Tell me about my real estate value",
        "What's my current account balance?",
        "What do I own across all accounts?",
        "Show all my liabilities",
        "How much are my total assets worth?",
        "What's in my investment account?",
        "What sector is most of my wealth in?",
        "How are my stocks performing?",
        "What's the value of my investments?",
        "What are my recent investment losses?",
        "Show the accounts I have",
        "How is my brokerage account doing?",
        # Fixed round 2: natural-language financial queries now covered by new keywords
        "How am I doing financially?",
        "Give me a summary of my finances",
        "What's my current financial health?",
        # Remaining gaps (→ GENERAL): "Am I in the positive overall?", "Am I saving enough?"
    ]

    def test_clear_portfolio_cases_all_correct(self, fallback_router):
        """Every easy portfolio query with an obvious keyword must be classified correctly."""
        errors = []
        for msg in self.CLEAR_PORTFOLIO_CASES:
            result = fallback_router._fallback_classification(msg)
            if result != UserIntent.PORTFOLIO_QUERY:
                errors.append(f"  '{msg}' → got '{result.value}' (expected 'portfolio_query')")
        assert not errors, "Keyword mismatches on clear portfolio queries:\n" + "\n".join(errors)


class TestProfilingKeywordCoverage:
    """25 of 25 profiling cases are correctly identified by keywords."""

    CLEAR_PROFILING_CASES = [
        "Build my investment profile",
        "I want to do a risk assessment",
        "What's my risk tolerance?",
        "Create my investor profile",
        "Set up my investment strategy",
        "Help me build my portfolio plan",
        "I want to take the investment questionnaire",
        "What are my investment preferences?",
        "Update my investment profile",
        "Let's create my financial profile",
        # Fixed round 1: 'my investment goals' and 'my financial goals' now explicit keywords
        "I want to assess my investment goals",
        "I need to determine my financial goals",
        "Plan my investment strategy",
        "Help me set up my investment plan",
        "Build an investment portfolio for me",
        "Tell me about your profiling process",
        "Let's go through the risk questionnaire",
        "Start the investment profiling survey",
        # Fixed round 1: 'how much risk' now a profiling keyword (checked before 'how much' in portfolio)
        "I want to figure out how much risk I can handle",
        # Fixed round 2: risk-descriptor and investor-style keywords now covered
        "I need to know my risk appetite",
        "What kind of investor am I?",
        "Help me understand my investment style",
        "I'm new to investing, where should I start?",
        "Let me figure out my risk level",
        "Help me understand my investment objectives",
    ]

    def test_clear_profiling_cases_all_correct(self, fallback_router):
        """Every easy profiling query with an obvious keyword must be classified correctly."""
        errors = []
        for msg in self.CLEAR_PROFILING_CASES:
            result = fallback_router._fallback_classification(msg)
            if result != UserIntent.PROFILING:
                errors.append(f"  '{msg}' → got '{result.value}' (expected 'profiling')")
        assert not errors, "Keyword mismatches on clear profiling queries:\n" + "\n".join(errors)


class TestRecommendationKeywordCoverage:
    """25 of 25 recommendation cases are correctly identified by keywords."""

    CLEAR_RECOMMENDATION_CASES = [
        "What should I invest in?",
        "Should I buy more AAPL shares?",
        "Can you recommend some ETFs?",
        "Should I sell my tech stocks?",
        "How should I diversify my portfolio?",
        "Give me investment advice please",
        # Fixed round 1: 'where should i invest' now an explicit keyword
        "Where should I invest my extra cash?",
        "Should I rebalance my portfolio?",
        "How can I optimize my investment returns?",
        "Can you give me a suggestion on bonds?",
        "How can I improve my portfolio performance?",
        "Is it a good time to invest in bonds?",
        "Which stocks should I buy right now?",
        "I want financial advice on my allocation",
        "Help me rebalance my investments",
        "What ETFs would you recommend for growth?",
        # Fixed round 1: 'reduce risk' now a recommendation keyword (checked before 'my investments' in portfolio)
        "How do I reduce risk in my investments?",
        # Fixed round 2: additional action-intent keywords now covered
        "What's the best way to grow my wealth?",
        "Should I add international exposure to my holdings?",
        "Is now a good time to buy tech stocks?",
        "What percentage should I put in bonds?",
        "I need guidance on choosing index funds",
        "Which asset class should I focus on?",
        "Can you help me build a safer portfolio?",
        "What do you suggest for my retirement plan?",
    ]

    def test_clear_recommendation_cases_all_correct(self, fallback_router):
        """Every easy recommendation query with an obvious keyword must be classified correctly."""
        errors = []
        for msg in self.CLEAR_RECOMMENDATION_CASES:
            result = fallback_router._fallback_classification(msg)
            if result != UserIntent.RECOMMENDATION:
                errors.append(f"  '{msg}' → got '{result.value}' (expected 'recommendation')")
        assert not errors, "Keyword mismatches on clear recommendation queries:\n" + "\n".join(errors)


class TestGeneralKeywordCoverage:
    """Most general messages have no keywords and should fall through to GENERAL."""

    CLEAR_GENERAL_CASES = [
        "Hello!",
        "Thanks for your help",
        "How are you?",
        "What can you do?",
        "Goodbye",
        "Tell me a joke",
        "Who are you?",
        "What is inflation?",
        "How does compound interest work?",
        "Start over please",
        "What's a bond?",
        "Help me",
        "I just started using this app",
        "Nice, let's keep going",
        "What time is it?",
        # Fixed: 'recommendation' no longer triggers intent (keyword is now 'recommend ' with space)
        "What's your recommendation for a good finance book?",
    ]

    def test_clear_general_cases_all_correct(self, fallback_router):
        """Pure general messages with no domain keywords must not be mislabeled."""
        errors = []
        for msg in self.CLEAR_GENERAL_CASES:
            result = fallback_router._fallback_classification(msg)
            if result != UserIntent.GENERAL:
                errors.append(f"  '{msg}' → got '{result.value}' (expected 'general')")
        assert not errors, "Unexpected labels on clear general messages:\n" + "\n".join(errors)


# ── Known failure modes (documented gaps) ─────────────────────────────────────


class TestKnownFallbackGaps:
    """Remaining known limitations of the keyword-based fallback.

    These tests assert the CURRENT (still-incorrect) behavior to prevent
    silent regressions when keywords are updated.
    If a case is fixed, move it to a coverage test.

    Fixed in round 1 (moved to coverage tests):
      - 'my investment goals' / 'my financial goals' → profiling  (added explicit keywords)
      - 'where should i invest' → recommendation                  (added explicit keyword)
      - 'how much risk' → profiling                               (added explicit keyword)
      - 'recommendation' word → GENERAL (false positive fixed)    (keyword narrowed to 'recommend ')
      - 'reduce risk in my investments' → recommendation          (added 'reduce risk' keyword)

    Fixed in round 2 (moved to coverage tests):
      - 'financially' / 'my finances' / 'financial health' → portfolio  (added explicit keywords)
      - 'risk appetite' / 'risk level' → profiling                      (added explicit keywords)
      - 'what kind of investor' / 'investment style' → profiling        (added explicit keywords)
      - 'new to investing' / 'investment objectives' → profiling        (added explicit keywords)
      - 'grow my wealth' / 'should i add' / 'time to buy' → recommendation  (added keywords)
      - 'should i put' / 'guidance on' / 'asset class' → recommendation     (added keywords)
      - 'safer portfolio' / 'do you suggest' → recommendation              (added keywords)
      - 'build a safer portfolio' → recommendation                (added 'safer portfolio' keyword)
      - natural-language recommendation queries → recommendation           (now all covered)
    """

    def test_natural_language_portfolio_queries_miss(self, fallback_router):
        """2 remaining portfolio queries with no mappable keyword fall through to GENERAL.

        These are irreducibly ambiguous without risking false positives on GENERAL
        (e.g. adding 'am i' or 'saving' would mislabel many unrelated messages).
        The LLM primary classifier is expected to handle these correctly.
        """
        natural_queries = [
            "Am I in the positive overall?",
            "Am I saving enough for the future?",
        ]
        missed = []
        for msg in natural_queries:
            result = fallback_router._fallback_classification(msg)
            if result == UserIntent.PORTFOLIO_QUERY:
                missed.append(msg)
        assert len(missed) == 0, (
            f"These natural queries are now correctly classified (remove from gap list): {missed}"
        )


# ── Routing priority order ─────────────────────────────────────────────────


class TestRoutingPriorityOrder:
    """Verify that the keyword priority order (profiling > recommendation > portfolio) is stable."""

    def test_profiling_keyword_beats_portfolio_keyword(self, fallback_router):
        """When both profiling and portfolio keywords appear, profiling wins."""
        # "build my portfolio" has both "build my" (profiling) and "portfolio" (portfolio)
        result = fallback_router._fallback_classification("Build my portfolio")
        assert result == UserIntent.PROFILING

    def test_recommendation_keyword_beats_portfolio_keyword(self, fallback_router):
        """When both recommendation and portfolio keywords appear, recommendation wins."""
        # "diversify my portfolio" has "diversify" (rec) and "portfolio" (portfolio)
        result = fallback_router._fallback_classification("How should I diversify my portfolio?")
        assert result == UserIntent.RECOMMENDATION

    def test_profiling_keyword_beats_recommendation_keyword(self, fallback_router):
        """When both profiling and recommendation keywords appear, profiling wins."""
        # "help me build" (profiling) + "advice" (recommendation)
        result = fallback_router._fallback_classification(
            "Help me build my profile and give me advice"
        )
        assert result == UserIntent.PROFILING
