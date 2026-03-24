"""Recommendation ranking and filtering system."""

import logging

from .schemas import (
    AggregatedContext,
    Recommendation,
    RecommendationCategory,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# Risk level compatibility: which RiskLevels are acceptable for each risk tolerance
RISK_COMPATIBILITY: dict[str, set[RiskLevel]] = {
    "conservative": {RiskLevel.LOW},
    "moderate": {RiskLevel.LOW, RiskLevel.MODERATE},
    "aggressive": {RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH},
}


class RecommendationRanker:
    """Scores, filters, and ranks recommendations based on user context.

    Scoring dimensions:
    1. Relevance (0-1): How relevant the recommendation is to the user's query
    2. Risk alignment (0-1): How well the risk level matches user's tolerance
    3. Diversification benefit (0-1): How much it improves portfolio diversification
    """

    def rank(
        self,
        recommendations: list[Recommendation],
        context: AggregatedContext,
        categories: list[RecommendationCategory] | None = None,
        min_composite_score: float = 0.55,
    ) -> list[Recommendation]:
        """Score, filter, and rank recommendations.

        Args:
            recommendations: Raw recommendations from generator.
            context: Aggregated context with user data.
            categories: Optional category filter.
            min_composite_score: Minimum composite score threshold (0-1).
                Recommendations below this score are suppressed.
                In finance, silence is better than a bad recommendation.

        Returns:
            Ranked and filtered list of recommendations.
        """
        if not recommendations:
            return []

        # Step 1: Filter by category if specified
        if categories:
            recommendations = [r for r in recommendations if r.category in categories]

        # Step 2: Filter by risk compatibility
        recommendations = self._filter_by_risk(recommendations, context)

        # Step 3: Filter by excluded sectors
        recommendations = self._filter_excluded_sectors(recommendations, context)

        # Step 4: Score each recommendation
        for rec in recommendations:
            rec.risk_alignment_score = self._score_risk_alignment(rec, context)
            rec.diversification_score = self._score_diversification(rec, context)
            rec.relevance_score = self._score_relevance(rec)

        # Step 5: Filter by minimum composite score
        recommendations = self._filter_by_min_score(recommendations, min_composite_score)

        # Step 6: Sort by composite score descending
        recommendations.sort(key=lambda r: r.composite_score, reverse=True)

        return recommendations

    def _filter_by_min_score(
        self,
        recs: list[Recommendation],
        min_score: float,
    ) -> list[Recommendation]:
        """Remove recommendations that fall below the minimum composite score.

        In financial recommendation systems, suppressing a weak recommendation
        is safer than surfacing it. A recommendation must clear every scoring
        dimension sufficiently — relevance, risk alignment, and diversification
        benefit — before it is shown to the user.

        Args:
            recs: Scored recommendations (scores must already be populated).
            min_score: Minimum composite_score required to pass (0.0 disables).

        Returns:
            Recommendations with composite_score >= min_score.
        """
        if min_score <= 0.0:
            return recs

        passed = [r for r in recs if r.composite_score >= min_score]
        removed = len(recs) - len(passed)
        if removed > 0:
            logger.info(
                f"Suppressed {removed} recommendation(s) below minimum composite "
                f"score threshold of {min_score:.2f}"
            )
        return passed

    def _filter_by_risk(
        self, recs: list[Recommendation], context: AggregatedContext
    ) -> list[Recommendation]:
        """Remove recommendations that exceed user's risk tolerance."""
        user_risk = context.user_risk_tolerance
        if not user_risk:
            return recs  # No profile, keep all

        acceptable_risks = RISK_COMPATIBILITY.get(
            user_risk, {RiskLevel.LOW, RiskLevel.MODERATE}
        )
        filtered = [r for r in recs if r.risk_level in acceptable_risks]

        removed_count = len(recs) - len(filtered)
        if removed_count > 0:
            logger.info(
                f"Filtered {removed_count} recommendations exceeding "
                f"{user_risk} risk tolerance"
            )

        return filtered

    def _filter_excluded_sectors(
        self, recs: list[Recommendation], context: AggregatedContext
    ) -> list[Recommendation]:
        """Remove recommendations in user's excluded sectors."""
        excluded = context.excluded_sectors
        if not excluded:
            return recs

        excluded_lower = [s.lower() for s in excluded]

        filtered = []
        for rec in recs:
            rec_text = f"{rec.title} {rec.summary} {rec.detailed_rationale}".lower()
            is_excluded = any(sector in rec_text for sector in excluded_lower)
            if not is_excluded:
                filtered.append(rec)

        removed_count = len(recs) - len(filtered)
        if removed_count > 0:
            logger.info(
                f"Filtered {removed_count} recommendations in excluded sectors"
            )

        return filtered

    def _score_risk_alignment(
        self, rec: Recommendation, context: AggregatedContext
    ) -> float:
        """Score how well a recommendation's risk matches user tolerance."""
        user_risk = context.user_risk_tolerance
        if not user_risk:
            return 0.5  # Neutral if no profile

        # Perfect match = 1.0, adjacent = 0.7, mismatched = 0.3
        risk_order = ["conservative", "moderate", "aggressive"]
        rec_risk_map = {
            "low": "conservative",
            "moderate": "moderate",
            "high": "aggressive",
        }

        try:
            user_idx = risk_order.index(user_risk)
            rec_idx = risk_order.index(
                rec_risk_map.get(rec.risk_level.value, "moderate")
            )
            distance = abs(user_idx - rec_idx)
        except ValueError:
            return 0.5

        if distance == 0:
            return 1.0
        elif distance == 1:
            return 0.7
        else:
            return 0.3

    def _score_diversification(
        self, rec: Recommendation, context: AggregatedContext
    ) -> float:
        """Score how much a recommendation improves portfolio diversification."""
        current_allocation = context.current_sector_allocation
        if not current_allocation:
            return 0.5  # No allocation data, neutral score

        rec_text = f"{rec.title} {rec.summary} {rec.detailed_rationale}".lower()

        # Get overweight sectors (above 30%)
        overweight_sectors = [
            s.lower() for s, pct in current_allocation.items() if pct > 30
        ]

        # Get underweight sectors (below 10%)
        underweight_sectors = [
            s.lower() for s, pct in current_allocation.items() if pct < 10
        ]

        # Check if recommendation addresses underweight sectors
        addresses_underweight = any(s in rec_text for s in underweight_sectors)

        # Check if recommendation adds to overweight sectors
        adds_to_overweight = any(s in rec_text for s in overweight_sectors)

        # Categories that inherently improve diversification
        diversifying_categories = {
            RecommendationCategory.DIVERSIFY,
            RecommendationCategory.REBALANCE,
        }

        score = 0.5  # Base score
        if rec.category in diversifying_categories:
            score += 0.2
        if addresses_underweight:
            score += 0.2
        if adds_to_overweight:
            score -= 0.2

        # Check if recommendation suggests tickers not already in portfolio
        if rec.tickers:
            existing = {t.upper() for t in context.portfolio_tickers}
            new_tickers = [t for t in rec.tickers if t.upper() not in existing]
            if new_tickers:
                score += 0.1  # New tickers improve diversification

        return max(0.0, min(1.0, score))

    def _score_relevance(self, rec: Recommendation) -> float:
        """Score how relevant a recommendation is.

        Uses priority and confidence as proxies, since the LLM has already
        determined relevance during generation.
        """
        score = 0.5

        # Priority boost
        if rec.priority.value == "high":
            score += 0.3
        elif rec.priority.value == "medium":
            score += 0.15

        # Confidence boost
        if rec.confidence.value == "high":
            score += 0.2
        elif rec.confidence.value == "medium":
            score += 0.1

        return min(1.0, score)
