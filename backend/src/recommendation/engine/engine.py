"""Main recommendation engine orchestrating the full pipeline."""

import logging
import time

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.mcp.integration.hybrid_data_provider import HybridDataProvider
from backend.src.recommendation.rag.initializer import RAGInitializer

from .context_builder import ContextBuilder
from .generator import RecommendationGenerator
from .ranker import RecommendationRanker
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine.

    Pipeline:
    1. ContextBuilder gathers data from portfolio, profile, RAG, MCP
    2. RecommendationGenerator sends context + prompt to LLM
    3. RecommendationRanker scores, filters, and ranks results
    4. Results are packaged into RecommendationResponse

    Usage:
        engine = RecommendationEngine(
            portfolio_aggregator=aggregator,
            rag_initializer=rag,
            llm_client=llm,
        )
        response = engine.generate_recommendations(
            RecommendationRequest(user_id="user_001", query="How should I diversify?")
        )
    """

    def __init__(
        self,
        portfolio_aggregator: PortfolioAggregator,
        rag_initializer: RAGInitializer,
        llm_client: LLMClient | None = None,
        hybrid_data_provider: HybridDataProvider | None = None,
    ):
        self._llm = llm_client or LLMClient()
        self._context_builder = ContextBuilder(
            portfolio_aggregator=portfolio_aggregator,
            rag_initializer=rag_initializer,
            hybrid_data_provider=hybrid_data_provider,
        )
        self._generator = RecommendationGenerator(llm_client=self._llm)
        self._ranker = RecommendationRanker()
        self._aggregator = portfolio_aggregator

    def generate_recommendations(
        self, request: RecommendationRequest
    ) -> RecommendationResponse:
        """Generate personalized recommendations.

        Args:
            request: RecommendationRequest with user_id, query, and options.

        Returns:
            RecommendationResponse with ranked recommendations.
        """
        start_time = time.time()

        try:
            # Verify user exists
            portfolio = self._aggregator.get_portfolio(request.user_id)
            if not portfolio:
                return RecommendationResponse(
                    user_id=request.user_id,
                    query=request.query,
                    success=False,
                    error=f"No portfolio data found for user {request.user_id}",
                )

            # Step 1: Build aggregated context
            context = self._context_builder.build_context(
                user_id=request.user_id,
                query=request.query,
                include_live_data=request.include_live_data,
            )

            # Step 2: Generate recommendations via LLM
            # Generate extra to compensate for filtering
            raw_recommendations = self._generator.generate(
                context=context,
                query=request.query,
                max_recommendations=request.max_recommendations + 2,
            )

            # Step 3: Rank, filter, and score
            ranked_recommendations = self._ranker.rank(
                recommendations=raw_recommendations,
                context=context,
                categories=request.categories,
                min_composite_score=request.min_composite_score,
            )

            # Trim to requested count
            final_recommendations = ranked_recommendations[
                : request.max_recommendations
            ]

            # Calculate profile completeness
            profile_completeness = 0.0
            if portfolio.investment_profile:
                profile_completeness = portfolio.investment_profile.profile_completeness

            # Build warnings
            warnings = self._build_warnings(context, profile_completeness, final_recommendations, request)

            # Build summary
            summary = self._build_summary(final_recommendations)

            elapsed_ms = (time.time() - start_time) * 1000

            return RecommendationResponse(
                user_id=request.user_id,
                query=request.query,
                recommendations=final_recommendations,
                summary=summary,
                data_sources_used=context.data_sources_used,
                profile_completeness=profile_completeness,
                warnings=warnings,
                processing_time_ms=elapsed_ms,
                success=True,
            )

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}", exc_info=True)
            elapsed_ms = (time.time() - start_time) * 1000
            return RecommendationResponse(
                user_id=request.user_id,
                query=request.query,
                success=False,
                error=str(e),
                processing_time_ms=elapsed_ms,
            )

    def _build_warnings(
        self, context: object, profile_completeness: float,
        final_recommendations: list | None = None,
        request: object | None = None,
    ) -> list[str]:
        """Build warning messages based on data availability."""
        warnings = []
        if profile_completeness < 0.5:
            warnings.append(
                "Your investment profile is less than 50% complete. "
                "Complete the Investment Profile questionnaire for more "
                "personalized recommendations."
            )
        if not context.rag_context and profile_completeness < 0.3:
            pass  # already covered by completeness warning below
        if profile_completeness >= 0.3 and final_recommendations is not None and not final_recommendations:
            min_score = request.min_composite_score if request is not None else 0.55
            warnings.append(
                "No recommendations met the minimum quality threshold "
                f"(composite score >= {min_score:.0%}). "
                "This means the available suggestions were not sufficiently relevant, "
                "risk-aligned, or diversifying for your profile. "
                "Try completing your investment profile or refining your query."
            )
        if "live_quotes" not in context.data_sources_used:
            if request is not None and not request.include_live_data:
                warnings.append(
                    "Live market data was not included. Check 'Include Live Data' "
                    "to enrich recommendations with real-time market information."
                )
            else:
                warnings.append(
                    "Live market data was not available. Recommendations are based on "
                    "portfolio data and knowledge base only."
                )
        if not context.rag_context:
            warnings.append(
                "Financial knowledge base returned no results. "
                "Recommendations may be less informed."
            )
        # Always include disclaimer
        warnings.append(
            "These recommendations are informational only and do not constitute "
            "financial advice. Consult a licensed financial advisor before making "
            "investment decisions."
        )
        return warnings

    def _build_summary(self, recommendations: list) -> str:
        """Build a one-paragraph summary of all recommendations."""
        if not recommendations:
            return "No recommendations could be generated based on available data."

        count = len(recommendations)
        categories = set(r.category.value for r in recommendations)
        cat_str = ", ".join(sorted(categories))
        high_priority = [r for r in recommendations if r.priority.value == "high"]

        summary = (
            f"Generated {count} recommendation(s) across categories: {cat_str}."
        )
        if high_priority:
            summary += (
                f" {len(high_priority)} high-priority action(s) identified."
            )

        return summary
