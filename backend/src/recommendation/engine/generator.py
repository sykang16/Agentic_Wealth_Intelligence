"""LLM-based recommendation generator."""

import json
import logging
import uuid
from datetime import datetime

from backend.src.common.llm_client import LLMClient

from ..prompts.explanation_prompt import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_PROMPT,
)
from ..prompts.recommendation_prompt import (
    DEFAULT_QUERY,
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_PROMPT,
)
from .schemas import (
    AggregatedContext,
    ConfidenceLevel,
    DataSource,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """Generates investment recommendations using LLM with structured prompts."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def generate(
        self,
        context: AggregatedContext,
        query: str = "",
        max_recommendations: int = 5,
    ) -> list[Recommendation]:
        """Generate recommendations from aggregated context.

        Args:
            context: Aggregated context from all data sources.
            query: User's query. Uses default if empty.
            max_recommendations: Maximum recommendations to generate.

        Returns:
            List of Recommendation objects.
        """
        effective_query = query or DEFAULT_QUERY

        system_prompt = RECOMMENDATION_SYSTEM_PROMPT.format(
            max_recommendations=max_recommendations
        )
        user_prompt = RECOMMENDATION_USER_PROMPT.format(
            context=context.get_full_context(max_length=6000),
            query=effective_query,
        )

        response = self.llm.chat(
            user_message=user_prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.4,
        )
        recommendations = self._parse_recommendations(response.content, context)
        return recommendations[:max_recommendations]

    def enhance_explanation(
        self,
        recommendation: Recommendation,
        context: AggregatedContext,
        experience_level: str = "intermediate",
    ) -> str:
        """Generate a detailed explanation for a single recommendation.

        Args:
            recommendation: The recommendation to explain.
            context: Aggregated context.
            experience_level: User's investment experience level.

        Returns:
            Detailed explanation string.
        """
        system_prompt = EXPLANATION_SYSTEM_PROMPT.format(
            experience_level=experience_level
        )
        user_prompt = EXPLANATION_USER_PROMPT.format(
            title=recommendation.title,
            summary=recommendation.summary,
            category=recommendation.category.value,
            suggested_action=recommendation.suggested_action,
            risk_level=recommendation.risk_level.value,
            context=context.rag_context[:2000],
            portfolio_context=context.portfolio_context[:2000],
        )

        try:
            response = self.llm.chat(
                user_message=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.5,
            )
            return response.content
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return recommendation.detailed_rationale

    def _parse_recommendations(
        self, llm_output: str, context: AggregatedContext
    ) -> list[Recommendation]:
        """Parse LLM JSON output into Recommendation objects.

        Handles common LLM output issues: markdown code blocks, trailing commas.
        """
        # Strip markdown code fences if present
        cleaned = llm_output.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            raw_list = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}")
            logger.debug(f"Raw output: {cleaned[:500]}")
            return []

        if not isinstance(raw_list, list):
            raw_list = [raw_list]

        recommendations = []
        for i, raw in enumerate(raw_list):
            try:
                rec = Recommendation(
                    id=f"rec_{uuid.uuid4().hex[:8]}",
                    category=RecommendationCategory(raw.get("category", "hold")),
                    title=raw.get("title", "Untitled Recommendation"),
                    summary=raw.get("summary", ""),
                    detailed_rationale=raw.get("detailed_rationale", ""),
                    tickers=raw.get("tickers", []),
                    suggested_action=raw.get("suggested_action", ""),
                    suggested_allocation_pct=raw.get("suggested_allocation_pct"),
                    risk_level=RiskLevel(raw.get("risk_level", "moderate")),
                    expected_return_range=raw.get("expected_return_range"),
                    time_horizon=raw.get("time_horizon"),
                    confidence=ConfidenceLevel(raw.get("confidence", "medium")),
                    priority=RecommendationPriority(raw.get("priority", "medium")),
                    data_sources=[
                        DataSource(
                            source_type=s, source_name=s, relevance_score=0.5
                        )
                        for s in context.data_sources_used
                    ],
                    generated_at=datetime.now(),
                )
                recommendations.append(rec)
            except Exception as e:
                logger.warning(f"Failed to parse recommendation {i}: {e}")
                continue

        return recommendations
