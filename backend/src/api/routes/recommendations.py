"""Recommendation endpoints."""

from fastapi import APIRouter, Depends

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.recommendation.engine import RecommendationEngine, RecommendationRequest
from backend.src.recommendation.rag.initializer import RAGInitializer
from backend.src.api.dependencies import get_aggregator, get_llm_client, get_rag
from backend.src.api.schemas import RecommendationGenerateRequest

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


@router.post("/generate")
def generate_recommendations(
    request: RecommendationGenerateRequest,
    aggregator: PortfolioAggregator = Depends(get_aggregator),
    llm_client: LLMClient = Depends(get_llm_client),
    rag: RAGInitializer = Depends(get_rag),
):
    """Generate personalized investment recommendations."""
    engine = RecommendationEngine(
        portfolio_aggregator=aggregator,
        rag_initializer=rag,
        llm_client=llm_client,
    )
    rec_request = RecommendationRequest(
        user_id=request.user_id,
        query=request.query,
        max_recommendations=request.max_recommendations,
        include_live_data=request.include_live_data,
    )
    response = engine.generate_recommendations(rec_request)
    return response.model_dump(mode="json")
