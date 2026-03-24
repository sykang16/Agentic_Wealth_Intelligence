"""Portfolio endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.src.agents.asset_agent import AssetAgent, AssetAgentResponse
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.api.dependencies import get_aggregator, get_llm_client
from backend.src.api.schemas import QueryRequest

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


@router.get("/{user_id}")
def get_portfolio_summary(
    user_id: str,
    aggregator: PortfolioAggregator = Depends(get_aggregator),
):
    """Get portfolio summary for a user."""
    portfolio = aggregator.get_portfolio(user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    metrics = aggregator.get_portfolio_metrics(user_id)
    allocation = aggregator.get_asset_allocation(user_id)
    top_holdings = aggregator.get_top_holdings(user_id, 5)

    return {
        "user_id": user_id,
        "name": portfolio.user.name,
        "metrics": metrics,
        "allocation": allocation,
        "top_holdings": [
            {
                "symbol": h.symbol,
                "name": h.name,
                "value": float(h.market_value),
                "gain_loss_pct": h.unrealized_gain_loss_percent,
            }
            for h in top_holdings
        ],
    }


@router.post("/{user_id}/query")
def query_portfolio(
    user_id: str,
    request: QueryRequest,
    aggregator: PortfolioAggregator = Depends(get_aggregator),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Process a natural language query about a user's portfolio."""
    agent = AssetAgent(aggregator, llm_client=llm_client)
    result = agent.process(user_id, request.query)
    return {
        "answer": result.answer,
        "query_type": result.query_type,
        "success": result.success,
        "error": result.error,
        "data": result.data,
    }
