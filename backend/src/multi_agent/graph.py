"""LangGraph orchestrator graph construction."""

from langgraph.graph import END, StateGraph

from backend.src.agents.asset_agent import AssetAgent
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.profiling import ProfilingAgent
from backend.src.recommendation.engine import RecommendationEngine

from .nodes import (
    create_general_node,
    create_portfolio_node,
    create_profiling_node,
    create_router_node,
    respond_node,
)
from .recommendation_supervisor import build_recommendation_supervisor
from .state import OrchestratorState, UserIntent


def route_by_intent(state: OrchestratorState) -> str:
    """Route to the appropriate module node based on classified intent."""
    intent = state.get("intent", UserIntent.GENERAL.value)
    mapping = {
        UserIntent.PORTFOLIO_QUERY.value: "portfolio",
        UserIntent.PROFILING.value: "profiling",
        UserIntent.RECOMMENDATION.value: "recommend",
        UserIntent.GENERAL.value: "general",
    }
    return mapping.get(intent, "general")


def build_orchestrator_graph(
    llm_client: LLMClient,
    asset_agent: AssetAgent,
    profiling_agent: ProfilingAgent,
    recommendation_engine: RecommendationEngine,
    aggregator: PortfolioAggregator,
) -> StateGraph:
    """Build the orchestrator StateGraph.

    Args:
        llm_client: LLM client for routing and general responses.
        asset_agent: AssetAgent instance for portfolio queries.
        profiling_agent: ProfilingAgent instance for profiling.
        recommendation_engine: RecommendationEngine for recommendations.
        aggregator: PortfolioAggregator used to persist completed profiles.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Build the recommendation supervisor subgraph (embedded as an opaque node)
    rec_supervisor = build_recommendation_supervisor(
        llm_client=llm_client,
        asset_agent=asset_agent,
        profiling_agent=profiling_agent,
        recommendation_engine=recommendation_engine,
    )

    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("router", create_router_node(llm_client))
    graph.add_node("portfolio", create_portfolio_node(asset_agent))
    graph.add_node("profiling", create_profiling_node(profiling_agent, aggregator))
    graph.add_node("recommend", rec_supervisor)  # supervisor subgraph
    graph.add_node("general", create_general_node(llm_client))
    graph.add_node("respond", respond_node)

    # Set entry point
    graph.set_entry_point("router")

    # Conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "portfolio": "portfolio",
            "profiling": "profiling",
            "recommend": "recommend",
            "general": "general",
        },
    )

    # Fixed edges from module nodes to respond
    graph.add_edge("portfolio", "respond")
    graph.add_edge("profiling", "respond")
    graph.add_edge("recommend", "respond")
    graph.add_edge("general", "respond")

    # End after respond
    graph.add_edge("respond", END)

    return graph
