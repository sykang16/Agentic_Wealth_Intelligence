"""High-level orchestrator wrapping the LangGraph multi-agent system."""

import logging
from typing import Any

from backend.src.agents.asset_agent import AssetAgent
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.multi_agent.graph import build_orchestrator_graph
from backend.src.multi_agent.state import OrchestratorResponse, OrchestratorState
from backend.src.profiling import ProfilingAgent
from backend.src.recommendation.engine import RecommendationEngine
from backend.src.recommendation.rag.initializer import RAGInitializer

logger = logging.getLogger(__name__)


class WealthOrchestrator:
    """High-level API for the multi-agent wealth management system.

    Wraps the LangGraph orchestrator graph and provides simple
    process_message / create_initial_state methods.
    """

    def __init__(
        self,
        aggregator: PortfolioAggregator,
        llm_client: LLMClient,
        rag: RAGInitializer | None = None,
    ):
        self.aggregator = aggregator
        self.llm_client = llm_client

        # Build module agents
        self._asset_agent = AssetAgent(aggregator, llm_client=llm_client)
        self._profiling_agent = ProfilingAgent(llm_client=llm_client)
        _rag = rag or RAGInitializer()
        from backend.src.mcp.integration.hybrid_data_provider import HybridDataProvider  # noqa: PLC0415
        _hybrid_provider = HybridDataProvider(
            rag_initializer=_rag,
            portfolio_aggregator=aggregator,
        )
        self._recommendation_engine = RecommendationEngine(
            portfolio_aggregator=aggregator,
            rag_initializer=_rag,
            llm_client=llm_client,
            hybrid_data_provider=_hybrid_provider,
        )

        # Build and compile graph
        graph = build_orchestrator_graph(
            llm_client=llm_client,
            asset_agent=self._asset_agent,
            profiling_agent=self._profiling_agent,
            recommendation_engine=self._recommendation_engine,
            aggregator=aggregator,
        )
        self._compiled = graph.compile()

    def create_initial_state(self, user_id: str) -> OrchestratorState:
        """Create a fresh orchestrator state for a user."""
        return OrchestratorState(
            user_id=user_id,
            messages=[],
            current_message="",
            intent="",
            response="",
            module_source="",
            profiling_state=None,
            error=None,
            include_live_data=True,
            process_log=[],
        )

    def process_message(
        self,
        message: str,
        state: OrchestratorState,
        include_live_data: bool | None = None,
    ) -> OrchestratorState:
        """Process a user message through the orchestrator graph.

        Args:
            message: User's message text.
            state: Current orchestrator state (from create_initial_state or previous call).

        Returns:
            Updated OrchestratorState with response populated.
        """
        # Set the current message, live data preference, and reset trace log
        state["current_message"] = message
        state["process_log"] = []
        if include_live_data is not None:
            state["include_live_data"] = include_live_data

        try:
            result = self._compiled.invoke(state)
            return result
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            # Return state with error
            messages = list(state.get("messages", []))
            messages.append({"role": "user", "content": message})
            messages.append({
                "role": "assistant",
                "content": f"I'm sorry, an error occurred: {e}",
            })
            state["messages"] = messages
            state["response"] = f"I'm sorry, an error occurred: {e}"
            state["error"] = str(e)
            state["module_source"] = "Error"
            return state

    def get_response(self, state: OrchestratorState) -> OrchestratorResponse:
        """Extract an OrchestratorResponse from the current state."""
        # Strip non-serializable fields (e.g. Plotly figures) for the API model
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in state.get("messages", [])
        ]
        return OrchestratorResponse(
            response=state.get("response", ""),
            intent=state.get("intent", ""),
            module_source=state.get("module_source", ""),
            messages=api_messages,
            error=state.get("error"),
            success=state.get("error") is None,
        )
