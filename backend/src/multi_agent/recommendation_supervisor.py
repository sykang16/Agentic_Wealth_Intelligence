"""Recommendation Supervisor Subgraph.

A LangGraph supervisor subgraph that orchestrates multi-agent data gathering
before synthesizing investment recommendations. Embedded as a single opaque
node in the outer orchestrator graph.

Flow:
  supervisor → portfolio_fetch → (back to supervisor)
  supervisor → profiling_fetch → (back to supervisor)
  supervisor → recommend       → (back to supervisor)
  supervisor → FINISH          → END (returns response/module_source to outer state)
"""

import logging
import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.src.agents.asset_agent import AssetAgent
from backend.src.common.llm_client import LLMClient
from backend.src.profiling import ProfilingAgent
from backend.src.recommendation.engine import (
    RecommendationEngine,
    RecommendationRequest,
)

logger = logging.getLogger(__name__)

# ── Iteration guard ────────────────────────────────────────────────────────────
MAX_SUPERVISOR_STEPS = 5


# ── Reducer helper ─────────────────────────────────────────────────────────────
def _append_list(left: list | None, right: list) -> list:
    """Safe list-append reducer that handles a None initial value."""
    if left is None:
        return list(right)
    return list(left) + list(right)


# ── State ──────────────────────────────────────────────────────────────────────
class SupervisorSubgraphState(TypedDict, total=False):
    """State for the recommendation supervisor subgraph.

    Fields shared with OrchestratorState (outer graph reads/writes these):
        user_id, current_message, profiling_state,
        response, module_source, visualization, error

    Fields internal to the subgraph (never propagated to outer graph):
        agent_scratchpad, supervisor_steps, next_agent
    """

    # ── read from outer state ──────────────────────────────────────────────
    user_id: str
    current_message: str          # outer graph's current_message == user query
    profiling_state: Any
    include_live_data: bool

    # ── internal accumulator (uses reducer to append entries) ──────────────
    agent_scratchpad: Annotated[list[dict], _append_list]
    supervisor_steps: int
    next_agent: str

    # ── written back to outer state ────────────────────────────────────────
    response: str
    module_source: str
    visualization: Any
    recommendation_response: Any  # full RecommendationResponse object for rich UI rendering
    error: str | None
    process_log: Annotated[list[dict], _append_list]


# ── Supervisor system prompt ───────────────────────────────────────────────────
SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor for a wealth management recommendation system.
Your job is to decide which agent to call next so the recommendation engine receives
the context it needs.

Available agents:
  portfolio_fetch  – retrieves current portfolio data (asset allocation, holdings, net worth)
  profiling_fetch  – retrieves investment profile (risk tolerance, goals, investment horizon)
  recommend        – generates personalized investment recommendations (call last)
  finish           – marks the task complete (call only AFTER recommend has run)

Decision rules:
  1. If the query mentions allocation, holdings, net worth, rebalance, or existing assets
     → call portfolio_fetch (if not already done)
  2. If the query mentions risk tolerance, investment tendency, goals, profile, or preferences
     → call profiling_fetch (if not already done)
  3. After gathering all needed context → call recommend
  4. After recommend has been called → call finish
  5. For simple queries with no cross-domain signals → call recommend directly

Respond with ONLY one word: portfolio_fetch, profiling_fetch, recommend, or finish"""


# ── Builder ────────────────────────────────────────────────────────────────────
def build_recommendation_supervisor(
    llm_client: LLMClient,
    asset_agent: AssetAgent,
    profiling_agent: ProfilingAgent,
    recommendation_engine: RecommendationEngine,
):
    """Build and compile the recommendation supervisor subgraph.

    Args:
        llm_client: LLM client used by the supervisor for routing decisions.
        asset_agent: AssetAgent for portfolio data retrieval.
        profiling_agent: ProfilingAgent for investment profile data.
        recommendation_engine: RecommendationEngine for final synthesis.

    Returns:
        Compiled LangGraph subgraph ready to be embedded as a node.
    """

    # ── Node: supervisor ───────────────────────────────────────────────────
    def supervisor_node(state: SupervisorSubgraphState) -> dict:
        """LLM-driven routing node. Decides which agent to call next."""
        steps = state.get("supervisor_steps", 0)

        # Iteration guard – prevent infinite loops
        if steps >= MAX_SUPERVISOR_STEPS:
            logger.warning("Supervisor reached max steps; forcing finish.")
            return {
                "next_agent": "finish",
                "supervisor_steps": steps + 1,
                "process_log": [{"step": steps + 1, "agent": "supervisor", "ts": datetime.now().isoformat(), "note": "Max steps reached — forcing finish"}],
            }

        user_query = state.get("current_message", "")
        scratchpad: list[dict] = state.get("agent_scratchpad") or []

        # Hard guard: never re-call an agent already in the scratchpad
        already_called = {entry["agent"] for entry in scratchpad}
        if "recommend" in already_called:
            return {"next_agent": "finish", "supervisor_steps": steps + 1}

        # Summarise scratchpad for the prompt
        if scratchpad:
            scratchpad_summary = "\n".join(
                f"  [{entry['agent']}]: {str(entry['result'])[:300]}"
                for entry in scratchpad
            )
        else:
            scratchpad_summary = "  (none yet)"

        user_message = (
            f"User query: {user_query}\n\n"
            f"Agents called so far:\n{scratchpad_summary}\n\n"
            "Which agent should run next?"
        )

        try:
            llm_response = llm_client.chat(
                user_message=user_message,
                system_prompt=SUPERVISOR_SYSTEM_PROMPT,
                max_tokens=20,
                temperature=0.0,
            )
            raw = llm_response.content.strip().lower().split()[0].rstrip(".,;")
            valid = {"portfolio_fetch", "profiling_fetch", "recommend", "finish"}
            # Disallow re-calling agents already in the scratchpad
            available = valid - already_called
            next_agent = raw if raw in available else "recommend"
        except Exception as exc:
            logger.warning("Supervisor LLM failed (%s); defaulting to recommend.", exc)
            next_agent = "recommend"

        return {
            "next_agent": next_agent,
            "supervisor_steps": steps + 1,
            "process_log": [{"step": steps + 1, "agent": "supervisor", "ts": datetime.now().isoformat(), "note": f"Routing to: {next_agent}"}],
        }

    # ── Node: portfolio_fetch ──────────────────────────────────────────────
    def portfolio_fetch_node(state: SupervisorSubgraphState) -> dict:
        """Fetch current portfolio data via AssetAgent."""
        user_id = state.get("user_id", "")
        query = state.get("current_message", "")

        try:
            result = asset_agent.process(user_id, query)
            result_text = result.answer
        except Exception as exc:
            logger.error("portfolio_fetch_node failed: %s", exc, exc_info=True)
            result_text = f"Error retrieving portfolio data: {exc}"

        return {
            "agent_scratchpad": [{"agent": "portfolio_fetch", "result": result_text}],
            "process_log": [{"step": None, "agent": "portfolio_fetch", "ts": datetime.now().isoformat(), "note": result_text[:120]}],
        }

    # ── Node: profiling_fetch ──────────────────────────────────────────────
    def profiling_fetch_node(state: SupervisorSubgraphState) -> dict:
        """Fetch investment profile from the existing profiling_state."""
        profiling_state = state.get("profiling_state")

        if profiling_state is None:
            result_text = "No investment profile found for this user."
        else:
            try:
                summary = profiling_agent.get_profile_summary(profiling_state)
                filled = summary.get("filled_slots", {})
                if filled:
                    lines = [
                        f"  - {display}: {info['value']}"
                        for display, info in filled.items()
                    ]
                    completion = summary.get("completion_percentage", 0)
                    result_text = (
                        f"Investment Profile ({completion:.0f}% complete):\n"
                        + "\n".join(lines)
                    )
                else:
                    result_text = (
                        "Investment profile exists but no slots are filled yet."
                    )
            except Exception as exc:
                logger.error("profiling_fetch_node failed: %s", exc, exc_info=True)
                result_text = f"Error retrieving investment profile: {exc}"

        return {
            "agent_scratchpad": [{"agent": "profiling_fetch", "result": result_text}],
            "process_log": [{"step": None, "agent": "profiling_fetch", "ts": datetime.now().isoformat(), "note": result_text[:120]}],
        }

    # ── Node: recommend_synthesis ──────────────────────────────────────────
    def recommend_synthesis_node(state: SupervisorSubgraphState) -> dict:
        """Synthesise recommendations using accumulated agent context."""
        user_id = state.get("user_id", "")
        user_query = state.get("current_message", "")
        scratchpad: list[dict] = state.get("agent_scratchpad") or []

        # Prefix the query with gathered context
        if scratchpad:
            context_sections = "\n\n".join(
                f"[{entry['agent']} output]\n{entry['result']}"
                for entry in scratchpad
            )
            enriched_query = (
                f"CONTEXT FROM AGENTS:\n{context_sections}\n\n"
                f"USER QUERY: {user_query}"
            )
        else:
            enriched_query = user_query

        try:
            request = RecommendationRequest(
                user_id=user_id,
                query=enriched_query,
                max_recommendations=5,
                include_live_data=state.get("include_live_data", True),
            )
            result = recommendation_engine.generate_recommendations(request)

            if result.success:
                parts = [result.summary]
                for i, rec in enumerate(result.recommendations, 1):
                    meta = []
                    if rec.expected_return_range:
                        meta.append(f"Expected return: {rec.expected_return_range}")
                    if rec.time_horizon:
                        meta.append(f"Horizon: {rec.time_horizon}")
                    meta.append(f"Risk: {rec.risk_level.value}")
                    meta.append(f"Confidence: {rec.confidence.value}")
                    meta_str = " | ".join(meta)

                    parts.append(
                        f"\n{i}. **{rec.title}**"
                        f" ({rec.category.value.replace('_', ' ').title()})"
                        f" — {rec.priority.value.upper()} priority\n"
                        f"   {rec.summary}\n"
                        f"\n   **Reasoning:** {rec.detailed_rationale}\n"
                        f"\n   **Action:** {rec.suggested_action}\n"
                        f"   {meta_str}"
                    )
                if result.warnings:
                    parts.append("\n---\n" + "\n".join(f"* {w}" for w in result.warnings))

                response_text = "\n".join(parts)
                synthesis_entry = {
                    "agent": "recommend",
                    "result": response_text[:300],
                }
                n = len(result.recommendations)
                sources = ", ".join(result.data_sources_used) if result.data_sources_used else "none"
                return {
                    "response": response_text,
                    "module_source": "Recommendation",
                    "recommendation_response": result,
                    "agent_scratchpad": [synthesis_entry],
                    "process_log": [{"step": None, "agent": "recommend", "ts": datetime.now().isoformat(), "note": f"Generated {n} recommendation(s). Data sources: {sources}"}],
                }
            else:
                error_text = f"I couldn't generate recommendations: {result.error}"
                return {
                    "response": error_text,
                    "module_source": "Recommendation",
                    "error": result.error,
                    "agent_scratchpad": [{"agent": "recommend", "result": error_text}],
                    "process_log": [{"step": None, "agent": "recommend", "ts": datetime.now().isoformat(), "note": f"Failed: {result.error}"}],
                }

        except Exception as exc:
            logger.error("recommend_synthesis_node failed: %s", exc, exc_info=True)
            error_text = f"I encountered an error generating recommendations: {exc}"
            return {
                "response": error_text,
                "module_source": "Recommendation",
                "error": str(exc),
                "agent_scratchpad": [{"agent": "recommend", "result": error_text}],
                "process_log": [{"step": None, "agent": "recommend", "ts": datetime.now().isoformat(), "note": f"Exception: {exc}"}],
            }

    # ── Assemble graph ─────────────────────────────────────────────────────
    graph = StateGraph(SupervisorSubgraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("portfolio_fetch", portfolio_fetch_node)
    graph.add_node("profiling_fetch", profiling_fetch_node)
    graph.add_node("recommend", recommend_synthesis_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next_agent", "recommend"),
        {
            "portfolio_fetch": "portfolio_fetch",
            "profiling_fetch": "profiling_fetch",
            "recommend": "recommend",
            "finish": "recommend",  # safety: treat stray "finish" as recommend
        },
    )

    graph.add_edge("portfolio_fetch", "supervisor")
    graph.add_edge("profiling_fetch", "supervisor")
    graph.add_edge("recommend", END)

    return graph.compile()
