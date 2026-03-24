"""Orchestrator state definitions for the multi-agent system."""

from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class UserIntent(str, Enum):
    """Intent categories for routing user messages."""

    PORTFOLIO_QUERY = "portfolio_query"
    PROFILING = "profiling"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


class OrchestratorState(TypedDict, total=False):
    """State for the LangGraph orchestrator.

    Uses TypedDict per LangGraph conventions for reducer-friendly state.
    """

    user_id: str
    messages: list[dict[str, str]]
    current_message: str
    intent: str
    response: str
    module_source: str
    visualization: Any  # Plotly figure from portfolio node
    recommendation_response: Any  # RecommendationResponse object from recommendation node
    profiling_state: Any
    error: str | None
    include_live_data: bool
    process_log: list[dict]  # step-by-step trace of agent calls


class OrchestratorResponse(BaseModel):
    """Public API response from the orchestrator."""

    response: str
    intent: str
    module_source: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    success: bool = True
