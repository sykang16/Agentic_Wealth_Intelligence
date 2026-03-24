"""Profiling endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.src.common.llm_client import LLMClient
from backend.src.profiling import ProfilingAgent
from backend.src.api.dependencies import get_llm_client
from backend.src.api.schemas import (
    ProfilingRespondRequest,
    ProfilingResponse,
    ProfilingStartRequest,
)

router = APIRouter(prefix="/api/v1/profiling", tags=["profiling"])

# In-memory session store (per the plan: full persistence deferred)
_profiling_sessions: dict[str, dict] = {}


@router.post("/{user_id}/start", response_model=ProfilingResponse)
def start_profiling(
    user_id: str,
    request: ProfilingStartRequest = ProfilingStartRequest(),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Start or restart a profiling conversation."""
    agent = ProfilingAgent(llm_client=llm_client)
    state = agent.start_conversation(user_id, request.user_name)
    summary = agent.get_profile_summary(state)

    _profiling_sessions[user_id] = {"agent": agent, "state": state}

    assistant_msg = agent.get_last_assistant_message(state) or ""
    return ProfilingResponse(
        assistant_message=assistant_msg,
        completion_percentage=summary["completion_percentage"],
        is_complete=summary["is_complete"],
        filled_slots=summary["filled_slots"],
        missing_required=summary["missing_required"],
    )


@router.post("/{user_id}/respond", response_model=ProfilingResponse)
def respond_profiling(
    user_id: str,
    request: ProfilingRespondRequest,
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Process a user response in an ongoing profiling conversation."""
    session = _profiling_sessions.get(user_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"No active profiling session for user {user_id}. Call /start first.",
        )

    agent = session["agent"]
    state = session["state"]

    state = agent.process_response(state, request.message)
    summary = agent.get_profile_summary(state)

    _profiling_sessions[user_id]["state"] = state

    assistant_msg = agent.get_last_assistant_message(state) or ""
    return ProfilingResponse(
        assistant_message=assistant_msg,
        completion_percentage=summary["completion_percentage"],
        is_complete=summary["is_complete"],
        filled_slots=summary["filled_slots"],
        missing_required=summary["missing_required"],
    )
