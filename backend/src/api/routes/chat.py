"""Unified chat endpoint powered by the orchestrator."""

import logging

from fastapi import APIRouter, Depends

from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.api.dependencies import get_log_repository, get_orchestrator
from backend.src.api.schemas import ChatRequest, ChatResponse
from backend.src.logging_db import LogRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    orchestrator: WealthOrchestrator = Depends(get_orchestrator),
    log_repo: LogRepository = Depends(get_log_repository),
):
    """Process a message through the multi-agent orchestrator."""
    # Restore persisted state if a session_id is provided
    session_id = request.session_id
    if session_id:
        persisted = log_repo.load_session_state(session_id)
        if persisted:
            state = persisted
        else:
            state = orchestrator.create_initial_state(request.user_id)
    else:
        state = orchestrator.create_initial_state(request.user_id)

    result = orchestrator.process_message(request.message, state)
    resp = orchestrator.get_response(result)

    # Log interaction and persist state (non-critical)
    try:
        if not session_id:
            session_id = log_repo.create_session(request.user_id)
        log_repo.log_interaction(
            session_id=session_id,
            user_message=request.message,
            response_content=resp.response,
            intent=resp.intent,
            module_source=resp.module_source,
            error=resp.error,
        )
        log_repo.save_session_state(session_id, result)
    except Exception:
        logger.exception("Failed to log interaction")

    return ChatResponse(
        response=resp.response,
        intent=resp.intent,
        module_source=resp.module_source,
        success=resp.success,
        error=resp.error,
        session_id=session_id,
    )
