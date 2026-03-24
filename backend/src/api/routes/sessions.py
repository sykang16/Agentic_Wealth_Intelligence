"""Session listing and detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.dependencies import get_log_repository
from backend.src.logging_db import LogRepository
from backend.src.logging_db.schemas import SessionDetailResponse, SessionListResponse

router = APIRouter(prefix="/api/v1", tags=["sessions"])


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    user_id: str | None = None,
    limit: int = 50,
    log_repo: LogRepository = Depends(get_log_repository),
):
    """List recent sessions, optionally filtered by user_id."""
    sessions = log_repo.list_sessions(user_id=user_id, limit=limit)
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    log_repo: LogRepository = Depends(get_log_repository),
):
    """Get full session detail with all interaction logs."""
    detail = log_repo.get_session_with_interactions(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetailResponse(**detail)
