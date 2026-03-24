"""Pydantic response models for session/interaction endpoints."""

from pydantic import BaseModel, Field


class InteractionLogResponse(BaseModel):
    """A single interaction log entry."""

    id: int
    node_name: str = ""
    user_message: str = ""
    response_content: str = ""
    intent: str = ""
    module_source: str = ""
    error: str | None = None
    created_at: str | None = None


class SessionListItem(BaseModel):
    """Summary item for session listing."""

    id: str
    user_id: str
    created_at: str | None = None
    updated_at: str | None = None
    interaction_count: int = 0


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[SessionListItem] = Field(default_factory=list)


class SessionDetailResponse(BaseModel):
    """Full session detail including interactions."""

    id: str
    user_id: str
    created_at: str | None = None
    updated_at: str | None = None
    state_json: str | None = None
    interactions: list[InteractionLogResponse] = Field(default_factory=list)
