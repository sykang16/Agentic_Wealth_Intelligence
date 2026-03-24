"""API request/response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request for the unified chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    user_id: str = Field(..., description="User ID")
    session_id: str | None = Field(default=None, description="Session ID for multi-turn conversations")


class ChatResponse(BaseModel):
    """Response from the unified chat endpoint."""

    response: str
    intent: str
    module_source: str
    success: bool = True
    error: str | None = None
    session_id: str | None = None


class QueryRequest(BaseModel):
    """Request for portfolio query endpoint."""

    query: str = Field(..., min_length=1, description="Natural language query")


class ProfilingStartRequest(BaseModel):
    """Request to start a profiling session."""

    user_name: str = Field(default="there", description="User's display name")


class ProfilingRespondRequest(BaseModel):
    """Request to continue a profiling conversation."""

    message: str = Field(..., min_length=1, description="User's response")


class ProfilingResponse(BaseModel):
    """Response from profiling endpoints."""

    assistant_message: str
    completion_percentage: float
    is_complete: bool
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)


class RecommendationGenerateRequest(BaseModel):
    """Request to generate recommendations."""

    user_id: str = Field(..., description="User ID")
    query: str = Field(default="", description="Optional query for targeted recommendations")
    max_recommendations: int = Field(default=5, ge=1, le=10)
    include_live_data: bool = Field(default=False)
