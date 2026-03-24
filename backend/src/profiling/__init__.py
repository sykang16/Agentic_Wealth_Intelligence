"""Investment Profiling Module (Module B).

This module provides conversational slot-filling for building investment profiles.
"""

from backend.src.profiling.agent import ProfilingAgent, create_profiling_agent
from backend.src.profiling.schemas import (
    ConversationState,
    Message,
    ProfileSlot,
    ProfileSlots,
    SlotStatus,
)

__all__ = [
    "ProfilingAgent",
    "create_profiling_agent",
    "ConversationState",
    "Message",
    "ProfileSlot",
    "ProfileSlots",
    "SlotStatus",
]
