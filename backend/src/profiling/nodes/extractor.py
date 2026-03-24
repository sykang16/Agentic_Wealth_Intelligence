"""Extractor node for parsing user responses and filling profile slots."""

import json
import re
from typing import Any

from backend.src.common.llm_client import LLMClient
from backend.src.profiling.prompts.extractor_prompt import EXTRACTOR_PROMPT
from backend.src.profiling.schemas import ConversationState, ProfileSlots, SlotStatus


def _format_slot_status(state: ConversationState) -> str:
    """Format current slot status for the prompt."""
    lines = []
    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        status_str = f"{'[FILLED]' if slot.status != SlotStatus.EMPTY else '[EMPTY]'}"
        required_str = "(required)" if slot.required else "(optional)"
        value_str = f" = {slot.value}" if slot.value is not None else ""
        lines.append(f"- {slot.display_name} {status_str} {required_str}{value_str}")
    return "\n".join(lines)


def _get_current_question(state: ConversationState) -> str:
    """Get the current question being asked."""
    if state.current_slot:
        slot = getattr(state.slots, state.current_slot, None)
        if slot:
            return slot.question or f"Tell me about your {slot.display_name}"
    return "General question about your investment preferences"


def _parse_extraction_response(response: str) -> dict[str, Any]:
    """Parse the LLM's extraction response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If no valid JSON, return empty dict
    return {}


def extract_profile_info(state: ConversationState, llm_client: LLMClient) -> ConversationState:
    """Extract profile information from the user's latest message.

    This node:
    1. Analyzes the user's most recent message
    2. Extracts any profile information mentioned
    3. Updates the state with extracted values

    Args:
        state: Current conversation state.
        llm_client: LLM client for extraction.

    Returns:
        Updated conversation state with extraction results.
    """
    # Get the last user message
    user_messages = [m for m in state.messages if m.role == "user"]
    if not user_messages:
        return state

    last_user_message = user_messages[-1].content

    # Format the prompt
    prompt = EXTRACTOR_PROMPT.format(
        slot_status=_format_slot_status(state),
        current_question=_get_current_question(state),
        user_message=last_user_message,
    )

    # Get extraction from LLM
    response = llm_client.chat(
        user_message=prompt,
        system_prompt="You are a precise information extractor. Return only valid JSON.",
        temperature=0.1,  # Low temperature for consistent extraction
    )

    # Parse the extraction
    extracted = _parse_extraction_response(response.content)

    # Store the extraction result
    state.last_extraction = extracted

    return state


def apply_extraction_to_state(state: ConversationState) -> ConversationState:
    """Apply validated extraction results to the state slots.

    This should be called after validation to update slot values.

    Args:
        state: Conversation state with validated extraction.

    Returns:
        Updated state with filled slots.
    """
    if not state.last_extraction:
        return state

    for slot_name, value in state.last_extraction.items():
        if value is None:
            continue

        # Check if this is a valid slot
        if not hasattr(state.slots, slot_name):
            continue

        slot = getattr(state.slots, slot_name)

        # Update the slot
        slot.value = value
        slot.status = SlotStatus.FILLED

    # Update completion percentage
    state.completion_percentage = state.calculate_completion()

    return state
