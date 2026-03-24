"""Completion checker node for determining if the profile is complete."""

from backend.src.profiling.schemas import ConversationState, ProfileSlots, SlotStatus


def check_completion(state: ConversationState) -> ConversationState:
    """Check if the investment profile is complete.

    This node:
    1. Checks if all required slots are filled
    2. Updates completion status
    3. Determines next action based on state

    Args:
        state: Current conversation state.

    Returns:
        Updated state with completion status.
    """
    # Count filled required slots
    required_filled = 0
    total_required = 0

    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        if slot.required:
            total_required += 1
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                required_filled += 1

    # Calculate completion percentage
    if total_required > 0:
        state.completion_percentage = round((required_filled / total_required) * 100, 1)
    else:
        state.completion_percentage = 100.0

    # Check if all required slots are filled
    all_required_filled = required_filled == total_required

    # Check iteration limit
    if state.iteration_count >= state.max_iterations:
        state.is_complete = True
        state.next_action = "complete"
        return state

    # Determine next action
    if all_required_filled:
        state.is_complete = True
        state.next_action = "complete"
    elif state.validation_errors:
        state.next_action = "clarify"
    else:
        state.next_action = "ask_question"

    return state


def should_continue(state: ConversationState) -> str:
    """Determine if the conversation should continue.

    This is used as a conditional edge in the LangGraph workflow.

    Args:
        state: Current conversation state.

    Returns:
        Next node to route to: "ask_question", "clarify", or "complete"
    """
    if state.is_complete:
        return "complete"

    if state.iteration_count >= state.max_iterations:
        return "complete"

    if state.validation_errors:
        return "clarify"

    # Check if there are remaining slots
    remaining = state.get_missing_required_slots()
    if not remaining:
        return "complete"

    return "ask_question"


def get_profile_summary(state: ConversationState) -> dict:
    """Get a summary of the filled profile.

    Args:
        state: Current conversation state.

    Returns:
        Dict with profile summary.
    """
    filled_slots = {}
    missing_slots = []

    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
            filled_slots[slot.display_name] = slot.value
        elif slot.required:
            missing_slots.append(slot.display_name)

    return {
        "filled": filled_slots,
        "missing": missing_slots,
        "completion_percentage": state.completion_percentage,
        "is_complete": state.is_complete,
        "total_messages": len(state.messages),
    }
