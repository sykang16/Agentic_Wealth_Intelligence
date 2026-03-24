"""Question generator node for creating contextual follow-up questions."""

from backend.src.common.llm_client import LLMClient
from backend.src.profiling.prompts.question_prompts import (
    GREETING_PROMPT,
    RESUME_GREETING_PROMPT,
    QUESTION_GENERATOR_PROMPT,
    COMPLETION_PROMPT,
)
from backend.src.profiling.schemas import ConversationState, Message, ProfileSlots, SlotStatus


def _format_conversation_history(state: ConversationState, max_messages: int = 10) -> str:
    """Format recent conversation history for the prompt."""
    if not state.messages:
        return "No previous conversation."

    recent = state.messages[-max_messages:]
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _format_filled_slots(state: ConversationState) -> str:
    """Format filled slots for the prompt."""
    filled = []
    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
            filled.append(f"- {slot.display_name}: {slot.value}")
    return "\n".join(filled) if filled else "None yet"


def _format_remaining_slots(state: ConversationState) -> str:
    """Format remaining slots for the prompt."""
    remaining = []
    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        if slot.status == SlotStatus.EMPTY:
            req = "(required)" if slot.required else "(optional)"
            remaining.append(f"- {slot.display_name} {req}")
    return "\n".join(remaining) if remaining else "All slots filled!"


def generate_greeting(
    state: ConversationState,
    llm_client: LLMClient,
    user_name: str = "there",
) -> ConversationState:
    """Generate an initial greeting to start the profiling conversation.

    Args:
        state: Current conversation state.
        llm_client: LLM client for generation.
        user_name: User's name for personalization.

    Returns:
        Updated state with greeting message.
    """
    prompt = GREETING_PROMPT.format(user_name=user_name)

    response = llm_client.chat(
        user_message=prompt,
        system_prompt="You are a friendly, professional financial advisor assistant.",
        temperature=0.7,
    )

    # Add greeting to messages
    state.messages.append(
        Message(role="assistant", content=response.content)
    )

    # Set the first slot to ask about
    next_slot = state.get_next_empty_slot()
    if next_slot:
        state.current_slot = next_slot.name

    state.next_action = "wait_for_response"

    return state


def generate_resume_greeting(
    state: ConversationState,
    llm_client: LLMClient,
    user_name: str = "there",
) -> ConversationState:
    """Generate a greeting that acknowledges already-filled slots and asks the first missing one.

    Args:
        state: Conversation state with pre-populated slots.
        llm_client: LLM client for generation.
        user_name: User's name for personalization.

    Returns:
        Updated state with resume greeting message.
    """
    prompt = RESUME_GREETING_PROMPT.format(
        user_name=user_name,
        filled_slots=_format_filled_slots(state),
        remaining_slots=_format_remaining_slots(state),
    )

    response = llm_client.chat(
        user_message=prompt,
        system_prompt="You are a friendly, professional financial advisor assistant.",
        temperature=0.7,
    )

    # Add greeting to messages
    state.messages.append(
        Message(role="assistant", content=response.content)
    )

    # Set the first empty slot to ask about
    next_slot = state.get_next_empty_slot()
    if next_slot:
        state.current_slot = next_slot.name

    state.next_action = "wait_for_response"

    return state


def generate_question(
    state: ConversationState,
    llm_client: LLMClient,
) -> ConversationState:
    """Generate the next question based on conversation context.

    This node:
    1. Identifies the next slot to fill
    2. Considers conversation context
    3. Generates a natural follow-up question

    Args:
        state: Current conversation state.
        llm_client: LLM client for generation.

    Returns:
        Updated state with new question.
    """
    # Check if profile is complete
    if state.is_complete:
        return state

    # Get next slot to fill
    next_slot = state.get_next_empty_slot()

    if not next_slot:
        # All slots filled, mark complete
        state.is_complete = True
        state.next_action = "complete"
        return state

    # For the first question or if no context, use the default question
    if len(state.messages) <= 1:
        question = next_slot.question or f"Could you tell me about your {next_slot.display_name}?"
    else:
        # Generate contextual question using LLM
        prompt = QUESTION_GENERATOR_PROMPT.format(
            conversation_history=_format_conversation_history(state),
            filled_slots=_format_filled_slots(state),
            remaining_slots=_format_remaining_slots(state),
            completion_percentage=state.completion_percentage,
            next_slot_name=next_slot.display_name,
            next_slot_description=next_slot.description,
            default_question=next_slot.question,
        )

        response = llm_client.chat(
            user_message=prompt,
            system_prompt="You are a friendly financial advisor having a natural conversation.",
            temperature=0.7,
        )

        question = response.content.strip()

    # Add question to messages
    state.messages.append(Message(role="assistant", content=question))

    # Update state
    state.current_slot = next_slot.name
    state.next_action = "wait_for_response"
    state.iteration_count += 1

    return state


def generate_completion_message(
    state: ConversationState,
    llm_client: LLMClient,
    user_name: str = "there",
) -> ConversationState:
    """Generate a completion message when profiling is done.

    Args:
        state: Current conversation state.
        llm_client: LLM client for generation.
        user_name: User's name for personalization.

    Returns:
        Updated state with completion message.
    """
    # Format profile summary
    profile_lines = []
    for field_name in ProfileSlots.model_fields:
        slot = getattr(state.slots, field_name)
        if slot.value is not None:
            profile_lines.append(f"- {slot.display_name}: {slot.value}")

    profile_summary = "\n".join(profile_lines)

    prompt = COMPLETION_PROMPT.format(
        user_name=user_name,
        profile_summary=profile_summary,
    )

    response = llm_client.chat(
        user_message=prompt,
        system_prompt="You are a friendly financial advisor wrapping up a conversation.",
        temperature=0.7,
    )

    # Add completion message
    state.messages.append(Message(role="assistant", content=response.content))

    state.is_complete = True
    state.next_action = "done"

    return state


def generate_clarification(
    state: ConversationState,
    llm_client: LLMClient,
) -> ConversationState:
    """Generate a clarification request when validation fails.

    Args:
        state: Current conversation state with validation errors.
        llm_client: LLM client for generation.

    Returns:
        Updated state with clarification request.
    """
    if not state.validation_errors:
        return state

    # Get the current slot being filled
    current_slot = None
    if state.current_slot and hasattr(state.slots, state.current_slot):
        current_slot = getattr(state.slots, state.current_slot)

    if not current_slot:
        # Default clarification
        clarification = "I didn't quite catch that. Could you please rephrase your answer?"
    else:
        # Generate specific clarification
        from backend.src.profiling.schemas import SLOT_VALIDATION_RULES

        rules = SLOT_VALIDATION_RULES.get(state.current_slot, {})
        valid_options = rules.get("values", [])

        if valid_options:
            options_str = ", ".join(valid_options)
            clarification = (
                f"I understand, but I need to know specifically about your {current_slot.display_name}. "
                f"Would you say it's {options_str}?"
            )
        else:
            clarification = (
                f"Could you clarify your answer about {current_slot.display_name}? "
                f"{current_slot.description}"
            )

    # Add clarification to messages
    state.messages.append(Message(role="assistant", content=clarification))

    # Clear validation errors
    state.validation_errors = []
    state.next_action = "wait_for_response"

    return state
