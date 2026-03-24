"""Investment Profiling Agent using LangGraph.

This module implements the conversational slot-filling agent for building
investment profiles through natural dialogue.
"""

from typing import Any

from langgraph.graph import StateGraph, END

from backend.src.common.llm_client import LLMClient
from backend.src.common.models import InvestmentProfile
from backend.src.profiling.schemas import (
    ConversationState,
    Message,
    ProfileSlot,
    ProfileSlots,
    SlotStatus,
)
from backend.src.profiling.nodes.extractor import extract_profile_info
from backend.src.profiling.nodes.validator import validate_extracted_values
from backend.src.profiling.nodes.question_gen import (
    generate_greeting,
    generate_resume_greeting,
    generate_question,
    generate_completion_message,
    generate_clarification,
)
from backend.src.profiling.nodes.checker import check_completion, should_continue


class ProfilingAgent:
    """Agent for conversational investment profile collection.

    This agent uses a LangGraph workflow to:
    1. Greet the user and explain the profiling process
    2. Ask questions to fill profile slots
    3. Extract information from user responses
    4. Validate extracted values
    5. Generate follow-up questions until complete

    The workflow is:
    START -> greet -> ask_question -> (user response) ->
    extract -> validate -> check_completion ->
      - if incomplete: ask_question (loop)
      - if validation error: clarify -> (loop)
      - if complete: complete -> END
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize the profiling agent.

        Args:
            llm_client: Optional LLM client. Creates default if not provided.
        """
        self.llm = llm_client or LLMClient()
        self._graph = None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with state schema
        workflow = StateGraph(ConversationState)

        # Define node functions that use self.llm
        def greet_node(state: ConversationState) -> ConversationState:
            return generate_greeting(state, self.llm)

        def ask_question_node(state: ConversationState) -> ConversationState:
            return generate_question(state, self.llm)

        def extract_node(state: ConversationState) -> ConversationState:
            return extract_profile_info(state, self.llm)

        def validate_node(state: ConversationState) -> ConversationState:
            return validate_extracted_values(state, self.llm)

        def check_node(state: ConversationState) -> ConversationState:
            return check_completion(state)

        def clarify_node(state: ConversationState) -> ConversationState:
            return generate_clarification(state, self.llm)

        def complete_node(state: ConversationState) -> ConversationState:
            return generate_completion_message(state, self.llm)

        # Add nodes
        workflow.add_node("greet", greet_node)
        workflow.add_node("ask_question", ask_question_node)
        workflow.add_node("extract", extract_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("check_completion", check_node)
        workflow.add_node("clarify", clarify_node)
        workflow.add_node("complete", complete_node)

        # Set entry point
        workflow.set_entry_point("greet")

        # Add edges
        workflow.add_edge("greet", "ask_question")

        # After asking, we wait for user input (this edge won't be auto-traversed)
        # The user input handling is external

        # After extraction, validate
        workflow.add_edge("extract", "validate")

        # After validation, check completion
        workflow.add_edge("validate", "check_completion")

        # Conditional routing after completion check
        workflow.add_conditional_edges(
            "check_completion",
            should_continue,
            {
                "ask_question": "ask_question",
                "clarify": "clarify",
                "complete": "complete",
            },
        )

        # After clarification, wait for user (external handling)
        # Clarify loops back to waiting for input

        # Complete ends the workflow
        workflow.add_edge("complete", END)

        return workflow

    def get_graph(self) -> StateGraph:
        """Get the compiled workflow graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def start_conversation(self, user_id: str, user_name: str = "there") -> ConversationState:
        """Start a new profiling conversation.

        Args:
            user_id: User ID to profile.
            user_name: User's name for personalization.

        Returns:
            Initial conversation state with greeting.
        """
        # Create initial state
        state = ConversationState(user_id=user_id)

        # Generate greeting with first question
        state = generate_greeting(state, self.llm, user_name)

        return state

    def resume_conversation(
        self,
        user_id: str,
        user_name: str,
        profile: InvestmentProfile,
    ) -> ConversationState:
        """Resume a profiling conversation with pre-populated slots from an existing profile.

        Args:
            user_id: User ID to profile.
            user_name: User's name for personalization.
            profile: Existing investment profile to pre-populate from.

        Returns:
            Conversation state with filled slots and a resume greeting.
        """
        state = ConversationState(user_id=user_id)

        # Map InvestmentProfile fields into ProfileSlots
        field_mapping: list[tuple[str, Any]] = []

        if profile.risk_tolerance is not None:
            field_mapping.append(("risk_tolerance", profile.risk_tolerance.value))
        if profile.investment_horizon is not None:
            field_mapping.append(("investment_horizon", profile.investment_horizon.value))
        if profile.investment_experience is not None:
            field_mapping.append(("experience_level", profile.investment_experience.value))
        if profile.liquidity_needs is not None:
            field_mapping.append(("liquidity_needs", profile.liquidity_needs.value))
        if profile.esg_preference is not None:
            field_mapping.append(("esg_preference", profile.esg_preference))
        if profile.goals:
            field_mapping.append(("primary_goal", profile.goals[0].goal_type.value))
        if profile.loss_comfort is not None:
            field_mapping.append(("loss_comfort", profile.loss_comfort))
        if profile.income_stability is not None:
            field_mapping.append(("income_stability", profile.income_stability))
        if profile.has_emergency_fund is not None:
            field_mapping.append(("has_emergency_fund", profile.has_emergency_fund))
        if profile.debt_level is not None:
            field_mapping.append(("debt_level", profile.debt_level))
        if profile.monthly_income is not None:
            field_mapping.append(("monthly_income", float(profile.monthly_income)))
        if profile.monthly_expenses is not None:
            field_mapping.append(("monthly_expenses", float(profile.monthly_expenses)))

        for slot_name, value in field_mapping:
            slot = getattr(state.slots, slot_name)
            slot.value = value
            slot.status = SlotStatus.FILLED

        # Calculate initial completion percentage
        state.completion_percentage = state.calculate_completion()

        # Generate resume greeting
        state = generate_resume_greeting(state, self.llm, user_name)

        return state

    def process_response(self, state: ConversationState, user_message: str) -> ConversationState:
        """Process a user response and generate the next question or completion.

        Args:
            state: Current conversation state.
            user_message: User's message.

        Returns:
            Updated conversation state.
        """
        # Add user message to history
        state.messages.append(Message(role="user", content=user_message))

        # Extract information
        state = extract_profile_info(state, self.llm)

        # Validate
        state = validate_extracted_values(state, self.llm)

        # Check completion
        state = check_completion(state)

        # Route based on state
        next_action = should_continue(state)

        if next_action == "complete":
            state = generate_completion_message(state, self.llm)
        elif next_action == "clarify":
            state = generate_clarification(state, self.llm)
        else:  # ask_question
            state = generate_question(state, self.llm)

        return state

    def get_last_assistant_message(self, state: ConversationState) -> str | None:
        """Get the most recent assistant message from the state.

        Args:
            state: Conversation state.

        Returns:
            Last assistant message or None.
        """
        for msg in reversed(state.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def get_profile_summary(self, state: ConversationState) -> dict[str, Any]:
        """Get a summary of the collected profile.

        Args:
            state: Conversation state.

        Returns:
            Profile summary dict.
        """
        filled = {}
        missing = []

        for field_name in ProfileSlots.model_fields:
            slot = getattr(state.slots, field_name)
            if slot.value is not None:
                filled[slot.display_name] = {
                    "value": slot.value,
                    "name": field_name,
                }
            elif slot.required:
                missing.append(slot.display_name)

        return {
            "user_id": state.user_id,
            "filled_slots": filled,
            "missing_required": missing,
            "completion_percentage": state.completion_percentage,
            "is_complete": state.is_complete,
            "conversation_turns": len([m for m in state.messages if m.role == "user"]),
        }

    def export_profile(self, state: ConversationState) -> dict[str, Any]:
        """Export the filled profile as a dictionary.

        Args:
            state: Conversation state.

        Returns:
            Profile data ready for storage.
        """
        return state.to_investment_profile()


def create_profiling_agent(llm_client: LLMClient | None = None) -> ProfilingAgent:
    """Factory function to create a profiling agent.

    Args:
        llm_client: Optional LLM client.

    Returns:
        Configured ProfilingAgent instance.
    """
    return ProfilingAgent(llm_client)
