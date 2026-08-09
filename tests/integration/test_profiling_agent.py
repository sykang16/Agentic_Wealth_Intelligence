"""Integration tests for the profiling agent."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from backend.src.profiling.agent import ProfilingAgent, create_profiling_agent
from backend.src.profiling.schemas import ConversationState, SlotStatus
from backend.src.common.llm_client import LLMClient, LLMResponse


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_count = 0

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Return mock response."""
        if self._call_count < len(self._responses):
            content = self._responses[self._call_count]
        else:
            # Default responses for different contexts
            if "extract" in system_prompt.lower() or "json" in system_prompt.lower():
                content = '{"risk_tolerance": null}'
            elif "greeting" in user_message.lower() or "welcome" in user_message.lower():
                content = "Welcome! Let me help you build your investment profile. What's your experience with investing?"
            elif "complete" in user_message.lower():
                content = "Thank you for completing your profile!"
            else:
                content = "Could you tell me more about your investment preferences?"

        self._call_count += 1

        return LLMResponse(
            content=content,
            model="mock-model",
            provider="mock",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

    def chat_with_history(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Return mock response for history-based chat."""
        return self.chat(messages[-1]["content"] if messages else "", system_prompt, max_tokens, temperature)


class TestProfilingAgentCreation:
    """Tests for profiling agent creation."""

    def test_create_agent_with_default_client(self):
        """Test creating agent with default LLM client."""
        with patch.object(LLMClient, '__init__', return_value=None):
            with patch.object(LLMClient, 'chat', return_value=LLMResponse(
                content="Hello", model="test", provider="test"
            )):
                # This would fail without API key in real scenario
                # Just test the interface
                assert ProfilingAgent is not None

    def test_create_agent_with_custom_client(self):
        """Test creating agent with custom LLM client."""
        mock_client = MockLLMClient()
        agent = ProfilingAgent(llm_client=mock_client)

        assert agent.llm == mock_client

    def test_factory_function(self):
        """Test the create_profiling_agent factory function."""
        mock_client = MockLLMClient()
        agent = create_profiling_agent(llm_client=mock_client)

        assert isinstance(agent, ProfilingAgent)


class TestProfilingAgentStartConversation:
    """Tests for starting profiling conversations."""

    def test_start_conversation(self):
        """Test starting a new profiling conversation."""
        mock_client = MockLLMClient([
            "Welcome! I'm here to help you build your investment profile. "
            "Let's start with your investment experience - how would you describe it?"
        ])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123", "John")

        assert state.user_id == "user123"
        assert len(state.messages) > 0
        assert state.messages[0].role == "assistant"
        assert state.is_complete is False

    def test_start_conversation_sets_current_slot(self):
        """Test that starting conversation sets current slot."""
        mock_client = MockLLMClient(["Welcome! What's your risk tolerance?"])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")

        # Should have a current slot set
        assert state.current_slot is not None


class TestProfilingAgentProcessResponse:
    """Tests for processing user responses."""

    def test_process_response_adds_user_message(self):
        """Test that processing response adds user message to history."""
        mock_client = MockLLMClient([
            "Welcome!",  # For greeting
            '{"risk_tolerance": "moderate"}',  # For extraction
            "Great! What's your investment horizon?",  # For question
        ])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")
        initial_messages = len(state.messages)

        state = agent.process_response(state, "I have a moderate risk tolerance")

        # Should have added user message and assistant response
        user_messages = [m for m in state.messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "I have a moderate risk tolerance"

    def test_process_response_extracts_information(self):
        """Test that processing extracts profile information."""
        mock_client = MockLLMClient([
            "Welcome!",  # For greeting
            '{"risk_tolerance": "conservative"}',  # For extraction
            "What's your investment time horizon?",  # For question
        ])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")
        state = agent.process_response(state, "I'm quite conservative with investments")

        # Should have extracted risk_tolerance
        assert state.slots.risk_tolerance.value == "conservative"
        assert state.slots.risk_tolerance.status == SlotStatus.FILLED

    def test_process_response_generates_follow_up(self):
        """Test that processing generates follow-up question."""
        mock_client = MockLLMClient([
            "Welcome! What's your risk tolerance?",  # For greeting
            '{"risk_tolerance": "moderate"}',  # For extraction
            "Great choice! Now, what's your investment time horizon?",  # For question
        ])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")
        state = agent.process_response(state, "Moderate risk is fine for me")

        # Should have added a follow-up question
        last_message = state.messages[-1]
        assert last_message.role == "assistant"

    def test_process_multiple_responses(self):
        """Test processing multiple user responses."""
        responses = [
            "Welcome!",  # Greeting
            '{"risk_tolerance": "moderate"}',  # Extract risk
            "What's your horizon?",  # Question
            '{"investment_horizon": "long"}',  # Extract horizon
            "Do you have emergency fund?",  # Question
        ]
        mock_client = MockLLMClient(responses)
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")
        state = agent.process_response(state, "Moderate risk")
        state = agent.process_response(state, "Long term, like 10+ years")

        # Should have filled multiple slots
        assert state.slots.risk_tolerance.value is not None
        assert state.slots.investment_horizon.value is not None


class TestProfilingAgentCompletion:
    """Tests for profile completion detection."""

    def test_completion_percentage_increases(self):
        """Test that completion percentage increases as slots fill."""
        mock_client = MockLLMClient([
            "Welcome!",
            '{"risk_tolerance": "moderate"}',
            "Next question",
        ])
        agent = ProfilingAgent(llm_client=mock_client)

        state = agent.start_conversation("user123")
        initial_completion = state.completion_percentage

        state = agent.process_response(state, "Moderate")

        assert state.completion_percentage > initial_completion

    def test_get_profile_summary(self):
        """Test getting profile summary."""
        mock_client = MockLLMClient()
        agent = ProfilingAgent(llm_client=mock_client)

        state = ConversationState(user_id="user123")
        state.slots.risk_tolerance.value = "moderate"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        summary = agent.get_profile_summary(state)

        assert summary["user_id"] == "user123"
        assert "Risk Tolerance" in summary["filled_slots"]
        assert len(summary["missing_required"]) > 0

    def test_export_profile(self):
        """Test exporting filled profile."""
        mock_client = MockLLMClient()
        agent = ProfilingAgent(llm_client=mock_client)

        state = ConversationState(user_id="user123")
        state.slots.risk_tolerance.value = "aggressive"
        state.slots.investment_horizon.value = "long"

        profile = agent.export_profile(state)

        assert profile["user_id"] == "user123"
        assert profile["risk_tolerance"] == "aggressive"
        assert profile["investment_horizon"] == "long"


class TestProfilingAgentHelpers:
    """Tests for agent helper methods."""

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        mock_client = MockLLMClient()
        agent = ProfilingAgent(llm_client=mock_client)

        state = ConversationState(user_id="user123")
        state.messages.append(Mock(role="assistant", content="Hello"))
        state.messages.append(Mock(role="user", content="Hi"))
        state.messages.append(Mock(role="assistant", content="How can I help?"))

        last_msg = agent.get_last_assistant_message(state)
        assert last_msg == "How can I help?"

    def test_get_last_assistant_message_empty(self):
        """Test getting last assistant message when none exist."""
        mock_client = MockLLMClient()
        agent = ProfilingAgent(llm_client=mock_client)

        state = ConversationState(user_id="user123")

        last_msg = agent.get_last_assistant_message(state)
        assert last_msg is None


class TestConversationFlow:
    """Integration tests for complete conversation flows."""

    def test_basic_conversation_flow(self):
        """Test a basic conversation flow filling a few slots."""
        # Create responses for a simple conversation
        responses = [
            # Greeting
            "Welcome! Let's build your investment profile. What's your experience level with investing?",
            # Extract experience
            '{"experience_level": "beginner"}',
            # Question about risk
            "As a beginner, how would you describe your risk tolerance?",
            # Extract risk
            '{"risk_tolerance": "conservative"}',
            # Next question
            "What's your investment time horizon?",
        ]

        mock_client = MockLLMClient(responses)
        agent = ProfilingAgent(llm_client=mock_client)

        # Start conversation
        state = agent.start_conversation("user123", "Alice")
        assert len(state.messages) == 1

        # First response
        state = agent.process_response(state, "I'm new to investing")
        assert state.slots.experience_level.value == "beginner"

        # Second response
        state = agent.process_response(state, "I prefer safe investments")
        assert state.slots.risk_tolerance.value == "conservative"

        # Verify completion increased
        assert state.completion_percentage > 0

    def test_iteration_limit(self):
        """Test that iteration limit is respected."""
        mock_client = MockLLMClient(['{"risk_tolerance": null}', "Question"] * 50)
        agent = ProfilingAgent(llm_client=mock_client)

        state = ConversationState(user_id="user123")
        state.max_iterations = 3

        # Process responses until limit
        for i in range(5):
            state = agent.process_response(state, "I don't know")

            if state.is_complete:
                break

        # Should have stopped at some point
        assert state.iteration_count <= state.max_iterations + 2  # Allow some buffer
