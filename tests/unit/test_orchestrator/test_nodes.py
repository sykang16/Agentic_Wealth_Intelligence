"""Unit tests for orchestrator node functions."""

from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.multi_agent.nodes import (
    create_general_node,
    create_portfolio_node,
    create_profiling_node,
    create_recommend_node,
    create_router_node,
    respond_node,
)
from backend.src.multi_agent.state import OrchestratorState, UserIntent


@pytest.fixture
def base_state() -> OrchestratorState:
    """Create a base orchestrator state."""
    return OrchestratorState(
        user_id="user_001",
        messages=[],
        current_message="test message",
        intent="",
        response="",
        module_source="",
        profiling_state=None,
        error=None,
    )


class TestRouterNode:
    """Tests for the router node."""

    def test_router_classifies_intent(self, base_state):
        """Test router node classifies intent correctly."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="portfolio_query", model="test", provider="test"
        )
        node = create_router_node(mock_llm)
        result = node(base_state)
        assert result["intent"] == "portfolio_query"

    def test_router_handles_error(self, base_state):
        """Test router node handles errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("LLM down")
        node = create_router_node(mock_llm)
        result = node(base_state)
        # Should fall back to general
        assert result["intent"] == UserIntent.GENERAL.value


class TestPortfolioNode:
    """Tests for the portfolio node."""

    def test_portfolio_success(self, base_state):
        """Test portfolio node processes query successfully."""
        mock_agent = MagicMock()
        mock_agent.process.return_value = MagicMock(
            success=True, answer="Your net worth is $100,000", error=None
        )
        node = create_portfolio_node(mock_agent)
        result = node(base_state)
        assert result["module_source"] == "Portfolio"
        assert "100,000" in result["response"]

    def test_portfolio_error(self, base_state):
        """Test portfolio node handles errors."""
        mock_agent = MagicMock()
        mock_agent.process.side_effect = Exception("Data error")
        node = create_portfolio_node(mock_agent)
        result = node(base_state)
        assert result["module_source"] == "Portfolio"
        assert result["error"] is not None


class TestProfilingNode:
    """Tests for the profiling node."""

    def test_profiling_starts_new(self, base_state):
        """Test profiling node starts new conversation."""
        mock_agent = MagicMock()
        mock_state = MagicMock()
        mock_agent.start_conversation.return_value = mock_state
        mock_agent.get_last_assistant_message.return_value = "Welcome! Let's start."

        node = create_profiling_node(mock_agent)
        result = node(base_state)
        assert result["module_source"] == "Profiling"
        assert "Welcome" in result["response"]
        assert result["profiling_state"] is not None

    def test_profiling_continues(self, base_state):
        """Test profiling node continues existing conversation."""
        mock_agent = MagicMock()
        mock_conv_state = MagicMock()
        mock_conv_state.is_complete = False
        mock_agent.process_response.return_value = mock_conv_state
        mock_agent.get_last_assistant_message.return_value = "What's your risk tolerance?"

        base_state["profiling_state"] = MagicMock()
        node = create_profiling_node(mock_agent)
        result = node(base_state)
        assert result["module_source"] == "Profiling"


class TestRecommendNode:
    """Tests for the recommendation node."""

    def test_recommend_success(self, base_state):
        """Test recommendation node generates response."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.summary = "3 recommendations generated."
        mock_rec = MagicMock()
        mock_rec.title = "Diversify into bonds"
        mock_rec.category.value = "diversify"
        mock_rec.summary = "Add bond exposure."
        mock_rec.suggested_action = "Buy BND."
        mock_response.recommendations = [mock_rec]
        mock_response.warnings = []
        mock_engine.generate_recommendations.return_value = mock_response

        node = create_recommend_node(mock_engine)
        result = node(base_state)
        assert result["module_source"] == "Recommendation"
        assert "Diversify" in result["response"]


class TestGeneralNode:
    """Tests for the general node."""

    def test_general_success(self, base_state):
        """Test general node returns LLM response."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="Hello! How can I help?", model="test", provider="test"
        )
        node = create_general_node(mock_llm)
        result = node(base_state)
        assert result["module_source"] == "General"
        assert "Hello" in result["response"]

    def test_general_fallback_on_error(self, base_state):
        """Test general node provides fallback on LLM error."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API error")
        node = create_general_node(mock_llm)
        result = node(base_state)
        assert result["module_source"] == "General"
        assert "Portfolio Analysis" in result["response"]


class TestRespondNode:
    """Tests for the respond node."""

    def test_respond_appends_messages(self, base_state):
        """Test respond node appends messages to history."""
        base_state["response"] = "Here is the answer."
        base_state["current_message"] = "What is my net worth?"
        result = respond_node(base_state)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_respond_preserves_existing_messages(self, base_state):
        """Test respond node preserves existing message history."""
        base_state["messages"] = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        base_state["response"] = "New answer."
        base_state["current_message"] = "New question?"
        result = respond_node(base_state)
        assert len(result["messages"]) == 4
