"""Integration tests for the multi-agent orchestrator."""

from unittest.mock import MagicMock

import pytest

from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.common.llm_client import LLMResponse
from backend.src.multi_agent.state import UserIntent


class MockLLM:
    """Mock LLM that returns intent-aware responses."""

    def __init__(self):
        self._call_count = 0
        self._responses = {}

    def set_response(self, keyword: str, content: str):
        self._responses[keyword] = content

    def chat(self, user_message: str, system_prompt: str | None = None,
             max_tokens: int = 2048, temperature: float = 0.7) -> LLMResponse:
        self._call_count += 1

        # Intent classification
        if system_prompt and "intent classifier" in system_prompt.lower():
            msg_lower = user_message.lower()
            if any(k in msg_lower for k in ["net worth", "portfolio", "holdings"]):
                return LLMResponse(content="portfolio_query", model="mock", provider="mock")
            elif any(k in msg_lower for k in ["profile", "risk tolerance"]):
                return LLMResponse(content="profiling", model="mock", provider="mock")
            elif any(k in msg_lower for k in ["recommend", "invest in", "should i"]):
                return LLMResponse(content="recommendation", model="mock", provider="mock")
            return LLMResponse(content="general", model="mock", provider="mock")

        # General responses
        if "extract" in (system_prompt or "").lower() or "json" in (system_prompt or "").lower():
            return LLMResponse(content='{}', model="mock", provider="mock")

        return LLMResponse(content="Mock response to your question.", model="mock", provider="mock")

    def chat_with_history(self, messages, system_prompt=None,
                          max_tokens=2048, temperature=0.7) -> LLMResponse:
        last_msg = messages[-1]["content"] if messages else ""
        return self.chat(last_msg, system_prompt, max_tokens, temperature)


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_aggregator(sample_aggregator_with_data):
    return sample_aggregator_with_data


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.is_initialized.return_value = True
    rag.search.return_value = {"context": "", "results": [], "query": "", "num_results": 0}
    rag.retriever = MagicMock()
    rag.retriever.get_context_for_query.return_value = ""
    return rag


@pytest.fixture
def orchestrator(mock_aggregator, mock_llm, mock_rag):
    return WealthOrchestrator(mock_aggregator, mock_llm, mock_rag)


class TestWealthOrchestrator:
    """Integration tests for WealthOrchestrator."""

    def test_create_initial_state(self, orchestrator):
        """Test creating initial state."""
        state = orchestrator.create_initial_state("user_001")
        assert state["user_id"] == "user_001"
        assert state["messages"] == []

    def test_process_general_message(self, orchestrator):
        """Test processing a general greeting message."""
        state = orchestrator.create_initial_state("user_001")
        result = orchestrator.process_message("Hello!", state)
        assert result.get("response")
        assert len(result["messages"]) == 2  # user + assistant

    def test_process_portfolio_query(self, orchestrator):
        """Test processing a portfolio query."""
        state = orchestrator.create_initial_state("user_001")
        result = orchestrator.process_message("What's my net worth?", state)
        assert result.get("response")
        assert result.get("module_source") == "Portfolio"

    def test_process_profiling_request(self, orchestrator):
        """Test processing a profiling request."""
        state = orchestrator.create_initial_state("user_001")
        result = orchestrator.process_message("Help me build my investment profile", state)
        assert result.get("response")
        assert result.get("module_source") == "Profiling"
        assert result.get("profiling_state") is not None

    def test_get_response(self, orchestrator):
        """Test extracting OrchestratorResponse from state."""
        state = orchestrator.create_initial_state("user_001")
        result = orchestrator.process_message("Hello!", state)
        response = orchestrator.get_response(result)
        assert response.response
        assert response.success

    def test_message_history_accumulates(self, orchestrator):
        """Test that message history grows across turns."""
        state = orchestrator.create_initial_state("user_001")
        state = orchestrator.process_message("Hello!", state)
        assert len(state["messages"]) == 2

        state = orchestrator.process_message("What can you do?", state)
        assert len(state["messages"]) == 4
