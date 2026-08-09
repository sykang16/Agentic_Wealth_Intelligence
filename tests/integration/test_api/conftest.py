"""Shared fixtures for API tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.src.api.app import create_app
from backend.src.api import dependencies as deps
from backend.src.common.llm_client import LLMResponse


class SimpleMockLLM:
    """Simple mock LLM for API tests."""

    def chat(self, user_message, system_prompt=None, max_tokens=2048, temperature=0.7):
        # Classify intent based on keywords
        if system_prompt and "classifier" in system_prompt.lower():
            msg_lower = user_message.lower()
            if "net worth" in msg_lower or "portfolio" in msg_lower:
                return LLMResponse(content="portfolio_query", model="mock", provider="mock")
            elif "profile" in msg_lower:
                return LLMResponse(content="profiling", model="mock", provider="mock")
            elif "recommend" in msg_lower:
                return LLMResponse(content="recommendation", model="mock", provider="mock")
            return LLMResponse(content="general", model="mock", provider="mock")

        if "extract" in (system_prompt or "").lower() or "json" in (system_prompt or "").lower():
            return LLMResponse(content='{}', model="mock", provider="mock")

        return LLMResponse(content="Mock response.", model="mock", provider="mock")

    def chat_with_history(self, messages, system_prompt=None, max_tokens=2048, temperature=0.7):
        last_msg = messages[-1]["content"] if messages else ""
        return self.chat(last_msg, system_prompt, max_tokens, temperature)


@pytest.fixture
def mock_llm_for_api():
    return SimpleMockLLM()


@pytest.fixture
def api_client(sample_aggregator_with_data, mock_llm_for_api):
    """Create a FastAPI test client with mocked dependencies."""
    mock_rag = MagicMock()
    mock_rag.is_initialized.return_value = True
    mock_rag.search.return_value = {"context": "", "results": [], "query": "", "num_results": 0}
    mock_rag.retriever = MagicMock()
    mock_rag.retriever.get_context_for_query.return_value = ""

    app = create_app()

    # Override dependencies
    app.dependency_overrides[deps.get_aggregator] = lambda: sample_aggregator_with_data
    app.dependency_overrides[deps.get_llm_client] = lambda: mock_llm_for_api
    app.dependency_overrides[deps.get_rag] = lambda: mock_rag

    from backend.src.agents.orchestrator import WealthOrchestrator
    orchestrator = WealthOrchestrator(sample_aggregator_with_data, mock_llm_for_api, mock_rag)
    app.dependency_overrides[deps.get_orchestrator] = lambda: orchestrator

    # Use in-memory SQLite for logging so tests don't create real DB files
    from backend.src.logging_db import LogRepository, get_engine, get_session_factory
    test_engine = get_engine("sqlite:///:memory:")
    test_factory = get_session_factory(test_engine)
    test_log_repo = LogRepository(test_factory)
    app.dependency_overrides[deps.get_log_repository] = lambda: test_log_repo

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
