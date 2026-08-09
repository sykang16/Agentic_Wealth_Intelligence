"""Shared E2E fixtures for full pipeline tests."""

from unittest.mock import MagicMock

import pytest

from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.common.llm_client import LLMResponse


class MultiIntentMockLLM:
    """Mock LLM that handles multiple intent types in sequence.

    Provides realistic routing and response behavior for E2E tests.
    """

    def __init__(self):
        self._call_count = 0

    def chat(self, user_message: str, system_prompt: str | None = None,
             max_tokens: int = 2048, temperature: float = 0.7) -> LLMResponse:
        self._call_count += 1
        sp = (system_prompt or "").lower()
        msg = user_message.lower()

        # Intent classification
        if "classifier" in sp or "classify" in sp:
            # Extract just the current message when conversation context is prepended
            classify_msg = msg
            if "current message:" in msg:
                classify_msg = msg.split("current message:")[-1].strip()
            if any(k in classify_msg for k in ["net worth", "portfolio", "holdings", "allocation"]):
                return LLMResponse(content="portfolio_query", model="mock", provider="mock")
            if any(k in classify_msg for k in ["profile", "risk tolerance", "investment profile"]):
                return LLMResponse(content="profiling", model="mock", provider="mock")
            if any(k in classify_msg for k in ["recommend", "invest in", "should i", "advice", "diversify"]):
                return LLMResponse(content="recommendation", model="mock", provider="mock")
            return LLMResponse(content="general", model="mock", provider="mock")

        # Extraction responses
        if "extract" in sp or "json" in sp:
            if "moderate" in msg:
                return LLMResponse(content='{"risk_tolerance": "moderate"}', model="mock", provider="mock")
            if "conservative" in msg:
                return LLMResponse(content='{"risk_tolerance": "conservative"}', model="mock", provider="mock")
            if "long" in msg:
                return LLMResponse(content='{"investment_horizon": "long"}', model="mock", provider="mock")
            return LLMResponse(content='{}', model="mock", provider="mock")

        # Greeting / question gen
        if "greeting" in sp or "welcome" in sp:
            return LLMResponse(
                content="Welcome! Let's build your investment profile.",
                model="mock", provider="mock",
            )

        # Recommendation generation
        if "recommend" in sp:
            import json
            return LLMResponse(
                content=json.dumps([{
                    "category": "diversify",
                    "title": "Diversify into bonds",
                    "summary": "Add bond exposure for stability.",
                    "detailed_rationale": "Your portfolio is equity-heavy.",
                    "tickers": ["BND", "AGG"],
                    "suggested_action": "Buy BND or AGG ETFs",
                    "risk_level": "low",
                    "confidence": "medium",
                    "priority": "medium",
                }]),
                model="mock", provider="mock",
            )

        # Default
        return LLMResponse(
            content="Thank you for your question. Here is a helpful response.",
            model="mock", provider="mock",
        )

    def chat_with_history(self, messages, system_prompt=None,
                          max_tokens=2048, temperature=0.7) -> LLMResponse:
        last_msg = messages[-1]["content"] if messages else ""
        return self.chat(last_msg, system_prompt, max_tokens, temperature)


@pytest.fixture
def e2e_mock_llm():
    """Multi-intent mock LLM for E2E tests."""
    return MultiIntentMockLLM()


@pytest.fixture
def e2e_mock_rag():
    """Mock RAG system for E2E tests."""
    rag = MagicMock()
    rag.is_initialized.return_value = True
    rag.search.return_value = {
        "context": "ETFs provide diversified exposure to markets.",
        "results": [{"source": "ETF Guide", "content": "VOO tracks S&P 500", "score": 0.9}],
        "query": "",
        "num_results": 1,
    }
    rag.retriever = MagicMock()
    rag.retriever.get_context_for_query.return_value = "ETFs provide diversified exposure."
    return rag


@pytest.fixture
def e2e_orchestrator(sample_aggregator_with_data, e2e_mock_llm, e2e_mock_rag):
    """Create a WealthOrchestrator for E2E testing."""
    return WealthOrchestrator(sample_aggregator_with_data, e2e_mock_llm, e2e_mock_rag)
