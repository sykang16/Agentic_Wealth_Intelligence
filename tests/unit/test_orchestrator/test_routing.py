"""Unit tests for the IntentRouter."""

from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.multi_agent.routing import IntentRouter
from backend.src.multi_agent.state import UserIntent


@pytest.fixture
def mock_llm():
    """Create a mock LLM client for routing tests."""
    mock = MagicMock()
    return mock


@pytest.fixture
def router(mock_llm):
    """Create an IntentRouter with mock LLM."""
    return IntentRouter(mock_llm)


class TestIntentRouterLLM:
    """Tests for LLM-based intent classification."""

    def test_classify_portfolio_query(self, router, mock_llm):
        """Test classification of portfolio queries."""
        mock_llm.chat.return_value = LLMResponse(
            content="portfolio_query", model="test", provider="test"
        )
        assert router.classify("What's my net worth?") == UserIntent.PORTFOLIO_QUERY

    def test_classify_profiling(self, router, mock_llm):
        """Test classification of profiling requests."""
        mock_llm.chat.return_value = LLMResponse(
            content="profiling", model="test", provider="test"
        )
        assert router.classify("Help me build my profile") == UserIntent.PROFILING

    def test_classify_recommendation(self, router, mock_llm):
        """Test classification of recommendation requests."""
        mock_llm.chat.return_value = LLMResponse(
            content="recommendation", model="test", provider="test"
        )
        assert router.classify("What should I invest in?") == UserIntent.RECOMMENDATION

    def test_classify_general(self, router, mock_llm):
        """Test classification of general messages."""
        mock_llm.chat.return_value = LLMResponse(
            content="general", model="test", provider="test"
        )
        assert router.classify("Hello!") == UserIntent.GENERAL

    def test_classify_strips_whitespace(self, router, mock_llm):
        """Test that LLM response whitespace is handled."""
        mock_llm.chat.return_value = LLMResponse(
            content="  portfolio_query  \n", model="test", provider="test"
        )
        assert router.classify("Show my holdings") == UserIntent.PORTFOLIO_QUERY

    def test_classify_unknown_falls_back(self, router, mock_llm):
        """Test that unknown LLM response falls back to keyword classification."""
        mock_llm.chat.return_value = LLMResponse(
            content="unknown_intent", model="test", provider="test"
        )
        # "net worth" is a portfolio keyword
        assert router.classify("What's my net worth?") == UserIntent.PORTFOLIO_QUERY


class TestIntentRouterFallback:
    """Tests for keyword-based fallback classification."""

    def test_fallback_portfolio_keywords(self, router, mock_llm):
        """Test fallback classification for portfolio keywords."""
        mock_llm.chat.side_effect = Exception("LLM failed")

        assert router.classify("What's my net worth?") == UserIntent.PORTFOLIO_QUERY
        assert router.classify("Show my portfolio") == UserIntent.PORTFOLIO_QUERY
        assert router.classify("What are my holdings?") == UserIntent.PORTFOLIO_QUERY

    def test_fallback_profiling_keywords(self, router, mock_llm):
        """Test fallback classification for profiling keywords."""
        mock_llm.chat.side_effect = Exception("LLM failed")

        assert router.classify("Build my profile") == UserIntent.PROFILING
        assert router.classify("What's my risk tolerance?") == UserIntent.PROFILING
        assert router.classify("Help me build my investment portfolio") == UserIntent.PROFILING
        assert router.classify("Help me build my portfolio") == UserIntent.PROFILING

    def test_fallback_recommendation_keywords(self, router, mock_llm):
        """Test fallback classification for recommendation keywords."""
        mock_llm.chat.side_effect = Exception("LLM failed")

        assert router.classify("Can you recommend something?") == UserIntent.RECOMMENDATION
        assert router.classify("Where to invest my money?") == UserIntent.RECOMMENDATION

    def test_fallback_general(self, router, mock_llm):
        """Test fallback returns general for unrecognized messages."""
        mock_llm.chat.side_effect = Exception("LLM failed")

        assert router.classify("Hello!") == UserIntent.GENERAL
        assert router.classify("Thanks!") == UserIntent.GENERAL
