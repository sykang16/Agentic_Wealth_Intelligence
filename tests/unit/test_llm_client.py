"""Unit tests for the LLM client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.src.common.llm_client import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    get_available_providers,
)


class TestLLMClientInit:
    """Tests for LLM client initialization."""

    def test_init_openai_sets_provider(self):
        """Test OpenAI provider initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                import httpx
                with patch.object(httpx, "Client"):
                    client = LLMClient(provider=LLMProvider.OPENAI)
                    assert client.provider == LLMProvider.OPENAI
                    assert client.model == "gpt-4o"

    def test_init_anthropic_sets_provider(self):
        """Test Anthropic provider initialization."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                client = LLMClient(provider=LLMProvider.ANTHROPIC)
                assert client.provider == LLMProvider.ANTHROPIC
                assert client.model == "claude-sonnet-4-20250514"

    def test_init_openai_no_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                LLMClient(provider=LLMProvider.OPENAI)

    def test_init_anthropic_no_key_raises(self):
        """Test that missing Anthropic key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                LLMClient(provider=LLMProvider.ANTHROPIC)

    def test_init_gemini_no_key_raises(self):
        """Test that missing Gemini key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                LLMClient(provider=LLMProvider.GEMINI)

    def test_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                import httpx
                with patch.object(httpx, "Client"):
                    client = LLMClient(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo")
                    assert client.model == "gpt-3.5-turbo"

    def test_explicit_api_key(self):
        """Test initialization with explicit API key."""
        with patch("openai.OpenAI"):
            import httpx
            with patch.object(httpx, "Client"):
                client = LLMClient(provider=LLMProvider.OPENAI, api_key="explicit-key")
                assert client._api_key == "explicit-key"


class TestLLMClientChat:
    """Tests for the chat method."""

    def test_chat_openai_dispatch(self):
        """Test chat dispatches to OpenAI correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai_cls:
                import httpx
                with patch.object(httpx, "Client"):
                    mock_client_inst = MagicMock()
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = "Hello!"
                    mock_response.model = "gpt-4o"
                    mock_response.usage.prompt_tokens = 10
                    mock_response.usage.completion_tokens = 5
                    mock_client_inst.chat.completions.create.return_value = mock_response
                    mock_openai_cls.return_value = mock_client_inst

                    client = LLMClient(provider=LLMProvider.OPENAI)
                    result = client.chat("Hello")

                    assert isinstance(result, LLMResponse)
                    assert result.content == "Hello!"
                    assert result.provider == "openai"

    def test_chat_error_no_retry_on_non_rate_limit(self):
        """Test that non-rate-limit errors are not retried."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai_cls:
                import httpx
                with patch.object(httpx, "Client"):
                    mock_client_inst = MagicMock()
                    mock_client_inst.chat.completions.create.side_effect = ValueError("Bad request")
                    mock_openai_cls.return_value = mock_client_inst

                    client = LLMClient(provider=LLMProvider.OPENAI)
                    with pytest.raises(ValueError, match="Bad request"):
                        client.chat("Hello")

                    assert mock_client_inst.chat.completions.create.call_count == 1


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_llm_response_creation(self):
        """Test LLMResponse can be created."""
        resp = LLMResponse(content="Hello", model="gpt-4o", provider="openai")
        assert resp.content == "Hello"
        assert resp.model == "gpt-4o"
        assert resp.usage is None

    def test_llm_response_with_usage(self):
        """Test LLMResponse with usage info."""
        resp = LLMResponse(
            content="Hi",
            model="gpt-4o",
            provider="openai",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert resp.usage["input_tokens"] == 10


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GEMINI.value == "gemini"


class TestGetAvailableProviders:
    """Tests for get_available_providers."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "key1"}, clear=True)
    def test_single_provider(self):
        """Test with single provider configured."""
        providers = get_available_providers()
        assert len(providers) == 1
        assert providers[0]["provider"] == LLMProvider.OPENAI

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "key1", "ANTHROPIC_API_KEY": "key2", "GEMINI_API_KEY": "key3"},
        clear=True,
    )
    def test_all_providers(self):
        """Test with all providers configured."""
        providers = get_available_providers()
        assert len(providers) == 3
        assert providers[0]["provider"] == LLMProvider.OPENAI

    @patch.dict(os.environ, {}, clear=True)
    def test_no_providers(self):
        """Test with no providers configured."""
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
            os.environ.pop(key, None)
        providers = get_available_providers()
        assert len(providers) == 0
