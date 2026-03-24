"""LLM client supporting multiple providers: Anthropic, OpenAI, and Gemini."""

import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel


# Rate limit configuration
MAX_RETRIES = 2  # Maximum retry attempts
RETRY_DELAY = 1  # Seconds to wait between retries


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] | None = None


class LLMClient:
    """Unified client for multiple LLM providers."""

    # Default models for each provider
    DEFAULT_MODELS = {
        LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
        LLMProvider.OPENAI: "gpt-4o",
        LLMProvider.GEMINI: "gemini-2.0-flash",
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,  # Default to OpenAI
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider to use.
            model: Model to use. Defaults to provider's default model.
            api_key: API key. Defaults to environment variable.
        """
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self._client = None
        self._api_key = api_key

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.provider == LLMProvider.GEMINI:
            self._init_gemini()

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        from anthropic import Anthropic

        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self._client = Anthropic(api_key=api_key)

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        from openai import OpenAI
        import httpx

        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        # Create client with explicit http_client to avoid proxy issues
        http_client = httpx.Client()
        self._client = OpenAI(api_key=api_key, http_client=http_client)

    def _init_gemini(self) -> None:
        """Initialize Gemini client using new google.genai package."""
        from google import genai

        api_key = self._api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        self._client = genai.Client(api_key=api_key)

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat message to the LLM with retry logic.

        Args:
            user_message: User's message.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            Exception: If all retries fail.
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if self.provider == LLMProvider.ANTHROPIC:
                    return self._chat_anthropic(user_message, system_prompt, max_tokens, temperature)
                elif self.provider == LLMProvider.OPENAI:
                    return self._chat_openai(user_message, system_prompt, max_tokens, temperature)
                elif self.provider == LLMProvider.GEMINI:
                    return self._chat_gemini(user_message, system_prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded for {self.provider.value}. "
                            f"Please try a different model or wait before retrying."
                        ) from e
                elif (
                    "authentication" in error_str
                    or "invalid_api_key" in error_str
                    or "invalid api key" in error_str
                    or "incorrect api key" in error_str
                    or "401" in error_str
                ):
                    raise Exception(
                        f"Authentication failed for {self.provider.value}. "
                        f"Please check your API key in the .env file."
                    ) from e
                elif (
                    "insufficient" in error_str
                    or "billing" in error_str
                    or "balance" in error_str
                    or "credits" in error_str
                    or "402" in error_str
                    or "out of" in error_str
                ):
                    raise Exception(
                        f"Insufficient API credits for {self.provider.value}. "
                        f"Please top up your account balance."
                    ) from e
                else:
                    # For other errors, don't retry
                    raise

        raise last_error if last_error else Exception("Unknown error occurred")

    def _chat_anthropic(
        self,
        user_message: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Chat using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _chat_openai(
        self,
        user_message: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Chat using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
        )

    def _chat_gemini(
        self,
        user_message: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Chat using Gemini API with new google.genai package."""
        from google.genai import types

        # Combine system prompt with user message
        full_prompt = user_message
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_message}"

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=config,
        )

        return LLMResponse(
            content=response.text,
            model=self.model,
            provider="gemini",
            usage=None,
        )

    def chat_with_history(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat with message history (with retry logic).

        Args:
            messages: List of messages with 'role' and 'content'.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and metadata.
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if self.provider == LLMProvider.ANTHROPIC:
                    return self._chat_with_history_anthropic(
                        messages, system_prompt, max_tokens, temperature
                    )
                elif self.provider == LLMProvider.OPENAI:
                    return self._chat_with_history_openai(
                        messages, system_prompt, max_tokens, temperature
                    )
                else:
                    # For Gemini, flatten to single message
                    combined = "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in messages
                    )
                    return self._chat_gemini(combined, system_prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded for {self.provider.value}. "
                            f"Please try a different model or wait before retrying."
                        ) from e
                elif (
                    "authentication" in error_str
                    or "invalid_api_key" in error_str
                    or "invalid api key" in error_str
                    or "incorrect api key" in error_str
                    or "401" in error_str
                ):
                    raise Exception(
                        f"Authentication failed for {self.provider.value}. "
                        f"Please check your API key in the .env file."
                    ) from e
                elif (
                    "insufficient" in error_str
                    or "billing" in error_str
                    or "balance" in error_str
                    or "credits" in error_str
                    or "402" in error_str
                    or "out of" in error_str
                ):
                    raise Exception(
                        f"Insufficient API credits for {self.provider.value}. "
                        f"Please top up your account balance."
                    ) from e
                else:
                    raise

        raise last_error if last_error else Exception("Unknown error occurred")

    def _chat_with_history_anthropic(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Chat with history using Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _chat_with_history_openai(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Chat with history using OpenAI API."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
        )


def get_available_providers() -> list[dict[str, Any]]:
    """Get list of available LLM providers based on configured API keys.

    Returns providers in order of preference: OpenAI first (most reliable),
    then Anthropic, then Gemini.
    """
    providers = []

    # OpenAI first (most reliable, no rate limit issues)
    if os.getenv("OPENAI_API_KEY"):
        providers.append({
            "provider": LLMProvider.OPENAI,
            "name": "OpenAI GPT-4o",
            "model": LLMClient.DEFAULT_MODELS[LLMProvider.OPENAI],
        })

    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({
            "provider": LLMProvider.ANTHROPIC,
            "name": "Anthropic Claude",
            "model": LLMClient.DEFAULT_MODELS[LLMProvider.ANTHROPIC],
        })

    # Gemini last (often has rate limit issues on free tier)
    if os.getenv("GEMINI_API_KEY"):
        providers.append({
            "provider": LLMProvider.GEMINI,
            "name": "Google Gemini",
            "model": LLMClient.DEFAULT_MODELS[LLMProvider.GEMINI],
        })

    return providers
