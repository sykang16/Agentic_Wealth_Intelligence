"""Intent routing for the multi-agent orchestrator."""

import json
import logging
import re

from backend.src.common.llm_client import LLMClient

from .state import UserIntent

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a wealth management system.
Given a user message, classify it into exactly one of these categories:

- portfolio_query: Questions about the user's EXISTING portfolio, such as current net worth, holdings, assets, accounts, allocation, gains/losses, liquidity, real estate, or financial metrics. This is for VIEWING or QUERYING data the user already has.
- profiling: Requests to BUILD, CREATE, SET UP, or UPDATE an investment profile, portfolio plan, or strategy. This includes risk assessments, preferences questionnaires, and any request where the user wants to establish or revise their investment approach. Keywords: "build", "create", "set up", "help me with my portfolio", "plan my investments".
- recommendation: Requests for specific investment advice, suggestions on what to buy/sell, how to diversify, or actionable financial recommendations for specific assets or strategies.
- general: Greetings, thanks, help requests, off-topic, or anything that doesn't fit the above.

IMPORTANT: If the user says "build", "create", "set up", or "help me build" related to their portfolio or investments, classify as "profiling" — NOT "portfolio_query". The "portfolio_query" category is strictly for querying existing data.

Respond with ONLY the category name (one of: portfolio_query, profiling, recommendation, general). No other text."""


class IntentRouter:
    """Routes user messages to the appropriate module via LLM classification."""

    # Keyword patterns for fallback classification
    PORTFOLIO_KEYWORDS = [
        "net worth", "portfolio", "holdings", "allocation", "assets",
        "liabilities", "bank account", "investment account", "liquidity",
        "gains", "losses", "real estate", "sector", "how much", "balance",
        "what do i own", "my stocks", "my investments", "account",
        "financially", "my finances", "financial health",
    ]
    PROFILING_KEYWORDS = [
        "profile", "profiling", "risk tolerance", "risk assessment",
        "build my profile", "investment profile", "questionnaire",
        "preferences", "my goals", "update my profile",
        "build my", "create my", "set up my", "help me build my",
        "plan my investment", "build an investment",
        "my investment goals", "my financial goals",
        "how much risk",
        "risk appetite", "risk level",
        "what kind of investor", "investment style",
        "new to investing", "investment objectives",
    ]
    RECOMMENDATION_KEYWORDS = [
        "recommend ", "suggestion", "advice", "should i buy", "should i sell",
        "diversify", "invest in", "what should i", "rebalance", "optimize",
        "improve my portfolio", "where to invest",
        "where should i invest", "reduce risk",
        "grow my wealth", "should i add", "should i put",
        "time to buy", "guidance on", "asset class",
        "safer portfolio", "do you suggest",
    ]

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def classify(self, message: str, history: list[dict[str, str]] | None = None) -> UserIntent:
        """Classify a user message into a UserIntent.

        Uses LLM classification with keyword fallback.
        Optionally accepts conversation history for context-aware routing.
        """
        try:
            # Build context from recent history so the LLM understands references
            context = ""
            if history:
                recent = history[-4:]  # last 2 exchanges
                context = "Recent conversation:\n" + "\n".join(
                    f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
                ) + "\n\n"

            response = self.llm.chat(
                user_message=context + f"Current message: {message}",
                system_prompt=ROUTER_SYSTEM_PROMPT,
                max_tokens=20,
                temperature=0.0,
            )
            intent_str = response.content.strip().lower()
            # Clean up potential extra text
            intent_str = intent_str.split("\n")[0].strip().strip(".")

            try:
                return UserIntent(intent_str)
            except ValueError:
                logger.warning(
                    f"LLM returned unknown intent '{intent_str}', using fallback"
                )
                return self._fallback_classification(message)

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using fallback")
            return self._fallback_classification(message)

    def _fallback_classification(self, message: str) -> UserIntent:
        """Keyword-based fallback classification."""
        msg_lower = message.lower()

        for keyword in self.PROFILING_KEYWORDS:
            if keyword in msg_lower:
                return UserIntent.PROFILING

        for keyword in self.RECOMMENDATION_KEYWORDS:
            if keyword in msg_lower:
                return UserIntent.RECOMMENDATION

        for keyword in self.PORTFOLIO_KEYWORDS:
            if keyword in msg_lower:
                return UserIntent.PORTFOLIO_QUERY

        return UserIntent.GENERAL
