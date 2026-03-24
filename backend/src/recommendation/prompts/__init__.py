"""Prompt templates for the recommendation engine."""

from .recommendation_prompt import (
    DEFAULT_QUERY,
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_PROMPT,
)
from .explanation_prompt import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_PROMPT,
)

__all__ = [
    "DEFAULT_QUERY",
    "EXPLANATION_SYSTEM_PROMPT",
    "EXPLANATION_USER_PROMPT",
    "RECOMMENDATION_SYSTEM_PROMPT",
    "RECOMMENDATION_USER_PROMPT",
]
