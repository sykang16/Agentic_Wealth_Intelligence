"""Prompts for the Investment Profiling Agent."""

from backend.src.profiling.prompts.extractor_prompt import EXTRACTOR_PROMPT
from backend.src.profiling.prompts.question_prompts import (
    QUESTION_GENERATOR_PROMPT,
    GREETING_PROMPT,
    COMPLETION_PROMPT,
)
from backend.src.profiling.prompts.validator_prompt import VALIDATOR_PROMPT

__all__ = [
    "EXTRACTOR_PROMPT",
    "QUESTION_GENERATOR_PROMPT",
    "GREETING_PROMPT",
    "COMPLETION_PROMPT",
    "VALIDATOR_PROMPT",
]
