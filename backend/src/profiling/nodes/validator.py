"""Validator node for validating extracted profile values."""

import json
import re
from typing import Any

from backend.src.common.llm_client import LLMClient
from backend.src.profiling.prompts.validator_prompt import VALIDATOR_PROMPT
from backend.src.profiling.schemas import (
    ConversationState,
    SlotStatus,
    SLOT_VALIDATION_RULES,
)


def _validate_value(slot_name: str, value: Any) -> tuple[Any, str | None]:
    """Validate a single slot value against its rules.

    Args:
        slot_name: Name of the slot.
        value: Value to validate.

    Returns:
        Tuple of (validated_value, error_message).
        If validation passes, error_message is None.
        If validation fails and cannot be auto-corrected, validated_value is None.
    """
    if value is None:
        return None, None

    rules = SLOT_VALIDATION_RULES.get(slot_name)
    if not rules:
        return value, None  # No rules, accept as-is

    rule_type = rules.get("type")

    if rule_type == "enum":
        valid_values = rules.get("values", [])
        # Normalize to lowercase for comparison
        if isinstance(value, str):
            normalized = value.lower().strip()
            if normalized in valid_values:
                return normalized, None

            # Try common mappings
            mappings = {
                # Risk tolerance
                "safe": "conservative",
                "low risk": "conservative",
                "cautious": "conservative",
                "careful": "conservative",
                "balanced": "moderate",
                "middle": "moderate",
                "medium": "moderate",
                "high risk": "aggressive",
                "risky": "aggressive",
                "growth": "aggressive",
                # Investment horizon
                "short term": "short",
                "short-term": "short",
                "soon": "short",
                "1-2 years": "short",
                "medium term": "medium",
                "medium-term": "medium",
                "mid term": "medium",
                "3-5 years": "medium",
                "5-10 years": "medium",
                "long term": "long",
                "long-term": "long",
                "retirement": "long",
                "10+ years": "long",
                # Income stability
                "regular": "stable",
                "steady": "stable",
                "salary": "stable",
                "freelance": "variable",
                "commission": "variable",
                "irregular": "variable",
                "unsure": "uncertain",
                "unknown": "uncertain",
                # Experience
                "new": "beginner",
                "novice": "beginner",
                "none": "beginner",
                "just started": "beginner",
                "some": "intermediate",
                "a bit": "intermediate",
                "experienced": "intermediate",
                "expert": "advanced",
                "professional": "advanced",
                "extensive": "advanced",
                # Debt level
                "zero": "none",
                "no debt": "none",
                "little": "low",
                "minimal": "low",
                "manageable": "low",
                "some debt": "moderate",
                "average": "moderate",
                "significant": "high",
                "a lot": "high",
                "lots": "high",
                # Liquidity needs
                "not important": "low",
                "somewhat": "medium",
                "very important": "high",
                "important": "high",
            }

            for phrase, mapped_value in mappings.items():
                if phrase in normalized and mapped_value in valid_values:
                    return mapped_value, None

        return None, f"Value '{value}' not in valid options: {valid_values}"

    elif rule_type == "int_range":
        min_val = rules.get("min", 0)
        max_val = rules.get("max", 100)
        try:
            int_value = int(value)
            if min_val <= int_value <= max_val:
                return int_value, None
            return None, f"Value {int_value} out of range [{min_val}, {max_val}]"
        except (ValueError, TypeError):
            return None, f"Cannot convert '{value}' to integer"

    elif rule_type == "float_range":
        min_val = rules.get("min", 0)
        max_val = rules.get("max", 100)
        try:
            float_value = float(value)
            if min_val <= float_value <= max_val:
                return float_value, None
            return None, f"Value {float_value} out of range [{min_val}, {max_val}]"
        except (ValueError, TypeError):
            return None, f"Cannot convert '{value}' to number"

    elif rule_type == "bool":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in ["yes", "yeah", "yep", "true", "i do", "definitely", "sure", "absolutely"]:
                return True, None
            if lower in ["no", "nope", "false", "i don't", "not really", "nah", "never"]:
                return False, None
        return None, f"Cannot interpret '{value}' as yes/no"

    elif rule_type == "list_string":
        if isinstance(value, list):
            return [str(v).strip() for v in value if v], None
        if isinstance(value, str):
            # Split by commas or "and"
            items = re.split(r',|\band\b', value)
            return [item.strip() for item in items if item.strip()], None
        return None, f"Cannot convert '{value}' to list"

    elif rule_type == "string":
        if value:
            return str(value).strip(), None
        return None, "Empty value"

    return value, None


def validate_extracted_values(state: ConversationState, llm_client: LLMClient | None = None) -> ConversationState:
    """Validate extracted values and update slots.

    This node:
    1. Validates each extracted value against defined rules
    2. Attempts auto-correction for common variations
    3. Updates slots with validated values
    4. Records validation errors

    Args:
        state: Current conversation state with extraction results.
        llm_client: Optional LLM client for complex validation.

    Returns:
        Updated conversation state with validated values.
    """
    if not state.last_extraction:
        return state

    validation_errors = []
    validated_values = {}

    for slot_name, value in state.last_extraction.items():
        if value is None:
            continue

        validated, error = _validate_value(slot_name, value)

        if error:
            validation_errors.append(f"{slot_name}: {error}")
        elif validated is not None:
            validated_values[slot_name] = validated

    # Update state with validated values
    state.last_extraction = validated_values
    state.validation_errors = validation_errors

    # Apply validated values to slots
    for slot_name, value in validated_values.items():
        if hasattr(state.slots, slot_name):
            slot = getattr(state.slots, slot_name)
            slot.value = value
            slot.status = SlotStatus.FILLED

    # Recalculate completion
    state.completion_percentage = state.calculate_completion()

    return state


def validate_with_llm(
    state: ConversationState,
    llm_client: LLMClient,
) -> ConversationState:
    """Use LLM to validate and potentially correct extracted values.

    This is a fallback for complex cases where rule-based validation fails.

    Args:
        state: Current conversation state.
        llm_client: LLM client for validation.

    Returns:
        Updated conversation state.
    """
    if not state.last_extraction:
        return state

    # Format validation rules for the prompt
    rules_str = json.dumps(SLOT_VALIDATION_RULES, indent=2)

    prompt = VALIDATOR_PROMPT.format(
        extracted_values=json.dumps(state.last_extraction, indent=2),
        validation_rules=rules_str,
    )

    response = llm_client.chat(
        user_message=prompt,
        system_prompt="You are a precise validator. Return only valid JSON.",
        temperature=0.1,
    )

    # Parse response
    try:
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            result = json.loads(json_match.group())
            validated = result.get("validated_values", {})
            errors = result.get("errors", [])

            # Apply validated values
            for slot_name, value in validated.items():
                if value is not None and hasattr(state.slots, slot_name):
                    slot = getattr(state.slots, slot_name)
                    slot.value = value
                    slot.status = SlotStatus.FILLED

            state.validation_errors = errors
            state.completion_percentage = state.calculate_completion()
    except (json.JSONDecodeError, KeyError):
        pass  # Keep existing state if parsing fails

    return state
