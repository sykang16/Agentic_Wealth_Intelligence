"""Validator prompt for validating extracted profile values."""

VALIDATOR_PROMPT = """You are an investment profile validator. Your task is to validate extracted values and suggest corrections if needed.

## Extracted Values
{extracted_values}

## Validation Rules
{validation_rules}

## Task
For each extracted value:
1. Check if it matches the expected type and constraints
2. If valid, keep the value as-is
3. If invalid but close (e.g., typo, different wording), suggest a correction
4. If invalid and cannot be corrected, mark as null

## Common Mappings to Apply
- "safe", "low risk", "cautious" → "conservative"
- "balanced", "middle ground" → "moderate"
- "high risk", "growth", "risky" → "aggressive"
- "yes", "yeah", "yep", "I do" → true
- "no", "nope", "not really", "I don't" → false
- "little", "minimal", "manageable" → "low"
- "some", "average" → "moderate"
- "lots", "significant", "a lot" → "high"
- "none", "zero", "nothing" → "none"
- "short term", "soon", "1-2 years" → "short"
- "medium term", "few years", "5 years" → "medium"
- "long term", "retirement", "10+ years" → "long"
- "new", "novice", "just started" → "beginner"
- "some experience", "a few years" → "intermediate"
- "expert", "professional", "years of experience" → "advanced"

## Response Format
Respond with a JSON object containing:
- "validated_values": object with slot name → validated value (or null if invalid)
- "corrections": object with slot name → explanation of correction made
- "errors": list of validation errors for values that couldn't be validated

Example:
```json
{{
    "validated_values": {{
        "risk_tolerance": "moderate",
        "loss_comfort": 7,
        "investment_horizon": null
    }},
    "corrections": {{
        "risk_tolerance": "Mapped 'balanced' to 'moderate'"
    }},
    "errors": [
        "investment_horizon: 'maybe 5 years' is ambiguous, needs clarification"
    ]
}}
```

Return ONLY the JSON object.
"""
