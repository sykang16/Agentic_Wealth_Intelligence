"""Evaluation: Profiling slot extraction — deterministic replay tests.

Phase 1 scope — offline, no real LLM calls:
  - Tests _parse_extraction_response (JSON parser) against fixture responses.
  - Tests apply_extraction_to_state (state updater) with known extraction dicts.
  - Tests slot coverage: all required slots can be filled, completion reaches 100%.

Run with:
    pytest tests/eval/test_profiling_extraction.py -v -s
"""

from __future__ import annotations

import pytest

from backend.src.profiling.nodes.extractor import (
    _parse_extraction_response,
    apply_extraction_to_state,
)
from backend.src.profiling.schemas import ConversationState, ProfileSlots, SlotStatus

from .conftest import (
    REQUIRED_SLOTS,
    SLOT_FILL_VALUES,
    fill_all_required,
    fill_slots,
    make_state,
)


# ── _parse_extraction_response ────────────────────────────────────────────────


class TestParseExtractionResponse:
    """Tests for the JSON parser that turns raw LLM output into a dict."""

    # ---- Success cases (from scripted_extractions fixture) ----

    def test_single_slot_clean_json(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "clean_json_single_slot")
        result = _parse_extraction_response(case["llm_response"])
        assert result == case["expected_values"]

    def test_multiple_slots_clean_json(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "clean_json_multiple_slots")
        result = _parse_extraction_response(case["llm_response"])
        assert result == case["expected_values"]

    def test_json_embedded_in_prose(self, scripted_extractions):
        """LLM often wraps JSON in explanation text — parser must find the JSON block."""
        case = next(c for c in scripted_extractions if c["name"] == "json_embedded_in_prose")
        result = _parse_extraction_response(case["llm_response"])
        assert result == case["expected_values"]

    def test_json_with_list_value(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "json_with_list_value")
        result = _parse_extraction_response(case["llm_response"])
        assert result["previous_investments"] == ["stocks", "ETFs", "bonds"]

    def test_json_with_null_values_preserved(self, scripted_extractions):
        """Null values in LLM JSON must be preserved in the parsed dict."""
        case = next(c for c in scripted_extractions if c["name"] == "json_with_null_values")
        result = _parse_extraction_response(case["llm_response"])
        assert result["risk_tolerance"] == "moderate"
        assert result["loss_comfort"] is None
        assert result["investment_horizon"] is None

    def test_json_with_boolean(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "json_with_boolean")
        result = _parse_extraction_response(case["llm_response"])
        assert result["has_emergency_fund"] is True
        assert result["debt_level"] == "low"

    def test_json_with_float(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "json_with_float")
        result = _parse_extraction_response(case["llm_response"])
        assert result["target_return"] == pytest.approx(8.5)
        assert result["primary_goal"] == "retirement"

    def test_empty_json_object(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "empty_extraction")
        result = _parse_extraction_response(case["llm_response"])
        assert result == {}

    def test_all_required_slots_extracted_at_once(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "all_required_slots_at_once")
        result = _parse_extraction_response(case["llm_response"])
        for slot in REQUIRED_SLOTS:
            assert slot in result, f"Required slot '{slot}' missing from parsed extraction"

    # ---- JSON embedded in markdown code block ----

    def test_json_after_code_fence(self, scripted_extractions):
        """Regex greedy match handles JSON inside markdown fences."""
        case = next(c for c in scripted_extractions if c["name"] == "json_after_code_fence")
        result = _parse_extraction_response(case["llm_response"])
        # The regex may pick up the outer ``` block — just verify required keys present
        assert "income_stability" in result
        assert "has_emergency_fund" in result

    # ---- Failure / edge cases ----

    def test_plain_text_no_json_returns_empty(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "invalid_json_no_braces")
        result = _parse_extraction_response(case["llm_response"])
        assert result == {}

    def test_malformed_json_returns_empty(self, scripted_extractions):
        case = next(c for c in scripted_extractions if c["name"] == "malformed_json_partial")
        result = _parse_extraction_response(case["llm_response"])
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _parse_extraction_response("")
        assert result == {}

    def test_whitespace_only_returns_empty_dict(self):
        result = _parse_extraction_response("   \n   ")
        assert result == {}

    def test_non_json_curly_braces_returns_empty(self):
        """Template strings with curly braces must not be parsed as JSON."""
        result = _parse_extraction_response("Please fill in {field_name} here.")
        assert result == {}

    def test_unknown_slot_passes_through_parser(self, scripted_extractions):
        """The parser is slot-agnostic; unknown keys come through as-is."""
        case = next(c for c in scripted_extractions if c["name"] == "unknown_slot_in_json")
        result = _parse_extraction_response(case["llm_response"])
        assert "risk_tolerance" in result
        assert "unknown_field" in result  # parser doesn't filter


# ── apply_extraction_to_state ─────────────────────────────────────────────────


class TestApplyExtractionToState:
    """Tests for applying a parsed extraction dict into ConversationState."""

    def test_single_slot_applied_correctly(self):
        state = make_state()
        state.last_extraction = {"risk_tolerance": "moderate"}
        state = apply_extraction_to_state(state)

        assert state.slots.risk_tolerance.value == "moderate"
        assert state.slots.risk_tolerance.status == SlotStatus.FILLED

    def test_multiple_slots_applied_correctly(self):
        state = make_state()
        state.last_extraction = {
            "risk_tolerance": "conservative",
            "loss_comfort": 3,
            "investment_horizon": "long",
        }
        state = apply_extraction_to_state(state)

        assert state.slots.risk_tolerance.value == "conservative"
        assert state.slots.loss_comfort.value == 3
        assert state.slots.investment_horizon.value == "long"
        assert state.slots.risk_tolerance.status == SlotStatus.FILLED
        assert state.slots.loss_comfort.status == SlotStatus.FILLED
        assert state.slots.investment_horizon.status == SlotStatus.FILLED

    def test_unknown_slot_name_is_silently_ignored(self):
        """Unrecognized slot names must not raise exceptions or corrupt state."""
        state = make_state()
        state.last_extraction = {
            "risk_tolerance": "moderate",
            "completely_unknown_field": "some_value",
        }
        state = apply_extraction_to_state(state)

        assert state.slots.risk_tolerance.value == "moderate"
        assert not hasattr(state.slots, "completely_unknown_field")

    def test_none_values_skip_slot_update(self):
        """Null extraction values must not overwrite existing slot values."""
        state = make_state()
        # Pre-fill a slot
        state.slots.risk_tolerance.value = "aggressive"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        # LLM returns null for this slot
        state.last_extraction = {"risk_tolerance": None}
        state = apply_extraction_to_state(state)

        # Original value must be preserved
        assert state.slots.risk_tolerance.value == "aggressive"

    def test_completion_percentage_updates_after_apply(self):
        """Completion percentage must increase after slots are applied."""
        state = make_state()
        assert state.completion_percentage == 0.0

        state.last_extraction = {"risk_tolerance": "moderate", "loss_comfort": 5}
        state = apply_extraction_to_state(state)

        assert state.completion_percentage > 0.0

    def test_empty_extraction_does_not_change_state(self):
        state = make_state()
        state.last_extraction = None
        before_completion = state.completion_percentage

        state = apply_extraction_to_state(state)

        assert state.completion_percentage == before_completion

    def test_overwrite_previously_filled_slot(self):
        """A subsequent extraction can update an already-filled slot."""
        state = make_state()
        state.slots.risk_tolerance.value = "conservative"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        state.last_extraction = {"risk_tolerance": "moderate"}
        state = apply_extraction_to_state(state)

        assert state.slots.risk_tolerance.value == "moderate"

    def test_boolean_slot_applied(self):
        state = make_state()
        state.last_extraction = {"has_emergency_fund": True}
        state = apply_extraction_to_state(state)

        assert state.slots.has_emergency_fund.value is True
        assert state.slots.has_emergency_fund.status == SlotStatus.FILLED

    def test_list_slot_applied(self):
        state = make_state()
        state.last_extraction = {"previous_investments": ["stocks", "ETFs"]}
        state = apply_extraction_to_state(state)

        assert state.slots.previous_investments.value == ["stocks", "ETFs"]
        assert state.slots.previous_investments.status == SlotStatus.FILLED


# ── Slot coverage ─────────────────────────────────────────────────────────────


class TestSlotCoverage:
    """Verify that all profile slots can be filled and completion logic is correct."""

    def test_all_required_slots_fillable(self):
        """Every required slot has a corresponding entry in SLOT_FILL_VALUES."""
        for slot_name in REQUIRED_SLOTS:
            assert slot_name in SLOT_FILL_VALUES, (
                f"Required slot '{slot_name}' has no test value in SLOT_FILL_VALUES — "
                f"update conftest.py"
            )

    def test_fill_all_required_reaches_100_percent(self):
        """Filling every required slot must bring completion to exactly 100%."""
        state = make_state()
        state = fill_all_required(state)

        assert state.calculate_completion() == 100.0

    def test_fill_zero_slots_is_0_percent(self):
        state = make_state()
        assert state.calculate_completion() == 0.0

    def test_fill_half_required_is_roughly_50_percent(self):
        """Filling 5 of 11 required slots → ~45.5% (not exactly 50% due to rounding)."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:5])
        completion = state.calculate_completion()
        assert 40.0 <= completion <= 55.0

    def test_optional_slots_do_not_count_toward_required_completion(self):
        """Filling only optional slots must leave completion at 0%."""
        optional_slots = ["previous_investments", "target_return", "esg_preference"]
        state = make_state()
        state = fill_slots(state, optional_slots)
        assert state.calculate_completion() == 0.0

    def test_required_slot_names_match_schema(self):
        """REQUIRED_SLOTS list must match what ProfileSlots actually defines as required."""
        schema_required = [
            field_name
            for field_name in ProfileSlots.model_fields
            if ProfileSlots.model_fields[field_name].default_factory().required  # type: ignore[misc]
        ]
        assert set(REQUIRED_SLOTS) == set(schema_required), (
            f"REQUIRED_SLOTS in conftest.py is out of sync with ProfileSlots schema. "
            f"Schema required: {schema_required}. Conftest: {REQUIRED_SLOTS}"
        )

    def test_get_missing_required_slots_decreases_as_slots_filled(self):
        """Missing required slot count must decrease as we fill slots."""
        state = make_state()
        initial_missing = len(state.get_missing_required_slots())
        assert initial_missing == len(REQUIRED_SLOTS)

        # Fill slots one by one
        for i, slot_name in enumerate(REQUIRED_SLOTS, start=1):
            state = fill_slots(state, [slot_name])
            missing = len(state.get_missing_required_slots())
            assert missing == len(REQUIRED_SLOTS) - i

    def test_get_filled_slots_increases_as_slots_filled(self):
        """Filled slot count must increase as we fill slots."""
        state = make_state()
        assert len(state.get_filled_slots()) == 0

        for i, slot_name in enumerate(REQUIRED_SLOTS, start=1):
            state = fill_slots(state, [slot_name])
            assert len(state.get_filled_slots()) == i

    def test_get_next_empty_slot_cycles_through_required_first(self):
        """get_next_empty_slot() should return required slots before optional ones."""
        state = make_state()
        next_slot = state.get_next_empty_slot()
        assert next_slot is not None
        assert next_slot.required is True

    def test_to_investment_profile_exports_all_filled_slots(self):
        """to_investment_profile() must include every filled slot in output dict."""
        state = make_state()
        state = fill_all_required(state)

        profile_dict = state.to_investment_profile()

        for slot_name in REQUIRED_SLOTS:
            assert slot_name in profile_dict, (
                f"Slot '{slot_name}' is filled but missing from to_investment_profile() output"
            )
