"""Evaluation: Profiling checker — completion logic and routing state machine.

Phase 1 scope — offline, no real LLM calls:
  - Tests check_completion() node across all state transitions.
  - Tests should_continue() routing function for all possible routes.
  - Tests completion percentage accuracy across partial and full fills.
  - Tests edge cases: max iterations, validation errors, optional vs required slots.

Run with:
    pytest tests/eval/test_profiling_checker.py -v -s
"""

from __future__ import annotations

import pytest

from backend.src.profiling.nodes.checker import (
    check_completion,
    get_profile_summary,
    should_continue,
)
from backend.src.profiling.schemas import ConversationState, SlotStatus

from .conftest import (
    REQUIRED_SLOTS,
    fill_all_required,
    fill_slots,
    make_state,
)


# ── check_completion ──────────────────────────────────────────────────────────


class TestCheckCompletion:
    """Tests for the check_completion() node function."""

    def test_empty_state_is_not_complete(self):
        state = make_state()
        result = check_completion(state)

        assert result.is_complete is False
        assert result.completion_percentage == 0.0

    def test_empty_state_routes_to_ask_question(self):
        state = make_state()
        result = check_completion(state)

        assert result.next_action == "ask_question"

    def test_all_required_filled_marks_complete(self):
        state = make_state()
        state = fill_all_required(state)
        result = check_completion(state)

        assert result.is_complete is True
        assert result.completion_percentage == 100.0

    def test_all_required_filled_routes_to_complete(self):
        state = make_state()
        state = fill_all_required(state)
        result = check_completion(state)

        assert result.next_action == "complete"

    def test_partial_fill_not_complete(self):
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:4])
        result = check_completion(state)

        assert result.is_complete is False
        assert result.next_action == "ask_question"

    def test_partial_fill_completion_percentage(self):
        """Filling 3 of 9 required slots → 33.3%."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:3])
        result = check_completion(state)

        expected_pct = round((3 / len(REQUIRED_SLOTS)) * 100, 1)
        assert result.completion_percentage == pytest.approx(expected_pct, abs=0.2)

    def test_validation_errors_trigger_clarify_route(self):
        """Partial fill + validation errors → routes to clarify (not ask_question)."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:3])
        state.validation_errors = ["risk_tolerance: invalid value 'extreme'"]
        result = check_completion(state)

        assert result.next_action == "clarify"
        assert result.is_complete is False

    def test_all_required_filled_with_validation_errors_still_completes(self):
        """When all required slots are filled, completion overrides validation errors."""
        state = make_state()
        state = fill_all_required(state)
        state.validation_errors = ["some_optional_slot: some_error"]
        result = check_completion(state)

        # Completion takes precedence — profile is done
        assert result.is_complete is True
        assert result.next_action == "complete"

    def test_max_iterations_forces_complete(self):
        """When max_iterations is hit, force complete even with missing slots."""
        state = make_state()
        state.iteration_count = state.max_iterations  # hit the limit
        result = check_completion(state)

        assert result.is_complete is True
        assert result.next_action == "complete"

    def test_iteration_below_max_does_not_force_complete(self):
        state = make_state()
        state.iteration_count = state.max_iterations - 1
        result = check_completion(state)

        assert result.is_complete is False

    def test_one_slot_below_completion(self):
        """All required slots filled except one → still incomplete."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:-1])  # all but last
        result = check_completion(state)

        assert result.is_complete is False
        expected_pct = round(((len(REQUIRED_SLOTS) - 1) / len(REQUIRED_SLOTS)) * 100, 1)
        assert result.completion_percentage == pytest.approx(expected_pct, abs=0.2)

    def test_completion_percentage_rounds_to_one_decimal(self):
        """Completion percentage must be rounded to 1 decimal place."""
        state = make_state()
        # 1 of 9 = 11.111...% → should be 11.1
        state = fill_slots(state, [REQUIRED_SLOTS[0]])
        result = check_completion(state)

        pct_str = str(result.completion_percentage)
        decimal_places = len(pct_str.split(".")[-1]) if "." in pct_str else 0
        assert decimal_places <= 1

    def test_optional_slots_do_not_affect_is_complete(self):
        """Filling optional slots only must not set is_complete=True."""
        optional_slots = ["previous_investments", "target_return", "esg_preference"]
        state = make_state()
        state = fill_slots(state, optional_slots)
        result = check_completion(state)

        assert result.is_complete is False
        assert result.completion_percentage == 0.0

    def test_confirmed_status_counts_as_filled(self):
        """Slots with CONFIRMED status must count the same as FILLED for completion."""
        state = make_state()
        state = fill_all_required(state)
        # Override last slot to CONFIRMED status
        last_slot = getattr(state.slots, REQUIRED_SLOTS[-1])
        last_slot.status = SlotStatus.CONFIRMED
        result = check_completion(state)

        assert result.is_complete is True
        assert result.completion_percentage == 100.0

    def test_invalid_status_does_not_count_as_filled(self):
        """Slots with INVALID status must not count toward completion."""
        state = make_state()
        state = fill_all_required(state)
        # Mark one slot as INVALID (validation failed)
        first_slot = getattr(state.slots, REQUIRED_SLOTS[0])
        first_slot.status = SlotStatus.INVALID

        result = check_completion(state)

        assert result.is_complete is False
        assert result.completion_percentage < 100.0


# ── should_continue ───────────────────────────────────────────────────────────


class TestShouldContinue:
    """Tests for the should_continue() routing function used as a LangGraph conditional edge."""

    def test_empty_state_routes_to_ask_question(self):
        state = make_state()
        result = should_continue(state)
        assert result == "ask_question"

    def test_is_complete_true_routes_to_complete(self):
        state = make_state()
        state.is_complete = True
        result = should_continue(state)
        assert result == "complete"

    def test_max_iterations_routes_to_complete(self):
        state = make_state()
        state.iteration_count = state.max_iterations
        result = should_continue(state)
        assert result == "complete"

    def test_validation_errors_routes_to_clarify(self):
        state = make_state()
        state.validation_errors = ["loss_comfort: value 15 out of range [1, 10]"]
        result = should_continue(state)
        assert result == "clarify"

    def test_all_required_slots_filled_routes_to_complete(self):
        """Even without is_complete=True, if no missing required slots → complete."""
        state = make_state()
        state = fill_all_required(state)
        # Do NOT run check_completion — should_continue checks independently
        result = should_continue(state)
        assert result == "complete"

    def test_partial_slots_no_errors_routes_to_ask_question(self):
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:3])
        result = should_continue(state)
        assert result == "ask_question"

    def test_is_complete_takes_priority_over_validation_errors(self):
        """If is_complete=True (max_iterations hit), ignore validation errors."""
        state = make_state()
        state.is_complete = True
        state.validation_errors = ["some_error"]
        result = should_continue(state)
        assert result == "complete"

    def test_route_values_are_valid_node_names(self):
        """Return values must be one of the three expected LangGraph node names."""
        valid_routes = {"ask_question", "clarify", "complete"}

        # Test all three branches
        empty_state = make_state()
        assert should_continue(empty_state) in valid_routes

        error_state = make_state()
        error_state.validation_errors = ["some_error"]
        assert should_continue(error_state) in valid_routes

        complete_state = make_state()
        complete_state = fill_all_required(complete_state)
        assert should_continue(complete_state) in valid_routes

    def test_empty_validation_errors_list_does_not_trigger_clarify(self):
        """An empty list of validation errors must NOT trigger clarify route."""
        state = make_state()
        state.validation_errors = []  # explicitly empty
        result = should_continue(state)
        assert result == "ask_question"  # back to asking questions


# ── State transitions: check_completion + should_continue together ────────────


class TestCompletionStateTransitions:
    """Integration: verify the complete flow from state → check → route."""

    @pytest.mark.parametrize("n_slots", [0, 1, 3, 6, 8])
    def test_partial_fill_always_routes_to_ask_question(self, n_slots):
        """Any partial fill (< 9 required) without errors → ask_question."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:n_slots])
        state = check_completion(state)
        route = should_continue(state)
        assert route == "ask_question"

    def test_full_fill_routes_to_complete(self):
        state = make_state()
        state = fill_all_required(state)
        state = check_completion(state)
        route = should_continue(state)
        assert route == "complete"

    def test_validation_error_clears_and_routes_back_to_ask(self):
        """After validation errors are cleared, routing returns to ask_question."""
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:3])
        state.validation_errors = ["loss_comfort: out of range"]
        state = check_completion(state)
        assert should_continue(state) == "clarify"

        # Simulate error being resolved
        state.validation_errors = []
        state = check_completion(state)
        assert should_continue(state) == "ask_question"

    def test_completion_percentage_monotonically_increases(self):
        """Completion percentage must never decrease as we fill slots."""
        state = make_state()
        previous_pct = -1.0
        for slot_name in REQUIRED_SLOTS:
            state = fill_slots(state, [slot_name])
            state = check_completion(state)
            assert state.completion_percentage >= previous_pct
            previous_pct = state.completion_percentage

    def test_final_completion_percentage_is_100(self):
        state = make_state()
        for slot_name in REQUIRED_SLOTS:
            state = fill_slots(state, [slot_name])
        state = check_completion(state)
        assert state.completion_percentage == 100.0
        assert state.is_complete is True


# ── get_profile_summary ───────────────────────────────────────────────────────


class TestGetProfileSummary:
    """Tests for the get_profile_summary() helper function."""

    def test_empty_state_summary(self):
        state = make_state()
        summary = get_profile_summary(state)

        assert summary["filled"] == {}
        assert len(summary["missing"]) == len(REQUIRED_SLOTS)
        assert summary["completion_percentage"] == 0.0
        assert summary["is_complete"] is False
        assert summary["total_messages"] == 0

    def test_full_state_summary(self):
        state = make_state()
        state = fill_all_required(state)
        state = check_completion(state)
        summary = get_profile_summary(state)

        assert len(summary["filled"]) == len(REQUIRED_SLOTS)
        assert summary["missing"] == []
        assert summary["completion_percentage"] == 100.0
        assert summary["is_complete"] is True

    def test_partial_state_summary_filled_and_missing(self):
        state = make_state()
        state = fill_slots(state, REQUIRED_SLOTS[:3])
        summary = get_profile_summary(state)

        assert len(summary["filled"]) == 3
        assert len(summary["missing"]) == len(REQUIRED_SLOTS) - 3

    def test_total_messages_counts_correctly(self):
        from backend.src.profiling.schemas import Message

        state = make_state()
        state.messages = [
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="I want moderate risk"),
            Message(role="assistant", content="Got it."),
            Message(role="user", content="About 10 years"),
        ]
        summary = get_profile_summary(state)
        assert summary["total_messages"] == 4

    def test_optional_slots_not_in_missing_list(self):
        """Optional slots must not appear in the 'missing' list even when empty."""
        state = make_state()
        summary = get_profile_summary(state)

        optional_display_names = {"Previous Investment Types", "Target Annual Return", "ESG Investing"}
        for name in optional_display_names:
            assert name not in summary["missing"], (
                f"Optional slot '{name}' must not appear in missing list"
            )
