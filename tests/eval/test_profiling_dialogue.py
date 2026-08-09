"""Evaluation: Profiling Agent — multi-turn conversation simulation.

Phase 2 scope — offline, scripted LLM responses:
  - Tests start_conversation() initial state and greeting.
  - Tests single-turn process_response() slot extraction and state update.
  - Tests multi-turn dialogue: simulates 9 turns to reach 100% completion.
  - Tests validation error recovery: invalid value → clarify → correct.
  - Tests max_iterations graceful exit.
  - Tests resume_conversation() with pre-populated slots.
  - Tests export_profile() after completion.

Key design: The LLM mock uses a queue (side_effect list) that returns
scripted responses in order. Each process_response() call makes:
  - 1 call for extract_profile_info() → must return JSON
  - 1 call for generate_question() or generate_completion_message() → must return a string
so every turn consumes exactly 2 queued LLM responses.

Run with:
    pytest tests/eval/test_profiling_dialogue.py -v -s
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.src.profiling.agent import ProfilingAgent
from backend.src.profiling.schemas import ConversationState, SlotStatus

from .conftest import REQUIRED_SLOTS, SLOT_FILL_VALUES, make_state


# ── Helpers ──────────────────────────────────────────────────────────────────


class _LLMResponse:
    """Minimal response object returned by the mock LLM."""

    def __init__(self, content: str):
        self.content = content


def _make_extraction_json(**slots) -> str:
    """Format a slot dict as a JSON string for the extractor LLM response."""
    import json
    return json.dumps(slots)


def _make_scripted_llm(responses: list[str]) -> MagicMock:
    """Return a MagicMock LLM whose .chat() returns responses in order."""
    llm = MagicMock()
    llm.chat.side_effect = [_LLMResponse(r) for r in responses]
    return llm


# LLM call pattern per turn in process_response():
#   extract_profile_info  → always 1 LLM call
#   generate_question     → 1 LLM call ONLY when len(state.messages) > 1
#   generate_clarification → NO LLM call (rule-based)
#
# Turn 1: state starts empty, len(messages)==1 after user append
#   → generate_question uses default slot question (no LLM)
#   → total: 1 LLM call (extract only)
# Turns 2-10: len(messages) > 1
#   → extract + question = 2 LLM calls
# Turn 11 (final): extract fills last slot → complete → generate_completion_message
#   → extract + completion = 2 LLM calls
# Total: 1 + 10 * 2 = 21 LLM calls for 11 turns

_FULL_DIALOGUE_RESPONSES: list[str] = [
    # Turn 1: extract only (question uses default slot text, no LLM call)
    _make_extraction_json(risk_tolerance="moderate"),
    # Turn 2: extract + question
    _make_extraction_json(investment_horizon="long"),
    "How stable is your current income?",
    # Turn 3
    _make_extraction_json(income_stability="stable"),
    "Do you have an emergency fund covering 3-6 months of expenses?",
    # Turn 4
    _make_extraction_json(has_emergency_fund=True),
    "How would you describe your current debt level?",
    # Turn 5
    _make_extraction_json(debt_level="low"),
    "How much investment experience do you have?",
    # Turn 6
    _make_extraction_json(experience_level="intermediate"),
    "How quickly might you need to access your invested money?",
    # Turn 7
    _make_extraction_json(liquidity_needs="low"),
    "What's your primary financial goal?",
    # Turn 8
    _make_extraction_json(primary_goal="retirement"),
    "On a scale of 1-10, how comfortable are you with potential losses?",
    # Turn 9
    _make_extraction_json(loss_comfort=5),
    "What is your approximate monthly income?",
    # Turn 10
    _make_extraction_json(monthly_income=7000.0),
    "And roughly how much are your monthly expenses?",
    # Turn 11: extract fills last required slot → complete → completion message (not question)
    _make_extraction_json(monthly_expenses=4500.0),
    "Your profile is complete! Great work.",
]

# User messages for each turn (content doesn't matter — LLM is scripted)
_USER_MESSAGES = [
    "I'd say moderate risk",
    "Long term, about 20 years",
    "My income is stable, I have a salaried job",
    "Yes, I have 6 months saved up",
    "My debt is low, just a small car loan",
    "I have some experience with stocks and ETFs",
    "My liquidity needs are low",
    "I'm saving for retirement",
    "I'm comfortable with about a 5 out of 10",
    "My monthly income is about $7,000",
    "My monthly expenses are around $4,500",
]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def greeting_llm() -> MagicMock:
    """LLM that returns a standard greeting for start_conversation()."""
    return _make_scripted_llm(
        ["Hello! I'm here to help build your investment profile. Let's get started!"]
    )


@pytest.fixture
def agent_greeting(greeting_llm) -> ProfilingAgent:
    """Agent with scripted greeting LLM."""
    return ProfilingAgent(llm_client=greeting_llm)


# ── start_conversation ────────────────────────────────────────────────────────


class TestStartConversation:
    """start_conversation initializes state and generates a greeting."""

    def test_returns_conversation_state(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert isinstance(state, ConversationState)

    def test_user_id_is_set(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert state.user_id == "eval_user"

    def test_initial_completion_is_zero(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert state.completion_percentage == 0.0

    def test_is_not_complete_initially(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert state.is_complete is False

    def test_greeting_message_added_to_state(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert len(state.messages) >= 1
        assert state.messages[0].role == "assistant"

    def test_get_last_assistant_message_returns_greeting(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        msg = agent_greeting.get_last_assistant_message(state)
        assert msg is not None
        assert len(msg) > 0

    def test_no_slots_filled_initially(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert len(state.get_filled_slots()) == 0

    def test_all_required_slots_missing_initially(self, agent_greeting):
        state = agent_greeting.start_conversation("eval_user")
        assert len(state.get_missing_required_slots()) == len(REQUIRED_SLOTS)


# ── single-turn process_response ──────────────────────────────────────────────


class TestSingleTurnExtraction:
    """process_response fills exactly the slots mentioned in the LLM extraction."""

    def test_risk_tolerance_slot_filled_after_first_turn(self):
        """First turn: extraction JSON has risk_tolerance → slot is FILLED."""
        llm = _make_scripted_llm([
            _make_extraction_json(risk_tolerance="moderate"),
            "How long do you plan to invest?",  # follow-up question
        ])
        agent = ProfilingAgent(llm_client=llm)

        # Simulate start (no greeting LLM call needed — we build state manually)
        state = ConversationState(user_id="test_user")
        state = agent.process_response(state, "I prefer moderate risk")

        assert state.slots.risk_tolerance.status == SlotStatus.FILLED
        assert state.slots.risk_tolerance.value == "moderate"

    def test_completion_percentage_increases_after_extraction(self):
        """Completion % must increase after a slot is filled."""
        llm = _make_scripted_llm([
            _make_extraction_json(investment_horizon="long"),
            "Got it. What about liquidity?",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state = agent.process_response(state, "Long term, about 20 years")

        assert state.completion_percentage > 0.0

    def test_assistant_message_added_after_turn(self):
        """process_response must add one assistant message (the next question)."""
        llm = _make_scripted_llm([
            _make_extraction_json(debt_level="low"),
            "What's your investment experience?",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        initial_msg_count = len([m for m in state.messages if m.role == "assistant"])
        state = agent.process_response(state, "I have very little debt")

        final_msg_count = len([m for m in state.messages if m.role == "assistant"])
        assert final_msg_count > initial_msg_count

    def test_user_message_added_to_history(self):
        """The user's message must appear in state.messages after process_response."""
        llm = _make_scripted_llm([
            _make_extraction_json(risk_tolerance="conservative"),
            "Next question here",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state = agent.process_response(state, "I am very risk-averse")

        user_msgs = [m for m in state.messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert "risk-averse" in user_msgs[0].content

    def test_extraction_with_no_json_does_not_crash(self):
        """If the LLM returns garbage for extraction, the agent must not crash."""
        llm = _make_scripted_llm([
            "This is not valid JSON at all!",
            "Could you please clarify?",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        result = agent.process_response(state, "I like moderate risk")
        # Should complete gracefully with 0 new slots filled
        assert result is not None

    def test_null_value_in_extraction_does_not_overwrite_existing(self):
        """Null extraction values must not overwrite already-filled slots."""
        # Pre-fill risk_tolerance
        llm = _make_scripted_llm([
            _make_extraction_json(risk_tolerance=None, investment_horizon="medium"),
            "What's your income situation?",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state.slots.risk_tolerance.value = "aggressive"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        state = agent.process_response(state, "Medium term goals")
        # risk_tolerance must still be aggressive (null extraction skipped)
        assert state.slots.risk_tolerance.value == "aggressive"
        # investment_horizon should be filled
        assert state.slots.investment_horizon.status == SlotStatus.FILLED


# ── multi-turn completion ─────────────────────────────────────────────────────


class TestMultiTurnCompletion:
    """Simulate a full 11-turn dialogue to reach 100% profile completion."""

    @pytest.fixture
    def completed_state(self) -> ConversationState:
        """Run 11 scripted turns and return the final state."""
        llm = _make_scripted_llm(_FULL_DIALOGUE_RESPONSES)
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="eval_user")
        for user_msg in _USER_MESSAGES:
            state = agent.process_response(state, user_msg)
        return state

    def test_profile_is_complete_after_all_turns(self, completed_state):
        assert completed_state.is_complete is True

    def test_completion_percentage_is_100(self, completed_state):
        assert completed_state.completion_percentage == pytest.approx(100.0, abs=1.0)

    def test_all_required_slots_are_filled(self, completed_state):
        missing = completed_state.get_missing_required_slots()
        assert missing == [], f"Still missing required slots: {missing}"

    def test_no_remaining_required_slots(self, completed_state):
        assert len(completed_state.get_missing_required_slots()) == 0

    def test_completion_percentage_was_monotonically_increasing(self):
        """Completion % must never decrease turn-over-turn."""
        llm = _make_scripted_llm(_FULL_DIALOGUE_RESPONSES)
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="eval_user")
        prev_pct = 0.0
        for user_msg in _USER_MESSAGES:
            state = agent.process_response(state, user_msg)
            assert state.completion_percentage >= prev_pct
            prev_pct = state.completion_percentage

    def test_message_history_has_correct_length(self, completed_state):
        """9 user messages + N assistant messages must all be present."""
        user_msgs = [m for m in completed_state.messages if m.role == "user"]
        assert len(user_msgs) == len(_USER_MESSAGES)

    def test_risk_tolerance_correct_value(self, completed_state):
        assert completed_state.slots.risk_tolerance.value == "moderate"

    def test_investment_horizon_correct_value(self, completed_state):
        assert completed_state.slots.investment_horizon.value == "long"


# ── validation error recovery ─────────────────────────────────────────────────


class TestValidationErrorRecovery:
    """Invalid extracted values generate a clarification; next turn they correct."""

    def test_invalid_value_does_not_fill_slot(self):
        """Slot must NOT be filled if the extracted value fails validation."""
        # loss_comfort must be 1-10; "fifty" is invalid
        llm = _make_scripted_llm([
            _make_extraction_json(loss_comfort="fifty"),  # invalid int
            "Could you say a number between 1 and 10?",
        ])
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")
        state = agent.process_response(state, "About fifty")
        assert state.slots.loss_comfort.status != SlotStatus.FILLED

    def test_corrected_value_fills_slot_on_next_turn(self):
        """After clarification, a valid response must fill the slot.

        generate_clarification is rule-based (no LLM call), so turn 1 uses
        only 1 LLM call (extract). Turn 2 uses 2 calls (extract + question).
        """
        llm = _make_scripted_llm([
            _make_extraction_json(loss_comfort="fifty"),   # turn 1: extract (invalid)
            # no LLM for clarification (rule-based)
            _make_extraction_json(loss_comfort=7),         # turn 2: extract (valid)
            "Great! What's your primary investment goal?",  # turn 2: question
        ])
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")

        state = agent.process_response(state, "About fifty")
        assert state.slots.loss_comfort.status != SlotStatus.FILLED

        state = agent.process_response(state, "I'd say 7")
        assert state.slots.loss_comfort.status == SlotStatus.FILLED
        assert state.slots.loss_comfort.value == 7

    def test_validation_errors_cleared_on_valid_retry(self):
        """After a valid extraction, state.validation_errors must be empty.

        Turn 1: extract "extreme" → validation fails → clarify (no LLM) = 1 call.
        Turn 2: extract "aggressive" → valid → question (LLM) = 2 calls.
        """
        llm = _make_scripted_llm([
            _make_extraction_json(risk_tolerance="extreme"),    # turn 1: extract (invalid)
            # no LLM for clarification (rule-based)
            _make_extraction_json(risk_tolerance="aggressive"), # turn 2: extract (valid)
            "What's your time horizon?",                        # turn 2: question
        ])
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")

        state = agent.process_response(state, "Extreme risk")
        state = agent.process_response(state, "I'm comfortable with high risk")
        assert state.validation_errors == []
        assert state.slots.risk_tolerance.value == "aggressive"

    def test_valid_enum_value_passes_validation(self):
        """Check that all canonical enum values are accepted without errors."""
        valid_cases = [
            ("risk_tolerance", "conservative"),
            ("risk_tolerance", "moderate"),
            ("risk_tolerance", "aggressive"),
            ("investment_horizon", "short"),
            ("investment_horizon", "medium"),
            ("investment_horizon", "long"),
            ("debt_level", "none"),
            ("debt_level", "low"),
            ("experience_level", "beginner"),
            ("experience_level", "intermediate"),
            ("experience_level", "advanced"),
        ]
        for slot_name, value in valid_cases:
            llm = _make_scripted_llm([
                _make_extraction_json(**{slot_name: value}),
                "Next question.",
            ])
            agent = ProfilingAgent(llm_client=llm)
            state = ConversationState(user_id="test_user")
            state = agent.process_response(state, f"My answer is {value}")
            slot = getattr(state.slots, slot_name)
            assert slot.status == SlotStatus.FILLED, (
                f"Expected {slot_name}={value!r} to pass validation, but status is {slot.status}"
            )


# ── max iterations graceful exit ──────────────────────────────────────────────


class TestMaxIterationsExit:
    """When iteration_count >= max_iterations, the agent must complete gracefully."""

    def test_at_max_iterations_state_is_complete(self):
        """Even with slots missing, hitting max_iterations forces completion."""
        llm = _make_scripted_llm([
            _make_extraction_json(risk_tolerance="moderate"),
            "Your session is complete. Thank you!",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state.iteration_count = state.max_iterations  # pre-hit the limit

        state = agent.process_response(state, "moderate risk please")
        assert state.is_complete is True

    def test_at_max_iterations_missing_slots_accepted(self):
        """Missing required slots must not block completion at max_iterations."""
        llm = _make_scripted_llm([
            _make_extraction_json(),   # no slots extracted
            "Thank you for your time.",
        ])
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state.iteration_count = state.max_iterations

        state = agent.process_response(state, "I'm done for now")
        assert state.is_complete is True


# ── resume_conversation ───────────────────────────────────────────────────────


class TestResumeConversation:
    """resume_conversation pre-populates slots from an existing InvestmentProfile."""

    @pytest.fixture
    def resume_llm(self) -> MagicMock:
        return _make_scripted_llm(
            ["Welcome back! We have some profile info already. Let's continue."]
        )

    def test_resume_pre_populates_risk_tolerance(self, resume_llm):
        from backend.src.common.models import InvestmentProfile, RiskTolerance

        agent = ProfilingAgent(llm_client=resume_llm)
        profile = InvestmentProfile(user_id="test_user", risk_tolerance=RiskTolerance.CONSERVATIVE)
        state = agent.resume_conversation("test_user", "Alex", profile)

        assert state.slots.risk_tolerance.status == SlotStatus.FILLED
        assert state.slots.risk_tolerance.value == "conservative"

    def test_resume_completion_percentage_reflects_pre_fills(self, resume_llm):
        from backend.src.common.models import InvestmentProfile, RiskTolerance

        agent = ProfilingAgent(llm_client=resume_llm)
        profile = InvestmentProfile(user_id="test_user", risk_tolerance=RiskTolerance.AGGRESSIVE)
        state = agent.resume_conversation("test_user", "Alex", profile)

        assert state.completion_percentage > 0.0

    def test_resume_state_is_not_complete_with_partial_profile(self, resume_llm):
        from backend.src.common.models import InvestmentProfile, RiskTolerance

        agent = ProfilingAgent(llm_client=resume_llm)
        profile = InvestmentProfile(user_id="test_user", risk_tolerance=RiskTolerance.MODERATE)
        state = agent.resume_conversation("test_user", "Alex", profile)

        # Only 1 of 9 required slots filled → not complete
        assert state.is_complete is False

    def test_resume_greeting_message_added(self, resume_llm):
        from backend.src.common.models import InvestmentProfile

        agent = ProfilingAgent(llm_client=resume_llm)
        profile = InvestmentProfile(user_id="test_user")
        state = agent.resume_conversation("test_user", "Alex", profile)

        assert len(state.messages) >= 1
        assert state.messages[0].role == "assistant"


# ── export_profile ────────────────────────────────────────────────────────────


class TestExportProfile:
    """export_profile returns a dict with all filled slot values."""

    def test_export_contains_all_filled_required_slots(self):
        """After completing all required slots, export must include every one."""
        from .conftest import fill_all_required

        llm = MagicMock()  # not called
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state = fill_all_required(state)
        state.is_complete = True

        profile_dict = agent.export_profile(state)
        for slot_name in REQUIRED_SLOTS:
            assert slot_name in profile_dict, (
                f"Required slot '{slot_name}' missing from export_profile() output"
            )

    def test_export_values_match_what_was_filled(self):
        from .conftest import fill_all_required, SLOT_FILL_VALUES

        llm = MagicMock()
        agent = ProfilingAgent(llm_client=llm)

        state = ConversationState(user_id="test_user")
        state = fill_all_required(state)

        profile_dict = agent.export_profile(state)
        for slot_name, expected_value in SLOT_FILL_VALUES.items():
            if slot_name in profile_dict:
                assert profile_dict[slot_name] == expected_value, (
                    f"Slot '{slot_name}': expected {expected_value!r}, "
                    f"got {profile_dict[slot_name]!r}"
                )

    def test_export_empty_state_has_no_slot_values(self):
        """to_investment_profile() on an empty state has user_id but no slot values."""
        llm = MagicMock()
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")
        profile_dict = agent.export_profile(state)
        # user_id is always present; no slot data should be present
        assert profile_dict.get("user_id") == "test_user"
        for slot_name in ["risk_tolerance", "investment_horizon", "loss_comfort"]:
            assert slot_name not in profile_dict


# ── get_profile_summary ───────────────────────────────────────────────────────


class TestAgentGetProfileSummary:
    """ProfilingAgent.get_profile_summary returns a well-structured dict."""

    def test_summary_keys_present(self):
        llm = MagicMock()
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")
        summary = agent.get_profile_summary(state)

        assert "user_id" in summary
        assert "filled_slots" in summary
        assert "missing_required" in summary
        assert "completion_percentage" in summary
        assert "is_complete" in summary

    def test_summary_user_id_matches(self):
        llm = MagicMock()
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="user_xyz")
        summary = agent.get_profile_summary(state)
        assert summary["user_id"] == "user_xyz"

    def test_summary_conversation_turns_counts_user_messages(self):
        from backend.src.profiling.schemas import Message

        llm = MagicMock()
        agent = ProfilingAgent(llm_client=llm)
        state = ConversationState(user_id="test_user")
        state.messages = [
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="Hi there"),
            Message(role="assistant", content="Question 2"),
            Message(role="user", content="Answer 2"),
        ]
        summary = agent.get_profile_summary(state)
        assert summary["conversation_turns"] == 2
