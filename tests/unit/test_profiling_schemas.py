"""Unit tests for profiling schemas."""

import pytest
from backend.src.profiling.schemas import (
    ConversationState,
    Message,
    ProfileSlot,
    ProfileSlots,
    SlotStatus,
    SLOT_VALIDATION_RULES,
)


class TestSlotStatus:
    """Tests for SlotStatus enum."""

    def test_slot_status_values(self):
        """Test that all expected status values exist."""
        assert SlotStatus.EMPTY == "empty"
        assert SlotStatus.FILLED == "filled"
        assert SlotStatus.INVALID == "invalid"
        assert SlotStatus.CONFIRMED == "confirmed"


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="How can I help?")
        assert msg.role == "assistant"
        assert msg.content == "How can I help?"


class TestProfileSlot:
    """Tests for ProfileSlot model."""

    def test_create_slot(self):
        """Test creating a profile slot."""
        slot = ProfileSlot(
            name="risk_tolerance",
            display_name="Risk Tolerance",
            description="How comfortable are you with risk?",
        )
        assert slot.name == "risk_tolerance"
        assert slot.status == SlotStatus.EMPTY
        assert slot.value is None
        assert slot.required is True

    def test_slot_with_question(self):
        """Test creating a slot with a question."""
        slot = ProfileSlot(
            name="experience_level",
            display_name="Experience",
            description="Your investment experience",
            question="How much experience do you have?",
        )
        assert slot.question == "How much experience do you have?"

    def test_optional_slot(self):
        """Test creating an optional slot."""
        slot = ProfileSlot(
            name="target_return",
            display_name="Target Return",
            description="Your target return",
            required=False,
        )
        assert slot.required is False


class TestProfileSlots:
    """Tests for ProfileSlots model."""

    def test_default_slots_created(self):
        """Test that default slots are created."""
        slots = ProfileSlots()

        # Check required slots exist
        assert hasattr(slots, "risk_tolerance")
        assert hasattr(slots, "investment_horizon")
        assert hasattr(slots, "experience_level")
        assert hasattr(slots, "primary_goal")

        # Check optional slots exist
        assert hasattr(slots, "target_return")
        assert hasattr(slots, "esg_preference")

    def test_slots_have_questions(self):
        """Test that all slots have questions defined."""
        slots = ProfileSlots()

        for field_name in ProfileSlots.model_fields:
            slot = getattr(slots, field_name)
            assert slot.question is not None, f"Slot {field_name} has no question"

    def test_required_slots(self):
        """Test required vs optional slots."""
        slots = ProfileSlots()

        required_slots = [
            "risk_tolerance",
            "loss_comfort",
            "investment_horizon",
            "liquidity_needs",
            "income_stability",
            "has_emergency_fund",
            "debt_level",
            "experience_level",
            "primary_goal",
        ]

        optional_slots = [
            "previous_investments",
            "target_return",
            "esg_preference",
        ]

        for slot_name in required_slots:
            slot = getattr(slots, slot_name)
            assert slot.required is True, f"Slot {slot_name} should be required"

        for slot_name in optional_slots:
            slot = getattr(slots, slot_name)
            assert slot.required is False, f"Slot {slot_name} should be optional"


class TestConversationState:
    """Tests for ConversationState model."""

    def test_create_state(self):
        """Test creating a conversation state."""
        state = ConversationState(user_id="user123")

        assert state.user_id == "user123"
        assert state.messages == []
        assert state.is_complete is False
        assert state.completion_percentage == 0.0

    def test_add_message(self):
        """Test adding messages to state."""
        state = ConversationState(user_id="user123")

        state.messages.append(Message(role="assistant", content="Hello!"))
        state.messages.append(Message(role="user", content="Hi there"))

        assert len(state.messages) == 2
        assert state.messages[0].role == "assistant"
        assert state.messages[1].role == "user"

    def test_get_filled_slots(self):
        """Test getting filled slots."""
        state = ConversationState(user_id="user123")

        # Initially no slots filled
        assert state.get_filled_slots() == []

        # Fill a slot
        state.slots.risk_tolerance.value = "moderate"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        filled = state.get_filled_slots()
        assert "risk_tolerance" in filled

    def test_get_missing_required_slots(self):
        """Test getting missing required slots."""
        state = ConversationState(user_id="user123")

        # All required slots should be missing initially
        missing = state.get_missing_required_slots()
        assert len(missing) > 0
        assert "risk_tolerance" in missing
        assert "investment_horizon" in missing

        # Fill a slot
        state.slots.risk_tolerance.value = "moderate"
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        missing = state.get_missing_required_slots()
        assert "risk_tolerance" not in missing

    def test_get_next_empty_slot(self):
        """Test getting the next empty slot."""
        state = ConversationState(user_id="user123")

        # Should return a required slot first
        next_slot = state.get_next_empty_slot()
        assert next_slot is not None
        assert next_slot.required is True

    def test_calculate_completion(self):
        """Test completion calculation."""
        state = ConversationState(user_id="user123")

        # Initially 0%
        assert state.calculate_completion() == 0.0

        # Fill one required slot
        state.slots.risk_tolerance.status = SlotStatus.FILLED

        completion = state.calculate_completion()
        assert completion > 0
        assert completion < 100

    def test_to_investment_profile(self):
        """Test converting state to investment profile dict."""
        state = ConversationState(user_id="user123")

        state.slots.risk_tolerance.value = "moderate"
        state.slots.investment_horizon.value = "long"

        profile = state.to_investment_profile()

        assert profile["user_id"] == "user123"
        assert profile["risk_tolerance"] == "moderate"
        assert profile["investment_horizon"] == "long"


class TestSlotValidationRules:
    """Tests for slot validation rules."""

    def test_rules_exist_for_all_slots(self):
        """Test that validation rules exist for all slots."""
        slots = ProfileSlots()

        for field_name in ProfileSlots.model_fields:
            assert field_name in SLOT_VALIDATION_RULES, f"No rules for {field_name}"

    def test_enum_rules_have_values(self):
        """Test that enum rules have valid values lists."""
        for slot_name, rules in SLOT_VALIDATION_RULES.items():
            if rules.get("type") == "enum":
                assert "values" in rules, f"Enum rule for {slot_name} has no values"
                assert len(rules["values"]) > 0, f"Enum rule for {slot_name} has empty values"

    def test_range_rules_have_bounds(self):
        """Test that range rules have min/max bounds."""
        for slot_name, rules in SLOT_VALIDATION_RULES.items():
            if rules.get("type") in ["int_range", "float_range"]:
                assert "min" in rules, f"Range rule for {slot_name} has no min"
                assert "max" in rules, f"Range rule for {slot_name} has no max"
                assert rules["min"] < rules["max"], f"Invalid range for {slot_name}"
