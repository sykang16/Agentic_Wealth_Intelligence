"""Unit tests for profiling validator node."""

import pytest
from backend.src.profiling.nodes.validator import _validate_value, validate_extracted_values
from backend.src.profiling.schemas import ConversationState, SlotStatus


class TestValidateValue:
    """Tests for the _validate_value function."""

    def test_validate_risk_tolerance_exact(self):
        """Test validating exact risk tolerance values."""
        value, error = _validate_value("risk_tolerance", "conservative")
        assert value == "conservative"
        assert error is None

        value, error = _validate_value("risk_tolerance", "moderate")
        assert value == "moderate"
        assert error is None

        value, error = _validate_value("risk_tolerance", "aggressive")
        assert value == "aggressive"
        assert error is None

    def test_validate_risk_tolerance_mappings(self):
        """Test mapping common phrases to risk tolerance values."""
        mappings = [
            ("safe", "conservative"),
            ("low risk", "conservative"),
            ("cautious", "conservative"),
            ("balanced", "moderate"),
            ("high risk", "aggressive"),
            ("risky", "aggressive"),
        ]

        for input_val, expected in mappings:
            value, error = _validate_value("risk_tolerance", input_val)
            assert value == expected, f"'{input_val}' should map to '{expected}'"
            assert error is None

    def test_validate_risk_tolerance_invalid(self):
        """Test invalid risk tolerance values."""
        value, error = _validate_value("risk_tolerance", "extreme")
        assert value is None
        assert error is not None

    def test_validate_loss_comfort_valid(self):
        """Test valid loss comfort values."""
        for i in range(1, 11):
            value, error = _validate_value("loss_comfort", i)
            assert value == i
            assert error is None

        # String numbers should also work
        value, error = _validate_value("loss_comfort", "5")
        assert value == 5
        assert error is None

    def test_validate_loss_comfort_out_of_range(self):
        """Test loss comfort values out of range."""
        value, error = _validate_value("loss_comfort", 0)
        assert value is None
        assert error is not None

        value, error = _validate_value("loss_comfort", 11)
        assert value is None
        assert error is not None

    def test_validate_investment_horizon(self):
        """Test investment horizon validation."""
        # Exact values
        value, error = _validate_value("investment_horizon", "short")
        assert value == "short"
        assert error is None

        # Mappings
        value, error = _validate_value("investment_horizon", "long term")
        assert value == "long"
        assert error is None

        value, error = _validate_value("investment_horizon", "retirement")
        assert value == "long"
        assert error is None

    def test_validate_boolean_yes(self):
        """Test boolean validation for yes answers."""
        yes_values = ["yes", "yeah", "yep", "true", "i do", "definitely", "sure"]

        for val in yes_values:
            value, error = _validate_value("has_emergency_fund", val)
            assert value is True, f"'{val}' should be True"
            assert error is None

    def test_validate_boolean_no(self):
        """Test boolean validation for no answers."""
        no_values = ["no", "nope", "false", "i don't", "not really"]

        for val in no_values:
            value, error = _validate_value("has_emergency_fund", val)
            assert value is False, f"'{val}' should be False"
            assert error is None

    def test_validate_list_string(self):
        """Test list string validation."""
        # List input
        value, error = _validate_value("previous_investments", ["stocks", "bonds"])
        assert value == ["stocks", "bonds"]
        assert error is None

        # String with commas
        value, error = _validate_value("previous_investments", "stocks, bonds, ETFs")
        assert "stocks" in value
        assert "bonds" in value
        assert "ETFs" in value
        assert error is None

        # String with "and"
        value, error = _validate_value("previous_investments", "stocks and bonds")
        assert "stocks" in value
        assert "bonds" in value
        assert error is None

    def test_validate_string(self):
        """Test string validation."""
        value, error = _validate_value("primary_goal", "retirement savings")
        assert value == "retirement savings"
        assert error is None

    def test_validate_float_range(self):
        """Test float range validation."""
        value, error = _validate_value("target_return", 7.5)
        assert value == 7.5
        assert error is None

        value, error = _validate_value("target_return", "8")
        assert value == 8.0
        assert error is None

        # Out of range
        value, error = _validate_value("target_return", 150)
        assert value is None
        assert error is not None

    def test_validate_null_value(self):
        """Test that null values pass through."""
        value, error = _validate_value("risk_tolerance", None)
        assert value is None
        assert error is None

    def test_validate_debt_level(self):
        """Test debt level validation with mappings."""
        mappings = [
            ("none", "none"),
            ("zero", "none"),
            ("no debt", "none"),
            ("little", "low"),
            ("minimal", "low"),
            ("some debt", "moderate"),
            ("significant", "high"),
            ("a lot", "high"),
        ]

        for input_val, expected in mappings:
            value, error = _validate_value("debt_level", input_val)
            assert value == expected, f"'{input_val}' should map to '{expected}'"

    def test_validate_experience_level(self):
        """Test experience level validation with mappings."""
        mappings = [
            ("beginner", "beginner"),
            ("new", "beginner"),
            ("novice", "beginner"),
            ("some", "intermediate"),
            ("expert", "advanced"),
            ("professional", "advanced"),
        ]

        for input_val, expected in mappings:
            value, error = _validate_value("experience_level", input_val)
            assert value == expected, f"'{input_val}' should map to '{expected}'"


class TestValidateExtractedValues:
    """Tests for validate_extracted_values function."""

    def test_validate_empty_extraction(self):
        """Test validating with no extraction."""
        state = ConversationState(user_id="user123")
        state.last_extraction = None

        result = validate_extracted_values(state)
        assert result.validation_errors == []

    def test_validate_single_value(self):
        """Test validating a single extracted value."""
        state = ConversationState(user_id="user123")
        state.last_extraction = {"risk_tolerance": "moderate"}

        result = validate_extracted_values(state)

        assert result.slots.risk_tolerance.value == "moderate"
        assert result.slots.risk_tolerance.status == SlotStatus.FILLED
        assert result.validation_errors == []

    def test_validate_multiple_values(self):
        """Test validating multiple extracted values."""
        state = ConversationState(user_id="user123")
        state.last_extraction = {
            "risk_tolerance": "conservative",
            "loss_comfort": 3,
            "investment_horizon": "long",
        }

        result = validate_extracted_values(state)

        assert result.slots.risk_tolerance.value == "conservative"
        assert result.slots.loss_comfort.value == 3
        assert result.slots.investment_horizon.value == "long"
        assert result.validation_errors == []

    def test_validate_with_errors(self):
        """Test validation with invalid values."""
        state = ConversationState(user_id="user123")
        state.last_extraction = {
            "risk_tolerance": "moderate",  # valid
            "loss_comfort": 15,  # invalid - out of range
        }

        result = validate_extracted_values(state)

        assert result.slots.risk_tolerance.value == "moderate"
        assert result.slots.loss_comfort.value is None  # Should not be set
        assert len(result.validation_errors) == 1

    def test_validate_updates_completion(self):
        """Test that validation updates completion percentage."""
        state = ConversationState(user_id="user123")
        state.last_extraction = {
            "risk_tolerance": "moderate",
            "loss_comfort": 5,
        }

        initial_completion = state.completion_percentage

        result = validate_extracted_values(state)

        assert result.completion_percentage > initial_completion
