"""Schemas for the Investment Profiling Agent.

This module defines the state schema and data models used by the
LangGraph-based profiling agent for conversational slot-filling.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.src.common.models import (
    ExperienceLevel,
    GoalType,
    InvestmentHorizon,
    Priority,
    RiskTolerance,
)


class SlotStatus(str, Enum):
    """Status of a profile slot."""

    EMPTY = "empty"
    FILLED = "filled"
    INVALID = "invalid"
    CONFIRMED = "confirmed"


class Message(BaseModel):
    """A message in the conversation."""

    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ProfileSlot(BaseModel):
    """A single slot in the investment profile with metadata."""

    name: str = Field(..., description="Slot name")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of what this slot captures")
    value: Any = Field(default=None, description="Current value")
    status: SlotStatus = Field(default=SlotStatus.EMPTY, description="Slot status")
    required: bool = Field(default=True, description="Whether this slot is required")
    question: str | None = Field(default=None, description="Question to ask for this slot")


class ProfileSlots(BaseModel):
    """All slots for the investment profile."""

    # Risk Assessment
    risk_tolerance: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="risk_tolerance",
            display_name="Risk Tolerance",
            description="How comfortable are you with investment risk?",
            question="How would you describe your risk tolerance? Are you conservative (prefer safety), moderate (balanced approach), or aggressive (comfortable with higher risk for higher returns)?",
        )
    )
    loss_comfort: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="loss_comfort",
            display_name="Loss Comfort Level",
            description="How comfortable are you with potential losses?",
            question="On a scale of 1-10, how comfortable are you with the possibility of your investments losing value in the short term? (1 = very uncomfortable, 10 = very comfortable)",
        )
    )

    # Time Horizon
    investment_horizon: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="investment_horizon",
            display_name="Investment Time Horizon",
            description="How long do you plan to keep your investments?",
            question="What's your investment time horizon? Are you investing for the short-term (less than 3 years), medium-term (3-10 years), or long-term (more than 10 years)?",
        )
    )
    liquidity_needs: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="liquidity_needs",
            display_name="Liquidity Needs",
            description="How important is having quick access to your money?",
            question="How important is it for you to have quick access to your invested funds? Would you say your liquidity needs are low, medium, or high?",
        )
    )

    # Financial Situation
    income_stability: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="income_stability",
            display_name="Income Stability",
            description="How stable is your current income?",
            question="How would you describe your income stability? Is it stable (regular salary), variable (commissions or freelance), or uncertain?",
        )
    )
    has_emergency_fund: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="has_emergency_fund",
            display_name="Emergency Fund",
            description="Do you have an emergency fund?",
            question="Do you currently have an emergency fund set aside (typically 3-6 months of expenses)?",
        )
    )
    debt_level: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="debt_level",
            display_name="Debt Level",
            description="What is your current debt level?",
            question="How would you describe your current debt level? None, low (manageable), moderate, or high?",
        )
    )

    # Investment Experience
    experience_level: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="experience_level",
            display_name="Investment Experience",
            description="How experienced are you with investing?",
            question="How would you describe your investment experience? Are you a beginner (little to no experience), intermediate (some experience with stocks/bonds), or advanced (experienced with various investment types)?",
        )
    )
    previous_investments: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="previous_investments",
            display_name="Previous Investment Types",
            description="What types of investments have you made before?",
            question="What types of investments have you made before? For example: stocks, bonds, ETFs, mutual funds, real estate, cryptocurrency, etc.",
            required=False,
        )
    )

    # Goals
    primary_goal: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="primary_goal",
            display_name="Primary Investment Goal",
            description="What is your main investment objective?",
            question="What is your primary investment goal? For example: retirement, buying a house, education funding, building wealth, or something else?",
        )
    )
    target_return: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="target_return",
            display_name="Target Annual Return",
            description="What annual return do you hope to achieve?",
            question="Do you have a target annual return in mind? If so, what percentage are you hoping to achieve?",
            required=False,
        )
    )

    # ESG Preference
    esg_preference: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="esg_preference",
            display_name="ESG Investing",
            description="Interest in sustainable/responsible investing",
            question="Are you interested in ESG (Environmental, Social, Governance) investing? This focuses on companies with sustainable and ethical practices.",
            required=False,
        )
    )

    # Financial Context (required for personalised recommendations)
    monthly_income: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="monthly_income",
            display_name="Monthly Income",
            description="Approximate monthly income in USD",
            question="To tailor recommendations to your financial capacity, could you share your approximate monthly income? A rough figure is fine (e.g. $5,000/month).",
        )
    )
    monthly_expenses: ProfileSlot = Field(
        default_factory=lambda: ProfileSlot(
            name="monthly_expenses",
            display_name="Monthly Expenses",
            description="Approximate monthly expenses in USD",
            question="And roughly how much are your monthly expenses? This helps me understand your savings potential.",
        )
    )


class ConversationState(BaseModel):
    """State for the profiling conversation managed by LangGraph."""

    # User info
    user_id: str = Field(..., description="User ID being profiled")

    # Conversation history
    messages: list[Message] = Field(default_factory=list, description="Conversation history")

    # Profile slots
    slots: ProfileSlots = Field(
        default_factory=ProfileSlots, description="Profile slots being filled"
    )

    # Workflow state
    current_slot: str | None = Field(
        default=None, description="Current slot being filled"
    )
    last_extraction: dict[str, Any] | None = Field(
        default=None, description="Last extraction result from LLM"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Current validation errors"
    )

    # Completion tracking
    is_complete: bool = Field(default=False, description="Whether profile is complete")
    completion_percentage: float = Field(
        default=0.0, description="Profile completion percentage"
    )

    # Workflow control
    next_action: str = Field(
        default="ask_question", description="Next action to take"
    )
    iteration_count: int = Field(default=0, description="Number of conversation turns")
    max_iterations: int = Field(default=30, description="Maximum conversation turns")

    def get_filled_slots(self) -> list[str]:
        """Get list of filled slot names."""
        filled = []
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                filled.append(field_name)
        return filled

    def get_missing_required_slots(self) -> list[str]:
        """Get list of missing required slot names."""
        missing = []
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if slot.required and slot.status == SlotStatus.EMPTY:
                missing.append(field_name)
        return missing

    def get_next_empty_slot(self) -> ProfileSlot | None:
        """Get the next empty required slot to fill."""
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if slot.required and slot.status == SlotStatus.EMPTY:
                return slot
        # If all required are filled, check optional
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if not slot.required and slot.status == SlotStatus.EMPTY:
                return slot
        return None

    def calculate_completion(self) -> float:
        """Calculate profile completion percentage."""
        total_required = 0
        filled_required = 0
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if slot.required:
                total_required += 1
                if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]:
                    filled_required += 1
        if total_required == 0:
            return 100.0
        return round((filled_required / total_required) * 100, 1)

    def to_investment_profile(self) -> dict[str, Any]:
        """Convert filled slots to investment profile dict."""
        profile = {"user_id": self.user_id}
        for field_name in ProfileSlots.model_fields:
            slot = getattr(self.slots, field_name)
            if slot.value is not None:
                profile[field_name] = slot.value
        return profile


# Validation rules for slot values
SLOT_VALIDATION_RULES = {
    "risk_tolerance": {
        "type": "enum",
        "values": ["conservative", "moderate", "aggressive"],
    },
    "loss_comfort": {
        "type": "int_range",
        "min": 1,
        "max": 10,
    },
    "investment_horizon": {
        "type": "enum",
        "values": ["short", "medium", "long"],
    },
    "liquidity_needs": {
        "type": "enum",
        "values": ["low", "medium", "high"],
    },
    "income_stability": {
        "type": "enum",
        "values": ["stable", "variable", "uncertain"],
    },
    "has_emergency_fund": {
        "type": "bool",
    },
    "debt_level": {
        "type": "enum",
        "values": ["none", "low", "moderate", "high"],
    },
    "experience_level": {
        "type": "enum",
        "values": ["beginner", "intermediate", "advanced"],
    },
    "previous_investments": {
        "type": "list_string",
    },
    "primary_goal": {
        "type": "string",
    },
    "target_return": {
        "type": "float_range",
        "min": 0,
        "max": 100,
        "optional": True,
    },
    "esg_preference": {
        "type": "bool",
        "optional": True,
    },
    "monthly_income": {
        "type": "float_range",
        "min": 0,
        "max": 10_000_000,
    },
    "monthly_expenses": {
        "type": "float_range",
        "min": 0,
        "max": 10_000_000,
    },
}
