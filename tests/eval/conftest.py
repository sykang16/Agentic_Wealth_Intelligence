"""Shared fixtures and helpers for the evaluation test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.src.multi_agent.routing import IntentRouter
from backend.src.profiling.schemas import ConversationState, SlotStatus


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


def load_json_fixture(filename: str) -> Any:
    """Load a JSON fixture file from the fixtures directory."""
    path = FIXTURES_DIR / filename
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def router_testset() -> list[dict]:
    """100-case labeled intent classification dataset."""
    return load_json_fixture("router_testset.json")


@pytest.fixture(scope="session")
def scripted_extractions() -> list[dict]:
    """Scripted LLM-response → expected parse output cases."""
    return load_json_fixture("scripted_extractions.json")


@pytest.fixture(scope="module")
def fallback_router() -> IntentRouter:
    """IntentRouter whose LLM always fails, forcing keyword fallback."""
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = RuntimeError("LLM disabled for offline eval")
    return IntentRouter(mock_llm)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


REQUIRED_SLOTS = [
    "risk_tolerance",
    "loss_comfort",
    "investment_horizon",
    "liquidity_needs",
    "income_stability",
    "has_emergency_fund",
    "debt_level",
    "experience_level",
    "primary_goal",
    "monthly_income",
    "monthly_expenses",
]

SLOT_FILL_VALUES: dict[str, Any] = {
    "risk_tolerance": "moderate",
    "loss_comfort": 5,
    "investment_horizon": "long",
    "liquidity_needs": "low",
    "income_stability": "stable",
    "has_emergency_fund": True,
    "debt_level": "low",
    "experience_level": "intermediate",
    "primary_goal": "retirement",
    "monthly_income": 7000.0,
    "monthly_expenses": 4500.0,
}


def make_state(user_id: str = "eval_user") -> ConversationState:
    """Return a fresh empty conversation state."""
    return ConversationState(user_id=user_id)


def fill_slots(state: ConversationState, slot_names: list[str]) -> ConversationState:
    """Fill the specified slots with canonical test values and mark FILLED."""
    for name in slot_names:
        slot = getattr(state.slots, name)
        slot.value = SLOT_FILL_VALUES.get(name, "test_value")
        slot.status = SlotStatus.FILLED
    return state


def fill_all_required(state: ConversationState) -> ConversationState:
    """Fill every required slot so the profile reaches 100% completion."""
    return fill_slots(state, REQUIRED_SLOTS)


# ---------------------------------------------------------------------------
# Metrics helpers (no sklearn dependency)
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1 and overall accuracy.

    Returns a dict shaped like:
      {
        "accuracy": float,
        "per_class": {
          label: {"precision": float, "recall": float, "f1": float,
                  "tp": int, "fp": int, "fn": int, "support": int}
        }
      }
    """
    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def print_metrics_report(metrics: dict[str, Any], title: str = "Classification Report") -> None:
    """Print a formatted metrics report to stdout (visible in pytest -s)."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Overall accuracy: {metrics['accuracy']:.1%}\n")
    print(f"  {'Class':<20} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-' * 50}")
    for label, m in metrics["per_class"].items():
        print(
            f"  {label:<20} {m['precision']:>6.2f} {m['recall']:>6.2f}"
            f" {m['f1']:>6.2f} {m['support']:>8}"
        )
    print(f"{'=' * 60}")
