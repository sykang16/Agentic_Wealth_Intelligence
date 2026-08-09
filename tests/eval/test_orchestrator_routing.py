"""Evaluation: Orchestrator — intent routing, state threading, node logic.

Phase 2 scope — offline, all submodules mocked:
  - Tests route_by_intent() pure routing function for all 4 intents.
  - Tests node factory functions (portfolio, profiling, general) with mock agents.
  - Tests respond_node message history accumulation.
  - Tests WealthOrchestrator.get_response() state-to-response extraction.
  - Tests profiling_state threading across multiple turns.
  - Tests error isolation: module failure returns error response, no crash.

Run with:
    pytest tests/eval/test_orchestrator_routing.py -v -s
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.src.multi_agent.graph import route_by_intent
from backend.src.multi_agent.nodes import (
    create_general_node,
    create_portfolio_node,
    create_profiling_node,
    respond_node,
)
from backend.src.multi_agent.state import OrchestratorState, UserIntent
from backend.src.profiling.schemas import ConversationState, Message


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_state(
    user_id: str = "user_001",
    current_message: str = "Hello",
    intent: str = "",
    messages: list | None = None,
    profiling_state=None,
    response: str = "",
    module_source: str = "",
    error: str | None = None,
) -> OrchestratorState:
    """Build a minimal OrchestratorState dict for testing."""
    return OrchestratorState(
        user_id=user_id,
        messages=messages or [],
        current_message=current_message,
        intent=intent,
        response=response,
        module_source=module_source,
        profiling_state=profiling_state,
        error=error,
    )


def _make_asset_response(answer: str = "Your net worth is $700k.", success: bool = True):
    """Build a minimal AssetAgentResponse mock."""
    r = MagicMock()
    r.answer = answer
    r.success = success
    r.visualization = None
    r.error = None if success else "Mock error"
    return r


def _make_profiling_conv_state(user_id: str = "user_001") -> ConversationState:
    state = ConversationState(user_id=user_id)
    state.messages.append(Message(role="assistant", content="What's your risk tolerance?"))
    return state


# ── route_by_intent ───────────────────────────────────────────────────────────


class TestRouteByIntent:
    """route_by_intent maps each UserIntent value to the correct node name."""

    @pytest.mark.parametrize(
        "intent_value, expected_node",
        [
            (UserIntent.PORTFOLIO_QUERY.value, "portfolio"),
            (UserIntent.PROFILING.value, "profiling"),
            (UserIntent.RECOMMENDATION.value, "recommend"),
            (UserIntent.GENERAL.value, "general"),
        ],
    )
    def test_intent_routes_to_correct_node(self, intent_value, expected_node):
        state = make_state(intent=intent_value)
        result = route_by_intent(state)
        assert result == expected_node

    def test_unknown_intent_falls_back_to_general(self):
        state = make_state(intent="unknown_intent_xyz")
        result = route_by_intent(state)
        assert result == "general"

    def test_empty_intent_falls_back_to_general(self):
        state = make_state(intent="")
        result = route_by_intent(state)
        assert result == "general"

    def test_all_four_intents_have_distinct_routes(self):
        routes = set()
        for intent in UserIntent:
            state = make_state(intent=intent.value)
            routes.add(route_by_intent(state))
        assert len(routes) == 4


# ── respond_node ──────────────────────────────────────────────────────────────


class TestRespondNode:
    """respond_node appends (user, assistant) pair to messages and clears visualization."""

    def test_adds_user_message_to_history(self):
        state = make_state(current_message="What is my net worth?", response="$700k.")
        result = respond_node(state)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "What is my net worth?"

    def test_adds_assistant_message_to_history(self):
        state = make_state(current_message="Hello", response="Hi there!")
        result = respond_node(state)
        asst_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert asst_msgs[0]["content"] == "Hi there!"

    def test_appends_to_existing_history(self):
        existing = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        state = make_state(
            messages=existing,
            current_message="What's my balance?",
            response="Your balance is $50k.",
        )
        result = respond_node(state)
        assert len(result["messages"]) == 4  # 2 existing + 2 new

    def test_clears_visualization_after_respond(self):
        state = make_state(response="Here's your chart.")
        state["visualization"] = object()  # mock figure
        result = respond_node(state)
        assert result["visualization"] is None

    def test_visualization_included_in_assistant_message_when_present(self):
        state = make_state(current_message="Show chart", response="Here is your chart.")
        state["visualization"] = {"type": "pie"}  # mock viz object
        result = respond_node(state)
        asst_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert "visualization" in asst_msgs[0]

    def test_multiple_turns_accumulate_correctly(self):
        """Simulate 3 turns of respond_node; history grows by 2 each turn."""
        messages = []
        for i in range(3):
            state = make_state(
                messages=messages,
                current_message=f"Q{i}",
                response=f"A{i}",
            )
            result = respond_node(state)
            messages = result["messages"]
        assert len(messages) == 6  # 3 * (user + assistant)


# ── create_portfolio_node ────────────────────────────────────────────────────


class TestPortfolioNode:
    """create_portfolio_node calls asset_agent.process() and returns response dict."""

    @pytest.fixture
    def mock_asset_agent(self):
        agent = MagicMock()
        agent.process.return_value = _make_asset_response("Your portfolio is worth $700k.")
        return agent

    @pytest.fixture
    def portfolio_node(self, mock_asset_agent):
        return create_portfolio_node(mock_asset_agent)

    def test_calls_asset_agent_process(self, portfolio_node, mock_asset_agent):
        state = make_state(user_id="user_001", current_message="What's my net worth?")
        portfolio_node(state)
        mock_asset_agent.process.assert_called_once_with("user_001", "What's my net worth?")

    def test_returns_response_string(self, portfolio_node):
        state = make_state(user_id="user_001", current_message="Show portfolio")
        result = portfolio_node(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_module_source_is_portfolio(self, portfolio_node):
        state = make_state(user_id="user_001", current_message="Net worth?")
        result = portfolio_node(state)
        assert result["module_source"] == "Portfolio"

    def test_agent_failure_sets_error(self, mock_asset_agent):
        mock_asset_agent.process.return_value = _make_asset_response(
            answer="Error occurred", success=False
        )
        node = create_portfolio_node(mock_asset_agent)
        state = make_state(user_id="user_001", current_message="Net worth?")
        result = node(state)
        assert result.get("error") is not None

    def test_agent_exception_caught_gracefully(self):
        agent = MagicMock()
        agent.process.side_effect = RuntimeError("Database down")
        node = create_portfolio_node(agent)
        state = make_state(user_id="user_001", current_message="Net worth?")
        result = node(state)
        assert "response" in result
        assert result["module_source"] == "Portfolio"
        assert result.get("error") is not None


# ── create_profiling_node ─────────────────────────────────────────────────────


class TestProfilingNode:
    """create_profiling_node starts a new conversation or continues an existing one."""

    @pytest.fixture
    def conv_state(self):
        """A profiling ConversationState with one greeting message."""
        return _make_profiling_conv_state()

    @pytest.fixture
    def mock_profiling_agent(self, conv_state):
        agent = MagicMock()
        agent.start_conversation.return_value = conv_state
        agent.process_response.return_value = conv_state
        agent.get_last_assistant_message.return_value = "What's your risk tolerance?"
        agent.get_profile_summary.return_value = {
            "completion_percentage": 0.0,
            "filled_slots": {},
            "missing_required": [],
            "is_complete": False,
        }
        return agent

    @pytest.fixture
    def mock_aggregator(self):
        return MagicMock()

    @pytest.fixture
    def profiling_node(self, mock_profiling_agent, mock_aggregator):
        return create_profiling_node(mock_profiling_agent, mock_aggregator)

    def test_starts_new_conversation_when_no_state(
        self, profiling_node, mock_profiling_agent
    ):
        state = make_state(user_id="user_001", current_message="I want to build a profile")
        profiling_node(state)
        mock_profiling_agent.start_conversation.assert_called_once_with("user_001")

    def test_continues_conversation_when_state_exists(
        self, profiling_node, mock_profiling_agent, conv_state
    ):
        state = make_state(
            user_id="user_001",
            current_message="Moderate risk",
            profiling_state=conv_state,
        )
        profiling_node(state)
        mock_profiling_agent.process_response.assert_called_once()

    def test_module_source_is_profiling(self, profiling_node):
        state = make_state(user_id="user_001", current_message="Build my profile")
        result = profiling_node(state)
        assert result["module_source"] == "Profiling"

    def test_profiling_state_is_threaded_in_response(
        self, profiling_node, conv_state
    ):
        state = make_state(user_id="user_001", current_message="Build my profile")
        result = profiling_node(state)
        assert result["profiling_state"] is not None

    def test_subsequent_call_passes_existing_state(
        self, profiling_node, mock_profiling_agent, conv_state
    ):
        """The profiling_state from the first call must be passed into the second call."""
        state1 = make_state(user_id="user_001", current_message="Start profiling")
        result1 = profiling_node(state1)

        state2 = make_state(
            user_id="user_001",
            current_message="Moderate risk",
            profiling_state=result1["profiling_state"],
        )
        profiling_node(state2)
        # process_response should now be called with the persisted conv_state
        mock_profiling_agent.process_response.assert_called_once()

    def test_profiling_agent_exception_caught_gracefully(self):
        agent = MagicMock()
        agent.start_conversation.side_effect = RuntimeError("LLM unavailable")
        node = create_profiling_node(agent, MagicMock())
        state = make_state(user_id="user_001", current_message="Build profile")
        result = node(state)
        assert "response" in result
        assert result["module_source"] == "Profiling"
        assert result.get("error") is not None


# ── create_general_node ───────────────────────────────────────────────────────


class TestGeneralNode:
    """create_general_node calls the LLM and returns its content."""

    @pytest.fixture
    def mock_llm_response(self):
        resp = MagicMock()
        resp.content = "I'm your wealth management assistant. How can I help?"
        return resp

    @pytest.fixture
    def mock_llm(self, mock_llm_response):
        llm = MagicMock()
        llm.chat.return_value = mock_llm_response
        llm.chat_with_history.return_value = mock_llm_response
        return llm

    @pytest.fixture
    def general_node(self, mock_llm):
        return create_general_node(mock_llm)

    def test_returns_response_string(self, general_node):
        state = make_state(current_message="Hello!")
        result = general_node(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_module_source_is_general(self, general_node):
        state = make_state(current_message="What can you do?")
        result = general_node(state)
        assert result["module_source"] == "General"

    def test_no_history_uses_chat(self, general_node, mock_llm):
        state = make_state(current_message="Hello!", messages=[])
        general_node(state)
        mock_llm.chat.assert_called_once()
        mock_llm.chat_with_history.assert_not_called()

    def test_with_history_uses_chat_with_history(self, general_node, mock_llm):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        state = make_state(current_message="Follow-up", messages=history)
        general_node(state)
        mock_llm.chat_with_history.assert_called_once()

    def test_llm_failure_returns_fallback_response(self, mock_llm):
        mock_llm.chat.side_effect = RuntimeError("LLM down")
        mock_llm.chat_with_history.side_effect = RuntimeError("LLM down")
        node = create_general_node(mock_llm)
        state = make_state(current_message="Hello!")
        result = node(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0


# ── WealthOrchestrator.get_response ──────────────────────────────────────────


class TestGetResponse:
    """WealthOrchestrator.get_response() converts OrchestratorState to OrchestratorResponse."""

    def _make_orchestrator(self):
        """Build WealthOrchestrator with all submodules mocked to avoid real init."""
        from backend.src.agents.orchestrator import WealthOrchestrator
        from backend.src.multi_agent.state import OrchestratorResponse

        orch = object.__new__(WealthOrchestrator)  # skip __init__
        return orch

    def test_returns_orchestrator_response(self):
        from backend.src.agents.orchestrator import WealthOrchestrator

        orch = self._make_orchestrator()
        state = make_state(
            response="Your balance is $50k.",
            intent="portfolio_query",
            module_source="Portfolio",
        )
        result = orch.get_response(state)
        assert result.response == "Your balance is $50k."
        assert result.intent == "portfolio_query"
        assert result.module_source == "Portfolio"

    def test_success_true_when_no_error(self):
        from backend.src.agents.orchestrator import WealthOrchestrator

        orch = self._make_orchestrator()
        state = make_state(response="OK", error=None)
        result = orch.get_response(state)
        assert result.success is True
        assert result.error is None

    def test_success_false_when_error_present(self):
        from backend.src.agents.orchestrator import WealthOrchestrator

        orch = self._make_orchestrator()
        state = make_state(response="Something went wrong", error="LLM timeout")
        result = orch.get_response(state)
        assert result.success is False
        assert result.error == "LLM timeout"

    def test_messages_serialized_to_role_content_dicts(self):
        from backend.src.agents.orchestrator import WealthOrchestrator

        orch = self._make_orchestrator()
        history = [
            {"role": "user", "content": "Hi", "visualization": object()},
            {"role": "assistant", "content": "Hello!"},
        ]
        state = make_state(messages=history, response="Done")
        result = orch.get_response(state)
        for msg in result.messages:
            assert "role" in msg
            assert "content" in msg
            # visualization must be stripped for the API response
            assert "visualization" not in msg

    def test_empty_state_returns_defaults(self):
        from backend.src.agents.orchestrator import WealthOrchestrator

        orch = self._make_orchestrator()
        state = OrchestratorState(
            user_id="u1",
            messages=[],
            current_message="",
            intent="",
            response="",
            module_source="",
            profiling_state=None,
            error=None,
        )
        result = orch.get_response(state)
        assert result.response == ""
        assert result.messages == []
        assert result.success is True


# ── Message history threading across node calls ────────────────────────────────


class TestMessageHistoryThreading:
    """Verify that message history grows correctly across simulated turns."""

    def test_history_grows_by_two_per_turn(self):
        """Each respond_node call adds one user + one assistant message."""
        messages = []
        for i in range(4):
            state = make_state(
                messages=messages,
                current_message=f"Question {i}",
                response=f"Answer {i}",
            )
            result = respond_node(state)
            messages = result["messages"]

        assert len(messages) == 8  # 4 turns * 2 messages

    def test_message_order_is_preserved(self):
        """User and assistant messages must alternate in chronological order."""
        messages = []
        for i in range(3):
            state = make_state(
                messages=messages,
                current_message=f"Q{i}",
                response=f"A{i}",
            )
            messages = respond_node(state)["messages"]

        for i in range(0, len(messages), 2):
            assert messages[i]["role"] == "user", f"Expected user at index {i}"
            assert messages[i + 1]["role"] == "assistant", (
                f"Expected assistant at index {i + 1}"
            )

    def test_profiling_state_persists_across_turns(self):
        """profiling_state from node response must be threaded into next turn's state."""
        conv_state_1 = _make_profiling_conv_state()
        conv_state_2 = _make_profiling_conv_state()
        conv_state_2.messages.append(Message(role="user", content="Moderate risk"))

        agent = MagicMock()
        agent.start_conversation.return_value = conv_state_1
        agent.process_response.return_value = conv_state_2
        agent.get_last_assistant_message.return_value = "What's your horizon?"
        agent.get_profile_summary.return_value = {
            "completion_percentage": 0.0,
            "is_complete": False,
            "filled_slots": {},
            "missing_required": [],
        }
        node = create_profiling_node(agent, MagicMock())

        # Turn 1: start conversation
        state1 = make_state(
            user_id="user_001", current_message="Build my profile", profiling_state=None
        )
        result1 = node(state1)
        assert result1["profiling_state"] is conv_state_1

        # Turn 2: continue with threaded state
        state2 = make_state(
            user_id="user_001",
            current_message="Moderate risk",
            profiling_state=result1["profiling_state"],
        )
        result2 = node(state2)
        # process_response should have been called with conv_state_1
        agent.process_response.assert_called_once_with(conv_state_1, "Moderate risk")
        assert result2["profiling_state"] is conv_state_2
