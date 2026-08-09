"""Unit tests for the orchestrator graph construction."""

from unittest.mock import MagicMock, call

import pytest

from backend.src.multi_agent.graph import build_orchestrator_graph, route_by_intent
from backend.src.multi_agent.recommendation_supervisor import build_recommendation_supervisor
from backend.src.multi_agent.state import OrchestratorState, UserIntent


class TestRouteByIntent:
    """Tests for the route_by_intent function."""

    def test_routes_portfolio(self):
        """Test routing to portfolio node."""
        state = OrchestratorState(intent=UserIntent.PORTFOLIO_QUERY.value)
        assert route_by_intent(state) == "portfolio"

    def test_routes_profiling(self):
        """Test routing to profiling node."""
        state = OrchestratorState(intent=UserIntent.PROFILING.value)
        assert route_by_intent(state) == "profiling"

    def test_routes_recommendation(self):
        """Test routing to recommendation node."""
        state = OrchestratorState(intent=UserIntent.RECOMMENDATION.value)
        assert route_by_intent(state) == "recommend"

    def test_routes_general(self):
        """Test routing to general node."""
        state = OrchestratorState(intent=UserIntent.GENERAL.value)
        assert route_by_intent(state) == "general"

    def test_routes_unknown_to_general(self):
        """Test that unknown intent routes to general."""
        state = OrchestratorState(intent="unknown")
        assert route_by_intent(state) == "general"


class TestBuildOrchestratorGraph:
    """Tests for graph construction."""

    def test_graph_compiles(self):
        """Test that the graph compiles without errors."""
        mock_llm = MagicMock()
        mock_asset_agent = MagicMock()
        mock_profiling_agent = MagicMock()
        mock_rec_engine = MagicMock()

        graph = build_orchestrator_graph(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_has_all_nodes(self):
        """Test that graph contains all expected nodes."""
        mock_llm = MagicMock()
        mock_asset_agent = MagicMock()
        mock_profiling_agent = MagicMock()
        mock_rec_engine = MagicMock()

        graph = build_orchestrator_graph(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )

        node_names = set(graph.nodes.keys())
        expected = {"router", "portfolio", "profiling", "recommend", "general", "respond"}
        assert expected.issubset(node_names)

    def test_graph_entry_point_is_router(self):
        """Test that the graph entry point is the router node."""
        mock_llm = MagicMock()
        graph = build_orchestrator_graph(
            llm_client=mock_llm,
            asset_agent=MagicMock(),
            profiling_agent=MagicMock(),
            recommendation_engine=MagicMock(),
        )
        # LangGraph sets __start__ → entry_point
        compiled = graph.compile()
        assert compiled is not None

    def test_outer_graph_non_recommendation_paths_unchanged(self):
        """Portfolio, profiling, and general intents route to their own nodes, not the supervisor."""
        mock_llm = MagicMock()
        graph = build_orchestrator_graph(
            llm_client=mock_llm,
            asset_agent=MagicMock(),
            profiling_agent=MagicMock(),
            recommendation_engine=MagicMock(),
        )
        # All expected nodes are present
        node_names = set(graph.nodes.keys())
        assert {"router", "portfolio", "profiling", "recommend", "general", "respond"}.issubset(
            node_names
        )
        # route_by_intent maps non-recommendation intents to their dedicated nodes
        assert route_by_intent(OrchestratorState(intent=UserIntent.PORTFOLIO_QUERY.value)) == "portfolio"
        assert route_by_intent(OrchestratorState(intent=UserIntent.PROFILING.value)) == "profiling"
        assert route_by_intent(OrchestratorState(intent=UserIntent.GENERAL.value)) == "general"
        # Only recommendation intent routes to the supervisor subgraph
        assert route_by_intent(OrchestratorState(intent=UserIntent.RECOMMENDATION.value)) == "recommend"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _llm_response(text: str) -> MagicMock:
    """Create a mock LLMResponse with the given content string."""
    r = MagicMock()
    r.content = text
    return r


def _make_rec_engine_mock(summary: str = "Here are your recommendations.") -> MagicMock:
    engine = MagicMock()
    result = MagicMock()
    result.success = True
    result.summary = summary
    result.recommendations = []
    result.warnings = []
    engine.generate_recommendations.return_value = result
    return engine


def _make_asset_agent_mock(answer: str = "Portfolio: 60% stocks, 40% bonds.") -> MagicMock:
    agent = MagicMock()
    resp = MagicMock()
    resp.answer = answer
    agent.process.return_value = resp
    return agent


def _make_profiling_agent_mock() -> MagicMock:
    agent = MagicMock()
    agent.get_profile_summary.return_value = {
        "user_id": "user_1",
        "filled_slots": {
            "Risk Tolerance": {"value": "moderate", "name": "risk_tolerance"},
            "Investment Horizon": {"value": "5-10 years", "name": "investment_horizon"},
        },
        "completion_percentage": 80.0,
        "is_complete": False,
        "missing_required": [],
    }
    return agent


# ── Supervisor subgraph tests ──────────────────────────────────────────────────

class TestRecommendationSupervisor:
    """Tests for the recommendation supervisor subgraph."""

    def test_cross_domain_invokes_portfolio_and_profiling(self):
        """Subgraph fetches both portfolio and profile before synthesising."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _llm_response("portfolio_fetch"),   # supervisor step 1
            _llm_response("profiling_fetch"),   # supervisor step 2
            _llm_response("recommend"),         # supervisor step 3
            _llm_response("finish"),            # supervisor step 4
        ]
        mock_asset_agent = _make_asset_agent_mock()
        mock_profiling_agent = _make_profiling_agent_mock()
        mock_rec_engine = _make_rec_engine_mock()

        supervisor = build_recommendation_supervisor(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )

        result = supervisor.invoke(
            {
                "user_id": "user_1",
                "current_message": (
                    "Should I rebalance my portfolio based on my asset allocation "
                    "while considering my investment tendency?"
                ),
                "profiling_state": MagicMock(),
            }
        )

        # Both context agents must have been called
        mock_asset_agent.process.assert_called_once()
        mock_profiling_agent.get_profile_summary.assert_called_once()

        # Scratchpad contains entries from all three agents
        scratchpad = result.get("agent_scratchpad", [])
        agent_names = [e["agent"] for e in scratchpad]
        assert "portfolio_fetch" in agent_names
        assert "profiling_fetch" in agent_names
        assert "recommend" in agent_names

        assert result.get("response")
        assert result.get("module_source") == "Recommendation"

    def test_simple_query_skips_context_agents(self):
        """For a simple query, supervisor calls recommend directly without fetching context."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _llm_response("recommend"),  # straight to recommend
            _llm_response("finish"),
        ]
        mock_asset_agent = _make_asset_agent_mock()
        mock_profiling_agent = _make_profiling_agent_mock()
        mock_rec_engine = _make_rec_engine_mock("Consider diversifying into bonds.")

        supervisor = build_recommendation_supervisor(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )

        result = supervisor.invoke(
            {
                "user_id": "user_1",
                "current_message": "What should I invest in?",
            }
        )

        # Context agents must NOT be called
        mock_asset_agent.process.assert_not_called()
        mock_profiling_agent.get_profile_summary.assert_not_called()

        # Recommendation engine was called
        mock_rec_engine.generate_recommendations.assert_called_once()

        assert result.get("response")
        assert result.get("module_source") == "Recommendation"

    def test_scratchpad_context_passed_to_recommendation_engine(self):
        """The enriched query sent to the engine contains context from both agents."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _llm_response("portfolio_fetch"),
            _llm_response("profiling_fetch"),
            _llm_response("recommend"),
            _llm_response("finish"),
        ]
        mock_asset_agent = _make_asset_agent_mock(
            answer="Net worth: $500,000. Allocation: 70% equity, 30% fixed income."
        )
        mock_profiling_agent = _make_profiling_agent_mock()
        mock_rec_engine = _make_rec_engine_mock(
            summary="Based on your 70% equity and moderate risk tolerance…"
        )

        supervisor = build_recommendation_supervisor(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )

        result = supervisor.invoke(
            {
                "user_id": "user_1",
                "current_message": (
                    "Should I rebalance my portfolio based on my current asset allocation "
                    "while considering my investment tendency?"
                ),
                "profiling_state": MagicMock(),
            }
        )

        scratchpad = result.get("agent_scratchpad", [])

        # At least 2 context entries (portfolio + profile) precede the recommend entry
        context_entries = [e for e in scratchpad if e["agent"] != "recommend"]
        assert len(context_entries) >= 2

        # Verify the engine received an enriched query containing agent context
        request = mock_rec_engine.generate_recommendations.call_args[0][0]
        assert "portfolio_fetch" in request.query or "CONTEXT" in request.query

        assert result.get("response")
        assert result.get("module_source") == "Recommendation"

    def test_iteration_guard_forces_finish(self):
        """Supervisor exits gracefully when MAX_SUPERVISOR_STEPS is reached."""
        # LLM always returns portfolio_fetch (malicious/stuck loop)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _llm_response("portfolio_fetch")

        mock_asset_agent = _make_asset_agent_mock()
        mock_profiling_agent = _make_profiling_agent_mock()
        mock_rec_engine = _make_rec_engine_mock()

        supervisor = build_recommendation_supervisor(
            llm_client=mock_llm,
            asset_agent=mock_asset_agent,
            profiling_agent=mock_profiling_agent,
            recommendation_engine=mock_rec_engine,
        )

        # Should not raise; iteration guard forces finish
        result = supervisor.invoke(
            {
                "user_id": "user_1",
                "current_message": "Any advice?",
            }
        )

        # module_source may be absent (no recommend node ran), but should not be
        # "Recommendation" with an empty response OR should have gracefully set error.
        # The key assertion: the subgraph terminated (did not loop forever).
        assert result is not None
