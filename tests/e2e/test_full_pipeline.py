"""End-to-end tests for the full wealth intelligence pipeline."""

import pytest

from backend.src.multi_agent.state import UserIntent


class TestFullPipeline:
    """E2E tests exercising the complete A -> B -> C journey."""

    def test_portfolio_query_flow(self, e2e_orchestrator):
        """Test: user asks portfolio question, gets answer."""
        state = e2e_orchestrator.create_initial_state("user_001")
        result = e2e_orchestrator.process_message("What's my net worth?", state)

        assert result["module_source"] == "Portfolio"
        assert result["response"]
        assert len(result["messages"]) == 2

    def test_profiling_flow(self, e2e_orchestrator):
        """Test: user starts profiling, provides answers."""
        state = e2e_orchestrator.create_initial_state("user_001")

        # Start profiling
        state = e2e_orchestrator.process_message(
            "Help me build my investment profile", state
        )
        assert state["module_source"] == "Profiling"
        assert state["profiling_state"] is not None

    def test_recommendation_flow(self, e2e_orchestrator):
        """Test: user asks for recommendations."""
        state = e2e_orchestrator.create_initial_state("user_001")
        result = e2e_orchestrator.process_message(
            "What should I invest in?", state
        )
        assert result["module_source"] == "Recommendation"
        assert result["response"]

    def test_multi_turn_module_handoff(self, e2e_orchestrator):
        """Test: user moves between modules across turns."""
        state = e2e_orchestrator.create_initial_state("user_001")

        # Turn 1: Portfolio query
        state = e2e_orchestrator.process_message("What's my net worth?", state)
        assert state["module_source"] == "Portfolio"
        assert len(state["messages"]) == 2

        # Turn 2: General greeting
        state = e2e_orchestrator.process_message("Thanks!", state)
        assert state["module_source"] == "General"
        assert len(state["messages"]) == 4

        # Turn 3: Recommendation
        state = e2e_orchestrator.process_message("Should I diversify?", state)
        assert state["module_source"] == "Recommendation"
        assert len(state["messages"]) == 6

    def test_context_preserved_across_turns(self, e2e_orchestrator):
        """Test that message history is preserved across module handoffs."""
        state = e2e_orchestrator.create_initial_state("user_001")

        state = e2e_orchestrator.process_message("Hello!", state)
        state = e2e_orchestrator.process_message("Show my portfolio", state)
        state = e2e_orchestrator.process_message("Give me advice", state)

        # All 6 messages (3 user + 3 assistant) should be in history
        assert len(state["messages"]) == 6
        # Verify alternating user/assistant pattern
        for i in range(0, 6, 2):
            assert state["messages"][i]["role"] == "user"
            assert state["messages"][i + 1]["role"] == "assistant"


class TestAPIFullPipeline:
    """E2E tests through the FastAPI layer."""

    @pytest.fixture
    def api_client(self, sample_aggregator_with_data, e2e_mock_llm, e2e_mock_rag):
        """Create API client for E2E tests."""
        from fastapi.testclient import TestClient
        from backend.src.api.app import create_app
        from backend.src.api import dependencies as deps
        from backend.src.agents.orchestrator import WealthOrchestrator

        app = create_app()
        orchestrator = WealthOrchestrator(
            sample_aggregator_with_data, e2e_mock_llm, e2e_mock_rag
        )

        app.dependency_overrides[deps.get_aggregator] = lambda: sample_aggregator_with_data
        app.dependency_overrides[deps.get_llm_client] = lambda: e2e_mock_llm
        app.dependency_overrides[deps.get_rag] = lambda: e2e_mock_rag
        app.dependency_overrides[deps.get_orchestrator] = lambda: orchestrator

        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_full_api_journey(self, api_client):
        """Test complete journey through API endpoints."""
        # 1. Get portfolio summary
        r = api_client.get("/api/v1/portfolio/user_001")
        assert r.status_code == 200
        assert r.json()["user_id"] == "user_001"

        # 2. Chat - portfolio query
        r = api_client.post(
            "/api/v1/chat",
            json={"message": "What's my net worth?", "user_id": "user_001"},
        )
        assert r.status_code == 200
        assert r.json()["module_source"] == "Portfolio"

        # 3. Start profiling
        r = api_client.post("/api/v1/profiling/user_001/start")
        assert r.status_code == 200
        assert r.json()["is_complete"] is False

        # 4. Generate recommendations
        r = api_client.post(
            "/api/v1/recommendations/generate",
            json={"user_id": "user_001", "max_recommendations": 3},
        )
        assert r.status_code == 200
