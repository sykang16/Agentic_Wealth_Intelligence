"""Integration tests for profiling API endpoints."""

import pytest


class TestProfilingAPI:
    """Tests for /api/v1/profiling endpoints."""

    def test_start_profiling(self, api_client):
        """Test POST start profiling session."""
        response = api_client.post(
            "/api/v1/profiling/user_001/start",
            json={"user_name": "Test User"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "assistant_message" in data
        assert data["is_complete"] is False
        assert data["completion_percentage"] >= 0

    def test_start_profiling_default_name(self, api_client):
        """Test POST start profiling with default name."""
        response = api_client.post("/api/v1/profiling/user_002/start")
        assert response.status_code == 200

    def test_respond_without_start(self, api_client):
        """Test POST respond without starting session returns 404."""
        response = api_client.post(
            "/api/v1/profiling/nonexistent_user/respond",
            json={"message": "I have moderate risk tolerance"},
        )
        assert response.status_code == 404

    def test_start_then_respond(self, api_client):
        """Test full profiling flow: start then respond."""
        # Start
        start_response = api_client.post(
            "/api/v1/profiling/user_flow/start",
            json={"user_name": "Flow User"},
        )
        assert start_response.status_code == 200

        # Respond
        respond_response = api_client.post(
            "/api/v1/profiling/user_flow/respond",
            json={"message": "I'm a moderate risk investor"},
        )
        assert respond_response.status_code == 200
        data = respond_response.json()
        assert "assistant_message" in data

    def test_respond_empty_message(self, api_client):
        """Test POST respond with empty message is rejected."""
        response = api_client.post(
            "/api/v1/profiling/user_001/respond",
            json={"message": ""},
        )
        assert response.status_code == 422
