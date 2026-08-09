"""Integration tests for the chat API endpoint."""

import pytest


class TestChatAPI:
    """Tests for POST /api/v1/chat."""

    def test_chat_general(self, api_client):
        """Test chat with a general greeting."""
        response = api_client.post(
            "/api/v1/chat",
            json={"message": "Hello!", "user_id": "user_001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["response"]

    def test_chat_portfolio_query(self, api_client):
        """Test chat with a portfolio query."""
        response = api_client.post(
            "/api/v1/chat",
            json={"message": "What's my net worth?", "user_id": "user_001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["module_source"] == "Portfolio"

    def test_chat_profiling(self, api_client):
        """Test chat with a profiling request."""
        response = api_client.post(
            "/api/v1/chat",
            json={"message": "Help me build my profile", "user_id": "user_001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["module_source"] == "Profiling"

    def test_chat_empty_message_rejected(self, api_client):
        """Test that empty message is rejected."""
        response = api_client.post(
            "/api/v1/chat",
            json={"message": "", "user_id": "user_001"},
        )
        assert response.status_code == 422

    def test_chat_missing_user_id(self, api_client):
        """Test that missing user_id is rejected."""
        response = api_client.post(
            "/api/v1/chat",
            json={"message": "Hello!"},
        )
        assert response.status_code == 422
