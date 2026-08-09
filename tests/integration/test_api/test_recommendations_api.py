"""Integration tests for recommendations API endpoints."""

import pytest


class TestRecommendationsAPI:
    """Tests for /api/v1/recommendations endpoints."""

    def test_generate_recommendations(self, api_client):
        """Test POST generate recommendations."""
        response = api_client.post(
            "/api/v1/recommendations/generate",
            json={"user_id": "user_001", "query": "How should I diversify?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "success" in data

    def test_generate_with_options(self, api_client):
        """Test generate recommendations with custom options."""
        response = api_client.post(
            "/api/v1/recommendations/generate",
            json={
                "user_id": "user_001",
                "query": "",
                "max_recommendations": 3,
                "include_live_data": False,
            },
        )
        assert response.status_code == 200

    def test_generate_missing_user_id(self, api_client):
        """Test generate without user_id is rejected."""
        response = api_client.post(
            "/api/v1/recommendations/generate",
            json={"query": "Help me invest"},
        )
        assert response.status_code == 422

    def test_generate_max_recommendations_limit(self, api_client):
        """Test generate with out-of-range max_recommendations."""
        response = api_client.post(
            "/api/v1/recommendations/generate",
            json={
                "user_id": "user_001",
                "max_recommendations": 20,  # > max of 10
            },
        )
        assert response.status_code == 422
