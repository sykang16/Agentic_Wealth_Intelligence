"""Integration tests for portfolio API endpoints."""

import pytest


class TestPortfolioAPI:
    """Tests for /api/v1/portfolio endpoints."""

    def test_get_portfolio_summary(self, api_client):
        """Test GET portfolio summary."""
        response = api_client.get("/api/v1/portfolio/user_001")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_001"
        assert "metrics" in data
        assert "allocation" in data
        assert "top_holdings" in data

    def test_get_portfolio_invalid_user(self, api_client):
        """Test GET portfolio for non-existent user returns 404."""
        # Need to configure mock to return None for unknown user
        from unittest.mock import MagicMock
        from backend.src.api import dependencies as deps

        original = api_client.app.dependency_overrides.get(deps.get_aggregator)
        mock_agg = original() if original else MagicMock()

        def get_portfolio_side_effect(uid):
            if uid == "nonexistent":
                return None
            return mock_agg.get_portfolio.return_value

        mock_agg.get_portfolio = MagicMock(side_effect=get_portfolio_side_effect)
        api_client.app.dependency_overrides[deps.get_aggregator] = lambda: mock_agg

        response = api_client.get("/api/v1/portfolio/nonexistent")
        assert response.status_code == 404

    def test_query_portfolio(self, api_client):
        """Test POST portfolio query."""
        response = api_client.post(
            "/api/v1/portfolio/user_001/query",
            json={"query": "What's my net worth?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "success" in data

    def test_query_portfolio_empty_query(self, api_client):
        """Test POST portfolio query with empty query."""
        response = api_client.post(
            "/api/v1/portfolio/user_001/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_portfolio_summary_has_holdings(self, api_client):
        """Test that portfolio summary includes holdings info."""
        response = api_client.get("/api/v1/portfolio/user_001")
        data = response.json()
        assert len(data["top_holdings"]) > 0
        assert "symbol" in data["top_holdings"][0]
