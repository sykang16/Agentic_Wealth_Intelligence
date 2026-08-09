"""Unit tests for Alpha Vantage collector."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.src.recommendation.collectors.alpha_vantage import AlphaVantageCollector
from backend.src.recommendation.collectors.base import CollectorStatus


class TestAlphaVantageCollector:
    """Tests for the AlphaVantageCollector."""

    def test_is_configured_with_key(self):
        """Test is_configured returns True when API key is set."""
        collector = AlphaVantageCollector(api_key="test-api-key")
        assert collector.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured returns False when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
            collector = AlphaVantageCollector(api_key=None)
            assert collector.is_configured() is False

    def test_is_configured_placeholder_key(self):
        """Test is_configured returns False for placeholder key."""
        collector = AlphaVantageCollector(api_key="your-alpha-vantage-key-here")
        assert collector.is_configured() is False

    def test_collect_not_configured(self):
        """Test collect returns error when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
            collector = AlphaVantageCollector(api_key=None)
            result = collector.collect()
            assert result.status == CollectorStatus.NO_API_KEY

    def test_collect_returns_collector_result(self):
        """Test collect returns a CollectorResult regardless of outcome."""
        collector = AlphaVantageCollector(api_key="test-key")
        # Mock the HTTP client to raise (simulating network error)
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Network error")
        collector._client = mock_client
        result = collector.collect(collect_etf_data=False)

        # Should still return a CollectorResult, not raise
        assert result.collector_name == "alpha_vantage"
        assert isinstance(result.status, CollectorStatus)

    def test_get_status(self):
        """Test get_status returns proper info."""
        collector = AlphaVantageCollector(api_key="test-key")
        status = collector.get_status()
        assert status["name"] == "alpha_vantage"
        assert status["configured"] is True
