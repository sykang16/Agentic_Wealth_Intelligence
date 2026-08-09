"""Unit tests for SEC EDGAR collector."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.src.recommendation.collectors.sec_edgar import SECEdgarCollector
from backend.src.recommendation.collectors.base import CollectorStatus


class TestSECEdgarCollector:
    """Tests for the SECEdgarCollector."""

    def test_is_configured_with_agent(self):
        """Test is_configured returns True when user agent is set."""
        collector = SECEdgarCollector(user_agent="Test User test@example.com")
        assert collector.is_configured() is True

    def test_is_configured_without_agent(self):
        """Test is_configured returns False when no user agent."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEC_USER_AGENT", None)
            collector = SECEdgarCollector(user_agent=None)
            assert collector.is_configured() is False

    def test_collect_not_configured(self):
        """Test collect returns error when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEC_USER_AGENT", None)
            collector = SECEdgarCollector(user_agent=None)
            result = collector.collect()
            assert result.status == CollectorStatus.NO_API_KEY

    def test_default_ciks(self):
        """Test default CIK mappings exist."""
        assert "AAPL" in SECEdgarCollector.DEFAULT_CIKS
        assert "MSFT" in SECEdgarCollector.DEFAULT_CIKS

    def test_get_status(self):
        """Test get_status returns proper info."""
        collector = SECEdgarCollector(user_agent="Test test@test.com")
        status = collector.get_status()
        assert status["name"] == "sec_edgar"
        assert status["configured"] is True

    def test_get_status_not_configured(self):
        """Test get_status for unconfigured collector."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEC_USER_AGENT", None)
            collector = SECEdgarCollector(user_agent=None)
            status = collector.get_status()
            assert status["configured"] is False
