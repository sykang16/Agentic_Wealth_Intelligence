"""Unit tests for News API collector."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.src.recommendation.collectors.news_api import NewsAPICollector
from backend.src.recommendation.collectors.base import CollectorStatus


class TestNewsAPICollector:
    """Tests for the NewsAPICollector."""

    def test_is_configured_with_key(self):
        """Test is_configured returns True when API key is set."""
        collector = NewsAPICollector(api_key="abcdef1234567890")
        assert collector.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured returns False when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEWS_API_KEY", None)
            collector = NewsAPICollector(api_key=None)
            assert collector.is_configured() is False

    def test_is_configured_short_key(self):
        """Test is_configured returns False for short placeholder key."""
        collector = NewsAPICollector(api_key="short")
        assert collector.is_configured() is False

    def test_is_configured_placeholder_key(self):
        """Test is_configured returns False for placeholder key."""
        collector = NewsAPICollector(api_key="your-key-here")
        assert collector.is_configured() is False

    def test_collect_not_configured(self):
        """Test collect returns error when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEWS_API_KEY", None)
            collector = NewsAPICollector(api_key=None)
            result = collector.collect()
            assert result.status == CollectorStatus.NO_API_KEY

    def test_get_status(self):
        """Test get_status returns proper info."""
        collector = NewsAPICollector(api_key="abcdef1234567890")
        status = collector.get_status()
        assert status["name"] == "news_api"
        assert status["configured"] is True
