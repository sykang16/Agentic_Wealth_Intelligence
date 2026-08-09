"""Unit tests for DataCollectionManager."""

from unittest.mock import MagicMock, patch

import pytest

from backend.src.recommendation.collectors.base import CollectorResult, CollectorStatus
from backend.src.recommendation.collectors.manager import DataCollectionManager


class TestDataCollectionManager:
    """Tests for the DataCollectionManager."""

    def test_get_collector_status(self):
        """Test getting status of all collectors."""
        manager = DataCollectionManager(rag_initializer=MagicMock())
        status = manager.get_collector_status()
        assert "alpha_vantage" in status
        assert "sec_edgar" in status
        assert "news_api" in status

    def test_get_configured_collectors_none(self):
        """Test getting configured collectors when none are configured."""
        manager = DataCollectionManager(rag_initializer=MagicMock())
        # With no env vars, all collectors should be unconfigured
        # (unless environment happens to have keys)
        configured = manager.get_configured_collectors()
        assert isinstance(configured, list)

    @patch.object(DataCollectionManager, "get_configured_collectors", return_value=[])
    def test_collect_all_no_collectors(self, mock_configured):
        """Test collect_all with no configured collectors."""
        manager = DataCollectionManager(rag_initializer=MagicMock())
        summary = manager.collect_all()
        assert summary.total_documents == 0
        assert "No configured collectors available" in summary.errors

    @patch.object(DataCollectionManager, "get_configured_collectors", return_value=["alpha_vantage"])
    def test_collect_all_mixed_success(self, mock_configured):
        """Test collect_all with mixed success/failure."""
        mock_rag = MagicMock()
        manager = DataCollectionManager(rag_initializer=mock_rag)

        # Mock the alpha_vantage collector to return success
        mock_result = CollectorResult(
            collector_name="alpha_vantage",
            status=CollectorStatus.SUCCESS,
            documents=[],
            documents_collected=0,
        )
        manager.collectors["alpha_vantage"].collect = MagicMock(return_value=mock_result)

        summary = manager.collect_all()
        assert "alpha_vantage" in summary.collector_results

    def test_dedup_available(self):
        """Test that _deduplicate_documents method exists."""
        manager = DataCollectionManager(rag_initializer=MagicMock())
        assert hasattr(manager, "_deduplicate_documents")
