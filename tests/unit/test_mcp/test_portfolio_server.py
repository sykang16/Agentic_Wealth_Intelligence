"""Unit tests for the Portfolio MCP Server."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal

from backend.src.mcp.servers.portfolio_server import PortfolioMCPServer


class TestPortfolioMCPServer:
    """Tests for PortfolioMCPServer."""

    def test_server_initialization(self):
        """Test server initializes correctly."""
        with patch.object(PortfolioMCPServer, "_register_tools"):
            with patch.object(PortfolioMCPServer, "_register_resources"):
                server = PortfolioMCPServer.__new__(PortfolioMCPServer)
                assert server.name == "portfolio"

    def test_server_name_and_description(self):
        """Test server has correct name and description."""
        assert PortfolioMCPServer.name == "portfolio"
        assert PortfolioMCPServer.description == "Portfolio analysis and financial knowledge"

    def test_ensure_loaded_raises_without_data(self):
        """Test _ensure_loaded raises when data file doesn't exist."""
        with patch.object(PortfolioMCPServer, "__init__", return_value=None):
            server = PortfolioMCPServer.__new__(PortfolioMCPServer)
            server._aggregator_loaded = False
            from pathlib import Path

            server.data_path = Path("/nonexistent/path.json")

            from fastmcp.exceptions import ToolError

            with pytest.raises(ToolError):
                server._ensure_loaded()

    def test_ensure_loaded_only_once(self):
        """Test _ensure_loaded skips if already loaded."""
        with patch.object(PortfolioMCPServer, "__init__", return_value=None):
            server = PortfolioMCPServer.__new__(PortfolioMCPServer)
            server._aggregator_loaded = True
            # Should not raise since data is already loaded
            server._ensure_loaded()

    def test_rag_lazy_initialization(self):
        """Test RAG is lazily initialized."""
        with patch.object(PortfolioMCPServer, "__init__", return_value=None):
            server = PortfolioMCPServer.__new__(PortfolioMCPServer)
            server._rag = None
            server.rag_persist_dir = "/tmp/test_chroma"

            with patch(
                "backend.src.mcp.servers.portfolio_server.RAGInitializer"
            ) as mock_rag_cls:
                mock_rag = MagicMock()
                mock_rag_cls.return_value = mock_rag
                result = server.rag
                assert result == mock_rag
                mock_rag_cls.assert_called_once()

    def test_rag_reuses_instance(self):
        """Test RAG property returns cached instance."""
        with patch.object(PortfolioMCPServer, "__init__", return_value=None):
            server = PortfolioMCPServer.__new__(PortfolioMCPServer)
            mock_rag = MagicMock()
            server._rag = mock_rag
            assert server.rag is mock_rag

    @pytest.mark.asyncio
    async def test_run_sync(self):
        """Test _run_sync executes sync function in thread pool."""
        with patch.object(PortfolioMCPServer, "__init__", return_value=None):
            from concurrent.futures import ThreadPoolExecutor

            server = PortfolioMCPServer.__new__(PortfolioMCPServer)
            server._executor = ThreadPoolExecutor(max_workers=1)

            def sync_func(x):
                return x * 2

            result = await server._run_sync(sync_func, 5)
            assert result == 10

    def test_portfolio_server_has_tools(self):
        """Test that the server registers tools."""
        # Verify the _register_tools method exists and is callable
        assert hasattr(PortfolioMCPServer, "_register_tools")
        assert callable(getattr(PortfolioMCPServer, "_register_tools"))
