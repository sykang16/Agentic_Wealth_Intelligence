"""Base MCP server class using FastMCP framework."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from ..schemas import MCPDataSource, MCPToolResult, MCPToolStatus

logger = logging.getLogger(__name__)


class BaseMCPServer(ABC):
    """Abstract base class for MCP servers.

    Provides common functionality for MCP servers including:
    - Tool and resource registration via FastMCP
    - Standardized result creation
    - Error handling helpers
    """

    name: str = "base"
    description: str = "Base MCP server"

    def __init__(self, data_source: MCPDataSource | None = None):
        """Initialize the MCP server.

        Args:
            data_source: The primary data source for this server.
        """
        self.data_source = data_source
        self._mcp = FastMCP(self.name)
        self._register_tools()
        self._register_resources()

    @property
    def mcp(self) -> FastMCP:
        """Get the FastMCP instance."""
        return self._mcp

    @abstractmethod
    def _register_tools(self) -> None:
        """Register MCP tools for this server.

        Subclasses must implement this method to register their tools
        using self._mcp.tool() decorator pattern.
        """
        pass

    def _register_resources(self) -> None:
        """Register MCP resources for this server.

        Subclasses can override this method to register resources
        using self._mcp.resource() decorator pattern.
        Default implementation does nothing.
        """
        pass

    def create_result(
        self,
        status: MCPToolStatus,
        data: dict | list | None = None,
        message: str | None = None,
        cached: bool = False,
    ) -> MCPToolResult:
        """Create a standardized MCP tool result.

        Args:
            status: The status of the tool execution.
            data: The result data.
            message: Optional message describing the result.
            cached: Whether the result was from cache.

        Returns:
            MCPToolResult instance.
        """
        return MCPToolResult(
            status=status,
            data=data,
            message=message,
            source=self.data_source,
            cached=cached,
            timestamp=datetime.now(),
        )

    def create_success(
        self,
        data: dict | list | None = None,
        message: str | None = None,
        cached: bool = False,
    ) -> MCPToolResult:
        """Create a successful result.

        Args:
            data: The result data.
            message: Optional success message.
            cached: Whether the result was from cache.

        Returns:
            MCPToolResult with SUCCESS status.
        """
        return self.create_result(
            status=MCPToolStatus.SUCCESS,
            data=data,
            message=message,
            cached=cached,
        )

    def create_error(
        self,
        message: str,
        data: dict | None = None,
    ) -> MCPToolResult:
        """Create an error result.

        Args:
            message: Error message.
            data: Optional additional error data.

        Returns:
            MCPToolResult with ERROR status.
        """
        return self.create_result(
            status=MCPToolStatus.ERROR,
            data=data,
            message=message,
        )

    def raise_tool_error(self, message: str) -> None:
        """Raise a ToolError for client-visible errors.

        Use this for errors that should be shown to the MCP client,
        such as invalid parameters or missing data.

        Args:
            message: Error message to show to client.

        Raises:
            ToolError: Always raises with the given message.
        """
        logger.error(f"Tool error in {self.name}: {message}")
        raise ToolError(message)

    def log_tool_call(self, tool_name: str, **kwargs) -> None:
        """Log a tool invocation for debugging.

        Args:
            tool_name: Name of the tool being called.
            **kwargs: Tool parameters.
        """
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"[{self.name}] {tool_name}({params})")

    def run(self) -> None:
        """Run the MCP server (blocking).

        This starts the server in stdio mode for MCP communication.
        """
        logger.info(f"Starting MCP server: {self.name}")
        self._mcp.run()

    async def run_async(self) -> None:
        """Run the MCP server asynchronously.

        This starts the server in stdio mode for MCP communication.
        """
        logger.info(f"Starting MCP server (async): {self.name}")
        await self._mcp.run_async()
