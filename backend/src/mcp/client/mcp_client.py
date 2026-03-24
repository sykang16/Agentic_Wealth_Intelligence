"""Unified MCP client for connecting to multiple servers."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .rate_limiter import MultiSourceRateLimiter, RateLimitConfig
from .retry_handler import RetryConfig, RetryHandler

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    rate_limit: RateLimitConfig | None = None


class MCPClient:
    """Unified client for connecting to multiple MCP servers.

    Features:
    - Connect to multiple MCP servers via stdio
    - Unified tool invocation across servers
    - Tool discovery
    - Resource reading
    - Rate limiting per server
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        server_configs: list[ServerConfig] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize the MCP client.

        Args:
            server_configs: List of server configurations.
            retry_config: Configuration for retry behavior.
        """
        self._server_configs: dict[str, ServerConfig] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._connected: dict[str, bool] = {}

        # Rate limiting
        rate_configs = {}
        if server_configs:
            for config in server_configs:
                self._server_configs[config.name] = config
                if config.rate_limit:
                    rate_configs[config.name] = config.rate_limit

        self._rate_limiter = MultiSourceRateLimiter(rate_configs)

        # Retry handler
        self._retry_handler = RetryHandler(retry_config)

    def add_server(self, config: ServerConfig) -> None:
        """Add a server configuration.

        Args:
            config: Server configuration.
        """
        self._server_configs[config.name] = config
        if config.rate_limit:
            self._rate_limiter._limiters[config.name] = self._rate_limiter.get_limiter(config.name)
            self._rate_limiter._limiters[config.name].config = config.rate_limit

    async def connect(self, server_name: str) -> ClientSession:
        """Connect to an MCP server.

        Args:
            server_name: Name of the server to connect to.

        Returns:
            ClientSession for the server.

        Raises:
            ValueError: If server is not configured.
            ConnectionError: If connection fails.
        """
        if server_name not in self._server_configs:
            raise ValueError(f"Server not configured: {server_name}")

        if server_name in self._sessions and self._connected.get(server_name):
            return self._sessions[server_name]

        config = self._server_configs[server_name]

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env or None,
        )

        try:
            # Create stdio client connection
            read, write = await stdio_client(server_params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()

            self._sessions[server_name] = session
            self._connected[server_name] = True

            logger.info(f"Connected to MCP server: {server_name}")
            return session

        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            raise ConnectionError(f"Failed to connect to {server_name}: {e}")

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of the server to disconnect from.
        """
        if server_name in self._sessions:
            try:
                # Close the session
                self._connected[server_name] = False
                del self._sessions[server_name]
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for server_name in list(self._sessions.keys()):
            await self.disconnect(server_name)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict | None = None,
        wait_for_rate_limit: bool = True,
    ) -> dict:
        """Call a tool on an MCP server.

        Args:
            server_name: Name of the server.
            tool_name: Name of the tool to call.
            arguments: Tool arguments.
            wait_for_rate_limit: Whether to wait if rate limited.

        Returns:
            Tool result as a dictionary.

        Raises:
            ValueError: If server not connected.
            RuntimeError: If tool call fails.
        """
        # Check rate limit
        if wait_for_rate_limit:
            acquired = await self._rate_limiter.wait_for_token(server_name)
        else:
            acquired = await self._rate_limiter.acquire(server_name)

        if not acquired:
            raise RuntimeError(f"Rate limited for server: {server_name}")

        # Get or create session
        session = await self.connect(server_name)

        async def _call():
            result = await session.call_tool(tool_name, arguments or {})
            return result

        try:
            result = await self._retry_handler.execute_with_retry(_call)

            # Parse result content
            if hasattr(result, "content") and result.content:
                content = result.content[0]
                if hasattr(content, "text"):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"text": content.text}

            return {"result": result}

        except Exception as e:
            logger.error(f"Tool call failed: {server_name}/{tool_name}: {e}")
            raise RuntimeError(f"Tool call failed: {e}")

    async def list_tools(self, server_name: str) -> list[dict]:
        """List available tools on a server.

        Args:
            server_name: Name of the server.

        Returns:
            List of tool descriptions.
        """
        session = await self.connect(server_name)
        tools = await session.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else None,
            }
            for tool in tools.tools
        ]

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read a resource from an MCP server.

        Args:
            server_name: Name of the server.
            uri: Resource URI.

        Returns:
            Resource content as string.
        """
        session = await self.connect(server_name)
        result = await session.read_resource(uri)

        if hasattr(result, "contents") and result.contents:
            content = result.contents[0]
            if hasattr(content, "text"):
                return content.text

        return str(result)

    async def list_resources(self, server_name: str) -> list[dict]:
        """List available resources on a server.

        Args:
            server_name: Name of the server.

        Returns:
            List of resource descriptions.
        """
        session = await self.connect(server_name)
        resources = await session.list_resources()

        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description if hasattr(resource, "description") else None,
            }
            for resource in resources.resources
        ]

    def get_rate_limit_status(self, server_name: str | None = None) -> dict:
        """Get rate limit status.

        Args:
            server_name: Specific server name, or None for all.

        Returns:
            Rate limit status dictionary.
        """
        if server_name:
            return self._rate_limiter.get_limiter(server_name).get_status()
        return self._rate_limiter.get_all_status()


def create_default_client() -> MCPClient:
    """Create an MCP client with default server configurations.

    Returns:
        Configured MCPClient instance.
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent.parent

    # Default server configurations
    servers = [
        ServerConfig(
            name="news",
            command="python",
            args=["-m", "backend.src.mcp.servers.news_server"],
            rate_limit=RateLimitConfig(calls_per_minute=5, calls_per_day=100),
        ),
        ServerConfig(
            name="market",
            command="python",
            args=["-m", "backend.src.mcp.servers.market_server"],
            rate_limit=RateLimitConfig(calls_per_minute=5, calls_per_day=500),
        ),
        ServerConfig(
            name="portfolio",
            command="python",
            args=["-m", "backend.src.mcp.servers.portfolio_server"],
            rate_limit=RateLimitConfig(calls_per_minute=60, calls_per_day=10000),
        ),
    ]

    return MCPClient(server_configs=servers)
