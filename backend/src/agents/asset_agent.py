"""Asset Management Agent - Module A.

This agent handles natural language queries about user's financial portfolio,
providing insights, metrics, and dynamic visualizations.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.asset_management.query_interface import (
    AssetQueryInterface,
    QueryResponse,
    QueryType,
)
from backend.src.asset_management.visualization import VisualizationEngine
from backend.src.common.llm_client import LLMClient


class AssetAgentState(BaseModel):
    """State for the Asset Agent."""

    user_id: str
    query: str
    response: QueryResponse | None = None
    visualization: Any | None = None
    error: str | None = None


class AssetAgentResponse(BaseModel):
    """Response from the Asset Agent."""

    answer: str
    query_type: str
    visualization_type: str | None = None
    visualization: Any | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None


class AssetAgent:
    """Agent for handling asset management queries.

    This agent:
    1. Receives natural language queries about user's portfolio
    2. Parses intent and extracts relevant data
    3. Generates natural language responses
    4. Creates appropriate visualizations
    """

    def __init__(
        self,
        aggregator: PortfolioAggregator,
        llm_client: LLMClient | None = None,
    ):
        """Initialize Asset Agent.

        Args:
            aggregator: Portfolio data aggregator with loaded data.
            llm_client: Optional LLM client.
        """
        self.aggregator = aggregator
        self.llm = llm_client or LLMClient()
        self.query_interface = AssetQueryInterface(aggregator, self.llm)
        self.viz_engine = VisualizationEngine()

    def process(self, user_id: str, query: str) -> AssetAgentResponse:
        """Process a user query.

        Args:
            user_id: User ID to query data for.
            query: Natural language query.

        Returns:
            AssetAgentResponse with answer and optional visualization.
        """
        try:
            # Verify user exists
            portfolio = self.aggregator.get_portfolio(user_id)
            if not portfolio:
                return AssetAgentResponse(
                    answer=f"No portfolio data found for user {user_id}.",
                    query_type="error",
                    success=False,
                    error="User not found",
                )

            # Process the query
            response = self.query_interface.process_query(query, user_id)

            # Generate visualization if needed
            visualization = None
            if response.visualization_type:
                visualization = self._create_visualization(
                    response.query_type,
                    response.visualization_type,
                    response.data,
                    portfolio,
                )

            return AssetAgentResponse(
                answer=response.answer,
                query_type=response.query_type.value,
                visualization_type=response.visualization_type,
                visualization=visualization,
                data=response.data,
                success=True,
            )

        except Exception as e:
            return AssetAgentResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                query_type="error",
                success=False,
                error=str(e),
            )

    def _create_visualization(
        self,
        query_type: QueryType,
        viz_type: str,
        data: dict[str, Any],
        portfolio: Any,
    ) -> Any:
        """Create appropriate visualization based on query type."""
        try:
            if query_type == QueryType.ASSET_ALLOCATION:
                excluded = data.get("excluded", [])
                title = "Asset Allocation"
                if excluded:
                    excluded_str = ", ".join(e.replace("_", " ").title() for e in excluded)
                    title = f"Asset Allocation (excluding {excluded_str})"
                return self.viz_engine.create_asset_allocation_pie(
                    data.get("allocation", {}), title=title
                )

            elif query_type == QueryType.SECTOR_ALLOCATION:
                return self.viz_engine.create_sector_allocation_pie(
                    data.get("allocation", {})
                )

            elif query_type == QueryType.NET_WORTH:
                return self.viz_engine.create_net_worth_breakdown(portfolio)

            elif query_type == QueryType.TOP_HOLDINGS:
                holdings = self.aggregator.get_top_holdings(
                    portfolio.user.user_id, 10
                )
                return self.viz_engine.create_holdings_bar(holdings, by="value")

            elif query_type == QueryType.HOLDINGS:
                return self.viz_engine.create_holdings_by_sector(portfolio.holdings)

            elif query_type == QueryType.GAINS_LOSSES:
                return self.viz_engine.create_gain_loss_chart(portfolio.holdings)

            elif query_type == QueryType.LIQUIDITY:
                metrics = data.get("metrics", {})
                return self.viz_engine.create_metrics_dashboard(metrics)

            elif query_type == QueryType.METRICS:
                metrics = data.get("metrics", {})
                return self.viz_engine.create_metrics_dashboard(metrics)

            elif query_type in [QueryType.BANK_ACCOUNTS, QueryType.INVESTMENT_ACCOUNTS]:
                return self.viz_engine.create_account_summary_table(portfolio)

            elif query_type == QueryType.REAL_ESTATE:
                # For real estate, we'll use a table-like visualization
                return self.viz_engine.create_account_summary_table(portfolio)

            else:
                # Default to metrics dashboard for general queries
                metrics = self.aggregator.get_portfolio_metrics(portfolio.user.user_id)
                return self.viz_engine.create_metrics_dashboard(metrics)

        except Exception:
            return None

    def get_quick_summary(self, user_id: str) -> dict[str, Any]:
        """Get a quick portfolio summary without LLM call.

        Args:
            user_id: User ID.

        Returns:
            Dict with key metrics.
        """
        metrics = self.aggregator.get_portfolio_metrics(user_id)
        allocation = self.aggregator.get_asset_allocation(user_id)
        top_holdings = self.aggregator.get_top_holdings(user_id, 5)

        return {
            "metrics": metrics,
            "allocation": allocation,
            "top_holdings": [
                {
                    "symbol": h.symbol,
                    "name": h.name,
                    "value": float(h.market_value),
                    "gain_loss_pct": h.unrealized_gain_loss_percent,
                }
                for h in top_holdings
            ],
        }

    def get_supported_queries(self) -> list[dict[str, str]]:
        """Get list of supported query types with examples."""
        return [
            {
                "type": "Net Worth",
                "examples": [
                    "What's my net worth?",
                    "Show me my total assets and liabilities",
                    "Give me a net worth breakdown",
                ],
            },
            {
                "type": "Asset Allocation",
                "examples": [
                    "Show my asset allocation",
                    "What's my portfolio breakdown?",
                    "How are my assets distributed?",
                ],
            },
            {
                "type": "Sector Allocation",
                "examples": [
                    "What sectors am I invested in?",
                    "Show sector breakdown",
                    "How diversified am I across sectors?",
                ],
            },
            {
                "type": "Liquidity",
                "examples": [
                    "What's my liquidity ratio?",
                    "How much cash do I have?",
                    "Am I liquid enough?",
                ],
            },
            {
                "type": "Holdings",
                "examples": [
                    "Show my top holdings",
                    "What are my largest positions?",
                    "List my investments",
                ],
            },
            {
                "type": "Gains & Losses",
                "examples": [
                    "What are my unrealized gains?",
                    "Which stocks are up/down?",
                    "Show my profit and loss",
                ],
            },
            {
                "type": "Real Estate",
                "examples": [
                    "What's my real estate worth?",
                    "Show my property portfolio",
                    "How much equity do I have in my home?",
                ],
            },
            {
                "type": "Accounts",
                "examples": [
                    "Show my bank accounts",
                    "What's in my investment accounts?",
                    "List all my accounts",
                ],
            },
            {
                "type": "Metrics",
                "examples": [
                    "Give me a portfolio overview",
                    "What are my key financial metrics?",
                    "Show my savings rate",
                ],
            },
        ]
