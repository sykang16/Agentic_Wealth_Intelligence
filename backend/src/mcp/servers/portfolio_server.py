"""Portfolio analysis MCP server.

Integrates portfolio data with RAG knowledge search.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal
from functools import partial
from pathlib import Path

from fastmcp.exceptions import ToolError

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.recommendation.rag.initializer import RAGInitializer

from ..schemas import (
    MCPDataSource,
    PortfolioAnalysisResult,
    PortfolioHolding,
    RAGSearchResult,
)
from .base import BaseMCPServer

logger = logging.getLogger(__name__)


class PortfolioMCPServer(BaseMCPServer):
    """MCP server for portfolio analysis.

    Provides tools for:
    - Analyzing portfolio metrics and risk
    - Getting portfolio holdings
    - Searching financial knowledge base
    - Getting portfolio recommendations

    Resources:
    - portfolio://users - Available user IDs
    """

    name = "portfolio"
    description = "Portfolio analysis and financial knowledge"

    def __init__(
        self,
        data_path: str | Path | None = None,
        rag_persist_dir: str | None = None,
    ):
        """Initialize portfolio MCP server.

        Args:
            data_path: Path to portfolio data JSON file.
            rag_persist_dir: Directory for RAG vector store.
        """
        # Set default paths
        project_root = Path(__file__).parent.parent.parent.parent.parent
        self.data_path = Path(data_path) if data_path else project_root / "data" / "synthetic" / "synthetic_portfolios.json"
        self.rag_persist_dir = rag_persist_dir or str(project_root / "data" / "chroma")

        # Initialize aggregator
        self._aggregator = PortfolioAggregator(self.data_path)
        self._aggregator_loaded = False

        # Initialize RAG (lazy)
        self._rag: RAGInitializer | None = None

        # Thread pool for sync operations
        self._executor = ThreadPoolExecutor(max_workers=2)

        super().__init__(data_source=MCPDataSource.PORTFOLIO)

    def _ensure_loaded(self) -> None:
        """Ensure portfolio data is loaded."""
        if not self._aggregator_loaded:
            if self.data_path.exists():
                self._aggregator.load_data()
                self._aggregator_loaded = True
            else:
                raise ToolError(f"Portfolio data not found: {self.data_path}")

    @property
    def rag(self) -> RAGInitializer:
        """Get or create RAG initializer."""
        if self._rag is None:
            self._rag = RAGInitializer(persist_directory=self.rag_persist_dir)
        return self._rag

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(func, *args, **kwargs),
        )

    def _register_tools(self) -> None:
        """Register MCP tools for portfolio analysis."""

        @self._mcp.tool()
        async def analyze_portfolio(
            user_id: str,
            include_risk: bool = True,
        ) -> dict:
            """Analyze portfolio with comprehensive metrics.

            Args:
                user_id: User ID to analyze.
                include_risk: Whether to include risk analysis.

            Returns:
                PortfolioAnalysisResult with metrics and analysis.
            """
            self.log_tool_call("analyze_portfolio", user_id=user_id, include_risk=include_risk)

            self._ensure_loaded()

            portfolio = self._aggregator.get_portfolio(user_id)
            if not portfolio:
                raise ToolError(f"Portfolio not found for user: {user_id}")

            summary = portfolio.summary
            if not summary:
                raise ToolError(f"No portfolio summary available for user: {user_id}")

            # Build holdings list
            holdings = []
            for h in portfolio.holdings[:10]:  # Top 10
                holdings.append(PortfolioHolding(
                    symbol=h.symbol,
                    name=h.name,
                    quantity=h.quantity,
                    average_cost=h.average_cost,
                    current_price=h.current_price,
                    market_value=h.market_value,
                    unrealized_gain_loss=h.unrealized_gain_loss,
                    unrealized_gain_loss_percent=h.unrealized_gain_loss_percent,
                    sector=h.sector,
                    asset_type=h.asset_type.value if h.asset_type else None,
                ))

            # Calculate risk factors
            risk_factors = []
            risk_score = 50.0  # Base score

            if include_risk:
                # Check concentration risk
                if holdings:
                    top_holding_pct = float(holdings[0].market_value or 0) / float(summary.total_assets) * 100
                    if top_holding_pct > 20:
                        risk_factors.append(f"High concentration: {holdings[0].symbol} is {top_holding_pct:.1f}% of portfolio")
                        risk_score += 10

                # Check liquidity
                if summary.liquidity_ratio < 0.1:
                    risk_factors.append("Low liquidity ratio (< 10%)")
                    risk_score += 15
                elif summary.liquidity_ratio > 0.3:
                    risk_score -= 10

                # Check debt
                if summary.debt_to_asset_ratio > 0.4:
                    risk_factors.append("High debt-to-asset ratio (> 40%)")
                    risk_score += 10
                elif summary.debt_to_asset_ratio < 0.2:
                    risk_score -= 5

                # Check diversification
                num_sectors = len(summary.allocation_by_sector)
                if num_sectors < 3:
                    risk_factors.append("Limited sector diversification")
                    risk_score += 10

            risk_score = max(0, min(100, risk_score))

            result = PortfolioAnalysisResult(
                user_id=user_id,
                net_worth=summary.total_net_worth,
                total_assets=summary.total_assets,
                total_liabilities=summary.total_liabilities,
                liquidity_ratio=summary.liquidity_ratio,
                debt_to_asset_ratio=summary.debt_to_asset_ratio,
                allocation_by_asset_type=summary.allocation_by_asset_type,
                allocation_by_sector=summary.allocation_by_sector,
                top_holdings=holdings,
                risk_score=risk_score if include_risk else None,
                risk_factors=risk_factors,
            )

            return result.model_dump(mode="json")

        @self._mcp.tool()
        async def get_portfolio_holdings(
            user_id: str,
            top_n: int = 10,
        ) -> dict:
            """Get top portfolio holdings.

            Args:
                user_id: User ID.
                top_n: Number of top holdings to return.

            Returns:
                List of PortfolioHolding objects.
            """
            self.log_tool_call("get_portfolio_holdings", user_id=user_id, top_n=top_n)

            self._ensure_loaded()

            holdings = self._aggregator.get_top_holdings(user_id, top_n)
            if not holdings:
                raise ToolError(f"No holdings found for user: {user_id}")

            result = []
            for h in holdings:
                result.append(PortfolioHolding(
                    symbol=h.symbol,
                    name=h.name,
                    quantity=h.quantity,
                    average_cost=h.average_cost,
                    current_price=h.current_price,
                    market_value=h.market_value,
                    unrealized_gain_loss=h.unrealized_gain_loss,
                    unrealized_gain_loss_percent=h.unrealized_gain_loss_percent,
                    sector=h.sector,
                    asset_type=h.asset_type.value if h.asset_type else None,
                ).model_dump(mode="json"))

            return {
                "user_id": user_id,
                "holdings": result,
                "count": len(result),
            }

        @self._mcp.tool()
        async def search_financial_knowledge(
            query: str,
            top_k: int = 5,
        ) -> dict:
            """Search the financial knowledge base using RAG.

            Args:
                query: Natural language search query.
                top_k: Number of results to return.

            Returns:
                RAGSearchResult with matching documents.
            """
            self.log_tool_call("search_financial_knowledge", query=query, top_k=top_k)

            # Run RAG search (sync operation in thread pool)
            try:
                search_result = await self._run_sync(
                    self.rag.search,
                    query,
                    top_k,
                )
            except Exception as e:
                raise ToolError(f"Knowledge search failed: {e}")

            result = RAGSearchResult(
                query=query,
                results=search_result.get("results", []),
                num_results=search_result.get("num_results", 0),
                context=search_result.get("context"),
                sources=[r.get("source", "") for r in search_result.get("results", [])],
            )

            return result.model_dump(mode="json")

        @self._mcp.tool()
        async def get_portfolio_recommendations(
            user_id: str,
            focus_area: str | None = None,
        ) -> dict:
            """Get portfolio recommendations based on analysis.

            Args:
                user_id: User ID.
                focus_area: Optional focus area (diversification, risk, income, growth).

            Returns:
                List of recommendations.
            """
            self.log_tool_call("get_portfolio_recommendations", user_id=user_id, focus_area=focus_area)

            self._ensure_loaded()

            portfolio = self._aggregator.get_portfolio(user_id)
            if not portfolio or not portfolio.summary:
                raise ToolError(f"Portfolio not found for user: {user_id}")

            summary = portfolio.summary
            recommendations = []

            # General recommendations
            if summary.liquidity_ratio < 0.1:
                recommendations.append({
                    "category": "liquidity",
                    "priority": "high",
                    "recommendation": "Consider increasing cash reserves to at least 10% of assets for emergency fund.",
                })

            if summary.debt_to_asset_ratio > 0.4:
                recommendations.append({
                    "category": "debt",
                    "priority": "high",
                    "recommendation": "Debt level is elevated. Consider prioritizing debt paydown.",
                })

            # Diversification recommendations
            if focus_area is None or focus_area == "diversification":
                allocation = summary.allocation_by_sector
                if len(allocation) < 5:
                    recommendations.append({
                        "category": "diversification",
                        "priority": "medium",
                        "recommendation": "Portfolio is concentrated in few sectors. Consider diversifying across more sectors.",
                    })

                # Check for over-allocation
                for sector, pct in allocation.items():
                    if pct > 40:
                        recommendations.append({
                            "category": "diversification",
                            "priority": "medium",
                            "recommendation": f"Over-allocated to {sector} ({pct:.1f}%). Consider rebalancing.",
                        })

            # Risk recommendations
            if focus_area is None or focus_area == "risk":
                top_holdings = self._aggregator.get_top_holdings(user_id, 1)
                if top_holdings:
                    top_pct = float(top_holdings[0].market_value) / float(summary.total_assets) * 100
                    if top_pct > 15:
                        recommendations.append({
                            "category": "risk",
                            "priority": "medium",
                            "recommendation": f"Single stock concentration risk: {top_holdings[0].symbol} is {top_pct:.1f}% of portfolio.",
                        })

            # Income recommendations
            if focus_area is None or focus_area == "income":
                # Check for dividend-focused holdings
                has_dividend_etfs = any(
                    h.symbol in ["VYM", "SCHD", "DVY", "HDV"]
                    for h in portfolio.holdings
                )
                if not has_dividend_etfs and portfolio.user.age and portfolio.user.age > 50:
                    recommendations.append({
                        "category": "income",
                        "priority": "low",
                        "recommendation": "Consider dividend-focused ETFs for income generation.",
                    })

            # Growth recommendations
            if focus_area is None or focus_area == "growth":
                savings_rate = summary.monthly_savings_rate or 0
                if savings_rate < 0.1 and portfolio.user.age and portfolio.user.age < 45:
                    recommendations.append({
                        "category": "growth",
                        "priority": "medium",
                        "recommendation": "Savings rate is low. Consider increasing monthly contributions.",
                    })

            return {
                "user_id": user_id,
                "focus_area": focus_area,
                "recommendations": recommendations,
                "count": len(recommendations),
            }

    def _register_resources(self) -> None:
        """Register MCP resources for portfolio data."""

        @self._mcp.resource("portfolio://users")
        def list_users() -> str:
            """List available user IDs."""
            import json

            try:
                self._ensure_loaded()
                user_ids = self._aggregator.get_user_ids()

                users = []
                for uid in user_ids:
                    portfolio = self._aggregator.get_portfolio(uid)
                    if portfolio:
                        users.append({
                            "user_id": uid,
                            "name": portfolio.user.name,
                            "age": portfolio.user.age,
                            "occupation": portfolio.user.occupation,
                        })

                return json.dumps(users, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})


def main():
    """Run the portfolio MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = PortfolioMCPServer()
    server.run()


if __name__ == "__main__":
    main()
