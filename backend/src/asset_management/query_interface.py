"""LLM-powered natural language query interface for asset management."""

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient


class QueryType(str, Enum):
    """Types of asset queries."""

    NET_WORTH = "net_worth"
    ASSET_ALLOCATION = "asset_allocation"
    SECTOR_ALLOCATION = "sector_allocation"
    LIQUIDITY = "liquidity"
    HOLDINGS = "holdings"
    TOP_HOLDINGS = "top_holdings"
    REAL_ESTATE = "real_estate"
    BANK_ACCOUNTS = "bank_accounts"
    INVESTMENT_ACCOUNTS = "investment_accounts"
    GAINS_LOSSES = "gains_losses"
    METRICS = "metrics"
    GENERAL = "general"


class QueryIntent(BaseModel):
    """Parsed query intent."""

    query_type: QueryType
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_visualization: bool = False
    visualization_type: str | None = None


class QueryResponse(BaseModel):
    """Response to a user query."""

    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    visualization_type: str | None = None
    query_type: QueryType


class AssetQueryInterface:
    """Natural language query interface for asset management."""

    SYSTEM_PROMPT = """You are a financial assistant analyzing a user's portfolio.
You have access to their complete financial data including bank accounts, investments,
real estate, and calculated metrics.

When answering questions:
1. Be specific with numbers and percentages
2. Provide context (e.g., compare to recommended thresholds)
3. Be concise but thorough
4. If relevant, suggest what visualization would help (pie chart, bar chart, etc.)

Always base your answers on the provided portfolio data. Do not make up numbers."""

    INTENT_PROMPT = """Analyze this user query about their financial portfolio and extract the intent.

Query: "{query}"

Respond with a JSON object containing:
- query_type: one of [net_worth, asset_allocation, sector_allocation, liquidity, holdings, top_holdings, real_estate, bank_accounts, investment_accounts, gains_losses, metrics, general]
- requires_visualization: boolean indicating if a chart would be helpful
- visualization_type: if visualization needed, one of [pie, bar, waterfall, treemap, table, gauge, none]
- parameters: object with specific parameters:
  - top_n: number for "show top N holdings"
  - exclude: array of asset types to exclude (e.g., ["real_estate"] for "except real estate")

Respond ONLY with the JSON object, no other text."""

    def __init__(
        self,
        aggregator: PortfolioAggregator,
        llm_client: LLMClient | None = None,
    ):
        """Initialize query interface.

        Args:
            aggregator: Portfolio data aggregator.
            llm_client: Optional LLM client. Creates new one if not provided.
        """
        self.aggregator = aggregator
        self.llm = llm_client or LLMClient()

    def parse_intent(self, query: str) -> QueryIntent:
        """Parse user query to extract intent.

        Args:
            query: Natural language query.

        Returns:
            Parsed query intent.
        """
        prompt = self.INTENT_PROMPT.format(query=query)

        try:
            response = self.llm.chat(prompt, max_tokens=256, temperature=0.1)
            content = response.content.strip()

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
            else:
                intent_data = json.loads(content)

            parameters = intent_data.get("parameters", {})

            # Always check for exclusions in the query (in case LLM missed them)
            exclude = self._extract_exclusions(query)
            if exclude:
                parameters["exclude"] = exclude

            return QueryIntent(
                query_type=QueryType(intent_data.get("query_type", "general")),
                parameters=parameters,
                requires_visualization=intent_data.get("requires_visualization", False),
                visualization_type=intent_data.get("visualization_type"),
            )
        except Exception:
            # Fallback to keyword-based intent detection
            return self._fallback_intent_detection(query)

    def _extract_exclusions(self, query: str) -> list[str]:
        """Extract asset types to exclude from query."""
        query_lower = query.lower()
        exclude = []

        # Check if query contains exclusion keywords
        if any(word in query_lower for word in ["except", "excluding", "without", "exclude"]):
            # Common asset types to check for exclusion
            asset_mappings = {
                "real estate": "real_estate",
                "realestate": "real_estate",
                "property": "real_estate",
                "stocks": "stocks",
                "stock": "stocks",
                "bonds": "bonds",
                "bond": "bonds",
                "etfs": "etfs",
                "etf": "etfs",
                "crypto": "crypto",
                "cryptocurrency": "crypto",
                "cash": "cash",
                "mutual funds": "mutual_funds",
                "mutual fund": "mutual_funds",
            }
            for keyword, asset_type in asset_mappings.items():
                if keyword in query_lower and asset_type not in exclude:
                    exclude.append(asset_type)

        return exclude

    def _fallback_intent_detection(self, query: str) -> QueryIntent:
        """Fallback intent detection using keywords."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["net worth", "worth", "total"]):
            return QueryIntent(
                query_type=QueryType.NET_WORTH,
                requires_visualization=True,
                visualization_type="waterfall",
            )
        elif any(word in query_lower for word in ["allocation", "breakdown", "distribution"]):
            if "sector" in query_lower:
                return QueryIntent(
                    query_type=QueryType.SECTOR_ALLOCATION,
                    requires_visualization=True,
                    visualization_type="pie",
                )
            # Check for exclusions
            exclude = self._extract_exclusions(query)

            return QueryIntent(
                query_type=QueryType.ASSET_ALLOCATION,
                requires_visualization=True,
                visualization_type="pie",
                parameters={"exclude": exclude} if exclude else {},
            )
        elif any(word in query_lower for word in ["liquidity", "liquid", "cash"]):
            return QueryIntent(
                query_type=QueryType.LIQUIDITY,
                requires_visualization=True,
                visualization_type="gauge",
            )
        elif any(word in query_lower for word in ["top", "best", "largest"]):
            return QueryIntent(
                query_type=QueryType.TOP_HOLDINGS,
                requires_visualization=True,
                visualization_type="bar",
            )
        elif any(word in query_lower for word in ["holding", "stock", "investment", "position"]):
            return QueryIntent(
                query_type=QueryType.HOLDINGS,
                requires_visualization=True,
                visualization_type="treemap",
            )
        elif any(word in query_lower for word in ["real estate", "property", "house", "home"]):
            return QueryIntent(
                query_type=QueryType.REAL_ESTATE,
                requires_visualization=True,
                visualization_type="table",
            )
        elif any(word in query_lower for word in ["gain", "loss", "profit", "return"]):
            return QueryIntent(
                query_type=QueryType.GAINS_LOSSES,
                requires_visualization=True,
                visualization_type="bar",
            )
        elif any(word in query_lower for word in ["bank", "checking", "savings"]):
            return QueryIntent(
                query_type=QueryType.BANK_ACCOUNTS,
                requires_visualization=True,
                visualization_type="table",
            )
        elif any(word in query_lower for word in ["metric", "ratio", "overview", "summary"]):
            return QueryIntent(
                query_type=QueryType.METRICS,
                requires_visualization=True,
                visualization_type="gauge",
            )
        else:
            return QueryIntent(query_type=QueryType.GENERAL)

    def process_query(self, query: str, user_id: str) -> QueryResponse:
        """Process a natural language query.

        Args:
            query: User's question.
            user_id: User ID to query data for.

        Returns:
            QueryResponse with answer and optional visualization info.
        """
        # Parse intent
        intent = self.parse_intent(query)

        # Get relevant data based on intent
        data = self._get_data_for_intent(intent, user_id)

        # Generate answer using LLM
        answer = self._generate_answer(query, user_id, intent, data)

        return QueryResponse(
            answer=answer,
            data=data,
            visualization_type=intent.visualization_type if intent.requires_visualization else None,
            query_type=intent.query_type,
        )

    def _get_data_for_intent(self, intent: QueryIntent, user_id: str) -> dict[str, Any]:
        """Get relevant data based on query intent."""
        data: dict[str, Any] = {}

        if intent.query_type == QueryType.NET_WORTH:
            data["net_worth"] = self.aggregator.get_net_worth(user_id)
            data["metrics"] = self.aggregator.get_portfolio_metrics(user_id)

        elif intent.query_type == QueryType.ASSET_ALLOCATION:
            exclude = intent.parameters.get("exclude", [])
            data["allocation"] = self.aggregator.get_asset_allocation(user_id, exclude=exclude)
            if exclude:
                data["excluded"] = exclude

        elif intent.query_type == QueryType.SECTOR_ALLOCATION:
            data["allocation"] = self.aggregator.get_sector_allocation(user_id)

        elif intent.query_type == QueryType.LIQUIDITY:
            data["liquidity_ratio"] = self.aggregator.get_liquidity_ratio(user_id)
            data["cash"] = self.aggregator.get_total_cash(user_id)
            data["metrics"] = self.aggregator.get_portfolio_metrics(user_id)

        elif intent.query_type == QueryType.TOP_HOLDINGS:
            n = intent.parameters.get("top_n", 5)
            holdings = self.aggregator.get_top_holdings(user_id, n)
            data["holdings"] = [
                {
                    "symbol": h.symbol,
                    "name": h.name,
                    "value": float(h.market_value),
                    "gain_loss": float(h.unrealized_gain_loss),
                    "gain_loss_pct": h.unrealized_gain_loss_percent,
                }
                for h in holdings
            ]

        elif intent.query_type == QueryType.HOLDINGS:
            portfolio = self.aggregator.get_portfolio(user_id)
            if portfolio:
                data["holdings"] = [
                    {
                        "symbol": h.symbol,
                        "name": h.name,
                        "type": h.asset_type.value,
                        "sector": h.sector,
                        "value": float(h.market_value),
                        "gain_loss": float(h.unrealized_gain_loss),
                    }
                    for h in portfolio.holdings
                ]
            data["by_type"] = {
                k: len(v) for k, v in self.aggregator.get_holdings_by_type(user_id).items()
            }

        elif intent.query_type == QueryType.REAL_ESTATE:
            data["real_estate"] = self.aggregator.get_real_estate_summary(user_id)

        elif intent.query_type == QueryType.BANK_ACCOUNTS:
            portfolio = self.aggregator.get_portfolio(user_id)
            if portfolio:
                data["accounts"] = [
                    {
                        "bank": acc.bank_name,
                        "type": acc.account_type.value,
                        "balance": float(acc.balance),
                        "interest_rate": float(acc.interest_rate),
                    }
                    for acc in portfolio.bank_accounts
                ]

        elif intent.query_type == QueryType.GAINS_LOSSES:
            portfolio = self.aggregator.get_portfolio(user_id)
            if portfolio:
                data["holdings"] = [
                    {
                        "symbol": h.symbol,
                        "gain_loss": float(h.unrealized_gain_loss),
                        "gain_loss_pct": h.unrealized_gain_loss_percent,
                    }
                    for h in sorted(
                        portfolio.holdings,
                        key=lambda x: x.unrealized_gain_loss,
                        reverse=True,
                    )
                ]

        elif intent.query_type == QueryType.METRICS:
            data["metrics"] = self.aggregator.get_portfolio_metrics(user_id)

        else:  # GENERAL
            data["metrics"] = self.aggregator.get_portfolio_metrics(user_id)
            data["allocation"] = self.aggregator.get_asset_allocation(user_id)

        return data

    def _generate_answer(
        self,
        query: str,
        user_id: str,
        intent: QueryIntent,
        data: dict[str, Any],
    ) -> str:
        """Generate natural language answer using LLM."""
        # Get full portfolio context
        context = self.aggregator.format_portfolio_context(user_id)

        # Format relevant data
        data_context = json.dumps(data, indent=2, default=str)

        prompt = f"""Based on the following portfolio data, answer the user's question.

{context}

Relevant data for this query:
{data_context}

User question: {query}

Provide a clear, specific answer with actual numbers from the data. Be concise but informative."""

        response = self.llm.chat(prompt, system_prompt=self.SYSTEM_PROMPT, max_tokens=512)
        return response.content
