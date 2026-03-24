"""Context builder that aggregates RAG, portfolio, profile, and live market data."""

import logging

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.models import AssetType, InvestmentProfile
from backend.src.recommendation.rag.initializer import RAGInitializer

from .schemas import AggregatedContext

logger = logging.getLogger(__name__)

# Asset types with tradable market quotes
_QUOTABLE = {AssetType.STOCK, AssetType.ETF, AssetType.BOND}


class ContextBuilder:
    """Builds aggregated context from all available data sources.

    Combines:
    - Portfolio data (holdings, allocations, metrics)
    - Investment profile (risk tolerance, goals, preferences)
    - RAG knowledge base (static financial documents)
    - Live market data via MCP -> MarketMCPServer -> yfinance (quotes, news)
    """

    def __init__(
        self,
        portfolio_aggregator: PortfolioAggregator,
        rag_initializer: RAGInitializer,
        hybrid_data_provider=None,
    ):
        self._aggregator = portfolio_aggregator
        self._rag = rag_initializer
        self._hybrid_provider = hybrid_data_provider

    def build_context(
        self,
        user_id: str,
        query: str = "",
        include_live_data: bool = True,
    ) -> AggregatedContext:
        """Build aggregated context for recommendation generation."""
        context = AggregatedContext()
        data_sources: list[str] = []

        # 1. Portfolio context
        self._add_portfolio_context(user_id, context, data_sources)

        # 2. Investment profile context
        self._add_profile_context(user_id, context, data_sources)

        # 3. RAG knowledge context
        search_query = query or self._build_rag_query(context)
        self._add_rag_context(search_query, context, data_sources)

        # 4. Live market data via MCP -> yfinance
        if include_live_data and context.portfolio_tickers and self._hybrid_provider:
            self._add_live_context(search_query, context.portfolio_tickers, context, data_sources)

        context.data_sources_used = data_sources
        return context

    # ------------------------------------------------------------------
    # Portfolio context
    # ------------------------------------------------------------------

    def _add_portfolio_context(
        self, user_id: str, context: AggregatedContext, sources: list[str]
    ) -> None:
        try:
            portfolio_text = self._aggregator.format_portfolio_context(user_id)
            context.portfolio_context = portfolio_text

            portfolio = self._aggregator.get_portfolio(user_id)
            if portfolio:
                context.portfolio_tickers = [
                    h.symbol for h in portfolio.holdings
                    if h.asset_type in _QUOTABLE
                ]
                if portfolio.summary:
                    context.current_sector_allocation = (
                        portfolio.summary.allocation_by_sector
                    )
                    context.current_asset_allocation = (
                        portfolio.summary.allocation_by_asset_type
                    )
            sources.append("portfolio")
        except Exception as e:
            logger.error("Failed to get portfolio context: %s", e)

    # ------------------------------------------------------------------
    # Profile context
    # ------------------------------------------------------------------

    def _add_profile_context(
        self, user_id: str, context: AggregatedContext, sources: list[str]
    ) -> None:
        try:
            portfolio = self._aggregator.get_portfolio(user_id)
            if portfolio and portfolio.investment_profile:
                profile = portfolio.investment_profile
                context.user_risk_tolerance = (
                    profile.risk_tolerance.value if profile.risk_tolerance else None
                )
                context.user_investment_horizon = (
                    profile.investment_horizon.value
                    if profile.investment_horizon else None
                )
                context.user_experience_level = (
                    profile.investment_experience.value
                    if profile.investment_experience else None
                )
                context.excluded_sectors = profile.excluded_sectors
                context.investment_profile_context = self._format_profile(profile)
                sources.append("investment_profile")
            else:
                context.investment_profile_context = (
                    "## Investment Profile\nNo investment profile available. "
                    "Using conservative defaults for recommendations."
                )
        except Exception as e:
            logger.error("Failed to get profile context: %s", e)

    def _format_profile(self, profile: InvestmentProfile) -> str:
        parts = ["## Investment Profile"]
        if profile.risk_tolerance:
            parts.append(f"- Risk Tolerance: {profile.risk_tolerance.value}")
        if profile.investment_horizon:
            parts.append(f"- Investment Horizon: {profile.investment_horizon.value}")
        if profile.investment_experience:
            parts.append(f"- Experience Level: {profile.investment_experience.value}")
        if profile.liquidity_needs:
            parts.append(f"- Liquidity Needs: {profile.liquidity_needs.value}")
        if profile.goals:
            goals_str = ", ".join(g.goal_type.value for g in profile.goals)
            parts.append(f"- Goals: {goals_str}")
        if profile.preferred_sectors:
            parts.append(f"- Preferred Sectors: {', '.join(profile.preferred_sectors)}")
        if profile.excluded_sectors:
            parts.append(f"- Excluded Sectors: {', '.join(profile.excluded_sectors)}")
        if profile.esg_preference is not None:
            parts.append(f"- ESG Preference: {'Yes' if profile.esg_preference else 'No'}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # RAG context
    # ------------------------------------------------------------------

    def _build_rag_query(self, context: AggregatedContext) -> str:
        query_parts = ["investment recommendations"]
        if context.user_risk_tolerance:
            query_parts.append(f"{context.user_risk_tolerance} risk")
        if context.user_investment_horizon:
            query_parts.append(f"{context.user_investment_horizon} term")
        top_sectors = sorted(
            context.current_sector_allocation.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for sector, _ in top_sectors:
            query_parts.append(sector)
        return " ".join(query_parts)

    def _add_rag_context(
        self, query: str, context: AggregatedContext, sources: list[str]
    ) -> None:
        try:
            results = self._rag.search(query, top_k=5)
            context.rag_context = results.get("context", "")
            if results.get("results"):
                sources.append("rag_knowledge_base")
        except Exception as e:
            logger.error("RAG search failed: %s", e)

    # ------------------------------------------------------------------
    # Live market context via MCP -> MarketMCPServer -> yfinance
    # ------------------------------------------------------------------

    def _add_live_context(
        self,
        query: str,
        tickers: list[str],
        context: AggregatedContext,
        sources: list[str],
    ) -> None:
        """Fetch live quotes and news via HybridDataProvider (MCP layer)."""
        try:
            response = self._hybrid_provider.get_enriched_context_sync(
                query=query,
                tickers=tickers[:10],
                include_quotes=True,
                include_news=True,
                top_k=0,  # RAG already fetched above; skip duplicate search
            )
        except Exception as e:
            logger.error("HybridDataProvider failed: %s", e)
            return

        # Build live quotes block
        quote_lines: list[str] = []
        for quote in response.live_quotes:
            change_str = ""
            if quote.change_percent is not None:
                sign = "+" if quote.change_percent >= 0 else ""
                change_str = f" ({sign}{quote.change_percent:.2f}%)"
            quote_lines.append(f"- {quote.symbol}: ${quote.price:.2f}{change_str}")

        if quote_lines:
            context.live_market_context = "### Live Quotes\n" + "\n".join(quote_lines)
            sources.append("live_quotes")

        # Build news headlines block
        news_lines: list[str] = []
        for article in response.live_news[:10]:
            ticker_tag = article.tickers[0] if article.tickers else "Market"
            news_lines.append(f"- [{ticker_tag}] {article.title}")

        if news_lines:
            news_block = "\n\n### Recent News\n" + "\n".join(news_lines)
            context.live_market_context = (context.live_market_context or "") + news_block
            sources.append("live_news")
