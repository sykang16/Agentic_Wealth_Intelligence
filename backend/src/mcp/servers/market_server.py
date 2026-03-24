"""Market data MCP server backed by yfinance (no API key required)."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

import yfinance as yf

from ..schemas import (
    ETFProfile,
    MCPDataSource,
    StockQuote,
)
from .base import BaseMCPServer

logger = logging.getLogger(__name__)


class MarketMCPServer(BaseMCPServer):
    """MCP server for real-time market data via yfinance.

    Provides tools for:
    - Getting real-time stock/ETF quotes
    - Getting ETF profiles and metrics
    - Getting recent news headlines per ticker
    - Batch quote retrieval

    Resources:
    - market://symbols/popular - List of popular tickers
    """

    name = "market"
    description = "Real-time market data and quotes via yfinance"

    POPULAR_SYMBOLS = [
        "SPY", "QQQ", "IWM", "DIA",           # Major ETFs
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Tech giants
        "VTI", "VOO", "VEA", "VWO",            # Vanguard ETFs
        "BND", "AGG", "TLT",                   # Bond ETFs
        "GLD", "SLV",                          # Commodities
    ]

    def __init__(self):
        # Quote cache (5 min TTL)
        self._cache: dict[str, tuple[datetime, object]] = {}
        self._cache_ttl = 300  # seconds

        # Public references to tool functions (set during _register_tools)
        self.get_stock_quote = None
        self.get_etf_profile = None
        self.get_multiple_quotes = None
        self.get_ticker_news = None

        super().__init__(data_source=MCPDataSource.YFINANCE)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return (datetime.now() - cached_time).total_seconds() < self._cache_ttl

    def _get_cached(self, key: str) -> object | None:
        if self._is_cache_valid(key):
            return self._cache[key][1]
        return None

    def _set_cached(self, key: str, value: object) -> None:
        self._cache[key] = (datetime.now(), value)

    # ------------------------------------------------------------------
    # Synchronous yfinance helpers (called via asyncio.to_thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_get_quote(symbol: str) -> dict:
        """Fetch current price/change from yfinance."""
        hist = yf.Ticker(symbol).history(period="2d")
        if hist.empty:
            raise ValueError(f"No price data returned for {symbol}")

        price = float(hist["Close"].iloc[-1])
        previous_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        change = price - previous_close if previous_close is not None else None
        change_percent = (change / previous_close * 100) if previous_close else None
        volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None

        return {
            "symbol": symbol,
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "volume": volume,
        }

    @staticmethod
    def _sync_get_etf_info(symbol: str) -> dict:
        """Fetch ETF/stock overview from yfinance."""
        info = yf.Ticker(symbol).info
        if not info or (not info.get("longName") and not info.get("shortName")):
            raise ValueError(f"No info data returned for {symbol}")

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "description": info.get("longBusinessSummary"),
            "asset_type": info.get("quoteType", "ETF"),
            "exchange": info.get("exchange"),
            "sector": info.get("sector") or info.get("category"),
            "dividend_yield": info.get("dividendYield"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
        }

    @staticmethod
    def _sync_get_news(symbol: str, limit: int = 5) -> list[dict]:
        """Fetch recent news headlines from yfinance."""
        news = yf.Ticker(symbol).news or []
        articles = []
        for item in news[:limit]:
            title = item.get("title", "")
            if title:
                articles.append({
                    "title": title,
                    "url": item.get("link") or item.get("url", ""),
                    "source": item.get("publisher", "Unknown"),
                    "published_at": item.get("providerPublishTime"),
                    "tickers": [symbol],
                })
        return articles

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        """Register MCP tools for market data."""

        @self._mcp.tool()
        async def get_stock_quote(symbol: str) -> dict:
            """Get real-time stock/ETF quote.

            Args:
                symbol: Ticker symbol (e.g., "AAPL", "SPY").

            Returns:
                StockQuote data with current price and daily change.
            """
            self.log_tool_call("get_stock_quote", symbol=symbol)
            symbol = symbol.upper()
            cache_key = f"quote:{symbol}"

            cached = self._get_cached(cache_key)
            if cached:
                return {"cached": True, **cached}

            try:
                data = await asyncio.to_thread(self._sync_get_quote, symbol)
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Failed to fetch quote for {symbol}: {e}")

            quote = StockQuote(
                symbol=symbol,
                price=Decimal(str(round(data["price"], 4))),
                previous_close=Decimal(str(round(data["previous_close"], 4))) if data.get("previous_close") is not None else None,
                change=Decimal(str(round(data["change"], 4))) if data.get("change") is not None else None,
                change_percent=data.get("change_percent"),
                volume=data.get("volume"),
            )

            quote_dict = quote.model_dump(mode="json")
            self._set_cached(cache_key, quote_dict)
            return quote_dict

        @self._mcp.tool()
        async def get_etf_profile(symbol: str) -> dict:
            """Get ETF profile and key financial metrics.

            Args:
                symbol: ETF ticker symbol (e.g., "SPY", "VOO").

            Returns:
                ETFProfile data with fund details and metrics.
            """
            self.log_tool_call("get_etf_profile", symbol=symbol)
            symbol = symbol.upper()
            cache_key = f"etf:{symbol}"

            # ETF profiles have a longer TTL (30 min)
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 1800:
                    return {"cached": True, **cached_data}

            try:
                data = await asyncio.to_thread(self._sync_get_etf_info, symbol)
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Failed to fetch ETF profile for {symbol}: {e}")

            profile = ETFProfile(
                symbol=symbol,
                name=data["name"],
                description=data.get("description"),
                asset_type=data.get("asset_type", "ETF"),
                exchange=data.get("exchange"),
                sector=data.get("sector"),
                dividend_yield=data.get("dividend_yield"),
                pe_ratio=data.get("pe_ratio"),
                market_cap=Decimal(str(int(data["market_cap"]))) if data.get("market_cap") else None,
                week_52_high=Decimal(str(round(data["week_52_high"], 4))) if data.get("week_52_high") else None,
                week_52_low=Decimal(str(round(data["week_52_low"], 4))) if data.get("week_52_low") else None,
            )

            profile_dict = profile.model_dump(mode="json")
            self._cache[cache_key] = (datetime.now(), profile_dict)
            return profile_dict

        @self._mcp.tool()
        async def get_ticker_news(symbol: str, limit: int = 3) -> dict:
            """Get recent news headlines for a ticker via yfinance.

            Args:
                symbol: Ticker symbol (e.g., "AAPL").
                limit: Maximum number of headlines to return (default 3).

            Returns:
                Dictionary with 'articles' list and 'symbol'.
            """
            self.log_tool_call("get_ticker_news", symbol=symbol, limit=limit)
            symbol = symbol.upper()
            cache_key = f"news:{symbol}"

            cached = self._get_cached(cache_key)
            if cached:
                return {"cached": True, **cached}

            try:
                articles = await asyncio.to_thread(self._sync_get_news, symbol, limit)
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Failed to fetch news for {symbol}: {e}")

            result = {"symbol": symbol, "articles": articles}
            self._set_cached(cache_key, result)
            return result

        @self._mcp.tool()
        async def get_multiple_quotes(symbols: list[str]) -> dict:
            """Get quotes for multiple symbols.

            Args:
                symbols: List of ticker symbols (max 10).

            Returns:
                Dictionary with 'quotes' mapping symbols to quote data and 'errors'.
            """
            self.log_tool_call("get_multiple_quotes", symbols=symbols)

            if len(symbols) > 10:
                from fastmcp.exceptions import ToolError
                raise ToolError("Maximum 10 symbols allowed per request")

            results = {}
            errors = []

            for symbol in symbols:
                symbol = symbol.upper()
                cache_key = f"quote:{symbol}"

                cached = self._get_cached(cache_key)
                if cached:
                    results[symbol] = {"cached": True, **cached}
                    continue

                try:
                    data = await asyncio.to_thread(self._sync_get_quote, symbol)
                    quote = StockQuote(
                        symbol=symbol,
                        price=Decimal(str(round(data["price"], 4))),
                        previous_close=Decimal(str(round(data["previous_close"], 4))) if data.get("previous_close") is not None else None,
                        change=Decimal(str(round(data["change"], 4))) if data.get("change") is not None else None,
                        change_percent=data.get("change_percent"),
                        volume=data.get("volume"),
                    )
                    quote_dict = quote.model_dump(mode="json")
                    self._set_cached(cache_key, quote_dict)
                    results[symbol] = quote_dict
                except Exception as e:
                    errors.append(f"{symbol}: {e}")

            return {
                "quotes": results,
                "errors": errors,
                "count": len(results),
            }

        # Save references for direct invocation
        self.get_stock_quote = get_stock_quote.fn
        self.get_etf_profile = get_etf_profile.fn
        self.get_ticker_news = get_ticker_news.fn
        self.get_multiple_quotes = get_multiple_quotes.fn

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self._mcp.resource("market://symbols/popular")
        def list_popular_symbols() -> str:
            """List popular stock and ETF symbols."""
            import json
            symbols = {
                "major_etfs": ["SPY", "QQQ", "IWM", "DIA"],
                "tech_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
                "vanguard_etfs": ["VTI", "VOO", "VEA", "VWO", "VIG"],
                "bond_etfs": ["BND", "AGG", "TLT", "LQD"],
                "commodities": ["GLD", "SLV", "USO"],
                "all": self.POPULAR_SYMBOLS,
            }
            return json.dumps(symbols, indent=2)

        self.list_popular_symbols = list_popular_symbols.fn


def main():
    """Run the market MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = MarketMCPServer()
    server.run()


if __name__ == "__main__":
    main()
