"""Data normalization utilities for cross-source data consistency."""

import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes data from different sources into consistent formats.

    Handles:
    - Price normalization
    - Percentage normalization
    - Volume normalization
    - Datetime normalization
    - Quote normalization
    - Article normalization
    """

    @staticmethod
    def normalize_price(value: str | float | Decimal | None) -> Decimal | None:
        """Normalize a price value to Decimal.

        Args:
            value: Price value in various formats.

        Returns:
            Normalized Decimal price or None.
        """
        if value is None:
            return None

        try:
            if isinstance(value, Decimal):
                return value

            # Handle string values
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = value.replace("$", "").replace(",", "").strip()
                if not cleaned or cleaned.lower() in ("n/a", "none", "null", "-"):
                    return None
                return Decimal(cleaned)

            return Decimal(str(value))

        except (InvalidOperation, ValueError) as e:
            logger.debug(f"Failed to normalize price '{value}': {e}")
            return None

    @staticmethod
    def normalize_percentage(value: str | float | None) -> float | None:
        """Normalize a percentage value.

        Args:
            value: Percentage value in various formats (e.g., "5.2%", 0.052, 5.2).

        Returns:
            Normalized float percentage (e.g., 5.2 for 5.2%) or None.
        """
        if value is None:
            return None

        try:
            if isinstance(value, str):
                cleaned = value.strip().rstrip("%")
                if not cleaned or cleaned.lower() in ("n/a", "none", "null", "-"):
                    return None
                return float(cleaned)

            # If value is less than 1, assume it's in decimal form (e.g., 0.052)
            if isinstance(value, float) and -1 < value < 1:
                return value * 100

            return float(value)

        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to normalize percentage '{value}': {e}")
            return None

    @staticmethod
    def normalize_volume(value: str | int | float | None) -> int | None:
        """Normalize a volume value.

        Args:
            value: Volume value in various formats (e.g., "1.5M", "1500000").

        Returns:
            Normalized integer volume or None.
        """
        if value is None:
            return None

        try:
            if isinstance(value, int):
                return value

            if isinstance(value, float):
                return int(value)

            if isinstance(value, str):
                cleaned = value.strip().replace(",", "").upper()
                if not cleaned or cleaned.lower() in ("n/a", "none", "null", "-"):
                    return None

                # Handle suffixes
                multiplier = 1
                if cleaned.endswith("K"):
                    multiplier = 1000
                    cleaned = cleaned[:-1]
                elif cleaned.endswith("M"):
                    multiplier = 1000000
                    cleaned = cleaned[:-1]
                elif cleaned.endswith("B"):
                    multiplier = 1000000000
                    cleaned = cleaned[:-1]

                return int(float(cleaned) * multiplier)

            return int(value)

        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to normalize volume '{value}': {e}")
            return None

    @staticmethod
    def normalize_datetime(
        value: str | datetime | None,
        default_tz: timezone = timezone.utc,
    ) -> datetime | None:
        """Normalize a datetime value.

        Args:
            value: Datetime value in various formats.
            default_tz: Default timezone if not specified.

        Returns:
            Normalized datetime or None.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=default_tz)
            return value

        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in ("n/a", "none", "null", "-"):
                return None

            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y%m%dT%H%M%S",
                "%Y%m%d",
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(cleaned[:len("2024-01-01T00:00:00.000000Z")], fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=default_tz)
                    return dt
                except ValueError:
                    continue

            # Try ISO format with timezone
            try:
                dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
                return dt
            except ValueError:
                pass

            logger.debug(f"Failed to parse datetime '{value}'")
            return None

        return None

    @staticmethod
    def normalize_quote(quote_data: dict) -> dict:
        """Normalize a stock quote from any source.

        Args:
            quote_data: Raw quote data dictionary.

        Returns:
            Normalized quote dictionary.
        """
        normalizer = DataNormalizer()

        # Common field mappings
        field_mappings = {
            "symbol": ["symbol", "ticker", "Symbol"],
            "price": ["price", "05. price", "current_price", "lastPrice", "close"],
            "open": ["open", "02. open", "openPrice"],
            "high": ["high", "03. high", "dayHigh", "highPrice"],
            "low": ["low", "04. low", "dayLow", "lowPrice"],
            "volume": ["volume", "06. volume", "totalVolume"],
            "previous_close": ["previous_close", "08. previous close", "previousClose"],
            "change": ["change", "09. change", "priceChange"],
            "change_percent": ["change_percent", "10. change percent", "changePercent", "percentChange"],
            "timestamp": ["timestamp", "07. latest trading day", "lastUpdated", "quoteTime"],
        }

        normalized = {}

        for norm_field, source_fields in field_mappings.items():
            value = None
            for source_field in source_fields:
                if source_field in quote_data and quote_data[source_field] is not None:
                    value = quote_data[source_field]
                    break

            if value is None:
                continue

            if norm_field in ("price", "open", "high", "low", "previous_close", "change"):
                normalized[norm_field] = normalizer.normalize_price(value)
            elif norm_field == "volume":
                normalized[norm_field] = normalizer.normalize_volume(value)
            elif norm_field == "change_percent":
                normalized[norm_field] = normalizer.normalize_percentage(value)
            elif norm_field == "timestamp":
                normalized[norm_field] = normalizer.normalize_datetime(value)
            else:
                normalized[norm_field] = value

        return normalized

    @staticmethod
    def normalize_article(article_data: dict) -> dict:
        """Normalize a news article from any source.

        Args:
            article_data: Raw article data dictionary.

        Returns:
            Normalized article dictionary.
        """
        normalizer = DataNormalizer()

        # Common field mappings
        field_mappings = {
            "title": ["title", "headline"],
            "summary": ["summary", "description", "snippet"],
            "source": ["source", "source_name"],
            "url": ["url", "link", "article_url"],
            "published_at": ["published_at", "publishedAt", "time_published", "pubDate"],
            "author": ["author", "byline"],
            "sentiment_score": ["sentiment_score", "overall_sentiment_score"],
            "sentiment_label": ["sentiment_label", "overall_sentiment_label"],
        }

        normalized = {}

        for norm_field, source_fields in field_mappings.items():
            value = None
            for source_field in source_fields:
                if source_field in article_data:
                    raw = article_data[source_field]
                    # Handle nested source object (e.g. News API: {"name": "Reuters"})
                    if isinstance(raw, dict) and "name" in raw:
                        value = raw["name"]
                    else:
                        value = raw
                    break

            if value is None:
                continue

            if norm_field == "published_at":
                normalized[norm_field] = normalizer.normalize_datetime(value)
            elif norm_field == "sentiment_score":
                # Sentiment scores are typically -1 to 1, not percentages
                try:
                    normalized[norm_field] = float(value) if value else None
                except (ValueError, TypeError):
                    normalized[norm_field] = None
            else:
                normalized[norm_field] = value

        # Handle tickers
        tickers = article_data.get("tickers", [])
        if not tickers and "ticker_sentiment" in article_data:
            tickers = [t.get("ticker") for t in article_data["ticker_sentiment"] if t.get("ticker")]
        normalized["tickers"] = tickers

        return normalized

    @staticmethod
    def format_currency(value: Decimal | float | None, symbol: str = "$") -> str:
        """Format a value as currency.

        Args:
            value: Numeric value to format.
            symbol: Currency symbol.

        Returns:
            Formatted currency string.
        """
        if value is None:
            return "N/A"

        try:
            amount = float(value)
            if abs(amount) >= 1_000_000_000:
                return f"{symbol}{amount / 1_000_000_000:.2f}B"
            elif abs(amount) >= 1_000_000:
                return f"{symbol}{amount / 1_000_000:.2f}M"
            elif abs(amount) >= 1_000:
                return f"{symbol}{amount / 1_000:.2f}K"
            else:
                return f"{symbol}{amount:,.2f}"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def format_percentage(value: float | None, include_sign: bool = True) -> str:
        """Format a value as percentage.

        Args:
            value: Percentage value.
            include_sign: Whether to include + sign for positive values.

        Returns:
            Formatted percentage string.
        """
        if value is None:
            return "N/A"

        try:
            if include_sign and value > 0:
                return f"+{value:.2f}%"
            return f"{value:.2f}%"
        except (ValueError, TypeError):
            return "N/A"
