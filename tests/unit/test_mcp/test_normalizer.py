"""Unit tests for data normalizer."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from backend.src.mcp.integration.data_normalizer import DataNormalizer


class TestNormalizePric:
    """Tests for price normalization."""

    def test_normalize_decimal(self):
        """Test normalizing a Decimal value."""
        result = DataNormalizer.normalize_price(Decimal("150.25"))
        assert result == Decimal("150.25")

    def test_normalize_float(self):
        """Test normalizing a float value."""
        result = DataNormalizer.normalize_price(150.25)
        assert result == Decimal("150.25")

    def test_normalize_string(self):
        """Test normalizing a string value."""
        result = DataNormalizer.normalize_price("150.25")
        assert result == Decimal("150.25")

    def test_normalize_string_with_currency(self):
        """Test normalizing a string with currency symbol."""
        result = DataNormalizer.normalize_price("$150.25")
        assert result == Decimal("150.25")

    def test_normalize_string_with_commas(self):
        """Test normalizing a string with commas."""
        result = DataNormalizer.normalize_price("1,500.25")
        assert result == Decimal("1500.25")

    def test_normalize_na_value(self):
        """Test normalizing N/A values."""
        assert DataNormalizer.normalize_price("N/A") is None
        assert DataNormalizer.normalize_price("none") is None
        assert DataNormalizer.normalize_price("-") is None

    def test_normalize_none(self):
        """Test normalizing None."""
        assert DataNormalizer.normalize_price(None) is None

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        assert DataNormalizer.normalize_price("") is None


class TestNormalizePercentage:
    """Tests for percentage normalization."""

    def test_normalize_float_percentage(self):
        """Test normalizing a float percentage."""
        result = DataNormalizer.normalize_percentage(5.2)
        assert result == 5.2

    def test_normalize_string_with_percent(self):
        """Test normalizing a string with percent sign."""
        result = DataNormalizer.normalize_percentage("5.2%")
        assert result == 5.2

    def test_normalize_decimal_form(self):
        """Test normalizing a decimal form percentage."""
        result = DataNormalizer.normalize_percentage(0.052)
        assert result == 5.2

    def test_normalize_negative_percentage(self):
        """Test normalizing a negative percentage."""
        result = DataNormalizer.normalize_percentage("-3.5%")
        assert result == -3.5

    def test_normalize_na_value(self):
        """Test normalizing N/A values."""
        assert DataNormalizer.normalize_percentage("N/A") is None


class TestNormalizeVolume:
    """Tests for volume normalization."""

    def test_normalize_integer(self):
        """Test normalizing an integer."""
        result = DataNormalizer.normalize_volume(1500000)
        assert result == 1500000

    def test_normalize_float(self):
        """Test normalizing a float."""
        result = DataNormalizer.normalize_volume(1500000.5)
        assert result == 1500000

    def test_normalize_string(self):
        """Test normalizing a string."""
        result = DataNormalizer.normalize_volume("1500000")
        assert result == 1500000

    def test_normalize_with_commas(self):
        """Test normalizing a string with commas."""
        result = DataNormalizer.normalize_volume("1,500,000")
        assert result == 1500000

    def test_normalize_with_k_suffix(self):
        """Test normalizing with K suffix."""
        result = DataNormalizer.normalize_volume("1.5K")
        assert result == 1500

    def test_normalize_with_m_suffix(self):
        """Test normalizing with M suffix."""
        result = DataNormalizer.normalize_volume("1.5M")
        assert result == 1500000

    def test_normalize_with_b_suffix(self):
        """Test normalizing with B suffix."""
        result = DataNormalizer.normalize_volume("1.5B")
        assert result == 1500000000


class TestNormalizeDatetime:
    """Tests for datetime normalization."""

    def test_normalize_datetime_object(self):
        """Test normalizing a datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = DataNormalizer.normalize_datetime(dt)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc

    def test_normalize_iso_format(self):
        """Test normalizing ISO format string."""
        result = DataNormalizer.normalize_datetime("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_normalize_date_only(self):
        """Test normalizing date-only string."""
        result = DataNormalizer.normalize_datetime("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_normalize_compact_format(self):
        """Test normalizing compact format."""
        result = DataNormalizer.normalize_datetime("20240115")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_normalize_none(self):
        """Test normalizing None."""
        assert DataNormalizer.normalize_datetime(None) is None


class TestNormalizeQuote:
    """Tests for quote normalization."""

    def test_normalize_alpha_vantage_quote(self):
        """Test normalizing an Alpha Vantage quote."""
        raw_quote = {
            "01. symbol": "AAPL",
            "05. price": "175.50",
            "02. open": "174.00",
            "03. high": "176.00",
            "04. low": "173.50",
            "06. volume": "50000000",
            "08. previous close": "174.25",
            "09. change": "1.25",
            "10. change percent": "0.72%",
        }

        result = DataNormalizer.normalize_quote(raw_quote)

        assert result["price"] == Decimal("175.50")
        assert result["open"] == Decimal("174.00")
        assert result["volume"] == 50000000
        assert result["change_percent"] == 0.72

    def test_normalize_generic_quote(self):
        """Test normalizing a generic quote format."""
        raw_quote = {
            "symbol": "AAPL",
            "price": 175.50,
            "volume": 50000000,
            "changePercent": 0.72,
        }

        result = DataNormalizer.normalize_quote(raw_quote)

        assert result["symbol"] == "AAPL"
        assert result["price"] == Decimal("175.5")


class TestNormalizeArticle:
    """Tests for article normalization."""

    def test_normalize_news_api_article(self):
        """Test normalizing a News API article."""
        raw_article = {
            "title": "Market Rally Continues",
            "description": "Stocks rose today...",
            "source": {"name": "Reuters"},
            "url": "https://example.com/article",
            "publishedAt": "2024-01-15T10:30:00Z",
            "author": "John Doe",
        }

        result = DataNormalizer.normalize_article(raw_article)

        assert result["title"] == "Market Rally Continues"
        assert result["summary"] == "Stocks rose today..."
        assert result["source"] == "Reuters"
        assert result["author"] == "John Doe"

    def test_normalize_alpha_vantage_article(self):
        """Test normalizing an Alpha Vantage article."""
        raw_article = {
            "title": "Tech Stocks Rise",
            "summary": "Technology sector gains...",
            "source": "Bloomberg",
            "url": "https://example.com",
            "time_published": "20240115T103000",
            "overall_sentiment_score": 0.5,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"},
            ],
        }

        result = DataNormalizer.normalize_article(raw_article)

        assert result["title"] == "Tech Stocks Rise"
        assert result["source"] == "Bloomberg"
        assert result["sentiment_score"] == 0.5
        assert result["tickers"] == ["AAPL", "MSFT"]


class TestFormatCurrency:
    """Tests for currency formatting."""

    def test_format_small_amount(self):
        """Test formatting a small amount."""
        result = DataNormalizer.format_currency(Decimal("150.25"))
        assert result == "$150.25"

    def test_format_thousands(self):
        """Test formatting thousands."""
        result = DataNormalizer.format_currency(Decimal("5000.00"))
        assert result == "$5.00K"

    def test_format_millions(self):
        """Test formatting millions."""
        result = DataNormalizer.format_currency(Decimal("5000000.00"))
        assert result == "$5.00M"

    def test_format_billions(self):
        """Test formatting billions."""
        result = DataNormalizer.format_currency(Decimal("5000000000.00"))
        assert result == "$5.00B"

    def test_format_none(self):
        """Test formatting None."""
        result = DataNormalizer.format_currency(None)
        assert result == "N/A"


class TestFormatPercentage:
    """Tests for percentage formatting."""

    def test_format_positive(self):
        """Test formatting a positive percentage."""
        result = DataNormalizer.format_percentage(5.25)
        assert result == "+5.25%"

    def test_format_negative(self):
        """Test formatting a negative percentage."""
        result = DataNormalizer.format_percentage(-3.50)
        assert result == "-3.50%"

    def test_format_without_sign(self):
        """Test formatting without sign."""
        result = DataNormalizer.format_percentage(5.25, include_sign=False)
        assert result == "5.25%"

    def test_format_none(self):
        """Test formatting None."""
        result = DataNormalizer.format_percentage(None)
        assert result == "N/A"
