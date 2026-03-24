"""Data collectors for real financial data sources."""

from .base import BaseCollector, CollectorResult
from .alpha_vantage import AlphaVantageCollector
from .sec_edgar import SECEdgarCollector
from .news_api import NewsAPICollector
from .manager import DataCollectionManager

__all__ = [
    "BaseCollector",
    "CollectorResult",
    "AlphaVantageCollector",
    "SECEdgarCollector",
    "NewsAPICollector",
    "DataCollectionManager",
]
