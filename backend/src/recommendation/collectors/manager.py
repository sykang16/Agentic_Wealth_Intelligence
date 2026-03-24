"""Data collection manager that coordinates all collectors and RAG integration."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from ..rag.schemas import Document
from ..rag.initializer import RAGInitializer
from .base import BaseCollector, CollectorResult, CollectorStatus
from .alpha_vantage import AlphaVantageCollector
from .sec_edgar import SECEdgarCollector
from .news_api import NewsAPICollector

logger = logging.getLogger(__name__)


@dataclass
class CollectionSummary:
    """Summary of a data collection run."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_documents: int = 0
    total_indexed: int = 0
    collector_results: dict[str, CollectorResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if collection was successful."""
        return self.total_documents > 0


class DataCollectionManager:
    """Manages data collection from multiple sources and RAG indexing.

    This class coordinates:
    - Multiple data collectors (Alpha Vantage, SEC, News API)
    - Document deduplication
    - Incremental indexing to RAG system
    - Collection status and history
    """

    def __init__(self, rag_initializer: RAGInitializer | None = None):
        """Initialize the data collection manager.

        Args:
            rag_initializer: RAG system for indexing. If None, creates new one.
        """
        self.rag = rag_initializer or RAGInitializer()

        # Initialize collectors
        self.collectors: dict[str, BaseCollector] = {
            "alpha_vantage": AlphaVantageCollector(),
            "sec_edgar": SECEdgarCollector(),
            "news_api": NewsAPICollector(),
        }

        # Collection history
        self._collection_history: list[CollectionSummary] = []

    def get_collector_status(self) -> dict[str, dict]:
        """Get status of all collectors.

        Returns:
            Dictionary mapping collector names to their status.
        """
        return {name: collector.get_status() for name, collector in self.collectors.items()}

    def get_configured_collectors(self) -> list[str]:
        """Get list of properly configured collectors.

        Returns:
            List of collector names that are ready to use.
        """
        return [name for name, collector in self.collectors.items() if collector.is_configured()]

    def collect_all(
        self,
        index_immediately: bool = True,
        collectors: list[str] | None = None,
    ) -> CollectionSummary:
        """Collect data from all configured collectors.

        Args:
            index_immediately: If True, index documents to RAG immediately.
            collectors: Specific collectors to use. If None, uses all configured.

        Returns:
            CollectionSummary with results.
        """
        summary = CollectionSummary()
        all_documents: list[Document] = []

        # Determine which collectors to use
        if collectors:
            collector_names = [c for c in collectors if c in self.collectors]
        else:
            collector_names = self.get_configured_collectors()

        if not collector_names:
            summary.errors.append("No configured collectors available")
            return summary

        logger.info(f"Starting collection from: {collector_names}")

        # Collect from each source
        for name in collector_names:
            collector = self.collectors[name]

            try:
                logger.info(f"Collecting from {name}...")
                result = collector.collect()
                summary.collector_results[name] = result

                if result.is_success:
                    all_documents.extend(result.documents)
                    logger.info(f"Collected {result.documents_collected} documents from {name}")
                else:
                    logger.warning(f"Collection from {name} failed: {result.errors}")

            except Exception as e:
                error_msg = f"Error collecting from {name}: {e}"
                logger.error(error_msg)
                summary.errors.append(error_msg)

        summary.total_documents = len(all_documents)

        # Index documents if requested
        if index_immediately and all_documents:
            try:
                # Deduplicate by doc_id
                unique_docs = self._deduplicate_documents(all_documents)
                logger.info(f"Indexing {len(unique_docs)} unique documents...")

                num_indexed = self.add_documents(unique_docs)
                summary.total_indexed = num_indexed

                logger.info(f"Indexed {num_indexed} document chunks")

            except Exception as e:
                error_msg = f"Error indexing documents: {e}"
                logger.error(error_msg)
                summary.errors.append(error_msg)

        # Store in history
        self._collection_history.append(summary)

        return summary

    def collect_news(self, index_immediately: bool = True) -> CollectionSummary:
        """Collect news from all news sources.

        Args:
            index_immediately: If True, index documents immediately.

        Returns:
            CollectionSummary with results.
        """
        summary = CollectionSummary()
        all_documents: list[Document] = []

        news_collectors = ["alpha_vantage", "news_api"]

        for name in news_collectors:
            collector = self.collectors.get(name)
            if not collector or not collector.is_configured():
                continue

            try:
                if name == "alpha_vantage":
                    result = collector.collect_news_only()
                else:
                    result = collector.collect()

                summary.collector_results[name] = result
                if result.is_success:
                    all_documents.extend(result.documents)
                else:
                    summary.errors.extend(result.errors)

            except Exception as e:
                summary.errors.append(f"Error collecting news from {name}: {e}")

        summary.total_documents = len(all_documents)

        if index_immediately and all_documents:
            unique_docs = self._deduplicate_documents(all_documents)
            summary.total_indexed = self.add_documents(unique_docs)

        self._collection_history.append(summary)
        return summary

    def collect_sec_filings(
        self,
        tickers: list[str] | None = None,
        filing_types: list[str] | None = None,
        index_immediately: bool = True,
    ) -> CollectionSummary:
        """Collect SEC filings for specified companies.

        Args:
            tickers: Company ticker symbols.
            filing_types: Types of filings (10-K, 10-Q, 8-K).
            index_immediately: If True, index documents immediately.

        Returns:
            CollectionSummary with results.
        """
        summary = CollectionSummary()

        collector = self.collectors.get("sec_edgar")
        if not collector or not collector.is_configured():
            summary.errors.append("SEC EDGAR collector not configured")
            return summary

        try:
            result = collector.collect(tickers=tickers, filing_types=filing_types)
            summary.collector_results["sec_edgar"] = result
            summary.total_documents = result.documents_collected

            if result.is_success and index_immediately:
                unique_docs = self._deduplicate_documents(result.documents)
                summary.total_indexed = self.add_documents(unique_docs)
            elif not result.is_success:
                summary.errors.extend(result.errors)

        except Exception as e:
            summary.errors.append(f"Error collecting SEC filings: {e}")

        self._collection_history.append(summary)
        return summary

    def collect_etf_data(
        self,
        symbols: list[str] | None = None,
        index_immediately: bool = True,
    ) -> CollectionSummary:
        """Collect ETF data from Alpha Vantage.

        Args:
            symbols: ETF ticker symbols.
            index_immediately: If True, index documents immediately.

        Returns:
            CollectionSummary with results.
        """
        summary = CollectionSummary()

        collector = self.collectors.get("alpha_vantage")
        if not collector:
            summary.errors.append("Alpha Vantage collector not available")
            return summary

        try:
            result = collector.collect_etf_only(symbols=symbols)
            summary.collector_results["alpha_vantage"] = result
            summary.total_documents = result.documents_collected

            if result.is_success and index_immediately:
                unique_docs = self._deduplicate_documents(result.documents)
                summary.total_indexed = self.add_documents(unique_docs)
            elif not result.is_success:
                summary.errors.extend(result.errors)

        except Exception as e:
            summary.errors.append(f"Error collecting ETF data: {e}")

        self._collection_history.append(summary)
        return summary

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the RAG index incrementally.

        This adds new documents without clearing existing ones.

        Args:
            documents: Documents to add.

        Returns:
            Number of chunks indexed.
        """
        if not documents:
            return 0

        # Use the RAG retriever to add documents
        return self.rag.retriever.index_documents(documents)

    def _deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        """Remove duplicate documents based on doc_id.

        Args:
            documents: List of documents.

        Returns:
            Deduplicated list.
        """
        seen_ids = set()
        unique = []

        for doc in documents:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                unique.append(doc)

        return unique

    def get_collection_history(self, limit: int = 10) -> list[CollectionSummary]:
        """Get recent collection history.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of CollectionSummary objects.
        """
        return self._collection_history[-limit:]

    def get_rag_stats(self) -> dict:
        """Get current RAG system statistics.

        Returns:
            Dictionary with RAG stats.
        """
        return self.rag.get_stats()

    def cleanup_expired(self) -> int:
        """Remove expired documents from the RAG index.

        Returns:
            Number of chunks removed.
        """
        return self.rag.retriever.vector_store.delete_expired()

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search the RAG system.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            Search results dictionary.
        """
        return self.rag.search(query, top_k=top_k)
