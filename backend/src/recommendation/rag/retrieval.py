"""Advanced retrieval system with hybrid search and re-ranking."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .schemas import (
    DocumentChunk,
    DocumentType,
    RetrievalResponse,
    RetrievalResult,
    SearchQuery,
)
from .vector_store import ChromaVectorStore


class ReRankingStrategy(str, Enum):
    """Re-ranking strategies for search results."""

    NONE = "none"  # No re-ranking, use original scores
    RECENCY = "recency"  # Boost recent documents
    DIVERSITY = "diversity"  # Ensure diverse sources
    COMBINED = "combined"  # Combine recency and diversity


@dataclass
class ReRankingConfig:
    """Configuration for re-ranking."""

    strategy: ReRankingStrategy = ReRankingStrategy.COMBINED
    recency_weight: float = 0.2  # Weight for recency boost (0-1)
    diversity_weight: float = 0.1  # Weight for diversity penalty (0-1)
    recency_half_life_days: int = 30  # Half-life for recency decay
    min_score_threshold: float = 0.1  # Minimum score to include


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search with re-ranking."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        reranking_config: ReRankingConfig | None = None,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: ChromaDB vector store instance.
            reranking_config: Configuration for re-ranking.
        """
        self.vector_store = vector_store
        self.config = reranking_config or ReRankingConfig()

    def search(
        self,
        query: str | SearchQuery,
        top_k: int = 5,
        use_keyword_boost: bool = True,
        rerank: bool = True,
    ) -> RetrievalResponse:
        """Perform hybrid search with optional re-ranking.

        Args:
            query: Query string or SearchQuery object.
            top_k: Number of results to return.
            use_keyword_boost: Whether to boost results with keyword matches.
            rerank: Whether to apply re-ranking.

        Returns:
            RetrievalResponse with ranked results.
        """
        # Convert string to SearchQuery if needed
        if isinstance(query, str):
            search_query = SearchQuery(query_text=query, top_k=top_k * 2)  # Get more for re-ranking
        else:
            search_query = query
            search_query.top_k = search_query.top_k * 2  # Get more for re-ranking

        # Perform semantic search
        response = self.vector_store.search(search_query)

        if not response.results:
            return response

        results = response.results

        # Apply keyword boosting
        if use_keyword_boost:
            results = self._apply_keyword_boost(results, search_query.query_text)

        # Apply re-ranking
        if rerank and self.config.strategy != ReRankingStrategy.NONE:
            results = self._rerank(results)

        # Filter by minimum score and limit results
        results = [r for r in results if r.score >= self.config.min_score_threshold]
        results = results[:top_k]

        # Re-assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return RetrievalResponse(
            query=search_query,
            results=results,
            total_found=len(results),
            processing_time_ms=response.processing_time_ms,
        )

    def search_by_document_type(
        self,
        query: str,
        document_types: list[DocumentType],
        top_k: int = 5,
    ) -> RetrievalResponse:
        """Search within specific document types.

        Args:
            query: Query string.
            document_types: List of document types to search.
            top_k: Number of results to return.

        Returns:
            RetrievalResponse with filtered results.
        """
        search_query = SearchQuery(
            query_text=query,
            top_k=top_k,
            document_types=document_types,
        )
        return self.search(search_query, top_k=top_k)

    def search_by_ticker(
        self,
        query: str,
        tickers: list[str],
        top_k: int = 5,
    ) -> RetrievalResponse:
        """Search for documents related to specific tickers.

        Args:
            query: Query string.
            tickers: List of ticker symbols.
            top_k: Number of results to return.

        Returns:
            RetrievalResponse with filtered results.
        """
        search_query = SearchQuery(
            query_text=query,
            top_k=top_k,
            tickers=tickers,
        )
        return self.search(search_query, top_k=top_k)

    def _apply_keyword_boost(
        self,
        results: list[RetrievalResult],
        query_text: str,
    ) -> list[RetrievalResult]:
        """Boost scores for results containing query keywords.

        Args:
            results: List of retrieval results.
            query_text: Original query text.

        Returns:
            Results with boosted scores.
        """
        # Extract keywords (simple tokenization)
        keywords = self._extract_keywords(query_text)

        if not keywords:
            return results

        boosted_results = []
        for result in results:
            content_lower = result.chunk.content.lower()

            # Count keyword matches
            match_count = sum(1 for kw in keywords if kw in content_lower)
            keyword_ratio = match_count / len(keywords) if keywords else 0

            # Boost score based on keyword matches (up to 20% boost)
            boost_factor = 1.0 + (keyword_ratio * 0.2)
            new_score = min(1.0, result.score * boost_factor)

            boosted_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=new_score,
                    rank=result.rank,
                )
            )

        return boosted_results

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text.

        Args:
            text: Input text.

        Returns:
            List of lowercase keywords.
        """
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "as", "until", "while",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves",
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _rerank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Re-rank results based on configured strategy.

        Args:
            results: List of retrieval results.

        Returns:
            Re-ranked results.
        """
        if self.config.strategy == ReRankingStrategy.RECENCY:
            return self._rerank_by_recency(results)
        elif self.config.strategy == ReRankingStrategy.DIVERSITY:
            return self._rerank_by_diversity(results)
        elif self.config.strategy == ReRankingStrategy.COMBINED:
            return self._rerank_combined(results)
        return results

    def _rerank_by_recency(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Re-rank results boosting more recent documents.

        Args:
            results: List of retrieval results.

        Returns:
            Results re-ranked by recency.
        """
        now = datetime.now()
        reranked = []

        for result in results:
            publish_date = result.chunk.metadata.publish_date
            days_old = (now - publish_date).days

            # Exponential decay based on age
            recency_factor = 0.5 ** (days_old / self.config.recency_half_life_days)

            # Blend original score with recency
            new_score = (
                result.score * (1 - self.config.recency_weight) +
                recency_factor * self.config.recency_weight
            )

            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=min(1.0, new_score),
                    rank=result.rank,
                )
            )

        # Sort by new score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    def _rerank_by_diversity(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Re-rank results to ensure source diversity.

        Args:
            results: List of retrieval results.

        Returns:
            Results re-ranked for diversity.
        """
        # Track sources seen
        source_counts: dict[str, int] = defaultdict(int)
        reranked = []

        for result in results:
            source = result.chunk.metadata.source
            count = source_counts[source]

            # Penalize repeated sources
            diversity_penalty = self.config.diversity_weight * count
            new_score = max(0, result.score - diversity_penalty)

            source_counts[source] += 1

            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=new_score,
                    rank=result.rank,
                )
            )

        # Sort by new score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    def _rerank_combined(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Apply combined recency and diversity re-ranking.

        Args:
            results: List of retrieval results.

        Returns:
            Results with combined re-ranking.
        """
        now = datetime.now()
        source_counts: dict[str, int] = defaultdict(int)
        reranked = []

        for result in results:
            # Recency factor
            publish_date = result.chunk.metadata.publish_date
            days_old = (now - publish_date).days
            recency_factor = 0.5 ** (days_old / self.config.recency_half_life_days)

            # Diversity penalty
            source = result.chunk.metadata.source
            diversity_penalty = self.config.diversity_weight * source_counts[source]
            source_counts[source] += 1

            # Combined score
            new_score = (
                result.score * (1 - self.config.recency_weight) +
                recency_factor * self.config.recency_weight -
                diversity_penalty
            )
            new_score = max(0, min(1.0, new_score))

            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=new_score,
                    rank=result.rank,
                )
            )

        # Sort by new score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked


class RAGRetriever:
    """High-level RAG retriever combining all retrieval capabilities."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        """Initialize the RAG retriever.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            collection_name: Name of the collection.
            embedding_model: Sentence transformer model name.
        """
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
        self.hybrid_retriever = HybridRetriever(self.vector_store)

    def index_documents(self, documents: list) -> int:
        """Index documents into the vector store.

        Args:
            documents: List of Document instances to index.

        Returns:
            Number of chunks indexed.
        """
        return self.vector_store.add_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_types: list[DocumentType] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
    ) -> RetrievalResponse:
        """Search for relevant documents.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            document_types: Optional filter by document types.
            tickers: Optional filter by ticker symbols.
            sectors: Optional filter by sectors.

        Returns:
            RetrievalResponse with ranked results.
        """
        search_query = SearchQuery(
            query_text=query,
            top_k=top_k,
            document_types=document_types,
            tickers=tickers,
            sectors=sectors,
        )
        return self.hybrid_retriever.search(search_query, top_k=top_k)

    def get_context_for_query(
        self,
        query: str,
        max_chunks: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Get formatted context string for LLM prompts.

        Args:
            query: Natural language query.
            max_chunks: Maximum chunks to include.
            separator: Separator between chunks.

        Returns:
            Formatted context string.
        """
        response = self.search(query, top_k=max_chunks)
        return response.get_context_string(max_chunks=max_chunks, separator=separator)

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        self.vector_store.clear()

    def get_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dictionary with index statistics.
        """
        return {
            "total_chunks": self.vector_store.get_document_count(),
            "unique_documents": len(self.vector_store.get_unique_documents()),
        }
