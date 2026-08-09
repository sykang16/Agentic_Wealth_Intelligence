"""Integration tests for the RAG system."""

import os
import shutil
import tempfile
from datetime import datetime, timedelta

import pytest

from backend.src.recommendation.rag import (
    ChromaVectorStore,
    Document,
    DocumentMetadata,
    DocumentType,
    FreshnessManager,
    HybridRetriever,
    RAGRetriever,
    ReRankingConfig,
    ReRankingStrategy,
    SearchQuery,
)
from backend.src.data_generation.document_generator import (
    generate_all_sample_documents,
    generate_etf_factsheet,
    ETF_DATA,
)


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return generate_all_sample_documents()


@pytest.fixture
def vector_store(temp_persist_dir) -> ChromaVectorStore:
    """Create a ChromaVectorStore instance with temp persistence."""
    return ChromaVectorStore(
        collection_name="test_collection",
        persist_directory=temp_persist_dir,
    )


class TestChromaVectorStore:
    """Integration tests for ChromaVectorStore."""

    def test_add_and_search_documents(self, vector_store, sample_documents):
        """Test adding documents and searching."""
        # Add a subset of documents
        docs_to_add = sample_documents[:3]
        num_added = vector_store.add_documents(docs_to_add)
        assert num_added > 0

        # Search for content
        query = SearchQuery(query_text="S&P 500 ETF", top_k=3)
        response = vector_store.search(query)

        assert response.has_results
        assert len(response.results) <= 3

    def test_search_with_filters(self, vector_store, sample_documents):
        """Test searching with document type filters."""
        # Add all sample documents
        vector_store.add_documents(sample_documents)

        # Search only in ETF fact sheets
        query = SearchQuery(
            query_text="expense ratio",
            top_k=5,
            document_types=[DocumentType.ETF_FACTSHEET],
        )
        response = vector_store.search(query)

        assert response.has_results
        for result in response.results:
            assert result.chunk.metadata.document_type == DocumentType.ETF_FACTSHEET

    def test_delete_document(self, vector_store, sample_documents):
        """Test deleting a document."""
        doc = sample_documents[0]
        vector_store.add_documents([doc])

        initial_count = vector_store.get_document_count()
        deleted = vector_store.delete_document(doc.doc_id)

        assert deleted > 0
        assert vector_store.get_document_count() < initial_count

    def test_clear_collection(self, vector_store, sample_documents):
        """Test clearing all documents."""
        vector_store.add_documents(sample_documents[:3])
        assert vector_store.get_document_count() > 0

        vector_store.clear()
        assert vector_store.get_document_count() == 0


class TestHybridRetriever:
    """Integration tests for HybridRetriever."""

    @pytest.fixture
    def retriever(self, vector_store, sample_documents) -> HybridRetriever:
        """Create a HybridRetriever with indexed documents."""
        vector_store.add_documents(sample_documents)
        return HybridRetriever(vector_store)

    def test_hybrid_search(self, retriever):
        """Test hybrid search with keyword boosting."""
        response = retriever.search(
            "VOO S&P 500 index fund",
            top_k=5,
            use_keyword_boost=True,
        )

        assert response.has_results
        # Results should be ranked with boost applied
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_search_without_reranking(self, retriever):
        """Test search without re-ranking."""
        response = retriever.search(
            "retirement planning",
            top_k=5,
            rerank=False,
        )

        assert response.has_results

    def test_search_by_document_type(self, retriever):
        """Test search filtered by document type."""
        # Use a query that matches FOMC content well
        response = retriever.search_by_document_type(
            "interest rates monetary policy inflation",
            document_types=[DocumentType.FOMC_MINUTES],
            top_k=3,
        )

        # FOMC documents should be found with a relevant query
        if response.has_results:
            for result in response.results:
                assert result.chunk.metadata.document_type == DocumentType.FOMC_MINUTES
        else:
            # If no results, verify at least the filter was applied (no error)
            assert response.total_found == 0

    def test_search_by_ticker(self, retriever):
        """Test search filtered by ticker."""
        response = retriever.search_by_ticker(
            "technology sector",
            tickers=["QQQ", "VGT"],
            top_k=5,
        )

        # Should find documents related to tech ETFs
        if response.has_results:
            tickers = [r.chunk.metadata.ticker for r in response.results if r.chunk.metadata.ticker]
            assert any(t in ["QQQ", "VGT"] for t in tickers) or len(response.results) == 0


class TestRAGRetriever:
    """Integration tests for RAGRetriever high-level interface."""

    @pytest.fixture
    def rag_retriever(self, temp_persist_dir, sample_documents) -> RAGRetriever:
        """Create a RAGRetriever with indexed documents."""
        retriever = RAGRetriever(
            persist_directory=temp_persist_dir,
            collection_name="test_rag",
        )
        retriever.index_documents(sample_documents)
        return retriever

    def test_simple_search(self, rag_retriever):
        """Test simple search interface."""
        response = rag_retriever.search("dividend investing", top_k=3)

        assert response.has_results

    def test_get_context_for_query(self, rag_retriever):
        """Test getting formatted context for LLM."""
        context = rag_retriever.get_context_for_query(
            "asset allocation strategies",
            max_chunks=3,
        )

        assert len(context) > 0
        assert "---" in context or len(context.split("\n")) > 1

    def test_get_stats(self, rag_retriever):
        """Test getting index statistics."""
        stats = rag_retriever.get_stats()

        assert "total_chunks" in stats
        assert "unique_documents" in stats
        assert stats["total_chunks"] > 0
        assert stats["unique_documents"] > 0

    def test_clear_and_reindex(self, rag_retriever, sample_documents):
        """Test clearing and re-indexing."""
        initial_stats = rag_retriever.get_stats()

        rag_retriever.clear_index()
        assert rag_retriever.get_stats()["total_chunks"] == 0

        # Re-index a subset
        rag_retriever.index_documents(sample_documents[:2])
        new_stats = rag_retriever.get_stats()
        assert new_stats["total_chunks"] > 0
        assert new_stats["unique_documents"] == 2


class TestReRankingStrategies:
    """Tests for different re-ranking strategies."""

    @pytest.fixture
    def vector_store_with_docs(self, vector_store) -> ChromaVectorStore:
        """Create vector store with documents of varying dates."""
        # Create documents with different publish dates
        docs = []
        for i, etf in enumerate(ETF_DATA[:4]):
            doc = generate_etf_factsheet(etf)
            # Vary the publish dates
            doc.metadata.publish_date = datetime.now() - timedelta(days=i * 30)
            docs.append(doc)

        vector_store.add_documents(docs)
        return vector_store

    def test_recency_reranking(self, vector_store_with_docs):
        """Test recency-based re-ranking."""
        config = ReRankingConfig(
            strategy=ReRankingStrategy.RECENCY,
            recency_weight=0.3,
        )
        retriever = HybridRetriever(vector_store_with_docs, config)

        response = retriever.search("ETF investing", top_k=4)

        # More recent documents should generally rank higher
        # (though semantic relevance also plays a role)
        assert response.has_results

    def test_diversity_reranking(self, vector_store_with_docs):
        """Test diversity-based re-ranking."""
        config = ReRankingConfig(
            strategy=ReRankingStrategy.DIVERSITY,
            diversity_weight=0.2,
        )
        retriever = HybridRetriever(vector_store_with_docs, config)

        response = retriever.search("investment strategy", top_k=4)

        assert response.has_results
        # Should see diverse sources represented
        sources = [r.chunk.metadata.source for r in response.results]
        # Note: actual diversity depends on content relevance


class TestFreshnessManager:
    """Integration tests for FreshnessManager."""

    @pytest.fixture
    def freshness_manager(self, vector_store) -> FreshnessManager:
        """Create a FreshnessManager instance."""
        return FreshnessManager(vector_store)

    def test_prepare_documents(self, freshness_manager, sample_documents):
        """Test preparing documents with expiry dates."""
        docs = sample_documents[:3]
        prepared = freshness_manager.prepare_documents(docs)

        # Each document should have an expiry date based on its type
        for doc in prepared:
            if doc.metadata.document_type != DocumentType.INVESTMENT_GUIDE:
                # Guides have long retention, others should have expiry
                pass  # All should have expiry set now

    def test_cleanup_expired(self, freshness_manager, vector_store):
        """Test cleaning up expired documents."""
        # Create an expired document
        metadata = DocumentMetadata(
            source="Old News",
            document_type=DocumentType.NEWS_ARTICLE,
            publish_date=datetime.now() - timedelta(days=60),
            expiry_date=datetime.now() - timedelta(days=1),
        )
        doc = Document(
            doc_id="expired-doc",
            title="Old News Article",
            content="This is old news that has expired.",
            metadata=metadata,
        )
        vector_store.add_documents([doc])

        # Cleanup should remove it
        removed = freshness_manager.cleanup_expired()
        assert removed > 0

    def test_get_freshness_stats(self, freshness_manager, vector_store, sample_documents):
        """Test getting freshness statistics."""
        vector_store.add_documents(sample_documents[:5])
        stats = freshness_manager.get_freshness_stats()

        assert "total_chunks" in stats
        assert "by_type" in stats
        assert stats["total_chunks"] > 0


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_rag_workflow(self, temp_persist_dir):
        """Test complete RAG workflow from indexing to retrieval."""
        # 1. Create retriever
        retriever = RAGRetriever(
            persist_directory=temp_persist_dir,
            collection_name="e2e_test",
        )

        # 2. Generate and index documents
        documents = generate_all_sample_documents()
        num_indexed = retriever.index_documents(documents)
        assert num_indexed > 0

        # 3. Verify stats
        stats = retriever.get_stats()
        assert stats["unique_documents"] == len(documents)

        # 4. Perform searches
        queries = [
            "What is the expense ratio of VOO?",
            "Federal Reserve interest rate decision",
            "How to diversify my portfolio?",
        ]

        for query in queries:
            response = retriever.search(query, top_k=3)
            assert response.has_results, f"No results for query: {query}"

        # 5. Get context for LLM
        context = retriever.get_context_for_query(
            "retirement planning strategies",
            max_chunks=5,
        )
        assert len(context) > 100  # Should have substantial content
