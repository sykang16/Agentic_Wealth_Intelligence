"""Unit tests for RAG schema models."""

from datetime import datetime, timedelta

import pytest

from backend.src.recommendation.rag.schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalResponse,
    RetrievalResult,
    SearchQuery,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_create_metadata(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            source="Vanguard",
            document_type=DocumentType.ETF_FACTSHEET,
            publish_date=datetime(2024, 1, 15),
        )
        assert metadata.source == "Vanguard"
        assert metadata.document_type == DocumentType.ETF_FACTSHEET
        assert metadata.tags == []

    def test_metadata_with_optional_fields(self):
        """Test metadata with all optional fields."""
        metadata = DocumentMetadata(
            source="Federal Reserve",
            document_type=DocumentType.FOMC_MINUTES,
            publish_date=datetime(2024, 1, 31),
            expiry_date=datetime(2024, 4, 30),
            tags=["fomc", "monetary policy"],
            author="FOMC",
            url="https://federalreserve.gov/fomc",
            ticker=None,
            sector=None,
        )
        assert metadata.expiry_date == datetime(2024, 4, 30)
        assert "fomc" in metadata.tags


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self):
        """Test creating a document."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.INVESTMENT_GUIDE,
            publish_date=datetime.now(),
        )
        doc = Document(
            doc_id="test-doc-1",
            title="Test Document",
            content="This is test content.",
            metadata=metadata,
        )
        assert doc.doc_id == "test-doc-1"
        assert doc.title == "Test Document"

    def test_document_not_expired(self):
        """Test is_expired for non-expired document."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.ETF_FACTSHEET,
            publish_date=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=30),
        )
        doc = Document(
            doc_id="test-1",
            title="Test",
            content="Content",
            metadata=metadata,
        )
        assert not doc.is_expired

    def test_document_expired(self):
        """Test is_expired for expired document."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.NEWS_ARTICLE,
            publish_date=datetime.now() - timedelta(days=60),
            expiry_date=datetime.now() - timedelta(days=1),
        )
        doc = Document(
            doc_id="test-2",
            title="Old News",
            content="Content",
            metadata=metadata,
        )
        assert doc.is_expired

    def test_document_no_expiry(self):
        """Test is_expired for document with no expiry."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.INVESTMENT_GUIDE,
            publish_date=datetime.now(),
            expiry_date=None,
        )
        doc = Document(
            doc_id="test-3",
            title="Guide",
            content="Content",
            metadata=metadata,
        )
        assert not doc.is_expired


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_create_chunk(self):
        """Test creating a document chunk."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.RESEARCH_REPORT,
            publish_date=datetime.now(),
        )
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            content="This is chunk content for testing.",
            chunk_index=0,
            start_char=0,
            end_char=35,
            metadata=metadata,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.chunk_index == 0

    def test_chunk_token_estimate(self):
        """Test token estimation."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.MARKET_ANALYSIS,
            publish_date=datetime.now(),
        )
        chunk = DocumentChunk(
            chunk_id="chunk-2",
            doc_id="doc-1",
            content="A" * 400,  # 400 chars
            chunk_index=1,
            start_char=0,
            end_char=400,
            metadata=metadata,
        )
        assert chunk.token_estimate == 100  # 400 / 4


class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_default_query(self):
        """Test query with default values."""
        query = SearchQuery(query_text="What is VOO?")
        assert query.top_k == 5
        assert query.min_score == 0.0
        assert query.include_expired is False

    def test_query_with_filters(self):
        """Test query with filters."""
        query = SearchQuery(
            query_text="ETF expense ratio",
            top_k=10,
            document_types=[DocumentType.ETF_FACTSHEET],
            tickers=["VOO", "VTI"],
        )
        assert query.top_k == 10
        assert DocumentType.ETF_FACTSHEET in query.document_types
        assert "VOO" in query.tickers


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_create_result(self):
        """Test creating a retrieval result."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.ETF_FACTSHEET,
            publish_date=datetime.now(),
        )
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
            metadata=metadata,
        )
        result = RetrievalResult(
            chunk=chunk,
            score=0.85,
            rank=1,
        )
        assert result.score == 0.85
        assert result.doc_id == "doc-1"


class TestRetrievalResponse:
    """Tests for RetrievalResponse model."""

    def test_empty_response(self):
        """Test response with no results."""
        query = SearchQuery(query_text="test")
        response = RetrievalResponse(
            query=query,
            results=[],
            total_found=0,
        )
        assert not response.has_results
        assert response.get_context_string() == ""

    def test_response_with_results(self):
        """Test response with results."""
        metadata = DocumentMetadata(
            source="Vanguard",
            document_type=DocumentType.ETF_FACTSHEET,
            publish_date=datetime.now(),
            ticker="VOO",
        )
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            content="VOO tracks the S&P 500",
            chunk_index=0,
            start_char=0,
            end_char=22,
            metadata=metadata,
        )
        result = RetrievalResult(chunk=chunk, score=0.9, rank=1)
        query = SearchQuery(query_text="VOO")
        response = RetrievalResponse(
            query=query,
            results=[result],
            total_found=1,
        )
        assert response.has_results
        context = response.get_context_string()
        assert "Vanguard" in context
        assert "VOO" in context
