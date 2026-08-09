"""Unit tests for RAG embeddings module."""

from datetime import datetime

import pytest

from backend.src.recommendation.rag.schemas import Document, DocumentMetadata, DocumentType
from backend.src.recommendation.rag.embeddings import (
    DocumentChunker,
    EmbeddingPipeline,
    SentenceTransformerEmbeddings,
)


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    @pytest.fixture
    def sample_document(self) -> Document:
        """Create a sample document for testing."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.INVESTMENT_GUIDE,
            publish_date=datetime.now(),
        )
        content = """# Investment Guide

## Introduction
This is an introduction to investing. It covers the basics of building a portfolio.

## Asset Allocation
Asset allocation is the process of dividing investments among different asset categories.
This helps manage risk and optimize returns based on your investment goals.

## Diversification
Diversification reduces portfolio volatility by spreading investments across various assets.
When one asset class underperforms, others may compensate.

## Conclusion
Regular review and rebalancing keeps your strategy on track.
"""
        return Document(
            doc_id="test-guide-1",
            title="Investment Guide",
            content=content,
            metadata=metadata,
        )

    def test_chunk_document_without_embeddings(self, sample_document):
        """Test chunking a document without generating embeddings."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(sample_document, embed=False)

        assert len(chunks) > 0
        assert all(c.embedding is None for c in chunks)
        assert all(c.doc_id == sample_document.doc_id for c in chunks)

    def test_chunk_indices_are_sequential(self, sample_document):
        """Test that chunk indices are sequential."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(sample_document, embed=False)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_content_covers_document(self, sample_document):
        """Test that chunks cover the document content."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk_document(sample_document, embed=False)

        # Each chunk should have content from the original document
        for chunk in chunks:
            assert chunk.content in sample_document.content or chunk.content.strip() in sample_document.content

    def test_chunk_id_is_deterministic(self, sample_document):
        """Test that chunk IDs are deterministic."""
        chunker = DocumentChunker(chunk_size=100)
        chunks1 = chunker.chunk_document(sample_document, embed=False)
        chunks2 = chunker.chunk_document(sample_document, embed=False)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id

    def test_multiple_documents(self, sample_document):
        """Test chunking multiple documents."""
        chunker = DocumentChunker(chunk_size=100)
        docs = [sample_document, sample_document]
        chunks = chunker.chunk_documents(docs, embed=False)

        assert len(chunks) > len(chunker.chunk_document(sample_document, embed=False))


class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformerEmbeddings class."""

    @pytest.fixture
    def embedding_model(self) -> SentenceTransformerEmbeddings:
        """Create an embedding model instance."""
        return SentenceTransformerEmbeddings()

    def test_encode_single(self, embedding_model):
        """Test encoding a single text."""
        embedding = embedding_model.encode_single("This is a test sentence.")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    def test_encode_multiple(self, embedding_model):
        """Test encoding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_model.encode(texts)

        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_embedding_dimension(self, embedding_model):
        """Test embedding dimension property."""
        dim = embedding_model.dimension
        assert dim > 0

        # Check that embeddings match the dimension
        embedding = embedding_model.encode_single("Test")
        assert len(embedding) == dim

    def test_similar_texts_have_similar_embeddings(self, embedding_model):
        """Test that similar texts have similar embeddings."""
        import numpy as np

        text1 = "The stock market went up today."
        text2 = "Equity markets rose today."
        text3 = "I like pizza for dinner."

        emb1 = embedding_model.encode_single(text1)
        emb2 = embedding_model.encode_single(text2)
        emb3 = embedding_model.encode_single(text3)

        # Calculate cosine similarities
        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_12 = cosine_sim(emb1, emb2)  # Similar texts
        sim_13 = cosine_sim(emb1, emb3)  # Dissimilar texts

        assert sim_12 > sim_13  # Similar texts should have higher similarity


class TestEmbeddingPipeline:
    """Tests for EmbeddingPipeline class."""

    @pytest.fixture
    def pipeline(self) -> EmbeddingPipeline:
        """Create an embedding pipeline."""
        return EmbeddingPipeline(chunk_size=100, chunk_overlap=10)

    @pytest.fixture
    def sample_document(self) -> Document:
        """Create a sample document."""
        metadata = DocumentMetadata(
            source="Test",
            document_type=DocumentType.RESEARCH_REPORT,
            publish_date=datetime.now(),
        )
        return Document(
            doc_id="test-doc",
            title="Test Document",
            content="This is a test document with enough content to be chunked. " * 10,
            metadata=metadata,
        )

    def test_process_document(self, pipeline, sample_document):
        """Test processing a document through the pipeline."""
        chunks = pipeline.process_document(sample_document)

        assert len(chunks) > 0
        assert all(c.embedding is not None for c in chunks)
        assert all(len(c.embedding) == pipeline.embedding_dimension for c in chunks)

    def test_embed_query(self, pipeline):
        """Test embedding a query."""
        query = "What is asset allocation?"
        embedding = pipeline.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == pipeline.embedding_dimension

    def test_embedding_dimension_property(self, pipeline):
        """Test that embedding dimension is accessible."""
        dim = pipeline.embedding_dimension
        assert dim > 0
        assert isinstance(dim, int)
