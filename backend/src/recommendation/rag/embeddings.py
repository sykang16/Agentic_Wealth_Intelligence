"""Embedding generation for RAG documents using sentence-transformers."""

import hashlib
import uuid
from typing import Protocol

from .schemas import Document, DocumentChunk, DocumentMetadata


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings."""
        ...


class SentenceTransformerEmbeddings:
    """Embedding generator using sentence-transformers library."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

    def __init__(self, model_name: str | None = None):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text to an embedding.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


class DocumentChunker:
    """Splits documents into chunks for embedding and retrieval."""

    DEFAULT_CHUNK_SIZE = 512  # Target tokens (approx 2048 chars)
    DEFAULT_CHUNK_OVERLAP = 50  # Overlap tokens (approx 200 chars)

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        """Initialize the document chunker.

        Args:
            chunk_size: Target chunk size in tokens (approx chars/4).
            chunk_overlap: Overlap between chunks in tokens.
            embedding_model: Optional embedding model to embed chunks.
        """
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        self.embedding_model = embedding_model

        # Convert token estimates to characters (rough approximation)
        self._char_chunk_size = self.chunk_size * 4
        self._char_overlap = self.chunk_overlap * 4

    def chunk_document(self, document: Document, embed: bool = True) -> list[DocumentChunk]:
        """Split a document into chunks.

        Args:
            document: Document to chunk.
            embed: Whether to generate embeddings for chunks.

        Returns:
            List of DocumentChunk instances.
        """
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            # Find end of chunk
            end = min(start + self._char_chunk_size, len(content))

            # Try to break at a natural boundary (newline, period, or space)
            if end < len(content):
                # Look for paragraph break first
                newline_pos = content.rfind("\n\n", start, end)
                if newline_pos > start + self._char_chunk_size // 2:
                    end = newline_pos + 2
                else:
                    # Look for sentence break
                    period_pos = content.rfind(". ", start, end)
                    if period_pos > start + self._char_chunk_size // 2:
                        end = period_pos + 2
                    else:
                        # Look for word break
                        space_pos = content.rfind(" ", start, end)
                        if space_pos > start + self._char_chunk_size // 2:
                            end = space_pos + 1

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunk_id = self._generate_chunk_id(document.doc_id, chunk_index, chunk_text)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - self._char_overlap
            if start >= len(content) - self._char_overlap:
                break

        # Generate embeddings if requested
        if embed and self.embedding_model and chunks:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        return chunks

    def chunk_documents(
        self, documents: list[Document], embed: bool = True
    ) -> list[DocumentChunk]:
        """Split multiple documents into chunks.

        Args:
            documents: List of documents to chunk.
            embed: Whether to generate embeddings for chunks.

        Returns:
            List of all DocumentChunk instances.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, embed=False)
            all_chunks.extend(chunks)

        # Batch embed all chunks for efficiency
        if embed and self.embedding_model and all_chunks:
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts)
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding

        return all_chunks

    def _generate_chunk_id(self, doc_id: str, chunk_index: int, content: str) -> str:
        """Generate a unique, deterministic chunk ID.

        Args:
            doc_id: Parent document ID.
            chunk_index: Index of chunk in document.
            content: Chunk content (used for hash).

        Returns:
            Unique chunk identifier.
        """
        # Create a hash of the content for deterministic IDs
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}-chunk-{chunk_index}-{content_hash}"


class EmbeddingPipeline:
    """End-to-end pipeline for document embedding."""

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize the embedding pipeline.

        Args:
            model_name: Sentence transformer model name.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
        """
        self.embedding_model = SentenceTransformerEmbeddings(model_name)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=self.embedding_model,
        )

    def process_document(self, document: Document) -> list[DocumentChunk]:
        """Process a single document through the pipeline.

        Args:
            document: Document to process.

        Returns:
            List of embedded chunks.
        """
        return self.chunker.chunk_document(document, embed=True)

    def process_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Process multiple documents through the pipeline.

        Args:
            documents: Documents to process.

        Returns:
            List of all embedded chunks.
        """
        return self.chunker.chunk_documents(documents, embed=True)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query.

        Args:
            query: Query text to embed.

        Returns:
            Query embedding vector.
        """
        return self.embedding_model.encode_single(query)

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_model.dimension
