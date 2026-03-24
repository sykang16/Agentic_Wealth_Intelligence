"""Build-time script: pre-index RAG documents into ChromaDB.

Run during Render build so the vector store exists before the app starts.
At runtime, initialize_with_sample_documents() detects existing data and
skips indexing entirely — no sentence-transformers embedding at runtime.
"""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from backend.src.recommendation.rag.initializer import RAGInitializer

persist_dir = str(repo_root / "data" / "chroma")
print(f"[build_rag_index] Building RAG index at: {persist_dir}")

rag = RAGInitializer(persist_directory=persist_dir)
result = rag.initialize_with_sample_documents()

print(f"[build_rag_index] Done: {result}")
