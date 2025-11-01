"""Dense retrieval components."""

from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever
from src.retrieval.vector_store import FAISSVectorStore

__all__ = ["Embedder", "DenseRetriever", "FAISSVectorStore"]

