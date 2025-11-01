"""FAISS-based vector store for dense retrieval."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS vector store for efficient similarity search."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        metric_type: str = "METRIC_INNER_PRODUCT",
    ):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index (IndexFlatIP, IndexFlatL2, etc.)
            metric_type: Distance metric (METRIC_INNER_PRODUCT, METRIC_L2)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric_type = metric_type
        self.doc_ids: List[str] = []
        
        # Create index
        self.index = self._create_index()
        logger.info(
            f"Initialized FAISS index: {index_type}, "
            f"dimension: {embedding_dim}"
        )

    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on type.

        Returns:
            FAISS index
        """
        if self.index_type == "IndexFlatIP":
            # Inner product (for normalized embeddings = cosine similarity)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return index

    def add(
        self,
        embeddings: np.ndarray,
        doc_ids: List[str],
    ) -> None:
        """
        Add embeddings to the index.

        Args:
            embeddings: Embeddings array of shape (n_docs, embedding_dim)
            doc_ids: List of document IDs corresponding to embeddings
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of doc_ids ({len(doc_ids)})"
            )

        # Ensure embeddings are float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Add to index
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)

        logger.info(
            f"Added {len(doc_ids)} documents to index. "
            f"Total documents: {len(self.doc_ids)}"
        )

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for similar documents.

        Args:
            query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
            top_k: Number of top results to return

        Returns:
            Tuple of (doc_ids, scores) where each is a list of lists
        """
        # Ensure embeddings are float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        # Handle single query
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embeddings, top_k)

        # Convert indices to doc_ids
        doc_ids_results = []
        scores_results = []

        for query_indices, query_scores in zip(indices, scores):
            query_doc_ids = []
            query_doc_scores = []
            
            for idx, score in zip(query_indices, query_scores):
                if idx != -1 and idx < len(self.doc_ids):  # -1 means no result
                    query_doc_ids.append(self.doc_ids[idx])
                    query_doc_scores.append(float(score))
            
            doc_ids_results.append(query_doc_ids)
            scores_results.append(query_doc_scores)

        return doc_ids_results, scores_results

    def save(self, index_path: str, doc_ids_path: str) -> None:
        """
        Save index and doc_ids to disk.

        Args:
            index_path: Path to save FAISS index
            doc_ids_path: Path to save doc_ids
        """
        # Create directories
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(doc_ids_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

        # Save doc_ids
        with open(doc_ids_path, "wb") as f:
            pickle.dump(self.doc_ids, f)
        logger.info(f"Saved doc_ids to {doc_ids_path}")

    @classmethod
    def load(
        cls,
        index_path: str,
        doc_ids_path: str,
        index_type: str = "IndexFlatIP",
    ) -> "FAISSVectorStore":
        """
        Load index and doc_ids from disk.

        Args:
            index_path: Path to FAISS index
            doc_ids_path: Path to doc_ids
            index_type: Type of index (for metadata)

        Returns:
            Loaded FAISSVectorStore instance
        """
        # Load index
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")

        # Load doc_ids
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)
        logger.info(f"Loaded {len(doc_ids)} doc_ids from {doc_ids_path}")

        # Create instance
        embedding_dim = index.d
        vector_store = cls(
            embedding_dim=embedding_dim,
            index_type=index_type,
        )
        vector_store.index = index
        vector_store.doc_ids = doc_ids

        return vector_store

    @property
    def num_documents(self) -> int:
        """Get number of documents in the index."""
        return len(self.doc_ids)

