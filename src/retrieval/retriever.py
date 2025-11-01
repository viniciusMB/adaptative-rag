"""Dense retriever combining embedder and vector store."""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever using embeddings and FAISS."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: int = 10,
    ):
        """
        Initialize dense retriever.

        Args:
            embedder: Embedder for encoding queries and documents
            vector_store: FAISS vector store (optional, can be built later)
            top_k: Default number of results to return
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.corpus_df: Optional[pd.DataFrame] = None

    def build_index(
        self,
        corpus_df: pd.DataFrame,
        doc_id_col: str = "doc_id",
        text_col: str = "text",
        show_progress: bool = True,
    ) -> None:
        """
        Build FAISS index from corpus.

        Args:
            corpus_df: DataFrame with documents
            doc_id_col: Column name for document IDs
            text_col: Column name for text
            show_progress: Whether to show progress bar
        """
        logger.info(f"Building index from {len(corpus_df)} documents")

        # Store corpus
        self.corpus_df = corpus_df

        # Encode passages
        texts = corpus_df[text_col].tolist()
        embeddings = self.embedder.encode_corpus(
            texts,
            show_progress=show_progress,
        )

        # Create vector store if not exists
        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedder.embedding_dim,
                index_type="IndexFlatIP",
            )

        # Add to index
        doc_ids = corpus_df[doc_id_col].astype(str).tolist()
        self.vector_store.add(embeddings, doc_ids)

        logger.info("Index building complete")

    def retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Retrieve documents for queries.

        Args:
            queries: List of query strings
            top_k: Number of results to return (uses default if None)

        Returns:
            Tuple of (doc_ids, scores) for each query
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_index first.")

        k = top_k if top_k is not None else self.top_k

        # Encode queries
        query_embeddings = self.embedder.encode_queries(queries)

        # Search
        doc_ids, scores = self.vector_store.search(query_embeddings, top_k=k)

        return doc_ids, scores

    def retrieve_single(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_texts: bool = False,
    ) -> Dict:
        """
        Retrieve documents for a single query.

        Args:
            query: Query string
            top_k: Number of results to return
            return_texts: Whether to include document texts in results

        Returns:
            Dictionary with doc_ids, scores, and optionally texts
        """
        doc_ids_list, scores_list = self.retrieve([query], top_k=top_k)
        
        doc_ids = doc_ids_list[0]
        scores = scores_list[0]

        result = {
            "query": query,
            "doc_ids": doc_ids,
            "scores": scores,
        }

        # Add texts if requested
        if return_texts and self.corpus_df is not None:
            texts = []
            for doc_id in doc_ids:
                doc_row = self.corpus_df[self.corpus_df["doc_id"] == doc_id]
                if len(doc_row) > 0:
                    texts.append(doc_row.iloc[0]["text"])
                else:
                    texts.append("")
            result["texts"] = texts

        return result

    def save(self, index_path: str, doc_ids_path: str) -> None:
        """
        Save the index to disk.

        Args:
            index_path: Path to save FAISS index
            doc_ids_path: Path to save doc_ids
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        self.vector_store.save(index_path, doc_ids_path)
        logger.info("Retriever saved successfully")

    def load(self, index_path: str, doc_ids_path: str) -> None:
        """
        Load index from disk.

        Args:
            index_path: Path to FAISS index
            doc_ids_path: Path to doc_ids
        """
        self.vector_store = FAISSVectorStore.load(
            index_path,
            doc_ids_path,
            index_type="IndexFlatIP",
        )
        logger.info("Retriever loaded successfully")

    @property
    def num_documents(self) -> int:
        """Get number of documents in the index."""
        if self.vector_store is None:
            return 0
        return self.vector_store.num_documents

