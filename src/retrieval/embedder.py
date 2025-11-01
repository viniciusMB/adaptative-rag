"""Embedding generation using sentence-transformers."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize embedder.

        Args:
            model_name: Name of sentence-transformers model
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings to unit length
            cache_folder: Folder to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder

        logger.info(f"Loading embedding model: {model_name}")
        
        # Create cache folder if specified
        if cache_folder:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)

        # Load model
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
        )
        
        # Set max sequence length
        self.model.max_seq_length = max_length
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded. Embedding dimension: {self.embedding_dim}, "
            f"Device: {device}"
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to return numpy array

        Returns:
            Embeddings as numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings

    def encode_corpus(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a large corpus with progress tracking.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            Embeddings as numpy array
        """
        logger.info(f"Encoding {len(texts)} passages...")

        embeddings = self.encode(
            texts,
            show_progress=show_progress,
            convert_to_numpy=True,
        )

        logger.info(
            f"Encoded {len(texts)} passages. "
            f"Embedding shape: {embeddings.shape}"
        )

        return embeddings

    def encode_queries(
        self,
        queries: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode queries.

        Args:
            queries: List of query texts
            show_progress: Whether to show progress bar

        Returns:
            Query embeddings as numpy array
        """
        logger.info(f"Encoding {len(queries)} queries...")

        embeddings = self.encode(
            queries,
            show_progress=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: Embeddings array
            path: Path to save to
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            path: Path to load from

        Returns:
            Embeddings array
        """
        embeddings = np.load(path)
        logger.info(
            f"Loaded embeddings from {path}. Shape: {embeddings.shape}"
        )
        return embeddings

