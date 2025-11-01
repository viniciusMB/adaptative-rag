"""Tests for FAISS vector store."""

import tempfile
from pathlib import Path

import numpy as np

from src.retrieval.vector_store import FAISSVectorStore


def test_vector_store_creation():
    """Test vector store creation."""
    vector_store = FAISSVectorStore(embedding_dim=128)
    
    assert vector_store.embedding_dim == 128
    assert vector_store.num_documents == 0


def test_add_and_search():
    """Test adding embeddings and searching."""
    embedding_dim = 128
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    
    # Create some random embeddings
    num_docs = 100
    embeddings = np.random.randn(num_docs, embedding_dim).astype(np.float32)
    
    # Normalize for IndexFlatIP (cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    doc_ids = [f"doc{i}" for i in range(num_docs)]
    
    # Add to index
    vector_store.add(embeddings, doc_ids)
    
    assert vector_store.num_documents == num_docs
    
    # Search with first embedding (should return itself as top result)
    query_embedding = embeddings[0:1]
    doc_ids_results, scores_results = vector_store.search(query_embedding, top_k=5)
    
    assert len(doc_ids_results) == 1
    assert len(doc_ids_results[0]) == 5
    assert doc_ids_results[0][0] == "doc0"  # First result should be itself
    assert scores_results[0][0] > 0.99  # Should be very high similarity


def test_save_and_load():
    """Test saving and loading vector store."""
    embedding_dim = 64
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    
    # Add some data
    num_docs = 50
    embeddings = np.random.randn(num_docs, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    doc_ids = [f"doc{i}" for i in range(num_docs)]
    vector_store.add(embeddings, doc_ids)
    
    # Save to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = str(Path(tmpdir) / "index.faiss")
        doc_ids_path = str(Path(tmpdir) / "doc_ids.pkl")
        
        vector_store.save(index_path, doc_ids_path)
        
        # Load
        loaded_store = FAISSVectorStore.load(index_path, doc_ids_path)
        
        assert loaded_store.num_documents == num_docs
        assert loaded_store.doc_ids == doc_ids
        
        # Search should work the same
        query_embedding = embeddings[10:11]
        orig_results = vector_store.search(query_embedding, top_k=5)
        loaded_results = loaded_store.search(query_embedding, top_k=5)
        
        assert orig_results[0][0] == loaded_results[0][0]  # Same doc_ids

