"""Tests for text preprocessor."""

import pandas as pd

from src.data.preprocessor import TextPreprocessor


def test_clean_text_basic():
    """Test basic text cleaning."""
    preprocessor = TextPreprocessor(remove_extra_whitespace=True)
    
    text = "This  has   extra    spaces"
    cleaned = preprocessor.clean_text(text)
    
    assert cleaned == "This has extra spaces"


def test_clean_text_lowercase():
    """Test lowercasing."""
    preprocessor = TextPreprocessor(lowercase=True)
    
    text = "Hello WORLD"
    cleaned = preprocessor.clean_text(text)
    
    assert cleaned == "hello world"


def test_clean_text_max_length():
    """Test max length truncation."""
    preprocessor = TextPreprocessor(max_length=10)
    
    text = "This is a very long text that should be truncated"
    cleaned = preprocessor.clean_text(text)
    
    assert len(cleaned) == 10


def test_preprocess_corpus():
    """Test corpus preprocessing."""
    preprocessor = TextPreprocessor(remove_extra_whitespace=True)
    
    corpus_df = pd.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3"],
        "text": [
            "Normal text",
            "Text  with   spaces",
            "",  # Empty text
        ]
    })
    
    processed = preprocessor.preprocess_corpus(corpus_df)
    
    # Empty text should be removed
    assert len(processed) == 2
    assert "Text with spaces" in processed["text"].values
    assert "text_length" in processed.columns
    assert "word_count" in processed.columns


def test_preprocess_queries():
    """Test query preprocessing."""
    preprocessor = TextPreprocessor(remove_extra_whitespace=True)
    
    queries_df = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "text": [
            "search  query",
            "another query",
        ]
    })
    
    processed = preprocessor.preprocess_queries(queries_df)
    
    assert len(processed) == 2
    assert "search query" in processed["text"].values

