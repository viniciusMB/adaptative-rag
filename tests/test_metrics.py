"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    compute_average_precision,
    compute_dcg,
    compute_mrr,
    compute_ndcg,
    compute_recall,
)


def test_recall():
    """Test Recall@k computation."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc4", "doc6"}

    # Recall@3: 1/3 (only doc2 found)
    recall_3 = compute_recall(retrieved, relevant, k=3)
    assert recall_3 == pytest.approx(1/3)

    # Recall@5: 2/3 (doc2 and doc4 found)
    recall_5 = compute_recall(retrieved, relevant, k=5)
    assert recall_5 == pytest.approx(2/3)


def test_recall_no_relevant():
    """Test Recall@k with no relevant documents."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = set()

    recall = compute_recall(retrieved, relevant, k=3)
    assert recall == 0.0


def test_ndcg():
    """Test nDCG@k computation."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc2": 1.0, "doc3": 1.0, "doc4": 1.0}

    # Should return non-zero since we have relevant docs in top-3
    ndcg = compute_ndcg(retrieved, relevant, k=3)
    assert 0 < ndcg <= 1.0


def test_ndcg_perfect():
    """Test nDCG@k with perfect ranking."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1": 2.0, "doc2": 1.0, "doc3": 1.0}

    # Perfect ranking should give nDCG = 1.0
    ndcg = compute_ndcg(retrieved, relevant, k=3)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_no_relevant():
    """Test nDCG@k with no relevant documents."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {}

    ndcg = compute_ndcg(retrieved, relevant, k=3)
    assert ndcg == 0.0


def test_dcg():
    """Test DCG computation."""
    relevances = [3, 2, 1, 0, 0]
    
    # DCG@3 = 3 + 2/log2(3) + 1/log2(4)
    dcg = compute_dcg(relevances, k=3)
    assert dcg > 0


def test_mrr():
    """Test MRR computation."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc2", "doc4"}

    # First relevant doc is at rank 2
    mrr = compute_mrr(retrieved, relevant)
    assert mrr == pytest.approx(1/2)


def test_mrr_first_rank():
    """Test MRR with first document relevant."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1"}

    mrr = compute_mrr(retrieved, relevant)
    assert mrr == 1.0


def test_mrr_no_relevant():
    """Test MRR with no relevant documents found."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc4", "doc5"}

    mrr = compute_mrr(retrieved, relevant)
    assert mrr == 0.0


def test_average_precision():
    """Test Average Precision computation."""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}

    # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
    ap = compute_average_precision(retrieved, relevant)
    assert ap == pytest.approx(0.833, abs=0.01)


def test_average_precision_no_relevant():
    """Test AP with no relevant documents."""
    retrieved = ["doc1", "doc2"]
    relevant = set()

    ap = compute_average_precision(retrieved, relevant)
    assert ap == 0.0

