"""Evaluation metrics for retrieval systems."""

import logging
from typing import Dict, List, Set

import numpy as np

logger = logging.getLogger(__name__)


def compute_recall(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Set of relevant document IDs
        k: Cutoff position

    Returns:
        Recall@k score (fraction of relevant docs found in top-k)
    """
    if len(relevant_docs) == 0:
        return 0.0

    # Take top-k retrieved docs
    top_k_docs = set(retrieved_docs[:k])

    # Count how many relevant docs are in top-k
    num_relevant_found = len(top_k_docs.intersection(relevant_docs))

    # Recall = (relevant found) / (total relevant)
    recall = num_relevant_found / len(relevant_docs)

    return recall


def compute_precision(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """
    Compute Precision@k.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: Set of relevant document IDs
        k: Cutoff position

    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0

    # Take top-k retrieved docs
    top_k_docs = set(retrieved_docs[:k])

    # Count how many are relevant
    num_relevant_found = len(top_k_docs.intersection(relevant_docs))

    # Precision = (relevant found) / k
    precision = num_relevant_found / k

    return precision


def compute_average_precision(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
) -> float:
    """
    Compute Average Precision (AP).

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: Set of relevant document IDs

    Returns:
        Average Precision score
    """
    if len(relevant_docs) == 0:
        return 0.0

    num_relevant_found = 0
    precision_sum = 0.0

    for k, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / k
            precision_sum += precision_at_k

    if num_relevant_found == 0:
        return 0.0

    avg_precision = precision_sum / len(relevant_docs)
    return avg_precision


def compute_dcg(relevances: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    Args:
        relevances: List of relevance scores (ordered by rank)
        k: Cutoff position

    Returns:
        DCG@k score
    """
    relevances = relevances[:k]
    
    if len(relevances) == 0:
        return 0.0

    # DCG = rel_1 + sum(rel_i / log2(i+1)) for i=2..k
    dcg = relevances[0]
    for i, rel in enumerate(relevances[1:], start=2):
        dcg += rel / np.log2(i + 1)

    return dcg


def compute_ndcg(
    retrieved_docs: List[str],
    relevant_docs: Dict[str, float],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by rank)
        relevant_docs: Dict mapping doc_id -> relevance score
        k: Cutoff position

    Returns:
        nDCG@k score
    """
    if len(relevant_docs) == 0:
        return 0.0

    # Get relevance scores for retrieved docs
    retrieved_relevances = []
    for doc_id in retrieved_docs[:k]:
        rel = relevant_docs.get(doc_id, 0.0)
        retrieved_relevances.append(rel)

    # Compute DCG
    dcg = compute_dcg(retrieved_relevances, k)

    # Compute ideal DCG (IDCG)
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)
    idcg = compute_dcg(ideal_relevances, k)

    # Compute nDCG
    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def compute_mrr(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (for a single query).

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: Set of relevant document IDs

    Returns:
        Reciprocal rank (1/rank of first relevant doc)
    """
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank

    return 0.0


def aggregate_metrics(
    metrics_per_query: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across queries.

    Args:
        metrics_per_query: List of metric dicts for each query

    Returns:
        Dictionary with mean metrics
    """
    if len(metrics_per_query) == 0:
        return {}

    # Get all metric names
    metric_names = set()
    for m in metrics_per_query:
        metric_names.update(m.keys())

    # Compute means
    aggregated = {}
    for metric_name in metric_names:
        values = [m.get(metric_name, 0.0) for m in metrics_per_query]
        aggregated[metric_name] = float(np.mean(values))

    return aggregated

