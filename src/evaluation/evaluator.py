"""Evaluator for retrieval systems."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import (
    aggregate_metrics,
    compute_average_precision,
    compute_mrr,
    compute_ndcg,
    compute_recall,
)
from src.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for retrieval systems."""

    def __init__(
        self,
        retriever: DenseRetriever,
        k_values: List[int] = [1, 5, 10, 20],
    ):
        """
        Initialize evaluator.

        Args:
            retriever: Dense retriever to evaluate
            k_values: List of k values for Recall@k and nDCG@k
        """
        self.retriever = retriever
        self.k_values = sorted(k_values)

    def evaluate(
        self,
        queries_df: pd.DataFrame,
        qrels: Dict[str, Dict[str, float]],
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate retriever on queries.

        Args:
            queries_df: DataFrame with columns: query_id, text
            qrels: Dict mapping query_id -> {doc_id: relevance_score}
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"Evaluating on {len(queries_df)} queries")

        metrics_per_query = []

        # Process queries
        iterator = tqdm(queries_df.iterrows(), total=len(queries_df)) \
            if show_progress else queries_df.iterrows()

        for _, row in iterator:
            query_id = str(row["query_id"])
            query_text = row["text"]

            # Skip if no relevance judgments
            if query_id not in qrels or len(qrels[query_id]) == 0:
                continue

            # Retrieve documents
            max_k = max(self.k_values)
            doc_ids, scores = self.retriever.retrieve([query_text], top_k=max_k)
            retrieved_docs = doc_ids[0]

            # Get relevant docs
            relevant_docs_dict = qrels[query_id]
            relevant_docs_set = set(relevant_docs_dict.keys())

            # Compute metrics for this query
            query_metrics = {}

            # Recall@k
            for k in self.k_values:
                recall = compute_recall(retrieved_docs, relevant_docs_set, k)
                query_metrics[f"recall@{k}"] = recall

            # nDCG@k
            for k in self.k_values:
                # Convert binary relevance to float if needed
                relevant_docs_float = {
                    doc_id: float(rel)
                    for doc_id, rel in relevant_docs_dict.items()
                }
                ndcg = compute_ndcg(retrieved_docs, relevant_docs_float, k)
                query_metrics[f"ndcg@{k}"] = ndcg

            # MAP (Mean Average Precision)
            ap = compute_average_precision(retrieved_docs, relevant_docs_set)
            query_metrics["map"] = ap

            # MRR (Mean Reciprocal Rank)
            rr = compute_mrr(retrieved_docs, relevant_docs_set)
            query_metrics["mrr"] = rr

            metrics_per_query.append(query_metrics)

        # Aggregate metrics
        aggregated_metrics = aggregate_metrics(metrics_per_query)

        logger.info("Evaluation complete")
        self._log_metrics(aggregated_metrics)

        return aggregated_metrics

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics in a readable format."""
        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info("=" * 50)

        # Group by metric type
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith("recall")}
        ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith("ndcg")}
        other_metrics = {
            k: v for k, v in metrics.items()
            if not k.startswith("recall") and not k.startswith("ndcg")
        }

        if recall_metrics:
            logger.info("\nRecall:")
            for k, v in sorted(recall_metrics.items()):
                logger.info(f"  {k}: {v:.4f}")

        if ndcg_metrics:
            logger.info("\nnDCG:")
            for k, v in sorted(ndcg_metrics.items()):
                logger.info(f"  {k}: {v:.4f}")

        if other_metrics:
            logger.info("\nOther:")
            for k, v in sorted(other_metrics.items()):
                logger.info(f"  {k}: {v:.4f}")

        logger.info("=" * 50)

    def save_results(
        self,
        metrics: Dict[str, float],
        output_path: str,
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            metrics: Metrics dictionary
            output_path: Path to save results
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    def generate_report(
        self,
        metrics: Dict[str, float],
    ) -> str:
        """
        Generate a formatted report.

        Args:
            metrics: Metrics dictionary

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "RETRIEVAL EVALUATION REPORT",
            "=" * 60,
            "",
            f"Number of documents: {self.retriever.num_documents}",
            "",
        ]

        # Recall metrics
        report_lines.append("Recall@k:")
        for k in self.k_values:
            key = f"recall@{k}"
            if key in metrics:
                report_lines.append(f"  Recall@{k:2d}: {metrics[key]:.4f}")

        report_lines.append("")

        # nDCG metrics
        report_lines.append("nDCG@k:")
        for k in self.k_values:
            key = f"ndcg@{k}"
            if key in metrics:
                report_lines.append(f"  nDCG@{k:2d}: {metrics[key]:.4f}")

        report_lines.append("")

        # Other metrics
        if "map" in metrics:
            report_lines.append(f"MAP: {metrics['map']:.4f}")
        if "mrr" in metrics:
            report_lines.append(f"MRR: {metrics['mrr']:.4f}")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

