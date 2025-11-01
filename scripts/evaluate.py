"""Script to evaluate retrieval system."""

import logging
import sys
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import MSMARCOLoader
from src.data.preprocessor import TextPreprocessor
from src.evaluation.evaluator import Evaluator
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate retrieval system."""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )

    logger.info("=" * 60)
    logger.info("Evaluating Retrieval System")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Load queries and qrels
    logger.info("\n[1/4] Loading queries and qrels...")
    loader = MSMARCOLoader(
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        cache_dir=cfg.paths.cache_dir,
    )
    
    queries_df = loader.load_queries(split=cfg.data.queries_split)
    qrels = loader.load_qrels(split=cfg.data.queries_split)
    
    # Sample for testing if needed
    # logger.warning("SAMPLING QUERIES FOR TESTING")
    # queries_df = queries_df.head(100)

    # 2. Preprocess queries
    logger.info("\n[2/4] Preprocessing queries...")
    preprocessor = TextPreprocessor(
        max_length=cfg.data.max_passage_length,
        remove_extra_whitespace=cfg.data.clean_text,
    )
    queries_df = preprocessor.preprocess_queries(queries_df)

    # 3. Load retriever
    logger.info("\n[3/4] Loading retriever...")
    
    # Initialize embedder
    embedder = Embedder(
        model_name=cfg.model.name,
        device=cfg.model.device,
        batch_size=cfg.model.batch_size,
        max_length=cfg.model.max_length,
        normalize_embeddings=cfg.model.normalize_embeddings,
        cache_folder=cfg.model.cache_folder,
    )

    # Load retriever
    retriever = DenseRetriever(
        embedder=embedder,
        top_k=cfg.retrieval.top_k,
    )
    
    retriever.load(
        index_path=cfg.retrieval.index_path,
        doc_ids_path=cfg.retrieval.doc_ids_path,
    )

    # Load corpus for retriever
    corpus_path = Path(cfg.paths.output_dir) / "corpus.parquet"
    if corpus_path.exists():
        retriever.corpus_df = pd.read_parquet(corpus_path)
        logger.info(f"Loaded corpus from {corpus_path}")

    # 4. Evaluate
    logger.info("\n[4/4] Running evaluation...")
    evaluator = Evaluator(
        retriever=retriever,
        k_values=cfg.retrieval.k_values,
    )
    
    metrics = evaluator.evaluate(
        queries_df=queries_df,
        qrels=qrels,
        show_progress=True,
    )

    # Save results
    results_path = Path(cfg.paths.output_dir) / "evaluation_results.json"
    evaluator.save_results(metrics, str(results_path))

    # Generate and print report
    report = evaluator.generate_report(metrics)
    print("\n" + report)

    elapsed_time = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed_time:.2f} seconds")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

