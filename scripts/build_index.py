"""Script to build FAISS index from corpus."""

import logging
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import MSMARCOLoader
from src.data.preprocessor import TextPreprocessor
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build and save FAISS index."""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )

    logger.info("=" * 60)
    logger.info("Building FAISS Index")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Load data
    logger.info("\n[1/4] Loading corpus...")
    loader = MSMARCOLoader(
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        cache_dir=cfg.paths.cache_dir,
    )
    
    corpus_df = loader.load_corpus(split=cfg.data.corpus_split)
    
    # Sample for testing if needed (comment out for full corpus)
    # logger.warning("SAMPLING CORPUS FOR TESTING")
    # corpus_df = corpus_df.head(10000)

    # 2. Preprocess
    logger.info("\n[2/4] Preprocessing text...")
    preprocessor = TextPreprocessor(
        max_length=cfg.data.max_passage_length,
        remove_extra_whitespace=cfg.data.clean_text,
    )
    corpus_df = preprocessor.preprocess_corpus(corpus_df)

    # 3. Initialize embedder
    logger.info("\n[3/4] Initializing embedder...")
    embedder = Embedder(
        model_name=cfg.model.name,
        device=cfg.model.device,
        batch_size=cfg.model.batch_size,
        max_length=cfg.model.max_length,
        normalize_embeddings=cfg.model.normalize_embeddings,
        cache_folder=cfg.model.cache_folder,
    )

    # 4. Build index
    logger.info("\n[4/4] Building FAISS index...")
    retriever = DenseRetriever(
        embedder=embedder,
        top_k=cfg.retrieval.top_k,
    )
    
    retriever.build_index(
        corpus_df=corpus_df,
        show_progress=True,
    )

    # 5. Save index
    logger.info("\nSaving index...")
    retriever.save(
        index_path=cfg.retrieval.index_path,
        doc_ids_path=cfg.retrieval.doc_ids_path,
    )

    # Save corpus for later reference
    corpus_path = Path(cfg.paths.output_dir) / "corpus.parquet"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(corpus_path, index=False)
    logger.info(f"Saved corpus to {corpus_path}")

    elapsed_time = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed_time:.2f} seconds")
    logger.info("Index building complete!")


if __name__ == "__main__":
    main()

