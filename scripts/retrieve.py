"""Interactive retrieval script."""

import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Interactive retrieval."""
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )

    logger.info("Loading retriever...")

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

    # Load corpus
    corpus_path = Path(cfg.paths.output_dir) / "corpus.parquet"
    if corpus_path.exists():
        retriever.corpus_df = pd.read_parquet(corpus_path)
        logger.info(f"Loaded corpus with {len(retriever.corpus_df)} documents")

    logger.info("Retriever ready!")
    logger.info(f"Index contains {retriever.num_documents} documents")
    
    print("\n" + "=" * 60)
    print("Interactive Retrieval")
    print("=" * 60)
    print("Enter your query (or 'quit' to exit)")
    print()

    # Interactive loop
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            # Retrieve
            result = retriever.retrieve_single(
                query,
                top_k=cfg.retrieval.top_k,
                return_texts=True,
            )

            # Display results
            print(f"\nTop {len(result['doc_ids'])} results:")
            print("-" * 60)

            for i, (doc_id, score, text) in enumerate(
                zip(result["doc_ids"], result["scores"], result["texts"]),
                start=1
            ):
                print(f"\n[{i}] Score: {score:.4f} | Doc ID: {doc_id}")
                print(f"    {text[:200]}..." if len(text) > 200 else f"    {text}")

            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()

