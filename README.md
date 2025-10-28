### Goal

Build an intelligent retrieval system that adapts to the query — dynamic chunking, hybrid retrieval, and reranking — achieving measurable gains in precision and latency.

### Outcome

A modular RAG pipeline benchmarked on accuracy, speed, and efficiency; reproducible and production-ready.

### Milestone 1 — Retrieval Foundations

- **Tasks**
    - Set up repository, config system, and data loaders.
    - Ingest corpus, clean, and prepare metadata.
    - Implement basic dense retrieval pipeline (embeddings + vector DB).
- **Tools**: `sentence-transformers`, `FAISS` / `Qdrant`, `Hydra`, `Poetry`, `DVC`, `W&B`
- **Deliverable**: Baseline retrieval pipeline and evaluation script (Recall@k, nDCG)

### Milestone 2 — Dynamic Chunking Engine

- **Tasks**
    - Implement adaptive chunker: query-based length, semantic boundaries.
    - Cache and profile different chunking strategies.
- **Tools**: `tiktoken`, `spacy`, `nltk`, custom `LangChain` splitters
- **Deliverable**: Dynamic chunking component with benchmark results

### Milestone 3 — Hybrid Retrieval Layer

- **Tasks**
    - Add sparse retrieval (BM25/TF-IDF).
    - Fuse with dense results via reciprocal rank fusion or learned weights.
- **Tools**: `Elasticsearch` / `OpenSearch`, `rank-bm25`, `scikit-learn`
- **Deliverable**: Dual-mode retriever with hybrid weighting module

### Milestone 4 — Re-Ranker & Adaptive Policy

- **Tasks**
    - Integrate cross-encoder reranker.
    - Build adaptive policy (ML or heuristic) to decide reranking, k, and chunk size.
- **Tools**: `cross-encoder/ms-marco`, `bge-reranker`, `scikit-learn`, `Ray Tune`
- **Deliverable**: Policy-driven retriever with reranker on/off decisions

### Milestone 5 — Benchmark & Optimization

- **Tasks**
    - Design evaluation harness and latency tests.
    - Profile p50/p95 latency across configs.
    - Visualize trade-offs in dashboard.
- **Tools**: `ragas`, `Locust`, `Prometheus`, `Grafana`, `pytest`
- **Deliverable**: Quality vs latency dashboard; final report

### Success Criteria

- ≥ 15% improvement in nDCG@10 over dense baseline
- Median latency ≤ 500 ms with rerank enabled
