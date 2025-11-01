# Architecture Documentation

## System Overview

The Adaptive RAG system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│         (Scripts: build_index.py, evaluate.py, etc.)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│              (DenseRetriever, Evaluator)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┬──────────────────┐
        │                              │                   │
┌───────▼────────┐          ┌─────────▼────────┐  ┌──────▼─────┐
│   DATA LAYER   │          │ RETRIEVAL LAYER   │  │  EVAL      │
│                │          │                   │  │  LAYER     │
│ • Loader       │          │ • Embedder        │  │ • Metrics  │
│ • Preprocessor │          │ • VectorStore     │  │ • Eval     │
└────────────────┘          └───────────────────┘  └────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                  ┌──────▼──────┐          ┌──────▼──────┐
                  │   MODELS    │          │   STORAGE   │
                  │  sentence-  │          │    FAISS    │
                  │ transformers│          │    Index    │
                  └─────────────┘          └─────────────┘
```

## Design Principles

### 1. Modularity

Each component has a **single, well-defined responsibility**:

```python
# Bad: Everything in one class
class SearchEngine:
    def load_data(self): ...
    def preprocess(self): ...
    def embed(self): ...
    def search(self): ...
    def evaluate(self): ...

# Good: Separate components
loader = MSMARCOLoader()          # Data loading
preprocessor = TextPreprocessor() # Text cleaning
embedder = Embedder()             # Embedding generation
vector_store = FAISSVectorStore() # Vector storage
retriever = DenseRetriever()      # Orchestration
evaluator = Evaluator()           # Evaluation
```

### 2. Configuration-Driven

All parameters are **externalized** in Hydra configs:

```yaml
# configs/config.yaml
model:
  name: all-MiniLM-L6-v2
  batch_size: 64

retrieval:
  top_k: 10
  index_type: IndexFlatIP
```

Benefits:
- No code changes for experiments
- Easy to track what worked
- Composable configs

### 3. Testability

Each component can be **tested independently**:

```python
# Test embedder without loading full dataset
embedder = Embedder()
embeddings = embedder.encode(["test1", "test2"])
assert embeddings.shape == (2, 384)

# Test metrics without running retrieval
recall = compute_recall(
    retrieved=["doc1", "doc2"],
    relevant={"doc1", "doc3"},
    k=10
)
assert recall == 0.5
```

### 4. Extensibility

Easy to **swap implementations**:

```python
# Future: Switch from FAISS to Qdrant
from src.retrieval.qdrant_store import QdrantVectorStore

# Same interface, different backend
vector_store = QdrantVectorStore(embedding_dim=384)
retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
```

## Component Deep Dive

### Data Layer

#### MSMARCOLoader

**Responsibilities:**
- Download dataset from HuggingFace
- Parse MS MARCO format
- Return pandas DataFrames

**Design Decisions:**
- Uses HuggingFace `datasets` for automatic caching
- Returns DataFrames (standard, easy to manipulate)
- Lazy loading (only downloads when needed)

**Future Extensions:**
- Support other datasets (BEIR, Natural Questions)
- Streaming for huge datasets
- Custom dataset loaders

#### TextPreprocessor

**Responsibilities:**
- Clean and normalize text
- Add metadata
- Filter invalid documents

**Design Decisions:**
- Configurable cleaning steps
- Preserves document IDs
- Returns modified DataFrame (immutable input)

**Future Extensions:**
- Language detection
- Advanced NLP (entity extraction, etc.)
- Custom cleaning pipelines

### Retrieval Layer

#### Embedder

**Responsibilities:**
- Load sentence-transformer models
- Batch encode texts
- Handle device management (CPU/GPU)

**Key Design:**
```python
class Embedder:
    def __init__(self, model_name, device, batch_size, ...):
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, texts):
        # Handles batching, progress, normalization
        return self.model.encode(texts, ...)
```

**Why This Design?**
- Wraps sentence-transformers for easier testing
- Centralizes embedding logic
- Easy to swap models

**Future Extensions:**
- Multi-modal embeddings (text + images)
- Fine-tuning on domain data
- Embedding caching strategies

#### FAISSVectorStore

**Responsibilities:**
- Store embeddings in searchable index
- Fast similarity search
- Save/load to disk

**Index Types:**
```python
# Current: IndexFlatIP (exact search)
index = faiss.IndexFlatIP(dim)
# Pros: 100% accurate
# Cons: Slower for >10M docs

# Future: IndexIVFFlat (approximate)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
# Pros: Much faster
# Cons: ~95-99% accurate
```

**Why FAISS?**
- Industry standard (Facebook Research)
- Extremely fast
- Many index types for different tradeoffs
- CPU and GPU support

**Future Extensions:**
- IVF indices for >50M docs
- Product Quantization for memory efficiency
- GPU acceleration

#### DenseRetriever

**Responsibilities:**
- Orchestrate embedder + vector store
- Manage corpus metadata
- Provide high-level retrieval API

**Design Pattern: Facade**
```python
# Hides complexity, provides simple interface
retriever = DenseRetriever(embedder, vector_store)
retriever.build_index(corpus_df)
results = retriever.retrieve(queries)
```

**Future Extensions:**
- Query preprocessing (expansion, reformulation)
- Result post-processing
- Multi-stage retrieval

### Evaluation Layer

#### Metrics

**Responsibilities:**
- Implement standard IR metrics
- Handle edge cases (no relevant docs, etc.)

**Metrics Explained:**

**Recall@k**: Coverage
```
Relevant docs found in top-k / Total relevant docs

Example:
Relevant: {doc1, doc2, doc3}
Retrieved top-10: [doc1, doc4, doc2, ...]
Recall@10 = 2/3 = 0.667 (found 2 out of 3)
```

**nDCG@k**: Ranking Quality
```
Rewards finding relevant docs higher in ranking

Example:
Perfect: [doc1(rel=3), doc2(rel=2), doc3(rel=1), ...]  → nDCG = 1.0
Bad:     [doc10(rel=0), doc11(rel=0), doc1(rel=3), ...] → nDCG = 0.3
```

**MAP**: Average Precision
```
Precision at each relevant doc, averaged

Example:
Retrieved: [doc1(R), doc2(X), doc3(R), doc4(X), doc5(R)]
AP = (1/1 + 2/3 + 3/5) / 3 = 0.756
```

**MRR**: First Relevant
```
Reciprocal rank of first relevant doc

Example:
First relevant at rank 3 → MRR = 1/3 = 0.333
```

#### Evaluator

**Responsibilities:**
- Run retrieval on test queries
- Compute metrics per query
- Aggregate and report results

**Design:**
```python
class Evaluator:
    def evaluate(self, queries_df, qrels):
        for query in queries:
            results = self.retriever.retrieve(query)
            metrics = compute_metrics(results, qrels[query])
        return aggregate(metrics)
```

## Data Flow

### Index Building Flow

```
1. MSMARCOLoader.load_corpus()
   ↓
   DataFrame: [doc_id, text]
   
2. TextPreprocessor.preprocess_corpus()
   ↓
   DataFrame: [doc_id, text, text_length, word_count]
   
3. Embedder.encode_corpus()
   ↓
   NumPy Array: (N, 384)
   
4. FAISSVectorStore.add()
   ↓
   FAISS Index: Searchable
   
5. Save to disk
   ↓
   Files: faiss_index, doc_ids.pkl
```

### Query Flow

```
1. User enters query
   ↓
   "How does machine learning work?"
   
2. Embedder.encode()
   ↓
   Query Vector: (1, 384)
   
3. FAISSVectorStore.search()
   ↓
   Indices: [12345, 67890, ...]
   Scores:  [0.89, 0.85, ...]
   
4. Map indices to doc_ids
   ↓
   Doc IDs: ["doc_12345", "doc_67890", ...]
   
5. Retrieve texts from corpus
   ↓
   Results: [(doc_id, score, text), ...]
```

### Evaluation Flow

```
1. Load queries + qrels
   ↓
   Queries: ["query1", "query2", ...]
   QRels: {query1: {doc1: 1, doc2: 1}, ...}
   
2. For each query:
   a. Retrieve top-k docs
   b. Compare with qrels
   c. Compute metrics
   ↓
   Per-query metrics
   
3. Aggregate across all queries
   ↓
   Final metrics: {recall@10: 0.75, nDCG@10: 0.48, ...}
```

## Configuration System

### Hydra Structure

```
configs/
├── config.yaml              # Main config, includes defaults
├── data/
│   └── msmarco.yaml        # Dataset-specific config
├── model/
│   ├── minilm.yaml         # Small, fast model
│   └── mpnet.yaml          # Larger, more accurate
└── retrieval/
    ├── dense.yaml          # Dense retrieval config
    └── hybrid.yaml         # (Future) Hybrid config
```

### Config Composition

**Default:**
```bash
python scripts/build_index.py
# Uses: data=msmarco, model=minilm, retrieval=dense
```

**Override:**
```bash
python scripts/build_index.py model=mpnet retrieval.top_k=20
# Uses: mpnet model, top_k=20
```

**Custom Config:**
```yaml
# configs/experiment/fast.yaml
defaults:
  - /data: msmarco
  - /model: minilm
  - /retrieval: dense
  - _self_

model:
  batch_size: 128  # Faster on GPU

data:
  corpus_split: train[:10000]  # Small subset for testing
```

```bash
python scripts/build_index.py +experiment=fast
```

## Error Handling

### Strategy

1. **Fail Fast**: Validate inputs early
2. **Informative Errors**: Clear messages with solutions
3. **Graceful Degradation**: Continue when possible

### Examples

```python
# Good error handling
if embeddings.shape[1] != self.embedding_dim:
    raise ValueError(
        f"Expected embeddings with dim {self.embedding_dim}, "
        f"got {embeddings.shape[1]}. "
        f"Check model configuration."
    )

# Graceful degradation
if query_id not in qrels:
    logger.warning(f"No qrels for {query_id}, skipping")
    continue  # Don't crash, just skip
```

## Performance Considerations

### Memory

**Embeddings Storage:**
```
8.8M passages × 384 dimensions × 4 bytes (float32)
= 13.5 GB for full corpus embeddings
```

**Optimization:**
- Use float16 (half precision): 6.75 GB
- Quantization (future): 3.4 GB with minimal accuracy loss

### Speed

**Bottlenecks:**
1. **Embedding Generation** (slowest)
   - CPU: ~500 passages/sec
   - GPU: ~5000 passages/sec
   - Solution: Batch processing, GPU, caching

2. **FAISS Search**
   - IndexFlatIP: ~50-100ms for 8.8M docs
   - IndexIVF: ~10-20ms (approximate)

3. **Data Loading**
   - HuggingFace datasets: Cached after first load
   - Use `.parquet` for fast loading

### Scalability

| Documents | Index Type   | Build Time | Query Time |
|-----------|--------------|------------|------------|
| 1M        | IndexFlat    | 10 min     | 10ms       |
| 10M       | IndexFlat    | 2 hours    | 100ms      |
| 100M      | IndexIVF     | 5 hours    | 20ms       |
| 1B        | IndexIVFPQ   | 20 hours   | 50ms       |

## Testing Strategy

### Unit Tests

Test **individual components** in isolation:
```python
def test_recall():
    # No dependencies, pure function
    recall = compute_recall(["doc1"], {"doc1", "doc2"}, k=1)
    assert recall == 0.5
```

### Integration Tests

Test **component interactions**:
```python
def test_retriever_pipeline():
    embedder = Embedder()
    store = FAISSVectorStore(dim=384)
    retriever = DenseRetriever(embedder, store)
    
    retriever.build_index(small_corpus)
    results = retriever.retrieve(["test query"])
    
    assert len(results[0]) > 0
```

### End-to-End Tests

Test **full workflows** (slower, fewer tests):
```python
def test_build_and_search():
    # Full pipeline
    subprocess.run(["python", "scripts/build_index.py", "..."])
    subprocess.run(["python", "scripts/evaluate.py", "..."])
    # Check outputs exist and are valid
```

## Future Architecture

### Milestone 2: Dynamic Chunking

```
Query → Chunker → [Select chunk size/strategy]
          ↓
      Chunked Corpus
          ↓
      Embeddings
          ↓
       Retrieval
```

### Milestone 3: Hybrid Retrieval

```
        Query
          ↓
    ┌─────┴─────┐
    ▼           ▼
  Dense      Sparse
 (semantic) (keyword)
    │           │
    └─────┬─────┘
          ▼
     Fusion Layer
     (combine)
          ↓
       Results
```

### Milestone 4: Reranking

```
Query → Dense Retrieval → Top 100
                ↓
        Cross-Encoder Reranker
        (more accurate, slower)
                ↓
           Top 10 (refined)
```

## Deployment Considerations

### Production Readiness

**Current State (Milestone 1):**
- ✅ Modular, testable code
- ✅ Configuration management
- ✅ Logging
- ⚠️ No API server (script-based)
- ⚠️ No monitoring
- ❌ No authentication

**Next Steps:**
- FastAPI REST endpoint
- Docker containerization
- Prometheus metrics
- Rate limiting
- Error tracking (Sentry)

### Example Production Architecture

```
┌─────────┐
│ Client  │
└────┬────┘
     │ HTTP
     ▼
┌─────────────┐
│  Load Bal.  │
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│ API │ │ API │  (FastAPI servers)
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       ▼
┌──────────────┐
│ FAISS Index  │  (Shared memory or replicated)
└──────────────┘
```

## Glossary

- **Dense Retrieval**: Using neural embeddings for semantic search
- **Sparse Retrieval**: Traditional keyword-based (BM25, TF-IDF)
- **Embedding**: Vector representation of text
- **FAISS**: Facebook AI Similarity Search library
- **MS MARCO**: Microsoft MAchine Reading COmprehension dataset
- **Recall@k**: Proportion of relevant docs in top-k results
- **nDCG**: Normalized Discounted Cumulative Gain (ranking quality)
- **Hydra**: Configuration management framework

