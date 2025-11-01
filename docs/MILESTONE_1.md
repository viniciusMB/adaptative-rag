# Milestone 1: Retrieval Foundations - Complete Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Components](#components)
6. [Usage Examples](#usage-examples)
7. [Performance Targets](#performance-targets)

## Overview

### What is Milestone 1?

Milestone 1 builds the **foundation of our retrieval system**: a dense retrieval pipeline using semantic embeddings. Think of it as creating a smart search engine that understands meaning, not just keywords.

### Why Dense Retrieval?

**Traditional keyword search** (like Google in the 90s):
```
Query: "How do I fix a leaky faucet?"
Matches: Documents containing words "fix", "leaky", "faucet"
Problem: Misses "repairing dripping tap" or "stop water leak"
```

**Dense retrieval** (what we're building):
```
Query: "How do I fix a leaky faucet?"
Matches: Documents with similar MEANING, even with different words
Finds: "repairing dripping tap", "stop water leak", "plumbing repairs"
```

### Real-World Analogy

Imagine you're a librarian:
1. **Traditional search**: You only help if someone asks using exact titles/keywords
2. **Dense retrieval**: You understand what people mean and recommend relevant books even if they use different words

## How It Works

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MILESTONE 1: DENSE RETRIEVAL                  │
└─────────────────────────────────────────────────────────────────┘

USER QUERY: "How does photosynthesis work?"
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. EMBEDDING MODEL                                               │
│    Converts text to numbers (vectors)                            │
│    Query → [0.23, -0.45, 0.67, ..., 0.12]  (384 dimensions)    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. VECTOR SEARCH (FAISS)                                         │
│    Finds similar vectors in our database                         │
│    Compares with 8.8M document embeddings                        │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. RANKED RESULTS                                                │
│    Returns top 10 most similar documents                         │
│    Score: 0.89 → "Photosynthesis is the process..."            │
│    Score: 0.85 → "Plants convert light energy..."              │
│    Score: 0.82 → "Chlorophyll captures sunlight..."            │
└──────────────────────────────────────────────────────────────────┘
```

### The Magic of Embeddings

**Embeddings** are the secret sauce. They convert text into numbers that capture meaning:

```
Text: "The cat sat on the mat"
         ↓
Embedding Model (Neural Network)
         ↓
Vector: [0.23, -0.45, 0.67, 0.12, ..., 0.89]
        └─── 384 numbers that represent the meaning ───┘
```

**Similar meanings = Similar vectors:**
```
"dog" and "puppy"     → Very close vectors (high similarity)
"dog" and "car"       → Far apart vectors (low similarity)
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  MSMARCOLoader          │  TextPreprocessor                     │
│  • Download dataset      │  • Clean text                         │
│  • Load passages         │  • Normalize whitespace               │
│  • Load queries          │  • Handle special chars               │
│  • Load relevance data   │  • Add metadata                       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Embedder               │  FAISSVectorStore  │  DenseRetriever  │
│  • Load model           │  • Store embeddings │  • Orchestrate   │
│  • Encode text          │  • Build index      │  • Query→Results │
│  • Batch processing     │  • Fast search      │  • Save/Load     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Metrics                │  Evaluator                             │
│  • Recall@k             │  • Run evaluation                      │
│  • nDCG@k               │  • Generate reports                    │
│  • MAP, MRR             │  • Compare performance                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Building the Index

```
┌─────────────┐
│  MS MARCO   │  8.8 Million passages
│  Dataset    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Load & Clean    │  Remove empty, normalize text
│ (Preprocessor)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Generate        │  Convert each passage to 384-dim vector
│ Embeddings      │  Uses: all-MiniLM-L6-v2 model
│ (Embedder)      │  Batches of 64 for efficiency
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Build FAISS     │  Create searchable index
│ Index           │  Type: IndexFlatIP (exact search)
│ (VectorStore)   │  Optimized for cosine similarity
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Save to Disk    │  index.faiss + doc_ids.pkl
└─────────────────┘  Ready for fast retrieval!
```

### Data Flow: Searching

```
User Query: "What is machine learning?"
    │
    ▼
┌────────────────────┐
│ Encode Query       │  Convert to 384-dim vector
│ (Embedder)         │  [0.34, -0.12, 0.78, ...]
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Search Index       │  Compare with 8.8M embeddings
│ (FAISS)            │  Find top-K most similar
└────────┬───────────┘  Time: ~50-100ms
         │
         ▼
┌────────────────────┐
│ Retrieve Docs      │  doc_id → actual text
│ (Retriever)        │  Return with scores
└────────┬───────────┘
         │
         ▼
    Top 10 Results
```

## Step-by-Step Walkthrough

### Step 1: Build the Index (One-Time Setup)

**Command:**
```bash
python scripts/build_index.py
```

**What Happens:**

1. **Downloads MS MARCO dataset** (~3GB)
   - 8.8 million passages
   - Takes 10-20 minutes first time
   - Cached for future use

2. **Preprocesses text**
   - Removes extra whitespace
   - Normalizes special characters
   - Adds metadata (length, word count)

3. **Generates embeddings**
   - Downloads all-MiniLM-L6-v2 model (~90MB)
   - Encodes each passage to 384-dim vector
   - Takes 2-4 hours on CPU (30 mins on GPU)
   - Progress bar shows status

4. **Builds FAISS index**
   - Creates searchable vector database
   - Uses IndexFlatIP (exact search)
   - Saves to `outputs/faiss_index`

**Output:**
```
outputs/
├── faiss_index           # Vector index
├── doc_ids.pkl          # Document ID mapping
└── corpus.parquet       # Original passages
```

### Step 2: Evaluate Performance

**Command:**
```bash
python scripts/evaluate.py
```

**What Happens:**

1. **Loads test queries** (~6,900 queries)
   - Each has known relevant documents
   - Used to measure accuracy

2. **Retrieves results** for each query
   - Top 10 documents per query
   - ~50-100ms per query

3. **Computes metrics**
   - **Recall@10**: What % of relevant docs we found
   - **nDCG@10**: Quality of ranking (better docs should rank higher)
   - **MAP**: Mean Average Precision
   - **MRR**: Mean Reciprocal Rank

4. **Generates report**

**Expected Output:**
```
============================================================
RETRIEVAL EVALUATION REPORT
============================================================

Number of documents: 8841823

Recall@k:
  Recall@ 1: 0.4123
  Recall@ 5: 0.6789
  Recall@10: 0.7541
  Recall@20: 0.8234

nDCG@k:
  nDCG@ 5: 0.4532
  nDCG@10: 0.4891

MAP: 0.4123
MRR: 0.5234

============================================================
```

### Step 3: Interactive Search

**Command:**
```bash
python scripts/retrieve.py
```

**Try queries like:**
- "How does solar power work?"
- "Best programming languages for beginners"
- "Symptoms of vitamin D deficiency"

**Example Output:**
```
Query: How does solar power work?

Top 10 results:
------------------------------------------------------------

[1] Score: 0.8923 | Doc ID: doc_1234567
    Solar panels convert sunlight into electricity using
    photovoltaic cells. When sunlight hits the cells...

[2] Score: 0.8756 | Doc ID: doc_2345678
    Photovoltaic systems harness energy from the sun by
    using semiconductor materials that generate electric...

[3] Score: 0.8534 | Doc ID: doc_3456789
    The process of converting solar radiation into usable
    electricity involves several key components...
```

## Components

### 1. MSMARCOLoader (`src/data/loader.py`)

**Purpose**: Download and load the MS MARCO dataset.

**Key Methods:**
- `load_corpus()`: Get all passages
- `load_queries()`: Get test queries
- `load_qrels()`: Get relevance judgments (which docs answer which queries)

**Usage:**
```python
from src.data.loader import MSMARCOLoader

loader = MSMARCOLoader()
corpus_df = loader.load_corpus(split="train")
queries_df = loader.load_queries(split="dev")
qrels = loader.load_qrels(split="dev")
```

### 2. TextPreprocessor (`src/data/preprocessor.py`)

**Purpose**: Clean and normalize text.

**Key Features:**
- Remove extra whitespace
- Normalize special characters
- Truncate to max length
- Add metadata

**Usage:**
```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(max_length=512)
cleaned_corpus = preprocessor.preprocess_corpus(corpus_df)
```

### 3. Embedder (`src/retrieval/embedder.py`)

**Purpose**: Convert text to embeddings using neural networks.

**Model**: all-MiniLM-L6-v2
- 384 dimensions
- Fast (2000 passages/second on CPU)
- Good quality for English text

**Usage:**
```python
from src.retrieval.embedder import Embedder

embedder = Embedder(model_name="all-MiniLM-L6-v2")
embeddings = embedder.encode(["Hello world", "Another text"])
# Returns: numpy array of shape (2, 384)
```

### 4. FAISSVectorStore (`src/retrieval/vector_store.py`)

**Purpose**: Store embeddings and enable fast similarity search.

**Index Type**: IndexFlatIP
- "Flat" = exhaustive search (checks all vectors)
- "IP" = Inner Product (for cosine similarity on normalized vectors)
- Accurate but slower for huge datasets (fine for 8.8M)

**Usage:**
```python
from src.retrieval.vector_store import FAISSVectorStore

store = FAISSVectorStore(embedding_dim=384)
store.add(embeddings, doc_ids)
results = store.search(query_embedding, top_k=10)
```

### 5. DenseRetriever (`src/retrieval/retriever.py`)

**Purpose**: Orchestrate embedding and searching.

**Combines:**
- Embedder (text → vectors)
- VectorStore (search vectors)
- Corpus (vectors → actual text)

**Usage:**
```python
from src.retrieval.retriever import DenseRetriever

retriever = DenseRetriever(embedder=embedder)
retriever.build_index(corpus_df)
results = retriever.retrieve(["your query"])
```

### 6. Evaluator (`src/evaluation/evaluator.py`)

**Purpose**: Measure retrieval quality.

**Metrics:**
- **Recall@k**: % of relevant docs in top-k results
- **nDCG@k**: Normalized Discounted Cumulative Gain (rewards ranking quality)
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank (position of first relevant doc)

## Usage Examples

### Example 1: Build Custom Index

```python
from src.data.loader import MSMARCOLoader
from src.data.preprocessor import TextPreprocessor
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever

# Load data
loader = MSMARCOLoader()
corpus_df = loader.load_corpus()

# Preprocess
preprocessor = TextPreprocessor()
corpus_df = preprocessor.preprocess_corpus(corpus_df)

# Create retriever
embedder = Embedder(model_name="all-MiniLM-L6-v2")
retriever = DenseRetriever(embedder=embedder)

# Build index
retriever.build_index(corpus_df)

# Save
retriever.save("outputs/faiss_index", "outputs/doc_ids.pkl")
```

### Example 2: Search

```python
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever

# Load retriever
embedder = Embedder()
retriever = DenseRetriever(embedder=embedder)
retriever.load("outputs/faiss_index", "outputs/doc_ids.pkl")

# Search
query = "How does machine learning work?"
result = retriever.retrieve_single(query, top_k=5, return_texts=True)

# Print results
for doc_id, score, text in zip(result['doc_ids'], result['scores'], result['texts']):
    print(f"Score: {score:.4f}")
    print(f"Text: {text[:200]}...")
    print()
```

### Example 3: Custom Configuration

Create `configs/model/custom.yaml`:
```yaml
# @package _global_

model:
  name: all-mpnet-base-v2  # More accurate model
  batch_size: 32
  normalize_embeddings: true
```

Run with:
```bash
python scripts/build_index.py --config-name=config model=custom
```

## Performance Targets

### Accuracy (Expected Baseline)

| Metric      | Target   | Meaning                                      |
|-------------|----------|----------------------------------------------|
| Recall@10   | ≥ 0.75   | Find 75% of relevant docs in top 10         |
| nDCG@10     | ≥ 0.45   | Good ranking quality                         |
| MAP         | ≥ 0.40   | Consistent precision across queries          |

### Speed

| Operation       | Target        | Notes                           |
|-----------------|---------------|---------------------------------|
| Single query    | < 100ms       | Including embedding + search    |
| Batch (100)     | < 2s          | Amortized ~20ms per query      |
| Index building  | 2-4 hours CPU | One-time cost, ~30min on GPU   |

### Resource Usage

| Resource | Requirement  | Notes                              |
|----------|--------------|------------------------------------|
| RAM      | 8-16 GB      | For loading index and model        |
| Disk     | 10 GB        | Dataset + embeddings + index       |
| GPU      | Optional     | 10x faster index building          |

## What's Next?

After Milestone 1, we'll add:

1. **Dynamic Chunking** (Milestone 2)
   - Split documents intelligently based on query type
   - Semantic boundary detection

2. **Hybrid Retrieval** (Milestone 3)
   - Combine dense (semantic) + sparse (keyword) search
   - Best of both worlds

3. **Reranking** (Milestone 4)
   - More accurate but slower cross-encoder models
   - Apply only to top candidates

4. **Adaptive Strategy** (Milestone 5)
   - Automatically choose the best approach for each query
   - Balance speed vs accuracy

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size in configs/model/minilm.yaml
model:
  batch_size: 32  # Reduce to 16 or 8
```

### Slow on CPU?
```bash
# Sample smaller corpus for testing
# Edit scripts/build_index.py, uncomment:
corpus_df = corpus_df.head(10000)  # Use only 10k docs
```

### Download Fails?
```bash
# Clear cache and retry
rm -rf data/cache
python scripts/build_index.py
```

## Questions?

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- See [CONFIGURATION.md](CONFIGURATION.md) for all config options
- Review tests in `tests/` for code examples

