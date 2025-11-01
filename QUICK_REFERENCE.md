# 🚀 Quick Reference Card

## What Is This Project?

An **intelligent semantic search system** that understands meaning, not just keywords.

**Goal**: Find relevant documents even when they use different words than the query.

## 📖 Documentation Map (Read in This Order)

| # | Document | Purpose | Time |
|---|----------|---------|------|
| 1 | [`README.md`](README.md) | Project overview | 5 min |
| 2 | [`SETUP.md`](SETUP.md) | Installation | 10 min |
| 3 | [`docs/VISUAL_GUIDE.md`](docs/VISUAL_GUIDE.md) | How it works (diagrams) | 15 min |
| 4 | [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md) | Detailed walkthrough | 30 min |
| 5 | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Customization | 15 min |
| 6 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Technical deep dive | 30 min |

**For first-time users**: Read 1-4 in order.

## ⚡ Quick Start

```bash
# 1. Install
poetry install

# 2. Build search index (small test)
python scripts/build_index.py data.corpus_split="train[:10000]"

# 3. Evaluate
python scripts/evaluate.py

# 4. Interactive search
python scripts/retrieve.py
```

## 🎯 Use Cases

| Use Case | Example |
|----------|---------|
| **Customer Support** | Search knowledge base for solutions |
| **Research** | Find relevant academic papers |
| **E-commerce** | Product search by description |
| **Legal** | Find similar case documents |
| **Healthcare** | Search medical literature |
| **RAG Systems** | Provide context to LLMs |

## 🔍 How It Works (30 Seconds)

```
Your Query: "How does solar power work?"
    ↓
[Neural Network] Converts to numbers (embedding)
    ↓
[FAISS Search] Finds similar embeddings in 8.8M documents
    ↓
Results: Relevant docs about solar energy (even with different words!)
```

**Key Insight**: Similar meanings → Similar number patterns (embeddings)

## 📁 File Structure (Where to Find Things)

```
configs/        → Settings (YAML files)
docs/           → Documentation (START HERE!)
scripts/        → Run these (build_index.py, evaluate.py, retrieve.py)
src/            → Source code (don't modify unless developing)
tests/          → Unit tests (examples of usage)
```

## 🛠️ Common Commands

### Building Index

```bash
# Full corpus (2-4 hours)
python scripts/build_index.py

# Small test (5 minutes)
python scripts/build_index.py data.corpus_split="train[:10000]"

# With GPU
python scripts/build_index.py device=cuda
```

### Evaluation

```bash
# Standard evaluation
python scripts/evaluate.py

# Different model
python scripts/evaluate.py model=mpnet

# More results
python scripts/evaluate.py retrieval.top_k=20
```

### Interactive Search

```bash
# Start interactive mode
python scripts/retrieve.py

# Then type queries like:
# - "What causes earthquakes?"
# - "Best practices for Python testing"
# - "How to train a neural network"
```

### Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=src

# Specific test
poetry run pytest tests/test_metrics.py -v
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/
```

## ⚙️ Configuration Quick Reference

All configs in `configs/` folder:

| File | What It Controls |
|------|------------------|
| `config.yaml` | Main settings (paths, device, logging) |
| `data/msmarco.yaml` | Dataset settings |
| `model/minilm.yaml` | Embedding model settings |
| `retrieval/dense.yaml` | Search settings |

**Override from command line:**
```bash
python scripts/build_index.py model.batch_size=128 retrieval.top_k=20
```

## 📊 Evaluation Metrics

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **Recall@10** | % of relevant docs found in top 10 | ≥ 0.75 |
| **nDCG@10** | Ranking quality (1.0 = perfect) | ≥ 0.45 |
| **MAP** | Mean Average Precision | ≥ 0.40 |
| **MRR** | How high first relevant doc ranks | ≥ 0.50 |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `model.batch_size=32` or smaller corpus |
| Slow processing | Use GPU: `device=cuda` |
| Import errors | Activate environment: `poetry shell` |
| Download fails | Clear cache: `rm -rf data/cache` |
| Tests fail | Install deps: `poetry install` |

## 🎯 Project Goals

**Current (Milestone 1)**: Dense retrieval baseline
**Next (Milestone 2)**: Dynamic chunking
**Future (M3-M5)**: Hybrid retrieval, reranking, optimization

**Target**: 15% improvement over baseline with <500ms latency

## 🧩 Key Components

| Component | File | What It Does |
|-----------|------|--------------|
| **Loader** | `src/data/loader.py` | Downloads MS MARCO dataset |
| **Preprocessor** | `src/data/preprocessor.py` | Cleans text |
| **Embedder** | `src/retrieval/embedder.py` | Text → Numbers |
| **VectorStore** | `src/retrieval/vector_store.py` | FAISS search |
| **Retriever** | `src/retrieval/retriever.py` | Orchestrates search |
| **Evaluator** | `src/evaluation/evaluator.py` | Measures performance |

## 💡 Key Concepts

### Embeddings
Numbers that represent meaning. Similar meanings = similar numbers.

```
"cat" → [0.2, 0.8, ...]  ←┐
"dog" → [0.2, 0.8, ...]  ←┘ Close together (similar)

"car" → [-0.5, 0.1, ...] ← Far away (different)
```

### Dense Retrieval
Search by meaning using embeddings (vs keywords).

### FAISS
Facebook's library for fast similarity search in millions of vectors.

## 📚 External Resources

- **MS MARCO**: https://microsoft.github.io/msmarco/
- **sentence-transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Hydra**: https://hydra.cc/

## 🎓 Learning Path

**Level 1 - User** (1 hour):
1. Read README
2. Run scripts
3. Try queries

**Level 2 - Configurator** (2 hours):
1. Read CONFIGURATION.md
2. Experiment with settings
3. Try different models

**Level 3 - Developer** (4 hours):
1. Read ARCHITECTURE.md
2. Understand components
3. Run/write tests

**Level 4 - Researcher** (ongoing):
1. Analyze metrics
2. Try improvements
3. Benchmark changes

## 🚦 Quick Status Check

After setup, you should have:

- ✅ `outputs/faiss_index` (search index)
- ✅ `outputs/doc_ids.pkl` (document mapping)
- ✅ `outputs/corpus.parquet` (processed documents)
- ✅ `data/cache/` (downloaded data and models)

Run `ls -lh outputs/` to verify.

## 🎯 What's Unique About This Implementation?

1. **Complete Documentation**: 6 docs, 3500+ lines
2. **Visual Guides**: Diagrams and explanations
3. **Production-Ready**: Tests, logging, error handling
4. **Configurable**: No code changes needed for experiments
5. **Modular**: Easy to extend and maintain
6. **Educational**: Clear explanations at every level

## 🔑 Most Important Files

| File | Why Important |
|------|---------------|
| [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md) | Complete walkthrough |
| [`docs/VISUAL_GUIDE.md`](docs/VISUAL_GUIDE.md) | Visual explanations |
| [`scripts/retrieve.py`](scripts/retrieve.py) | Try the system |
| [`SETUP.md`](SETUP.md) | Get started |

## 💬 Example Queries to Try

```
"How does machine learning work?"
"What causes climate change?"
"Best way to learn Python"
"Symptoms of vitamin D deficiency"
"How to start a business"
"Benefits of meditation"
"Quantum computing basics"
"History of the internet"
```

## 📞 Need Help?

1. Check troubleshooting in SETUP.md
2. Read relevant doc in `docs/`
3. Look at tests for code examples
4. Review configs for options

## 🎊 You're Ready!

With this foundation, you can:
- Build semantic search applications
- Understand RAG systems
- Experiment with embeddings
- Scale to millions of documents
- Extend with new features

**Happy searching! 🚀**

