# Adaptive RAG: Intelligent Retrieval System

## 🎯 Project Goal

Build an **intelligent retrieval system that adapts to queries** using dynamic chunking, hybrid retrieval, and reranking to achieve measurable improvements in both precision and latency.

## 🏆 Target Outcome

A **modular, production-ready RAG (Retrieval-Augmented Generation) pipeline** that is:
- Benchmarked on accuracy, speed, and efficiency
- Reproducible and well-documented
- Adaptable to different query types and use cases

## 💡 Use Cases

This system is designed for scenarios where you need to:

1. **Question Answering Systems**: Find the most relevant documents to answer user questions
   - Customer support chatbots
   - Internal knowledge base search
   - Research paper retrieval

2. **Semantic Search**: Go beyond keyword matching to understand query intent
   - E-commerce product search
   - Legal document retrieval
   - Medical literature search

3. **RAG Applications**: Provide relevant context to LLMs for better responses
   - AI assistants with company knowledge
   - Code search and documentation
   - Academic research helpers

## 🚀 Quick Start

### Installation

```bash
# Install dependencies with Poetry
poetry install

# Or with pip (if not using Poetry)
pip install -e .
```

### Basic Usage

```bash
# 1. Build the search index (downloads MS MARCO and creates embeddings)
python scripts/build_index.py

# 2. Evaluate the retrieval system
python scripts/evaluate.py

# 3. Try interactive search
python scripts/retrieve.py
```

## 📊 Project Milestones

### ✅ Milestone 1 — Retrieval Foundations (Current)

**What**: Build a baseline dense retrieval system using semantic embeddings.

**Why**: Establish a solid foundation and baseline metrics before adding complexity.

**Deliverable**: 
- Working retrieval pipeline with FAISS vector database
- Evaluation metrics (Recall@k, nDCG)
- Baseline performance benchmarks

**[See detailed documentation →](docs/MILESTONE_1.md)**

### 🔜 Milestone 2 — Dynamic Chunking Engine

Implement adaptive text chunking based on query characteristics and semantic boundaries.

### 🔜 Milestone 3 — Hybrid Retrieval Layer

Combine dense (semantic) and sparse (keyword) retrieval methods for better coverage.

### 🔜 Milestone 4 — Re-Ranker & Adaptive Policy

Add cross-encoder reranking and intelligent decision-making for when to use different strategies.

### 🔜 Milestone 5 — Benchmark & Optimization

Comprehensive evaluation with latency profiling and visualization dashboard.

## 📈 Success Criteria

- ≥ 15% improvement in nDCG@10 over dense baseline
- Median latency ≤ 500ms with rerank enabled
- Modular, maintainable codebase

## 📁 Project Structure

```
adaptative-rag/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── data/                  # Dataset configs
│   ├── model/                 # Model configs
│   └── retrieval/             # Retrieval configs
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── retrieval/             # Retrieval components
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Shared utilities
├── scripts/                   # Entry point scripts
│   ├── build_index.py         # Build search index
│   ├── evaluate.py            # Run evaluation
│   └── retrieve.py            # Interactive search
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── data/                      # Data storage (created on first run)
```

## 🛠️ Technology Stack

### Core Libraries
- **sentence-transformers**: Neural text embeddings
- **FAISS**: Fast similarity search
- **Hydra**: Configuration management
- **Poetry**: Dependency management

### Dataset
- **MS MARCO**: Microsoft MAchine Reading COmprehension dataset
  - ~8.8M passages
  - ~6.9k evaluation queries
  - Industry-standard benchmark

## 📖 Documentation

- **[Milestone 1 Guide](docs/MILESTONE_1.md)**: Detailed walkthrough of the current system
- **[Architecture](docs/ARCHITECTURE.md)**: System design and components
- **[Configuration](docs/CONFIGURATION.md)**: How to customize settings

## 🧪 Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_metrics.py
```

## 📝 Development

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/
```

### Configuration

All settings are managed through Hydra configs in `configs/`. You can:
- Override any parameter from command line
- Create new config variants
- Compose multiple configs

Example:
```bash
# Use different model
python scripts/build_index.py model.name=all-mpnet-base-v2

# Change top_k results
python scripts/evaluate.py retrieval.top_k=20
```

## 🤝 Contributing

This is a learning/research project. Feel free to:
- Experiment with different models
- Add new evaluation metrics
- Optimize performance
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- MS MARCO dataset by Microsoft
- sentence-transformers by UKP Lab
- FAISS by Facebook Research
