# Setup Guide

This guide will help you get the Adaptive RAG system up and running.

## Prerequisites

- Python 3.9 or higher
- 10GB free disk space (for dataset and models)
- 8-16GB RAM
- (Optional) NVIDIA GPU with CUDA for faster processing

## Installation Steps

### Option 1: Using Poetry (Recommended)

```bash
# 1. Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone/navigate to the project
cd adaptative-rag

# 3. Install dependencies
poetry install

# 4. Activate the virtual environment
poetry shell
```

### Option 2: Using pip

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -e .
```

## Quick Start

### Step 1: Build the Index

This will download the MS MARCO dataset and create the search index:

```bash
python scripts/build_index.py
```

**What happens:**
- Downloads ~3GB of data (first time only)
- Processes 8.8 million passages
- Creates embeddings using neural network
- Builds FAISS search index
- **Time**: 2-4 hours on CPU, ~30 minutes on GPU

**For quick testing**, you can use a smaller dataset:
```bash
# Only 10,000 passages for testing
python scripts/build_index.py data.corpus_split="train[:10000]"
```

### Step 2: Evaluate the System

Run evaluation on test queries:

```bash
python scripts/evaluate.py
```

**Output**: Metrics like Recall@10, nDCG@10, etc.

### Step 3: Try Interactive Search

```bash
python scripts/retrieve.py
```

**Try queries like:**
- "How does photosynthesis work?"
- "Best programming language for machine learning"
- "Symptoms of the common cold"

## Verify Installation

### Run Tests

```bash
# With Poetry
poetry run pytest

# With pip
pytest
```

### Check Configuration

```bash
# View loaded configuration
python scripts/build_index.py --cfg job
```

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
python scripts/build_index.py model.batch_size=32
```

**Solution 2**: Use smaller subset
```bash
python scripts/build_index.py data.corpus_split="train[:10000]"
```

### Issue: Slow Processing

**Solution 1**: Use GPU (if available)
```bash
python scripts/build_index.py device=cuda
```

**Solution 2**: Install with GPU support
```bash
poetry add faiss-gpu  # Instead of faiss-cpu
```

### Issue: Download Fails

**Solution**: Clear cache and retry
```bash
rm -rf data/cache
python scripts/build_index.py
```

### Issue: Import Errors

**Solution**: Ensure virtual environment is activated
```bash
poetry shell  # or: source venv/bin/activate
```

## Directory Structure After Setup

```
adaptative-rag/
├── configs/              # ✅ Configuration files
├── src/                  # ✅ Source code
├── scripts/              # ✅ Entry scripts
├── tests/                # ✅ Unit tests
├── docs/                 # ✅ Documentation
├── data/                 # ⚠️ Created on first run
│   └── cache/           # Downloaded datasets and models
├── outputs/              # ⚠️ Created after build_index
│   ├── faiss_index      # Search index
│   ├── doc_ids.pkl      # Document mapping
│   └── corpus.parquet   # Processed corpus
└── pyproject.toml        # ✅ Dependencies
```

## Next Steps

1. **Learn the System**
   - Read [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md) for detailed walkthrough
   - Try the interactive retrieval

2. **Customize**
   - Read [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
   - Experiment with different models

3. **Develop**
   - Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
   - Run tests: `pytest tests/`
   - Check code quality: `black src/ tests/`

## System Requirements

### Minimum

- **CPU**: 2+ cores
- **RAM**: 8 GB
- **Disk**: 10 GB
- **Time**: 3-4 hours for full index

### Recommended

- **CPU**: 4+ cores
- **RAM**: 16 GB
- **GPU**: NVIDIA with 6GB+ VRAM
- **Disk**: 20 GB (for experiments)
- **Time**: 30 minutes for full index (with GPU)

## Getting Help

- **Documentation**: See `docs/` folder
- **Examples**: Check `tests/` for code examples
- **Configuration**: See `configs/` for all options

## Development Setup

### Additional Tools

```bash
# Install dev dependencies (already in pyproject.toml)
poetry install

# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking (optional)
poetry add --group dev mypy
poetry run mypy src/
```

### Running Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific test file
poetry run pytest tests/test_metrics.py -v

# With output
poetry run pytest -s
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## FAQ

### Q: Can I use my own dataset?

**A**: Yes! Create a custom loader in `src/data/` following the `MSMARCOLoader` pattern, then create a new config in `configs/data/`.

### Q: Can I use different embedding models?

**A**: Yes! Any sentence-transformers model works. Create a config in `configs/model/`:
```yaml
model:
  name: sentence-transformers/all-mpnet-base-v2
  batch_size: 32
```

### Q: How do I deploy this to production?

**A**: Current milestone is for development. Future milestones will add:
- FastAPI REST endpoint
- Docker containers
- Monitoring and logging
- Rate limiting

### Q: Can I use this for non-English text?

**A**: Yes, but you'll need multilingual models like:
- `paraphrase-multilingual-MiniLM-L12-v2`
- `paraphrase-multilingual-mpnet-base-v2`

## License

MIT License - See LICENSE file for details.

