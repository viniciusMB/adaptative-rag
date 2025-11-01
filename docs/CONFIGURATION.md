# Configuration Guide

## Overview

This project uses [Hydra](https://hydra.cc/) for configuration management. All settings are stored in YAML files under `configs/`.

## Configuration Structure

```
configs/
├── config.yaml              # Main config (includes others)
├── data/
│   └── msmarco.yaml        # MS MARCO dataset config
├── model/
│   └── minilm.yaml         # Embedding model config
└── retrieval/
    └── dense.yaml          # Retrieval config
```

## Main Configuration

**File**: `configs/config.yaml`

```yaml
defaults:
  - data: msmarco          # Use configs/data/msmarco.yaml
  - model: minilm          # Use configs/model/minilm.yaml
  - retrieval: dense       # Use configs/retrieval/dense.yaml
  - _self_                 # Include this config last

# Project paths
paths:
  data_dir: data
  output_dir: outputs
  cache_dir: ${paths.data_dir}/cache

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Runtime
device: cpu              # or "cuda" for GPU
seed: 42
```

## Data Configuration

**File**: `configs/data/msmarco.yaml`

```yaml
# @package _global_

data:
  name: msmarco
  dataset_name: ms_marco
  dataset_config: v1.1
  
  # Subsets to load
  load_corpus: true
  load_queries: true
  load_qrels: true
  
  # Splits
  corpus_split: train      # Which split for passages
  queries_split: dev       # Which split for queries
  
  # Preprocessing
  max_passage_length: 512  # Max chars per passage
  clean_text: true         # Apply text cleaning
  
  # Caching
  cache_embeddings: true
  embeddings_cache_path: ${paths.cache_dir}/embeddings.npy
  metadata_cache_path: ${paths.cache_dir}/metadata.pkl
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dataset_name` | str | `ms_marco` | HuggingFace dataset name |
| `dataset_config` | str | `v1.1` | Dataset version |
| `corpus_split` | str | `train` | Split for passages |
| `queries_split` | str | `dev` | Split for queries |
| `max_passage_length` | int | `512` | Max characters |
| `clean_text` | bool | `true` | Enable preprocessing |

## Model Configuration

**File**: `configs/model/minilm.yaml`

```yaml
# @package _global_

model:
  name: all-MiniLM-L6-v2   # Model from sentence-transformers
  type: sentence-transformer
  
  # Encoding parameters
  batch_size: 64           # Passages per batch
  max_length: 512          # Max tokens
  normalize_embeddings: true  # For cosine similarity
  
  # Device
  device: ${device}        # Inherit from main config
  
  # Model cache
  cache_folder: ${paths.cache_dir}/models
```

### Available Models

| Model | Dim | Speed | Quality | Size |
|-------|-----|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | 90MB |
| `all-mpnet-base-v2` | 768 | Medium | Better | 420MB |
| `all-MiniLM-L12-v2` | 384 | Medium | Good | 120MB |

**Create custom model config:**

```yaml
# configs/model/mpnet.yaml
# @package _global_

model:
  name: all-mpnet-base-v2
  batch_size: 32          # Smaller batch for larger model
  max_length: 512
  normalize_embeddings: true
  device: ${device}
  cache_folder: ${paths.cache_dir}/models
```

**Use it:**
```bash
python scripts/build_index.py model=mpnet
```

## Retrieval Configuration

**File**: `configs/retrieval/dense.yaml`

```yaml
# @package _global_

retrieval:
  type: dense
  
  # FAISS parameters
  index_type: IndexFlatIP         # Exact search, inner product
  metric_type: METRIC_INNER_PRODUCT
  
  # Retrieval parameters
  top_k: 10                       # Results to return
  k_values: [1, 5, 10, 20]       # For evaluation
  
  # Index paths
  index_path: ${paths.output_dir}/faiss_index
  doc_ids_path: ${paths.output_dir}/doc_ids.pkl
  
  # Search parameters
  search_batch_size: 32
```

### FAISS Index Types

| Type | Description | Speed | Accuracy | Best For |
|------|-------------|-------|----------|----------|
| `IndexFlatIP` | Exact search, inner product | Slow | 100% | <10M docs, baseline |
| `IndexFlatL2` | Exact search, L2 distance | Slow | 100% | When not using cosine |
| `IndexIVFFlat` | Approximate, clustered | Fast | 95-99% | >10M docs |
| `IndexIVFPQ` | Quantized, compressed | Very Fast | 90-95% | >100M docs |

## Command-Line Overrides

### Basic Override

```bash
# Override single parameter
python scripts/build_index.py retrieval.top_k=20

# Override multiple parameters
python scripts/build_index.py model.batch_size=128 retrieval.top_k=20
```

### Switch Config Group

```bash
# Use different model
python scripts/build_index.py model=mpnet

# Use different dataset (if you create configs/data/custom.yaml)
python scripts/build_index.py data=custom
```

### Nested Overrides

```bash
# Override nested values
python scripts/evaluate.py data.queries_split=test

# Override paths
python scripts/build_index.py paths.output_dir=/tmp/outputs
```

## Common Configurations

### Fast Testing (Small Dataset)

```bash
# Use first 10,000 passages only
python scripts/build_index.py \
  data.corpus_split="train[:10000]" \
  data.queries_split="dev[:100]"
```

### GPU Acceleration

```bash
# Use GPU for faster embedding
python scripts/build_index.py device=cuda model.batch_size=256
```

### High Accuracy Model

```bash
# Use more accurate (but slower) model
python scripts/build_index.py model=mpnet
```

### Different Top-K

```bash
# Retrieve more results
python scripts/evaluate.py retrieval.top_k=50 retrieval.k_values=[1,10,20,50]
```

## Creating Custom Configs

### Example: Fast Experimentation Config

**File**: `configs/experiment/fast.yaml`

```yaml
# @package _global_

# Inherit from defaults
defaults:
  - /data: msmarco
  - /model: minilm
  - /retrieval: dense
  - _self_

# Override for fast iteration
data:
  corpus_split: "train[:5000]"
  queries_split: "dev[:50]"

model:
  batch_size: 128

retrieval:
  top_k: 5
  k_values: [1, 5]
```

**Usage:**
```bash
python scripts/build_index.py +experiment=fast
```

### Example: Production Config

**File**: `configs/experiment/production.yaml`

```yaml
# @package _global_

defaults:
  - /data: msmarco
  - /model: mpnet          # More accurate model
  - /retrieval: dense
  - _self_

model:
  batch_size: 64
  device: cuda             # Assume GPU available

retrieval:
  top_k: 10
  k_values: [1, 5, 10, 20, 50]

logging:
  level: INFO
```

## Environment Variables

You can use environment variables in configs:

```yaml
# configs/config.yaml
paths:
  data_dir: ${oc.env:DATA_DIR,data}  # Use $DATA_DIR or default to "data"
  
device: ${oc.env:DEVICE,cpu}         # Use $DEVICE or default to "cpu"
```

**Usage:**
```bash
export DATA_DIR=/mnt/data
export DEVICE=cuda
python scripts/build_index.py
```

## Interpolation

Configs can reference other values:

```yaml
paths:
  data_dir: data
  output_dir: outputs
  cache_dir: ${paths.data_dir}/cache          # data/cache
  index_path: ${paths.output_dir}/index       # outputs/index

model:
  name: all-MiniLM-L6-v2
  cache_folder: ${paths.cache_dir}/models     # data/cache/models

device: cuda

retrieval:
  index_path: ${paths.index_path}/faiss      # outputs/index/faiss
```

## Validation

### Check Loaded Config

```bash
# Print final config without running
python scripts/build_index.py --cfg job
```

### Debug Config Loading

```bash
# Show config resolution
python scripts/build_index.py -cd configs -cn config hydra.verbose=true
```

## Best Practices

### 1. Don't Hardcode Values

❌ **Bad:**
```python
batch_size = 64
model_name = "all-MiniLM-L6-v2"
```

✅ **Good:**
```python
@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    batch_size = cfg.model.batch_size
    model_name = cfg.model.name
```

### 2. Use Config Groups

❌ **Bad:**
```yaml
# One monolithic config
model_name: all-MiniLM-L6-v2
model_batch_size: 64
data_name: msmarco
data_split: train
retrieval_top_k: 10
```

✅ **Good:**
```yaml
# Modular configs
defaults:
  - data: msmarco
  - model: minilm
  - retrieval: dense
```

### 3. Document Configs

✅ **Good:**
```yaml
model:
  batch_size: 64           # Passages per batch (reduce if OOM)
  max_length: 512          # Max tokens (must match model limit)
  normalize_embeddings: true  # Required for IndexFlatIP
```

### 4. Sensible Defaults

Choose defaults that work for most users:
- CPU (not everyone has GPU)
- Moderate batch sizes (don't assume 64GB RAM)
- Standard model names (from HuggingFace)

## Troubleshooting

### Config Not Found

```
Error: Cannot find primary config 'config.yaml'
```

**Solution:** Run from project root or specify config path:
```bash
cd /path/to/adaptative-rag
python scripts/build_index.py
```

### Override Not Working

```bash
# This doesn't work (typo)
python scripts/build_index.py model.batchsize=128

# This works
python scripts/build_index.py model.batch_size=128
```

**Tip:** Use `--cfg job` to verify overrides.

### Interpolation Error

```
omegaconf.errors.InterpolationKeyError: Interpolation key 'device' not found
```

**Solution:** Ensure referenced keys exist:
```yaml
device: cpu              # Define before use

model:
  device: ${device}      # Now works
```

## Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Patterns](https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/)

