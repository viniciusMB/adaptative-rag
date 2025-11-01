# Documentation Index

Welcome to the Adaptive RAG documentation!

## 📚 Documentation Files

### For Users

1. **[MILESTONE_1.md](MILESTONE_1.md)** - **START HERE!**
   - Complete walkthrough of the current system
   - How everything works with diagrams
   - Step-by-step usage guide
   - Use cases and examples

2. **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration Guide
   - How to customize settings
   - All available options
   - Command-line overrides
   - Creating custom configs

### For Developers

3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System Architecture
   - Design principles
   - Component deep dive
   - Data flows
   - Performance considerations
   - Future extensions

## 🚀 Quick Links

### New to the Project?
→ Read [../README.md](../README.md) for project overview
→ Then dive into [MILESTONE_1.md](MILESTONE_1.md) for details

### Want to Run It?
```bash
# 1. Install
poetry install

# 2. Build index
python scripts/build_index.py

# 3. Evaluate
python scripts/evaluate.py

# 4. Interactive search
python scripts/retrieve.py
```

### Want to Customize?
→ See [CONFIGURATION.md](CONFIGURATION.md) for all options

### Want to Understand the Code?
→ See [ARCHITECTURE.md](ARCHITECTURE.md) for design details

### Want to Contribute?
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Check `tests/` for examples
3. Run tests: `poetry run pytest`
4. Format code: `poetry run black src/ tests/`

## 📖 Additional Resources

- **MS MARCO Dataset**: https://microsoft.github.io/msmarco/
- **sentence-transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Hydra**: https://hydra.cc/

## 💡 Learning Path

**Level 1: User**
1. Run the scripts
2. Try interactive retrieval
3. Experiment with different queries

**Level 2: Configurator**
1. Read CONFIGURATION.md
2. Try different models
3. Adjust parameters
4. Create custom configs

**Level 3: Developer**
1. Read ARCHITECTURE.md
2. Understand components
3. Run and write tests
4. Extend functionality

**Level 4: Researcher**
1. Read evaluation metrics
2. Analyze results
3. Try improvements
4. Benchmark changes

## 🎯 Project Goals Recap

**Ultimate Goal**: Build an adaptive retrieval system that:
- Understands query intent (semantic search)
- Adapts chunking strategy to query type
- Combines multiple retrieval methods
- Reranks results intelligently
- Balances accuracy and speed

**Current Status** (Milestone 1):
- ✅ Dense retrieval foundation
- ✅ Baseline metrics
- 🔜 Dynamic chunking (M2)
- 🔜 Hybrid retrieval (M3)
- 🔜 Reranking (M4)
- 🔜 Optimization (M5)

## 📞 Need Help?

- Check the troubleshooting sections in each doc
- Review test files for code examples
- Look at configs for available options

