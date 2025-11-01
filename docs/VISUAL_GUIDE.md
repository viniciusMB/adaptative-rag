# Visual Guide to Adaptive RAG

A visual explanation of how the system works, from query to results.

## 🎯 The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE RAG JOURNEY                          │
│                                                                  │
│  Milestone 1      Milestone 2      Milestone 3      Milestone 4  │
│  Dense ────────►  Dynamic ───────► Hybrid ───────►  Reranking   │
│  Retrieval        Chunking         Retrieval        + Adaptive   │
│                                                                  │
│  [YOU ARE HERE]    [COMING NEXT]   [FUTURE]        [FUTURE]     │
└─────────────────────────────────────────────────────────────────┘
```

## 📚 What Problem Are We Solving?

### Traditional Keyword Search

```
User Query: "How do I improve my sleep quality?"

┌─────────────────────────────────────────┐
│  Traditional Keyword Search             │
│  Looks for exact words: "improve",     │
│  "sleep", "quality"                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
        ❌ Misses documents about:
           "better rest habits"
           "insomnia solutions"
           "sleeping disorder treatments"
        
        ❌ Returns documents with:
           "improve video quality while sleeping" (wrong!)
```

### Our Semantic Search

```
User Query: "How do I improve my sleep quality?"

┌─────────────────────────────────────────┐
│  Semantic Search (Dense Retrieval)      │
│  Understands MEANING, not just words    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
        ✅ Finds documents about:
           "better rest habits" ← Different words, same meaning!
           "insomnia solutions" ← Related concept
           "sleeping disorder treatments" ← Synonymous topic
        
        ✅ Ignores:
           "improve video quality while sleeping" ← Wrong context
```

## 🔄 How It Works: The Complete Flow

### Phase 1: Building the Index (One-Time Setup)

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: GET DATA                                                 │
└─────────────────────────────────────────────────────────────────┘

    Internet
       │
       ▼
  [MS MARCO Dataset]
  8,841,823 passages
       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: CLEAN DATA                                               │
└─────────────────────────────────────────────────────────────────┘

  Raw: "The  cat     sat on   the mat."
       │
       ▼ [TextPreprocessor]
       │
  Clean: "The cat sat on the mat."

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: CONVERT TO NUMBERS (Embeddings)                         │
└─────────────────────────────────────────────────────────────────┘

  Text: "The cat sat on the mat."
       │
       ▼ [Neural Network Model]
       │   (all-MiniLM-L6-v2)
       │
  Vector: [0.23, -0.45, 0.67, 0.12, ..., 0.89]
          └──────── 384 numbers ─────────┘
  
  This vector "encodes" the meaning!

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: BUILD SEARCH INDEX                                      │
└─────────────────────────────────────────────────────────────────┘

  8.8M Vectors
       │
       ▼ [FAISS]
       │ (Facebook AI Similarity Search)
       │
  Searchable Index
  ✅ Saved to disk
```

### Phase 2: Searching (Fast!)

```
┌─────────────────────────────────────────────────────────────────┐
│ USER QUERY                                                       │
└─────────────────────────────────────────────────────────────────┘

  "How does machine learning work?"

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ EMBED QUERY                                                      │
└─────────────────────────────────────────────────────────────────┘

  [all-MiniLM-L6-v2]
       │
       ▼
  Query Vector: [0.34, -0.12, 0.78, ..., 0.45]

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ SEARCH INDEX                                                     │
└─────────────────────────────────────────────────────────────────┘

  Compare query vector with 8.8M document vectors
  Find most similar (cosine similarity)
  
  Similarity Scores:
  doc_1234567: 0.89 ← Very similar!
  doc_2345678: 0.85
  doc_3456789: 0.82
  ...

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVE TEXTS                                                   │
└─────────────────────────────────────────────────────────────────┘

  Map doc IDs back to actual text
  
  Result 1 (0.89): "Machine learning is a subset of AI..."
  Result 2 (0.85): "ML algorithms learn from data..."
  Result 3 (0.82): "Neural networks are inspired by..."
  
       │
       ▼
  
  Return to user! ⏱️ ~50-100ms
```

## 🎨 The Math Behind Embeddings (Simplified)

### What Are Embeddings?

Think of embeddings as coordinates in "meaning space":

```
         Dimension 2
              ↑
              |
         dog •    • puppy
              |
              |
    ──────────┼──────────────► Dimension 1
              |
              |         • car
              |
              |
```

- Words with similar meanings are **close together**
- Words with different meanings are **far apart**

In reality, we use 384 dimensions (not just 2), but the idea is the same!

### How Similarity Works

```
Query: "How does solar power work?"
       ↓
Vector: [0.2, 0.8, 0.1, ...]

Documents:
1. "Solar panels convert sunlight..." [0.19, 0.82, 0.09, ...] → Score: 0.89 ✅
2. "Wind turbines generate power..." [0.1, 0.5, 0.3, ...]   → Score: 0.65
3. "How to bake a cake..."          [-0.3, 0.1, 0.7, ...]   → Score: 0.12 ❌

Cosine Similarity = how "aligned" vectors are
- 1.0 = Identical meaning
- 0.5 = Somewhat related
- 0.0 = Completely unrelated
```

## 📊 System Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: USER INTERFACE                                          │
│ (scripts/retrieve.py, scripts/evaluate.py)                       │
│                                                                  │
│ What: Entry points for users                                    │
│ Why: Simple commands to use the system                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ LAYER 2: ORCHESTRATION                                           │
│ (DenseRetriever, Evaluator)                                      │
│                                                                  │
│ What: Coordinates multiple components                           │
│ Why: Simplifies complex workflows                               │
└─────────────┬────────────────────────┬───────────────────────────┘
              │                        │
┌─────────────▼────────┐    ┌─────────▼──────────┐
│ LAYER 3A: DATA       │    │ LAYER 3B: RETRIEVAL│
│ (Loader, Processor)  │    │ (Embedder, Store)  │
│                      │    │                    │
│ What: Load & clean   │    │ What: Embed & find │
│ Why: Prepare inputs  │    │ Why: Core search   │
└─────────────┬────────┘    └─────────┬──────────┘
              │                       │
┌─────────────▼───────────────────────▼──────────────────────────┐
│ LAYER 4: INFRASTRUCTURE                                          │
│ (FAISS, sentence-transformers, Hydra)                            │
│                                                                  │
│ What: External libraries and tools                              │
│ Why: Don't reinvent the wheel                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🔍 Code Flow Example

Let's trace what happens when you search:

```python
# 1. You type a query
query = "What causes earthquakes?"

# 2. Embedder converts it to a vector
embedder = Embedder(model_name="all-MiniLM-L6-v2")
query_vector = embedder.encode([query])
# Result: numpy array of shape (1, 384)

# 3. FAISS searches for similar vectors
vector_store = FAISSVectorStore.load("outputs/faiss_index", ...)
doc_ids, scores = vector_store.search(query_vector, top_k=10)
# Result: 
#   doc_ids = [["doc_123", "doc_456", ...]]
#   scores = [[0.89, 0.85, ...]]

# 4. Retriever gets the actual text
retriever = DenseRetriever(embedder, vector_store)
retriever.corpus_df = load_corpus()
result = retriever.retrieve_single(query, return_texts=True)
# Result:
#   {
#     "doc_ids": ["doc_123", "doc_456", ...],
#     "scores": [0.89, 0.85, ...],
#     "texts": ["Earthquakes occur when...", ...]
#   }

# 5. Display to user
for i, (doc_id, score, text) in enumerate(zip(...)):
    print(f"[{i+1}] Score: {score:.2f}")
    print(f"     {text[:200]}...")
```

## 📈 Performance Metrics Explained

### Recall@10

```
Question: "Out of ALL relevant documents, how many did we find in top-10?"

Example:
Relevant documents: {doc1, doc2, doc3, doc4, doc5}
Retrieved top-10: [doc1, doc2, doc6, doc7, doc8, doc9, doc10, doc11, doc12, doc13]

Found: doc1, doc2 (2 out of 5)
Recall@10 = 2/5 = 0.4 = 40%
```

### nDCG@10 (Normalized Discounted Cumulative Gain)

```
Question: "Are the best results ranked at the top?"

Example:
Perfect ranking: [doc1(★★★), doc2(★★), doc3(★), ...]  → nDCG = 1.0
Good ranking:    [doc1(★★★), doc3(★), doc2(★★), ...] → nDCG = 0.95
Bad ranking:     [doc5(☆), doc6(☆), doc1(★★★), ...]  → nDCG = 0.4

Higher nDCG = Better ranking quality
```

## 🎯 Use Case Example

### Scenario: Customer Support Chatbot

```
┌─────────────────────────────────────────────────────────────────┐
│ CUSTOMER QUESTION                                                │
└─────────────────────────────────────────────────────────────────┘

"My laptop won't charge. What should I do?"

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ ADAPTIVE RAG RETRIEVAL                                           │
└─────────────────────────────────────────────────────────────────┘

Searches knowledge base of 1 million support articles

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ TOP RESULTS                                                      │
└─────────────────────────────────────────────────────────────────┘

1. "Troubleshooting laptop charging issues" (Score: 0.92)
2. "Battery not charging solutions" (Score: 0.88)
3. "Power adapter troubleshooting" (Score: 0.85)

       │
       ▼

┌─────────────────────────────────────────────────────────────────┐
│ LLM GENERATES ANSWER                                             │
└─────────────────────────────────────────────────────────────────┘

GPT/Claude reads retrieved documents and answers:

"I understand your laptop isn't charging. Here are steps to resolve:
1. Check the power adapter connection...
2. Inspect the charging port for debris...
3. Try a different outlet...

Based on: [Article 1234], [Article 5678]"
```

### Benefits

- ✅ **Accurate**: Finds relevant support articles even with different wording
- ✅ **Fast**: Returns results in <100ms
- ✅ **Scalable**: Works with millions of articles
- ✅ **Contextual**: Understands "won't charge" = "not charging" = "charging problem"

## 🚀 Future Milestones Preview

### Milestone 2: Dynamic Chunking

```
Long Document:
┌─────────────────────────────────────────────────────────────────┐
│ Introduction... [500 words]                                      │
│ Methods... [1000 words]                                          │
│ Results... [800 words]                                           │
│ Conclusion... [300 words]                                        │
└─────────────────────────────────────────────────────────────────┘

Current (M1): Split into fixed 512-char chunks ← Not optimal!

Future (M2): Smart splitting based on:
- Semantic boundaries (paragraph breaks)
- Query type (short answer vs long explanation)
- Content structure (headings, sections)
```

### Milestone 3: Hybrid Retrieval

```
Current (M1): Only dense retrieval

Future (M3): Dense + Sparse
┌────────────────┐         ┌────────────────┐
│ Dense          │         │ Sparse (BM25)  │
│ (Semantic)     │         │ (Keywords)     │
│                │         │                │
│ "ML concepts"  │         │ Exact matches: │
│ "AI learning"  │         │ "machine"      │
└────────┬───────┘         └────────┬───────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
             Fusion Layer
             (Combine results)
                    │
                    ▼
             Best of both!
```

### Milestone 4: Adaptive Reranking

```
Current (M1): Return top-10 directly

Future (M4): Smart refinement
┌────────────────────────────────────┐
│ Dense Retrieval → Top 100          │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│ Reranker (Cross-Encoder)           │
│ More accurate, but slower          │
│ Only on top candidates             │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│ Final Top 10 (Higher Quality!)     │
└────────────────────────────────────┘

Adaptive: Decide when to use based on:
- Query complexity
- Confidence scores
- Latency requirements
```

## 💡 Key Takeaways

1. **Semantic Search**: Understands meaning, not just keywords
2. **Embeddings**: Convert text to numbers that encode meaning
3. **FAISS**: Fast search through millions of vectors
4. **Modular Design**: Each component has one job
5. **Configurable**: Easy to experiment without code changes
6. **Scalable**: Ready for production use cases

## 📚 Further Learning

- **[MILESTONE_1.md](MILESTONE_1.md)**: Detailed walkthrough
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical deep dive
- **[CONFIGURATION.md](CONFIGURATION.md)**: Customization guide

## 🎓 Understanding Check

After reading this guide, you should be able to answer:

1. ✅ What's the difference between keyword and semantic search?
2. ✅ What are embeddings and why do we use them?
3. ✅ How does the system find similar documents?
4. ✅ What happens when you run `build_index.py`?
5. ✅ What happens when you search for something?
6. ✅ What do Recall@10 and nDCG@10 mean?

If you can answer these, you're ready to use and extend the system! 🎉

