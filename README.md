# RAG + Memory + LLM Router

A production-grade RAG system with short-term memory, long-term memory, semantic caching, and an LLM router that eliminates unnecessary retrieval.

---

## Architecture

```
User query
    │
    ▼
STM Query Rewriter          — resolves pronouns, makes query standalone
    │
    ▼
Semantic Cache              — returns stored answer if cosine ≥ 0.92 (zero LLM on hit)
    │ MISS
    ▼
LLM Router  ──────────────────────────────────────────────┐
    │                                                      │
    ├── DIRECT                  ├── RAG                   └── DECOMPOSE
    │   Answer immediately      │   Single-pass retrieve       Sub-questions →
    │   (no retrieval)          │   → LLM synthesis            per-sub retrieve →
    │                           │                              synthesise
    └───────────────────────────┴──────────────────────────────────┘
                                        │
                                        ▼
                               LTM personalisation
                                        │
                                        ▼
                               Memory write-back
                         (cache + STM + LTM fact extraction)
```

### LLM calls per turn

| Path | LLM calls |
|---|---|
| Cache hit | 0 |
| DIRECT | 1 (router = answer) |
| RAG | 2 (router + synthesis) |
| DECOMPOSE (N sub-questions) | 2+N (router + sub-answers + synthesis) |

All paths include +1 for STM query rewriting on turns after the first.

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | CLIP (`openai/clip-vit-large-patch14`) — fully local, text + image |
| Vector DB | Qdrant (HNSW, cosine similarity) — in-memory by default |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Router | LLM-as-router (DIRECT / RAG / DECOMPOSE) |
| STM | In-process sliding window (last 8 turns) |
| LTM | Per-user TurboQuant collection (extracted facts) |
| Cache | Semantic cache (TurboQuant, cosine ≥ 0.92, TTL 7d) |

---

## Setup

### Quick start

```bash
pip install -r requirements.txt
cp config.env .env   # add GROQ_API_KEY
export $(grep -v '^#' .env | xargs)
python ingest.py --source ./your_docs --user my_user
python example_usage.py
```

### Detailed steps

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Set environment variables
```bash
cp config.env .env
# Edit .env — add your GROQ_API_KEY
export $(grep -v '^#' .env | xargs)
```

#### 3. Get a Groq API key
Sign up free at https://console.groq.com → API Keys → Create key.
Add it to your .env file as `GROQ_API_KEY`.

#### 4. Ingest documents
```bash
# Index a folder of PDFs, docx, txt, md files
python ingest.py --source ./your_docs --user my_user

# Index a single file
python ingest.py --source ./manual.pdf --user my_user

# Custom chunk size
python ingest.py --source ./docs --chunk-tokens 512 --overlap-tokens 64
```

#### 5. Run the demo
```bash
python example_usage.py
```

---

## Files

```
rag_memory_core.py   Core pipeline — all classes and logic
ingest.py            CLI tool to chunk and index documents
example_usage.py     8-turn demo exercising all router paths
requirements.txt     Python dependencies
config.env           Environment variable template
README.md            This file
```

---

## Key classes

### `RAGMemoryPipeline`
Main entrypoint. One public method: `pipeline.query(user_input) → str`.

```python
pipeline = RAGMemoryPipeline(user_id="alice")
pipeline.index("chunk_001", "TurboQuant uses HNSW indexing...")
answer = pipeline.query("What indexing method does TurboQuant use?")
```

### `LLMRouter`
Single LLM call that classifies the query and prepares the retrieval strategy.
Output format enforced via a strict system prompt. Falls back to RAG on parse failure.

### `ShortTermMemory`
Sliding window of last N turns. `rewrite_query()` resolves pronouns before routing.

### `LongTermMemory`
Per-user TurboQuant collection. Facts are extracted from every user turn and stored
as embeddings. Retrieved at query time for personalisation.

### `SemanticCache`
Stores full answers keyed by query embedding. On a hit: zero retrieval, zero LLM.
TTL eviction prevents stale answers.

### `QueryDecomposer`
Handles the DECOMPOSE path. Retrieves independently per sub-question, answers each,
then synthesises into one final response.

### `DocumentStore`
Wraps TurboQuant for document indexing and retrieval. Supports text and image chunks
(multimodal via CLIP).

---

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `CACHE_SIM_THRESHOLD` | 0.92 | Lower → more cache hits, risk stale answers |
| `CACHE_TTL_SECONDS` | 604800 (7d) | Lower → fresher answers, lower hit rate |
| `STM_WINDOW` | 8 | Higher → more context, longer prompts |
| `RAG_TOP_K` | 5 | Higher → more context, higher latency |
| `SUB_RAG_TOP_K` | 3 | Chunks per sub-question in DECOMPOSE path |
| `DECOMP_MAX_SUBQ` | 4 | Hard cap on sub-questions |

---

## Production notes

- CLIP model (~1.7GB) is downloaded from HuggingFace on first run and cached
  in `~/.cache/huggingface`. Set `CLIP_MODEL_ID` env var to use a different
  CLIP variant (e.g. `openai/clip-vit-base-patch32` for lower memory usage).
- `SemanticCache._meta` is in-process RAM — replace with Redis or SQLite for persistence across restarts.
- `DocumentStore._chunks` is in-process RAM — replace with a database for large corpora.
- `LongTermMemory` uses a Qdrant collection per user — fine up to ~100k users; shard beyond that.
- Qdrant runs in-memory by default (no server needed). For persistence across
  restarts, change `QdrantClient(":memory:")` to `QdrantClient(path="./qdrant_data")`
  in `_get_qdrant()`. For a remote Qdrant cluster, use
  `QdrantClient(host="localhost", port=6333)` or the Qdrant Cloud URL.
- Groq enforces rate limits on the free tier (6000 tokens/min on llama-3.3-70b-versatile).
  For high-throughput production use, add exponential backoff around `llm_call()`:
  catch `groq.RateLimitError` and retry with `time.sleep(2 ** attempt)`.
- CLIP still runs fully locally — Groq only handles LLM inference.
- For high-throughput production use, batch the embedding calls to CLIP and run sub-question retrievals concurrently with `asyncio.gather`.
