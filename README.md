# AxionRAG

![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![llm](https://img.shields.io/badge/LLM-llama--3.3--70b-brightgreen?style=flat-square&logo=meta&logoColor=white)
![groq](https://img.shields.io/badge/inference-Groq-orange?style=flat-square)
![vector db](https://img.shields.io/badge/vector_db-Qdrant-red?style=flat-square)
![embeddings](https://img.shields.io/badge/embeddings-CLIP-purple?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

**Agentic RAG with persistent memory, LLM routing, and semantic caching.**

AxionRAG is a production-grade Retrieval-Augmented Generation pipeline that goes beyond naive chunk-and-retrieve. A single LLM call routes every query into one of three strategies — direct answer, single-pass retrieval, or multi-hop decomposition — while short-term and long-term memory keep responses coherent and personalised across sessions.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Ingesting Documents](#ingesting-documents)
  - [Querying the Pipeline](#querying-the-pipeline)
- [LLM Calls Per Turn](#llm-calls-per-turn)
- [API Reference](#api-reference)
- [Tuning](#tuning)
- [Production Deployment](#production-deployment)
- [Contributing](#contributing)

---

## Features

- **LLM Router** — One Groq call classifies every query as `DIRECT`, `RAG`, or `DECOMPOSE`, eliminating unnecessary retrieval on conversational and factual queries the model already knows.
- **Semantic Cache** — Full answers are stored keyed by query embedding. Repeated or near-identical queries return instantly with zero LLM calls and zero retrieval.
- **Short-Term Memory (STM)** — Sliding-window conversation buffer that rewrites pronouns and elliptic references into standalone queries before routing.
- **Long-Term Memory (LTM)** — Per-user vector collection of extracted facts and preferences that personalises every response across sessions.
- **Multi-hop Decomposition** — Complex multi-part questions are decomposed into focused sub-questions, retrieved independently, answered individually, then synthesised into a single coherent response.
- **Multimodal Embeddings** — Both text and images are embedded via OpenAI CLIP, enabling mixed-content corpora.
- **Zero infrastructure** — Qdrant runs in-memory by default. No Docker, no server, no external dependencies to start.

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│  STM Query Rewriter             │  Resolves pronouns, makes query
│  (skipped on first turn)        │  self-contained. 1 LLM call.
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Semantic Cache                 │  Cosine similarity vs stored queries.
│  threshold: 0.92, TTL: 7 days   │  Hit → return answer. 0 LLM calls.
└───────────────┬─────────────────┘
                │ MISS
                ▼
┌─────────────────────────────────┐
│  LLM Router                     │  Single Groq call. Classifies query
│                                 │  and acts on it simultaneously.
└─────┬───────────┬───────────────┘
      │           │              │
   DIRECT        RAG         DECOMPOSE
      │           │              │
      │           ▼              ▼
      │   ┌──────────────┐  ┌────────────────────────┐
      │   │ Vector Search│  │ Generate sub-questions │
      │   │ (Qdrant HNSW)│  │ Retrieve per-sub       │
      │   │ top_k=5      │  │ Answer per-sub         │
      │   └──────┬───────┘  │ Synthesise all         │
      │          │          └───────────┬────────────┘
      │          ▼                      │
      │   ┌──────────────┐              │
      │   │  Synthesise  │              │
      │   │  (LTM + STM) │              │
      │   └──────┬───────┘              │
      └──────────┴──────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Write-back                     │
│  · Cache store                  │
│  · STM update                   │
│  · LTM fact extraction          │
└─────────────────────────────────┘
                 │
                 ▼
            Final Answer
```

---

## Stack

| Layer          | Technology                           | Notes                                                |
| -------------- | ------------------------------------ | ---------------------------------------------------- |
| **Embeddings** | CLIP `openai/clip-vit-large-patch14` | Text + image, 768-dim, runs locally                  |
| **Vector DB**  | Qdrant (HNSW, cosine)                | In-memory by default; one-line swap to disk or cloud |
| **LLM**        | Groq API — `llama-3.3-70b-versatile` | Fast cloud inference, free tier available            |
| **Router**     | LLM-as-router                        | `DIRECT` / `RAG` / `DECOMPOSE` in one call           |
| **STM**        | In-process sliding window            | Last 8 turns, with query rewriting                   |
| **LTM**        | Per-user Qdrant collection           | Extracted facts, persistent cross-session            |
| **Cache**      | Semantic cache (Qdrant)              | Cosine ≥ 0.92, TTL 7 days                            |

---

## Project Structure

```
AxionRAG/
├── rag_memory_core.py   # Core pipeline — all classes, embedding, routing, memory
├── ingest.py            # CLI tool: chunk and index documents into the pipeline
├── example_usage.py     # 8-turn demo exercising all router paths
├── requirements.txt     # Python dependencies
├── config.env           # Environment variable template (copy to .env)
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) — sign up, go to **API Keys → Create key**

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Rahul-14507/AxionRAG.git
cd AxionRAG
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The CLIP model (~1.7 GB) is downloaded from HuggingFace automatically on first run and cached in `~/.cache/huggingface`.

### 3. Configure environment variables

```bash
cp config.env .env
```

Open `.env` and fill in your key:

```env
GROQ_API_KEY=gsk_your_key_here
```

Load the environment:

```bash
# macOS / Linux
export $(grep -v '^#' .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#][^=]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

### 4. Run the demo

```bash
python example_usage.py
```

The demo indexes 11 document chunks and runs an 8-turn conversation exercising all three router paths: `DIRECT`, `RAG`, and `DECOMPOSE`, plus an STM rewrite and a semantic cache hit.

---

## Configuration

All parameters are controlled via environment variables or constants in `rag_memory_core.py`.

| Variable              | Default                         | Description                                         |
| --------------------- | ------------------------------- | --------------------------------------------------- |
| `GROQ_API_KEY`        | _(required)_                    | Groq API key                                        |
| `CLIP_MODEL_ID`       | `openai/clip-vit-large-patch14` | HuggingFace CLIP model ID                           |
| `CACHE_SIM_THRESHOLD` | `0.92`                          | Minimum cosine similarity for a cache hit           |
| `CACHE_TTL_SECONDS`   | `604800` (7 days)               | Cache entry lifetime                                |
| `STM_WINDOW`          | `8`                             | Conversation turns retained in short-term memory    |
| `LTM_TOP_K`           | `3`                             | User facts retrieved per query                      |
| `RAG_TOP_K`           | `5`                             | Document chunks retrieved per RAG query             |
| `SUB_RAG_TOP_K`       | `3`                             | Chunks retrieved per sub-question in DECOMPOSE path |
| `DECOMP_MAX_SUBQ`     | `4`                             | Hard cap on sub-questions                           |

---

## Usage

### Ingesting Documents

`ingest.py` is a CLI tool that reads documents, splits them into overlapping token-aware chunks, and indexes them into the pipeline's `DocumentStore`.

**Supported formats:** `.pdf`, `.docx`, `.txt`, `.md`

```bash
# Index a folder
python ingest.py --source ./docs --user alice

# Index a single file
python ingest.py --source ./manual.pdf --user alice

# Custom chunk size and overlap
python ingest.py --source ./docs --chunk-tokens 512 --overlap-tokens 64 --user alice
```

| Flag               | Default      | Description                                     |
| ------------------ | ------------ | ----------------------------------------------- |
| `--source`         | _(required)_ | Path to a file or directory                     |
| `--user`           | `default`    | User ID — controls which LTM collection is used |
| `--chunk-tokens`   | `256`        | Target chunk size in tokens                     |
| `--overlap-tokens` | `32`         | Overlap between consecutive chunks              |

Chunks are content-addressed using an MD5 hash of `(file path + chunk index + first 64 chars)`. Re-ingesting the same file is safe and idempotent.

---

### Querying the Pipeline

```python
from rag_memory_core import RAGMemoryPipeline

pipeline = RAGMemoryPipeline(user_id="alice")

# Index some text directly
pipeline.index("doc_001", "TurboQuant uses HNSW indexing...", metadata={"source": "docs/db.md"})

# Index an image
with open("diagram.png", "rb") as f:
    pipeline.index_image("img_001", f.read(), caption="System architecture diagram")

# Ask a question — the router, cache, STM, and LTM all fire automatically
answer = pipeline.query("What indexing method does TurboQuant use?")
print(answer)
```

The pipeline is stateful. Each `query()` call updates STM, extracts facts into LTM, and stores the answer in the semantic cache.

---

## LLM Calls Per Turn

| Path                            | LLM calls | Triggered when                                                    |
| ------------------------------- | --------- | ----------------------------------------------------------------- |
| **Cache hit**                   | 0         | Query cosine ≥ 0.92 to a stored query                             |
| **DIRECT**                      | 1         | Greetings, math, general knowledge — no document lookup needed    |
| **RAG**                         | 2         | Single focused document lookup (+ 1 for STM rewrite after turn 1) |
| **DECOMPOSE (N sub-questions)** | 2 + N     | Multi-part or comparative questions                               |

The STM rewrite adds +1 LLM call on every turn after the first. It is skipped on the first turn.

---

## API Reference

### `RAGMemoryPipeline`

The main entrypoint. All state is encapsulated inside a single instance.

```python
pipeline = RAGMemoryPipeline(user_id="alice")
```

| Method        | Signature                                            | Description                                  |
| ------------- | ---------------------------------------------------- | -------------------------------------------- |
| `index`       | `(chunk_id: str, text: str, metadata: dict \| None)` | Embeds and indexes a text chunk              |
| `index_image` | `(chunk_id: str, image_bytes: bytes, caption: str)`  | Embeds and indexes an image chunk            |
| `query`       | `(user_input: str) → str`                            | Runs the full pipeline and returns an answer |

---

### `ShortTermMemory`

Sliding-window conversation buffer.

| Method                     | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `add(role, content)`       | Appends a turn to the buffer                          |
| `rewrite_query(raw_query)` | Rewrites the query as a standalone question using LLM |
| `format_for_prompt()`      | Returns formatted conversation history                |

---

### `LongTermMemory`

Per-user persistent fact store backed by Qdrant.

| Method                            | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `store_fact(fact)`                | Embeds and upserts a fact into the user's collection |
| `retrieve(query, top_k)`          | Returns the top-k most relevant facts                |
| `extract_and_store(user_message)` | Calls LLM to extract memorable facts from a turn     |

---

### `SemanticCache`

Full-answer cache keyed by query embedding.

| Method                               | Description                                                            |
| ------------------------------------ | ---------------------------------------------------------------------- |
| `lookup(query) → CacheEntry \| None` | Returns a cached answer if cosine ≥ threshold and entry is not expired |
| `store(query, answer, chunk_ids)`    | Stores a new cache entry                                               |

---

### `DocumentStore`

Vector store for document chunks.

| Method                                                  | Description                              |
| ------------------------------------------------------- | ---------------------------------------- |
| `add_text(chunk_id, text, metadata)`                    | Embeds and upserts a text chunk          |
| `add_image(chunk_id, image_bytes, caption, metadata)`   | Embeds and upserts an image chunk        |
| `retrieve(query, top_k) → list[tuple[str, str, float]]` | Returns `[(chunk_id, text, score), ...]` |
| `fetch(chunk_ids) → list[tuple[str, str]]`              | Fetches raw text by chunk ID             |

---

## Tuning

### Raising cache precision

Increase `CACHE_SIM_THRESHOLD` (e.g. `0.95`) to only hit the cache on near-verbatim repeats. Lower it (e.g. `0.88`) for more aggressive caching on paraphrase variants.

### Adjusting retrieval breadth

Increase `RAG_TOP_K` for longer, more context-rich synthesis prompts. Increase `SUB_RAG_TOP_K` for deeper sub-question answers in the DECOMPOSE path. Both increase latency and token usage.

### LLM temperature

`llm_call()` uses `temperature=0.2` for deterministic routing, retrieval query generation, and sub-question answering. If you want more expressive final synthesis answers, add a `temperature` parameter to `_synthesise()` and pass `0.7`.

### Groq rate limits

The free tier allows **6,000 tokens/min** on `llama-3.3-70b-versatile`. Add exponential backoff for high-throughput use:

```python
import groq, time

def llm_call_with_retry(prompt, system="", max_tokens=1024, max_attempts=4):
    for attempt in range(max_attempts):
        try:
            return llm_call(prompt, system=system, max_tokens=max_tokens)
        except groq.RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)
```

---

## Production Deployment

### Qdrant persistence

By default, Qdrant runs in-memory and all data is lost when the process exits. To persist data across restarts, change one line in `_get_qdrant()`:

```python
# On-disk persistence (no server required)
_qdrant = QdrantClient(path="./qdrant_data")

# Remote Qdrant server or Qdrant Cloud
_qdrant = QdrantClient(host="localhost", port=6333)
_qdrant = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-qdrant-key")
```

### SemanticCache persistence

`SemanticCache._meta` is an in-process Python dict. For multi-process or multi-restart deployments, replace it with a Redis or SQLite-backed store keyed by the Qdrant point ID.

### LongTermMemory at scale

Each user gets their own Qdrant collection (`ltm_{user_id}`). This is practical up to ~100k users on a single node. For larger user bases, shard by user ID prefix or use namespaced collections on a Qdrant cluster.

### Async and batching

For high-throughput scenarios:

- Batch Groq calls for the DECOMPOSE sub-question answer step using `asyncio.gather`.
- Batch CLIP embedding calls for bulk document ingestion.
- Use `AsyncQdrantClient` to avoid blocking the event loop on vector operations.

### CLIP model size

The default `openai/clip-vit-large-patch14` model is ~1.7 GB and uses 768-dim vectors. For lower-memory deployments, switch to `openai/clip-vit-base-patch32` (340 MB, 512-dim) by setting `CLIP_MODEL_ID=openai/clip-vit-base-patch32`. Reindex all documents after changing the model — vector dimensions must match the collection.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes. Keep new classes and functions in `rag_memory_core.py` unless the change is a standalone CLI tool.
3. Test against `example_usage.py` and ensure all three router paths produce sensible output.
4. Open a pull request with a clear description of the change and why it improves the system.

---

## License

MIT
