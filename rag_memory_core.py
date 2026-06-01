import os
import re
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    SearchRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GROQ_MODEL          = "llama-3.3-70b-versatile"

CLIP_MODEL_ID       = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
EMBED_DIM           = 768

CACHE_SIM_THRESHOLD = 0.92          # cosine similarity for a cache hit
CACHE_TTL_SECONDS   = 7 * 24 * 3600 # 7-day TTL on cache entries

STM_WINDOW          = 8             # conversation turns kept in short-term memory
LTM_TOP_K           = 3             # user facts fetched per query
RAG_TOP_K           = 5             # document chunks fetched per query
SUB_RAG_TOP_K       = 3             # chunks fetched per sub-question
DECOMP_MAX_SUBQ     = 4             # hard cap on sub-questions

# ── CLIP singleton — loaded once, reused across all embed calls ────────
_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        log.info("Loading CLIP model: %s", CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
    return _clip_model, _clip_processor

# ── Groq client singleton ──────────────────────────────────────────────
_groq_client: Groq | None = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


@dataclass
class Turn:
    role: str       # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class CacheEntry:
    query_text: str
    answer: str             # store the full answer, not just chunk IDs
    chunk_ids: list[str]
    timestamp: float

@dataclass
class RouterDecision:
    """Parsed output of the LLM router call."""
    action: str             # "DIRECT" | "RAG" | "DECOMPOSE"
    direct_answer: str = "" # populated when action == DIRECT
    rag_query: str = ""     # populated when action == RAG (may differ from original)
    sub_questions: list[str] = field(default_factory=list)  # DECOMPOSE

@dataclass
class SubQuestion:
    question: str
    answer: str = ""
    chunk_ids: list[str] = field(default_factory=list)

# ──────────────────────────────────────────────────────────────────────────────
# Embedding helpers  (CLIP — multimodal)
# ──────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    model, processor = _get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vec = model.get_text_features(**inputs)
    if hasattr(vec, "pooler_output"):
        vec = vec.pooler_output
    elif hasattr(vec, "last_hidden_state"):
        vec = vec.last_hidden_state[:, 0, :]
    vec = vec / vec.norm(dim=-1, keepdim=True)   # L2 normalise for cosine
    return vec.squeeze().numpy().astype(np.float32)

def embed_image(image_bytes: bytes) -> np.ndarray:
    import PIL.Image, io
    model, processor = _get_clip()
    img    = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vec = model.get_image_features(**inputs)
    if hasattr(vec, "pooler_output"):
        vec = vec.pooler_output
    elif hasattr(vec, "last_hidden_state"):
        vec = vec.last_hidden_state[:, 0, :]
    vec = vec / vec.norm(dim=-1, keepdim=True)   # L2 normalise for cosine
    return vec.squeeze().numpy().astype(np.float32)

# ── Qdrant client singleton (in-memory) ───────────────────────────────
_qdrant: QdrantClient | None = None

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(":memory:")
        log.info("Qdrant in-memory client initialised")
    return _qdrant

def make_collection(name: str, dim: int = EMBED_DIM):
    """Creates the collection if it doesn't exist. Returns the collection name."""
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name = name,
            vectors_config  = VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection: %s (dim=%d)", name, dim)
    return name

# ──────────────────────────────────────────────────────────────────────────────
# LLM wrapper  (Groq)
# ──────────────────────────────────────────────────────────────────────────────

def llm_call(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _get_groq().chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = 0.2,        # low temp for deterministic routing + retrieval
    )
    return resp.choices[0].message.content.strip()

def llm_short(prompt: str, max_tokens: int = 256) -> str:
    """Lightweight wrapper for short utility calls — rewriting, fact extraction."""
    return llm_call(prompt, max_tokens=max_tokens)

# ──────────────────────────────────────────────────────────────────────────────
# Short-term memory
# ──────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """Sliding-window conversation buffer with query rewriter."""

    def __init__(self, window: int = STM_WINDOW):
        self.window = window
        self._turns: list[Turn] = []

    def add(self, role: str, content: str):
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self.window:
            self._turns = self._turns[-self.window:]

    def format_for_prompt(self) -> str:
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in self._turns)

    def rewrite_query(self, raw_query: str) -> str:
        """
        Resolves pronouns and elliptic references so the query can be
        understood without conversation context. Skipped on first turn.
        """
        history = self.format_for_prompt()
        if not history:
            return raw_query
        prompt = (
            "Given the conversation history, rewrite the LAST USER message as a "
            "fully self-contained question. Output ONLY the rewritten question.\n\n"
            f"History:\n{history}\n\nLast message: {raw_query}\nRewritten:"
        )
        return llm_short(prompt, max_tokens=128).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Long-term memory
# ──────────────────────────────────────────────────────────────────────────────

class LongTermMemory:
    """Per-user vector store of extracted facts and preferences."""

    def __init__(self, user_id: str):
        self.col = make_collection(f"ltm_{user_id}")

    def store_fact(self, fact: str):
        vec = embed_text(fact)
        fid = hashlib.md5(fact.encode()).hexdigest()
        _get_qdrant().upsert(
            collection_name = self.col,
            points = [PointStruct(
                id      = int(fid, 16) % (2**63),
                vector  = vec.tolist(),
                payload = {"_str_id": fid, "fact": fact, "ts": time.time()}
            )]
        )
        log.info("LTM stored: %s", fact)

    def retrieve(self, query: str, top_k: int = LTM_TOP_K) -> list[str]:
        vec = embed_text(query)
        results = _get_qdrant().query_points(
            collection_name = self.col,
            query           = vec.tolist(),
            limit           = top_k,
            with_payload    = True,
        ).points
        return [r.payload["fact"] for r in results]

    def extract_and_store(self, user_message: str):
        """Extract memorable facts from a user turn and persist them."""
        prompt = (
            "Extract factual statements about the user's preferences, goals, or background "
            "from the message below. One fact per line, no bullets. "
            "If nothing worth remembering, output exactly: NONE\n\n"
            f"Message: {user_message}"
        )
        raw = llm_short(prompt, max_tokens=200).strip()
        if raw.upper() == "NONE" or not raw:
            return
        for line in raw.splitlines():
            line = line.strip()
            if line:
                self.store_fact(line)

# ──────────────────────────────────────────────────────────────────────────────
# Semantic cache
# ──────────────────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Stores full answers (not just chunk IDs) keyed by query embedding.
    On a hit the answer is returned directly — zero retrieval, zero LLM call.
    """

    def __init__(self):
        self.col   = make_collection("query_cache")
        self._meta: dict[str, CacheEntry] = {}   # persist to SQLite/Redis in production

    def lookup(self, query: str) -> Optional[CacheEntry]:
        vec     = embed_text(query)
        results = _get_qdrant().query_points(
            collection_name = self.col,
            query           = vec.tolist(),
            limit           = 1,
            with_payload    = True,
        ).points
        if not results:
            return None
        best  = results[0]
        score = best.score
        if score < CACHE_SIM_THRESHOLD:
            return None
        str_id = best.payload.get("_str_id")
        if not str_id:
            return None
        entry = self._meta.get(str_id)
        if entry is None:
            return None
        if time.time() - entry.timestamp > CACHE_TTL_SECONDS:
            log.info("Cache entry expired — evicting")
            from qdrant_client.models import PointIdsList
            _get_qdrant().delete(
                collection_name = self.col,
                points_selector = PointIdsList(points=[best.id]),
            )
            del self._meta[str_id]
            return None
        log.info("Cache HIT (score=%.3f): %s", score, query[:60])
        return entry

    def store(self, query: str, answer: str, chunk_ids: list[str]):
        vec = embed_text(query)
        eid = hashlib.md5(query.encode()).hexdigest()
        _get_qdrant().upsert(
            collection_name = self.col,
            points = [PointStruct(
                id      = int(eid, 16) % (2**63),
                vector  = vec.tolist(),
                payload = {"_str_id": eid, "query": query}
            )]
        )
        self._meta[eid] = CacheEntry(
            query_text=query, answer=answer,
            chunk_ids=chunk_ids, timestamp=time.time(),
        )
        log.info("Cache stored: %s", query[:60])

# ──────────────────────────────────────────────────────────────────────────────
# Document store
# ──────────────────────────────────────────────────────────────────────────────

class DocumentStore:
    """Indexes text and image chunks via CLIP."""

    def __init__(self):
        self.col     = make_collection("documents")
        self._chunks: dict[str, str] = {}   # id → text (replace with DB in production)

    def add_text(self, chunk_id: str, text: str, metadata: dict | None = None):
        vec = embed_text(text)
        _get_qdrant().upsert(
            collection_name = self.col,
            points = [PointStruct(
                id      = abs(hash(chunk_id)) % (2**63),
                vector  = vec.tolist(),
                payload = {"_str_id": chunk_id, "text": text, **(metadata or {})}
            )]
        )
        self._chunks[chunk_id] = text

    def add_image(self, chunk_id: str, image_bytes: bytes, caption: str = "", metadata: dict | None = None):
        vec = embed_image(image_bytes)
        _get_qdrant().upsert(
            collection_name = self.col,
            points = [PointStruct(
                id      = abs(hash(chunk_id)) % (2**63),
                vector  = vec.tolist(),
                payload = {"_str_id": chunk_id, "caption": caption, "is_image": True, **(metadata or {})}
            )]
        )
        self._chunks[chunk_id] = f"[IMAGE] {caption}"

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[tuple[str, str, float]]:
        """Returns [(chunk_id, text, score), ...]"""
        vec     = embed_text(query)
        results = _get_qdrant().query_points(
            collection_name = self.col,
            query           = vec.tolist(),
            limit           = top_k,
            with_payload    = True,
        ).points
        return [
            (r.payload.get("_str_id", str(r.id)),
             r.payload.get("text") or r.payload.get("caption", ""),
             r.score)
            for r in results
        ]

    def fetch(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        return [(cid, self._chunks.get(cid, "")) for cid in chunk_ids]

# ──────────────────────────────────────────────────────────────────────────────
# LLM Router  — the heart of the new architecture
# ──────────────────────────────────────────────────────────────────────────────

# Strict output format the router must follow
_ROUTER_SYSTEM = """You are a query router for a RAG assistant. Given a user query, decide the best action and respond in EXACTLY one of these three formats — no other text:

DIRECT: <your complete answer here>
  Use when: the query is conversational (greeting, thanks, simple math, general knowledge the LLM knows well, no document lookup needed).

RAG: <rewritten retrieval query>
  Use when: the query needs information from documents but is a single focused question. Output a clean retrieval query, not the original message.

DECOMPOSE: <sub-question 1> | <sub-question 2> | <sub-question 3>
  Use when: the query is multi-part, comparative, or requires chaining multiple facts. Break into the minimum number of focused sub-questions (max 4), pipe-separated.

Rules:
- Prefer DIRECT for anything you can answer confidently without documents.
- Prefer RAG over DECOMPOSE unless the query genuinely needs multiple independent retrievals.
- Never output anything outside the three formats above.
- For DECOMPOSE, each sub-question must be self-contained and retrievable independently."""

_DIRECT_RE   = re.compile(r'^DIRECT:\s*(.+)', re.S)
_RAG_RE      = re.compile(r'^RAG:\s*(.+)',    re.S)
_DECOMP_RE   = re.compile(r'^DECOMPOSE:\s*(.+)', re.S)

def parse_router_response(raw: str) -> RouterDecision:
    """
    Parse the router's structured response into a RouterDecision.
    Falls back to RAG with the original query if the format is unexpected
    so we never silently drop a query.
    """
    raw = raw.strip()

    m = _DIRECT_RE.match(raw)
    if m:
        return RouterDecision(action="DIRECT", direct_answer=m.group(1).strip())

    m = _RAG_RE.match(raw)
    if m:
        return RouterDecision(action="RAG", rag_query=m.group(1).strip())

    m = _DECOMP_RE.match(raw)
    if m:
        sub_qs = [q.strip() for q in m.group(1).split("|") if q.strip()]
        return RouterDecision(action="DECOMPOSE", sub_questions=sub_qs[:DECOMP_MAX_SUBQ])

    # Fallback — malformed output → safe RAG
    log.warning("Router returned unexpected format — falling back to RAG: %s", raw[:80])
    return RouterDecision(action="RAG", rag_query=raw)


class LLMRouter:
    """
    Single LLM call that simultaneously:
      - decides whether retrieval is needed at all (DIRECT)
      - decides retrieval mode (RAG vs DECOMPOSE)
      - for DIRECT: produces the answer immediately
      - for RAG: produces a clean retrieval query
      - for DECOMPOSE: produces ready-to-retrieve sub-questions

    This replaces both the old complexity check AND the old query rewriter
    for the routing step — one call does the work of two.
    """

    def route(self, query: str, history: str, ltm_facts: list[str]) -> RouterDecision:
        context_parts = []
        if ltm_facts:
            context_parts.append("User facts: " + "; ".join(ltm_facts))
        if history:
            context_parts.append(f"Conversation so far:\n{history}")
        context_parts.append(f"Query: {query}")

        prompt = "\n\n".join(context_parts)
        raw    = llm_call(prompt, system=_ROUTER_SYSTEM, max_tokens=512)
        log.info("Router raw output: %s", raw[:120])
        decision = parse_router_response(raw)
        log.info("Router decision: %s", decision.action)
        return decision

# ──────────────────────────────────────────────────────────────────────────────
# Query Decomposer  (used only when router returns DECOMPOSE)
# ──────────────────────────────────────────────────────────────────────────────

class QueryDecomposer:
    """
    Handles the DECOMPOSE path: per-sub-question retrieve + answer,
    then synthesise into one final answer.
    """

    def __init__(self, docs: DocumentStore):
        self.docs = docs

    def _answer_sub(self, sub_q: str) -> SubQuestion:
        results     = self.docs.retrieve(sub_q, top_k=SUB_RAG_TOP_K)
        chunk_texts = [t for _, t, _ in results]
        chunk_ids   = [c for c, _, _ in results]
        context     = "\n\n".join(chunk_texts) if chunk_texts else "No context found."
        prompt      = (
            f"Answer the question below using only the context provided. "
            f"Be concise.\n\nContext:\n{context}\n\nQuestion: {sub_q}\nAnswer:"
        )
        answer = llm_short(prompt, max_tokens=512)
        return SubQuestion(question=sub_q, answer=answer, chunk_ids=chunk_ids)

    def run(self, original_query: str, sub_questions: list[str],
            ltm_block: str = "", history_block: str = "") -> tuple[str, list[str]]:
        """
        Returns (final_synthesised_answer, all_chunk_ids_used).
        """
        log.info("Decomposing into %d sub-questions", len(sub_questions))
        sub_results = [self._answer_sub(sq) for sq in sub_questions]

        qa_pairs = "\n\n".join(
            f"Sub-question: {sr.question}\nAnswer: {sr.answer}"
            for sr in sub_results
        )

        prompt_parts = []
        if ltm_block:
            prompt_parts.append(ltm_block)
        if history_block:
            prompt_parts.append(f"Conversation so far:\n{history_block}")
        prompt_parts.append(
            f"You answered the following sub-questions to address a complex query.\n"
            f"Synthesise them into a single coherent answer to the original question.\n\n"
            f"{qa_pairs}\n\n"
            f"Original question: {original_query}\nSynthesised answer:"
        )

        final    = llm_call("\n\n".join(prompt_parts), max_tokens=1024)
        all_ids  = list({cid for sr in sub_results for cid in sr.chunk_ids})
        return final, all_ids

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGMemoryPipeline:
    """
    Full pipeline — one public method: query(user_input) → answer.

    LLM calls per turn:
      DIRECT query   → 1  (router = answer)
      Simple RAG     → 2  (router + synthesis)     [+ 1 if STM rewrite needed]
      Complex DECOMP → 2+N (router + N sub-answers + synthesis)

    All paths benefit from the semantic cache which can reduce any of
    the above to 0 LLM calls + 0 retrieval on repeat queries.
    """

    def __init__(self, user_id: str = "default"):
        self.stm        = ShortTermMemory(window=STM_WINDOW)
        self.ltm        = LongTermMemory(user_id=user_id)
        self.cache      = SemanticCache()
        self.docs       = DocumentStore()
        self.router     = LLMRouter()
        self.decomposer = QueryDecomposer(docs=self.docs)

    # ── Public: indexing ────────────────────────────────────────────────

    def index(self, chunk_id: str, text: str, metadata: dict | None = None):
        self.docs.add_text(chunk_id, text, metadata)

    def index_image(self, chunk_id: str, image_bytes: bytes, caption: str = ""):
        self.docs.add_image(chunk_id, image_bytes, caption)

    # ── Public: query ───────────────────────────────────────────────────

    def query(self, user_input: str) -> str:
        log.info("─── query: %s", user_input[:80])

        # Step 1 — STM query rewrite (resolves pronouns / ellipsis)
        standalone = self.stm.rewrite_query(user_input)

        # Step 2 — Semantic cache (full answer stored, so zero LLM on hit)
        cached = self.cache.lookup(standalone)
        if cached:
            log.info("Returning cached answer")
            self._write_back(user_input, cached.answer, cached.chunk_ids, cache_hit=True)
            return cached.answer

        # Step 3 — LLM Router (classify + act in one call)
        history_block = self.stm.format_for_prompt()
        ltm_facts     = self.ltm.retrieve(standalone)
        ltm_block     = ("Known facts about this user:\n" + "\n".join(f"- {f}" for f in ltm_facts)) if ltm_facts else ""

        decision = self.router.route(standalone, history_block, ltm_facts)

        # ── Branch A: DIRECT — no retrieval needed ──────────────────────
        if decision.action == "DIRECT":
            log.info("DIRECT answer — no retrieval")
            answer      = decision.direct_answer
            chunk_ids   = []

        # ── Branch B: RAG — single-pass retrieval ───────────────────────
        elif decision.action == "RAG":
            retrieval_query = decision.rag_query or standalone
            results         = self.docs.retrieve(retrieval_query, top_k=RAG_TOP_K)
            chunk_texts     = [t for _, t, _ in results]
            chunk_ids       = [c for c, _, _ in results]
            answer          = self._synthesise(user_input, chunk_texts, ltm_block, history_block)

        # ── Branch C: DECOMPOSE — agentic multi-hop ─────────────────────
        else:
            answer, chunk_ids = self.decomposer.run(
                standalone, decision.sub_questions, ltm_block, history_block
            )

        # Step 4 — Write-back (cache, STM, LTM)
        self._write_back(user_input, answer, chunk_ids)
        return answer

    # ── Private helpers ─────────────────────────────────────────────────

    def _synthesise(self, question: str, chunk_texts: list[str],
                    ltm_block: str, history_block: str) -> str:
        context = "\n\n".join(chunk_texts) if chunk_texts else "No relevant context found."
        system  = (
            "You are a helpful assistant. Use the context, conversation history, "
            "and user facts to give accurate, personalised answers. "
            "If the context doesn't contain the answer, say so honestly."
        )
        parts = []
        if ltm_block:     parts.append(ltm_block)
        if history_block: parts.append(f"Conversation so far:\n{history_block}")
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {question}\nAnswer:")
        return llm_call("\n\n".join(parts), system=system)

    def _write_back(self, user_input: str, answer: str,
                    chunk_ids: list[str], cache_hit: bool = False):
        # Always update STM
        self.stm.add("user", user_input)
        self.stm.add("assistant", answer)
        # Only store to cache and extract LTM facts on non-cache-hit turns
        if not cache_hit:
            standalone = self.stm.rewrite_query(user_input)
            self.cache.store(standalone, answer, chunk_ids)
            self.ltm.extract_and_store(user_input)
        log.info("Write-back complete (cache_hit=%s)", cache_hit)
