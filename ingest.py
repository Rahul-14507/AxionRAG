import os
import re
import sys
import argparse
import hashlib
import logging
from pathlib import Path

import tiktoken
from rag_memory_core import RAGMemoryPipeline

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Default chunking config ────────────────────────────────────────────────
DEFAULT_CHUNK_TOKENS   = 256    # target chunk size in tokens
DEFAULT_OVERLAP_TOKENS = 32     # overlap between consecutive chunks
TOKENIZER              = "cl100k_base"  # tiktoken encoding (GPT-4 / Gemini compatible)

# ──────────────────────────────────────────────────────────────────────────────
# Text extractors
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)

def extract_docx(path: Path) -> str:
    from docx import Document
    doc  = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def extract_md(path: Path) -> str:
    import markdown, html
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
        def handle_data(self, data):
            self.parts.append(data)

    raw  = path.read_text(encoding="utf-8", errors="replace")
    html_str = markdown.markdown(raw)
    s    = _Stripper()
    s.feed(html_str)
    return " ".join(s.parts)

EXTRACTORS = {
    ".pdf":  extract_pdf,
    ".docx": extract_docx,
    ".txt":  extract_txt,
    ".md":   extract_md,
}

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {ext}")
    return EXTRACTORS[ext](path)

# ──────────────────────────────────────────────────────────────────────────────
# Token-aware chunker
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Splits text into overlapping token windows.
    Splits on sentence boundaries where possible to avoid cutting mid-sentence.
    """
    enc      = tiktoken.get_encoding(TOKENIZER)
    # Split into sentences first for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks:  list[str] = []
    current: list[str] = []
    current_tokens      = 0

    for sentence in sentences:
        s_tokens = len(enc.encode(sentence))

        # If a single sentence exceeds chunk size, hard-split it by tokens
        if s_tokens > chunk_tokens:
            if current:
                chunks.append(" ".join(current))
                current = current[-overlap_tokens:] if overlap_tokens else []
                current_tokens = sum(len(enc.encode(s)) for s in current)

            tokens = enc.encode(sentence)
            for i in range(0, len(tokens), chunk_tokens - overlap_tokens):
                window = tokens[i: i + chunk_tokens]
                chunks.append(enc.decode(window))
            continue

        if current_tokens + s_tokens > chunk_tokens:
            chunks.append(" ".join(current))
            # Keep tail for overlap
            overlap_sents: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = len(enc.encode(s))
                if overlap_count + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += t
            current       = overlap_sents
            current_tokens = overlap_count

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]

# ──────────────────────────────────────────────────────────────────────────────
# Chunk ID generation — deterministic, content-addressed
# ──────────────────────────────────────────────────────────────────────────────

def make_chunk_id(source_path: Path, chunk_index: int, text: str) -> str:
    """
    Deterministic ID: hash of (file path + chunk index + first 64 chars).
    Re-indexing the same file produces the same IDs → safe to re-run (upsert).
    """
    key = f"{source_path}::{chunk_index}::{text[:64]}"
    return hashlib.md5(key.encode()).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# File ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest_file(
    path: Path,
    pipeline: RAGMemoryPipeline,
    chunk_tokens: int,
    overlap_tokens: int,
) -> int:
    """
    Extracts, chunks, and indexes a single file.
    Returns the number of chunks indexed.
    """
    log.info("Ingesting: %s", path)
    try:
        text = extract_text(path)
    except Exception as e:
        log.error("Failed to extract %s: %s", path, e)
        return 0

    if not text.strip():
        log.warning("Empty content — skipping: %s", path)
        return 0

    chunks = chunk_text(text, chunk_tokens, overlap_tokens)
    log.info("  → %d chunks", len(chunks))

    for i, chunk in enumerate(chunks):
        chunk_id = make_chunk_id(path, i, chunk)
        metadata = {
            "source": str(path),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        pipeline.index(chunk_id, chunk, metadata)

    return len(chunks)

# ──────────────────────────────────────────────────────────────────────────────
# Directory ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest_source(
    source: Path,
    pipeline: RAGMemoryPipeline,
    chunk_tokens: int,
    overlap_tokens: int,
) -> dict[str, int]:
    """
    Ingests a file or every supported file under a directory.
    Returns {file_path: chunks_indexed}.
    """
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = [
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTRACTORS
        ]
        if not files:
            log.warning("No supported files found in %s", source)
            return {}
    else:
        log.error("Source does not exist: %s", source)
        sys.exit(1)

    results: dict[str, int] = {}
    total_chunks = 0
    for f in sorted(files):
        n = ingest_file(f, pipeline, chunk_tokens, overlap_tokens)
        results[str(f)] = n
        total_chunks += n

    log.info("Ingestion complete — %d file(s), %d total chunks", len(files), total_chunks)
    return results

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline.")
    parser.add_argument("--source",         required=True,              help="File or directory to ingest")
    parser.add_argument("--user",           default="default",          help="User ID for the pipeline (default: 'default')")
    parser.add_argument("--chunk-tokens",   type=int, default=DEFAULT_CHUNK_TOKENS,   help=f"Chunk size in tokens (default: {DEFAULT_CHUNK_TOKENS})")
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS, help=f"Overlap in tokens (default: {DEFAULT_OVERLAP_TOKENS})")
    args = parser.parse_args()

    pipeline = RAGMemoryPipeline(user_id=args.user)
    results  = ingest_source(
        source         = Path(args.source),
        pipeline       = pipeline,
        chunk_tokens   = args.chunk_tokens,
        overlap_tokens = args.overlap_tokens,
    )

    print("\nIngestion summary:")
    print(f"  {'File':<60} Chunks")
    print(f"  {'─'*60} ──────")
    for file_path, n_chunks in results.items():
        short = file_path[-57:] if len(file_path) > 57 else file_path
        print(f"  {short:<60} {n_chunks}")
    print(f"\n  Total chunks indexed: {sum(results.values())}")

if __name__ == "__main__":
    main()
