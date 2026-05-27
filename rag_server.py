"""rag_server.py — Local HTTP API for the browser extension.

Delegates all indexing and search to the MCP server (mcp_server.py) via a
persistent MCP ClientSession — the same pattern agent7.py uses. No chunking
or memory logic lives here; index_document and search_knowledge are the
single source of truth.

Endpoints, CORS-open on localhost:
  POST /index   → index page text; supports reindex=true to replace old vectors
  POST /search  → ranked chunk retrieval (with cosine scores)
  POST /rag     → confidence-gated retrieval + LLM answer assembly
  GET  /status  → {page_count, chunk_count, last_indexed_url, index_size_bytes}
  GET  /health  → liveness check

Start:
    uv run rag_server.py

Confidence tiers (FR-5.1–FR-5.3):
  max_score ≥ 0.70  → "high"   — full RAG answer with sources
  max_score ≥ 0.30  → "medium" — RAG answer with disclaimer
  max_score <  0.30 → "none"   — no LLM call; return informational message

Tool-selection guidance lives in this docstring only — no tool names leak into
any SYSTEM prompt sent to the LLM (FR-5.5, AC-2, NFR-8).
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

MCP_SERVER  = Path(__file__).parent / "mcp_server.py"
STATE_DIR   = Path(__file__).parent / "state"
_STATUS_FILE = STATE_DIR / "rag_status.json"


# ── status helpers ────────────────────────────────────────────────────────────

def _load_status() -> dict:
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text())
        except Exception:
            pass
    return {"last_indexed_url": None, "indexed_urls": [], "url_to_file": {}}


def _save_status(s: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(json.dumps(s, indent=2))


# ── re-index: remove old fact items + rebuild FAISS ──────────────────────────

def _remove_and_rebuild(safe_name: str) -> int:
    """Remove all fact items whose descriptor references *safe_name* from
    memory.json, then rebuild the FAISS index from the surviving items.

    Returns the number of items removed.
    """
    import numpy as np
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        return 0

    mem_path = STATE_DIR / "memory.json"
    if not mem_path.exists():
        return 0

    items = json.loads(mem_path.read_text())
    prefix = f"[sandbox:{safe_name}"
    kept = [i for i in items if not (
        i.get("kind") == "fact" and i.get("descriptor", "").startswith(prefix)
    )]
    removed = len(items) - len(kept)

    if removed == 0:
        return 0

    # Write back surviving items
    mem_path.write_text(json.dumps(kept, indent=2))

    # Rebuild FAISS from scratch using surviving items that have embeddings
    idx_path  = STATE_DIR / "index.faiss"
    ids_path  = STATE_DIR / "index_ids.json"

    dim = 768
    index = faiss.IndexFlatIP(dim)
    ids: list[str] = []
    for it in kept:
        emb = it.get("embedding")
        if emb is None:
            continue
        vec = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        if vec.shape[0] != dim:
            continue
        index.add(vec.reshape(1, -1))
        ids.append(it["id"])

    faiss.write_index(index, str(idx_path))
    ids_path.write_text(json.dumps(ids))
    return removed


# ── MCP session lifecycle ─────────────────────────────────────────────────────

_session: ClientSession | None = None
_read = _write = None
_cm = None
_session_cm = None


async def _start_mcp() -> None:
    global _session, _read, _write, _cm, _session_cm
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER)])
    _cm = stdio_client(params)
    _read, _write = await _cm.__aenter__()
    _session_cm = ClientSession(_read, _write)
    _session = await _session_cm.__aenter__()
    await _session.initialize()


async def _stop_mcp() -> None:
    global _session, _session_cm, _cm
    if _session_cm:
        await _session_cm.__aexit__(None, None, None)
    if _cm:
        await _cm.__aexit__(None, None, None)
    _session = _session_cm = _cm = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await _start_mcp()
    yield
    await _stop_mcp()


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Extension API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── MCP call helper ───────────────────────────────────────────────────────────

async def _call(tool: str, args: dict[str, Any]) -> Any:
    """Call an MCP tool and return the parsed result (dict or list).

    FastMCP serialises tool return values in two different ways:
      - A single dict/scalar  → one TextContent containing the JSON
      - A list[dict]          → one TextContent *per item* (not a JSON array)
    We try the whole payload as JSON first; if that fails we parse each
    TextContent individually and return them as a list.
    """
    if _session is None:
        raise HTTPException(503, "MCP session not ready")
    result = await _session.call_tool(tool, arguments=args)
    parts = [getattr(c, "text", str(c)) for c in (result.content or [])]

    if len(parts) == 1:
        try:
            return json.loads(parts[0])
        except json.JSONDecodeError:
            return {"raw": parts[0]}

    items = []
    for part in parts:
        try:
            items.append(json.loads(part))
        except json.JSONDecodeError:
            items.append({"raw": part})
    return items


# ── LLM call for /rag ─────────────────────────────────────────────────────────

async def _rag_answer(query: str, chunks: list[dict]) -> str:
    """Assemble a RAG prompt and call the gateway LLM.

    System prompt (FR-5.4): answer only from context; admit absence clearly.
    Numbered context chunks carry source labels (FR-5.4).
    Runs the synchronous gateway LLM client in a thread pool so the FastAPI
    event loop is not blocked.
    """
    context_lines = "\n\n".join(
        f"[{i + 1}] source: {c.get('source', '?')}\n{c.get('preview', '')}"
        for i, c in enumerate(chunks)
    )
    system = (
        "Answer only from the provided context. "
        "If the answer is absent from the context, "
        "say 'I don't have that in your indexed pages.'"
    )
    prompt = f"Context:\n{context_lines}\n\nQuestion: {query}"

    def _sync_chat() -> str:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from gateway import LLM, ensure_gateway
        ensure_gateway()
        resp = LLM().chat(prompt=prompt, system=system, auto_route="memory")
        return resp.get("content") or resp.get("text") or str(resp)

    return await asyncio.to_thread(_sync_chat)


# ── request / response models ─────────────────────────────────────────────────

class IndexRequest(BaseModel):
    text: str
    url: str = ""
    title: str = ""
    chunk_size: int = 400
    overlap: int = 80
    reindex: bool = False


class SearchRequest(BaseModel):
    query: str
    k: int = 6


class RagRequest(BaseModel):
    query: str
    k: int = 6
    high_threshold: float = 0.70
    medium_threshold: float = 0.30


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index")
async def index_page(req: IndexRequest) -> dict:
    """Write page text into the sandbox temp file then call index_document.

    When reindex=True, all prior fact vectors for the same URL are removed from
    memory.json and FAISS before the new chunks are added (FR-3.7).
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty page text")

    sandbox = Path(__file__).parent / "sandbox"
    sandbox.mkdir(exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_"
                        for c in (req.title or "page")[:40]) + "_ext.txt"
    tmp_path = sandbox / safe_name

    # Re-index: remove old vectors before writing new content
    removed = 0
    if req.reindex and tmp_path.exists():
        removed = await asyncio.to_thread(_remove_and_rebuild, safe_name)

    tmp_path.write_text(
        f"URL: {req.url}\nTitle: {req.title}\n\n{text}",
        encoding="utf-8",
    )

    result = await _call("index_document", {
        "path": safe_name,
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
    })

    chunks_indexed = result.get("chunks_indexed", 0) if isinstance(result, dict) else 0

    # Persist status
    status = _load_status()
    if req.url:
        status["last_indexed_url"] = req.url
        urls: list = status.setdefault("indexed_urls", [])
        if req.url not in urls:
            urls.append(req.url)
        status.setdefault("url_to_file", {})[req.url] = safe_name
    _save_status(status)

    return {
        "ok": True,
        "chunks_indexed": chunks_indexed,
        "removed_old_chunks": removed,
        "source": result.get("source", safe_name) if isinstance(result, dict) else safe_name,
        "reindexed": req.reindex and removed > 0,
        "tool": "index_document",
    }


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    """Call search_knowledge and return ranked chunks with cosine similarity scores."""
    result = await _call("search_knowledge", {
        "query": req.query,
        "k": req.k,
    })

    items = result if isinstance(result, list) else result.get("results", [])

    return {
        "query": req.query,
        "results": [
            {
                "id":           r.get("id", ""),
                "descriptor":   r.get("descriptor", ""),
                "source":       r.get("source", ""),
                "score":        r.get("score"),
                "preview":      r.get("chunk_preview", "")[:280],
                "chunk_index":  r.get("metadata", {}).get("chunk_index"),
                "total_chunks": r.get("metadata", {}).get("total_chunks"),
            }
            for r in items
        ],
        "tool": "search_knowledge",
    }


@app.post("/rag")
async def rag_query(req: RagRequest) -> dict:
    """Retrieve relevant chunks then apply a 3-tier confidence gate (FR-5.1–5.3).

    Confidence tiers:
      max_score ≥ high_threshold (0.70)   → "high"   — LLM answer with sources
      max_score ≥ medium_threshold (0.30) → "medium" — LLM answer with disclaimer
      max_score <  medium_threshold       → "none"   — no LLM call

    RAG prompt assembly (FR-5.4):
      System: answer only from context; admit absence explicitly.
      Context: numbered chunks with source labels.
      User: the original query.
    """
    result = await _call("search_knowledge", {
        "query": req.query,
        "k": req.k,
    })

    items = result if isinstance(result, list) else result.get("results", [])

    if not items:
        return {
            "query": req.query,
            "confidence": "none",
            "answer": "I don't have that in your indexed pages.",
            "sources": [],
            "max_score": None,
        }

    # Extract scores; items from search_knowledge now carry a "score" field
    scores = [r.get("score") for r in items if r.get("score") is not None]
    max_score = max(scores) if scores else 0.0

    chunks = [
        {
            "source":  r.get("source", ""),
            "preview": r.get("chunk_preview", r.get("preview", ""))[:400],
        }
        for r in items
    ]

    sources = [
        {"source": r.get("source", ""), "descriptor": r.get("descriptor", "")}
        for r in items
    ]

    # Confidence gate
    if max_score >= req.high_threshold:
        confidence = "high"
        answer = await _rag_answer(req.query, chunks)
    elif max_score >= req.medium_threshold:
        confidence = "medium"
        llm_answer = await _rag_answer(req.query, chunks)
        answer = (
            f"{llm_answer}\n\n"
            "_(Confidence: medium — the indexed content is only partially relevant.)_"
        )
    else:
        confidence = "none"
        answer = "I don't have that in your indexed pages."

    return {
        "query": req.query,
        "confidence": confidence,
        "max_score": round(max_score, 4),
        "answer": answer,
        "sources": sources,
    }


@app.get("/status")
async def status() -> dict:
    """Return index health: page count, chunk count, last URL, FAISS size.

    Fields (FR-3.6):
      page_count        — unique URLs that have been indexed
      chunk_count       — total fact vectors in the FAISS index
      last_indexed_url  — most recently indexed page URL (null if none)
      index_size_bytes  — size of state/index.faiss on disk (0 if not built yet)
    """
    s = _load_status()

    ids_path = STATE_DIR / "index_ids.json"
    chunk_count = 0
    if ids_path.exists():
        try:
            chunk_count = len(json.loads(ids_path.read_text()))
        except Exception:
            pass

    idx_path = STATE_DIR / "index.faiss"
    index_size_bytes = idx_path.stat().st_size if idx_path.exists() else 0

    return {
        "page_count":       len(s.get("indexed_urls", [])),
        "chunk_count":      chunk_count,
        "last_indexed_url": s.get("last_indexed_url"),
        "index_size_bytes": index_size_bytes,
    }


@app.get("/health")
async def health() -> dict:
    tools = []
    if _session:
        listed = await _session.list_tools()
        tools = [t.name for t in listed.tools]
    return {"ok": True, "port": 8108, "mcp_tools": tools}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8108)
