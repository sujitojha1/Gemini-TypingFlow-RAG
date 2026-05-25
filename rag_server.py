"""rag_server.py — Local HTTP API for the browser extension.

Delegates all indexing and search to the MCP server (mcp_server.py) via a
persistent MCP ClientSession — the same pattern agent7.py uses. No chunking
or memory logic lives here; index_document and search_knowledge are the
single source of truth.

Two endpoints, CORS-open on localhost:
  POST /index   → calls MCP tool  index_document
  POST /search  → calls MCP tool  search_knowledge
  GET  /health  → liveness check

Start:
    uv run rag_server.py
"""

from __future__ import annotations

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

MCP_SERVER = Path(__file__).parent / "mcp_server.py"


# ── MCP session lifecycle ─────────────────────────────────────────────────────

_session: ClientSession | None = None
_read = _write = None
_cm = None                      # stdio_client context manager
_session_cm = None              # ClientSession context manager


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

app = FastAPI(title="RAG Extension API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── MCP call helper ───────────────────────────────────────────────────────────

async def _call(tool: str, args: dict[str, Any]) -> Any:
    """Call an MCP tool and return the parsed result (dict or list)."""
    if _session is None:
        raise HTTPException(503, "MCP session not ready")
    result = await _session.call_tool(tool, arguments=args)
    # MCP returns content as a list of TextContent objects
    parts = [getattr(c, "text", str(c)) for c in (result.content or [])]
    raw = "\n".join(parts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


# ── request / response models ─────────────────────────────────────────────────

class IndexRequest(BaseModel):
    text: str
    url: str = ""
    title: str = ""
    chunk_size: int = 400
    overlap: int = 80


class SearchRequest(BaseModel):
    query: str
    k: int = 6


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index")
async def index_page(req: IndexRequest) -> dict:
    """Write page text into the sandbox temp file then call index_document."""
    import tempfile, os

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty page text")

    # Write content to a sandbox temp file so index_document can read it
    sandbox = Path(__file__).parent / "sandbox"
    sandbox.mkdir(exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_"
                        for c in (req.title or "page")[:40]) + "_ext.txt"
    tmp_path = sandbox / safe_name
    tmp_path.write_text(
        f"URL: {req.url}\nTitle: {req.title}\n\n{text}",
        encoding="utf-8",
    )

    result = await _call("index_document", {
        "path": safe_name,
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
    })

    return {
        "ok": True,
        "chunks_indexed": result.get("chunks_indexed", 0),
        "source": result.get("source", safe_name),
        "tool": "index_document",
    }


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    """Call search_knowledge and return ranked chunks."""
    result = await _call("search_knowledge", {
        "query": req.query,
        "k": req.k,
    })

    # search_knowledge returns a list directly
    items = result if isinstance(result, list) else result.get("results", [])

    return {
        "query": req.query,
        "results": [
            {
                "id":           r.get("id", ""),
                "descriptor":   r.get("descriptor", ""),
                "source":       r.get("source", ""),
                "preview":      r.get("chunk_preview", "")[:280],
                "chunk_index":  r.get("metadata", {}).get("chunk_index"),
                "total_chunks": r.get("metadata", {}).get("total_chunks"),
            }
            for r in items
        ],
        "tool": "search_knowledge",
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
