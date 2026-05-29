#!/usr/bin/env python3
"""scripts/run_custom_queries.py — FR-7.1–FR-7.4

Runs 5 custom queries against the 50+ page corpus in two modes:
  1. with_index  — calls the running rag_server at localhost:8108 (full FAISS index)
  2. without_index — temporarily hides FAISS files so memory returns empty results

Saves 10 trace files:
  traces/custom/query_N_with_index.json
  traces/custom/query_N_without_index.json

Usage:
    uv run scripts/run_custom_queries.py
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
TRACES_DIR = ROOT / "traces" / "custom"
STATE_DIR = ROOT / "state"

# ── 5 custom queries (FR-7.1–FR-7.3) ─────────────────────────────────────────
# Types: "semantic" = query words absent from matching chunks (FR-7.3)
#        "keyword"  = exact term match expected
#        "mixed"    = some conceptual, some keyword

QUERIES = [
    {
        "id": 1,
        "type": "semantic",
        "text": (
            "What makes it possible for a language model to weigh the importance "
            "of different words when reading input?"
        ),
        "expected_topic": "attention mechanism / transformer architecture",
        "semantic_check_words": ["weigh", "importance", "reading input"],
    },
    {
        "id": 2,
        "type": "semantic",
        "text": (
            "How do deep networks avoid the problem of learning signals "
            "becoming too small to be useful during training?"
        ),
        "expected_topic": "vanishing gradient / batch normalization / residual connections",
        "semantic_check_words": ["learning signals", "too small", "useful"],
    },
    {
        "id": 3,
        "type": "keyword",
        "text": "What does asyncio.gather do in Python?",
        "expected_topic": "asyncio — Coroutines and Tasks",
        "semantic_check_words": [],
    },
    {
        "id": 4,
        "type": "keyword",
        "text": (
            "What are the differences between LoRA and full fine-tuning "
            "of large language models?"
        ),
        "expected_topic": "LoRA: Low-Rank Adaptation of Large Language Models",
        "semantic_check_words": [],
    },
    {
        "id": 5,
        "type": "mixed",
        "text": (
            "Why might a very large language model sometimes produce "
            "plausible but factually incorrect responses?"
        ),
        "expected_topic": "large language model hallucination / alignment",
        "semantic_check_words": ["plausible", "factually incorrect"],
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def query_with_index(query_text: str, k: int = 6) -> dict:
    """POST /rag against the running rag_server."""
    resp = httpx.post(
        "http://127.0.0.1:8108/rag",
        json={"query": query_text, "k": k},
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()


# Inline script run in a fresh subprocess so memory module is imported with no
# FAISS files on disk.  The script renames the index files, runs search, then
# restores them — all in a finally block so restoration always happens.
_WITHOUT_INDEX_SCRIPT = textwrap.dedent("""\
    import sys, json, pathlib

    root = pathlib.Path(sys.argv[1])
    query = sys.argv[2]
    k = int(sys.argv[3])

    sys.path.insert(0, str(root))

    state_dir = root / "state"
    # Hide FAISS files AND memory.json so _index() cannot rebuild from embeddings.
    # memory._index() falls back to rebuilding from memory.json when FAISS is
    # absent, so we must hide the memory store as well.
    to_hide = [
        (state_dir / "index.faiss",    state_dir / "index.faiss.bak"),
        (state_dir / "index_ids.json", state_dir / "index_ids.json.bak"),
        (state_dir / "memory.json",    state_dir / "memory.json.bak"),
    ]

    moved = []
    try:
        for orig, bak in to_hide:
            if orig.exists():
                orig.rename(bak)
                moved.append((bak, orig))

        import memory
        results = memory.search_with_scores(query, kinds=["fact"], top_k=k)
        out = [
            {
                "id": item.id,
                "descriptor": item.descriptor,
                "source": item.source,
                "score": round(score, 4),
                "chunk_preview": (item.value.get("chunk") or "")[:240],
            }
            for score, item in results
        ]
        print(json.dumps({"results": out}))
    finally:
        for bak, orig in moved:
            if bak.exists():
                bak.rename(orig)
""")


def query_without_index(query_text: str, k: int = 6) -> dict:
    """Run a search in a subprocess with FAISS files temporarily hidden."""
    proc = subprocess.run(
        [sys.executable, "-c", _WITHOUT_INDEX_SCRIPT, str(ROOT), query_text, str(k)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"without-index subprocess failed (rc={proc.returncode}):\n{proc.stderr}"
        )

    raw = proc.stdout.strip()
    data = json.loads(raw) if raw else {"results": []}
    items = data.get("results", [])

    if not items:
        return {
            "query": query_text,
            "confidence": "none",
            "max_score": None,
            "answer": "No relevant indexed content found.",
            "sources": [],
        }

    scores = [r.get("score", 0.0) for r in items]
    max_score = max(scores)
    return {
        "query": query_text,
        "confidence": "none",
        "max_score": round(max_score, 4),
        "answer": "No relevant indexed content found.",
        "sources": [
            {"source": r["source"], "descriptor": r["descriptor"]} for r in items
        ],
    }


def save_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    print(f"  saved → {path.relative_to(ROOT)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Verify server is reachable
    try:
        health = httpx.get("http://127.0.0.1:8108/health", timeout=5).json()
        print(f"[health] rag_server OK  tools={health.get('mcp_tools', [])[:3]}…\n")
    except Exception as exc:
        print(f"[ERROR] rag_server not reachable: {exc}")
        print("Start it first:  uv run rag_server.py")
        sys.exit(1)

    summary: list[dict] = []

    for q in QUERIES:
        qid = q["id"]
        qtext = q["text"]
        print(f"── Query {qid} ({q['type']}) ──────────────────────────────────")
        print(f"  {qtext}")

        # ── with index ────────────────────────────────────────────────────────
        print("  [with_index] querying rag_server …")
        wi_result = query_with_index(qtext)
        wi_trace = {
            "query_id": qid,
            "query_text": qtext,
            "query_type": q["type"],
            "expected_topic": q["expected_topic"],
            "mode": "with_index",
            "timestamp": _now(),
            **wi_result,
        }
        save_trace(TRACES_DIR / f"query_{qid}_with_index.json", wi_trace)
        print(f"  confidence={wi_result.get('confidence')}  max_score={wi_result.get('max_score')}")

        # ── without index ─────────────────────────────────────────────────────
        print("  [without_index] hiding FAISS files in subprocess …")
        wo_result = query_without_index(qtext)
        wo_trace = {
            "query_id": qid,
            "query_text": qtext,
            "query_type": q["type"],
            "expected_topic": q["expected_topic"],
            "mode": "without_index",
            "timestamp": _now(),
            **wo_result,
        }
        save_trace(TRACES_DIR / f"query_{qid}_without_index.json", wo_trace)
        print(f"  confidence={wo_result.get('confidence')}  max_score={wo_result.get('max_score')}")

        summary.append({
            "id": qid,
            "type": q["type"],
            "with_confidence": wi_result.get("confidence"),
            "with_max_score": wi_result.get("max_score"),
            "without_confidence": wo_result.get("confidence"),
        })
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for row in summary:
        flag = "✓" if row["with_confidence"] in ("high", "medium") else "✗"
        woflag = "✓ (degraded)" if row["without_confidence"] == "none" else "✗ (still answered)"
        print(
            f"  Q{row['id']} [{row['type']:8s}]  with={row['with_confidence']}({row['with_max_score']})  "
            f"without={row['without_confidence']}  {flag} / {woflag}"
        )

    trace_files = sorted(TRACES_DIR.glob("query_*.json"))
    print(f"\nTrace files written: {len(trace_files)} / 10")
    for tf in trace_files:
        print(f"  {tf.name}")


if __name__ == "__main__":
    main()
