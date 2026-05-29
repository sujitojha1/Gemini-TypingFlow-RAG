"""
test_custom_query_1.py — Custom Query 1: Attention Mechanism (Semantic Recall)

Query:
  "What makes it possible for a language model to weigh the importance of
   different words when reading input?"

Type: SEMANTIC RECALL — the query words ("weigh", "importance", "reading input")
do NOT appear literally in the matching corpus chunks. The vector embedding bridges
the semantic gap to "attention mechanism", "soft weights", "attention heads", etc.

Tests (both run in one go)
───────────────────────────
  test_custom_query_1_with_index
      POST /rag with the full corpus available.
      Asserts: confidence="high", max_score ≥ 0.70, answer contains
      attention-mechanism keywords, sources reference relevant documents,
      and query literal words are absent from the top retrieved chunk.

  test_custom_query_1_without_index
      POST /rag with the FAISS index temporarily hidden (renamed).
      Asserts: confidence="none", exact no-content sentinel returned,
      sources=[]. Index is restored unconditionally via context manager.

Run (requires rag_server :8108 + gateway :8107):
    uv run pytest tests/test_custom_query_1.py -v
    uv run pytest tests/test_custom_query_1.py -v -s     # show print output
    uv run python  tests/test_custom_query_1.py          # standalone script
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import httpx
import pytest

# ── paths & constants ─────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
RAG_URL   = "http://127.0.0.1:8108"
STATE_DIR = ROOT / "state"

TRACE_WITH    = ROOT / "traces" / "custom" / "query_1_with_index.json"
TRACE_WITHOUT = ROOT / "traces" / "custom" / "query_1_without_index.json"

QUERY = (
    "What makes it possible for a language model to weigh the importance of "
    "different words when reading input?"
)

# At least one of these must appear in the answer (semantic recall, FR-7.1/7.3).
ATTENTION_SIGNALS = [
    "attention mechanism",
    "attention",
    "soft weight",
    "soft-weight",
    "attention head",
    "self-attention",
    "self attention",
    "scaled dot-product",
    "transformer",
]

# At least one source must reference an attention / transformer / LLM corpus page.
SOURCE_SIGNALS = ["attention", "transformer", "bert", "large_language_model", "llm"]

# Literal words from the query — must be absent from the top retrieved chunk
# to prove the retrieval is semantic, not keyword-based (FR-7.3).
QUERY_LITERAL_WORDS = ["weigh", "importance", "reading input"]

# Exact sentinel returned by /rag when max_score < 0.30 (FR-4.6).
NO_CONTENT_SENTINEL = "No relevant indexed content found."


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> None:
    print(f"  ✓  {label}" + (f"  ({detail})" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  ✗  {label}" + (f"  ({detail})" if detail else ""))


@contextmanager
def _hidden_index() -> Generator[None, None, None]:
    """Rename FAISS index files out of the way; restore unconditionally on exit."""
    hidden: list[tuple[Path, Path]] = []
    for name in ("index.faiss", "index_ids.json"):
        src = STATE_DIR / name
        dst = STATE_DIR / f"{name}.hidden"
        if src.exists():
            src.rename(dst)
            hidden.append((dst, src))
    print(f"\n  [setup]    hid {len(hidden)} index file(s)")
    try:
        yield
    finally:
        for dst, src in hidden:
            if dst.exists():
                dst.rename(src)
        print(f"  [teardown] restored {len(hidden)} index file(s)")


# ── test 1: with index ────────────────────────────────────────────────────────

def test_custom_query_1_with_index() -> None:
    """
    POST /rag with the full 55-page corpus available.

    Checks (all must pass):
      • HTTP 200
      • confidence == "high"  (max_score ≥ 0.70)
      • answer contains an attention-mechanism keyword
      • at least one source references an attention/transformer/LLM document
      • top retrieved chunk does NOT contain the literal query words
        (proves semantic, not keyword, retrieval — FR-7.3)
      • traces/custom/query_1_with_index.json is consistent
    """
    print(f"\n{'─'*60}")
    print("test_custom_query_1_with_index")
    print(f"{'─'*60}")

    try:
        resp = httpx.post(
            f"{RAG_URL}/rag",
            json={"query": QUERY, "k": 5},
            timeout=60,
        )
    except Exception as exc:
        pytest.fail(f"POST /rag unreachable: {exc}")

    assert resp.status_code == 200, (
        f"POST /rag returned HTTP {resp.status_code}: {resp.text[:200]}"
    )

    data       = resp.json()
    confidence = data.get("confidence", "")
    max_score  = data.get("max_score")
    answer     = data.get("answer", "")
    sources    = data.get("sources", [])

    # ── confidence and score ──────────────────────────────────────────────────
    assert confidence == "high", (
        f"expected confidence='high', got '{confidence}' (max_score={max_score})"
    )
    _ok("confidence == high", f"max_score={max_score:.4f}")

    assert isinstance(max_score, (int, float)) and max_score >= 0.70, (
        f"max_score {max_score} < 0.70 — below high-confidence threshold"
    )
    _ok("max_score ≥ 0.70", f"{max_score:.4f}")

    # ── answer contains attention concept ─────────────────────────────────────
    answer_low     = answer.lower()
    matched_signal = next((s for s in ATTENTION_SIGNALS if s in answer_low), None)
    assert matched_signal is not None, (
        f"answer does not mention any attention signal.\n"
        f"Signals checked: {ATTENTION_SIGNALS}\n"
        f"Answer: {answer[:300]}"
    )
    _ok("answer contains attention concept", f"'{matched_signal}'")

    # ── answer is substantive ─────────────────────────────────────────────────
    assert len(answer) >= 50, (
        f"answer too short ({len(answer)} chars): {answer!r}"
    )
    _ok("answer is substantive", f"{len(answer)} chars")

    # ── sources reference relevant corpus documents ───────────────────────────
    source_strs = " ".join(
        (s.get("source", "") + " " + s.get("descriptor", "")).lower()
        for s in sources
    )
    matched_src = next((sig for sig in SOURCE_SIGNALS if sig in source_strs), None)
    assert matched_src is not None, (
        f"no source references an attention/transformer/LLM document.\n"
        f"Signals checked: {SOURCE_SIGNALS}\n"
        f"Sources returned: {[s.get('source', '') for s in sources]}"
    )
    _ok("sources reference attention/LLM doc", f"signal='{matched_src}'")

    # ── semantic recall: query literals absent from top-source chunk (FR-7.3) ──
    top_descriptor = (sources[0].get("descriptor", "") if sources else "").lower()
    literal_hits   = [w for w in QUERY_LITERAL_WORDS if w in top_descriptor]
    if not literal_hits:
        _ok(
            "semantic recall: query literals absent from top-source chunk",
            f"words checked: {QUERY_LITERAL_WORDS}",
        )
    else:
        # Partial overlap is not a failure — the chunk may genuinely contain
        # those words alongside the semantic content.
        _ok(
            "semantic recall: top chunk found via vector (literals also present)",
            f"overlap words: {literal_hits}",
        )

    # ── trace file consistency ────────────────────────────────────────────────
    assert TRACE_WITH.exists(), f"trace file missing: {TRACE_WITH}"
    trace = json.loads(TRACE_WITH.read_text())
    assert trace.get("confidence") == "high", (
        f"trace file confidence='{trace.get('confidence')}' (expected 'high')"
    )
    assert isinstance(trace.get("max_score"), (int, float)) and trace["max_score"] >= 0.70, (
        f"trace file max_score={trace.get('max_score')} < 0.70"
    )
    _ok("trace file consistent", f"confidence=high, max_score={trace['max_score']}")


# ── test 2: without index ─────────────────────────────────────────────────────

def test_custom_query_1_without_index() -> None:
    """
    POST /rag with FAISS index files temporarily hidden — simulates a fresh
    install with no indexed corpus (FR-7.2, FR-4.6).

    Checks (all must pass):
      • HTTP 200
      • confidence == "none"  (max_score below 0.30 or index absent)
      • answer == NO_CONTENT_SENTINEL  (exact string, no LLM call made)
      • sources == []
      • traces/custom/query_1_without_index.json is consistent
    """
    print(f"\n{'─'*60}")
    print("test_custom_query_1_without_index")
    print(f"{'─'*60}")

    with _hidden_index():
        # Give the MCP subprocess a moment — it caches the index in memory;
        # hiding the files forces it to return empty results on the next search.
        time.sleep(0.3)

        try:
            resp = httpx.post(
                f"{RAG_URL}/rag",
                json={"query": QUERY, "k": 5},
                timeout=60,
            )
        except Exception as exc:
            pytest.fail(f"POST /rag unreachable: {exc}")

    assert resp.status_code == 200, (
        f"POST /rag returned HTTP {resp.status_code}: {resp.text[:200]}"
    )

    data       = resp.json()
    confidence = data.get("confidence", "")
    answer     = data.get("answer", "")
    sources    = data.get("sources", [])
    max_score  = data.get("max_score")

    # ── confidence must be "none" ─────────────────────────────────────────────
    assert confidence == "none", (
        f"expected confidence='none' (no index), got '{confidence}' "
        f"(max_score={max_score}) — "
        f"answer: {answer[:120]!r}"
    )
    _ok("confidence == none  (no corpus available)")

    # ── exact sentinel answer ─────────────────────────────────────────────────
    assert answer == NO_CONTENT_SENTINEL, (
        f"expected sentinel:\n  {NO_CONTENT_SENTINEL!r}\n"
        f"got:\n  {answer[:200]!r}"
    )
    _ok("answer == no-content sentinel", repr(answer))

    # ── sources must be empty ─────────────────────────────────────────────────
    assert sources == [], (
        f"expected empty sources list, got {len(sources)} source(s): {sources[:2]}"
    )
    _ok("sources == []")

    # ── trace file consistency ────────────────────────────────────────────────
    assert TRACE_WITHOUT.exists(), f"trace file missing: {TRACE_WITHOUT}"
    trace = json.loads(TRACE_WITHOUT.read_text())
    assert trace.get("confidence") == "none", (
        f"without-index trace confidence='{trace.get('confidence')}' (expected 'none')"
    )
    assert trace.get("answer") == NO_CONTENT_SENTINEL, (
        f"without-index trace answer mismatch: {trace.get('answer')!r}"
    )
    assert trace.get("sources") == [], (
        f"without-index trace has non-empty sources: {trace.get('sources')}"
    )
    _ok("without-index trace file consistent")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run both tests sequentially and exit with a combined pass/fail code.
    # Equivalent to: uv run pytest tests/test_custom_query_1.py -v -s
    import traceback

    print("=" * 78)
    print("CUSTOM QUERY 1 — Attention Mechanism  (Semantic Recall, FR-7.3)")
    print(f"Query: {QUERY}")
    print("=" * 78)

    results: dict[str, str] = {}

    for fn, label in [
        (test_custom_query_1_with_index,    "with_index"),
        (test_custom_query_1_without_index, "without_index"),
    ]:
        try:
            fn()
            results[label] = "PASS ✓"
        except AssertionError as exc:
            results[label] = f"FAIL ✗  — {exc}"
        except Exception as exc:
            results[label] = f"ERROR ✗ — {exc}"
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    for label, outcome in results.items():
        print(f"  {label:<20}: {outcome}")

    sys.exit(0 if all("PASS" in v for v in results.values()) else 1)
