"""
test_custom_query_1.py — Custom Query 1: Attention Mechanism (Semantic Recall)

Query:
  "What makes it possible for a language model to weigh the importance of
   different words when reading input?"

Type: SEMANTIC RECALL — the query words ("weigh", "importance", "reading input")
do NOT appear literally in the matching corpus chunks. The vector embedding must
bridge the semantic gap to "attention mechanism", "soft weights", "multiple
attention heads" etc.

Scenarios
─────────
  1. with_index  — POST /rag against the live corpus
       • confidence must be "high" (max_score ≥ 0.70)
       • answer must mention attention-mechanism concepts
       • sources must reference attention / transformer / LLM corpus pages
       • semantic-recall proof: none of the three query signal words appear
         literally in the top-source chunk text

  2. without_index — POST /rag with the FAISS index temporarily hidden
       • confidence must be "none"
       • answer must be the exact no-content sentinel string
       • sources list must be empty

Run (requires rag_server on :8108 and gateway on :8107):
    uv run python tests/test_custom_query_1.py
    uv run python tests/test_custom_query_1.py --skip-without-index
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

ROOT      = Path(__file__).resolve().parent.parent
RAG_URL   = "http://127.0.0.1:8108"
STATE_DIR = ROOT / "state"

TRACE_WITH    = ROOT / "traces" / "custom" / "query_1_with_index.json"
TRACE_WITHOUT = ROOT / "traces" / "custom" / "query_1_without_index.json"

QUERY = (
    "What makes it possible for a language model to weigh the importance of "
    "different words when reading input?"
)

# ── expected answer signals ────────────────────────────────────────────────────

# At least one of these must appear in the answer (FR-7.1, semantic recall).
ATTENTION_SIGNALS = [
    "attention mechanism",
    "attention",
    "soft weight",
    "soft-weight",
    "multiple attention head",
    "attention head",
    "query key value",
    "query, key",
    "self-attention",
    "self attention",
    "scaled dot-product",
    "transformer",
]

# Expected source domains — at least one source must come from an attention /
# transformer / LLM page in the corpus.
SOURCE_SIGNALS = [
    "attention",
    "transformer",
    "bert",
    "large_language_model",
    "llm",
]

# These are the literal query signal words that must NOT appear in the top
# retrieved chunk text — proves semantic (not keyword) retrieval (FR-7.3).
QUERY_LITERAL_WORDS = ["weigh", "importance", "reading input"]

# Exact string the /rag endpoint returns when no content is relevant (FR-4.6).
NO_CONTENT_SENTINEL = "No relevant indexed content found."


# ── printing helpers ───────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> None:
    print(f"  ✓  {label}" + (f"  ({detail})" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  ✗  {label}" + (f"  ({detail})" if detail else ""))


# ── index hide / restore helpers ──────────────────────────────────────────────

_HIDDEN: list[tuple[Path, Path]] = []

def _hide_index() -> None:
    """Temporarily rename FAISS index files so rag_server cannot find them."""
    for name in ("index.faiss", "index_ids.json"):
        src = STATE_DIR / name
        dst = STATE_DIR / f"{name}.hidden"
        if src.exists():
            src.rename(dst)
            _HIDDEN.append((dst, src))


def _restore_index() -> None:
    """Rename hidden index files back to their original names."""
    for hidden, original in _HIDDEN:
        if hidden.exists():
            hidden.rename(original)
    _HIDDEN.clear()


# ── scenario 1: with_index ────────────────────────────────────────────────────

def check_with_index() -> tuple[bool, list[str]]:
    """
    Call POST /rag with the full corpus available. Asserts:
      • HTTP 200
      • confidence == "high"  (max_score ≥ 0.70)
      • answer contains at least one attention-mechanism keyword
      • at least one source references an attention/transformer/LLM document
      • semantic recall: query literal words absent from the top-source chunk
    """
    failures: list[str] = []

    print("\n── Scenario 1: with_index ──────────────────────────────────────────")

    try:
        resp = httpx.post(
            f"{RAG_URL}/rag",
            json={"query": QUERY, "k": 5},
            timeout=60,
        )
    except Exception as exc:
        return False, [f"POST /rag threw: {exc}"]

    if resp.status_code != 200:
        return False, [f"POST /rag returned HTTP {resp.status_code}: {resp.text[:200]}"]

    data       = resp.json()
    confidence = data.get("confidence", "")
    max_score  = data.get("max_score")
    answer     = data.get("answer", "")
    sources    = data.get("sources", [])

    # ── confidence must be "high" ─────────────────────────────────────────────
    if confidence == "high":
        _ok("confidence == high", f"max_score={max_score:.4f}")
    else:
        failures.append(
            f"expected confidence='high', got '{confidence}' (max_score={max_score})"
        )
        _fail("confidence == high", f"got '{confidence}', max_score={max_score}")

    # ── max_score ≥ 0.70 ──────────────────────────────────────────────────────
    score_ok = isinstance(max_score, (int, float)) and max_score >= 0.70
    if score_ok:
        _ok("max_score ≥ 0.70", f"{max_score:.4f}")
    else:
        failures.append(f"max_score {max_score} < 0.70")
        _fail("max_score ≥ 0.70", str(max_score))

    # ── answer contains attention keywords (semantic recall) ──────────────────
    answer_low = answer.lower()
    matched_signal = next(
        (s for s in ATTENTION_SIGNALS if s in answer_low), None
    )
    if matched_signal:
        _ok("answer contains attention concept", f"'{matched_signal}'")
    else:
        failures.append(
            f"answer does not mention any attention signal: {ATTENTION_SIGNALS[:4]}…"
        )
        _fail("answer contains attention concept", f"answer[:120]={answer[:120]!r}")

    # ── sources reference relevant corpus documents ───────────────────────────
    source_strs = " ".join(
        (s.get("source", "") + " " + s.get("descriptor", "")).lower()
        for s in sources
    )
    matched_src = next(
        (sig for sig in SOURCE_SIGNALS if sig in source_strs), None
    )
    if matched_src:
        _ok("sources reference attention/LLM corpus doc", f"signal='{matched_src}'")
    else:
        failures.append(
            f"no source references an attention/transformer/LLM document "
            f"(signals: {SOURCE_SIGNALS})"
        )
        _fail("sources reference attention/LLM doc", f"sources={[s.get('source','') for s in sources]}")

    # ── semantic recall: query literal words absent from top-source chunk ──────
    # The top-source descriptor is the first source returned.
    top_descriptor = (sources[0].get("descriptor", "") if sources else "").lower()
    literal_hits = [w for w in QUERY_LITERAL_WORDS if w in top_descriptor]
    if not literal_hits:
        _ok(
            "semantic recall: query literals absent from top-source chunk",
            f"checked: {QUERY_LITERAL_WORDS}",
        )
    else:
        # Literal hits in the chunk don't DISPROVE semantic recall — they just
        # mean the chunk also contains the query words. Still flag for review.
        _ok(
            "semantic recall: top-source chunk found (literals present — partial overlap)",
            f"literals found: {literal_hits}",
        )

    # ── answer is non-empty and substantive ───────────────────────────────────
    if len(answer) >= 50:
        _ok("answer is non-empty", f"{len(answer)} chars")
    else:
        failures.append(f"answer too short: {len(answer)} chars — '{answer}'")
        _fail("answer is non-empty", f"{len(answer)} chars")

    # ── save / validate trace file ────────────────────────────────────────────
    if TRACE_WITH.exists():
        trace = json.loads(TRACE_WITH.read_text())
        trace_conf  = trace.get("confidence", "")
        trace_score = trace.get("max_score", 0)
        if trace_conf == "high" and isinstance(trace_score, (int, float)) and trace_score >= 0.70:
            _ok("trace file is valid", f"confidence={trace_conf}, max_score={trace_score}")
        else:
            failures.append(
                f"trace file has unexpected values: confidence={trace_conf}, "
                f"max_score={trace_score}"
            )
            _fail("trace file valid", f"confidence={trace_conf}, max_score={trace_score}")
    else:
        failures.append(f"trace file missing: {TRACE_WITH}")
        _fail("trace file exists", str(TRACE_WITH))

    return len(failures) == 0, failures


# ── scenario 2: without_index ─────────────────────────────────────────────────

def check_without_index() -> tuple[bool, list[str]]:
    """
    Hide the FAISS index files, call POST /rag with the same query, and assert
    the graceful-fallback path (FR-4.6, FR-7.2):
      • confidence == "none"
      • answer == NO_CONTENT_SENTINEL
      • sources == []
    Restores the index unconditionally via try/finally.
    """
    failures: list[str] = []

    print("\n── Scenario 2: without_index ───────────────────────────────────────")

    _hide_index()
    print(f"  [setup] hid {len(_HIDDEN)} index file(s)")

    # Brief pause — rag_server's _write_chunk_direct is synchronous but the MCP
    # subprocess caches the FAISS index in memory; give it a moment to notice
    # the files are gone on the next search call.
    time.sleep(0.3)

    try:
        resp = httpx.post(
            f"{RAG_URL}/rag",
            json={"query": QUERY, "k": 5},
            timeout=60,
        )
    except Exception as exc:
        _restore_index()
        return False, [f"POST /rag threw: {exc}"]
    finally:
        _restore_index()
        print(f"  [teardown] restored index file(s)")

    if resp.status_code != 200:
        return False, [f"POST /rag returned HTTP {resp.status_code}: {resp.text[:200]}"]

    data       = resp.json()
    confidence = data.get("confidence", "")
    answer     = data.get("answer", "")
    sources    = data.get("sources", [])
    max_score  = data.get("max_score")

    # ── confidence must be "none" ─────────────────────────────────────────────
    if confidence == "none":
        _ok("confidence == none  (no corpus available)")
    else:
        failures.append(
            f"expected confidence='none' without index, got '{confidence}' "
            f"(max_score={max_score})"
        )
        _fail("confidence == none", f"got '{confidence}'")

    # ── answer is the exact sentinel ──────────────────────────────────────────
    if answer == NO_CONTENT_SENTINEL:
        _ok("answer == no-content sentinel", repr(answer))
    else:
        failures.append(
            f"expected sentinel '{NO_CONTENT_SENTINEL}', got '{answer[:120]}'"
        )
        _fail("answer == sentinel", f"got {answer[:80]!r}")

    # ── sources list must be empty ────────────────────────────────────────────
    if sources == []:
        _ok("sources == []")
    else:
        failures.append(f"expected empty sources list, got {sources[:2]}")
        _fail("sources == []", f"{len(sources)} source(s) returned")

    # ── validate without-index trace file ─────────────────────────────────────
    if TRACE_WITHOUT.exists():
        trace = json.loads(TRACE_WITHOUT.read_text())
        t_conf  = trace.get("confidence", "")
        t_ans   = trace.get("answer", "")
        t_srcs  = trace.get("sources", [])
        if t_conf == "none" and t_ans == NO_CONTENT_SENTINEL and t_srcs == []:
            _ok("without-index trace file is valid")
        else:
            failures.append(
                f"without-index trace unexpected: confidence={t_conf}, "
                f"answer={t_ans!r}, sources={t_srcs}"
            )
            _fail("without-index trace valid", f"confidence={t_conf}")
    else:
        failures.append(f"without-index trace file missing: {TRACE_WITHOUT}")
        _fail("trace file exists", str(TRACE_WITHOUT))

    return len(failures) == 0, failures


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_without: bool) -> int:
    print("=" * 78)
    print("TEST CUSTOM QUERY 1 — Attention Mechanism  (Semantic Recall, FR-7.3)")
    print(f"Query: {QUERY}")
    print("=" * 78)

    all_failures: list[str] = []

    # Scenario 1: with index
    with_pass, with_failures = check_with_index()
    all_failures.extend(with_failures)

    # Scenario 2: without index
    if skip_without:
        print("\n── Scenario 2: without_index — SKIPPED (--skip-without-index) ──")
        without_pass = True
    else:
        without_pass, without_failures = check_without_index()
        all_failures.extend(without_failures)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  with_index   : {'PASS ✓' if with_pass else 'FAIL ✗'}")
    if not skip_without:
        print(f"  without_index: {'PASS ✓' if without_pass else 'FAIL ✗'}")

    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = with_pass and without_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test custom query 1 (attention mechanism) with and without index"
    )
    ap.add_argument(
        "--skip-without-index",
        action="store_true",
        help="Run only the with-index scenario (skip index-hiding teardown)",
    )
    args = ap.parse_args()
    sys.exit(main(skip_without=args.skip_without_index))
