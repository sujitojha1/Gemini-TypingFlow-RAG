"""
test_custom_query_4.py — Custom Query 4: LoRA vs Full Fine-Tuning (Keyword)

Query:
  "What are the differences between LoRA and full fine-tuning of large
   language models?"

Type: KEYWORD — query terms (LoRA, fine-tuning, large language models) appear
directly in the LoRA arxiv paper indexed in the corpus.

Tests (both run in one go)
───────────────────────────
  test_custom_query_4_with_index
      POST /rag with the full corpus available.
      Asserts: confidence="high", max_score ≥ 0.70, answer explains the core
      LoRA vs fine-tuning distinction (frozen weights, rank decomposition,
      trainable parameters), sources reference the LoRA / LLM corpus pages.

  test_custom_query_4_without_index
      POST /rag with the FAISS index temporarily hidden (renamed).
      Asserts: confidence="none", exact no-content sentinel returned,
      sources=[]. Index is restored unconditionally via context manager.

Run (requires rag_server :8108 + gateway :8107):
    uv run pytest tests/test_custom_query_4.py -v
    uv run pytest tests/test_custom_query_4.py -v -s     # show print output
    uv run python  tests/test_custom_query_4.py          # standalone script
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

TRACE_WITH    = ROOT / "traces" / "custom" / "query_4_with_index.json"
TRACE_WITHOUT = ROOT / "traces" / "custom" / "query_4_without_index.json"

QUERY = (
    "What are the differences between LoRA and full fine-tuning of large "
    "language models?"
)

# At least one must appear in the answer.
LORA_SIGNALS = [
    "lora",
    "low-rank",
    "low rank",
    "rank decomposition",
    "trainable parameter",
    "freeze",
    "frozen",
    "adapter",
    "fine-tuning",
    "fine tuning",
    "weight matrix",
    "intrinsic rank",
]

# At least one source must reference the LoRA paper or LLM corpus pages.
SOURCE_SIGNALS = [
    "lora__low-rank",
    "lora",
    "large_language_model",
    "direct_preference",
    "bert",
]

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

def test_custom_query_4_with_index() -> None:
    """
    POST /rag with the full 55-page corpus available.

    Checks (all must pass):
      • HTTP 200
      • confidence == "high"  (max_score ≥ 0.70)
      • answer explains LoRA vs full fine-tuning distinction
      • at least one source references the LoRA paper or LLM corpus pages
      • answer is substantive (≥ 50 chars)
      • traces/custom/query_4_with_index.json is consistent
    """
    print(f"\n{'─'*60}")
    print("test_custom_query_4_with_index")
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

    # ── answer explains LoRA vs fine-tuning ───────────────────────────────────
    answer_low     = answer.lower()
    matched_signal = next((s for s in LORA_SIGNALS if s.lower() in answer_low), None)
    assert matched_signal is not None, (
        f"answer does not explain LoRA vs fine-tuning distinction.\n"
        f"Signals checked: {LORA_SIGNALS}\n"
        f"Answer: {answer[:300]}"
    )
    _ok("answer explains LoRA vs fine-tuning", f"'{matched_signal}'")

    # ── answer is substantive ─────────────────────────────────────────────────
    assert len(answer) >= 50, (
        f"answer too short ({len(answer)} chars): {answer!r}"
    )
    _ok("answer is substantive", f"{len(answer)} chars")

    # ── sources reference LoRA / LLM corpus documents ────────────────────────
    source_strs = " ".join(
        (s.get("source", "") + " " + s.get("descriptor", "")).lower()
        for s in sources
    )
    matched_src = next((sig for sig in SOURCE_SIGNALS if sig in source_strs), None)
    assert matched_src is not None, (
        f"no source references a LoRA / LLM corpus document.\n"
        f"Signals checked: {SOURCE_SIGNALS}\n"
        f"Sources returned: {[s.get('source', '') for s in sources]}"
    )
    _ok("sources reference LoRA/LLM corpus doc", f"signal='{matched_src}'")

    # ── trace file consistency ────────────────────────────────────────────────
    assert TRACE_WITH.exists(), f"trace file missing: {TRACE_WITH}"
    trace = json.loads(TRACE_WITH.read_text())
    assert trace.get("confidence") == "high", (
        f"trace confidence='{trace.get('confidence')}' (expected 'high')"
    )
    assert isinstance(trace.get("max_score"), (int, float)) and trace["max_score"] >= 0.70, (
        f"trace max_score={trace.get('max_score')} < 0.70"
    )
    _ok("trace file consistent", f"confidence=high, max_score={trace['max_score']}")


# ── test 2: without index ─────────────────────────────────────────────────────

def test_custom_query_4_without_index() -> None:
    """
    POST /rag with FAISS index files temporarily hidden — simulates a fresh
    install with no indexed corpus (FR-7.2, FR-4.6).

    Checks (all must pass):
      • HTTP 200
      • confidence == "none"
      • answer == NO_CONTENT_SENTINEL  (exact string, no LLM call made)
      • sources == []
      • traces/custom/query_4_without_index.json is consistent
    """
    print(f"\n{'─'*60}")
    print("test_custom_query_4_without_index")
    print(f"{'─'*60}")

    with _hidden_index():
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

    assert confidence == "none", (
        f"expected confidence='none' (no index), got '{confidence}' "
        f"(max_score={max_score}) — answer: {answer[:120]!r}"
    )
    _ok("confidence == none  (no corpus available)")

    assert answer == NO_CONTENT_SENTINEL, (
        f"expected sentinel:\n  {NO_CONTENT_SENTINEL!r}\ngot:\n  {answer[:200]!r}"
    )
    _ok("answer == no-content sentinel", repr(answer))

    assert sources == [], (
        f"expected empty sources list, got {len(sources)} source(s): {sources[:2]}"
    )
    _ok("sources == []")

    # ── trace file consistency ────────────────────────────────────────────────
    assert TRACE_WITHOUT.exists(), f"trace file missing: {TRACE_WITHOUT}"
    trace = json.loads(TRACE_WITHOUT.read_text())
    assert trace.get("confidence") == "none"
    assert trace.get("answer") == NO_CONTENT_SENTINEL
    assert trace.get("sources") == []
    _ok("without-index trace file consistent")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    print("=" * 78)
    print("CUSTOM QUERY 4 — LoRA vs Full Fine-Tuning  (Keyword, FR-7.1)")
    print(f"Query: {QUERY}")
    print("=" * 78)

    results: dict[str, str] = {}

    for fn, label in [
        (test_custom_query_4_with_index,    "with_index"),
        (test_custom_query_4_without_index, "without_index"),
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
