"""
Test Query E — Single-document index and extract

Query: Index the file papers/attention.md and tell me what the three key
       contributions of the Transformer architecture are according to this paper.

Expected flow (3 iterations):
  Iter 1: Decision dispatches index_document on papers/attention.md. Action chunks it 
          into windows, embeds them, and writes facts to memory.
  Iter 2: Perception scans memory hits, recognises chunks, emits attach hint. Decision 
          calls answer based on the text.
  Iter 3: Agent finishes.

Run:
    uv run python tests/test_query_e.py
    uv run python tests/test_query_e.py --no-clean   # keep existing state
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY = (
    "Index the file papers/attention.md and tell me what the three key "
    "contributions of the Transformer architecture are according to this paper."
)

STATE_DIR    = ROOT / "state"
ARTIFACT_DIR = STATE_DIR / "artifacts"
MEMORY_FILE  = STATE_DIR / "memory.json"

# ── state helpers ─────────────────────────────────────────────────────────────

def clean_state() -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
        print(f"[test] cleared {STATE_DIR}")
    STATE_DIR.mkdir(parents=True)
    ARTIFACT_DIR.mkdir(parents=True)


def _memory_items() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text())
    except Exception:
        return []


# ── checks ────────────────────────────────────────────────────────────────────

def check_tool_calls_in_memory() -> tuple[bool, list[str]]:
    """Memory must have records for index_document."""
    items = _memory_items()
    failures = []
    
    tool_outcomes = [
        m.get("value", {}) for m in items 
        if m.get("kind") == "tool_outcome"
    ]
    
    index_calls = [t for t in tool_outcomes if t.get("tool") == "index_document"]
    
    if not index_calls:
        failures.append("No 'index_document' tool outcome recorded in memory")
        
    return len(failures) == 0, failures


def check_facts_in_memory() -> tuple[bool, list[str]]:
    """Memory must contain fact chunks from the indexed document."""
    items = _memory_items()
    failures = []
    
    # Check for facts with descriptor starting with [sandbox:papers/attention.md
    chunks = [
        m for m in items 
        if m.get("kind") == "fact" and 
        "[sandbox:papers/attention.md" in m.get("descriptor", "")
    ]
    
    if len(chunks) < 2:
        failures.append(
            f"Expected multiple chunk facts from papers/attention.md, "
            f"found {len(chunks)}"
        )
        
    return len(failures) == 0, failures


def check_answer_content(answer: str) -> tuple[bool, list[str]]:
    """Answer must mention attention and parallel computation."""
    failures = []
    low = answer.lower()
    
    keywords = ["attention", "parallel"]
    missing = [kw for kw in keywords if kw not in low]
    
    if missing:
        failures.append(
            f"Answer is missing key contributions: {missing}"
        )
            
    return len(failures) == 0, failures


# ── main ──────────────────────────────────────────────────────────────────────

async def main(clean: bool = True) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY E — Single-document index and extract")
    print("=" * 78)

    answer = await agent7.run(QUERY)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    tools_pass, tools_failures = check_tool_calls_in_memory()
    facts_pass, facts_failures = check_facts_in_memory()
    content_pass, content_failures = check_answer_content(answer)

    # Tool outcomes check
    items = _memory_items()
    tool_outcomes = [m.get("value", {}) for m in items if m.get("kind") == "tool_outcome"]
    index_calls = [t for t in tool_outcomes if t.get("tool") == "index_document"]
    search_calls = [t for t in tool_outcomes if t.get("tool") == "search_knowledge"]
    print(f"  correct tool calls recorded          : {'✓' if tools_pass else '✗'}"
          + f"  (index_document: {len(index_calls)}, search_knowledge: {len(search_calls)})")

    # Facts check
    chunks = [m for m in items if m.get("kind") == "fact" and "[sandbox:papers/attention.md" in m.get("descriptor", "")]
    print(f"  chunks written to memory             : {'✓' if facts_pass else '✗'}"
          + f"  (found {len(chunks)} chunks)")

    # Content check
    print(f"  key contributions cited in answer    : {'✓' if content_pass else '✗'}")
    print(f"  Final answer: {answer!r}")

    all_failures = tools_failures + facts_failures + content_failures
    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = tools_pass and facts_pass and content_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
