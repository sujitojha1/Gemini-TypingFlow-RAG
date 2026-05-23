"""
Test Query D — Asyncio research (multi-source synthesis, carryover)

Query: Search for "Python asyncio best practices", read the top 3 results,
       and give me a short numbered list of the advice they agree on.

Expected flow (6 iterations):
  Iter 1: Decision calls web_search on "Python asyncio best practices"
  Iter 2-4: Decision fetches the top 3 result URLs using fetch_url (each stored as an artifact)
  Iter 5: Decision synthesises the common advice from the three attached artifacts
  Iter 6: All goals done

Run:
    uv run python tests/test_query_d.py
    uv run python tests/test_query_d.py --no-clean   # keep existing state
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
    'Search for "Python asyncio best practices", read the top 3 results, '
    'and give me a short numbered list of the advice they agree on.'
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


def _memory_full_text() -> str:
    parts: list[str] = []
    for m in _memory_items():
        parts.append(m.get("descriptor", ""))
        val = m.get("value", {})
        if isinstance(val, dict):
            parts.extend(str(v) for v in val.values())
    return " ".join(parts).lower()


# ── checks ────────────────────────────────────────────────────────────────────

def check_artifact_store() -> tuple[bool, list[str]]:
    """At least 3 artifacts (fetched pages) ≥ 2 KB must exist in state/artifacts/."""
    failures = []
    if not ARTIFACT_DIR.exists():
        return False, ["state/artifacts/ directory does not exist"]
    bins = list(ARTIFACT_DIR.glob("*.bin"))
    large = [b for b in bins if b.stat().st_size >= 2048]
    
    if len(large) < 3:
        failures.append(
            f"Expected at least 3 fetched page artifacts ≥ 2 KB, "
            f"found {len(large)} (sizes: {[b.stat().st_size for b in bins]})"
        )
    return len(failures) == 0, failures


def check_tool_calls_in_memory() -> tuple[bool, list[str]]:
    """Memory must have records for 1 web_search and at least 3 fetch_url calls."""
    items = _memory_items()
    failures = []
    
    tool_outcomes = [
        m.get("value", {}) for m in items 
        if m.get("kind") == "tool_outcome"
    ]
    
    searches = [t for t in tool_outcomes if t.get("tool") == "web_search"]
    fetches = [t for t in tool_outcomes if t.get("tool") == "fetch_url"]
    
    if not searches:
        failures.append("No 'web_search' tool outcome recorded in memory")
    if len(fetches) < 3:
        failures.append(
            f"Expected at least 3 'fetch_url' tool outcomes recorded in memory, "
            f"found {len(fetches)}"
        )
        
    return len(failures) == 0, failures


def check_synthesised_list(answer: str) -> tuple[bool, list[str]]:
    """Answer must contain a numbered list of asyncio-related advice."""
    failures = []
    low = answer.lower()
    
    # Check for asyncio-related keywords in the synthesis
    keywords = ["asyncio", "loop", "task", "gather", "run", "create_task", "await"]
    matches = [kw for kw in keywords if kw in low]
    if len(matches) < 2:
        failures.append(
            f"Answer does not seem highly focused on asyncio concepts. "
            f"Found keywords: {matches}"
        )
        
    # Check for numbered list indicators (e.g., '1.', '2.', '3.')
    indicators = ["1.", "2.", "3."]
    if not all(ind in low for ind in indicators):
        # Allow alternate list formats like 1), 2), 3) or bullet lists if they are formatted cleanly,
        # but let's check for standard numbering first.
        has_numbered = any(f"{i}." in low for i in (1, 2, 3)) or any(f"{i})" in low for i in (1, 2, 3))
        if not has_numbered:
            failures.append(f"Answer does not appear to be a numbered list: {answer!r}")
            
    return len(failures) == 0, failures


# ── main ──────────────────────────────────────────────────────────────────────

async def main(clean: bool) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY D — Asyncio research (multi-source synthesis, carryover)")
    print("=" * 78)

    answer = await agent7.run(QUERY)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    art_pass, art_failures = check_artifact_store()
    tools_pass, tools_failures = check_tool_calls_in_memory()
    list_pass, list_failures = check_synthesised_list(answer)

    # Artifacts check
    bins = list(ARTIFACT_DIR.glob("*.bin")) if ARTIFACT_DIR.exists() else []
    large_bins = [b for b in bins if b.stat().st_size >= 2048]
    print(f"  artifacts ≥ 2KB stored in state      : {'✓' if art_pass else '✗'}"
          + f"  (found {len(large_bins)} artifact(s) ≥ 2KB"
          + (f", sizes: {[b.stat().st_size for b in large_bins]})" if large_bins else ")"))

    # Tool outcomes check
    items = _memory_items()
    tool_outcomes = [m.get("value", {}) for m in items if m.get("kind") == "tool_outcome"]
    searches = [t for t in tool_outcomes if t.get("tool") == "web_search"]
    fetches = [t for t in tool_outcomes if t.get("tool") == "fetch_url"]
    print(f"  correct tool calls recorded          : {'✓' if tools_pass else '✗'}"
          + f"  (web_search: {len(searches)}, fetch_url: {len(fetches)})")

    # Synthesis list check
    print(f"  numbered list synthesised correctly  : {'✓' if list_pass else '✗'}")
    print(f"  Final answer: {answer!r}")

    all_failures = art_failures + tools_failures + list_failures
    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = art_pass and tools_pass and list_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
