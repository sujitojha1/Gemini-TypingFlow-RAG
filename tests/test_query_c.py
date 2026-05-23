"""
Test Query C — Mom's birthday (durable memory across runs, carryover)

Run 1: My mom's birthday is 15 May 2026. Remember that and create
       reminders for two weeks before and on the day.
Run 2: When is mom's birthday?
Run 1 takes four iterations. The user's query is passed through memory.remember, the classifier extracts a fact item with the date in both the descriptor and the value, and two create_file calls produce reminder files in the sandbox. The fact item carries an embedding.

Run 2 takes three iterations and zero tool calls. The agent starts with empty in-process state. Memory's read on iteration one runs the query "When is mom's birthday?" through the gateway's embed endpoint, FAISS returns the fact item from run 1, Perception sees the date in the rendered hit, and Decision answers from memory.

Run:
    uv run python tests/test_query_c.py
    uv run python tests/test_query_c.py --no-clean   # keep existing state
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY_1 = (
    "My mom's birthday is 15 May 2026. Remember that and create "
    "reminders for two weeks before and on the day."
)
QUERY_2 = "When is mom's birthday?"

STATE_DIR   = ROOT / "state"
SANDBOX_DIR = ROOT / "sandbox"
MEMORY_FILE = STATE_DIR / "memory.json"

# ── state helpers ─────────────────────────────────────────────────────────────

def clean_state() -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
        print(f"[test] cleared {STATE_DIR}")
    STATE_DIR.mkdir(parents=True)

    # Also clean mom's birthday reminder files from sandbox to start fresh
    for f in SANDBOX_DIR.glob("*birthday*"):
        if f.is_file():
            f.unlink()
            print(f"[test] removed sandbox file {f.name}")
    for f in SANDBOX_DIR.glob("reminder*"):
        if f.is_file():
            f.unlink()
            print(f"[test] removed sandbox file {f.name}")


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

def check_run1_files() -> tuple[bool, list[str]]:
    """Two reminder files must exist in the sandbox."""
    failures = []
    files = list(SANDBOX_DIR.glob("*birthday*")) + list(SANDBOX_DIR.glob("reminder*"))
    unique_files = list({f.resolve() for f in files})
    
    if len(unique_files) < 2:
        failures.append(
            f"Expected at least 2 birthday/reminder files in sandbox, "
            f"found {len(unique_files)}: {[f.name for f in unique_files]}"
        )
    return len(failures) == 0, failures


def check_run1_memory() -> tuple[bool, list[str]]:
    """Memory must have a fact item with mom's birthday date and carrying an embedding."""
    items = _memory_items()
    failures = []
    
    fact_items = [
        m for m in items
        if m.get("kind") == "fact"
        and ("15 may 2026" in m.get("descriptor", "").lower() or "birthday" in m.get("descriptor", "").lower())
    ]
    
    if not fact_items:
        failures.append("No fact item about mom's birthday found in memory.json")
        return False, failures
        
    has_embedding = any(m.get("embedding") is not None for m in fact_items)
    if not has_embedding:
        failures.append("Fact item about mom's birthday does not carry an embedding")
        
    return len(failures) == 0, failures


def check_run2_answer(answer: str) -> tuple[bool, list[str]]:
    """Run 2 answer must contain the birthday date."""
    failures = []
    low = answer.lower()
    if "15" not in low or "may" not in low or "2026" not in low:
        failures.append(f"Run 2 answer '{answer}' does not contain '15 May 2026'")
    return len(failures) == 0, failures


def check_run2_no_tools() -> tuple[bool, list[str]]:
    """Run 2 must not call any tools (zero tool outcomes in memory for Run 2)."""
    items = _memory_items()
    failures = []
    
    # Identify Run 2 run_id by looking for the user_query item corresponding to QUERY_2
    run2_ids = {
        m.get("run_id") for m in items
        if m.get("source") == "user_query" and "when is mom" in m.get("descriptor", "").lower()
    }
    
    if not run2_ids:
        failures.append("Could not identify Run 2 run_id in memory.json")
        return False, failures
        
    run2_id = list(run2_ids)[0]
    
    # Check if there are any tool outcomes for Run 2
    tool_outcomes = [
        m for m in items
        if m.get("run_id") == run2_id and m.get("kind") == "tool_outcome"
    ]
    
    if tool_outcomes:
        tools = [m.get("value", {}).get("tool", "?") for m in tool_outcomes]
        failures.append(f"Run 2 called tools recorded in memory: {tools}")
        
    return len(failures) == 0, failures


# ── main ──────────────────────────────────────────────────────────────────────

async def main(clean: bool) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY C — Mom's birthday (durable memory across runs, carryover)")
    print("=" * 78)

    print("\n--- RUN 1: Set Birthday & Reminders ---")
    answer1 = await agent7.run(QUERY_1)

    print("\n--- RUN 2: Retrieve Birthday ---")
    answer2 = await agent7.run(QUERY_2)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    files_pass, files_failures = check_run1_files()
    mem_pass, mem_failures = check_run1_memory()
    ans_pass, ans_failures = check_run2_answer(answer2)
    tools_pass, tools_failures = check_run2_no_tools()

    # Files created check
    files = list(SANDBOX_DIR.glob("*birthday*")) + list(SANDBOX_DIR.glob("reminder*"))
    unique_files = list({f.resolve() for f in files})
    print(f"  reminder files created in sandbox   : {'✓' if files_pass else '✗'}"
          + f"  (found {len(unique_files)}: {[f.name for f in unique_files]})")

    # Memory fact item check
    items = _memory_items()
    fact_items = [
        m for m in items
        if m.get("kind") == "fact"
        and ("15 may 2026" in m.get("descriptor", "").lower() or "birthday" in m.get("descriptor", "").lower())
    ]
    has_emb = any(m.get("embedding") is not None for m in fact_items)
    print(f"  memory contains embedded fact       : {'✓' if mem_pass else '✗'}"
          + f"  ({len(fact_items)} fact item(s) found, embedded={has_emb})")

    # Run 2 answer check
    print(f"  Run 2 answers birthday correctly    : {'✓' if ans_pass else '✗'}"
          + f"  (answer: {answer2!r})")

    # Run 2 tools check
    print(f"  Run 2 executes zero tool calls      : {'✓' if tools_pass else '✗'}")

    all_failures = files_failures + mem_failures + ans_failures + tools_failures
    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = files_pass and mem_pass and ans_pass and tools_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ and sandbox/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
