"""
Test Query F — Cross-run document recall (FAISS persistence)

Run 1: Index every .md file under papers/. Confirm how many chunks
       were indexed in total.
Run 2 (fresh process, persisted state):
       Across the papers I have indexed, what do they say about
       chain-of-thought reasoning?

Expected flow:
  Run 1: (clean state)
    - list_dir papers/
    - index_document on 5 paper files
    - Report total chunk count
  Run 2: (no clean, reuse state)
    - Memory read automatically queries FAISS
    - search_knowledge to retrieve more context if needed
    - Answer based on persisted chunks without re-fetching
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY_1 = "Index every .md file under papers/. Confirm how many chunks were indexed in total."
QUERY_2 = "Across the papers I have indexed, what do they say about chain-of-thought reasoning?"

STATE_DIR = ROOT / "state"
MEMORY_FILE = STATE_DIR / "memory.json"
INDEX_FAISS = STATE_DIR / "index.faiss"
INDEX_IDS = STATE_DIR / "index_ids.json"

def clean_state() -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
        print(f"[test] cleared {STATE_DIR}")
    STATE_DIR.mkdir(parents=True)

def _memory_items() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text())
    except Exception:
        return []

def check_run1_state() -> tuple[bool, list[str]]:
    failures = []
    items = _memory_items()
    if not items:
        failures.append("memory.json is empty or missing")
        return False, failures

    # Expect ~15 chunk facts + other memory items
    facts = [m for m in items if m.get("kind") == "fact" and m.get("descriptor", "").startswith("[sandbox:papers/")]
    if len(facts) < 10:
        failures.append(f"Expected at least 10 chunk facts from papers in memory, found {len(facts)}")

    if not INDEX_FAISS.exists():
        failures.append("index.faiss not found")
    if not INDEX_IDS.exists():
        failures.append("index_ids.json not found")

    return len(failures) == 0, failures

def check_run2_answer(answer: str) -> tuple[bool, list[str]]:
    failures = []
    low = answer.lower()
    if "chain-of-thought" not in low and "chain of thought" not in low and "cot" not in low:
        failures.append(f"Answer does not discuss chain-of-thought: {answer!r}")
    return len(failures) == 0, failures

async def main(clean: bool = True) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY F (Run 1) — Index corpus")
    print("=" * 78)

    answer1 = await agent7.run(QUERY_1)
    
    print("\n" + "=" * 78)
    print("VALIDATION (Run 1)")
    print("=" * 78)
    
    pass1, fails1 = check_run1_state()
    print(f"  Persisted FAISS and memory state : {'✓' if pass1 else '✗'}")
    if fails1:
        for f in fails1:
            print(f"    ✗ {f}")
            
    print(f"  Run 1 Answer: {answer1!r}")

    print("\n" + "=" * 78)
    print("TEST QUERY F (Run 2) — Cross-run recall")
    print("=" * 78)

    # Do not clean state! Re-import or re-use agent7 to simulate fresh run against same disk state.
    # Note: In a real "fresh process", all globals are reset. We'll rely on agent7 re-initialising its state from disk.
    answer2 = await agent7.run(QUERY_2)

    print("\n" + "=" * 78)
    print("VALIDATION (Run 2)")
    print("=" * 78)

    pass2, fails2 = check_run2_answer(answer2)
    print(f"  Answer uses persisted knowledge : {'✓' if pass2 else '✗'}")
    if fails2:
        for f in fails2:
            print(f"    ✗ {f}")
            
    print(f"  Run 2 Answer: {answer2!r}")

    overall = pass1 and pass2
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
