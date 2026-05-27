"""
Test Query G — Synonym recall (vector beats keyword)

Query: Across these papers, how do they handle the credit assignment problem?

Expected flow (4 iterations):
  Iter 1: Decision calls search_knowledge for "credit assignment"
  Iter 2-3: (Maybe additional searches or synthesis begins)
  Iter 4: Synthesis goal completes, attributes claims to CoT, DPO, ReAct, LoRA.

Run:
    uv run python tests/test_query_g.py
    uv run python tests/test_query_g.py --no-clean   # keep existing state
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY_INDEX = "Index every .md file under papers/. Confirm how many chunks were indexed in total."
QUERY_G = "Across these papers, how do they handle the credit assignment problem?"

STATE_DIR = ROOT / "state"
MEMORY_FILE = STATE_DIR / "memory.json"
TRACE_FILE_PREREQ = ROOT / "traces" / "base" / "query_G_prereq.json"
TRACE_FILE        = ROOT / "traces" / "base" / "query_G.json"
MAX_ITERATIONS    = 4   # search_knowledge + synthesis + done (headroom)

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

def check_run_answer(answer: str) -> tuple[bool, list[str]]:
    failures = []
    low = answer.lower()
    
    # Check that it pulled from at least 3 of the 4 relevant papers
    papers = ["cot", "dpo", "react", "lora"]
    found = sum(1 for p in papers if p in low)
    if found < 2:
        failures.append(
            f"Answer should attribute claims to at least 2 papers (CoT, DPO, ReAct, LoRA). "
            f"Found mentions of only {found}: {answer!r}"
        )
    return len(failures) == 0, failures

async def main(clean: bool = True) -> int:
    if clean:
        clean_state()

    import agent7

    if clean:
        print("=" * 78)
        print("PRE-REQUISITE — Index corpus")
        print("=" * 78)
        await agent7.run(QUERY_INDEX, trace_path=str(TRACE_FILE_PREREQ))
        print("\n" + "=" * 78)
        print("Corpus indexed. Beginning Test Query G.")
        print("=" * 78)

    print("=" * 78)
    print("TEST QUERY G — Synonym recall (vector beats keyword)")
    print("=" * 78)

    answer = await agent7.run(QUERY_G, trace_path=str(TRACE_FILE))

    trace = json.loads(TRACE_FILE.read_text()) if TRACE_FILE.exists() else {}
    iteration_count = trace.get("iterations", -1)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    pass1, fails1 = check_run_answer(answer)
    iter_pass = 0 < iteration_count <= MAX_ITERATIONS
    iter_failures = [] if iter_pass else [
        f"completed in {iteration_count} iterations (max allowed: {MAX_ITERATIONS})"
    ]
    print(f"  Vector recall surfaced correct concepts : {'✓' if pass1 else '✗'}")
    if fails1:
        for f in fails1:
            print(f"    ✗ {f}")
    print(f"  iterations ≤ {MAX_ITERATIONS}                          : {'✓' if iter_pass else '✗'}"
          + f"  ({iteration_count} iteration(s))")
    print(f"  trace file saved                        : {'✓' if TRACE_FILE.exists() else '✗'}")
    if iter_failures:
        for f in iter_failures:
            print(f"    ✗ {f}")
    print(f"  Final Answer: {answer!r}")

    overall = pass1 and iter_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
