"""
Test Query H — Cross-document synthesis

Query: Compare how the ReAct paper and the Chain-of-Thought paper differ
       in their treatment of intermediate reasoning.

Expected flow (3 iterations):
  Iter 1: Decision calls search_knowledge for ReAct and Chain-of-Thought
  Iter 2: Synthesis goal completes, comparing ReAct's interleaving of reasoning
          and actions with CoT's linear stepwise reasoning.

Run:
    uv run python tests/test_query_h.py
    uv run python tests/test_query_h.py --no-clean   # keep existing state
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
QUERY_H = "Compare how the ReAct paper and the Chain-of-Thought paper differ in their treatment of intermediate reasoning."

STATE_DIR = ROOT / "state"
TRACE_FILE_PREREQ = ROOT / "traces" / "base" / "query_H_prereq.json"
TRACE_FILE        = ROOT / "traces" / "base" / "query_H.json"
MAX_ITERATIONS    = 3   # search_knowledge + synthesis + done

def clean_state() -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
        print(f"[test] cleared {STATE_DIR}")
    STATE_DIR.mkdir(parents=True)

def check_run_answer(answer: str) -> tuple[bool, list[str]]:
    failures = []
    low = answer.lower()
    
    if "react" not in low:
        failures.append("Answer does not mention ReAct paper.")
    
    if "chain-of-thought" not in low and "chain of thought" not in low and "cot" not in low:
        failures.append("Answer does not mention Chain-of-Thought paper.")
        
    # Check that ReAct's distinction (actions/tools/interleaving) is mentioned
    react_concepts = ["interleav", "action", "tool", "external", "environment"]
    if not any(c in low for c in react_concepts):
        failures.append("Answer fails to mention ReAct's interleaving of reasoning with actions/tools.")
        
    # Check that CoT's distinction (step-by-step/linear/prompting) is mentioned
    cot_concepts = ["step-by-step", "step by step", "linear", "prompt", "intermediate step", "trace"]
    if not any(c in low for c in cot_concepts):
        failures.append("Answer fails to mention CoT's linear or step-by-step reasoning structure.")
        
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
        print("Corpus indexed. Beginning Test Query H.")
        print("=" * 78)

    print("=" * 78)
    print("TEST QUERY H — Cross-document synthesis")
    print("=" * 78)

    answer = await agent7.run(QUERY_H, trace_path=str(TRACE_FILE))

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
    print(f"  Synthesis accurately compares ReAct and CoT : {'✓' if pass1 else '✗'}")
    if fails1:
        for f in fails1:
            print(f"    ✗ {f}")
    print(f"  iterations ≤ {MAX_ITERATIONS}                            : {'✓' if iter_pass else '✗'}"
          + f"  ({iteration_count} iteration(s))")
    print(f"  trace file saved                             : {'✓' if TRACE_FILE.exists() else '✗'}")
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
