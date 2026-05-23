"""
Test Query A — Shannon Wikipedia (artifact attach path)

Expected flow:
  Iter 1: Decision calls fetch_url → result >4KB → stored as artifact
  Iter 2: Perception sets attach_artifact_id on extraction goal
          Decision receives attached bytes → answers with birth/death/contributions
  Iter 3: All goals done

Run:
    uv run python tests/test_query_a.py
    uv run python tests/test_query_a.py --no-clean   # keep existing state
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
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
    "birth date, death date, and three key contributions to information theory."
)

STATE_DIR    = ROOT / "state"
ARTIFACT_DIR = STATE_DIR / "artifacts"
MEMORY_FILE  = STATE_DIR / "memory.json"

# ── answer content checks ─────────────────────────────────────────────────────

BIRTH_YEAR  = "1916"
BIRTH_MONTH = ["april", "apr"]

DEATH_YEAR  = "2001"
DEATH_MONTH = ["february", "feb"]

# Each sub-list = one topic area; at least one signal must appear.
# THREE of the FIVE groups must match.
CONTRIBUTION_GROUPS = [
    # Information theory foundations: the 1948 paper, entropy, bit, the field itself
    ["mathematical theory", "theory of communication", "1948", "bit",
     "information theory", "entropy", "founded", "founding", "father of"],
    # Digital circuits / Boolean algebra / switching / relay logic
    ["boolean", "circuit", "switching", "relay", "digital", "logic design",
     "symbolic analysis"],
    # Data compression / source coding / lossless / Shannon entropy bound
    ["source coding", "data compression", "compression", "lossless",
     "huffman", "redundancy"],
    # Noisy-channel theorem / channel capacity / error correction / coding
    ["noisy channel", "noisy-channel", "channel capacity", "shannon limit",
     "error correction", "error rate", "coding theorem", "reliable transmission"],
    # Cryptography / secrecy / information-theoretic security
    ["cryptograph", "cryptanalysis", "secrecy", "one-time pad", "perfect secrecy",
     "communication theory of secrecy"],
]

MIN_GROUPS_REQUIRED = 3


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
    return json.loads(MEMORY_FILE.read_text())


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
    """At least one artifact ≥ 4 KB must exist (the Wikipedia fetch result)."""
    if not ARTIFACT_DIR.exists():
        return False, ["state/artifacts/ directory does not exist"]
    bins = list(ARTIFACT_DIR.glob("*.bin"))
    if not bins:
        return False, ["no .bin files in state/artifacts/"]
    large = [b for b in bins if b.stat().st_size >= 4096]
    if not large:
        return False, [
            f"no artifact ≥ 4 KB (sizes: {[b.stat().st_size for b in bins]})"
        ]
    return True, []


def check_memory_tool_record() -> tuple[bool, list[str]]:
    """Memory must have a tool_outcome record that carries an artifact_id."""
    items = _memory_items()
    if not items:
        return False, ["state/memory.json is empty or missing"]
    with_artifact = [
        m for m in items
        if m.get("kind") == "tool_outcome" and m.get("artifact_id") is not None
    ]
    if not with_artifact:
        kinds = list({m.get("kind") for m in items})
        tools = [m.get("value", {}).get("tool", "?")
                 for m in items if m.get("kind") == "tool_outcome"]
        return False, [
            f"no tool_outcome with artifact_id in memory.json "
            f"(kinds: {kinds}, tools called: {tools})"
        ]
    return True, []


def check_bio_dates(answer: str) -> tuple[bool, list[str]]:
    """Birth/death dates must appear in the final answer or memory records."""
    low      = answer.lower()
    mem_text = _memory_full_text()
    combined = low + " " + mem_text
    failures = []

    if BIRTH_YEAR not in combined:
        failures.append(f"birth year '{BIRTH_YEAR}' not in answer or memory")
    if not any(m in combined for m in BIRTH_MONTH):
        failures.append("birth month (April) not in answer or memory")
    if DEATH_YEAR not in combined:
        failures.append(f"death year '{DEATH_YEAR}' not in answer or memory")
    if not any(m in combined for m in DEATH_MONTH):
        failures.append("death month (February) not in answer or memory")

    return len(failures) == 0, failures


def check_contributions(answer: str) -> tuple[bool, list[str]]:
    """≥ MIN_GROUPS_REQUIRED of the 5 contribution areas must appear."""
    low     = answer.lower()
    matched = [
        i for i, group in enumerate(CONTRIBUTION_GROUPS, 1)
        if any(signal in low for signal in group)
    ]
    if len(matched) < MIN_GROUPS_REQUIRED:
        missing = [i for i in range(1, len(CONTRIBUTION_GROUPS) + 1)
                   if i not in matched]
        return False, [
            f"only {len(matched)}/{len(CONTRIBUTION_GROUPS)} contribution groups matched "
            f"(need ≥{MIN_GROUPS_REQUIRED}); missing group(s): {missing}; "
            f"signals: {[CONTRIBUTION_GROUPS[i - 1][:3] for i in missing]}"
        ]
    return True, []


# ── main ──────────────────────────────────────────────────────────────────────

async def main(clean: bool) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY A — Claude Shannon Wikipedia (artifact attach path)")
    print("=" * 78)

    answer = await agent7.run(QUERY)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    low      = answer.lower()
    mem_text = _memory_full_text()

    bio_pass, bio_failures = check_bio_dates(answer)
    con_pass, con_failures = check_contributions(answer)
    art_pass, art_failures = check_artifact_store()
    mem_pass, mem_failures = check_memory_tool_record()

    # Birth / death — note whether found in answer or memory fallback
    for year, months, label in [
        (BIRTH_YEAR, BIRTH_MONTH, "birth  (April 30 1916)"),
        (DEATH_YEAR, DEATH_MONTH, "death  (Feb 24 2001) "),
    ]:
        in_ans = year in low and any(m in low for m in months)
        in_mem = not in_ans and year in mem_text and any(m in mem_text for m in months)
        src    = "answer" if in_ans else ("memory" if in_mem else "—")
        ok     = in_ans or in_mem
        print(f"  {label} found : {'✓' if ok else '✗'}  ({src})")

    # Contributions
    matched_count = 0
    for i, group in enumerate(CONTRIBUTION_GROUPS, 1):
        hit = next((s for s in group if s in low), None)
        if hit:
            matched_count += 1
        print(f"  contribution group {i}                  : {'✓' if hit else '✗'}"
              + (f"  ('{hit}')" if hit else f"  (signals: {group[:3]}...)"))
    print(f"  ≥{MIN_GROUPS_REQUIRED} of {len(CONTRIBUTION_GROUPS)} groups matched       : "
          f"{'✓' if matched_count >= MIN_GROUPS_REQUIRED else '✗'}  ({matched_count} matched)")

    # Artifact store
    bins       = list(ARTIFACT_DIR.glob("*.bin")) if ARTIFACT_DIR.exists() else []
    large_bins = [b for b in bins if b.stat().st_size >= 4096]
    print(f"  artifact ≥4KB stored                 : {'✓' if art_pass else '✗'}"
          + f"  ({len(large_bins)} artifact(s) ≥4KB"
          + (f", largest {max(b.stat().st_size for b in large_bins):,}B)" if large_bins else ")"))

    # Memory tool record
    items      = _memory_items()
    tool_names = [m.get("value", {}).get("tool", "?")
                  for m in items if m.get("kind") == "tool_outcome"]
    print(f"  memory records tool outcome          : {'✓' if mem_pass else '✗'}"
          + f"  (tools: {tool_names})")

    all_failures = bio_failures + con_failures + art_failures + mem_failures
    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = bio_pass and con_pass and art_pass and mem_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
