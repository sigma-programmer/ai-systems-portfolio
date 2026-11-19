"""End-to-end demo of the 3-tier memory system (fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p07_memory_system/demo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent.parent / "common")]

from memory_system import DAY, ManualClock, MemoryManager  # noqa: E402
from memory_system.bench import benchmark, replay  # noqa: E402
from memory_system.scenario import PROBES, QUERY_DAY  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "agent_memory.db"
        clock = ManualClock()
        start = clock.now()
        print("=== Replaying 4 scripted sessions over 20 simulated days ===")
        mm, _ = replay(db, clock)
        for ep in sorted(mm.episodes.values(), key=lambda e: e.start):
            print(f"[episode {ep.session_id}] {ep.raw_tokens}->{ep.summary_tokens} tokens: "
                  f"{ep.summary[:110]}...")
        print("\nSuperseded beliefs:")
        for f in mm.semantic.facts.values():
            if f.status == "superseded":
                print(f"  - {f.text}")
        mm.close()

        print(f"\n=== New MemoryManager instance on the same SQLite file, day {QUERY_DAY:.0f} ===")
        clock.t = start + QUERY_DAY * DAY
        fresh = MemoryManager(db, clock=clock)
        print("TTL eviction at startup:", fresh.evict())
        print("Profile:", json.dumps(fresh.profile()))
        for probe in PROBES[:6] + PROBES[-1:]:
            hits = fresh.recall(probe.query, k=2, reinforce=False)
            top = hits[0] if hits else None
            print(f"Q: {probe.query}\n   -> [{top.tier}] {top.text}  (score={top.score:.3f}, "
                  f"sim={top.similarity:.2f}, imp={top.importance:.2f}, rec={top.recency:.2f})")
        print("\nPrompt context block for 'plan a team dinner for the user':")
        print(fresh.context("plan a team dinner: food allergies, diet, drink preference", k=4))
        fresh.close()

    print("\n=== Benchmark ===")
    print(json.dumps(benchmark(), indent=2))


if __name__ == "__main__":
    main()
