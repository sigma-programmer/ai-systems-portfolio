"""End-to-end demo of the typed multi-agent swarm (fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p08_agent_swarm/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "common")]

from aikit import MockLLM  # noqa: E402
from agent_swarm import (  # noqa: E402
    TASKS_BY_NAME, Budget, Coder, ConsensusPanel, ExecutionReviewer, Researcher, Rule, Sandbox, Supervisor,
    Topology, TopologyError, build_supervisor, build_topology, passes_hidden, transcript,
)
from agent_swarm.bench import benchmark  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    sandbox = Sandbox()

    banner("1. Typed topology: built and type-checked before anything runs")
    topo = build_topology("swarm_consensus", rule=Rule.WEIGHTED)
    print(topo.describe())
    bad = Topology().add(Researcher("researcher"), Coder("coder"), ConsensusPanel("panel"))
    bad.connect("researcher", "coder").connect("coder", "panel")  # CodeArtifact -> panel expecting ReviewVerdict
    try:
        bad.validate()
    except TopologyError as e:
        print("\nRejected ill-typed topology (coder wired straight into the panel):")
        print("  " + str(e).replace("\n", "\n  "))

    banner("2. Swarm with consensus on 'merge_intervals' (CEGIS revise loop)")
    task = TASKS_BY_NAME["merge_intervals"]
    st = build_supervisor("swarm_consensus", sandbox=sandbox, rule=Rule.WEIGHTED).run(task.spec)
    print(transcript(st))
    print(f"\nPlan subgoals: {st.plan.subgoals if st.plan else []}")
    print(f"Final code ({st.final_artifact.origin}):\n{st.final_artifact.source}")
    print(f"Passes hidden tests: {passes_hidden(st.final_artifact, task, sandbox)}")

    banner("3. Single agent on the same task (no research, no review)")
    st1 = build_supervisor("single_agent", sandbox=sandbox).run(task.spec)
    print(f"status={st1.status}  origin={st1.final_artifact.origin}  "
          f"passes hidden tests: {passes_hidden(st1.final_artifact, task, sandbox)}")

    banner("4. Fault injection: adversarial reviewer joins the panel ('second_largest')")
    task = TASKS_BY_NAME["second_largest"]
    st = build_supervisor("swarm_consensus", sandbox=sandbox, rule=Rule.MAJORITY, faulty="adversarial").run(task.spec)
    for e in st.events:
        if e.kind in ("output", "escalation", "revise", "terminate") and e.node in ("rogue", "exec", "panel", "supervisor", "coder"):
            print(f"  {e.kind:<11} {e.node:<10} {e.detail[:100]}")
    print(f"Passes hidden tests: {passes_hidden(st.final_artifact, task, sandbox)}  "
          f"(escalations settled by re-execution: {st.escalations})")
    st_bad = build_supervisor("swarm_review", sandbox=sandbox, faulty="adversarial").run(task.spec)
    print(f"Same adversary as the ONLY reviewer: status={st_bad.status}, "
          f"passes hidden tests: {passes_hidden(st_bad.final_artifact, task, sandbox)}")

    banner("5. Pluggable LLM coder with structured repair (MockLLM, offline)")
    replies = iter([
        "Sure! Here is the code you asked for.",                           # not JSON
        json.dumps({"source": "def chunks(xs, size):\n    return []\n"}),  # wrong function name
        json.dumps({"source": "def chunk(xs, size):\n    if size <= 0:\n        raise ValueError('size')\n"
                              "    return [xs[i:i + size] for i in range(0, len(xs), size)]\n",
                    "rationale": "validated size, slicing"}),
    ])
    llm = MockLLM(default=lambda p, s: next(replies, "{}"))
    task = TASKS_BY_NAME["chunk"]
    topo = Topology().add(Researcher("researcher"), Coder("coder", llm=llm), ExecutionReviewer("exec"),
                          ConsensusPanel("panel"))
    topo.connect("researcher", "coder").connect("coder", "exec").connect("exec", "panel")
    topo.connect("panel", "coder", feedback=True).validate()
    st = Supervisor(topo, budget=Budget(max_revisions=2), sandbox=sandbox).run(task.spec)
    for e in st.events:
        if e.node == "coder" or e.kind == "terminate":
            print(f"  {e.kind:<17} {e.node:<10} {e.detail[:100]}")
    print(f"LLM calls: {len(llm.calls)}, repairs: {st.repairs}, "
          f"passes hidden tests: {passes_hidden(st.final_artifact, task, sandbox)}")

    banner("6. Budget enforcement (token budget too small)")
    st = build_supervisor("swarm_consensus", sandbox=sandbox, budget=Budget(max_tokens=800)).run(
        TASKS_BY_NAME["palindrome"].spec)
    print(f"status={st.status} steps={st.steps} tokens={st.tokens}")

    banner("7. Benchmark (15 bundled spec tasks)")
    r = benchmark()
    print(f"{'configuration':<30}{'success':>8}{'verdict acc':>12}{'revs':>6}{'tokens':>8}{'escal':>6}{'wall s':>8}")
    for group in ("modes", "resilience"):
        for name, m in r[group].items():
            acc = "-" if m["verdict_accuracy"] is None else f"{m['verdict_accuracy']:.3f}"
            print(f"{name:<30}{m['success_rate']:>8.3f}{acc:>12}{m['avg_revisions']:>6.2f}"
                  f"{m['avg_tokens']:>8.0f}{m['escalations']:>6}{m['wall_s']:>8.2f}")
    print(f"total benchmark wall time: {r['total_wall_s']}s")


if __name__ == "__main__":
    main()
