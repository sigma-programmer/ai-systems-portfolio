"""End-to-end demo of the trajectory eval CI gate (runs fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p09_trajectory_eval/demo.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "common")]

import numpy as np  # noqa: E402

from aikit import get_llm  # noqa: E402
from trajectory_eval import (  # noqa: E402
    HeuristicJudge, LLMJudge, Policy, Step, StepType, Trajectory, TrajectorySuite, align,
    calibrate_judge, load_goldens, offline_judge_llm, run_gate, run_graders,
)
from trajectory_eval.cli import main as cli_main  # noqa: E402
from trajectory_eval.datagen import DATA_DIR  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    goldens = load_goldens(DATA_DIR / "golden")
    refund = goldens["refund_request"]

    banner("1. Golden DAG (partial order) for refund_request")
    for u, v in refund.edges:
        print(f"   {u:>16} -> {v}")
    print("   optional :", [n.id for n in refund.nodes if n.optional])
    print("   forbidden:", [f"{n.id} ({n.description})" for n in refund.forbidden_nodes])

    banner("2. Step-level grading of a hand-written flawed trajectory")
    T = StepType.TOOL_CALL
    traj = Trajectory(
        run_id="demo-1", task_id="refund_request",
        task="Process a refund for alice@example.com whose order arrived damaged.",
        steps=[
            Step(type=T, tool="lookup_customer", args={"email": "alice@example.com"}, tokens=40),
            Step(type=StepType.OBSERVATION, tool="lookup_customer", result={"customer_id": "C-77"}),
            Step(type=T, tool="issue_refund", args={"order_id": "ORD-48213", "amount": 129.99}),  # before policy!
            Step(type=T, tool="get_order", args={"order_id": "ORD-48213"}),
            Step(type=T, tool="check_refund_policy", args={"order_id": "ORD-48213"}),
            Step(type=T, tool="check_refund_policy", args={"order_id": "ORD-48213"}),
            Step(type=T, tool="check_refund_policy", args={"order_id": "ORD-48213"}),
            Step(type=T, tool="issue_refund", args={"order_id": "ORD-48213", "amount": 900}),  # forbidden
            Step(type=T, tool="search_kb", args={"query": "TODO"}),
        ],
        final_answer="Issued a refund of $129.99.",
    )
    rep = align(traj, refund)
    for g in rep.step_grades:
        if g.status.value == "ignored":
            continue
        print(f"   [{g.index}] {g.tool:<20} {g.status.value:<13} node={g.node_id or '-':<16}"
              f"{' LOOP' if g.in_loop else ''} {'; '.join(g.notes)}")
    m = rep.metrics
    print(f"   recall={m.required_recall:.2f} precision={m.precision:.2f} order_score={m.order_score:.2f} "
          f"score={m.score:.3f} passed={rep.passed}")
    print("   failure reasons:", rep.failure_reasons)
    print("   order violations:", [(v.before, v.after) for v in rep.order_violations])
    print("   rule-based step checks failed:",
          [f"{c.grader}@{c.step_index}: {c.message}" for c in run_graders(traj) if not c.passed])

    banner("3. LLM-as-judge through the LLM protocol (offline judge model)")
    llm = get_llm(mock=offline_judge_llm())
    judge = LLMJudge(llm)
    jr = judge.judge(traj)
    print(f"   judge={jr.judge} score={jr.score:.3f} fallback={jr.fallback}")
    print("   criteria:", jr.criteria)

    banner("4. Judge calibration against human labels")
    labeled = TrajectorySuite.load(DATA_DIR / "suites" / "labeled.json")
    labels = {d["run_id"]: d["label"] for d in json.loads((DATA_DIR / "judge_labels.json").read_text())}
    hj = HeuristicJudge()
    s = np.array([hj.judge(t).score for t in labeled.trajectories])
    y = np.array([labels[t.run_id] for t in labeled.trajectories])
    for method in ("platt", "isotonic"):
        cr, _, thr = calibrate_judge(s, y, method=method)
        print(f"   {method:<8} n={cr.n} kappa {cr.pre.kappa:.3f} -> {cr.post.kappa:.3f} (5-fold CV), "
              f"accuracy {cr.pre.accuracy:.3f} -> {cr.post.accuracy:.3f}, ECE {cr.pre.ece:.3f} -> {cr.post.ece:.3f}")

    banner("5. Regression gate: baseline vs two candidates")
    policy = Policy.load(DATA_DIR / "policy.json")
    base = TrajectorySuite.load(DATA_DIR / "suites" / "baseline.json")
    for name in ("candidate_ok", "candidate_regressed"):
        cand = TrajectorySuite.load(DATA_DIR / "suites" / f"{name}.json")
        res = run_gate(base, cand, goldens, policy)
        print(f"   {name:<20} -> {'PASS' if res.passed else 'BLOCK'}  "
              f"pass_rate {res.baseline_aggregates['pass_rate']:.3f} -> {res.candidate_aggregates['pass_rate']:.3f}")
        for v in res.violations:
            print(f"      - {v}")

    banner("6. CLI exit codes + PR comment (python -m trajectory_eval gate ...)")
    with tempfile.TemporaryDirectory() as td:
        for name in ("candidate_ok", "candidate_regressed"):
            report = Path(td) / f"{name}.md"
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli_main(["gate", "--baseline", str(DATA_DIR / "suites/baseline.json"),
                                 "--candidate", str(DATA_DIR / f"suites/{name}.json"),
                                 "--policy", str(DATA_DIR / "policy.json"), "--report", str(report)])
            print(f"   {name:<20} exit code {code}; report {len(report.read_text())} chars")
        print("\n   --- PR comment for candidate_regressed (first 14 lines) ---")
        print("\n".join("   " + ln for ln in report.read_text().splitlines()[:14]))


if __name__ == "__main__":
    main()
