"""Command-line interface.

    python -m trajectory_eval gate --baseline B.json --candidate C.json --policy policy.json \
        [--goldens DIR] [--report pr_comment.md] [--json result.json]
    python -m trajectory_eval grade --suite S.json [--goldens DIR] [--run-id ID]
    python -m trajectory_eval calibrate --suite S.json --labels labels.json [--method platt|isotonic]

Exit codes: 0 = gate passed, 1 = gate blocked (policy violated), 2 = bad input/config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .align import align
from .calibration import calibrate_judge
from .datagen import DATA_DIR
from .gate import Policy, render_markdown, run_gate
from .golden import load_goldens
from .graders import run_graders
from .judge import HeuristicJudge
from .schema import TrajectorySuite

EXIT_PASS, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2
DEFAULT_GOLDENS = DATA_DIR / "golden"


def _load_goldens(path: str) -> dict:
    goldens = load_goldens(path)
    if not goldens:
        raise FileNotFoundError(f"no golden DAGs found in {path}")
    return goldens


def cmd_gate(a: argparse.Namespace) -> int:
    policy = Policy.load(a.policy)
    res = run_gate(TrajectorySuite.load(a.baseline), TrajectorySuite.load(a.candidate),
                   _load_goldens(a.goldens), policy)
    md = render_markdown(res)
    if a.report:
        Path(a.report).write_text(md)
    if a.json:
        Path(a.json).write_text(res.model_dump_json(indent=1))
    print(md)
    return EXIT_PASS if res.passed else EXIT_BLOCKED


def cmd_grade(a: argparse.Namespace) -> int:
    suite = TrajectorySuite.load(a.suite)
    goldens = _load_goldens(a.goldens)
    passed = 0
    for t in suite.trajectories:
        if a.run_id and t.run_id != a.run_id:
            continue
        g = goldens.get(t.task_id)
        if g is None:
            print(f"{t.run_id}: no golden for task {t.task_id!r}")
            continue
        r = align(t, g)
        passed += r.passed
        m = r.metrics
        print(f"{t.run_id}: {'PASS' if r.passed else 'FAIL'} score={m.score:.3f} recall={m.required_recall:.2f} "
              f"precision={m.precision:.2f} order={m.order_score:.2f} {'; '.join(r.failure_reasons)}")
        if a.run_id:
            for sg in r.step_grades:
                print(f"  [{sg.index:2d}] {sg.step_type.value:<11} {sg.tool or '':<20} {sg.status.value:<12} "
                      f"{sg.node_id or '':<20} {'LOOP ' if sg.in_loop else ''}{'; '.join(sg.notes)}")
            for c in run_graders(t):
                if not c.passed:
                    print(f"  check {c.grader} @step {c.step_index}: {c.message}")
    print(f"{passed}/{len(suite.trajectories) if not a.run_id else 1} passed")
    return EXIT_PASS


def cmd_calibrate(a: argparse.Namespace) -> int:
    suite = TrajectorySuite.load(a.suite)
    labels = {d["run_id"]: int(d["label"]) for d in json.loads(Path(a.labels).read_text())}
    judge = HeuristicJudge()
    trajs = [t for t in suite.trajectories if t.run_id in labels]
    s = np.array([judge.judge(t).score for t in trajs])
    y = np.array([labels[t.run_id] for t in trajs])
    rep, _, _ = calibrate_judge(s, y, method=a.method)
    print(rep.model_dump_json(indent=1))
    return EXIT_PASS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trajectory_eval", description="Trajectory eval CI gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="compare candidate vs baseline and enforce policy")
    g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--policy", required=True)
    g.add_argument("--goldens", default=str(DEFAULT_GOLDENS))
    g.add_argument("--report", help="write markdown PR comment here")
    g.add_argument("--json", help="write machine-readable result here")
    g.set_defaults(fn=cmd_gate)
    r = sub.add_parser("grade", help="step-level grading of a suite")
    r.add_argument("--suite", required=True)
    r.add_argument("--goldens", default=str(DEFAULT_GOLDENS))
    r.add_argument("--run-id")
    r.set_defaults(fn=cmd_grade)
    c = sub.add_parser("calibrate", help="judge/human agreement before and after calibration")
    c.add_argument("--suite", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    c.set_defaults(fn=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:  # argparse usage errors
        return EXIT_ERROR if e.code else EXIT_PASS
    try:
        return a.fn(a)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"trajectory_eval: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
