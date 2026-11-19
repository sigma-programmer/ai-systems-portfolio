"""Worker process used by crash-recovery tests and the benchmark.

    python -m durable_agent.worker --db engine.db --ledger ledger.db --run-id r1 \
        [--start] [--crash-at after_effect --crash-step charge] [--hang-after charge]

Runs the order workflow for one run and prints a JSON status line. With
``--crash-at`` the process really dies (``os._exit``) at that crash point.
With ``--hang-after`` it prints ``EFFECT_DONE <step>`` after that step's side
effect and then blocks, so the parent can ``SIGKILL`` it at a known point.
"""
from __future__ import annotations

import argparse
import json
import sys

from .engine import Engine
from .faults import ExitAt, FaultInjector
from .samples import Ledger, order_workflow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="durable_agent.worker")
    ap.add_argument("--db", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--start", action="store_true", help="create the run before executing it")
    ap.add_argument("--crash-at", choices=["before_handler", "after_effect", "after_handler", "after_commit"])
    ap.add_argument("--crash-step")
    ap.add_argument("--hang-after")
    args = ap.parse_args(argv)

    ledger = Ledger(args.ledger)

    def marker(step: str) -> None:
        print(f"EFFECT_DONE {step}", flush=True)

    wf = order_workflow(ledger, hang_after=args.hang_after, on_effect=marker if args.hang_after else None)
    faults: FaultInjector = ExitAt(args.crash_at, args.crash_step) if args.crash_at else FaultInjector()
    engine = Engine(args.db, [wf], faults=faults)
    if args.start:
        engine.start("order", {"sku": "A-1", "qty": 1}, run_id=args.run_id)
    info = engine.run(args.run_id)
    print(json.dumps({"status": info.status, "path": info.path}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
