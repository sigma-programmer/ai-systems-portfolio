"""CLI: python -m teardown {bench,cost,diagrams,adrs,site}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adr import load_adrs
from .diagrams import platform_graph, swarm_graph, to_ascii, to_mermaid
from .registry import PROJECTS
from .report import cost_markdown
from .runner import run_all, save
from .site import build_single_page, build_site, load_results

HERE = Path(__file__).resolve().parents[1]
DOCS, RESULTS = HERE / "docs", HERE / "results" / "benchmarks.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="teardown")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="run every project's benchmark in isolated subprocesses")
    b.add_argument("--only", nargs="*", help="project ids, e.g. p02 p11")
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--out", type=Path, default=RESULTS)
    sub.add_parser("cost", help="print cost-per-task table")
    d = sub.add_parser("diagrams", help="print topology diagrams")
    d.add_argument("--format", choices=["mermaid", "ascii"], default="mermaid")
    sub.add_parser("adrs", help="validate ADRs (non-zero exit on errors)")
    s = sub.add_parser("site", help="build static site from docs + results")
    s.add_argument("--out", type=Path, default=HERE / "site")
    s.add_argument("--single", type=Path, help="also write a single-page HTML here")
    a = ap.parse_args(argv)

    if a.cmd == "bench":
        projects = [p for p in PROJECTS if not a.only or p.id in a.only]
        res = run_all(projects, repeats=a.repeats)
        save(res, a.out)
        for r in res["results"]:
            flag = "ok " if r["ok"] else "ERR"
            rep = "" if "reproducible" not in r else f" reproducible={r['reproducible']}"
            print(f"[{flag}] {r['project']} {r.get('wall_s', '-')}s{rep}")
            if not r["ok"]:
                print("      " + r["error"].strip().splitlines()[-1] if r["error"].strip() else "")
        print(f"wrote {a.out}")
        return 0 if all(r["ok"] for r in res["results"]) else 1
    if a.cmd == "cost":
        print(cost_markdown())
        return 0
    if a.cmd == "diagrams":
        for name, g in (("platform", platform_graph()), ("swarm", swarm_graph())):
            print(f"--- {name} ---")
            print(to_mermaid(g) if a.format == "mermaid" else to_ascii(g))
        return 0
    if a.cmd == "adrs":
        adrs, errors = load_adrs(DOCS / "adr")
        for x in adrs:
            print(f"ADR-{x.number:04d} [{x.status}] {x.title}")
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        return 1 if errors else 0
    if a.cmd == "site":
        results = load_results(RESULTS)
        files = build_site(DOCS, a.out, results)
        print(f"wrote {len(files)} pages to {a.out}")
        if a.single:
            print("wrote", build_single_page(DOCS, a.single, results))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
