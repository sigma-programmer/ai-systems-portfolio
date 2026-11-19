"""Reproducible benchmark runner.

Each project's `<pkg>.bench.benchmark()` runs in its own subprocess
(isolation: no shared caches/global state, hard timeout, offline mode
forced, fixed PYTHONHASHSEED). Results are fingerprinted over their
*deterministic* fields (timing-like keys excluded) so repeated runs can be
checked for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .registry import PROJECTS, ROOT, Project

_VOLATILE = ("ms", "latency", "time", "sec", "qps", "rps", "throughput", "duration", "elapsed", "overhead",
             "pareto", "frontier")  # frontier membership is a function of measured latency

_CHILD = r"""
import importlib, json, sys
sys.path[:0] = [{common!r}, {proj!r}]
mod = importlib.import_module({pkg!r} + ".bench")
res = mod.benchmark()
sys.stdout.write("\n__BENCH_JSON__" + json.dumps(res, default=str))
"""


def _is_volatile(key: str) -> bool:
    k = key.lower()
    return any(tok in k for tok in _VOLATILE) or "wall" in k or k.endswith("_s")


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in sorted(obj.items()) if not _is_volatile(str(k))}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def stable_fingerprint(result: dict) -> str:
    blob = json.dumps(_strip_volatile(result), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_benchmark(project: Project, timeout: float = 300.0, python: str | None = None) -> dict:
    code = _CHILD.format(common=str(ROOT / "common"), proj=str(project.path), pkg=project.package)
    env = {**os.environ, "AIKIT_OFFLINE": "1", "PYTHONHASHSEED": "0"}
    env.pop("ANTHROPIC_API_KEY", None)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([python or sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout, env=env, cwd=str(project.path))
    except subprocess.TimeoutExpired:
        return {"project": project.id, "ok": False, "error": f"timeout after {timeout}s"}
    wall = time.perf_counter() - t0
    marker = "__BENCH_JSON__"
    if proc.returncode != 0 or marker not in proc.stdout:
        return {"project": project.id, "ok": False, "wall_s": round(wall, 2),
                "error": (proc.stderr or proc.stdout)[-2000:]}
    result = json.loads(proc.stdout.rsplit(marker, 1)[1])
    return {"project": project.id, "title": project.title, "ok": True,
            "wall_s": round(wall, 2), "fingerprint": stable_fingerprint(result), "metrics": result}


def environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor() or "unknown",
            "offline": True, "pythonhashseed": "0"}


def run_all(projects: list[Project] | None = None, repeats: int = 1, timeout: float = 300.0) -> dict:
    projects = projects or [p for p in PROJECTS if (p.path / p.package / "bench.py").exists()]
    runs = []
    for p in projects:
        first = run_benchmark(p, timeout)
        if first["ok"] and repeats > 1:
            fps = {first["fingerprint"]}
            for _ in range(repeats - 1):
                again = run_benchmark(p, timeout)
                if again["ok"]:
                    fps.add(again["fingerprint"])
            first["reproducible"] = len(fps) == 1
        runs.append(first)
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": environment(), "repeats": repeats, "results": runs}


def save(results: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str))
    return path
