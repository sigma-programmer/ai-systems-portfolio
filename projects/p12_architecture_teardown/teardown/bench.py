"""Benchmark for the teardown toolkit itself: cost model + site build."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .cost import compare_strategies, reference_tasks
from .site import build_site

DOCS = Path(__file__).resolve().parents[1] / "docs"


def benchmark() -> dict:
    rows = compare_strategies(reference_tasks())
    base = sum(r["always_opus_usd"] for r in rows)
    best = sum(r["routed_cached_usd"] for r in rows)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        pages = build_site(DOCS, Path(d), {"environment": {}, "results": []})
    return {
        "reference_tasks": len(rows),
        "always_opus_usd_total": round(base, 6),
        "routed_cached_usd_total": round(best, 6),
        "portfolio_savings_pct": round(100 * (1 - best / base), 1),
        "site_pages": len(pages),
        "site_build_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
