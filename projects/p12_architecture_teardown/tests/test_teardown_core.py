import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from teardown import (ADR, LLMCall, PRICING, Project, TaskProfile, compare_strategies, cost_of_call,
                      load_adrs, platform_graph, run_benchmark, stable_fingerprint, to_ascii, to_mermaid)
from teardown.adr import parse_adr, validate_adr
from teardown.cost import reference_tasks, task_cost
from teardown.markdown import to_html
from teardown.report import benchmark_markdown, flatten
from teardown.site import build_single_page, build_site

HERE = Path(__file__).resolve().parents[1]

ADR_TEXT = """# ADR-0007: Use RRF for fusion
- Status: Accepted
- Date: 2026-11-01
- Projects: p02
## Context
Scores are incomparable.
## Decision
Use RRF k=60.
## Consequences
Robust, parameter-light.
## Alternatives Considered
Weighted min-max fusion.
"""


# ---------- cost ----------
def test_cost_of_call_math():
    c = LLMCall("x", 1_000_000, 1_000_000)
    assert cost_of_call(c, "claude-opus-5-5") == pytest.approx(24.0)
    cached = LLMCall("x", 1_000_000, 0, cacheable_prefix=1_000_000)
    assert cost_of_call(cached, "claude-opus-5-5", prompt_cache=True) == pytest.approx(0.20)
    with pytest.raises(KeyError):
        cost_of_call(c, "gpt-whatever")


def test_strategies_ordering_and_savings():
    rows = compare_strategies(reference_tasks())
    for r in rows:
        assert r["routed_cached_usd"] <= r["routed_usd"] <= r["always_opus_usd"]
    t = TaskProfile("t", [LLMCall("a", 1000, 100, "low")], semantic_cache_hit_rate=0.5)
    assert task_cost(t, "routed_cached") == pytest.approx(task_cost(t, "routed") * 0.5)
    with pytest.raises(ValueError):
        task_cost(t, "bogus")


def test_pricing_table_has_current_models():
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} <= set(PRICING)


# ---------- diagrams ----------
def test_mermaid_and_ascii():
    g = platform_graph()
    m = to_mermaid(g)
    assert m.startswith("flowchart LR")
    assert m.count("-->") == g.number_of_edges()
    assert 'subgraph Edge["Edge"]' in m
    a = to_ascii(g)
    assert "Security Proxy (p10)" in a and "edges:" in a
    cyc = nx.DiGraph([("a", "b"), ("b", "a")])
    assert "edges:" in to_ascii(cyc)


# ---------- ADRs ----------
def test_parse_and_validate_adr():
    adr = parse_adr(ADR_TEXT)
    assert adr.number == 7 and adr.projects == ["p02"]
    assert validate_adr(adr) == []
    bad = parse_adr(ADR_TEXT.replace("## Consequences\nRobust, parameter-light.\n", ""))
    assert any("Consequences" in e for e in validate_adr(bad))
    with pytest.raises(Exception):
        parse_adr(ADR_TEXT.replace("Accepted", "Maybe"))
    with pytest.raises(ValueError):
        parse_adr("no header")


def test_load_adrs_detects_dupes_and_dangling_supersede(tmp_path):
    (tmp_path / "a.md").write_text(ADR_TEXT)
    (tmp_path / "b.md").write_text(ADR_TEXT.replace("Accepted", "Superseded by ADR-0099"))
    adrs, errors = load_adrs(tmp_path)
    assert len(adrs) == 2
    assert any("duplicate" in e for e in errors)
    assert any("unknown ADR-0099" in e for e in errors)


def test_repo_adrs_are_valid():
    adrs, errors = load_adrs(HERE / "docs" / "adr")
    assert errors == []
    assert len(adrs) >= 8


# ---------- markdown ----------
def test_markdown_rendering_and_escaping():
    md = "# T\n\nhello **b** *i* `c<d>` [x](https://e.com) [y](javascript:alert(1))\n\n- a\n- b\n\n1. one\n\n" \
         "| h1 | h2 |\n|---|---|\n| v | w |\n\n```mermaid\nflowchart LR\n a-->b\n```\n\n> q\n\n<script>x</script>"
    h = to_html(md)
    assert '<h1 id="t">T</h1>' in h and "<strong>b</strong>" in h and "<em>i</em>" in h
    assert "<code>c&lt;d&gt;</code>" in h and 'href="https://e.com"' in h and "javascript" not in h
    assert "<ul><li>a</li><li>b</li></ul>" in h and "<ol>" in h and "<table>" in h
    assert '<pre class="mermaid">' in h and "<blockquote>" in h
    assert "<script>" not in h


# ---------- runner ----------
def _fake_project(tmp_path: Path, body: str) -> Project:
    pkg = tmp_path / "pX_fake" / "fakepkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "bench.py").write_text(body)
    return Project("pX", "pX_fake", "fakepkg", "Fake", "fake", base=tmp_path)


def test_run_benchmark_success_and_fingerprint(tmp_path):
    p = _fake_project(tmp_path, "import time\ndef benchmark():\n    print('noise')\n"
                                "    return {'recall': 0.9, 'latency_ms': time.perf_counter()}\n")
    r1, r2 = run_benchmark(p), run_benchmark(p)
    assert r1["ok"] and r1["metrics"]["recall"] == 0.9
    assert r1["fingerprint"] == r2["fingerprint"]  # latency excluded


def test_run_benchmark_failure_and_timeout(tmp_path):
    p = _fake_project(tmp_path, "def benchmark():\n    raise RuntimeError('boom')\n")
    r = run_benchmark(p)
    assert not r["ok"] and "boom" in r["error"]
    p2 = _fake_project(tmp_path / "t", "import time\ndef benchmark():\n    time.sleep(5)\n")
    assert "timeout" in run_benchmark(p2, timeout=0.5)["error"]


def test_fingerprint_ignores_volatile_keys():
    assert stable_fingerprint({"a": 1, "p50_ms": 3}) == stable_fingerprint({"a": 1, "p50_ms": 9})
    assert stable_fingerprint({"a": 1}) != stable_fingerprint({"a": 2})


# ---------- report + site ----------
def test_flatten_and_benchmark_markdown():
    f = flatten({"a": {"b": 1, "c": [1, 2]}, "d": [{"x": 1}]})
    assert f == {"a.b": 1, "a.c": "1, 2", "d": "[1 items]"}
    res = {"environment": {"python": "3"}, "results": [
        {"project": "p01", "title": "T", "ok": True, "wall_s": 1, "fingerprint": "f", "metrics": {"m": 0.5}},
        {"project": "p02", "ok": False, "error": "bad"}]}
    md = benchmark_markdown(res)
    assert "| `m` | 0.5 |" in md and "FAILED" in md


def test_build_site(tmp_path):
    res = {"environment": {}, "results": []}
    files = build_site(HERE / "docs", tmp_path / "site", res)
    names = {f.name for f in files}
    assert {"index.html", "architecture.html", "cost.html", "benchmarks.html", "adrs.html"} <= names
    assert "<!DOCTYPE html>" in (tmp_path / "site" / "index.html").read_text()
    single = build_single_page(HERE / "docs", tmp_path / "one.html", res)
    assert 'id="adrs"' in single.read_text()


def test_cli_cost_and_adrs():
    env_py = sys.executable
    out = subprocess.run([env_py, "-m", "teardown", "cost"], cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0 and "Savings" in out.stdout
    out = subprocess.run([env_py, "-m", "teardown", "adrs"], cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_fingerprint_ignores_latency_derived_frontiers():
    a = {"ndcg": 0.8, "pareto_frontier": [{"config": "a"}]}
    b = {"ndcg": 0.8, "pareto_frontier": [{"config": "a"}, {"config": "b"}]}
    assert stable_fingerprint(a) == stable_fingerprint(b)


def test_fingerprint_ignores_wall_seconds():
    assert stable_fingerprint({"acc": 1, "total_wall_s": 1.2, "x": {"chaos_wall_s": 3}}) == \
        stable_fingerprint({"acc": 1, "total_wall_s": 9.9, "x": {"chaos_wall_s": 0.1}})
