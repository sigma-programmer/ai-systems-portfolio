"""Turn benchmark results + cost model + diagrams into Markdown sections."""
from __future__ import annotations

from .cost import PRICING, compare_strategies, reference_tasks
from .diagrams import platform_graph, swarm_graph, to_mermaid


def flatten(d, prefix: str = "", max_list: int = 6) -> dict[str, object]:
    """Flatten nested metrics into dotted keys (lists summarized)."""
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}.", max_list))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d) and len(d) <= max_list:
            out[prefix.rstrip(".")] = ", ".join(map(_fmt, d))
        else:
            out[prefix.rstrip(".")] = f"[{len(d)} items]"
    else:
        out[prefix.rstrip(".")] = d
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def benchmark_markdown(results: dict, max_rows: int = 40) -> str:
    env = results.get("environment", {})
    md = ["## Reproducible benchmarks", "",
          f"Generated {results.get('generated_at', '?')} · Python {env.get('python', '?')} · "
          f"{env.get('platform', '?')} · offline mode · PYTHONHASHSEED=0 · repeats={results.get('repeats', 1)}",
          "", "Reproduce with `python -m teardown bench --repeats 2` from `projects/p12_architecture_teardown`.", "",
          "| Project | Status | Wall (s) | Reproducible | Fingerprint |", "|---|---|---|---|---|"]
    for r in results.get("results", []):
        status = "ok" if r.get("ok") else "FAILED"
        rep = {True: "yes", False: "NO", None: "n/a"}[r.get("reproducible")]
        md.append(f"| {r.get('title', r['project'])} | {status} | {r.get('wall_s', '-')} | {rep} | "
                  f"`{r.get('fingerprint', '-')}` |")
    for r in results.get("results", []):
        md += ["", f"### {r.get('title', r['project'])}", ""]
        if not r.get("ok"):
            md += ["```", str(r.get("error", ""))[-800:], "```"]
            continue
        flat = flatten(r["metrics"])
        md += ["| Metric | Value |", "|---|---|"]
        for k, v in list(flat.items())[:max_rows]:
            md.append(f"| `{k}` | {_fmt(v)} |")
        if len(flat) > max_rows:
            md.append(f"| … | {len(flat) - max_rows} more in benchmarks.json |")
    return "\n".join(md)


def cost_markdown() -> str:
    rows = compare_strategies(reference_tasks())
    md = ["## Cost-per-task analysis", "",
          "List prices (USD / million tokens): " + ", ".join(
              f"`{m}` ${p.input:g} in / ${p.output:g} out / ${p.cache_read:g} cache-read"
              for m, p in PRICING.items()) + ".", "",
          "Strategies: **always-Opus** baseline; **routed** = each call goes to the tier the "
          "complexity router (p11) assigns; **routed+cached** adds prompt caching of stable prefixes "
          "and the semantic cache hit rate for repeated tasks.", "",
          "| Task | LLM calls | Tokens | Always-Opus | Routed | Routed + cached | Savings |",
          "|---|---|---|---|---|---|---|"]
    tot = [0.0, 0.0, 0.0]
    for r in rows:
        tot[0] += r["always_opus_usd"]; tot[1] += r["routed_usd"]; tot[2] += r["routed_cached_usd"]
        md.append(f"| {r['task']} | {r['llm_calls']} | {r['tokens']:,} | ${r['always_opus_usd']:.4f} | "
                  f"${r['routed_usd']:.4f} | ${r['routed_cached_usd']:.4f} | {r['savings_pct']}% |")
    md.append(f"| **Total** | | | **${tot[0]:.4f}** | **${tot[1]:.4f}** | **${tot[2]:.4f}** | "
              f"**{100 * (1 - tot[2] / tot[0]):.1f}%** |")
    return "\n".join(md)


def diagrams_markdown() -> str:
    return "\n".join([
        "## Platform topology", "",
        "How the eleven subsystems compose into one production agent platform.", "",
        "```mermaid", to_mermaid(platform_graph(), "TB"), "```", "",
        "## Swarm topology (p08)", "",
        "Typed edges: every handoff is a Pydantic model validated at the boundary; "
        "the topology itself is type-checked when the graph is built.", "",
        "```mermaid", to_mermaid(swarm_graph(), "LR"), "```",
    ])
