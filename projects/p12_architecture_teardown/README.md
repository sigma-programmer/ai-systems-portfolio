# Project 12 — Public Architecture Teardown

> You're a system designer, not a wrapper importer.

The other eleven projects are the systems. This one is the **evidence layer** around them: reproducible benchmarks, cost-per-task economics, topology diagrams, Architecture Decision Records, and published deep-dives, all generated from the code rather than written by hand and left to rot.

## What's here

```
teardown/
  registry.py   single source of truth: the 11 projects, their packages and paths
  runner.py     isolated, fingerprinted benchmark runner (subprocess per project)
  cost.py       cost-per-task model: always-Opus vs routed vs routed+cached
  diagrams.py   networkx -> Mermaid / ASCII (platform + swarm topology)
  adr.py        ADR parser + validator (Pydantic), supersession checks
  markdown.py   dependency-free, HTML-escaping Markdown renderer
  report.py     benchmark/cost/diagram -> Markdown sections
  site.py       static multi-page site + single-page build
  __main__.py   CLI
docs/
  adr/          12 ADRs (validated in CI)
  deep-dives/   one write-up per project, with measured numbers
results/
  benchmarks.json   last full run (environment, fingerprints, all metrics)
```

## Reproducible benchmarks

```bash
cd projects/p12_architecture_teardown
python -m teardown bench --repeats 2          # all projects
python -m teardown bench --only p02 p11       # a subset
```

How reproducibility is enforced:

1. **Process isolation.** Each project's `bench.benchmark()` runs in a fresh subprocess with a hard timeout, so there is no shared cache or global state between projects.
2. **Pinned nondeterminism.** `AIKIT_OFFLINE=1` (no network or LLM variance), `PYTHONHASHSEED=0`, and seeded RNGs inside every benchmark. `ANTHROPIC_API_KEY` is removed from the child environment.
3. **Fingerprints.** Each result is hashed over its *deterministic* fields. Keys that measure wall-clock (`*_ms`, `latency`, `throughput`, `rps`…) and latency-derived structures (Pareto frontiers) are excluded, and floats are rounded to 4 places. With `--repeats N`, a project is marked `reproducible` only if all N fingerprints match.
4. **Environment capture.** Python version, platform and machine are recorded next to the results.

## Cost-per-task model

`python -m teardown cost` prices representative task profiles for each subsystem (token counts mirror the prompts the projects build) under three strategies:

| Strategy | Meaning |
|---|---|
| always-Opus | every call to `claude-opus-5-5` ($4 / $20 per MTok) |
| routed | each call to the tier the p11 complexity router would pick (Haiku 5.5 / Sonnet 5.5 / Opus 5.5) |
| routed + cached | plus prompt-cache reads on stable prefixes and the semantic-cache hit rate |

The portfolio-level result is about **65% lower cost per task** than always-Opus, and it is consistent with p11's measured 73.7% on a replayed workload. The cost model is a planning tool, and the p11 benchmark is the measurement.

## Diagrams

`python -m teardown diagrams [--format ascii]` renders the platform topology (how p01–p11 compose into one agent platform) and the swarm topology from networkx graphs. Diagrams are code, so they're versioned and diffable.

## ADRs

`python -m teardown adrs` validates every record: header, status (`Proposed | Accepted | Deprecated | Superseded by ADR-NNNN`), date, project refs, and four required sections. It also checks for duplicate numbers and dangling supersessions, and exits non-zero on any error so CI can enforce it.

## Site

```bash
python -m teardown site --single site/teardown.html
```

This builds `site/*.html` (index, architecture, cost, benchmarks, ADRs, one page per deep-dive) and optionally one self-contained page. Mermaid renders client-side, and all Markdown is HTML-escaped before formatting.

## Tests

```bash
python -m pytest projects/p12_architecture_teardown
```

The tests cover the cost math, strategy ordering, Mermaid/ASCII output, ADR parsing and validation (duplicates, dangling supersede, bad status), Markdown escaping (no `javascript:` links, no raw `<script>`), the runner's success, failure and timeout paths, fingerprint volatility rules, and site build plus CLI exit codes.
