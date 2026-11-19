# AI Systems Engineering Portfolio

Twelve production-grade AI systems projects, built from first principles: context engineering, retrieval, knowledge graphs, grounded RAG, durable agents, protocol-level tooling, memory, multi-agent orchestration, evaluation, security, cost control, and a public teardown that ties them together.

**1,142 tests · 12 runnable demos · 11 reproducible benchmarks · 12 ADRs · 0 network calls required**

| # | Project | Shows | Headline result (measured) |
|---|---|---|---|
| 01 | [Context Assembler Engine](projects/p01_context_assembler) | You master the context window, not fight it | 0 overflows over 250 randomized assemblies; 100% critical-info retention vs 0% for naive truncation; 78% fewer input tokens |
| 02 | [Hybrid Reranking Retrieval](projects/p02_hybrid_retrieval) | You build retrieval that actually finds things | MRR 0.59 (BM25) → 0.73 (hybrid) → **0.95** (hybrid + rerank); Recall@10 0.99 |
| 03 | [GraphRAG Knowledge Pipeline](projects/p03_graphrag) | You can make AI reason across connected data | Multi-hop accuracy **1.00** vs 0.14 for vector RAG, with 30% less context |
| 04 | [Self-Correcting Agentic RAG](projects/p04_agentic_rag) | You can stop RAG from confidently lying | 140/140 injected hallucinations caught, 0 leaked; abstention P/R 1.0/1.0 |
| 05 | [Durable Agent State Machine](projects/p05_durable_agent) | You build agents that survive production failures | 200 workflows, 112 crashes, **0 duplicate side effects**; real SIGKILL recovery |
| 06 | [Raw MCP Tool Server](projects/p06_mcp_server) | You understand tooling at the protocol level | No SDK; 54/54 fuzz cases handled correctly; 29/29 sandbox escapes blocked; ~67k req/s over stdio |
| 07 | [3-Tier Memory System](projects/p07_memory_system) | You can build agents that truly remember | precision@1 0.83 vs 0.42 flat store; stale facts never outrank current ones |
| 08 | [Typed Multi-Agent Swarm](projects/p08_agent_swarm) | You orchestrate swarms, not fragile single agents | Task success 0% → 93%; with an adversarial reviewer, verdict accuracy stays 1.00 |
| 09 | [Trajectory Eval CI Gate](projects/p09_trajectory_eval) | You grade the journey, not just the answer | Statistical gate: FPR **0%** vs 50% for a naive gate; judge κ 0.00 → 0.53 after calibration |
| 10 | [Inter-Agent Security Proxy](projects/p10_security_proxy) | You can secure agents against each other | Held-out injection P/R 1.00/0.84 with 0% FPR; PII recall 1.0 across 8 types |
| 11 | [Semantic Cache + Routing Gateway](projects/p11_semantic_gateway) | You can slash inference costs by 40-60% | **73.7% cost reduction** vs always-Opus, 0 cache false hits, 99.8% success under outage |
| 12 | [Public Architecture Teardown](projects/p12_architecture_teardown) | You're a system designer, not a wrapper importer | Fingerprinted benchmark harness, cost-per-task model, diagrams-as-code, validated ADRs, static site |

Each project has its own README (architecture, design decisions, measured numbers). The cross-cutting write-ups live in [`projects/p12_architecture_teardown/docs/deep-dives`](projects/p12_architecture_teardown/docs/deep-dives) and the decisions in [`docs/adr`](projects/p12_architecture_teardown/docs/adr).

## How they fit together

```
 Client ──► Security Proxy (p10) ──► Swarm Supervisor (p08) ──► Durable State Machine (p05)
                                                                   │          │          │
                                                       Context Assembler   MCP Tools   Trajectory
                                                            (p01)            (p06)     CI Gate (p09)
                                                       │     │      │
                                               Memory (p07)  │   Semantic Gateway (p11) ──► Claude
                                                       Agentic RAG (p04)
                                                       │              │
                                             Hybrid Retrieval (p02)  GraphRAG (p03)
```

## See it all running: the dashboard

```bash
make dashboard     # → http://127.0.0.1:8765
```

One local web app for all 12 projects:
- run any project's **demo**, **test suite** or **benchmark** live, with output streamed to the page
- **Run all tests** fills in a pass/fail light for every project
- **7 interactive playgrounds** where you type input and the real system responds: pack a context window (p01), compare four retrievers (p02), ask multi-hop questions (p03), attack the RAG with a hallucinating generator (p04), test memory across a process restart (p07), send injection attacks between agents (p10), route prompts and hit the semantic cache (p11)

`make snapshot` records every run into a static `dashboard/snapshot.html` you can open or share without Python.

## Quickstart

```bash
make setup         # venv + numpy, networkx, pydantic, anthropic, pytest
make test          # 1,142 tests, ~40 s, fully offline
make demos         # run all 12 end-to-end demos
make bench         # reproducible benchmarks → projects/p12_architecture_teardown/results/benchmarks.json
make site          # static teardown site → projects/p12_architecture_teardown/site/
make dashboard     # live control panel at http://127.0.0.1:8765
```

Run a single project:

```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p05_durable_agent/demo.py
.venv/bin/python -m pytest projects/p05_durable_agent
```

## Design principles

- **Offline-first, LLM-pluggable** ([ADR-0001](projects/p12_architecture_teardown/docs/adr/0001-offline-first-llm-protocol.md)). Every project depends on one small `LLM` protocol in `common/aikit`. Each LLM-backed step ships a *genuine* algorithmic default (pattern extraction, extractive summarization, heuristic judges, program search), so the whole portfolio tests and benchmarks deterministically in seconds. Set `ANTHROPIC_API_KEY` (and unset `AIKIT_OFFLINE`), and `get_llm()` returns a real Claude client: `claude-opus-5-5`, adaptive thinking, server-side refusal fallback.
- **Minimal dependencies.** numpy, networkx, pydantic, anthropic. BM25, IVF, logistic regression, Platt/isotonic calibration, JSON-RPC, MCP, event sourcing, community summaries: all implemented directly.
- **Measured, not claimed.** Every project has a `bench.py`. The p12 harness runs each one in an isolated subprocess, fingerprints the deterministic fields, and verifies they reproduce across repeated runs.
- **Honest numbers.** Several benchmarks score 1.0 on bundled synthetic sets. Where that's a development-set artifact, the README says so, and held-out numbers are reported where they exist (e.g. p10 held-out recall 0.84, p04 paraphrase false-flag rate 16%, p08's one deliberately preserved correlated-error failure).

## Repository layout

```
common/aikit/          shared: LLM protocol (Claude + Mock), token estimator, offline embeddings
projects/pNN_*/        <package>/  tests/  demo.py  README.md  (<package>/bench.py)
dashboard/             live control panel (server.py, playgrounds.py, index.html)
.github/workflows/     CI: tests + reproducible benchmarks + ADR validation
Makefile  pyproject.toml  requirements.txt
```
