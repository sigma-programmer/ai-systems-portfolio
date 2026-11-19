"""Static site generator for the published teardown."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .adr import load_adrs
from .markdown import to_html
from .registry import PROJECTS
from .report import benchmark_markdown, cost_markdown, diagrams_markdown

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#ffffff;--accent:#c2410c;--code:#f3f1ea}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161614;--fg:#ecebe6;--muted:#a19f97;--line:#33322e;--card:#1e1e1b;--accent:#fb923c;--code:#24231f}}
:root[data-theme=dark]{--bg:#161614;--fg:#ecebe6;--muted:#a19f97;--line:#33322e;--card:#1e1e1b;--accent:#fb923c;--code:#24231f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 Inter,system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 80px}nav{border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:16px;flex-wrap:wrap;font-size:14px}
nav a{color:var(--muted);text-decoration:none}nav a:hover{color:var(--accent)}
h1{font-size:2rem;line-height:1.2;margin:.2em 0 .4em}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}h3{margin-top:1.6em}
a{color:var(--accent)}code{background:var(--code);padding:.1em .35em;border-radius:4px;font-size:.88em}
pre{background:var(--code);padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.45}pre code{background:none;padding:0}
pre.mermaid{background:var(--card);border:1px solid var(--line);text-align:center}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
blockquote{border-left:3px solid var(--accent);margin:1em 0;padding:.2em 1em;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h3{margin:0 0 4px;font-size:1rem}.card p{margin:0;color:var(--muted);font-size:14px}.card a{text-decoration:none}
"""

MERMAID = ('<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";'
           'mermaid.initialize({startOnLoad:true,theme:matchMedia("(prefers-color-scheme: dark)").matches?"dark":"neutral"});</script>')


def page(title: str, body_html: str, nav: str = "") -> str:
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{nav}<main>{body_html}</main>{MERMAID}</body></html>")


def _nav(pages: list[tuple[str, str]]) -> str:
    return "<nav>" + "".join(f'<a href="{href}">{html.escape(t)}</a>' for t, href in pages) + "</nav>"


def collect_sections(docs: Path, results: dict | None) -> dict[str, str]:
    """Return {page_slug: markdown} for every page of the site."""
    pages: dict[str, str] = {}
    cards = ["# Public Architecture Teardown", "",
             "Eleven production-grade AI subsystems, torn down: design, topology, cost-per-task, "
             "reproducible benchmarks and the decisions behind them.", ""]
    pages["index"] = "\n".join(cards)
    for f in sorted((docs / "deep-dives").glob("*.md")):
        pages[f.stem] = f.read_text()
    pages["architecture"] = "# Architecture\n\n" + diagrams_markdown()
    pages["cost"] = "# Cost per task\n\n" + cost_markdown()
    if results:
        pages["benchmarks"] = "# Benchmarks\n\n" + benchmark_markdown(results)
    adrs, errors = load_adrs(docs / "adr")
    adr_md = ["# Architecture Decision Records", "", "| ADR | Title | Status | Projects |", "|---|---|---|---|"]
    adr_md += [f"| [{a.number:04d}](#adr-{a.number:04d}) | {a.title} | {a.status} | {', '.join(a.projects)} |"
               for a in adrs]
    for a in adrs:
        adr_md += ["", f"## ADR-{a.number:04d}: {a.title}", "",
                   f"**Status:** {a.status} · **Date:** {a.date} · **Projects:** {', '.join(a.projects)}"]
        for name, body in a.sections.items():
            adr_md += ["", f"### {name}", "", body]
    if errors:
        adr_md += ["", "> Validation errors: " + "; ".join(errors)]
    pages["adrs"] = "\n".join(adr_md)
    return pages


def build_site(docs: Path, out: Path, results: dict | None = None) -> list[Path]:
    pages = collect_sections(docs, results)
    out.mkdir(parents=True, exist_ok=True)
    order = ["index", "architecture", "cost", "benchmarks", "adrs"]
    nav_items = [(s.title() if s != "adrs" else "ADRs", f"{s}.html") for s in order if s in pages]
    nav = _nav(nav_items)
    deep = [k for k in pages if k not in order]
    written = []
    for slug, md in pages.items():
        body = to_html(md)
        if slug == "index":
            items = "".join(
                f'<div class="card"><a href="{k}.html"><h3>{html.escape(_title(pages[k]))}</h3></a>'
                f"<p>{html.escape(_subtitle(k))}</p></div>" for k in deep)
            body += f'<h2 id="deep-dives">Deep dives</h2><div class="grid">{items}</div>'
        p = out / f"{slug}.html"
        p.write_text(page(_title(md), body, nav))
        written.append(p)
    return written


def build_single_page(docs: Path, out_file: Path, results: dict | None = None) -> Path:
    """Everything on one page (for publishing as a single artifact)."""
    pages = collect_sections(docs, results)
    order = ["index", "architecture", "cost"] + [k for k in pages if k not in
             ("index", "architecture", "cost", "benchmarks", "adrs")] + ["benchmarks", "adrs"]
    parts = []
    for slug in order:
        if slug not in pages:
            continue
        md = pages[slug]
        if slug != "index":  # demote headings one level under the page title
            md = "\n".join(("#" + l) if l.startswith("#") else l for l in md.splitlines())
        parts.append(f'<section id="{slug}">{to_html(md)}</section>')
    toc = "".join(f'<a href="#{s}">{html.escape(_title(pages[s]))}</a>' for s in order if s in pages and s != "index")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page("Architecture Teardown", "".join(parts), f"<nav>{toc}</nav>"))
    return out_file


def _title(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _subtitle(slug: str) -> str:
    for p in PROJECTS:
        if slug.startswith(p.id):
            return p.shows
    return ""


def load_results(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
