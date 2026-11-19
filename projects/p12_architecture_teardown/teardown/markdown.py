"""Minimal, dependency-free Markdown -> HTML converter.

Supports what our docs use: ATX headings, paragraphs, fenced code
(```mermaid fences become <pre class="mermaid">), bullet/numbered lists,
pipe tables, blockquotes, hr, inline code, bold, italic, links.
All text is HTML-escaped first, so docs can't inject markup.
"""
from __future__ import annotations

import html
import re

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
        else:
            s = html.escape(p, quote=False)
            for pat, rep in _INLINE:
                s = pat.sub(rep, s)
            s = re.sub(r'href="(?!https?://|#|/|\.|[\w-]+\.html)[^"]*"', 'href="#"', s)
            out.append(s)
    return "".join(out)


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], [r for r in cells[2:]]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            lang = s[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            code = html.escape("\n".join(body))
            cls = "mermaid" if lang == "mermaid" else f"code lang-{lang or 'text'}"
            out.append(f'<pre class="{cls}">{code}</pre>' if lang == "mermaid"
                       else f'<pre class="{cls}"><code>{code}</code></pre>')
        elif not s:
            flush()
        elif m := re.match(r"^(#{1,6})\s+(.*)$", s):
            flush()
            lvl, txt = len(m.group(1)), m.group(2)
            out.append(f'<h{lvl} id="{slugify(txt)}">{inline(txt)}</h{lvl}>')
        elif re.fullmatch(r"-{3,}|\*{3,}", s):
            flush()
            out.append("<hr>")
        elif s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1].strip()):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_table(rows))
            continue
        elif s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(quote))}</blockquote>")
            continue
        elif re.match(r"^([-*]|\d+\.)\s+", s):
            flush()
            ordered = bool(re.match(r"^\d+\.", s))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{tag}>")
            continue
        else:
            para.append(s)
        i += 1
    flush()
    return "\n".join(out)
