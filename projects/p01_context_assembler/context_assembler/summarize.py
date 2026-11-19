"""Summarisers: extractive (offline default) and LLM-backed, both budget-exact."""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

from aikit import LLM, HashingEmbedder, count_tokens

from .formatting import Counter, fit_prefix

_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_SIGNAL = re.compile(r"\d+|\$|%|\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()] if text else []


class Summarizer(Protocol):
    def summarize(self, texts: Sequence[str], max_tokens: int, query: str | None = None) -> str: ...


class ExtractiveSummarizer:
    """Pick the most informative sentences that fit the budget.

    Each sentence is scored by
        w_c  * centrality      (cosine to the centroid of all sentences)
      + w_q  * query relevance (cosine to the query; w_c=0.10, w_q=0.65 with a
                                query, w_c=0.65 without)
      + 0.10 * position        (earlier sentences within a text score higher)
      + 0.15 * signal          (numbers, money, percentages, proper names)
    and greedily selected by score, skipping near-duplicates (cosine above
    ``redundancy``) of sentences already chosen; selected sentences are
    emitted in their original order. Texts of the form ``"Role: ..."`` keep their speaker label
    so conversation summaries stay attributable.
    """

    def __init__(self, counter: Counter = count_tokens, embedder: HashingEmbedder | None = None,
                 redundancy: float = 0.85):
        self.counter = counter
        self.embedder = embedder or HashingEmbedder()
        self.redundancy = redundancy

    def summarize(self, texts: Sequence[str], max_tokens: int, query: str | None = None) -> str:
        if max_tokens <= 0:
            return ""
        units: list[tuple[int, int, str, str]] = []  # (text idx, sent idx, label, sentence)
        for ti, text in enumerate(texts):
            label, body = "", text
            m = re.match(r"^(User|Assistant|Tool|System):\s*", text)
            if m:
                label, body = m.group(1), text[m.end():]
            for si, s in enumerate(split_sentences(body)):
                units.append((ti, si, label, s))
        if not units:
            return ""
        sents = [u[3] for u in units]
        vecs = self.embedder.embed(sents)
        centroid = vecs.mean(axis=0)
        n = float(np.linalg.norm(centroid))
        central = vecs @ (centroid / n) if n else np.zeros(len(sents))
        rel = vecs @ self.embedder.embed([query])[0] if query else np.zeros(len(sents))
        # Centrality rewards what is said often; with a query, relevance matters more.
        w_c, w_q = (0.10, 0.65) if query else (0.65, 0.0)
        scores = []
        for i, (_ti, si, _l, s) in enumerate(units):
            pos = 1.0 / (1.0 + si)
            signal = min(1.0, len(_SIGNAL.findall(s)) / 3)
            scores.append(w_c * central[i] + w_q * rel[i] + 0.10 * pos + 0.15 * signal)
        order = sorted(range(len(units)), key=lambda i: (-scores[i], i))

        # Upper-bound cost per sentence (as if it carried its speaker label); the
        # rendered summary can only be cheaper, so the greedy pack is O(n).
        # Near-duplicates of already chosen sentences are skipped (redundancy filter).
        costs = [self.counter(f"{u[2]}: {u[3]}" if u[2] else u[3]) for u in units]
        chosen: set[int] = set()
        chosen_vecs: list[np.ndarray] = []
        used = 0
        for i in order:
            if used + costs[i] > max_tokens:
                continue
            if any(float(vecs[i] @ v) > self.redundancy for v in chosen_vecs):
                continue
            chosen.add(i)
            chosen_vecs.append(vecs[i])
            used += costs[i]
        if not chosen:  # even the best sentence is too long: truncate it
            return fit_prefix(self._render(units, {order[0]}), max_tokens, self.counter)
        return self._render(units, chosen)

    @staticmethod
    def _render(units: list[tuple[int, int, str, str]], chosen: set[int]) -> str:
        parts: list[str] = []
        last_text = None
        for i in sorted(chosen):
            ti, _si, label, s = units[i]
            if ti != last_text and label:
                parts.append(f"{label}: {s}")
            else:
                parts.append(s)
            last_text = ti
        return " ".join(parts)


SUMMARY_SYSTEM = "You compress context for another model. Preserve facts, numbers, names, decisions and open questions."
SUMMARY_PROMPT = """Summarize the following content in at most {words} words.
{focus}
CONTENT:
{content}"""


class LLMSummarizer:
    """Abstractive summaries through any :class:`aikit.LLM`.

    The reply is hard-fitted to ``max_tokens`` with the same counter the
    assembler uses, so a verbose model can never break the budget. On any LLM
    error it falls back to the extractive summariser.
    """

    def __init__(self, llm: LLM, counter: Counter = count_tokens, fallback: Summarizer | None = None):
        self.llm = llm
        self.counter = counter
        self.fallback = fallback or ExtractiveSummarizer(counter)
        self.calls = 0
        self.failures = 0

    def summarize(self, texts: Sequence[str], max_tokens: int, query: str | None = None) -> str:
        if max_tokens <= 0 or not texts:
            return ""
        focus = f"Focus on what is relevant to: {query}\n" if query else ""
        prompt = SUMMARY_PROMPT.format(words=max(5, int(max_tokens * 0.7)), focus=focus, content="\n".join(texts))
        try:
            self.calls += 1
            text = self.llm.complete(prompt, system=SUMMARY_SYSTEM, max_tokens=max(64, max_tokens * 2)).text.strip()
        except Exception:  # noqa: BLE001 - any provider failure degrades to extractive
            self.failures += 1
            return self.fallback.summarize(texts, max_tokens, query)
        return fit_prefix(text, max_tokens, self.counter)
