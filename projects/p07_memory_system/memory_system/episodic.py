"""Episodic memory: compressed, timestamped windows of conversation.

Compression is extractive by default: sentences are scored by
`0.5 * centrality + 0.5 * salience` (centrality = mean cosine similarity to the
other sentences in the window, so on-topic sentences win; salience from
`importance.salience_of`), assistant sentences are down-weighted (users carry
the facts), filler is dropped, and the top sentences are kept — in original
order — until a token budget of `ratio * raw_tokens` is filled.
"""
from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

from aikit import LLM, HashingEmbedder, count_tokens

from .importance import FILLER, salience_of

_SENT = re.compile(r"(?<=[.!?])\s+")


class Summarizer(Protocol):
    def summarize(self, turns: Sequence[str]) -> str: ...


def _split_turn(turn: str) -> tuple[str, list[str]]:
    role, _, text = turn.partition(": ")
    if not text:
        role, text = "user", turn
    return role, [s.strip() for s in _SENT.split(text) if s.strip()]


class ExtractiveSummarizer:
    def __init__(self, embedder: HashingEmbedder | None = None, ratio: float = 0.4,
                 min_tokens: int = 16, assistant_weight: float = 0.6):
        self.embedder = embedder or HashingEmbedder()
        self.ratio = ratio
        self.min_tokens = min_tokens
        self.assistant_weight = assistant_weight

    def summarize(self, turns: Sequence[str]) -> str:
        units: list[tuple[str, str]] = []
        for t in turns:
            role, sents = _split_turn(t)
            units.extend((role, s) for s in sents if not FILLER.match(s))
        if not units:
            return ""
        raw_tokens = sum(count_tokens(t) for t in turns)
        budget = max(self.min_tokens, int(self.ratio * raw_tokens))
        vecs = self.embedder.embed([s for _, s in units])
        sims = vecs @ vecs.T
        n = len(units)
        centrality = ((sims.sum(axis=1) - 1.0) / (n - 1)) if n > 1 else np.ones(1)
        scores = []
        for i, (role, s) in enumerate(units):
            sc = 0.5 * float(centrality[i]) + 0.5 * salience_of(s)
            if role != "user":
                sc *= self.assistant_weight
            scores.append(sc)
        chosen: list[int] = []
        used = 0
        for i in sorted(range(n), key=lambda i: -scores[i]):
            cost = count_tokens(f"{units[i][0]}: {units[i][1]}")
            if used + cost > budget and chosen:
                continue
            chosen.append(i)
            used += cost
        return " ".join(f"{units[i][0]}: {units[i][1]}" for i in sorted(chosen))


class LLMSummarizer:
    """Abstractive episode summary via any `aikit.LLM`."""

    SYSTEM = ("Summarise this conversation window in at most three sentences. Keep every "
              "personal fact, preference, date and commitment; drop small talk.")

    def __init__(self, llm: LLM, max_tokens: int = 200):
        self.llm = llm
        self.max_tokens = max_tokens

    def summarize(self, turns: Sequence[str]) -> str:
        prompt = "SUMMARIZE EPISODE\n" + "\n".join(turns)
        return self.llm.complete(prompt, system=self.SYSTEM, max_tokens=self.max_tokens).text.strip()
