"""Second-stage rerankers.

* `CrossEncoderReranker` - pairwise feature model (see features.py) whose
  weights are learned with logistic regression on labeled (query, doc) pairs.
  Untrained, it falls back to sensible hand-set weights.
* `LLMReranker` - listwise relevance scoring via any `aikit.LLM`.

Both implement `rerank(query, qvec, hits) -> hits` and never drop hits.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from aikit import LLM, count_tokens, truncate_to_tokens

from .features import FEATURE_NAMES, FeatureExtractor
from .logreg import LogisticRegression
from .types import Hit

# Hand-set weights used before training (raw feature space).
DEFAULT_WEIGHTS = np.array([0.6, 1.0, 4.0, 3.0, 1.0, 2.0, 0.5, 0.5, 1.5, 0.5, 1.0])


class Reranker(Protocol):
    def rerank(self, query: str, qvec: np.ndarray, hits: Sequence[Hit]) -> list[Hit]: ...


def _reorder(hits: Sequence[Hit], scores: np.ndarray, key: str) -> list[Hit]:
    out = []
    for h, s in zip(hits, scores):
        comps = dict(h.components)
        comps[key] = float(s)
        out.append(Hit(h.index, h.doc_id, float(s), comps))
    # stable: ties keep first-stage order
    order = sorted(range(len(out)), key=lambda j: (-out[j].score, j))
    return [out[j] for j in order]


class CrossEncoderReranker:
    """Logistic-regression scorer over query-document interaction features."""

    def __init__(self, extractor: FeatureExtractor, model: LogisticRegression | None = None) -> None:
        self.extractor = extractor
        self.model = model

    @property
    def trained(self) -> bool:
        return self.model is not None and self.model.coef_ is not None

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> "CrossEncoderReranker":
        self.model = LogisticRegression(l2=l2).fit(X, y)
        return self

    def score(self, query: str, qvec: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        X = self.extractor.extract(query, qvec, indices)
        if self.trained:
            return self.model.decision_function(X)
        return X @ DEFAULT_WEIGHTS

    def rerank(self, query: str, qvec: np.ndarray, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        return _reorder(hits, self.score(query, qvec, [h.index for h in hits]), "rerank")

    def feature_weights(self) -> dict[str, float]:
        """Feature weights on the standardized scale (comparable importances)."""
        if not self.trained:
            return dict(zip(FEATURE_NAMES, DEFAULT_WEIGHTS.tolist()))
        return {n: round(float(w), 4) for n, w in zip(FEATURE_NAMES, self.model.coef_)}


RERANK_SYSTEM = (
    "You are a search relevance judge. Score how well each passage answers the query "
    "on a 0-10 scale (10 = directly answers it, 0 = unrelated)."
)


class LLMReranker:
    """Listwise LLM reranking: one prompt with numbered passages -> JSON scores.

    Expected reply: {"scores": [{"id": 1, "score": 8}, ...]} (a bare list of
    numbers is accepted too). Passages missing from the reply keep their
    first-stage order below all scored ones; an unparsable reply leaves the
    ranking unchanged, so the LLM can only ever help, never break, retrieval.
    """

    def __init__(self, llm: LLM, texts: Sequence[str], max_passage_tokens: int = 96, max_tokens: int = 512) -> None:
        self.llm = llm
        self.texts = texts
        self.max_passage_tokens = max_passage_tokens
        self.max_tokens = max_tokens
        self.last_prompt_tokens = 0
        self.last_error: str | None = None

    def build_prompt(self, query: str, hits: Sequence[Hit]) -> str:
        lines = [f"Query: {query}", "", "Passages:"]
        for n, h in enumerate(hits, start=1):
            passage = truncate_to_tokens(self.texts[h.index].replace("\n", " "), self.max_passage_tokens)
            lines.append(f"[{n}] {passage}")
        lines += ["", 'Return JSON: {"scores": [{"id": <passage number>, "score": <0-10>}, ...]}']
        return "\n".join(lines)

    def _parse(self, payload, n: int) -> dict[int, float]:
        if isinstance(payload, dict):
            payload = payload.get("scores", payload.get("results", []))
        scores: dict[int, float] = {}
        if isinstance(payload, list):
            for pos, item in enumerate(payload, start=1):
                if isinstance(item, dict) and "id" in item and "score" in item:
                    pid, sc = int(item["id"]), float(item["score"])
                elif isinstance(item, (int, float)):
                    pid, sc = pos, float(item)
                else:
                    continue
                if 1 <= pid <= n:
                    scores[pid] = sc
        return scores

    def rerank(self, query: str, qvec: np.ndarray, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        prompt = self.build_prompt(query, hits)
        self.last_prompt_tokens = count_tokens(RERANK_SYSTEM) + count_tokens(prompt)
        self.last_error = None
        try:
            resp = self.llm.complete(prompt, system=RERANK_SYSTEM, max_tokens=self.max_tokens, json_mode=True)
            scores = self._parse(resp.json(), len(hits))
        except (ValueError, TypeError, KeyError) as exc:
            self.last_error = str(exc)
            scores = {}
        if not scores:
            return list(hits)
        n = len(hits)
        # unscored passages sink below scored ones, preserving their order
        final = np.array(
            [scores.get(j, -1.0 - j / (n + 1)) for j in range(1, n + 1)], dtype=np.float64
        )
        return _reorder(hits, final, "llm")

