"""Citation verification and hallucination detection.

`HeuristicEntailment` decides whether a cited passage entails a claim:

* find the passage window (one sentence, or two adjacent sentences if that
  helps) with the best IDF-weighted coverage of the claim's content terms;
* **numbers**: every number in the claim must appear in the passage; a missing
  number when the window states a different one is a *contradiction*;
* **units**: a number whose following unit word differs from the unit the
  passage attaches to the same number is a contradiction (18 hours vs 18 minutes);
* **negation**: a differing count of negation cues between claim and its best
  window (with high lexical overlap) is a *contradiction* ("is not
  waterproof" vs "is waterproof");
* **entities**: a proper noun in the claim absent from the passage is a
  contradiction if the rest matches (entity swap), otherwise unsupported;
* otherwise the claim is *entailed* iff coverage >= `min_coverage`.

`HallucinationDetector` applies a verifier to every claim/citation pair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from aikit import LLM, tokenize_words

from .generator import Claim
from .grader import TermStats
from .text import NEGATIONS, STOPWORDS, entities_in, numbers_in, split_sentences, stem

ENTAILED, CONTRADICTED, UNSUPPORTED = "entailed", "contradicted", "unsupported"


@dataclass
class Verdict:
    label: str
    score: float
    reason: str
    doc_id: str | None = None
    evidence: str = ""


class EntailmentChecker(Protocol):
    def check(self, claim: str, passage: str) -> Verdict: ...


_NUM_UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)\s+([a-zA-Z]+)")


def _unit_pairs(text: str) -> dict[str, set[str]]:
    """Map each number to the stemmed words that directly follow it (its unit)."""
    out: dict[str, set[str]] = {}
    for m in _NUM_UNIT_RE.finditer(text):
        word = m.group(2).lower()
        if word not in STOPWORDS:
            out.setdefault(m.group(1).replace(",", ""), set()).add(stem(word))
    return out


def unit_mismatches(claim: str, passage: str) -> list[str]:
    """Numbers whose unit in the claim differs from every unit attached to them in the passage."""
    src = _unit_pairs(passage)
    bad = []
    for num, units in _unit_pairs(claim).items():
        if num in src and not (units & src[num]):
            bad.append(f"{num} {sorted(units)[0]} (source: {num} {sorted(src[num])[0]})")
    return bad


def _neg_count(text: str) -> int:
    toks = tokenize_words(text.replace("n't", " not"))
    return sum(1 for t in toks if t in NEGATIONS)


class HeuristicEntailment:
    """Offline entailment via term coverage plus number/negation/entity contradiction checks."""

    def __init__(self, stats: TermStats, min_coverage: float = 0.75, contra_overlap: float = 0.6):
        self.stats = stats
        self.min_coverage = min_coverage
        self.contra_overlap = contra_overlap

    def _best_window(self, claim: str, passage: str) -> tuple[float, str]:
        sents = split_sentences(passage) or [passage]
        best_cov, best = -1.0, sents[0]
        for s in sents:
            cov, _ = self.stats.coverage(claim, s)
            if cov > best_cov:
                best_cov, best = cov, s
        if best_cov < self.min_coverage:
            for a, b in zip(sents, sents[1:]):
                cov, _ = self.stats.coverage(claim, f"{a} {b}")
                if cov > best_cov + 0.1:
                    best_cov, best = cov, f"{a} {b}"
        return max(best_cov, 0.0), best

    def check(self, claim: str, passage: str) -> Verdict:
        cov, window = self._best_window(claim, passage)
        missing_nums = numbers_in(claim) - numbers_in(passage)
        if missing_nums:
            win_nums = numbers_in(window)
            if win_nums and cov >= 0.4:
                return Verdict(CONTRADICTED, 0.0,
                               f"number mismatch: claim says {sorted(missing_nums)}, "
                               f"source says {sorted(win_nums)}", evidence=window)
            return Verdict(UNSUPPORTED, cov * 0.5,
                           f"numbers {sorted(missing_nums)} not in source", evidence=window)
        units = unit_mismatches(claim, passage)
        if units:
            return Verdict(CONTRADICTED, 0.0, f"unit mismatch: {units[0]}", evidence=window)
        if cov >= self.contra_overlap and _neg_count(claim) != _neg_count(window):
            return Verdict(CONTRADICTED, 0.0, "negation mismatch with source", evidence=window)
        passage_words = set(tokenize_words(passage))
        missing_ents = {e for e in entities_in("X " + claim)
                        if not set(tokenize_words(e)) <= passage_words}
        if missing_ents:
            # Entity swap = everything *except* the foreign entity matches the source well.
            rest = re.sub("|".join(rf"\b{re.escape(e)}\b" for e in missing_ents), " ", claim,
                          flags=re.I)
            rest_cov, _ = self.stats.coverage(rest, window)
            label = CONTRADICTED if rest_cov >= self.min_coverage else UNSUPPORTED
            return Verdict(label, 0.0, f"entities not in source: {sorted(missing_ents)}",
                           evidence=window)
        if cov >= self.min_coverage:
            return Verdict(ENTAILED, cov, "supported", evidence=window)
        return Verdict(UNSUPPORTED, cov, f"low overlap with source ({cov:.2f})", evidence=window)


class LLMEntailment:
    """LLM judge; expects JSON {"label": entailed|contradicted|unsupported, "reason": str}."""

    SYSTEM = ("Decide whether the PASSAGE entails the CLAIM. Reply as JSON "
              "{\"label\": \"entailed\"|\"contradicted\"|\"unsupported\", \"reason\": \"...\"}.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def check(self, claim: str, passage: str) -> Verdict:
        prompt = f"VERIFY CLAIM\nCLAIM: {claim}\nPASSAGE: {passage}"
        data = self.llm.complete(prompt, system=self.SYSTEM, json_mode=True).json()
        label = str(data.get("label", UNSUPPORTED)).lower()
        if label not in (ENTAILED, CONTRADICTED, UNSUPPORTED):
            label = UNSUPPORTED
        return Verdict(label, 1.0 if label == ENTAILED else 0.0, str(data.get("reason", "")))


@dataclass
class ClaimCheck:
    claim: Claim
    verdict: Verdict

    @property
    def flagged(self) -> bool:
        return self.verdict.label != ENTAILED


class HallucinationDetector:
    """Verify each claim against each passage it cites."""

    def __init__(self, checker: EntailmentChecker):
        self.checker = checker

    def verify_claim(self, claim: Claim, passages: Mapping[str, str]) -> Verdict:
        if not claim.citations:
            return Verdict(UNSUPPORTED, 0.0, "no citation")
        verdicts: list[Verdict] = []
        for doc_id in claim.citations:
            if doc_id not in passages:
                verdicts.append(Verdict(UNSUPPORTED, 0.0, f"cited unknown document {doc_id}",
                                        doc_id))
                continue
            v = self.checker.check(claim.text, passages[doc_id])
            v.doc_id = doc_id
            verdicts.append(v)
        for label in (ENTAILED, CONTRADICTED):
            hits = [v for v in verdicts if v.label == label]
            if hits:
                return max(hits, key=lambda v: v.score)
        return max(verdicts, key=lambda v: v.score)

    def check(self, claims: list[Claim], passages: Mapping[str, str]) -> list[ClaimCheck]:
        return [ClaimCheck(c, self.verify_claim(c, passages)) for c in claims]
