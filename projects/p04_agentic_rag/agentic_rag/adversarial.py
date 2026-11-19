"""A deliberately hallucinating generator for red-teaming the grounding checks.

Wraps any generator and corrupts its output in the ways real LLMs fail:
number drift, negation flips, entity swaps, mis-citations, and wholesale
fabricated sentences with a plausible citation. Every corrupted claim is
marked `injected=True` with its `mutation`, so the benchmark can measure the
detector's catch rate against ground truth.
"""
from __future__ import annotations

import random
import re
from typing import Sequence

from .generator import AnswerGenerator, Claim, Evidence
from .text import entities_in

MUTATIONS = ("number_swap", "negation_flip", "entity_swap", "miscite", "unit_swap")
_UNITS = {"hours": "minutes", "days": "weeks", "weeks": "months", "dollars": "euros",
          "meters": "feet", "grams": "kilograms", "percent": "times", "year": "month",
          "months": "years", "kilometers": "miles", "minutes": "seconds"}
_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?\s+(?:\w+\s)?)(" + "|".join(_UNITS) + r")\b")
# Meaning-preserving rewrites used to test the verifier's false-flag rate on paraphrases.
_PARAPHRASES = (("lasts", "runs for"), ("receive", "get"), ("weighs", "has a weight of"),
                ("roughly", "about"), ("allows", "lets"), ("include", "come with"),
                ("is headquartered in", "has its headquarters in"), ("offers", "provides"),
                ("created", "designed"), ("retails for", "is priced at"),
                ("Customers may", "Buyers can"), ("is available", "operates"),
                ("raised", "secured"), ("highest", "tallest"))
_AUX_RE = re.compile(r"\b(is|are|was|were|can|has|have|does|will|may)\b")
_NOT_RE = re.compile(r"\s+not\b|\bno\s+", re.I)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_FAKE_ENTITIES = ("Berlin", "Marcus Holt", "Helsinki", "Orion", "Kestrel", "Jupiter")


def _swap_number(text: str, rng: random.Random) -> str | None:
    nums = list(_NUM_RE.finditer(text))
    if not nums:
        return None
    m = rng.choice(nums)
    val = float(m.group())
    new = val * rng.choice((2, 3)) + rng.choice((1, 5, 7))
    rep = str(int(new)) if "." not in m.group() else f"{new:.1f}"
    return text[: m.start()] + rep + text[m.end():]


def _flip_negation(text: str) -> str:
    if _NOT_RE.search(text):
        return _NOT_RE.sub(" ", text, count=1).replace("  ", " ")
    m = _AUX_RE.search(text)
    if m:
        return text[: m.end()] + " not" + text[m.end():]
    return "It is not true that " + text[0].lower() + text[1:]


def _swap_unit(text: str) -> str | None:
    m = _UNIT_RE.search(text)
    if not m:
        return None
    return text[: m.start(2)] + _UNITS[m.group(2)] + text[m.end(2):]


def paraphrase(text: str) -> str:
    """Apply meaning-preserving phrase substitutions (first match wins per phrase)."""
    for a, b in _PARAPHRASES:
        text = re.sub(rf"\b{re.escape(a)}\b", b, text, count=1)
    return text


def _swap_entity(text: str, rng: random.Random) -> str | None:
    ents = sorted(entities_in(text))
    if not ents:
        return None
    target = rng.choice(ents)
    fake = rng.choice([f for f in _FAKE_ENTITIES if f.lower() != target])
    return re.sub(rf"\b{re.escape(target)}\b", fake, text, count=1, flags=re.I)


class HallucinatingGenerator:
    """Corrupt `rate` of the base generator's claims and append a fabricated one."""

    def __init__(self, base: AnswerGenerator, doc_ids: Sequence[str], seed: int = 0,
                 rate: float = 1.0, fabricate: bool = True):
        self.base = base
        self.doc_ids = list(doc_ids)
        self.rng = random.Random(seed)
        self.rate = rate
        self.fabricate = fabricate

    def _mutate(self, claim: Claim) -> Claim:
        order = list(MUTATIONS)
        self.rng.shuffle(order)
        for kind in order:
            if kind == "number_swap":
                new = _swap_number(claim.text, self.rng)
            elif kind == "negation_flip":
                new = _flip_negation(claim.text)
            elif kind == "entity_swap":
                new = _swap_entity(claim.text, self.rng)
            elif kind == "unit_swap":
                new = _swap_unit(claim.text)
            else:
                others = [d for d in self.doc_ids if d not in claim.citations]
                if not others:
                    continue
                return Claim(claim.text, [self.rng.choice(others)], claim.need, True, "miscite")
            if new and new != claim.text:
                return Claim(new, list(claim.citations), claim.need, True, kind)
        return claim

    def generate(self, question: str, evidence: Evidence) -> list[Claim]:
        out: list[Claim] = []
        for c in self.base.generate(question, evidence):
            out.append(self._mutate(c) if self.rng.random() < self.rate else c)
        if self.fabricate and evidence:
            first_need = next(iter(evidence))
            graded = evidence[first_need]
            if graded:
                subj = sorted(entities_in("X " + question)) or ["the company"]
                year = self.rng.choice((2017, 2020, 2024))
                amount = self.rng.choice((12, 55, 300))
                text = (f"In {year} {subj[0].title()} also secured a {amount} million euro "
                        f"government innovation grant for quantum research.")
                out.append(Claim(text, [graded[0][0].doc_id], first_need, True, "fabricated"))
        return out


class ParaphrasingGenerator:
    """Honest generator that rewords true claims (tests verifier false flags)."""

    def __init__(self, base: AnswerGenerator):
        self.base = base

    def generate(self, question: str, evidence: Evidence) -> list[Claim]:
        out = []
        for c in self.base.generate(question, evidence):
            new = paraphrase(c.text)
            out.append(Claim(new, list(c.citations), c.need, False,
                             "paraphrase" if new != c.text else None))
        return out
