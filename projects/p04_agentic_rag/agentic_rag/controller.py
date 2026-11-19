"""The self-correcting agentic RAG controller and its trace.

Loop (bounded by `max_steps`):

    needs = [question]
    repeat:
        retrieve top-k for every unmet need's queries
        grade each passage against its need
        if every need has a relevant passage: break
        rewrite the unmet needs (decompose / expand / keywords)
    generate cited claims from relevant passages
    verify every claim against its cited passage (hallucination detector)
    repair: re-cite unsupported claims, drop the rest, regenerate empty needs
    confidence = calibrated(retrieval, coverage of needs, generator grounding)
    abstain with an explanation if nothing is supported or confidence < threshold
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .corpus import Document, load_corpus
from .generator import AnswerGenerator, Claim, Evidence, ExtractiveGenerator
from .grader import Grade, HeuristicGrader, RelevanceGrader, TermStats
from .retriever import HybridRetriever, RetrievedPassage
from .rewriter import HeuristicRewriter, QueryRewriter, decompose
from .verifier import (ENTAILED, ClaimCheck, EntailmentChecker, HallucinationDetector,
                       HeuristicEntailment)


@dataclass
class TraceStep:
    action: str
    data: dict[str, Any]
    ms: float


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, action: str, t0: float, **data: Any) -> None:
        self.steps.append(TraceStep(action, data, round((time.perf_counter() - t0) * 1000, 3)))

    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]

    def pretty(self) -> str:
        lines = []
        for i, s in enumerate(self.steps):
            brief = {k: v for k, v in s.data.items() if k not in ("passages",)}
            lines.append(f"  {i:2d}. {s.action:<14} {s.ms:7.2f}ms  {brief}")
        return "\n".join(lines)


@dataclass
class Confidence:
    """Logistic calibration over three normalised signals (weights documented in README)."""

    w_retrieval: float = 0.45
    w_answered: float = 0.35
    w_grounding: float = 0.20
    midpoint: float = 0.55
    steepness: float = 9.0

    def score(self, retrieval: float, answered: float, grounding: float) -> float:
        z = (self.w_retrieval * retrieval + self.w_answered * answered
             + self.w_grounding * grounding)
        return 1.0 / (1.0 + math.exp(-self.steepness * (z - self.midpoint)))


@dataclass
class RAGResult:
    question: str
    answer: str
    abstained: bool
    confidence: float
    claims: list[Claim]
    checks: list[ClaimCheck]
    unanswered: list[str]
    missing_terms: list[str]
    signals: dict[str, float]
    trace: Trace

    @property
    def citations(self) -> list[str]:
        return list(dict.fromkeys(c for cl in self.claims for c in cl.citations))

    @property
    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.flagged]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer, "abstained": self.abstained,
            "confidence": round(self.confidence, 4), "citations": self.citations,
            "claims": [{"text": c.text, "citations": c.citations} for c in self.claims],
            "flagged": [{"text": c.claim.text, "label": c.verdict.label,
                         "reason": c.verdict.reason} for c in self.flagged],
            "unanswered": self.unanswered, "missing_terms": self.missing_terms,
            "signals": self.signals, "trace": self.trace.to_dict(),
        }


class AgenticRAG:
    """Retrieve → grade → rewrite → generate → verify → repair/abstain."""

    def __init__(
        self,
        docs: Sequence[Document] | None = None,
        *,
        retriever: HybridRetriever | None = None,
        grader: RelevanceGrader | None = None,
        rewriter: QueryRewriter | None = None,
        generator: AnswerGenerator | None = None,
        checker: EntailmentChecker | None = None,
        repair_generator: AnswerGenerator | None = None,
        confidence: Confidence | None = None,
        max_steps: int = 3,
        decompose_upfront: bool = True,
        top_k: int = 4,
        abstain_threshold: float = 0.6,
    ):
        self.docs = list(docs) if docs is not None else load_corpus()
        self.retriever = retriever or HybridRetriever(self.docs)
        self.stats = TermStats([d.title + " " + d.text for d in self.docs])
        self.grader = grader or HeuristicGrader(self.stats)
        self.rewriter = rewriter or HeuristicRewriter(self.stats)
        self.generator = generator or ExtractiveGenerator(self.stats)
        self.repair_generator = repair_generator or ExtractiveGenerator(self.stats)
        self.detector = HallucinationDetector(checker or HeuristicEntailment(self.stats))
        self.confidence = confidence or Confidence()
        self.max_steps = max(1, max_steps)
        self.decompose_upfront = decompose_upfront
        self.top_k = top_k
        self.abstain_threshold = abstain_threshold
        self.passage_text = {d.doc_id: d.title + ". " + d.text for d in self.docs}

    # ------------------------------------------------------------------ retrieval loop
    def _retrieve(self, question: str, trace: Trace
                  ) -> tuple[list[str], dict[str, dict[str, tuple[RetrievedPassage, Grade]]]]:
        t0 = time.perf_counter()
        needs = decompose(question) if self.decompose_upfront else [question]
        queries: dict[str, list[str]] = {n: [n] for n in needs}
        trace.add("plan", t0, needs=needs)
        graded: dict[str, dict[str, tuple[RetrievedPassage, Grade]]] = {}
        for step in range(self.max_steps):
            for need in needs:
                if any(g.relevant for _, g in graded.get(need, {}).values()):
                    continue
                pool = graded.setdefault(need, {})
                for q in queries.get(need, [need]):
                    t0 = time.perf_counter()
                    hits = self.retriever.search(q, self.top_k)
                    trace.add("retrieve", t0, step=step, need=need, query=q,
                              hits=[(h.doc_id, round(h.score, 3)) for h in hits])
                    t0 = time.perf_counter()
                    grades = []
                    for h in hits:
                        g = self.grader.grade(need, h)
                        prev = pool.get(h.doc_id)
                        if prev is None or g.relevance > prev[1].relevance:
                            pool[h.doc_id] = (h, g)
                        grades.append((h.doc_id, round(g.relevance, 3), g.relevant))
                    trace.add("grade", t0, step=step, need=need, grades=grades)
            unmet = [n for n in needs
                     if not any(g.relevant for _, g in graded.get(n, {}).values())]
            t0 = time.perf_counter()
            trace.add("sufficiency", t0, step=step, needs=len(needs), unmet=unmet)
            if not unmet or step == self.max_steps - 1:
                break
            t0 = time.perf_counter()
            plan = self.rewriter.rewrite(question, unmet, step + 1)
            needs = [n for n in needs if n not in unmet] + plan.needs
            queries.update(plan.queries)
            trace.add("rewrite", t0, step=step, strategy=plan.strategy, needs=plan.needs,
                      queries=plan.queries)
        return needs, graded

    # ------------------------------------------------------------------ repair
    def _repair(self, question: str, checks: list[ClaimCheck], evidence: Evidence,
                trace: Trace) -> list[Claim]:
        t0 = time.perf_counter()
        pool_ids = list(dict.fromkeys(p.doc_id for g in evidence.values() for p, _ in g))
        kept: list[Claim] = []
        actions: list[dict[str, Any]] = []
        for chk in checks:
            if not chk.flagged:
                kept.append(chk.claim)
                continue
            fixed = None
            if chk.verdict.label != "contradicted":
                for doc_id in pool_ids:
                    if doc_id in chk.claim.citations:
                        continue
                    v = self.detector.checker.check(chk.claim.text, self.passage_text[doc_id])
                    if v.label == ENTAILED:
                        fixed = Claim(chk.claim.text, [doc_id], chk.claim.need,
                                      chk.claim.injected, chk.claim.mutation)
                        break
            if fixed is not None:
                kept.append(fixed)
                actions.append({"action": "recited", "claim": chk.claim.text,
                                "to": fixed.citations})
            else:
                actions.append({"action": "dropped", "claim": chk.claim.text,
                                "reason": chk.verdict.reason})
        covered = {c.need for c in kept}
        for need, graded in evidence.items():
            if need in covered or not graded:
                continue
            regen = self.repair_generator.generate(question, {need: graded})
            for chk in self.detector.check(regen, self.passage_text):
                if not chk.flagged and chk.claim.text not in {c.text for c in kept}:
                    kept.append(chk.claim)
                    actions.append({"action": "regenerated", "need": need,
                                    "claim": chk.claim.text})
        trace.add("repair", t0, actions=actions)
        return kept

    # ------------------------------------------------------------------ main entry
    def run(self, question: str) -> RAGResult:
        trace = Trace()
        question = question.strip()
        needs, graded = self._retrieve(question, trace)

        evidence: Evidence = {}
        best_rel: dict[str, float] = {}
        missing: list[str] = []
        closest: list[str] = []
        for need in needs:
            pool = sorted(graded.get(need, {}).values(), key=lambda x: -x[1].relevance)
            best_rel[need] = pool[0][1].relevance if pool else 0.0
            rel = [(p, g) for p, g in pool if g.relevant]
            if rel:
                evidence[need] = rel
            elif pool:
                missing.extend(t for t in pool[0][1].missing_terms if t not in missing)
                closest.append(pool[0][0].doc_id)
        unanswered = [n for n in needs if n not in evidence]
        retrieval = sum(best_rel.values()) / len(best_rel) if best_rel else 0.0

        if not evidence or not question:
            return self._abstain(question, trace, unanswered or [question], missing,
                                 {"retrieval": round(retrieval, 4), "answered": 0.0,
                                  "grounding": 0.0}, [], closest=closest)

        t0 = time.perf_counter()
        claims = self.generator.generate(question, evidence)
        for c in claims:
            if c.need not in evidence:
                c.need = max(evidence, key=lambda n: self.stats.coverage(n, c.text)[0])
        trace.add("generate", t0, claims=[c.render() for c in claims])

        t0 = time.perf_counter()
        checks = self.detector.check(claims, self.passage_text)
        trace.add("verify", t0, verdicts=[(c.claim.text[:60], c.verdict.label, c.verdict.reason)
                                          for c in checks])
        grounding = (sum(not c.flagged for c in checks) / len(checks)) if checks else 0.0

        final = self._repair(question, checks, evidence, trace) if any(
            c.flagged for c in checks) or not checks else [c.claim for c in checks]
        answered_needs = {c.need for c in final} & set(needs)
        answered = len(answered_needs) / len(needs) if needs else 0.0
        unanswered = [n for n in needs if n not in answered_needs]
        signals = {"retrieval": round(retrieval, 4), "answered": round(answered, 4),
                   "grounding": round(grounding, 4)}
        conf = self.confidence.score(retrieval, answered, grounding)
        t0 = time.perf_counter()
        trace.add("confidence", t0, confidence=round(conf, 4), **signals)
        if not final or conf < self.abstain_threshold:
            return self._abstain(question, trace, unanswered, missing, signals, checks, conf)

        answer = " ".join(c.render() for c in final)
        if unanswered:
            answer += " (Not found in the knowledge base: " + "; ".join(unanswered) + ")"
        return RAGResult(question, answer, False, conf, final, checks, unanswered, missing,
                         signals, trace)

    def _abstain(self, question: str, trace: Trace, unanswered: list[str], missing: list[str],
                 signals: dict[str, float], checks: list[ClaimCheck],
                 conf: float | None = None, closest: list[str] | None = None) -> RAGResult:
        if conf is None:
            conf = self.confidence.score(signals.get("retrieval", 0.0), 0.0, 0.0)
        parts = ["I don't know."]
        if missing:
            where = (f"The closest passage{'s' if len(closest or []) > 1 else ''} "
                     f"[{', '.join(closest)}] do{'' if len(closest) > 1 else 'es'} not mention"
                     if closest else "No retrieved passage mentions")
            parts.append(f"{where}: " + ", ".join(f"'{m}'" for m in missing[:5]) + ".")
        if unanswered and unanswered != [question]:
            parts.append("Unanswered parts: " + "; ".join(unanswered) + ".")
        if checks and any(c.flagged for c in checks):
            parts.append(f"{sum(c.flagged for c in checks)} drafted claim(s) failed "
                         "citation verification.")
        t0 = time.perf_counter()
        trace.add("abstain", t0, confidence=round(conf, 4), missing=missing[:5])
        return RAGResult(question, " ".join(parts), True, conf, [], checks, unanswered, missing,
                         signals, trace)
