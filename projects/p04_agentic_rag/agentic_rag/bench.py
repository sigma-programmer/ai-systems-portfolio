"""Benchmark: answer accuracy, abstention P/R, hallucination catch rate, citation precision.

Three slices of the labeled set in `dataset.py`:

* answerable   - the honest extractive pipeline must answer with all gold facts;
* unanswerable - near-miss questions the corpus cannot answer; must abstain;
* adversarial  - answerable questions answered by `HallucinatingGenerator`, which
  corrupts every claim (number/negation/entity/citation) and appends a fabricated
  sentence; the detector must flag injected claims and the final answer must not
  contain any surviving hallucination.

Everything is deterministic (seeded) and runs offline in well under a second.
"""
from __future__ import annotations

import json
import statistics
import time

from .adversarial import HallucinatingGenerator, ParaphrasingGenerator
from .controller import AgenticRAG, RAGResult
from .dataset import ADVERSARIAL, ANSWERABLE, UNANSWERABLE, LabeledQuestion
from .generator import ExtractiveGenerator


def _correct(q: LabeledQuestion, r: RAGResult) -> bool:
    return not r.abstained and all(a.lower() in r.answer.lower() for a in q.answers)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def benchmark(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    rag = AgenticRAG()

    latencies: list[float] = []
    steps: list[int] = []
    rewrites = 0
    rows: list[tuple[LabeledQuestion, RAGResult]] = []
    for q in ANSWERABLE + UNANSWERABLE:
        t0 = time.perf_counter()
        r = rag.run(q.question)
        latencies.append((time.perf_counter() - t0) * 1000)
        steps.append(sum(1 for a in r.trace.actions() if a == "sufficiency"))
        rewrites += sum(1 for a in r.trace.actions() if a == "rewrite")
        rows.append((q, r))

    answerable = [(q, r) for q, r in rows if q.kind == "answerable"]
    unanswerable = [(q, r) for q, r in rows if q.kind == "unanswerable"]
    accuracy = sum(_correct(q, r) for q, r in answerable) / len(answerable)
    multi = [(q, r) for q, r in answerable if "multi-part" in q.tags]
    abstained = [(q, r) for q, r in rows if r.abstained]
    abst_tp = sum(1 for q, _ in abstained if q.kind == "unanswerable")
    abst_precision = abst_tp / len(abstained) if abstained else 1.0
    abst_recall = abst_tp / len(unanswerable) if unanswerable else 1.0

    cites = [(q, c) for q, r in answerable if not r.abstained for cl in r.claims
             for c in cl.citations]
    citation_precision = sum(c in q.gold_docs for q, c in cites) / len(cites) if cites else 0.0
    clean_claims = sum(len(r.checks) for _, r in answerable)
    false_flags = sum(len(r.flagged) for _, r in answerable)

    # Brier score of confidence as a predictor of "the response is right" (answer or abstain).
    brier = statistics.fmean(
        (r.confidence - (1.0 if (q.kind == "answerable" and _correct(q, r)) else 0.0)) ** 2
        for q, r in rows)

    injected = caught = leaked = 0
    by_mutation: dict[str, list[int]] = {}
    adv_correct = adv_abstain = 0
    adv_lat: list[float] = []
    adv_runs = [(seed, q) for seed in seeds for q in ADVERSARIAL]
    adv_rags = {seed: AgenticRAG(generator=HallucinatingGenerator(
        ExtractiveGenerator(rag.stats), [d.doc_id for d in rag.docs], seed=seed))
        for seed in seeds}
    for seed, q in adv_runs:
        t0 = time.perf_counter()
        r = adv_rags[seed].run(q.question)
        adv_lat.append((time.perf_counter() - t0) * 1000)
        for chk in r.checks:
            if chk.claim.injected:
                injected += 1
                caught += chk.flagged
                stats = by_mutation.setdefault(chk.claim.mutation or "?", [0, 0])
                stats[0] += 1
                stats[1] += chk.flagged
        final_texts = {c.text for c in r.claims}
        leaked += sum(1 for chk in r.checks if chk.claim.injected
                      and chk.claim.mutation != "miscite" and chk.claim.text in final_texts)
        adv_correct += _correct(q, r)
        adv_abstain += r.abstained

    # Honest-but-reworded answers: how often does the verifier wrongly flag a true paraphrase?
    para_rag = AgenticRAG(generator=ParaphrasingGenerator(ExtractiveGenerator(rag.stats)))
    para_total = para_flagged = para_correct = 0
    for q in ANSWERABLE:
        r = para_rag.run(q.question)
        for chk in r.checks:
            if chk.claim.mutation == "paraphrase":
                para_total += 1
                para_flagged += chk.flagged
        para_correct += _correct(q, r)

    return {
        "n_questions": {"answerable": len(ANSWERABLE), "unanswerable": len(UNANSWERABLE),
                        "adversarial": len(ADVERSARIAL),
                        "adversarial_runs": len(adv_runs)},
        "answer_accuracy": round(accuracy, 4),
        "multi_part_accuracy": round(sum(_correct(q, r) for q, r in multi) / len(multi), 4),
        "abstention_precision": round(abst_precision, 4),
        "abstention_recall": round(abst_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "false_flag_rate_clean_claims": round(false_flags / clean_claims, 4) if clean_claims else 0,
        "confidence_brier": round(brier, 4),
        "hallucination_catch_rate": round(caught / injected, 4) if injected else 0.0,
        "injected_claims": injected,
        "catch_rate_by_mutation": {k: round(v[1] / v[0], 4) for k, v in sorted(by_mutation.items())},
        "hallucinations_leaked_to_final_answer": leaked,
        "adversarial_accuracy_after_repair": round(adv_correct / len(adv_runs), 4),
        "adversarial_abstentions": adv_abstain,
        "paraphrased_true_claims": para_total,
        "paraphrase_false_flag_rate": round(para_flagged / para_total, 4) if para_total else 0.0,
        "paraphrase_run_accuracy_after_repair": round(para_correct / len(ANSWERABLE), 4),
        "avg_retrieval_steps": round(statistics.fmean(steps), 3),
        "rewrites_issued": rewrites,
        "latency_ms": {"mean": round(statistics.fmean(latencies), 3),
                       "p50": round(_pct(latencies, 50), 3), "p95": round(_pct(latencies, 95), 3),
                       "adversarial_mean": round(statistics.fmean(adv_lat), 3)},
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
