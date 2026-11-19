"""Benchmark: injection detection quality, PII masking recall, latency.

Run: ``AIKIT_OFFLINE=1 .venv/bin/python -m security_proxy.bench`` (with the
project dir on ``PYTHONPATH``) or call :func:`benchmark`.

All quality metrics are deterministic; latency numbers are wall-clock and
vary by machine.
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Any

from .corpus import (
    ATTACKS,
    BENIGN,
    HELDOUT_ATTACKS,
    HELDOUT_BENIGN,
    PII_HELDOUT,
    PII_NEGATIVES,
    PII_SAMPLES,
    obfuscated_attacks,
)
from .injection import InjectionDetector
from .pii import PIIDetector, PIIType
from .policy import Policy
from .proxy import Envelope, SecurityProxy

__all__ = ["benchmark", "detection_metrics"]


def detection_metrics(detector: InjectionDetector, attacks: list[str], benign: list[str]) -> dict[str, Any]:
    """Precision / recall / F1 / FPR of ``detector`` on labelled texts."""
    tp = sum(detector.scan(t).is_injection for t in attacks)
    fp = sum(detector.scan(t).is_injection for t in benign)
    fn = len(attacks) - tp
    tn = len(benign) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_attacks": len(attacks),
        "n_benign": len(benign),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / len(benign), 4) if benign else 0.0,
    }


def _pii_metrics(samples: list[tuple[str, list[tuple[str, str]]]]) -> dict[str, Any]:
    det = PIIDetector()
    expected: dict[str, int] = {t: 0 for t in PIIType.ALL}
    found: dict[str, int] = {t: 0 for t in PIIType.ALL}
    extra = 0
    for text, labels in samples:
        got = {(m.type, m.value) for m in det.detect(text)}
        for typ, value in labels:
            expected[typ] += 1
            found[typ] += (typ, value) in got
        extra += len(got - {tuple(lbl) for lbl in labels})
    negatives_flagged = sum(1 for t in PII_NEGATIVES if det.detect(t))
    per_type = {t: round(found[t] / expected[t], 4) for t in PIIType.ALL if expected[t]}
    total_e, total_f = sum(expected.values()), sum(found.values())
    return {
        "recall_per_type": per_type,
        "recall_overall": round(total_f / total_e, 4),
        "n_labelled_values": total_e,
        "unexpected_detections_in_samples": extra,
        "negatives_flagged": negatives_flagged,
        "n_negatives": len(PII_NEGATIVES),
    }


def _latency(n_rounds: int = 3) -> dict[str, Any]:
    policy = Policy.from_dict(
        {"agents": {"a": {"can_message": ["b"]}, "b": {"can_message": ["a"]}}}
    )
    proxy = SecurityProxy(policy)
    texts = ATTACKS + BENIGN + [t for t, _ in PII_SAMPLES]
    samples: list[float] = []
    det = InjectionDetector()
    scan_samples: list[float] = []
    for _ in range(n_rounds):
        for t in texts:
            t0 = time.perf_counter()
            proxy.route(Envelope(sender="a", recipient="b", content=t, session_id="bench"))
            samples.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            det.scan(t)
            scan_samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    ok = proxy.audit.verify().ok
    return {
        "messages": len(samples),
        "route_ms_mean": round(statistics.fmean(samples), 3),
        "route_ms_p50": round(samples[len(samples) // 2], 3),
        "route_ms_p95": round(samples[int(len(samples) * 0.95)], 3),
        "scan_ms_mean": round(statistics.fmean(scan_samples), 3),
        "audit_chain_valid": ok,
    }


def benchmark() -> dict[str, Any]:
    """Run all benchmarks and return JSON-serialisable metrics."""
    det = InjectionDetector()
    obf = obfuscated_attacks()
    obf_detected = sum(det.scan(t).is_injection for t in obf)
    return {
        "injection_dev": detection_metrics(det, ATTACKS, BENIGN),
        "injection_heldout": detection_metrics(det, HELDOUT_ATTACKS, HELDOUT_BENIGN),
        "injection_combined": detection_metrics(det, ATTACKS + HELDOUT_ATTACKS, BENIGN + HELDOUT_BENIGN),
        "obfuscated_recall": round(obf_detected / len(obf), 4),
        "n_obfuscated": len(obf),
        "pii": _pii_metrics(PII_SAMPLES),
        "pii_heldout": _pii_metrics(PII_HELDOUT),
        "latency": _latency(),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
