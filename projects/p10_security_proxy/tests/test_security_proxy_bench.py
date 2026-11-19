"""Benchmark contract."""
from __future__ import annotations

import json

from security_proxy.bench import benchmark


def test_benchmark_shape_and_quality() -> None:
    m = benchmark()
    json.dumps(m)
    assert m["injection_dev"]["n_attacks"] >= 60 and m["injection_dev"]["n_benign"] >= 60
    assert m["injection_combined"]["f1"] >= 0.9
    assert m["injection_combined"]["false_positive_rate"] <= 0.05
    assert m["obfuscated_recall"] >= 0.9
    assert m["pii"]["recall_overall"] >= 0.95
    assert set(m["pii"]["recall_per_type"]) == {"EMAIL", "PHONE", "CREDIT_CARD", "SSN", "IP_ADDRESS", "IBAN", "API_KEY", "PERSON"}
    assert m["latency"]["audit_chain_valid"] is True
    assert m["latency"]["route_ms_mean"] > 0
