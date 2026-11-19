"""The benchmark runs and reports perfect rejection/blocking."""
from __future__ import annotations

import json

from mcp_server.bench import benchmark, fuzz_cases


def test_benchmark_small():
    m = benchmark(inproc_calls=300, stdio_calls=100)
    json.dumps(m)
    assert m["fuzz_accuracy"] == 1.0 and m["fuzz_failures"] == []
    assert m["sandbox_escapes_blocked"] == m["sandbox_escape_attempts"] and m["sandbox_not_blocked"] == []
    assert m["inproc_tools_call_rps"] > 0 and m["stdio_tools_call_rps_sequential"] > 0


def test_fuzz_corpus_size():
    assert len(fuzz_cases()) >= 50
