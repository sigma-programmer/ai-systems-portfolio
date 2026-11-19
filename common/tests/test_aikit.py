import os
import numpy as np
from aikit import count_tokens, truncate_to_tokens, HashingEmbedder, cosine_sim, MockLLM, get_llm


def test_tokens_monotonic_and_truncate():
    t = "The quick brown fox jumps over the lazy dog. " * 20
    assert count_tokens(t) > count_tokens(t[:50]) > 0
    out = truncate_to_tokens(t, 30)
    assert count_tokens(out) <= 30


def test_embedder_semantics():
    e = HashingEmbedder()
    q, rel, irr = e.embed(["purchase a vehicle", "how to buy a car", "baking sourdough bread"])
    assert cosine_sim(q, rel) > cosine_sim(q, irr)
    assert abs(np.linalg.norm(q) - 1) < 1e-5


def test_mock_llm_routing():
    m = MockLLM(default="nope").on(r"capital", "Paris").on(r"json", lambda p, s: '{"a": 1}')
    assert m.complete("capital of France?").text == "Paris"
    assert m.complete("give json").json() == {"a": 1}
    assert m.complete("other").text == "nope"
    assert len(m.calls) == 3


def test_get_llm_offline(monkeypatch):
    monkeypatch.setenv("AIKIT_OFFLINE", "1")
    assert isinstance(get_llm(), MockLLM)
