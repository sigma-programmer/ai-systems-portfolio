"""Prompt complexity analysis → score in [0,1] → tier.

Features (each normalised to [0,1]):

    entropy    Shannon entropy of the word distribution, / 7 bits
    length     log(1 + tokens) / log(1 + 400)
    rare       share of long (>= 9 char) words: a technical-vocabulary proxy
    reasoning  reasoning-cue hits / 2  ("prove", "derive", "step by step", "trade-off", ...)
    code       1 if code is present (fences, def/class/import, braces+semicolons, SQL)
    parts      (number of sub-questions - 1) / 3
    generative 1 if an open-ended generation cue is present ("summarize", "draft", "explain")

    score = 0.12·entropy + 0.13·length + 0.10·rare + 0.35·reasoning + 0.18·code
          + 0.12·parts + 0.20·generative          (clamped to 1)

    tier  = simple < 0.30 <= moderate < 0.50 <= complex

Reasoning cues carry the most weight because they are the strongest signal that a
small model will fail; length/entropy alone would route long-but-easy prompts
("summarise this paragraph") to the most expensive tier.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from aikit import count_tokens, tokenize_words

REASONING_CUES = (
    "prove", "proof", "derive", "derivation", "step by step", "step-by-step", "explain why",
    "analyze", "analyse", "compare", "trade-off", "tradeoff", "optimize", "optimise", "design",
    "architecture", "algorithm", "complexity", "theorem", "rigorous", "evaluate", "critique",
    "reasoning", "justify", "implement", "refactor", "debug", "formal", "invariant", "big-o",
)
# Open-ended generation/explanation tasks: more than a lookup, less than a proof.
GENERATIVE_CUES = ("summarize", "summarise", "summarizing", "explain", "write", "draft",
                   "describe", "pros and cons", "list the", "difference", "differ", "how does",
                   "outline", "paragraph", "email", "essay", "examples")
_CODE_RE = re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|\bimport \w+|#include|\bfunction\s*\(|"
                      r"=>|\bSELECT\b.+\bFROM\b|[{};]\s*\n|\w+\(\)\s*[{;]", re.S)
_ENUM_RE = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+")


@dataclass(frozen=True)
class ComplexityReport:
    tokens: int
    entropy_bits: float
    lexical_diversity: float
    reasoning_hits: int
    code_present: bool
    parts: int
    score: float
    tier: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ComplexityAnalyzer:
    def __init__(self, moderate_at: float = 0.30, complex_at: float = 0.50):
        if not 0 <= moderate_at < complex_at <= 1:
            raise ValueError("need 0 <= moderate_at < complex_at <= 1")
        self.moderate_at = moderate_at
        self.complex_at = complex_at

    @staticmethod
    def entropy(words: list[str]) -> float:
        if not words:
            return 0.0
        n = len(words)
        return -sum(c / n * math.log2(c / n) for c in Counter(words).values())

    def analyze(self, prompt: str) -> ComplexityReport:
        words = tokenize_words(prompt)
        tokens = count_tokens(prompt)
        low = prompt.lower()
        h = self.entropy(words)
        ttr = len(set(words)) / len(words) if words else 0.0
        rare = sum(1 for w in words if len(w) >= 9) / len(words) if words else 0.0
        hits = sum(1 for cue in REASONING_CUES if re.search(rf"\b{re.escape(cue)}\b", low))
        gen = sum(1 for cue in GENERATIVE_CUES if re.search(rf"\b{re.escape(cue)}\b", low))
        code = bool(_CODE_RE.search(prompt))
        parts = max(1, prompt.count("?") + len(_ENUM_RE.findall(prompt))
                    - (1 if prompt.count("?") and _ENUM_RE.findall(prompt) else 0))
        score = (0.12 * min(h / 7.0, 1.0)
                 + 0.13 * min(math.log1p(tokens) / math.log1p(400), 1.0)
                 + 0.10 * min(rare * 4, 1.0)
                 + 0.35 * min(hits / 2.0, 1.0)
                 + 0.18 * float(code)
                 + 0.12 * min((parts - 1) / 3.0, 1.0)
                 + 0.20 * min(gen, 1))
        score = round(min(1.0, score), 4)
        tier = ("complex" if score >= self.complex_at
                else "moderate" if score >= self.moderate_at else "simple")
        return ComplexityReport(tokens, round(h, 3), round(ttr, 3), hits, code, parts, score, tier)
