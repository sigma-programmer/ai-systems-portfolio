"""Importance scoring shared by all memory tiers.

    salience  ∈ [0,1]   static, computed once when a memory is written:
                        0.25 base
                      + 0.35 explicit cue   ("remember", "important", "don't forget", ...)
                      + 0.20 personal fact  (matches a semantic-extraction pattern)
                      + 0.15 emotional cue  ("love", "hate", "worried", "excited", ...)
                      + 0.05 concrete detail (contains a number/date)
                      - 0.15 filler         (greetings / acknowledgements only)
    recency   = 2 ** (-(now - last_access) / half_life_of_tier)
    frequency = 1 - exp(-access_count / 3)

    importance = 0.45 * salience + 0.30 * recency + 0.25 * frequency

Salience dominates so a "remember that ..." from last month outranks small talk from
five minutes ago; recency and access frequency let unused memories fade.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

EXPLICIT_CUES = re.compile(
    r"\b(remember|important|don'?t forget|do not forget|note that|keep in mind|always|never|"
    r"make sure|crucial)\b", re.I)
EMOTION_CUES = re.compile(
    r"\b(love|hate|angry|upset|excited|worried|anxious|scared|thrilled|happy|sad|stressed|"
    r"frustrated|grateful|afraid|delighted|furious)\b", re.I)
PERSONAL_CUES = re.compile(
    r"\b(my name is|i work (?:at|for)|i live in|i moved to|i prefer|i like|i love|i hate|"
    r"i dislike|my favou?rite|i am allergic|i'?m allergic|my birthday|i'?m an? |i am an? |"
    r"my \w+'?s name)\b", re.I)
FILLER = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok(?:ay)?|cool|great|sure|bye|lol|"
                    r"good morning|nice)[\s!.,]*$", re.I)
NUMBER = re.compile(r"\d")


@dataclass(frozen=True)
class ImportanceWeights:
    salience: float = 0.45
    recency: float = 0.30
    frequency: float = 0.25


def salience_of(text: str) -> float:
    """Static salience of a piece of text from lexical cues (see module docstring)."""
    s = 0.25
    if EXPLICIT_CUES.search(text):
        s += 0.35
    if PERSONAL_CUES.search(text):
        s += 0.20
    if EMOTION_CUES.search(text):
        s += 0.15
    if NUMBER.search(text):
        s += 0.05
    if FILLER.match(text):
        s -= 0.15
    return max(0.0, min(1.0, s))


def recency(age_seconds: float, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return 2.0 ** (-max(0.0, age_seconds) / half_life)


def frequency(access_count: int) -> float:
    return 1.0 - math.exp(-max(0, access_count) / 3.0)


class ImportanceScorer:
    """Combine salience, recency and frequency into one importance value in [0,1]."""

    def __init__(self, weights: ImportanceWeights | None = None):
        self.w = weights or ImportanceWeights()

    def static(self, salience: float, access_count: int) -> float:
        """Time-independent part of importance, renormalised to [0,1] (used by recall)."""
        w = self.w.salience + self.w.frequency
        return (self.w.salience * salience + self.w.frequency * frequency(access_count)) / w

    def score(self, salience: float, last_access: float, access_count: int, now: float,
              half_life: float) -> float:
        return (self.w.salience * salience
                + self.w.recency * recency(now - last_access, half_life)
                + self.w.frequency * frequency(access_count))
