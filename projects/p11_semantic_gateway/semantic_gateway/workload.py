"""Seeded synthetic workload: repeats, paraphrases, near-duplicate traps, mixed complexity.

Every request carries its ground-truth `intent`; a cache hit whose cached entry
came from a *different* intent is a false hit (e.g. "Convert 5 km to miles"
answered with the cached answer for 8 km).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    id: str
    tier: str
    paraphrases: tuple[str, ...]


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    tenant: str
    intent: str
    tier: str
    t: float


_CODE = ("Refactor this Python function to run in O(n log n) and explain the complexity:\n"
         "```python\ndef uniques(xs):\n    return [x for x in xs if xs.count(x) == 1]\n```")


def _intents() -> list[Intent]:
    I = Intent
    out = [
        I("capital_france", "simple", ("What is the capital of France?",
          "what's the capital of france", "Tell me the capital city of France.",
          "Which city is the capital of France?")),
        I("capital_japan", "simple", ("What is the capital of Japan?",
          "Which city is the capital of Japan?", "capital city of Japan?")),
        I("hello_spanish", "simple", ("How do you say hello in Spanish?",
          "Translate hello into Spanish.", "What is hello in Spanish?")),
        I("photosynthesis", "simple", ("Define photosynthesis.", "What is photosynthesis?",
          "Give me a short definition of photosynthesis.")),
        I("boiling_point", "simple", ("What is the boiling point of water?",
          "At what temperature does water boil?", "water boiling point?")),
        I("dogs_safe", "simple", ("Which foods are safe for dogs?",
          "What foods are safe for dogs to eat?")),
        I("dogs_unsafe", "simple", ("Which foods are not safe for dogs?",
          "What foods are not safe for dogs to eat?")),
        I("font_increase", "simple", ("How do I increase the font size in VS Code?",
          "How can I increase font size in VS Code?")),
        I("font_decrease", "simple", ("How do I decrease the font size in VS Code?",
          "How can I decrease font size in VS Code?")),
        I("ts_reasons", "simple", ("What are reasons to use TypeScript?",
          "Give me reasons to use TypeScript.")),
        I("ts_reasons_not", "simple", ("What are reasons not to use TypeScript?",
          "Give me reasons not to use TypeScript.")),
        I("ww1", "moderate", ("Summarize the main causes of World War I in a paragraph.",
          "Give me a paragraph summarizing the main causes of World War I.",
          "Briefly summarize the main causes of World War I in one paragraph.")),
        I("decline_email", "moderate", (
          "Write a polite email declining a meeting invitation for next Tuesday and "
          "suggesting two alternative times later in the week.",
          "Draft a polite email that declines a meeting invitation for next Tuesday and "
          "suggests two alternative times later in the week.")),
        I("https", "moderate", (
          "Explain how HTTPS keeps data private between a browser and a server, covering "
          "certificates and the handshake.",
          "How does HTTPS keep data private between a browser and a server? Cover "
          "certificates and the handshake.")),
        I("remote_work", "moderate", (
          "What are the pros and cons of remote work for software teams? What about "
          "onboarding new engineers?",
          "List the pros and cons of remote work for software teams. What about onboarding "
          "new engineers?")),
        I("tcp_udp", "moderate", ("Explain why TCP and UDP differ, with concrete examples of "
                                  "applications that use each protocol.",
          "Explain why TCP differs from UDP, with concrete examples of applications that use "
          "each protocol.")),
        I("sqrt2", "complex", ("Prove that the square root of 2 is irrational, step by step.",
          "Give a rigorous step by step proof that the square root of 2 is irrational.")),
        I("rate_limiter", "complex", (
          "Design a rate limiter architecture for a multi-region API and analyze the "
          "trade-offs between token bucket and sliding window algorithms.",
          "Design a multi-region API rate limiter architecture and analyze the trade-offs of "
          "token bucket versus sliding window algorithms.")),
        I("refactor", "complex", (_CODE,)),
        I("ols", "complex", (
          "Derive the closed-form solution for ordinary least squares regression and justify "
          "each step of the derivation.",
          "Derive the ordinary least squares closed-form solution and justify each step of "
          "the derivation.")),
        I("raft_paxos", "complex", (
          "Compare Raft and Paxos consensus algorithms:\n1. safety guarantees\n2. liveness\n"
          "3. operational complexity\nWhich would you choose for a 5-node cluster and why?",)),
    ]
    for n in (5, 8, 10, 21, 42):
        out.append(I(f"km_{n}", "simple", (f"Convert {n} km to miles.",
                                           f"How many miles is {n} km?")))
    for n in (40, 85, 120):
        out.append(I(f"tip_{n}", "simple", (f"What is a 15% tip on a ${n} bill?",
                                            f"Calculate a 15% tip on a ${n} bill.")))
    return out


INTENTS = _intents()

# Long-tail templates: combinatorial fills make most of these unique, and same-template
# siblings ("capital of Spain" vs "capital of Peru") are exactly the false-hit trap.
_COUNTRIES = ("Spain", "Peru", "Kenya", "Norway", "Vietnam", "Chile", "Egypt", "Poland",
              "Canada", "India", "Ghana", "Portugal", "Mexico", "Austria", "Nepal")
_TOPICS = ("the printing press", "the Roman Empire", "the Silk Road", "jazz music",
           "the Apollo program", "the Industrial Revolution", "the internet", "vaccines")
_CONCEPTS = ("recursion", "inflation", "photosynthesis in C4 plants", "public key cryptography",
             "compound interest", "gradient descent", "plate tectonics", "DNS caching")
_TASKS = ("parses ISO 8601 dates", "merges two sorted lists", "validates email addresses",
          "deduplicates a list while preserving order", "computes a moving average")
_LANGS = ("Python", "Go", "Rust", "TypeScript")
_SYSTEMS = ("a URL shortener", "a chat service", "a ride-sharing dispatcher",
            "a metrics pipeline", "a feature-flag service")
_TAIL: list[tuple[str, str]] = [
    ("simple", "What is the capital of {country}?"),
    ("simple", "What currency is used in {country}?"),
    ("simple", "What is {a} times {b}?"),
    ("moderate", "Summarize the history of {topic} in a short paragraph."),
    ("moderate", "Explain {concept} to a beginner with one example."),
    ("moderate", "Write a {lang} function that {task}."),
    ("complex", "Design the architecture for {system} handling {a}0000 requests per second "
                "and analyze the trade-offs of your storage choices step by step."),
    ("complex", "Prove that the sum of the first {a} odd numbers equals {a} squared, and derive "
                "the general formula rigorously."),
]


def _tail_item(rng: random.Random) -> tuple[str, str]:
    tier, tpl = rng.choice(_TAIL)
    prompt = tpl.format(country=rng.choice(_COUNTRIES), topic=rng.choice(_TOPICS),
                        concept=rng.choice(_CONCEPTS), task=rng.choice(_TASKS),
                        lang=rng.choice(_LANGS), system=rng.choice(_SYSTEMS),
                        a=rng.randint(3, 60), b=rng.randint(3, 60))
    return tier, prompt


def generate(n: int = 500, seed: int = 11, tenants: tuple[str, ...] = ("acme", "globex",
             "initech"), mean_gap_s: float = 2.0, tail_fraction: float = 0.5
             ) -> list[WorkItem]:
    """Head: Zipf-ish popular intents with paraphrases. Tail: templated, mostly unique."""
    rng = random.Random(seed)
    order = INTENTS[:]
    rng.shuffle(order)
    weights = [1.0 / (rank + 1) ** 0.8 for rank in range(len(order))]
    items: list[WorkItem] = []
    t = 0.0
    for _ in range(n):
        tenant = rng.choices(tenants, (0.5, 0.3, 0.2)[: len(tenants)] or None)[0]
        t += rng.expovariate(1.0 / mean_gap_s)
        if rng.random() < tail_fraction:
            tier, prompt = _tail_item(rng)
            items.append(WorkItem(prompt, tenant, "tail:" + prompt, tier, t))
            continue
        intent = rng.choices(order, weights)[0]
        items.append(WorkItem(rng.choice(intent.paraphrases), tenant, intent.id, intent.tier, t))
    return items
