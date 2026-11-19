"""Labeled evaluation set: answerable, unanswerable, and adversarial questions."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabeledQuestion:
    question: str
    kind: str  # "answerable" | "unanswerable" | "adversarial"
    answers: tuple[str, ...] = ()   # all must appear (case-insensitive) in a correct answer
    gold_docs: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default=())


def _a(q: str, answers: tuple[str, ...], docs: tuple[str, ...], *tags: str) -> LabeledQuestion:
    return LabeledQuestion(q, "answerable", answers, docs, tags)


def _u(q: str) -> LabeledQuestion:
    return LabeledQuestion(q, "unanswerable")


ANSWERABLE: list[LabeledQuestion] = [
    _a("When was Lumora Labs founded?", ("2014",), ("lumora-overview", "lumora-ceo")),
    _a("Where is Lumora headquartered?", ("Oslo",), ("lumora-overview",)),
    _a("How long does the Beacon battery last?", ("18 hours",), ("beacon-specs",)),
    _a("Is the Beacon waterproof?", ("not waterproof",), ("beacon-specs",)),
    _a("How much does the Beacon cost?", ("149",), ("beacon-pricing",), "synonym"),
    _a("How many vacation days do Lumora employees get?", ("25 days",), ("pto",), "synonym"),
    _a("Within how many days can customers return a product for a refund?", ("30 days",),
       ("refunds",)),
    _a("How long is the warranty on Lumora devices?", ("2 year",), ("warranty",)),
    _a("Does the warranty cover water damage?", ("does not cover water damage",), ("warranty",)),
    _a("How many days per week can employees work remotely?", ("3 days",), ("remote-work",)),
    _a("How long is parental leave at Lumora?", ("16 weeks",), ("parental-leave",)),
    _a("When is customer support available?", ("8 am to 8 pm",), ("support",)),
    _a("How much does the Lumora bug bounty pay?", ("20000",), ("security",)),
    _a("Does the Pulse support Z-Wave?", ("does not support Z-Wave",), ("pulse-specs",)),
    _a("How much did Lumora raise in its Series B?", ("40 million",), ("funding",)),
    _a("How many moons does Mars have?", ("two small moons",), ("mars",)),
    _a("How long is a day on Mars?", ("24.6",), ("mars",)),
    _a("Who created Python?", ("Guido van Rossum",), ("python",)),
    _a("How tall is Mount Everest?", ("8849",), ("everest",), "synonym"),
    _a("Which coffee species has more caffeine?", ("Robusta",), ("coffee",)),
    _a("When was Lumora founded and who is its CEO?", ("2014", "Dana Whitfield"),
       ("lumora-overview", "lumora-ceo"), "multi-part"),
    _a("What is the Beacon battery life and how much does it weigh?", ("18 hours", "640 grams"),
       ("beacon-specs",), "multi-part"),
    _a("How many PTO days do employees get and how many public holidays?",
       ("25 days", "10 public holidays"), ("pto",), "multi-part"),
    _a("Within how many days can customers return a product and how quickly are refunds issued?",
       ("30 days", "5 business days"), ("refunds",), "multi-part"),
]

UNANSWERABLE: list[LabeledQuestion] = [
    _u("What is the salary of the Lumora CEO?"),
    _u("Does Lumora offer a stock option plan for employees?"),
    _u("What is the population of Oslo?"),
    _u("How many moons does Jupiter have?"),
    _u("What is the battery life of the Lumora Pulse?"),
    _u("Who founded Fjord Ventures?"),
    _u("What is the boiling point of water on Mars?"),
    _u("When was the Pulse thermostat discontinued?"),
    _u("Who climbed Everest in 1975?"),
    _u("What programming language is the Lumora firmware written in?"),
    _u("How many employees work in the Lumora Berlin office?"),
    _u("What is the dental insurance plan at Lumora?"),
]

# Adversarial = answerable questions answered by a hallucinating generator.
ADVERSARIAL: list[LabeledQuestion] = [
    LabeledQuestion(q.question, "adversarial", q.answers, q.gold_docs, q.tags)
    for q in ANSWERABLE[::2]
]


def all_questions() -> list[LabeledQuestion]:
    return ANSWERABLE + UNANSWERABLE + ADVERSARIAL
