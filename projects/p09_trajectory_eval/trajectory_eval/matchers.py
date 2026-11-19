"""Argument matchers for golden steps.

A golden step constrains a tool call's arguments with one matcher per key:

* ``exact``    – value equality (after JSON normalisation)
* ``regex``    – ``re.fullmatch`` against ``str(value)``
* ``schema``   – a small JSON-Schema subset (type/enum/min/max/pattern/
                 minLength/maxLength/required/properties/items)
* ``semantic`` – cosine similarity of HashingEmbedder vectors >= threshold
* ``any``      – key must be present, any value

Each matcher returns a score in [0, 1]; a match requires score >= its
pass threshold (1.0 for crisp matchers, ``threshold`` for semantic).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from aikit import HashingEmbedder, cosine_sim

MatcherKind = Literal["exact", "regex", "schema", "semantic", "any"]

_EMBEDDER = HashingEmbedder(dim=512)


@lru_cache(maxsize=4096)
def _semantic(a: str, b: str) -> float:
    return max(0.0, float(cosine_sim(_EMBEDDER.embed_one(a), _EMBEDDER.embed_one(b))))


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of hashed embeddings, clipped to [0, 1]."""
    return _semantic(str(a), str(b))


_TYPE_CHECK = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``value`` against a JSON-Schema subset; return a list of errors."""
    errors: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPE_CHECK[x](value) for x in types):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} !~ /{schema['pattern']}/")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


class ArgMatcher(BaseModel):
    """Constraint on one argument key of a tool call."""

    key: str
    kind: MatcherKind = "exact"
    value: Any = None
    pattern: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    required: bool = True

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_fields(self) -> "ArgMatcher":
        if self.kind == "regex":
            if not self.pattern:
                raise ValueError("regex matcher needs 'pattern'")
            re.compile(self.pattern)
        if self.kind == "schema" and self.schema_ is None:
            raise ValueError("schema matcher needs 'schema'")
        if self.kind == "semantic" and not isinstance(self.value, str):
            raise ValueError("semantic matcher needs a string 'value'")
        return self

    def score(self, args: dict[str, Any]) -> float:
        """Return a match score in [0, 1] for the given argument dict."""
        if self.key not in args:
            return 0.0 if self.required else 1.0
        v = args[self.key]
        if self.kind == "any":
            return 1.0
        if self.kind == "exact":
            norm = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
            return 1.0 if norm(v) == norm(self.value) else 0.0
        if self.kind == "regex":
            return 1.0 if re.fullmatch(self.pattern or "", str(v)) else 0.0
        if self.kind == "schema":
            return 0.0 if validate_schema(v, self.schema_ or {}) else 1.0
        return semantic_similarity(str(v), str(self.value))

    def matches(self, args: dict[str, Any]) -> bool:
        s = self.score(args)
        return s >= (self.threshold if self.kind == "semantic" else 1.0)


def match_args(matchers: list[ArgMatcher], args: dict[str, Any]) -> tuple[bool, float, list[str]]:
    """Apply all matchers. Returns (all matched, mean score, failure notes)."""
    if not matchers:
        return True, 1.0, []
    scores: list[float] = []
    notes: list[str] = []
    ok = True
    for m in matchers:
        s = m.score(args)
        scores.append(s)
        if not m.matches(args):
            ok = False
            notes.append(f"arg {m.key!r} failed {m.kind} matcher (score={s:.2f})")
    return ok, sum(scores) / len(scores), notes
