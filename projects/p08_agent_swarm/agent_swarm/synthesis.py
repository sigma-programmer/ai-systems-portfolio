"""Program-synthesis-lite: a candidate space of library programs + AST mutants.

The offline Coder is a small counterexample-guided inductive synthesis (CEGIS)
loop. The *search space* is built here:

1. library templates for the function (``library.CANDIDATES``), and
2. single-point AST mutants of each template — comparison flips
   (``<``↔``<=``, ``>``↔``>=``, ``==``↔``!=``), integer constant ±1, and
   ``and``↔``or`` swaps — the classic off-by-one / boundary repair operators.

Candidates are ranked by an Occam prior (library order, simpler first; mutants
slightly below their parent) plus a bonus for tags that match words in the
researcher's requirements. The coder then picks the best-ranked candidate that
is consistent with every example and counterexample seen so far.
"""
from __future__ import annotations

import ast
import copy
import functools
from dataclasses import dataclass, field

from aikit import tokenize_words

from .library import CANDIDATES, CandidateTemplate
from .messages import source_hash

TAG_BONUS = 0.6
MUTANT_PENALTY = 0.5

_CMP_SWAPS: dict[type, type] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}


@dataclass
class Candidate:
    """One point in the program search space."""

    source: str
    origin: str
    prior: float
    tags: tuple[str, ...] = ()
    score: float = 0.0
    hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = source_hash(self.source)


def _sites(tree: ast.AST) -> list[tuple[str, int]]:
    """Enumerate mutation sites as (operator, ordinal) pairs in walk order."""
    sites: list[tuple[str, int]] = []
    cmp_i = const_i = bool_i = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for _ in node.ops:
                sites.append(("cmp", cmp_i))
                cmp_i += 1
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(("const+", const_i))
            sites.append(("const-", const_i))
            const_i += 1
        elif isinstance(node, ast.BoolOp):
            sites.append(("bool", bool_i))
            bool_i += 1
    return sites


def _apply(tree: ast.AST, op: str, ordinal: int) -> bool:
    """Apply one mutation in place; return False if the site does not mutate."""
    cmp_i = const_i = bool_i = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for k, o in enumerate(node.ops):
                if op == "cmp" and cmp_i == ordinal:
                    swap = _CMP_SWAPS.get(type(o))
                    if swap is None:
                        return False
                    node.ops[k] = swap()
                    return True
                cmp_i += 1
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            if op in ("const+", "const-") and const_i == ordinal:
                node.value = node.value + (1 if op == "const+" else -1)
                return True
            const_i += 1
        elif isinstance(node, ast.BoolOp):
            if op == "bool" and bool_i == ordinal:
                node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                return True
            bool_i += 1
    return False


def mutants(source: str) -> list[tuple[str, str]]:
    """All single-point mutants of ``source`` as (description, new_source)."""
    return list(_mutants_cached(source))


@functools.lru_cache(maxsize=512)
def _mutants_cached(source: str) -> tuple[tuple[str, str], ...]:
    try:
        base = ast.parse(source)
    except SyntaxError:
        return ()
    out: list[tuple[str, str]] = []
    seen = {source_hash(source)}
    for op, i in _sites(base):
        tree = copy.deepcopy(base)
        if not _apply(tree, op, i):
            continue
        new = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        h = source_hash(new)
        if h not in seen:
            seen.add(h)
            out.append((f"{op}#{i}", new))
    return tuple(out)


def requirement_words(requirements: list[str]) -> set[str]:
    """Lower-cased word set over all requirement strings."""
    return {w for r in requirements for w in tokenize_words(r)}


def build_pool(func_name: str, templates: list[CandidateTemplate] | None = None,
               with_mutants: bool = True) -> list[Candidate]:
    """Library templates + mutants for ``func_name`` (deduplicated, unranked)."""
    templates = templates if templates is not None else CANDIDATES.get(func_name, [])
    pool: list[Candidate] = []
    seen: set[str] = set()
    for idx, tpl in enumerate(templates):
        c = Candidate(tpl.source, f"library[{idx}]", prior=-float(idx), tags=tpl.tags)
        if c.hash not in seen:
            seen.add(c.hash)
            pool.append(c)
    if with_mutants:
        for idx, tpl in enumerate(templates):
            for j, (desc, src) in enumerate(mutants(tpl.source)):
                c = Candidate(src, f"mutant(library[{idx}], {desc})",
                              prior=-float(idx) - MUTANT_PENALTY - 0.001 * j, tags=tpl.tags)
                if c.hash not in seen:
                    seen.add(c.hash)
                    pool.append(c)
    return pool


def rank(pool: list[Candidate], requirements: list[str]) -> list[Candidate]:
    """Score = Occam prior + TAG_BONUS per tag found in the requirements; stable sort."""
    words = requirement_words(requirements)
    for c in pool:
        c.score = c.prior + TAG_BONUS * sum(1 for t in c.tags if t in words)
    return sorted(pool, key=lambda c: -c.score)
