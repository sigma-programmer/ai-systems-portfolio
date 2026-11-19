"""Architecture Decision Records: parse + validate Markdown ADRs.

Format (Michael Nygard style):

    # ADR-0001: Title
    - Status: Accepted | Proposed | Superseded by ADR-XXXX | Deprecated
    - Date: YYYY-MM-DD
    - Projects: p01, p11
    ## Context
    ## Decision
    ## Consequences
    ## Alternatives Considered
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel, field_validator

REQUIRED_SECTIONS = ("Context", "Decision", "Consequences", "Alternatives Considered")
_STATUS = re.compile(r"^(Proposed|Accepted|Deprecated|Superseded by ADR-\d{4})$")


class ADR(BaseModel):
    number: int
    title: str
    status: str
    date: date
    projects: list[str]
    sections: dict[str, str]
    path: str = ""

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if not _STATUS.match(v):
            raise ValueError(f"invalid status {v!r}")
        return v


def parse_adr(text: str, path: str = "") -> ADR:
    m = re.search(r"^#\s*ADR-(\d{4}):\s*(.+)$", text, re.M)
    if not m:
        raise ValueError(f"{path}: missing '# ADR-NNNN: Title' header")
    meta = dict(re.findall(r"^-\s*(Status|Date|Projects):\s*(.+?)\s*$", text, re.M))
    for k in ("Status", "Date", "Projects"):
        if k not in meta:
            raise ValueError(f"{path}: missing '- {k}:' field")
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+(.+?)\s*$", text, flags=re.M)
    for name, body in zip(parts[1::2], parts[2::2]):
        sections[name.strip()] = body.strip()
    return ADR(
        number=int(m.group(1)), title=m.group(2).strip(), status=meta["Status"],
        date=date.fromisoformat(meta["Date"]),
        projects=[p.strip() for p in meta["Projects"].split(",") if p.strip()],
        sections=sections, path=path,
    )


def validate_adr(adr: ADR, known_numbers: set[int] | None = None) -> list[str]:
    errors = []
    for s in REQUIRED_SECTIONS:
        if not adr.sections.get(s):
            errors.append(f"ADR-{adr.number:04d}: section '{s}' missing or empty")
    for p in adr.projects:
        if not re.fullmatch(r"p\d{2}|all", p):
            errors.append(f"ADR-{adr.number:04d}: bad project ref {p!r}")
    sup = re.match(r"Superseded by ADR-(\d{4})", adr.status)
    if sup and known_numbers is not None and int(sup.group(1)) not in known_numbers:
        errors.append(f"ADR-{adr.number:04d}: superseded by unknown ADR-{sup.group(1)}")
    return errors


def load_adrs(directory: Path) -> tuple[list[ADR], list[str]]:
    adrs, errors = [], []
    for f in sorted(Path(directory).glob("*.md")):
        if f.name.lower() == "readme.md":
            continue
        try:
            adrs.append(parse_adr(f.read_text(), str(f)))
        except Exception as e:  # noqa: BLE001 - collect all parse failures
            errors.append(str(e))
    nums = [a.number for a in adrs]
    dupes = {n for n in nums if nums.count(n) > 1}
    errors += [f"duplicate ADR number {n:04d}" for n in sorted(dupes)]
    known = set(nums)
    for a in adrs:
        errors += validate_adr(a, known)
    return sorted(adrs, key=lambda a: a.number), errors
