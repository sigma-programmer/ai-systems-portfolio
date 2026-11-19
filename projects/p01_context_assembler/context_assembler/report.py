"""Assembly report: what was included, dropped and compressed, and where tokens went."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BlockReport(BaseModel):
    name: str
    kind: str
    priority: int
    required: bool
    strategy: str
    demand: int
    floor: int
    allocated: int
    used: int
    items_in: int
    items_kept: int
    dropped_ids: list[str] = Field(default_factory=list)
    compressed_ids: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)

    @property
    def status(self) -> str:
        if self.items_in == 0:
            return "empty"
        if self.items_kept == 0:
            return "dropped"
        if self.dropped_ids or self.compressed_ids:
            return "compressed"
        return "full"


class AssemblyReport(BaseModel):
    window: int
    reserve_output: int
    safety_margin: int = 0
    budget: int  # tokens available to the input after reserve, margin and framing
    estimated_tokens: int
    payload_tokens: int
    by_kind: dict[str, int] = Field(default_factory=dict)
    blocks: list[BlockReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    redistribution_rounds: int = 0
    repair_rounds: int = 0
    latency_ms: float = 0.0

    @property
    def input_limit(self) -> int:
        return self.window - self.reserve_output - self.safety_margin

    @property
    def headroom(self) -> int:
        return self.input_limit - self.payload_tokens

    @property
    def utilization(self) -> float:
        return round(self.payload_tokens / self.input_limit, 4) if self.input_limit > 0 else 0.0

    def block(self, name: str) -> BlockReport:
        return next(b for b in self.blocks if b.name == name)

    def to_dict(self) -> dict:
        d = self.model_dump()
        d.update(input_limit=self.input_limit, headroom=self.headroom, utilization=self.utilization)
        for b, rep in zip(d["blocks"], self.blocks):
            b["status"] = rep.status
        return d

    def render_text(self) -> str:
        lines = [
            f"window={self.window} reserve_output={self.reserve_output} input_limit={self.input_limit} "
            f"payload={self.payload_tokens} (est {self.estimated_tokens}) utilization={self.utilization:.1%} "
            f"headroom={self.headroom} latency={self.latency_ms:.2f}ms",
            f"{'block':16} {'kind':12} {'pri':>3} {'req':>3} {'demand':>7} {'floor':>6} {'alloc':>6} {'used':>6} "
            f"{'kept':>7}  status",
        ]
        for b in self.blocks:
            lines.append(
                f"{b.name[:16]:16} {b.kind:12} {b.priority:>3} {'y' if b.required else '':>3} {b.demand:>7} "
                f"{b.floor:>6} {b.allocated:>6} {b.used:>6} {f'{b.items_kept}/{b.items_in}':>7}  {b.status}"
            )
            for a in b.actions:
                lines.append(f"{'':20}- {a}")
        lines.append("tokens by kind: " + ", ".join(f"{k}={v}" for k, v in self.by_kind.items()))
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)
