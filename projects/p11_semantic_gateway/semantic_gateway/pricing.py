"""Model catalogue and token pricing (USD per million tokens)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    tier: str            # "simple" | "moderate" | "complex"
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


HAIKU = ModelSpec("claude-haiku-5-5", "simple", 0.10, 0.50)
SONNET = ModelSpec("claude-sonnet-5-5", "moderate", 2.00, 10.00)
OPUS = ModelSpec("claude-opus-5-5", "complex", 4.00, 20.00)

MODELS: dict[str, ModelSpec] = {m.model_id: m for m in (HAIKU, SONNET, OPUS)}
TIER_ORDER = ("simple", "moderate", "complex")
DEFAULT_TIER_MODEL = {"simple": HAIKU.model_id, "moderate": SONNET.model_id,
                      "complex": OPUS.model_id}


def cost_of(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call; raises KeyError for unknown models."""
    return MODELS[model_id].cost(input_tokens, output_tokens)
