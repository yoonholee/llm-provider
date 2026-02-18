"""Per-token pricing loaded from prices.csv (USD per 1M tokens).

Last updated: 2026-02-17

Pricing sources:
  OpenAI:    https://platform.openai.com/docs/pricing
  Gemini:    https://ai.google.dev/gemini-api/docs/pricing
  Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
  Together:  https://www.together.ai/pricing

Notes:
  - Gemini output prices include thinking tokens
  - Gemini Pro prices are for prompts <=200k tokens (>200k is 2x)
  - Together batch is 50% of serverless for most models
  - OpenAI/Anthropic batch is 50% of standard
"""

import csv
from pathlib import Path

_CSV_PATH = Path(__file__).parent / "prices.csv"

# {model_id: {"input": float, "output": float, "cached_input": float|None, ...}}
PRICES: dict[str, dict[str, float | None]] = {}

with open(_CSV_PATH) as f:
    for row in csv.DictReader(f):
        PRICES[row["model"]] = {
            k: float(row[k]) if row[k] else None
            for k in ("input", "output", "cached_input", "batch_input", "batch_output")
        }


def cost(
    model_id: str, input_tokens: int, output_tokens: int, batch: bool = False
) -> float | None:
    """Calculate cost in USD. Returns None if model not in pricing table."""
    p = PRICES.get(model_id)
    if p is None:
        return None
    ik = "batch_input" if batch else "input"
    ok = "batch_output" if batch else "output"
    if p[ik] is None or p[ok] is None:
        return None
    return (input_tokens * p[ik] + output_tokens * p[ok]) / 1_000_000
