"""Observability: a local trace for every run, plus optional Langfuse export.

The local trace always runs, so tests and CI can assert on what happened (which guard fired,
which tools ran, token counts, latency). When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
and the `langfuse` package is installed, the LangGraph run is also sent to Langfuse.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    events: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    def log(self, kind: str, **data: Any) -> None:
        self.events.append({"t_ms": round((time.perf_counter() - self.started) * 1000, 2),
                            "kind": kind, **data})

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)

    def cost_usd(self) -> float:
        """Estimated cost from per-million-token prices you set for your model.

        Prices are read from CLAUDE_INPUT_USD_PER_MTOK / CLAUDE_OUTPUT_USD_PER_MTOK so they stay
        current with your contract; both default to 0, which reports token counts only.
        """
        in_price = float(os.environ.get("CLAUDE_INPUT_USD_PER_MTOK", "0"))
        out_price = float(os.environ.get("CLAUDE_OUTPUT_USD_PER_MTOK", "0"))
        return round(self.input_tokens / 1e6 * in_price + self.output_tokens / 1e6 * out_price, 6)


def langfuse_callbacks() -> list[Any]:
    """Return a Langfuse LangChain callback handler if Langfuse is configured, else []."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse >= 3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse 2.x
        except ImportError:
            return []
    return [CallbackHandler()]
