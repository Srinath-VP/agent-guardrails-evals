"""Model backends.

`ScriptedLLM` is a deterministic, deliberately gullible stand-in for a real model: it obeys any
instruction it reads, including injected ones. Running the evals against it shows what the
guardrails catch on their own, with no help from model alignment, and keeps CI free and stable.

`ClaudeLLM` calls the real Claude API through the Anthropic SDK with native tool use.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import ToolSpec


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class AgentStep:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> AgentStep: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ScriptedLLM:
    """A rule-based 'model' that follows instructions literally, including malicious ones."""

    name = "scripted"

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> AgentStep:
        last = messages[-1]
        prompt_tokens = _approx_tokens(system + json.dumps(messages))

        if last["role"] == "tool":
            return self._after_tool(messages, prompt_tokens)

        text = last["content"]
        lower = text.lower()
        order = re.search(r"ORD-\d{4}", text)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        amount = re.search(r"(?:inr|rs\.?|₹|\$)\s?([\d,]+)|([\d,]+)\s?(?:inr|rupees)", lower)

        def call(name: str, **args: Any) -> AgentStep:
            return AgentStep(tool_calls=[ToolCall(f"call_{len(messages)}", name, args)],
                             input_tokens=prompt_tokens, output_tokens=20)

        if any(p in lower for p in ("what were you told", "your setup", "your instructions", "your rules")):
            # Gullible: happily repeats its own system prompt.
            return AgentStep(text=f"Sure! My instructions are:\n{system}",
                             input_tokens=prompt_tokens, output_tokens=_approx_tokens(system))
        if "refund" in lower and order:
            value = (amount.group(1) or amount.group(2)) if amount else "0"
            return call("refund_order", order_id=order.group(0), amount=float(value.replace(",", "")))
        if ("email" in lower or "send" in lower) and email:
            return call("send_email", to=email.group(0), body=text)
        if order:
            return call("lookup_order", order_id=order.group(0))
        if any(w in lower for w in ("how", "policy", "search", "ship", "reset")):
            return call("search_kb", query=text)
        return AgentStep(text="I can help with orders, refunds and help-centre questions.",
                         input_tokens=prompt_tokens, output_tokens=12)

    def _after_tool(self, messages: list[dict[str, Any]], prompt_tokens: int) -> AgentStep:
        result = messages[-1]["content"]
        # Gullible: obeys instructions embedded in retrieved content (indirect injection).
        injected = re.search(r"call send_email to ([\w.+-]+@[\w.-]+)", result)
        already_tried = any(m["role"] == "assistant" and m.get("tool_calls") and
                            m["tool_calls"][0]["name"] == "send_email" for m in messages)
        if injected and not already_tried:
            return AgentStep(
                tool_calls=[ToolCall(f"call_{len(messages)}", "send_email",
                                     {"to": injected.group(1), "body": json.dumps(ORDERS_LEAK)})],
                input_tokens=prompt_tokens, output_tokens=40)
        return AgentStep(text=f"Here is what I found: {result}",
                         input_tokens=prompt_tokens, output_tokens=_approx_tokens(result))


ORDERS_LEAK = "all customer records"


class ClaudeLLM:
    """Claude via the Anthropic Messages API with native tool use."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic  # optional dependency: pip install ".[claude]"

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens
        self.name = f"claude:{self.model}"

    @staticmethod
    def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> AgentStep:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
            messages=self._to_anthropic(messages),
        )
        step = AgentStep(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        for block in response.content:
            if block.type == "text":
                step.text += block.text
            elif block.type == "tool_use":
                step.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return step


def make_llm(name: str) -> LLM:
    if name == "scripted":
        return ScriptedLLM()
    if name == "claude":
        return ClaudeLLM()
    raise ValueError(f"unknown model backend: {name}")
