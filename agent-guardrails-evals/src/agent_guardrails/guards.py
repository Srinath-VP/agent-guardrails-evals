"""Guardrails that sit outside the model.

The design assumption is that the model *will* be fooled sometimes, so every control here is
deterministic and runs before or after the model, never inside it:

  input guard   -> size limit, prompt-injection detection (incl. base64-smuggled text), PII redaction
  tool guard    -> allowlist, argument checks, spend limits, human approval, egress domain allowlist
  output guard  -> system-prompt leakage (canary), secret leakage, PII redaction
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 4_000

# A random marker embedded in the system prompt. If it ever appears in output, the prompt leaked.
SYSTEM_PROMPT_CANARY = "CANARY-7f3a9c"

INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|prompts?|directions)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|guidelines|policy|policies)\b",
    r"\bforget\b.{0,30}\b(instructions?|rules|everything)\b",
    r"\b(reveal|print|show|repeat|output|display)\b.{0,40}\b(system prompt|your instructions|hidden instructions|initial prompt)\b",
    r"\byou are now\b.{0,40}\b(dan|unrestricted|jailbroken|developer mode|no rules)\b",
    r"\b(developer|god|admin|sudo) mode\b",
    r"\bact as\b.{0,30}\bwithout (any )?(restrictions|rules|filters)\b",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "AADHAAR": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "PHONE": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?\d{5}[ -]?\d{5}(?!\w)"),
}
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),  # API-key shaped strings
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _decoded_base64_chunks(text: str) -> list[str]:
    decoded = []
    for chunk in _BASE64_RE.findall(text):
        try:
            raw = base64.b64decode(chunk + "=" * (-len(chunk) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        candidate = raw.decode("utf-8", errors="ignore")
        if candidate and sum(c.isprintable() for c in candidate) / len(candidate) > 0.9:
            decoded.append(candidate)
    return decoded


def detect_injection(text: str) -> list[str]:
    """Return the reasons the text looks like a prompt-injection attempt (empty if none)."""
    reasons = []
    for source, body in [("text", text)] + [("base64", d) for d in _decoded_base64_chunks(text)]:
        for pattern in _INJECTION_RE:
            if pattern.search(body):
                reasons.append(f"prompt injection pattern in {source}: /{pattern.pattern[:40]}.../")
                break
    return reasons


# Inbound, email addresses stay visible because the agent needs them to route messages (the tool
# guard's egress allowlist decides where mail may go). High-risk identifiers never reach the model.
INPUT_REDACT = ("CARD", "AADHAAR", "PAN", "PHONE")
OUTPUT_REDACT = tuple(PII_PATTERNS)


def redact_pii(text: str, kinds: tuple[str, ...] = OUTPUT_REDACT) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def _sub(label: str, pattern: re.Pattern[str], value: str) -> str:
        def repl(match: re.Match[str]) -> str:
            if label == "CARD" and not _luhn_ok(match.group(0)):
                return match.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"
        return pattern.sub(repl, value)

    for label, pattern in PII_PATTERNS.items():
        if label in kinds:
            text = _sub(label, pattern, text)
    return text, counts


def input_guard(text: str) -> GuardResult:
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, "", [f"input exceeds {MAX_INPUT_CHARS} characters (LLM10)"])
    reasons = detect_injection(text)
    if reasons:
        return GuardResult(False, "", reasons)
    clean, counts = redact_pii(text, INPUT_REDACT)
    return GuardResult(True, clean, [], counts)


def output_guard(text: str, system_prompt: str) -> GuardResult:
    reasons = []
    if SYSTEM_PROMPT_CANARY in text:
        reasons.append("system prompt leakage: canary found in output (LLM07)")
    # Long verbatim overlap with the system prompt is also leakage, even without the canary.
    for line in (ln.strip() for ln in system_prompt.splitlines()):
        if len(line) > 40 and line in text:
            reasons.append("system prompt leakage: verbatim instructions in output (LLM07)")
            break
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            reasons.append("secret-shaped string in output (LLM02)")
            break
    if reasons:
        return GuardResult(False, "", reasons)
    clean, counts = redact_pii(text)
    return GuardResult(True, clean, [], counts)
