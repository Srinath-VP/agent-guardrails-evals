from __future__ import annotations

import json

from agent_guardrails.evals import runner


def test_suite_matches_committed_baseline():
    summary = runner.summarize(runner.run_suite("scripted"))
    baseline = json.loads(runner.BASELINE.read_text())
    assert runner.compare(summary, baseline) == []


def test_every_owasp_category_and_benign_cases_are_covered():
    categories = {c["category"].split()[0] for c in runner.load_cases()}
    assert {"LLM01", "LLM02", "LLM06", "LLM07", "LLM10", "Benign"} <= categories


def test_regression_gate_catches_a_weakened_guard(monkeypatch):
    baseline = json.loads(runner.BASELINE.read_text())
    # Simulate someone loosening the egress allowlist to "make a test pass".
    from agent_guardrails import tools
    monkeypatch.setattr(tools, "ALLOWED_EMAIL_DOMAINS", {"example.com", "evil.example", "gmail.com"})
    summary = runner.summarize(runner.run_suite("scripted"))
    regressions = runner.compare(summary, baseline)
    assert regressions and "LLM01-indirect-kb" in regressions[0]
