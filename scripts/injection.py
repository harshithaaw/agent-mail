"""Compatibility wrapper for local prompt-injection detection.

Gemini-backed injection analysis was removed. The active Security Agent uses
the same local heuristic directly from agents.security.injection.
"""
from agents.security.injection import heuristic_injection_check


def check_injection(cleaned_text):
    flagged, matches = heuristic_injection_check(cleaned_text)
    return flagged, {"method": "local_heuristic", "matched_patterns": matches}


def llm_injection_check(_cleaned_text):
    raise RuntimeError("Cloud LLM injection checks were removed; use the local Security Agent.")
