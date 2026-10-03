import re


# Local deterministic copy of the heuristic patterns used by the original
# injection checker. This module intentionally has no model or network calls.
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"act as (an?|the)",
    r"forget (?:your|the)? (?:previous|prior|above|all) instructions",
    r"forget everything",
    r"override (?:your|the)? (?:system|default) behavior",
    r"reveal your (instructions|prompt|system prompt)",
    r"do not (tell|inform|notify) the user",
    r"forward (this|all) (email|emails|message)s? to (?:everyone|all contacts)",
]


def heuristic_injection_check(cleaned_text: str) -> tuple[bool, list[str]]:
    """Flag known injection phrases using the local regex pattern list."""
    matches = [
        pattern
        for pattern in INJECTION_PATTERNS
        if re.search(pattern, cleaned_text, re.IGNORECASE)
    ]
    return bool(matches), matches


def run_injection(cleaned_text: str) -> dict:
    try:
        flagged, matches = heuristic_injection_check(cleaned_text)
        return {"flagged": flagged, "details": {"matched_patterns": matches}}
    except Exception as exc:
        return {"flagged": None, "details": {}, "error": str(exc)}
