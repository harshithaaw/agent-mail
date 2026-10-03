"""
Reply-generation eligibility gate.

The security screening routing decides how far an email is allowed to travel
through the pipeline. This module is the separate, explicit gate for the
next checkpoint: whether an email is allowed to reach RAG retrieval + Gmail
draft creation.

This is intentionally its own small module rather than an inline `if` inside
the pipeline loop, so it can be unit-tested and audited on its own.

DESIGN DECISION (assumption — confirm or override):
Only "clean_full_pipeline" is treated as eligible for reply drafting.
"flagged_classify_summarize_only" gets full classify+summarize+task-extract
treatment, but its name explicitly says "...summarize_only" — read as a
deliberate signal that it should NOT proceed to reply generation, even though
it's not spam/injection/high-risk-phishing. If you want
flagged_classify_summarize_only to also be reply-eligible (e.g. so the user
can review a low-confidence draft rather than get nothing), add it to
REPLY_ELIGIBLE_ROUTES below — that's a one-line, explicit change, not a
silent one.
"""
import re

# All four security screening routes, per screen_email().
ALL_KNOWN_ROUTES = {
    "clean_full_pipeline",
    "flagged_classify_summarize_only",
    "low_priority_skip_downstream",
    "flagged_skip_reply_generation",
}

# Only routes in this set may proceed to RAG + Gmail draft generation.
REPLY_ELIGIBLE_ROUTES = {"clean_full_pipeline"}

# Automated/bulk sender patterns - follows heuristic style from phishing.py/injection.py
AUTOMATED_SENDER_PATTERNS = [
    r"no-?reply",
    r"do-?not-?reply",
    r"notifications?",
    r"mailer-daemon",
    r"noreply",
    r"donotreply",
    r"auto-?reply",
    r"bounce",
    r"postmaster",
    r"support@",  # Common support addresses (will match domain part)
    r"info@",     # Generic info addresses
    r"admin@",    # Generic admin addresses
]


def is_automated_sender(sender_email: str, list_unsubscribe: str = "", precedence: str = "") -> bool:
    """
    Check if a sender email appears to be from an automated/bulk system.

    Uses regex/substring heuristics on the local-part and domain to match
    common automated sender patterns like no-reply, notifications, mailer-daemon, etc.
    Also checks List-Unsubscribe and Precedence headers.

    Args:
        sender_email: The sender's email address
        list_unsubscribe: Optional List-Unsubscribe header value
        precedence: Optional Precedence header value

    Returns:
        bool: True if the sender appears to be automated, False otherwise
    """
    if list_unsubscribe and list_unsubscribe.strip():
        return True

    if precedence and precedence.lower().strip() in {"bulk", "list", "junk"}:
        return True

    if not sender_email or "@" not in sender_email:
        return False

    # Split into local-part and domain
    local_part = sender_email.split("@")[0].lower()
    domain = sender_email.split("@")[1].lower()
    full_lower = sender_email.lower()

    # Check against automated patterns
    for pattern in AUTOMATED_SENDER_PATTERNS:
        if re.search(pattern, full_lower, re.IGNORECASE):
            return True

    # Additional common bulk sender domain patterns
    bulk_domains = [
        "noreply",
        "no-reply",
        "donotreply",
        "do-not-reply",
        "notifications",
        "mailer",
        "bounce",
    ]

    for bulk_pattern in bulk_domains:
        if bulk_pattern in domain:
            return True

    return False


def should_generate_reply(route: str, sender_email: str, list_unsubscribe: str = "", precedence: str = "") -> bool:
    """
    Returns True only if the given security screening route is allowed to proceed to
    RAG-based reply generation and Gmail draft creation, AND the sender is not
    an automated/bulk system.

    This fails CLOSED on unrecognized input: an unknown route string raises
    rather than silently defaulting to True or False. This is a security
    gate deciding what gets a real Gmail draft written on the user's
    behalf — a typo'd or drifted route string should surface loudly, not
    silently let an email through or silently block one.

    Args:
        route: one of the four security screening route strings from screen_email().
        sender_email: the sender's email address
        list_unsubscribe: Optional List-Unsubscribe header value
        precedence: Optional Precedence header value

    Returns:
        bool: True if reply generation may proceed for this route and sender.

    Raises:
        ValueError: if route is not one of the four known security screening routes.
    """
    if route not in ALL_KNOWN_ROUTES:
        raise ValueError(
            f"should_generate_reply() received an unrecognized route: {route!r}. "
            f"Known routes are: {sorted(ALL_KNOWN_ROUTES)}. "
            f"This usually means screening.py's routing logic and reply_gate.py "
            f"have drifted out of sync — update ALL_KNOWN_ROUTES/REPLY_ELIGIBLE_ROUTES "
            f"deliberately rather than letting this pass silently."
        )
    return route in REPLY_ELIGIBLE_ROUTES and not is_automated_sender(sender_email, list_unsubscribe, precedence)


"""
DOCUMENTATION ADDENDUM:

SUMMARY:
Security checkpoint that determines whether an email is allowed to proceed to
LangGraph-based reply generation with local qwen3:8b. This is the final gate
before LLM operations and Gmail API calls.

FLOW:
1. Called by pipeline code (app.py, gmail/draft.py) before invoking
   the Reply Agent
2. Checks if the security screening route is in REPLY_ELIGIBLE_ROUTES
3. Checks if sender appears to be an automated/bulk system
4. Returns True only if both checks pass
5. Fails closed on unrecognized routes (raises ValueError)

WHY IT'S CALLED:
To prevent wasteful or unsafe reply generation for:
- Emails that failed security screening (spam, phishing, injection)
- Low-priority emails that don't merit full pipeline processing
- Automated senders (no-reply, notifications, etc.) that don't expect replies
- Routes explicitly marked as "summarize only" or "skip reply generation"

WHO CALLS IT:
- gmail/draft.py: create_gmail_draft() function (line 50, 197)
- app.py: Main incoming-mail pipeline orchestrator
- agents/reply/agent.py: Smoke test script (line 222)
- tests/test_reply_gate.py: Unit tests

MAIN FUNCTIONALITY:
- should_generate_reply(route, sender_email): Main gate function that combines
  route-based eligibility with automated sender detection
- is_automated_sender(sender_email): Helper function that uses regex patterns
  to detect common automated/bulk sender patterns

NOTE: Core logic is untouched - this module works the same regardless of whether
the Reply Agent uses the old generator.py approach or the new LangGraph approach.
"""
