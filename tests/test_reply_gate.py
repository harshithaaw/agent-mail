"""
Verifies should_generate_reply() against the real four-route fixture
(test_all_routes.json).

This reuses that fixture rather than inventing new test data, since it
already exercises all four security screening routes with real screen_email() output.

Run this from the project root, same place run_pipeline.py runs from.
"""
import json

from pathlib import Path

from screening import screen_email
from agents.reply.gate import should_generate_reply, REPLY_ELIGIBLE_ROUTES, is_automated_sender

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data/fixtures/test_all_routes.json"


def main():
    with open(TEST_FILE, "r") as f:
        emails = json.load(f)

    print(f"Loaded {len(emails)} emails from {TEST_FILE}.\n")
    print(f"REPLY_ELIGIBLE_ROUTES = {REPLY_ELIGIBLE_ROUTES}\n")

    failures = []
    route_seen = set()

    for i, email in enumerate(emails, 1):
        message_id = email.get("message_id", f"email_{i}")
        sender_email = email["sender_email"]
        body = email["body"]

        screening_result = screen_email(body, sender_email)
        route = screening_result["route"]
        route_seen.add(route)

        eligible = should_generate_reply(route, sender_email)
        expected = route in REPLY_ELIGIBLE_ROUTES and not is_automated_sender(sender_email)

        status = "OK" if eligible == expected else "FAIL"
        if status == "FAIL":
            failures.append((message_id, route, eligible, expected))

        print(f"[{i}/{len(emails)}] {message_id}")
        print(f"    route: {route}")
        print(f"    should_generate_reply: {eligible}  [{status}]")
        print()

    # Unknown-route defensive check: confirm the gate actually raises,
    # rather than trusting the docstring's claim.
    print("--- Fail-closed check on an unrecognized route ---")
    try:
        should_generate_reply("not_a_real_route", "test@example.com")
        print("  FAIL: should_generate_reply() did not raise on an unknown route.")
        failures.append(("<synthetic>", "not_a_real_route", "no exception raised", "ValueError expected"))
    except ValueError as e:
        print(f"  OK: raised ValueError as expected: {e}")

    # Automated sender filter tests
    print("\n--- Automated sender filter tests ---")
    automated_test_cases = [
        ("no-reply@service.com", "clean_full_pipeline", False, "no-reply address should be blocked"),
        ("noreply@company.com", "clean_full_pipeline", False, "noreply address should be blocked"),
        ("notifications@service.com", "clean_full_pipeline", False, "notifications address should be blocked"),
        ("john.smith@company.com", "clean_full_pipeline", True, "normal human sender should be allowed"),
        ("support@service.com", "clean_full_pipeline", False, "support address should be blocked"),
    ]

    for sender, route, expected, description in automated_test_cases:
        result = should_generate_reply(route, sender)
        status = "OK" if result == expected else "FAIL"
        if status == "FAIL":
            failures.append((f"<automated: {sender}>", route, result, expected))
        print(f"  {description}: {result} [{status}]")

    print("\n--- Route coverage ---")
    missing = should_generate_reply.__globals__["ALL_KNOWN_ROUTES"] - route_seen
    if missing:
        print(f"  NOTE: fixture did not exercise these routes: {missing}")
    else:
        print("  All four known routes were exercised by this fixture.")

    print("\n--- Result ---")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    {f}")
    else:
        print("  All checks passed.")


def test_devin_list_unsubscribe_blocked():
    sender = "devin@mail.marketing-example.com"
    route = "clean_full_pipeline"
    list_unsubscribe = "<mailto:unsubscribe@mail.marketing-example.com>"
    allowed = should_generate_reply(route, sender, list_unsubscribe=list_unsubscribe)
    assert not allowed, "Devin-style email with List-Unsubscribe header must be blocked"


def test_reply_gate():
    main()


if __name__ == "__main__":
    main()
