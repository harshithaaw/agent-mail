from datetime import datetime, timedelta, timezone

from agents.understanding import deadline_detector as detector


IST = timezone(timedelta(hours=5, minutes=30))
BASE = datetime(2026, 10, 3, 10, 0, tzinfo=IST)


def test_explicit_deadline_date_and_action():
    assert detector.detect_b(
        "Please upload the final report by October 9, 2026.", BASE.isoformat()
    ) == [{"date": "2026-10-09", "kind": "submission", "action": "upload the final report"}]


def test_relative_date_uses_received_at():
    assert detector.detect_b("Please reply by tomorrow.", BASE.isoformat()) == [
        {"date": "2026-10-04", "kind": "response", "action": "reply"}
    ]


def test_end_of_day_uses_received_day():
    assert detector.detect_b("Pay by EOD.", BASE.isoformat()) == [
        {"date": "2026-10-03", "kind": "payment", "action": "pay"}
    ]


def test_event_date_is_not_a_deadline():
    assert detector.detect_b("The webinar is on October 9, 2026.", BASE.isoformat()) == []


def test_past_date_is_ignored():
    assert detector.detect_b("Please submit by 2026-10-02.", BASE.isoformat()) == []


def test_quoted_text_is_ignored_and_forwarded_text_is_live():
    quoted = "> Reply by October 9.\nPlease respond by October 11."
    assert detector.detect_b(quoted, BASE.isoformat()) == [
        {"date": "2026-10-11", "kind": "response", "action": "respond"}
    ]
    forwarded = "Forwarded message\nPlease pay by October 9."
    assert detector.detect_b(forwarded, BASE.isoformat()) == [
        {"date": "2026-10-09", "kind": "payment", "action": "pay"}
    ]


def test_before_only_governs_a_nearby_date():
    received_at = BASE.isoformat()
    text = "Please pay the balance by October 24 before the account renewal date arrives November 3."
    assert [r["date"] for r in detector.detect_b(text, received_at)] == ["2026-10-24"]
    assert [r["date"] for r in detector.detect_b(
        "The report is needed before 17 October.", received_at
    )] == ["2026-10-17"]
    assert [r["date"] for r in detector.detect_b(
        "Please confirm before 18 October.", received_at
    )] == ["2026-10-18"]


def test_sale_end_phrase_is_vetoed_and_other_expiry_cues_work():
    received_at = BASE.isoformat()
    assert detector.detect_b("The flash sale offer ends tonight at midnight.", received_at) == []
    assert detector.detect_b("The store sale ends next Sunday.", received_at) == []
    assert [r["date"] for r in detector.detect_b(
        "The trial expires on 21 October.", received_at
    )] == ["2026-10-21"]
    assert [r["date"] for r in detector.detect_b(
        "Registration closes on 19 October.", received_at
    )] == ["2026-10-19"]


def test_elliptical_sender_promise_is_ignored():
    assert detector.detect_b(
        "Please send the agenda by October 16, and will forward the notes by October 17.",
        BASE.isoformat(),
    ) == [{"date": "2026-10-16", "kind": "submission", "action": "send the agenda"}]


def test_same_date_and_kind_is_deduplicated_across_actions():
    assert detector.detect_b(
        "Please upload the draft by October 18 and send the revision by October 18.",
        BASE.isoformat(),
    ) == [{"date": "2026-10-18", "kind": "submission", "action": "upload the draft"}]


def test_clock_and_zone_before_date_are_converted_to_ist_day():
    assert detector.detect_b(
        "Please send the binder at 8 PM EST on December 31, 2026.",
        BASE.isoformat(),
    ) == [{"date": "2027-01-01", "kind": "submission", "action": "send the binder at 8 pm est"}]


def test_clock_and_zone_after_date_are_converted_to_ist_day():
    assert detector.detect_b(
        "Please submit by December 31, 2026 at 8 PM EST.",
        BASE.isoformat(),
    ) == [{"date": "2027-01-01", "kind": "submission", "action": "submit"}]


def test_missing_or_timezone_naive_received_at_is_skipped(caplog):
    assert detector.detect_b("Please reply by October 9.", None) == []
    assert "missing received_at" in caplog.text
    assert detector.detect_b("Please reply by October 9.", "2026-10-03T10:00:00") == []
    assert "has no timezone" in caplog.text
