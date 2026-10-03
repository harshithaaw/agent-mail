import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import dashboard


@pytest.fixture
def dashboard_db(tmp_path):
    path = tmp_path / "processed.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE processed (
                message_id TEXT, thread_id TEXT, sender TEXT, subject TEXT,
                route TEXT, status TEXT, draft_id TEXT, processed_at TEXT, attempts INTEGER,
                body TEXT
            );
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY, started_at TEXT, finished_at TEXT, trigger TEXT,
                status TEXT, reason TEXT, counts_json TEXT
            );
        """)
        db.executemany(
            "INSERT INTO processed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("m1", "t1", "a@example.com", "Draft", "reply", "draft_created", "d1", "2026-10-01T08:00:00+00:00", 1, "secret body"),
                ("m2", "t2", "b@example.com", "Skip", "automated_sender", "skipped_automated_sender", None, "2026-10-01T08:01:00+00:00", 1, "secret body"),
                ("m3", "t3", "c@example.com", "Flag", "flagged_phishing", "flagged_phishing", None, "2026-10-01T08:02:00+00:00", 1, "secret body"),
                ("m4", "t4", "d@example.com", "Failure", "reply", "generation_failed", None, "2026-10-01T08:03:00+00:00", 1, "secret body"),
                ("m5", "t5", "e@example.com", "Low priority", "low_priority_skip_downstream", "low_priority_skip_downstream", None, "2026-10-01T08:04:00+00:00", 1, "secret body"),
                ("old", "t5", "e@example.com", "Yesterday", "reply", "draft_created", None, "2026-09-30T23:59:00+00:00", 1, "secret body"),
            ],
        )
        db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "2026-10-01T08:00:00+00:00", "2026-10-01T08:04:00+00:00", "test", "preflight_failed", "gmail_login_expired", json.dumps({"drafts_created": 1})),
        )
    return path


def test_load_dashboard_data_returns_today_counters_safe_rows_and_runs(dashboard_db):
    data = dashboard.load_dashboard_data(dashboard_db, datetime(2026, 10, 1, 12, tzinfo=timezone.utc))
    assert data["counters"] == {"drafts_created": 1, "skipped": 2, "flagged": 1, "failed": 1}
    assert len(data["processed_emails"]) == 6
    assert list(data["processed_emails"][0]) == ["time", "sender", "subject", "route", "status"]
    assert "body" not in data["processed_emails"][0]
    assert data["last_run"]["reason"] == "gmail_login_expired"
    assert data["runs"][0]["counts"] == {"drafts_created": 1}


def test_database_connection_is_read_only(dashboard_db):
    with dashboard._open_readonly_db(dashboard_db) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("CREATE TABLE should_not_exist (id INTEGER)")


def test_pause_resume_creates_and_deletes_flag(tmp_path):
    flag = tmp_path / "data" / "PAUSED"
    dashboard.pause_processing(flag)
    assert flag.is_file()
    assert dashboard.get_ui_status(flag, tmp_path / "run.lock") == "Paused"
    dashboard.resume_processing(flag)
    assert not flag.exists()
    assert dashboard.get_ui_status(flag, tmp_path / "run.lock") == "Idle"


def test_running_status_takes_priority_over_pause(tmp_path):
    flag, lock = tmp_path / "PAUSED", tmp_path / "run.lock"
    flag.touch()
    lock.touch()
    assert dashboard.get_ui_status(flag, lock) == "Running"


def test_timestamp_uses_local_clock_and_relative_age():
    now = datetime(2026, 10, 1, 16, 40, tzinfo=timezone.utc).astimezone()
    timestamp = (now - timedelta(minutes=3)).astimezone(timezone.utc).isoformat()
    expected_clock = (now - timedelta(minutes=3)).strftime("%I:%M %p").lstrip("0")
    assert dashboard.format_local_timestamp(timestamp, now) == f"{expected_clock} (3 min ago)"


def test_worker_countdown_uses_local_time(tmp_path):
    path = tmp_path / "worker_status.json"
    now = datetime(2026, 10, 1, 16, 40, tzinfo=timezone.utc).astimezone()
    next_run = now + timedelta(minutes=3)
    heartbeat = now.astimezone(timezone.utc).isoformat()
    path.write_text(json.dumps({
        "worker_pid": 123,
        "last_heartbeat": heartbeat,
        "next_inbox_run_at": next_run.astimezone(timezone.utc).isoformat(),
        "next_ingest_run_at": next_run.astimezone(timezone.utc).isoformat(),
    }))
    expected_clock = next_run.strftime("%I:%M %p").lstrip("0")
    assert dashboard.worker_status_message(path, now) == f"Worker: running, next automatic check in 3 min (at {expected_clock})"


@pytest.mark.parametrize("heartbeat", [None, "2026-10-01T16:20:00+00:00"])
def test_worker_missing_or_stale_heartbeat_message(tmp_path, heartbeat):
    path = tmp_path / "worker_status.json"
    now = datetime(2026, 10, 1, 16, 40, tzinfo=timezone.utc)
    if heartbeat is not None:
        path.write_text(json.dumps({
            "last_heartbeat": heartbeat,
            "next_inbox_run_at": "2026-10-01T16:45:00+00:00",
        }))
    assert dashboard.worker_status_message(path, now) == (
        "Worker not running: no automatic checks. Start it with: python -m worker"
    )


def test_locked_run_reason_is_friendly():
    assert dashboard.display_run_reason({"status": "locked", "reason": "run_lock_active"}) == (
        "Skipped: another run was already in progress"
    )
