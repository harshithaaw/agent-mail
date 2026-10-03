"""Small run coordinator for the existing AgentMail pipelines."""
from __future__ import annotations

import json
import os
import sqlite3
import traceback
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import PROCESSED_DB, _initialize_processed_db, process_inbox
from gmail.fetch import INBOX_MAX_RESULTS, fetch_inbox_emails

MAX_DRAFTS_PER_RUN = 5
MAX_DRAFTS_PER_DAY = 30
LOCK_STALE_MINUTES = 30

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = PROCESSED_DB
PAUSE_PATH = DATA_DIR / "PAUSED"
LOCK_PATH = DATA_DIR / "run.lock"

COUNT_KEYS = (
    "fetched", "already_processed", "drafts_created", "skipped_automated",
    "flagged", "low_priority", "blocked", "failed",
)


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat()


def _initialize_runs_db(db_path=DB_PATH):
    _initialize_processed_db(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            trigger TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            counts_json TEXT NOT NULL
        )""")


def _start_run(db_path, trigger, started_at):
    with sqlite3.connect(db_path) as db:
        cursor = db.execute(
            "INSERT INTO runs(started_at, trigger, status, counts_json) VALUES (?, ?, 'error', '{}')",
            (_iso(started_at), trigger),
        )
        return cursor.lastrowid


def _finish_run(db_path, run_id, finished_at, status, reason, counts):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE runs SET finished_at=?, status=?, reason=?, counts_json=? WHERE id=?",
            (_iso(finished_at), status, reason, json.dumps(counts, sort_keys=True), run_id),
        )


def check_ollama():
    request = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=3) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"ollama_http_{response.status}")
    return True


def check_gmail():
    from gmail.auth import get_gmail_service
    get_gmail_service()
    return True


def _gmail_failure_reason(exc):
    message = f"{type(exc).__name__}: {exc}".lower()
    expired_markers = (
        "invalid_grant", "expired", "revoked", "invalid token", "unauthorized",
        "unauthenticated", "401", "invalid authentication credentials",
    )
    if any(marker in message for marker in expired_markers):
        return "gmail_login_expired"
    return f"gmail_preflight_failed: {type(exc).__name__}: {exc}"


def _acquire_lock(lock_path, now):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age = now.timestamp() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= LOCK_STALE_MINUTES * 60:
                return None
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            continue
        else:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(token)
            return token
    return None


def _release_lock(lock_path, token):
    try:
        if lock_path.read_text() == token:
            lock_path.unlink()
    except FileNotFoundError:
        pass


def _drafts_today(db_path, now):
    day_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT COUNT(*) FROM processed WHERE status='draft_created' AND processed_at >= ? AND processed_at < ?",
            (_iso(day_start), _iso(day_end)),
        ).fetchone()
    return row[0]


def _existing_processed_ids(db_path):
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT message_id, status, attempts FROM processed").fetchall()
    return {
        message_id for message_id, status, attempts in rows
        if status != "generation_failed" or attempts >= 3
    }


def _ids_in_processed(db_path):
    with sqlite3.connect(db_path) as db:
        return {row[0] for row in db.execute("SELECT message_id FROM processed")}


def _count_results(emails, outcomes, already_processed):
    counts = {key: 0 for key in COUNT_KEYS}
    counts["fetched"] = len(emails)
    counts["already_processed"] = already_processed
    for outcome in outcomes:
        route = outcome.get("route", "")
        status = outcome.get("status", "")
        if status == "draft_created":
            counts["drafts_created"] += 1
        if status == "skipped_automated_sender":
            counts["skipped_automated"] += 1
            counts["blocked"] += 1
        if route.startswith("flagged_") or status.startswith("flagged_"):
            counts["flagged"] += 1
        if "low_priority" in route or "low_priority" in status:
            counts["low_priority"] += 1
        if status in {"generation_failed", "draft_failed"}:
            counts["failed"] += 1
        if status == "flagged_skip_reply_generation":
            counts["blocked"] += 1
    return counts


def _run_sent_ingestion():
    from scripts.ingest_sent_batch import run
    return run()


def run_once(run_inbox=True, run_ingest=False, trigger="manual") -> dict:
    """Run selected existing pipelines after pause, lock, and service checks."""
    started_at = _now()
    _initialize_runs_db(DB_PATH)
    run_id = _start_run(DB_PATH, trigger, started_at)
    counts = {key: 0 for key in COUNT_KEYS}
    status = "error"
    reason = None
    lock_token = None
    remaining_unrecorded = 0

    try:
        if PAUSE_PATH.exists():
            status, reason = "paused", "data_paused"
            return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts)

        lock_token = _acquire_lock(LOCK_PATH, started_at)
        if lock_token is None:
            status, reason = "locked", "run_lock_active"
            return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts)

        try:
            check_ollama()
        except Exception as exc:
            status, reason = "preflight_failed", f"ollama_preflight_failed: {type(exc).__name__}: {exc}"
            return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts)
        try:
            check_gmail()
        except Exception as exc:
            status, reason = "preflight_failed", _gmail_failure_reason(exc)
            return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts)

        if run_inbox:
            now = _now()
            daily_remaining = max(0, MAX_DRAFTS_PER_DAY - _drafts_today(DB_PATH, now))
            draft_limit = min(MAX_DRAFTS_PER_RUN, daily_remaining)
            if draft_limit == 0:
                reason = "draft_cap_reached"
            else:
                previous_ids = _existing_processed_ids(DB_PATH)
                fetched_emails = []

                def recording_fetcher(max_emails=INBOX_MAX_RESULTS):
                    emails = fetch_inbox_emails(max_emails=max_emails)
                    fetched_emails.extend(emails)
                    return emails

                outcomes = process_inbox(
                    fetcher=recording_fetcher,
                    processed_db=DB_PATH,
                    draft_limit=draft_limit,
                )
                already_processed = sum(
                    1 for email in fetched_emails
                    if email.get("message_id") in previous_ids
                )
                counts = _count_results(fetched_emails, outcomes, already_processed)
                if counts["drafts_created"] >= draft_limit:
                    reason = "draft_cap_reached"
                    recorded_ids = _ids_in_processed(DB_PATH)
                    remaining_unrecorded = sum(
                        1 for email in fetched_emails
                        if email.get("message_id") not in recorded_ids
                    )

        if run_ingest:
            _run_sent_ingestion()

        status = "ok"
        return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts, remaining_unrecorded)
    except Exception as exc:
        status, reason = "error", traceback.format_exc()
        return _recorded_result(DB_PATH, run_id, status, reason, started_at, counts, remaining_unrecorded)
    finally:
        if lock_token is not None:
            _release_lock(LOCK_PATH, lock_token)


def _recorded_result(db_path, run_id, status, reason, started_at, counts, remaining_unrecorded=0):
    finished_at = _now()
    _finish_run(db_path, run_id, finished_at, status, reason, counts)
    return {
        **counts,
        "run_id": run_id,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "status": status,
        "reason": reason,
        "counts": counts,
        "remaining_unrecorded": remaining_unrecorded,
    }
