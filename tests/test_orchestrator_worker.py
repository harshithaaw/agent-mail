import os
import sqlite3
import threading
from datetime import datetime, timezone
from logging import Logger

import app
import orchestrator
import worker
import pytest


def _setup(tmp_path, monkeypatch, *, ollama_ok=True, gmail_error=None):
    db_path = tmp_path / "processed.db"
    pause_path = tmp_path / "PAUSED"
    lock_path = tmp_path / "run.lock"
    monkeypatch.setattr(orchestrator, "DB_PATH", db_path)
    monkeypatch.setattr(orchestrator, "PAUSE_PATH", pause_path)
    monkeypatch.setattr(orchestrator, "LOCK_PATH", lock_path)
    monkeypatch.setattr(orchestrator, "check_ollama", lambda: True if ollama_ok else (_ for _ in ()).throw(ConnectionError("offline")))
    monkeypatch.setattr(orchestrator, "check_gmail", lambda: (_ for _ in ()).throw(gmail_error) if gmail_error else True)
    monkeypatch.setattr(orchestrator, "fetch_inbox_emails", lambda max_emails: [])
    return db_path, pause_path, lock_path


def _run_rows(db_path):
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT status, reason, counts_json FROM runs ORDER BY id").fetchall()


def test_pause_stops_run_and_writes_runs_row(tmp_path, monkeypatch):
    db_path, pause_path, _ = _setup(tmp_path, monkeypatch)
    pause_path.touch()
    monkeypatch.setattr(orchestrator, "process_inbox", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not process")))

    result = orchestrator.run_once()

    assert result["status"] == "paused"
    assert _run_rows(db_path)[0][0:2] == ("paused", "data_paused")


def test_active_lock_blocks_run_and_writes_runs_row(tmp_path, monkeypatch):
    db_path, _, lock_path = _setup(tmp_path, monkeypatch)
    lock_path.write_text("other-run")
    monkeypatch.setattr(orchestrator, "process_inbox", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not process")))

    result = orchestrator.run_once()

    assert result["status"] == "locked"
    assert _run_rows(db_path)[0][0:2] == ("locked", "run_lock_active")
    assert lock_path.read_text() == "other-run"


def test_second_concurrent_run_is_blocked(tmp_path, monkeypatch):
    db_path, _, lock_path = _setup(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def blocking_process(**kwargs):
        entered.set()
        assert release.wait(5), "fixture failed to release first run"
        return []

    monkeypatch.setattr(orchestrator, "process_inbox", blocking_process)
    first_results = []
    first = threading.Thread(target=lambda: first_results.append(orchestrator.run_once()))
    first.start()
    assert entered.wait(5)

    second_result = orchestrator.run_once()
    release.set()
    first.join(5)

    assert not first.is_alive()
    assert second_result["status"] == "locked"
    assert first_results[0]["status"] == "ok"
    assert not lock_path.exists()
    assert len(_run_rows(db_path)) == 2


def test_stale_lock_is_ignored_and_new_lock_is_released(tmp_path, monkeypatch):
    db_path, _, lock_path = _setup(tmp_path, monkeypatch)
    lock_path.write_text("stale-run")
    stale = datetime.now().timestamp() - (orchestrator.LOCK_STALE_MINUTES + 1) * 60
    os.utime(lock_path, (stale, stale))

    result = orchestrator.run_once(run_inbox=False)

    assert result["status"] == "ok"
    assert not lock_path.exists()
    assert _run_rows(db_path)[0][0] == "ok"


def test_preflight_failure_processes_nothing_and_records_reason(tmp_path, monkeypatch):
    db_path, _, _ = _setup(tmp_path, monkeypatch, ollama_ok=False)
    called = []
    monkeypatch.setattr(orchestrator, "process_inbox", lambda **kwargs: called.append(kwargs))

    result = orchestrator.run_once()

    assert result["status"] == "preflight_failed"
    assert result["reason"] == "ollama_preflight_failed: ConnectionError: offline"
    assert called == []
    assert _run_rows(db_path)[0][0:2] == ("preflight_failed", "ollama_preflight_failed: ConnectionError: offline")


def test_expired_gmail_preflight_uses_expired_reason(tmp_path, monkeypatch):
    db_path, _, _ = _setup(tmp_path, monkeypatch, gmail_error=RuntimeError("invalid_grant: expired token"))
    result = orchestrator.run_once()
    assert result["status"] == "preflight_failed"
    assert result["reason"] == "gmail_login_expired"
    assert _run_rows(db_path)[0][1] == "gmail_login_expired"


def test_per_run_draft_cap_leaves_remaining_messages_unrecorded(tmp_path, monkeypatch):
    db_path, _, _ = _setup(tmp_path, monkeypatch)
    cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
    emails = [{
        "message_id": f"cap-{i}", "thread_id": f"thread-{i}",
        "sender_email": "person@example.org", "subject": "Fixture", "body": "Hi",
        "received_at": "2026-09-26T15:13:40+00:00", "timestamp": "naive-preserved",
    } for i in range(8)]
    monkeypatch.setattr(orchestrator, "fetch_inbox_emails", lambda max_emails: emails[:max_emails])

    def fakeable_process(**kwargs):
        return app.process_inbox(
            **kwargs,
            security=lambda *args: {"route": "clean_full_pipeline"},
            understanding=lambda text: {"category": "Personal"},
            drafter=lambda email, understood: "draft-id",
            gate=lambda route, sender, *headers: True,
            now=cutoff,
        )

    monkeypatch.setattr(orchestrator, "process_inbox", fakeable_process)
    result = orchestrator.run_once()

    assert result["status"] == "ok"
    assert result["counts"]["fetched"] == 8
    assert result["counts"]["drafts_created"] == orchestrator.MAX_DRAFTS_PER_RUN == 5
    assert result["reason"] == "draft_cap_reached"
    assert result["remaining_unrecorded"] == 3
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM processed WHERE message_id LIKE 'cap-%'").fetchone()[0] == 5


def test_daily_draft_cap_stops_before_processing(tmp_path, monkeypatch):
    db_path, _, _ = _setup(tmp_path, monkeypatch)
    orchestrator._initialize_runs_db(db_path)
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO processed(message_id,status,processed_at) VALUES(?,?,?)",
            [(f"today-{i}", "draft_created", today) for i in range(orchestrator.MAX_DRAFTS_PER_DAY)],
        )
    called = []
    monkeypatch.setattr(orchestrator, "process_inbox", lambda **kwargs: called.append(kwargs) or [])

    result = orchestrator.run_once()

    assert result["status"] == "ok"
    assert result["reason"] == "draft_cap_reached"
    assert called == []
    assert result["counts"]["drafts_created"] == 0
    assert _run_rows(db_path)[0][0] == "ok"


def test_lock_is_released_after_processing_error_and_run_is_recorded(tmp_path, monkeypatch):
    db_path, _, lock_path = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "process_inbox", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")))

    result = orchestrator.run_once()

    assert result["status"] == "error"
    assert "RuntimeError: fixture failure" in result["reason"]
    assert "Traceback (most recent call last)" in result["reason"]
    assert not lock_path.exists()
    row = _run_rows(db_path)[0]
    assert row[0] == "error"
    assert "RuntimeError: fixture failure" in row[1]
    assert "Traceback (most recent call last)" in row[1]


def test_worker_once_invokes_orchestrator_once_without_ingest(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "configure_logging", lambda: Logger("fake"))
    monkeypatch.setattr(worker, "run_once", lambda **kwargs: calls.append(kwargs) or {"status": "ok"})

    worker.main(["--once"])

    assert calls == [{"run_inbox": True, "run_ingest": False, "trigger": "worker_once"}]


def test_worker_schedules_sent_ingestion_every_three_hours(monkeypatch, tmp_path):
    class StopLoop(Exception):
        pass

    clock = [0.0]
    calls = []
    monkeypatch.setattr(worker, "configure_logging", lambda: Logger("fake"))

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= worker.SENT_INGEST_INTERVAL_SECONDS + worker.INBOX_INTERVAL_SECONDS:
            raise StopLoop

    with pytest.raises(StopLoop):
        worker.run_forever(
            run_once_fn=lambda **kwargs: calls.append(kwargs) or {"status": "ok"},
            sleep_fn=sleep,
            monotonic_fn=lambda: clock[0],
            heartbeat_path=tmp_path / "worker_status.json",
        )

    assert calls[0] == {"run_inbox": True, "run_ingest": False, "trigger": "worker"}
    assert calls[-1] == {"run_inbox": True, "run_ingest": True, "trigger": "worker"}
    heartbeat = __import__("json").loads((tmp_path / "worker_status.json").read_text())
    assert {"worker_pid", "last_heartbeat", "next_inbox_run_at", "next_ingest_run_at"} <= set(heartbeat)


def test_worker_marks_heartbeat_stopped_on_keyboard_interrupt(monkeypatch, tmp_path):
    from logging import Logger

    path = tmp_path / "worker_status.json"
    monkeypatch.setattr(worker, "configure_logging", lambda: Logger("fake"))
    monkeypatch.setattr(worker, "WORKER_STATUS_PATH", path)
    monkeypatch.setattr(worker, "run_forever", lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    worker.main([])

    import json
    assert json.loads(path.read_text())["status"] == "stopped"
