import sqlite3
import json
from datetime import datetime, timezone

import app


def test_processing_same_fixture_twice_creates_only_one_draft(tmp_path):
    email = {
        "message_id": "fixture_message_once",
        "thread_id": "fixture_thread",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "sender_email": "person@example.org",
        "subject": "Fixture subject",
        "body": "Fixture body",
    }
    created = []
    dependencies = {
        "fetcher": lambda max_emails: [email],
        "security": lambda *args: {"route": "clean_full_pipeline"},
        "gate": lambda route, sender, *headers: True,
        "understanding": lambda text: {"category": "Personal"},
        "drafter": lambda current, understood: created.append(current["message_id"]) or "fixture-draft-id",
        "processed_db": tmp_path / "processed.db",
        "now": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }

    first = app.process_inbox(**dependencies)
    second = app.process_inbox(**dependencies)

    assert first[0]["draft"] == "fixture-draft-id"
    assert second == []
    assert created == [email["message_id"]]
    with sqlite3.connect(dependencies["processed_db"]) as db:
        row = db.execute(
            "SELECT message_id, thread_id, sender, subject, route, status, draft_id "
            "FROM processed WHERE message_id = ?",
            (email["message_id"],),
        ).fetchone()
    assert row == (
        email["message_id"], email["thread_id"], email["sender_email"],
        email["subject"], "clean_full_pipeline", "draft_created", "fixture-draft-id",
    )


def test_initialize_processed_db_seeding(tmp_path):
    db_no_seed = tmp_path / "no_seed.db"
    app._initialize_processed_db(db_no_seed, seed_path=tmp_path / "missing_seed.json")
    with sqlite3.connect(db_no_seed) as db:
        rows_no_seed = db.execute("SELECT message_id, route, status, processed_at FROM processed").fetchall()
    assert rows_no_seed == []

    seed_file = tmp_path / "seed.json"
    seed_file.write_text(json.dumps(["test_id_1", "test_id_2"]), encoding="utf-8")
    db_with_seed = tmp_path / "with_seed.db"
    app._initialize_processed_db(db_with_seed, seed_path=seed_file)
    with sqlite3.connect(db_with_seed) as db:
        rows_with_seed = db.execute("SELECT message_id, route, status, processed_at FROM processed ORDER BY message_id").fetchall()
    assert rows_with_seed == [
        ("test_id_1", "clean_full_pipeline", "draft_created", "1970-01-01T00:00:00+00:00"),
        ("test_id_2", "clean_full_pipeline", "draft_created", "1970-01-01T00:00:00+00:00"),
    ]

