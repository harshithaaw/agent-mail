"""Pipeline A orchestration. Fetch, security, understanding, then optional draft.

All external edges are injectable so fixtures never need Gmail or a model API.
"""
import sqlite3
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from gmail.fetch import INBOX_MAX_RESULTS


PROCESSED_DB = Path(__file__).resolve().parent / "data" / "processed.db"
SEED_PROCESSED_IDS_FILE = Path(__file__).resolve().parent / "data" / "local" / "seed_processed_ids.json"
GO_LIVE_ENV = "AGENTMAIL_GO_LIVE_AT"


def _load_seed_processed_ids(seed_path=SEED_PROCESSED_IDS_FILE):
    path = Path(seed_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return data
    except Exception:
        pass
    return []


def _initialize_processed_db(db_path, seed_path=SEED_PROCESSED_IDS_FILE):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS processed (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT,
            sender TEXT,
            subject TEXT,
            route TEXT,
            status TEXT,
            draft_id TEXT,
            processed_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(processed)")}
        if "attempts" not in columns:
            db.execute("ALTER TABLE processed ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        seed_ids = _load_seed_processed_ids(seed_path)
        if seed_ids:
            db.executemany(
                "INSERT OR IGNORE INTO processed (message_id, route, status, processed_at) VALUES (?, ?, ?, ?)",
                [(message_id, "clean_full_pipeline", "draft_created", "1970-01-01T00:00:00+00:00")
                 for message_id in seed_ids],
            )


def _is_processed(db_path, message_id):
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT status, attempts FROM processed WHERE message_id = ?", (message_id,)).fetchone()
        return row is not None and not (row[0] == "generation_failed" and row[1] < 3)


def _go_live_at(db_path, now=None):
    override = os.environ.get(GO_LIVE_ENV)
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT value FROM settings WHERE key = 'go_live_at'").fetchone()
        if override:
            value = override
            # Validate eagerly and normalize to UTC ISO format.
            value = _parse_received_at(value).isoformat()
            db.execute("INSERT INTO settings(key, value) VALUES('go_live_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (value,))
            return _parse_received_at(value)
        if row:
            return _parse_received_at(row[0])
        value = _parse_received_at(now or datetime.now(timezone.utc)).isoformat()
        db.execute("INSERT INTO settings(key, value) VALUES('go_live_at', ?)", (value,))
        return _parse_received_at(value)


def _parse_received_at(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("received timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def reply_to_email(*args, **kwargs):
    """Module-level, patchable lazy reference to the existing reply pipeline."""
    from pipelines.reply import reply_to_email as implementation
    return implementation(*args, **kwargs)


def _save_processed(db_path, email, outcome):
    with sqlite3.connect(db_path) as db:
        existing = db.execute("SELECT attempts FROM processed WHERE message_id = ?", (email.get("message_id"),)).fetchone()
        attempts = (existing[0] if existing else 0) + (1 if outcome.get("status") == "generation_failed" else 0)
        db.execute("""INSERT INTO processed
            (message_id, thread_id, sender, subject, route, status, draft_id, processed_at, attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET thread_id=excluded.thread_id,
            sender=excluded.sender, subject=excluded.subject, route=excluded.route,
            status=excluded.status, draft_id=excluded.draft_id,
            processed_at=excluded.processed_at, attempts=excluded.attempts""",
            (email.get("message_id"), email.get("thread_id"), email.get("sender_email"),
             email.get("subject"), outcome.get("route"), outcome.get("status"),
             outcome.get("draft"), datetime.now(timezone.utc).isoformat(), attempts))


def _log_outcome(outcome):
    print(
        f"[{outcome.get('message_id', 'unknown')}] "
        f"route={outcome.get('route', 'unknown')} status={outcome.get('status', 'unknown')}"
    )


def process_inbox(fetcher=None, security=None,
                  understanding=None, drafter=None, collection=None,
                  embed=None, gate=None, limit=INBOX_MAX_RESULTS, processed_db=PROCESSED_DB,
                  now=None, draft_limit=None):
    # Delay integration imports so offline fixture tests don't initialize
    # Gmail credentials, model runtimes, or persistent Chroma stores.
    if fetcher is None:
        from gmail.fetch import fetch_inbox_emails as fetcher
    if security is None:
        from agents.security.agent import run_security_agent as security
    use_production_understanding = understanding is None
    if understanding is None:
        from agents.understanding.runner import understand_email as understanding
    _initialize_processed_db(processed_db)
    go_live_at = _go_live_at(processed_db, now=now)
    emails = fetcher(max_emails=limit)
    results = []
    drafts_created = 0
    for email in emails:
        if draft_limit is not None and drafts_created >= draft_limit:
            break
        received_at = email.get("received_at") or email.get("timestamp")
        try:
            if not received_at:
                raise ValueError("timestamp is missing")
            received_at_utc = _parse_received_at(received_at)
        except (TypeError, ValueError) as exc:
            print(f"[{email.get('message_id', 'unknown')}] skipped_missing_or_invalid_timestamp: {exc}")
            continue
        # Both values are timezone-aware UTC datetimes before comparison.
        if received_at_utc < go_live_at:
            print(f"[{email.get('message_id', 'unknown')}] skipped_before_go_live_at")
            continue
        if email.get("message_id") and _is_processed(processed_db, email["message_id"]):
            _log_outcome({"message_id": email["message_id"], "route": "already_processed", "status": "already_processed"})
            continue
        text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
        try:
            screened = security(text, email.get("sender_email", ""), [])
        except Exception:
            _log_outcome({"message_id": email.get("message_id"), "route": "unknown", "status": "error"})
            raise
        route = screened["route"]
        outcome = {"message_id": email.get("message_id"), "route": route, "draft": None, "status": route}
        if gate is None:
            from agents.reply.gate import should_generate_reply as gate
        try:
            eligible = gate(route, email.get("sender_email", ""), email.get("list_unsubscribe", ""), email.get("precedence", ""))
        except Exception:
            _log_outcome({**outcome, "status": "error"})
            raise
        if route == "clean_full_pipeline" and not eligible:
            outcome["status"] = "skipped_automated_sender"
            _save_processed(processed_db, email, outcome)
            _log_outcome(outcome)
            results.append(outcome)
            continue
        if route not in {"flagged_skip_reply_generation", "low_priority_skip_downstream"}:
            if use_production_understanding:
                understood = understanding(
                    text,
                    received_at=email.get("received_at"),
                    message_id=email.get("message_id"),
                )
            else:
                understood = understanding(text)
            outcome["understanding"] = understood
            if route == "clean_full_pipeline":
                if drafter:
                    try:
                        outcome["draft"] = drafter(email, understood)
                    except Exception as exc:
                        outcome["status"] = "generation_failed"
                        outcome["generation_error"] = str(exc)
                        print(f"[{outcome['message_id']}] status=generation_failed error={exc}")
                        _save_processed(processed_db, email, outcome)
                        _log_outcome(outcome)
                        results.append(outcome)
                        continue
                    if not outcome["draft"]:
                        outcome["status"] = "generation_failed"
                        outcome["generation_error"] = "Reply Agent returned an empty draft"
                        print(f"[{outcome['message_id']}] status=generation_failed error={outcome['generation_error']}")
                        _save_processed(processed_db, email, outcome)
                        _log_outcome(outcome)
                        results.append(outcome)
                        continue
                else:
                    if embed is None:
                        from rag.retrieve import embed_text as embed
                    if collection is None:
                        from rag.retrieve import get_or_create_collection
                        active_collection = get_or_create_collection()
                    else:
                        active_collection = collection
                    try:
                        reply = reply_to_email(email, understood, active_collection, embed)
                    except Exception as exc:
                        outcome["status"] = "generation_failed"
                        outcome["generation_error"] = str(exc)
                        print(f"[{outcome['message_id']}] status=generation_failed error={exc}")
                        _save_processed(processed_db, email, outcome)
                        _log_outcome(outcome)
                        results.append(outcome)
                        continue
                    if not reply.get("draft"):
                        outcome["status"] = "generation_failed"
                        outcome["generation_error"] = "Reply Agent returned an empty draft"
                        outcome["reply_result"] = reply
                        print(f"[{outcome['message_id']}] status=generation_failed error={outcome['generation_error']} retrieved={len(reply.get('retrieved', []))}")
                        _save_processed(processed_db, email, outcome)
                        _log_outcome(outcome)
                        results.append(outcome)
                        continue
                    from gmail.draft import create_gmail_draft
                    try:
                        draft_result = create_gmail_draft(
                            {**email, "sender_email": email.get("sender_email", "")},
                            understood, route, generated_reply=reply["draft"],
                        )
                    except Exception:
                        _log_outcome({**outcome, "status": "error"})
                        raise
                    outcome["draft"] = draft_result["draft_id"]
                    outcome["draft_result"] = draft_result
                    if not outcome["draft"]:
                        outcome["status"] = "draft_failed"
                        outcome["draft_error"] = draft_result.get("error")
                    else:
                        outcome["status"] = "draft_created"
                if drafter and outcome["draft"]:
                    outcome["status"] = "draft_created"
        if outcome.get("status") == "draft_created":
            drafts_created += 1
        _save_processed(processed_db, email, outcome)
        _log_outcome(outcome)
        results.append(outcome)
    return results


if __name__ == "__main__":
    for item in process_inbox():
        print(item)
