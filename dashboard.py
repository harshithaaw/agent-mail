"""Read-only Streamlit viewer for local AgentMail state."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "processed.db"
PAUSE_PATH = ROOT / "data" / "PAUSED"
LOCK_PATH = ROOT / "data" / "run.lock"
WORKER_STATUS_PATH = ROOT / "data" / "worker_status.json"
CHROMA_PATH = ROOT / "chroma_db" / "chroma.sqlite3"
REPLY_TRACE_PATH = ROOT / "logs" / "reply_trace.jsonl"


def _open_readonly_db(db_path):
    """Open an existing SQLite database without permitting writes or creation."""
    absolute = Path(db_path).resolve()
    return sqlite3.connect(f"file:{quote(str(absolute))}?mode=ro", uri=True)


def _day_bounds(now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    start = datetime.combine(now.astimezone(timezone.utc).date(), time.min, timezone.utc)
    return start.isoformat(), (start + timedelta(days=1)).isoformat()


def load_dashboard_data(db_path=DB_PATH, now=None):
    """Return dashboard-safe rows; never select email bodies or mutate the DB."""
    counters = {"drafts_created": 0, "skipped": 0, "flagged": 0, "failed": 0}
    processed, runs, last_run = [], [], None
    try:
        with closing(_open_readonly_db(db_path)) as db:
            start, end = _day_bounds(now)
            for status, route, count in db.execute(
                "SELECT status, route, COUNT(*) FROM processed "
                "WHERE processed_at >= ? AND processed_at < ? GROUP BY status, route",
                (start, end),
            ):
                if status == "draft_created":
                    counters["drafts_created"] += count
                if status in {"skipped_automated_sender", "low_priority_skip_downstream"}:
                    counters["skipped"] += count
                if str(route or "").startswith("flagged_") or str(status or "").startswith("flagged_"):
                    counters["flagged"] += count
                if status in {"generation_failed", "draft_failed"}:
                    counters["failed"] += count

            processed = [
                {"time": row[0], "sender": row[1], "subject": row[2], "route": row[3], "status": row[4]}
                for row in db.execute(
                    "SELECT processed_at, sender, subject, route, status FROM processed "
                    "ORDER BY processed_at DESC LIMIT 50"
                )
            ]
            run_rows = db.execute(
                "SELECT id, started_at, finished_at, trigger, status, reason, counts_json "
                "FROM runs ORDER BY id DESC LIMIT 10"
            ).fetchall()
            runs = [
                {"id": row[0], "started_at": row[1], "finished_at": row[2], "trigger": row[3],
                 "status": row[4], "reason": row[5], "counts": _parse_counts(row[6])}
                for row in run_rows
            ]
            last_run = runs[0] if runs else None
    except (sqlite3.Error, OSError):
        pass
    return {"counters": counters, "processed_emails": processed, "runs": runs, "last_run": last_run}


def _parse_counts(value):
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def load_reply_trace(path=REPLY_TRACE_PATH, limit=20):
    """Read recent traces, grouped by incoming message, newest first."""
    traces = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict):
                    traces.append(record)
    except OSError:
        return []

    emails = []
    for record in traces:
        examples = record.get("examples")
        if not isinstance(examples, list):
            continue
        switch = "on" if record.get("switch") == "on" else "off"
        rows = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            rows.append({
                "rank": example.get("rank"),
                "past-reply subject": example.get("subject", ""),
                "distance": example.get("distance"),
                "sent to this sender": "yes" if example.get("sent_to_sender") else "no",
            })
        emails.append({
            "time": record.get("time", ""),
            "email id": record.get("message_id", ""),
            "switch": switch,
            "examples used": len(rows),
            "threshold label": "dropped from the prompt" if switch == "on" else "weak, still used",
            "threshold count": sum(bool(ex.get("above_threshold")) for ex in examples if isinstance(ex, dict)),
            "sent count": sum(bool(ex.get("sent_to_sender")) for ex in examples if isinstance(ex, dict)),
            "examples": rows,
        })
    # A message can have only one trace record; reverse file order is newest first.
    return list(reversed(emails))[:max(0, limit)]


def _parse_datetime(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return parsed.astimezone() if parsed.tzinfo is not None else parsed.astimezone()


def _relative_time(value, now=None):
    parsed = _parse_datetime(value)
    if parsed is None:
        return ""
    now_local = (now or datetime.now().astimezone()).astimezone()
    seconds = (now_local - parsed).total_seconds()
    if abs(seconds) < 60:
        return "just now" if seconds >= 0 else "in less than a min"
    minutes = int(abs(seconds) // 60)
    if minutes < 60:
        amount, unit = minutes, "min"
    elif minutes < 24 * 60:
        amount, unit = minutes // 60, "hr"
    else:
        amount, unit = minutes // (24 * 60), "day"
    if amount != 1 and unit == "day":
        unit += "s"
    return f"{amount} {unit} {'ago' if seconds >= 0 else 'from now'}"


def format_local_timestamp(value, now=None):
    parsed = _parse_datetime(value)
    if parsed is None:
        return value or ""
    clock = parsed.strftime("%I:%M %p").lstrip("0")
    relative = _relative_time(parsed, now)
    return f"{clock} ({relative})" if relative else clock


def display_run_reason(run):
    if run and run.get("status") == "locked" and run.get("reason") == "run_lock_active":
        return "Skipped: another run was already in progress"
    return (run or {}).get("reason") or "-"


def worker_status_message(status_path=WORKER_STATUS_PATH, now=None,
                          inbox_interval_seconds=5 * 60):
    fallback = "Worker not running: no automatic checks. Start it with: python -m worker"
    try:
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        heartbeat = _parse_datetime(status.get("last_heartbeat"))
        if heartbeat is None or status.get("status") == "stopped":
            return fallback
        current = (now or datetime.now().astimezone()).astimezone()
        age = (current - heartbeat).total_seconds()
        if age > 2 * inbox_interval_seconds:
            return fallback
        next_run = _parse_datetime(status.get("next_inbox_run_at"))
        if next_run is None:
            return fallback
        seconds = max(0, (next_run - current).total_seconds())
        if seconds < 60:
            countdown = "less than 1 min" if seconds else "now"
        else:
            minutes = int((seconds + 59) // 60)
            countdown = f"{minutes} min"
        next_clock = next_run.strftime("%I:%M %p").lstrip("0")
        return f"Worker: running, next automatic check in {countdown} (at {next_clock})"
    except (OSError, TypeError, json.JSONDecodeError):
        return fallback


def _format_dashboard_rows(data, now=None):
    emails = [
        {**row, "time": format_local_timestamp(row.get("time"), now)}
        for row in data["processed_emails"]
    ]
    runs = []
    for row in data["runs"]:
        runs.append({
            **row,
            "started_at": format_local_timestamp(row.get("started_at"), now),
            "finished_at": format_local_timestamp(row.get("finished_at"), now),
            "reason": display_run_reason(row),
        })
    return emails, runs


def _processed_display_rows(rows):
    status_labels = {
        "draft_created": "✅ Draft",
        "skipped_automated_sender": "⏭️ Skipped",
        "low_priority_skip_downstream": "⏭️ Skipped",
        "generation_failed": "❌ Failed",
        "draft_failed": "❌ Failed",
    }
    return [
        {"Time": row["time"], "From": row["sender"], "Subject": row["subject"],
         "Route": row["route"], "Status": status_labels.get(row["status"], row["status"])}
        for row in rows
    ]


def _reply_example_display_rows(rows):
    return [
        {"Rank": row["rank"], "Past reply": row["past-reply subject"],
         "Distance": row["distance"],
         "Sent to this sender": "✓" if row["sent to this sender"] == "yes" else "—"}
        for row in rows
    ]


def load_chroma_count(chroma_path=CHROMA_PATH, collection_name="email_memory"):
    """Count existing collection embeddings through a read-only SQLite connection."""
    if not Path(chroma_path).exists():
        return None
    try:
        with closing(_open_readonly_db(chroma_path)) as db:
            return db.execute(
                "SELECT COUNT(*) FROM embeddings e "
                "JOIN segments s ON s.id=e.segment_id "
                "JOIN collections c ON c.id=s.collection WHERE c.name=?",
                (collection_name,),
            ).fetchone()[0]
    except sqlite3.Error:
        return None


def get_ui_status(paused_path=PAUSE_PATH, lock_path=LOCK_PATH):
    if Path(lock_path).exists():
        return "Running"
    if Path(paused_path).exists():
        return "Paused"
    return "Idle"


def pause_processing(paused_path=PAUSE_PATH):
    flag = Path(paused_path)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch(exist_ok=True)


def resume_processing(paused_path=PAUSE_PATH):
    Path(paused_path).unlink(missing_ok=True)


def launch_worker(root=ROOT):
    return subprocess.Popen([sys.executable, "-m", "worker", "--once"], cwd=str(root))


def main():
    import streamlit as st

    st.set_page_config(page_title="AgentMail", page_icon="✉️", layout="wide")
    st.title("AgentMail")

    @st.fragment(run_every="30s")
    def dashboard():
        status = get_ui_status()
        st.caption(f"Status: {status}")
        worker_message = worker_status_message()
        if worker_message.startswith("Worker not running:"):
            st.warning(worker_message, icon="⚠️")
        else:
            st.caption(worker_message)
        st.caption("Run once now or pause automatic checks.")
        left, right = st.columns(2)
        if left.button("Run now", disabled=LOCK_PATH.exists(), use_container_width=True):
            launch_worker()
            st.rerun()
        if PAUSE_PATH.exists():
            if right.button("Resume", use_container_width=True):
                resume_processing()
                st.rerun()
        elif right.button("Pause", use_container_width=True):
            pause_processing()
            st.rerun()
        st.markdown("[Open Gmail drafts](https://mail.google.com/mail/#drafts)")

        data = load_dashboard_data()
        run = data["last_run"]
        st.divider()
        with st.container(border=True):
            st.caption("Last run")
            if run:
                run_time = run["finished_at"] or run["started_at"]
                reason = display_run_reason(run)
                st.write(
                    f"{format_local_timestamp(run_time)} · {run['status']} · "
                    f"{str(reason)[:100]}"
                )
                if "gmail_login_expired" in str(run["reason"] or ""):
                    st.error("Gmail login expired, sign in again")
            else:
                st.write("No runs recorded")

        st.subheader("Today's counters")
        cols = st.columns(4)
        labels = (("Drafts created", "drafts_created"), ("Skipped", "skipped"), ("Flagged", "flagged"), ("Failed", "failed"))
        for col, (label, key) in zip(cols, labels):
            col.metric(label, data["counters"][key])
        chroma_count = load_chroma_count()
        st.caption(f"Chroma records: {chroma_count if chroma_count is not None else 'unavailable'}")

        st.divider()
        st.subheader("Last 50 processed emails")
        st.caption("Most recently processed messages")
        processed, runs = _format_dashboard_rows(data)
        processed_display = _processed_display_rows(processed)
        st.dataframe(processed_display, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Reply retrieval (last 10 emails)")
        reply_trace = load_reply_trace(limit=10)
        if reply_trace:
            for email in reply_trace:
                timestamp = format_local_timestamp(email["time"]) if email["time"] else ""
                st.write(
                    f"{timestamp} · switch {email['switch']} · {email['examples used']} used · "
                    f"{email['threshold count']} {email['threshold label']} · "
                    f"{email['sent count']} to sender"
                )
                st.caption(f"Internal email id: {email['email id']}")
                with st.expander(f"Examples for {email['email id']}"):
                    examples = _reply_example_display_rows(email["examples"])
                    st.dataframe(
                        examples, use_container_width=True, hide_index=True,
                        column_config={
                            "Distance": st.column_config.NumberColumn(
                                "Distance (lower = more similar)", format="%.3f"
                            )
                        },
                    )
        else:
            st.write("No reply traces yet")
        with st.expander("Last 10 runs"):
            st.dataframe(runs, use_container_width=True, hide_index=True)

    dashboard()


if __name__ == "__main__":
    main()
