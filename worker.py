"""Timed local worker that delegates every run to orchestrator.run_once."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from orchestrator import run_once

INBOX_INTERVAL_SECONDS = 5 * 60
SENT_INGEST_INTERVAL_SECONDS = 3 * 60 * 60
LOG_PATH = Path(__file__).resolve().parent / "logs" / "agentmail.log"
WORKER_STATUS_PATH = Path(__file__).resolve().parent / "data" / "worker_status.json"


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(current, RotatingFileHandler) and current.baseFilename == handler.baseFilename
               for current in root.handlers):
        root.addHandler(handler)
    return logging.getLogger("agentmail.worker")


def _write_worker_status(path, next_inbox, next_ingest, *, stopped=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "worker_pid": os.getpid(),
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "next_inbox_run_at": next_inbox.isoformat(),
        "next_ingest_run_at": next_ingest.isoformat(),
    }
    if stopped:
        status["status"] = "stopped"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_forever(run_once_fn=run_once, sleep_fn=time.sleep, monotonic_fn=time.monotonic, logger=None,
                heartbeat_path=WORKER_STATUS_PATH):
    logger = logger or configure_logging()
    now = monotonic_fn()
    next_inbox = now
    next_ingest = now + SENT_INGEST_INTERVAL_SECONDS
    wall_now = datetime.now(timezone.utc)
    next_inbox_at = wall_now
    next_ingest_at = wall_now + timedelta(seconds=SENT_INGEST_INTERVAL_SECONDS)
    _write_worker_status(heartbeat_path, next_inbox_at, next_ingest_at)
    while True:
        now = monotonic_fn()
        if now >= next_inbox:
            run_ingest = now >= next_ingest
            result = run_once_fn(run_inbox=True, run_ingest=run_ingest, trigger="worker")
            logger.info("run result=%s", result)
            next_inbox = now + INBOX_INTERVAL_SECONDS
            if run_ingest:
                next_ingest = now + SENT_INGEST_INTERVAL_SECONDS
        wall_now = datetime.now(timezone.utc)
        monotonic_now = monotonic_fn()
        next_inbox_at = wall_now + timedelta(seconds=max(0, next_inbox - monotonic_now))
        next_ingest_at = wall_now + timedelta(seconds=max(0, next_ingest - monotonic_now))
        _write_worker_status(heartbeat_path, next_inbox_at, next_ingest_at)
        sleep_for = max(0, min(next_inbox, next_ingest) - monotonic_fn())
        sleep_fn(sleep_for)


def main(argv=None):
    logger = configure_logging()
    parser = argparse.ArgumentParser(description="Run the local AgentMail worker")
    parser.add_argument("--once", action="store_true", help="run inbox processing once and exit")
    args = parser.parse_args(argv)
    if args.once:
        result = run_once(run_inbox=True, run_ingest=False, trigger="worker_once")
        logger.info("run result=%s", result)
        counts = result.get("counts", {})
        print(f"Run status={result.get('status', 'unknown')} reason={result.get('reason') or '-'} counts={counts}")
        return result
    try:
        run_forever(logger=logger)
    except KeyboardInterrupt:
        now = datetime.now(timezone.utc)
        _write_worker_status(
            WORKER_STATUS_PATH,
            now + timedelta(seconds=INBOX_INTERVAL_SECONDS),
            now + timedelta(seconds=SENT_INGEST_INTERVAL_SECONDS),
            stopped=True,
        )
        logger.info("worker stopped by Ctrl+C")


if __name__ == "__main__":
    main()
