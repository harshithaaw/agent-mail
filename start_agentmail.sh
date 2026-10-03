#!/bin/bash

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${AGENTMAIL_PYTHON_BIN:-python}"
WORKER_STATUS_FILE="${AGENTMAIL_WORKER_STATUS_FILE:-$SCRIPT_DIR/data/worker_status.json}"
WORKER_LOG_FILE="${AGENTMAIL_WORKER_LOG_FILE:-$SCRIPT_DIR/logs/agentmail_worker.out}"
STALE_AFTER_SECONDS=$((2 * 5 * 60))

if ! "$PYTHON_BIN" -c 'import chromadb' >/dev/null 2>&1; then
    PATH="/opt/anaconda3/bin:$PATH"
    export PATH
    if [ "$PYTHON_BIN" = "python" ]; then
        PYTHON_BIN="$(command -v python)"
    fi
fi

if [ -e "$WORKER_STATUS_FILE" ]; then
    if "$PYTHON_BIN" - "$WORKER_STATUS_FILE" "$STALE_AFTER_SECONDS" <<'PY'
import json
import sys
from datetime import datetime, timezone

try:
    with open(sys.argv[1], encoding="utf-8") as status_file:
        status = json.load(status_file)
    heartbeat = datetime.fromisoformat(status["last_heartbeat"].replace("Z", "+00:00"))
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - heartbeat.astimezone(timezone.utc)).total_seconds()
    active = status.get("status") != "stopped" and 0 <= age <= int(sys.argv[2])
except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
    active = False
sys.exit(0 if active else 1)
PY
    then
        echo "AgentMail worker already appears to be running; not starting another one."
        exit 1
    fi
elif pgrep -f '[p]ython -m worker' >/dev/null 2>&1; then
    echo "AgentMail worker process already exists; not starting another one."
    exit 1
fi

mkdir -p "$(dirname -- "$WORKER_LOG_FILE")"

caffeinate_pid=""
streamlit_pid=""

stop_worker() {
    if [ -z "$caffeinate_pid" ] || ! kill -0 "$caffeinate_pid" 2>/dev/null; then
        return
    fi

    child_pids="$(pgrep -P "$caffeinate_pid" 2>/dev/null || true)"
    for child_pid in $child_pids; do
        kill -INT "$child_pid" 2>/dev/null || true
    done

    attempts=0
    while [ "$attempts" -lt 10 ]; do
        children_alive=0
        for child_pid in $child_pids; do
            if kill -0 "$child_pid" 2>/dev/null; then
                children_alive=1
            fi
        done
        [ "$children_alive" -eq 0 ] && break
        sleep 0.1
        attempts=$((attempts + 1))
    done
    for child_pid in $child_pids; do
        kill -TERM "$child_pid" 2>/dev/null || true
    done

    attempts=0
    while kill -0 "$caffeinate_pid" 2>/dev/null && [ "$attempts" -lt 30 ]; do
        sleep 0.1
        attempts=$((attempts + 1))
    done
    kill -TERM "$caffeinate_pid" 2>/dev/null || true
    wait "$caffeinate_pid" 2>/dev/null || true
    caffeinate_pid=""
}

cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ -n "$streamlit_pid" ] && kill -0 "$streamlit_pid" 2>/dev/null; then
        kill -TERM "$streamlit_pid" 2>/dev/null || true
        wait "$streamlit_pid" 2>/dev/null || true
    fi
    stop_worker
    return "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

caffeinate -i "$PYTHON_BIN" -m worker >>"$WORKER_LOG_FILE" 2>&1 &
caffeinate_pid=$!

echo "AgentMail dashboard: http://localhost:8501 — stop with Ctrl+C."
streamlit run dashboard.py &
streamlit_pid=$!
wait "$streamlit_pid"
