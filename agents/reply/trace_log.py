"""Privacy-conscious JSONL logging for reply retrieval traces."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from email.utils import getaddresses
from pathlib import Path

from agents.reply.context import DISTANCE_THRESHOLD

DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[2] / "logs" / "reply_trace.jsonl"
_ADDRESS_RE = re.compile(r"[\w.+%-]+@[\w.-]+", re.IGNORECASE)


def _normalize_address(raw_address):
    addresses = getaddresses([raw_address or ""])
    return addresses[0][1].strip().lower() if addresses else ""


def _without_addresses(value):
    if isinstance(value, str):
        return _ADDRESS_RE.sub("[address]", value)
    if isinstance(value, list):
        return [_without_addresses(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_addresses(item)
            for key, item in value.items()
            if key not in {"metadata", "sender_email", "recipient", "recipient_norm", "address"}
        }
    return value


def _safe_validation(validation):
    """Keep validation's scalar result fields, excluding embedded example data."""
    if not isinstance(validation, dict):
        return validation
    return {
        key: _without_addresses(value)
        for key, value in validation.items()
        if key != "usable_examples"
    }


def write_reply_trace(email, reply_result, path=DEFAULT_TRACE_PATH):
    """Append a draft's retrieval summary; logging failures never escape."""
    try:
        sender = _normalize_address((email or {}).get("sender_email"))
        examples = (reply_result or {}).get("retrieved_examples")
        if examples is None:
            examples = (reply_result or {}).get("retrieved", [])
        trace = {
            "time": datetime.now(timezone.utc).isoformat(),
            "message_id": _without_addresses((email or {}).get("message_id")),
            "switch": "on" if os.getenv("AGENTMAIL_PERSON_AWARE") == "1" else "off",
            "stopped_reason": _without_addresses((reply_result or {}).get("stopped_reason")),
            "validation": _safe_validation((reply_result or {}).get("validation")),
            "examples": [],
        }
        for rank, example in enumerate(examples or [], start=1):
            example = example if isinstance(example, dict) else {}
            metadata = example.get("metadata") or {}
            distance = example.get("distance")
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                distance = None
            document = example.get("document") or ""
            subject = document.splitlines()[0][:80] if isinstance(document, str) and document else ""
            trace["examples"].append({
                "rank": rank,
                "id": _without_addresses(example.get("id")),
                "subject": _without_addresses(subject),
                "distance": round(distance, 4) if distance is not None else None,
                "sent_to_sender": bool(sender and _normalize_address(metadata.get("recipient_norm")) == sender),
                "above_threshold": bool(distance is not None and distance >= DISTANCE_THRESHOLD),
            })
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # Trace logging must never interrupt drafting.
        return
