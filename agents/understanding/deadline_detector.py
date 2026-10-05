"""Deadline detection for an incoming email, anchored to the time it was received.

Design (v2):
  * Standard library only. No dateparser: every date form we accept is matched by
    one regex and resolved with plain calendar math, so there is no per-sentence
    n-gram search, no language detection and no import cost.
  * No model ever does date arithmetic. Naive/missing/ambiguous dates are skipped
    and logged, never guessed.
  * One small lexicon (_ACTIONS) drives triggers, `kind` and `action`; the
    old per-context regex pile is gone.

Public API (unchanged):
    detect_b(text, received_at) -> [{"date": "YYYY-MM-DD", "kind": str, "action": str}, ...]
Extra, for swapping into the Understanding Agent (NOT wired in):
    detect_deadline_v2(text, received_at) -> {"has_deadline", "deadline_date", "raw_phrase"}
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30), "IST")
_TZ = {
    "UTC": timezone.utc, "GMT": timezone.utc, "IST": IST,
    "EST": timezone(timedelta(hours=-5)), "EDT": timezone(timedelta(hours=-4)),
    "PST": timezone(timedelta(hours=-8)), "PDT": timezone(timedelta(hours=-7)),
}

NEAR = 90                # max characters between a date and the words that govern it
BEFORE_MAX_GAP = 12      # `before` must directly govern its date, not a later context date
OVERDUE_GRACE_DAYS = 60  # a year-less "5 September" up to this far in the past is still reported
EOD_HOUR = 17

# ----------------------------------------------------------------------------
# Calendar vocabulary (built from the calendar module, not typed by hand)
# ----------------------------------------------------------------------------
_MONTHS = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9
_WEEKDAYS = [n.lower() for n in calendar.day_name]
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY = "|".join(_WEEKDAYS)

# One regex for every date form. Each alternative has its own group names.
_DATE_RE = re.compile(
    rf"""
    (?<![\w/-])(?P<iso>\d{{4}})-(?P<iso_m>\d{{1,2}})-(?P<iso_d>\d{{1,2}})(?!\d)
  | (?<![\w/-])(?P<n1>\d{{1,2}})(?P<sep>[/-])(?P<n2>\d{{1,2}})(?P=sep)(?P<n3>\d{{4}}|\d{{2}})(?!\d)
  | \b(?:(?:{_DAY}),?\s+)?(?P<dm_d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<dm_m>{_MON})\b\.?
        (?:,?\s+(?P<dm_y>\d{{4}})(?!\d))?
  | \b(?P<md_m>{_MON})\b\.?\s*(?P<md_d>\d{{1,2}})(?:st|nd|rd|th)?(?!\d)
        (?:,?\s+(?P<md_y>\d{{4}})(?!\d))?
  | \b(?:in|within)\s+(?P<rel_n>\d{{1,3}})\s+(?P<rel_u>business\s+days?|hours?|days?|weeks?)\b
  | \b(?P<word>day\s+after\s+tomorrow|tomorrow|today|tonight)\b
  | \b(?:(?P<wd_mod>next|this|coming)\s+)?(?P<wd>{_DAY})\b
  | \b(?P<eod>EOD|COB|end\s+of\s+(?:the\s+)?day|close\s+of\s+business)\b
  | \b(?P<eow>EOW|end\s+of\s+(?:the\s+)?week)\b
  | \b(?P<eom>end\s+of\s+(?:the\s+month|(?P<eom_m>{_MON})\b))
    """,
    re.I | re.X,
)

# Optional clock time (and zone) right after a date: "... 9 October by 5 pm EST".
_TIME_RE = re.compile(
    r"""[\s,]*(?:(?:at|by|@)\s*)?
        (?:(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?\s*(?P<ap>[ap])\.?m\b\.?
          |(?P<h24>[01]?\d|2[0-3]):(?P<m24>\d{2})(?!\d))
        (?:\s*(?P<tz>EST|EDT|PST|PDT|UTC|GMT|IST)\b)?""",
    re.I | re.X,
)

# ----------------------------------------------------------------------------
# Words that make a date a deadline, and what they mean
# ----------------------------------------------------------------------------
# kind -> action-verb patterns. This single table gives triggers, `kind`, `action`.
_ACTIONS = [
    ("response", r"confirm\w*|repl(?:y|ies|ied|ying)|respon(?:d|ds|ded|ding|se)|rsvp\w*|approv\w*"
                 r"|acknowledg\w*|registration|accept(?:s|ed|ing)?"),
    ("submission", r"submit\w*|submission|upload\w*|send(?:s|ing)?|sent|complet(?:e|es|ed|ing|ion)"
                   r"|finali[sz]\w*|fill\w*|fil(?:e|es|ed|ing)|return(?:s|ed|ing)?|provid\w*"
                   r"|review\w*|share\w*|forward\w*|sign(?:s|ed|ing)?|regist(?:er|ers|ered|ering)"),
    ("payment", r"pay(?:s|ing|ed|ment)?|paid|settl\w*|remit\w*"),
]
_ACTION_RES = [(kind, re.compile(pat, re.I)) for kind, pat in _ACTIONS]
_ACTION_RE = re.compile(r"\b(?:" + "|".join(f"(?:{p})" for _, p in _ACTIONS) + r")\b", re.I)

_CUE = (r"no\s+later\s+than|on\s+or\s+before|last\s+day|closing\s+date|deadline|due|cut-?off"
        r"|by|before|until|till|through|within|expir\w*|clos(?:es|ing|ed)|ends?|valid")
_CUE_RE = re.compile(rf"\b(?:{_CUE})\b", re.I)
_EXPIRY_CUE_RE = re.compile(r"^(?:last\s+day|closing\s+date|expir\w*|clos\w*|ends?|valid)$", re.I)
_OPEN_RE = re.compile(r"\b(?:remains?\s+open|open|valid)\b", re.I)

# An action verb only counts as a trigger inside a request ("Please send ..."), so
# "Thanks for sending the files. See you on 12 Oct" is not a deadline.
_REQUEST_RE = re.compile(
    r"\b(?:please|kindly|must|should|need(?:s|ed)?\s+to|have\s+to|has\s+to|required|requested|"
    r"ensure|remember|reminder|let\s+me\s+know|(?:can|could|would|will)\s+you|urgent|asap|"
    r"action\s+required)\b",
    re.I,
)

_PAYMENT_NOUN = re.compile(
    r"\b(?:payments?|invoices?|fees?|installments?|tuition|retainer|dues|renewal\s+fee)\b", re.I)
_EVENT_RE = re.compile(
    r"\b(?:meeting|interview|call|event|webinar|session|launch|rehears\w+|presentation|"
    r"conference|kickoff|demo|workshop|appointment|ceremony)\b", re.I)
_SALE_RE = re.compile(r"\bsale\b(?:\W+\w+){0,4}\W+ends?\b", re.I)
_PAST_RE = re.compile(r"\b(?:was|were|already|completed|filed|shipped|signed|met)\b", re.I)
_PASSIVE_REQUEST_RE = re.compile(r"\b(?:must|should|needs?\s+to|has\s+to|will|can|is|are|be)\s*$", re.I)
_PROMISE_RE = re.compile(
    r"\b(?:I(?:'ll|\s+(?:will|shall|can|may))|and\s+will)\s+(?:" + "|".join(f"(?:{p})" for _, p in _ACTIONS) + r")\b",
    re.I)

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|[\n;]+")
_QUOTE_HEADER_RE = re.compile(r"^On\s+.+\s+wrote:\s*$", re.I)
_STOP = {
    "the", "a", "an", "is", "was", "are", "were", "be", "to", "of", "for", "on", "at", "in", "and",
    "your", "our", "this", "that", "please", "kindly", "it", "we", "you", "i", "with", "as", "date",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _received_at_ist(received_at: Any) -> datetime | None:
    if isinstance(received_at, datetime):
        value = received_at
    elif isinstance(received_at, str) and received_at.strip():
        try:
            value = datetime.fromisoformat(received_at.strip().replace("Z", "+00:00"))
        except ValueError:
            log.warning("Skipping deadline detection: unparseable received_at %r", received_at)
            return None
    else:
        log.warning("Skipping deadline detection: missing received_at")
        return None
    if value.tzinfo is None:
        log.warning("Skipping deadline detection: received_at has no timezone")
        return None
    return value.astimezone(IST)


def _live_text(text: str) -> str:
    """Drop quoted replies ('>' lines and 'On ... wrote:' blocks); forwarded bodies stay."""
    keep, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if line.lstrip().startswith(">"):
            continue
        if _QUOTE_HEADER_RE.match(s):
            in_block = True
            continue
        if in_block:
            in_block = bool(s)
            continue
        keep.append(line)
    return "\n".join(keep)


def _gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Characters between two spans (0 when they overlap)."""
    return max(0, b_start - a_end, a_start - b_end)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _verb_kind(word: str) -> str | None:
    return next((k for k, rx in _ACTION_RES if rx.fullmatch(word)), None)


# ----------------------------------------------------------------------------
# Date resolution: pure calendar math
# ----------------------------------------------------------------------------
def _resolve(g: dict[str, Any], base: datetime) -> tuple[date | None, bool]:
    """Return (calendar date, overdue_is_ok). None means skip."""
    today = base.date()

    if g["iso"]:
        return _safe_date(int(g["iso"]), int(g["iso_m"]), int(g["iso_d"])), False

    if g["n1"]:
        a, b, y = int(g["n1"]), int(g["n2"]), int(g["n3"])
        y += 2000 if y < 100 else 0
        if a > 12 >= b:
            day, mon = a, b
        elif b > 12 >= a:
            mon, day = a, b
        elif a == b <= 12:
            day = mon = a
        else:
            log.warning("Skipping ambiguous numeric date %s/%s/%s", a, b, y)
            return None, False
        return _safe_date(y, mon, day), False

    month_tok = g["dm_m"] or g["md_m"]
    if month_tok:
        if month_tok.islower() and month_tok == "may":   # the verb, not the month
            return None, False
        mon = _MONTHS[month_tok.lower()]
        day = int(g["dm_d"] or g["md_d"])
        year = g["dm_y"] or g["md_y"]
        if year:
            return _safe_date(int(year), mon, day), False
        cand = _safe_date(today.year, mon, day)
        if cand is None:
            return None, False
        past = (today - cand).days
        if past > OVERDUE_GRACE_DAYS:
            cand = _safe_date(today.year + 1, mon, day)
        return cand, 0 < past <= OVERDUE_GRACE_DAYS

    if g["rel_n"]:
        n, unit = int(g["rel_n"]), g["rel_u"].lower()
        if unit.startswith("hour"):
            return (base + timedelta(hours=n)).date(), False
        if unit.startswith("week"):
            return today + timedelta(weeks=n), False
        if unit.startswith("business"):
            d, left = today, n
            while left:
                d += timedelta(days=1)
                left -= d.weekday() < 5
            return d, False
        return today + timedelta(days=n), False

    if g["word"]:
        w = g["word"].lower()
        return today + timedelta(days=2 if w.startswith("day") else 1 if w == "tomorrow" else 0), False

    if g["wd"]:
        # "Friday" / "next Friday" / "this Friday" all mean the coming one (documented choice).
        target = _WEEKDAYS.index(g["wd"].lower())
        return today + timedelta(days=(target - today.weekday()) % 7 or 7), False

    if g["eow"]:
        wd = today.weekday()
        return today + timedelta(days=4 - wd if wd <= 4 else 11 - wd), False

    if g["eom"]:
        year, mon = today.year, today.month
        if g["eom_m"]:
            mon = _MONTHS[g["eom_m"].lower()]
            if (mon, calendar.monthrange(year, mon)[1]) < (today.month, today.day):
                year += 1
        return date(year, mon, calendar.monthrange(year, mon)[1]), False

    return None, False


def _apply_zone(d: date, t: re.Match) -> date:
    """If the email gives a clock time in another zone, convert to the IST calendar day."""
    tz = t.group("tz")
    if not tz:
        return d
    hour = int(t.group("h") or t.group("h24"))
    minute = int(t.group("mi") or t.group("m24") or 0)
    if t.group("ap"):
        hour = hour % 12 + (12 if t.group("ap").lower() == "p" else 0)
    if not 0 <= hour < 24:
        return d
    return datetime.combine(d, time(hour, minute), tzinfo=_TZ[tz.upper()]).astimezone(IST).date()


def _candidates(sentence: str, base: datetime):
    """Yield (mention, date, start, end, overdue_ok) for every date expression."""
    today = base.date()
    found, eods = [], []
    for m in _DATE_RE.finditer(sentence):
        g = m.groupdict()
        if g["eod"]:
            eods.append(m)
            continue
        day, overdue_ok = _resolve(g, base)
        if day is None:
            continue
        start, end = m.span()
        if not g["rel_n"]:
            t = _TIME_RE.match(sentence, end)
            if t:
                day, end = _apply_zone(day, t), t.end()
            else:
                prefix = sentence[:start]
                t = _TIME_RE.search(prefix)
                if t and t.group("tz") and re.fullmatch(r"\s+on\s+", prefix[t.end():], re.I):
                    day = _apply_zone(day, t)
        if not base.year - 1 <= day.year <= base.year + 5:
            log.warning("Skipping date outside plausible range: %r", m.group(0))
            continue
        found.append((sentence[start:end].strip(), day, start, end, overdue_ok))
    # "EOD" only means something by itself; next to a real date the date wins.
    if not found:
        found = [(m.group(0), today, *m.span(), False) for m in eods]
    return found


# ----------------------------------------------------------------------------
# Is this date a deadline? What is it for?
# ----------------------------------------------------------------------------
def _triggers(sentence: str) -> list[re.Match]:
    cues = list(_CUE_RE.finditer(sentence))
    imperative = re.match(r"\W*(?:" + "|".join(f"(?:{p})" for _, p in _ACTIONS) + r")\b", sentence, re.I)
    if imperative or _REQUEST_RE.search(sentence):
        cues += list(_ACTION_RE.finditer(sentence))
    return sorted(cues, key=lambda m: m.start())


def _nearest(matches, start: int, end: int):
    return min(matches, key=lambda m: _gap(m.start(), m.end(), start, end), default=None)


def _vetoed(sentence: str, start: int, end: int, trig: re.Match, trig_gap: int) -> bool:
    """True when the date is an event, sale, past fact or the sender's own promise."""
    sale = _SALE_RE.search(sentence)
    if sale:
        g = _gap(sale.start(), sale.end(), start, end)
        if g <= NEAR and g <= trig_gap:
            return True
    for ev in _EVENT_RE.finditer(sentence):
        g = _gap(ev.start(), ev.end(), start, end)
        if g <= NEAR and g < trig_gap:
            return True
    for past in _PAST_RE.finditer(sentence):
        if _PASSIVE_REQUEST_RE.search(sentence[max(0, past.start() - 30):past.start()]):
            continue
        g = _gap(past.start(), past.end(), start, end)
        if past.group(0).lower() in ("was", "were"):
            if (g <= NEAR and g < trig_gap) or (past.end() <= trig.start() and trig.start() - past.end() <= 40):
                return True
        elif g <= NEAR and past.end() <= end:
            return True
    promise = None
    for promise in _PROMISE_RE.finditer(sentence[:start]):
        pass
    return bool(promise and start - promise.end() <= NEAR)


def _kind_and_action(sentence: str, start: int, end: int, trig: re.Match) -> tuple[str, str]:
    near = [m for m in _ACTION_RE.finditer(sentence) if _gap(m.start(), m.end(), start, end) <= NEAR]
    verb = _nearest([m for m in near if m.end() <= start] or near, start, end)
    cue = trig.group(0).lower()
    ctx = sentence[max(0, min(start, trig.start()) - 40): max(end, trig.end()) + 40]

    verb_kind = _verb_kind(verb.group(0)) if verb else None
    if _EXPIRY_CUE_RE.match(cue) or (cue in ("until", "till", "through") and _OPEN_RE.search(ctx)):
        kind = "expiry"
    elif _PAYMENT_NOUN.search(ctx):
        kind = "payment"
    else:
        kind = verb_kind or "other"

    return kind, _action_phrase(sentence, start, end, trig, verb, kind)


def _action_phrase(sentence, start, end, trig, verb, kind) -> str:
    if verb:
        if verb.start() < start:
            stop = trig.start() if verb.end() <= trig.start() <= start else start
            seg = sentence[verb.start():stop]
        else:
            nxt = next((m for m in _CUE_RE.finditer(sentence, verb.end()) if m.start() > end), None)
            seg = sentence[verb.start(): nxt.start() if nxt else verb.end() + 60]
        words = re.sub(r"\s+", " ", seg).strip(" ,:.-").split()
        while words and words[-1].lower() in _STOP:
            words.pop()
        if words:
            return " ".join(words[:8]).lower()
    # No action verb: name what the deadline is about from the nouns around it.
    for text, from_end in ((sentence[:min(start, trig.start())], True), (sentence[end:], False)):
        words = [w for w in re.findall(r"[A-Za-z][\w'-]*", text)
                 if w.lower() not in _STOP and not _CUE_RE.fullmatch(w)]
        if words:
            return " ".join(words[-4:] if from_end else words[:4]).lower()
    return {"expiry": "expires", "payment": "payment due"}.get(kind, "respond")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def _detect_all(text: str, received_at: Any) -> list[dict[str, str]]:
    base = _received_at_ist(received_at)
    if base is None:
        return []
    today = base.date()
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for sentence in _SPLIT_RE.split(_live_text(text or "")):
        if not sentence.strip():
            continue
        triggers = _triggers(sentence)
        if not triggers:
            continue
        best: dict[tuple[int, int], tuple[int, tuple, re.Match]] = {}  # one date per trigger
        for cand in _candidates(sentence, base):
            mention, day, start, end, overdue_ok = cand
            if day < today and not overdue_ok:
                log.warning("Skipping past date %r", mention)
                continue
            trig = _nearest(triggers, start, end)
            trig_gap = _gap(trig.start(), trig.end(), start, end)
            if trig.group(0).lower() == "before" and trig_gap > BEFORE_MAX_GAP:
                continue
            if trig_gap > NEAR or _vetoed(sentence, start, end, trig, trig_gap):
                continue
            key = (trig.start(), trig.end())
            if key not in best or trig_gap < best[key][0]:
                best[key] = (trig_gap, cand, trig)
        for _, (mention, day, start, end, _ov), trig in best.values():
            kind, action = _kind_and_action(sentence, start, end, trig)
            rec = (day.isoformat(), kind, action)
            dedupe_key = (day.isoformat(), kind)
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                out.append({"date": rec[0], "kind": kind, "action": action, "mention": mention})
    return out


def detect_b(text: str, received_at: Any) -> list[dict[str, str]]:
    """Return deadline records [{date, kind, action}] with IST calendar dates."""
    return [{k: r[k] for k in ("date", "kind", "action")} for r in _detect_all(text, received_at)]


def detect_deadline_v2(text: str, received_at: Any) -> dict[str, Any]:
    """Drop-in shape for the existing detect_deadline: earliest upcoming deadline."""
    records = sorted(_detect_all(text, received_at), key=lambda r: r["date"])
    if not records:
        return {"has_deadline": False, "deadline_date": None, "raw_phrase": None}
    first = records[0]
    return {"has_deadline": True, "deadline_date": first["date"], "raw_phrase": first["mention"]}
