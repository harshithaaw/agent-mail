# agents/understanding/prioritizer.py
from datetime import date, datetime
from dateutil import parser as dateutil_parser
import logging
import os
import spacy
import re

from agents.understanding.deadline_detector import detect_deadline_v2

log = logging.getLogger(__name__)

# Loaded once at import time — same pattern as screening.py loading the spam model once.
nlp = spacy.load("en_core_web_sm")

URGENCY_KEYWORDS = [
    "urgent", "asap", "immediately", "action required",
    "deadline", "due today", "due tomorrow", "final notice",
    "expires", "right away", "as soon as possible",
]


def has_urgency_words(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in URGENCY_KEYWORDS)


def detect_deadline_legacy(text: str) -> dict:
    """
    Detect deadline information in email text using spaCy NLP and dateutil parsing.
    
    Args:
        text: The email text to analyze
    
    Returns:
        Dict with keys:
        - has_deadline: bool - whether a deadline was detected
        - deadline_date: str or None - ISO format date (YYYY-MM-DD) if found
        - raw_phrase: str or None - the matched text containing the deadline
    """
    # Enhanced deadline indicator patterns
    deadline_indicators = [
        r'due\s+(?:by|before|on|this|that|next)?',  # "due March 15", "due by Friday", "due next Monday"
        r'deadline\s+(?:is|:)?',
        r'submit\s+(?:by|before|no later than)',
        r'application\s+(?:due|deadline)',
        r'response\s+(?:required|due)\s+(?:by|before)',
        r'complete\s+(?:by|before)',
        r'finish\s+(?:by|before)',
        r'action\s+required\s+(?:by|before)',
        r'payment\s+due',
        r'urgent.*due',
        r'by\s+(?:march|april|may|june|july|august|september|october|november|december|january|february|\d{1,2})',  # "by March 15", "by 15th"
    ]
    
    # Check if text contains deadline indicators
    has_deadline_keyword = any(
        re.search(pattern, text, re.IGNORECASE) 
        for pattern in deadline_indicators
    )
    
    # If no deadline keyword, return early (reduces false positives)
    if not has_deadline_keyword:
        return {
            "has_deadline": False,
            "deadline_date": None,
            "raw_phrase": None
        }
    
    # Use spaCy to identify DATE and TIME entities
    doc = nlp(text)
    date_spans = [ent for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    
    # Filter out meeting/scheduling language that creates false positives
    meeting_keywords = ['meeting', 'scheduled', 'appointment', 'call', 'conference', 'session']
    has_meeting_context = any(keyword in text.lower() for keyword in meeting_keywords)
    
    # Parse dates with deadline context
    for span in date_spans:
        try:
            parsed = dateutil_parser.parse(span.text, fuzzy=True, default=datetime.now())
            deadline_date = parsed.date()
            
            span_start = span.start_char
            span_end = span.end_char
            
            # If there's meeting context, be more strict about deadline indicators
            if has_meeting_context:
                # Only accept if deadline keyword is very close to the date
                nearby_text = text[max(0, span_start - 50):min(len(text), span_end + 50)]
                
                if not any(re.search(pattern, nearby_text, re.IGNORECASE) for pattern in deadline_indicators):
                    continue  # Skip this date, it's likely a meeting time
            
            # Find the deadline phrase that led to this detection
            raw_phrase = extract_deadline_phrase(text, span_start, span_end, deadline_indicators)
            
            return {
                "has_deadline": True,
                "deadline_date": deadline_date.isoformat(),
                "raw_phrase": raw_phrase
            }
        except (ValueError, OverflowError):
            continue  # spaCy flagged it as a date-like phrase, but dateutil couldn't parse it — skip, try next span

    # Found deadline keyword but no parseable date
    return {
        "has_deadline": True,
        "deadline_date": None,
        "raw_phrase": extract_deadline_phrase(text, 0, len(text), deadline_indicators)
    }


def _received_at_is_aware(received_at) -> bool:
    if isinstance(received_at, datetime):
        try:
            return received_at.tzinfo is not None and received_at.utcoffset() is not None
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(received_at, str) and received_at.strip():
        try:
            value = datetime.fromisoformat(received_at.strip().replace("Z", "+00:00"))
        except ValueError:
            return False
        return value.tzinfo is not None and value.utcoffset() is not None
    return False


# TODO(REAL-eval): v2 evaluated on synthetic cases only. Validate on real
# labeled emails (logs/deadline_shadow.jsonl) before treating as final.
def detect_deadline(text: str, received_at=None) -> dict:
    """Detect a deadline using v2 only when a valid received timestamp is supplied."""
    if os.environ.get("AGENTMAIL_DEADLINE_DETECTOR") == "legacy":
        return detect_deadline_legacy(text)
    if not _received_at_is_aware(received_at):
        log.warning("Skipping v2 deadline detection: missing, invalid, or timezone-naive received_at")
        return {"has_deadline": False, "deadline_date": None, "raw_phrase": None}
    return detect_deadline_v2(text, received_at)


def extract_deadline_phrase(text: str, span_start: int, span_end: int, patterns: list) -> str:
    """Extract the deadline phrase that contains or is near the detected date."""
    # Search for deadline patterns near the date span
    search_start = max(0, span_start - 100)
    search_end = min(len(text), span_end + 100)
    nearby_text = text[search_start:search_end]
    
    for pattern in patterns:
        match = re.search(pattern, nearby_text, re.IGNORECASE)
        if match:
            # Return a reasonable context around the match
            match_start = search_start + match.start()
            match_end = search_start + match.end()
            context_start = max(0, match_start - 20)
            context_end = min(len(text), match_end + 20)
            return text[context_start:context_end].strip()
    
    # Fallback: return text around the date span
    context_start = max(0, span_start - 30)
    context_end = min(len(text), span_end + 30)
    return text[context_start:context_end].strip()

def compute_priority(has_urgency: bool, deadline_info: dict) -> str:
    """
    Compute priority level based on urgency and deadline information.
    
    Args:
        has_urgency: bool - whether urgency keywords were detected
        deadline_info: dict - result from detect_deadline() function
    
    Returns:
        Priority level: "High", "Medium", or "Low"
    """
    deadline_date = None
    if deadline_info and deadline_info.get('deadline_date'):
        try:
            deadline_date = datetime.fromisoformat(deadline_info['deadline_date']).date()
        except (ValueError, TypeError):
            pass
    
    if deadline_date is not None:
        days_until = (deadline_date - date.today()).days
        if days_until < 0:
            return "High"  # overdue deadline — always High, regardless of urgency wording
        deadline_soon = 0 <= days_until <= 2
    else:
        deadline_soon = False

    score = int(has_urgency) + int(deadline_soon)

    if score == 2:
        return "High"
    elif score == 1:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    # Smoke test with example strings
    print("Running prioritization smoke test...")
    
    examples = [
        "Please submit your assignment by March 15, 2024.",
        "The deadline is in 2 days. Make sure to complete it on time.",
        "Hope you're doing well! Let's catch up soon.",
        "The meeting is scheduled for Friday at 3 PM.",
        "Project due March 15, but also note that the midterm is on April 20th and the final is May 10th.",
        "Urgent: Action required by tomorrow at 5 PM.",
        "Final notice: Payment due immediately.",
        "The proposal is due next Monday for review.",
    ]
    
    for i, example in enumerate(examples, 1):
        has_urgency = has_urgency_words(example)
        deadline_info = detect_deadline_legacy(example)
        priority = compute_priority(has_urgency, deadline_info)
        
        print(f"\nExample {i}: {example}")
        print(f"Has urgency: {has_urgency}")
        print(f"Deadline info: {deadline_info}")
        print(f"Priority: {priority}")
    
    print("\nSmoke test complete.")
