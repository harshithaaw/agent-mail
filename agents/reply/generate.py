"""
Reply generation using local qwen3:8b via Ollama.

Local model only (qwen3:8b via Ollama). No Gemini, no API key. Takes the
current email, the Understanding Agent's output, and whatever validated
context survived context.py's check (possibly none), and produces the
final draft text.

This is the ONLY place in the pipeline that writes the actual reply.
Retrieval (retriever.py) and validation (context.py) upstream only decide
what context, if any, this call gets to see -- generate_reply itself makes
no retrieval/sufficiency decisions of its own.

CHANGE (fabrication fix): the original prompt only told the model not to
copy dates/names from the retrieved examples. It said nothing about
inventing NEW specifics that appear in neither the email nor the examples.
scratch/fabrication_check.py caught this for real: given an email that
only said "Free this week?" with no day named, the model proposed
"Tuesday or Wednesday" -- invented, not copied, not grounded in anything.
The prompt below adds an explicit instruction against that. This has NOT
yet been re-verified against the fixtures -- re-run
scratch/fabrication_check.py after this change and confirm Fixture 3 no
longer proposes a specific day before considering this resolved.
"""
import os
import re
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

MODEL_NAME = os.getenv("REPLY_AGENT_MODEL", "qwen3:8b")

# Separate LLM instance from the one agent.py uses for the retrieval
# decision. Keeping them separate means a prompt/temperature change for
# one call can't accidentally leak into the other.
_generation_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    num_predict=384,
    reasoning=False,
)

# qwen3 models can emit <think>...</think> reasoning blocks even when not
# explicitly asked to. Strip them so a stray block never ends up in a
# Gmail draft.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def _remove_placeholders(text: str, usable_examples: List[Dict[str, Any]]) -> str:
    """Remove template placeholders; never guess names from other threads."""
    user_name = _infer_user_signature(usable_examples)
    if user_name:
        text = re.sub(r"\[(?:your\s+)?name\]", user_name, text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]{1,60}\]", "", text)
    if user_name:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.fullmatch(r"\s*[—–-]\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s*", line) and line.strip().lstrip("—–- ") != user_name:
                lines[i] = re.sub(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?", user_name, line)
            if re.fullmatch(r"\s*(?:best|regards|thanks|cheers|sincerely),?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s*", line, re.IGNORECASE):
                prefix = re.match(r"\s*(?:best|regards|thanks|cheers|sincerely),?\s+", line, re.IGNORECASE).group(0)
                lines[i] = prefix + user_name
            if i > 0 and lines[i - 1].strip().lower().rstrip(",") in {"best", "regards", "thanks", "cheers", "sincerely"} and re.fullmatch(r"\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s*", line):
                lines[i] = user_name
        text = "\n".join(lines)
    text = re.sub(r"(?im)^(\s*(?:hi|hello|dear))\s*,\s*$", r"\1,", text)
    text = re.sub(r"[ \t]+([,.;])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_user_signature(usable_examples: List[Dict[str, Any]]) -> Optional[str]:
    """Infer the user's name only from explicit signature lines in sent mail."""
    signatures = []
    for example in usable_examples:
        lines = [line.strip() for line in example.get("document", "").splitlines() if line.strip()]
        for line in lines[-3:]:
            match = re.fullmatch(r"[—–-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", line)
            if match:
                signatures.append(match.group(1))
        tail = lines[-3:]
        for i, line in enumerate(tail[:-1]):
            if line.lower().rstrip(",") in {"best", "regards", "thanks", "cheers", "sincerely"}:
                match = re.fullmatch(r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?", tail[i + 1])
                if match:
                    signatures.append(match.group(0))
    if not signatures:
        return None
    # Normalize a full name and first-name-only signature to the same identity.
    signatures = [name.split()[0] for name in signatures]
    counts = {name: signatures.count(name) for name in set(signatures)}
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def _format_examples(usable_examples: List[Dict[str, Any]]) -> str:
    if not usable_examples:
        return "(none -- write the reply without relying on past examples)"
    lines = []
    for i, ex in enumerate(usable_examples, 1):
        doc = ex.get("document", "")
        dist = ex.get("distance")
        dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "?"
        lines.append(f"[Example {i}, distance={dist_str}]\n{doc}")
    return "\n\n".join(lines)


def build_generation_prompt(
    email_text: str,
    agent_output: Dict[str, Any],
    usable_examples: List[Dict[str, Any]],
) -> str:
    category = agent_output.get("category", "Unknown")
    priority = agent_output.get("priority", "Unknown")
    summary = agent_output.get("summary", "")
    tasks = agent_output.get("tasks", [])

    return f"""You are drafting a reply to an email on the user's behalf.

Email analysis (from the Understanding Agent):
- Category: {category}
- Priority: {priority}
- Summary: {summary}
- Tasks: {tasks}
- Sender of the email being answered: {agent_output.get('sender_name') or agent_output.get('sender_email') or 'unknown'}

Current email:
{email_text}

Past sent-email examples (learn the user's actual tone, greeting, level of
detail, sentence patterns, sign-off, and reusable wording/content when it
fits this conversation):
{_format_examples(usable_examples)}

Grounding rule -- use relevant example content carefully, do not invent:
Do not propose or state a specific day, date, or clock time (e.g. a
weekday name like "Tuesday", a calendar date, or a time like "2 PM")
unless that exact day/date/time is explicitly written in the current
email or the retrieved correspondence is clearly about this same event or
thread. Never carry unrelated names, dates, commitments, or factual claims
from another conversation into this reply. Address the email's sender above,
not the user's name found in sent examples. Do not thank someone for a
confirmation unless their email actually confirms something. Do not claim
the user has completed an action unless the current email or tool results
say so. The words "I" and "we" in the current email refer to its sender,
not the user; never turn the sender's actions into the user's actions. Do not
commit the user to availability or a plan unless the current email or analysis
explicitly states that availability. If the current email asks
about availability, timing, or scheduling without giving a specific
day/date/time and the user's calendar is not supplied, say that you will
check your schedule and confirm; do not accept the proposed time or ask the
sender for availability that they already supplied.
The same rule applies to any other concrete fact (a name, a number, a
place) that is not present in the current email.

Write only the reply body, in 2–4 concise sentences. Match the user's voice
from the closest relevant sent example, including a natural greeting or
sign-off when appropriate. Do not include a subject line, placeholders such
as [Name], analysis, markdown, or <think> text. Answer the current email
directly and do not claim an attachment, action, availability, or fact that
the current email does not support."""


def generate_reply(
    email_text: str,
    agent_output: Dict[str, Any],
    usable_examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Produce the final draft reply from qwen3:8b.

    Single, non-tool-calling LLM call by design -- by the time this runs,
    the graph has already decided whether context is usable. This
    function's only job is writing text.
    """
    usable_examples = usable_examples or []
    prompt = build_generation_prompt(email_text, agent_output, usable_examples)
    system = SystemMessage(content="/no_think Write concise, grounded email replies in the user's voice. Never invent facts or placeholders.")
    for attempt in range(5):
        attempt_prompt = prompt
        availability_unknown = not agent_output.get("user_availability")
        asks_availability = bool(re.search(r"\b(free|available|availability|what works|would .{0,40} work|schedule|when can)\b", email_text, re.IGNORECASE))
        if asks_availability and availability_unknown:
            attempt_prompt += "\n\nThe user's calendar availability is unknown. Do not accept a proposed date or time. Say you will check your schedule and confirm, in the user's established tone."
        if attempt:
            attempt_prompt += "\n\nRewrite your previous response. Keep the retrieved examples' tone and structure, but remove unsupported claims, placeholders, and commitments about availability. If asked about a date or schedule and the user's availability is unknown, say you will check and confirm. Return the complete reply body only."
        if attempt == 3 and asks_availability and availability_unknown:
            attempt_prompt += "\n\nREQUIRED SAFE REPLY: The user's availability is unknown. Thank the sender, say you will check your own schedule and confirm later, and use the sender's name above. Do not accept Saturday or any time."
        if attempt == 4 and asks_availability and availability_unknown:
            attempt_prompt = f"""Write a complete reply body to this incoming email in the user's style, using the past sent examples below for tone only.
Sender: {agent_output.get('sender_name') or 'the sender'}
Incoming email: {email_text}
Relevant past examples: {_format_examples(usable_examples)}
The user's availability is unknown. Do not accept or propose a time. Say the user will check their schedule and confirm. Use no placeholder. Body only."""
        if attempt >= 3:
            attempt_prompt += "\n\nDo not say you reviewed a paper, opened a file, or verified an attachment unless a tool result explicitly says that. If the sender asks you to check something, say you will check it instead."
        response = _generation_llm.invoke([system, HumanMessage(content=attempt_prompt)])
        raw = response.content if isinstance(response.content, str) else ""
        cleaned = _remove_placeholders(_strip_think(raw), usable_examples)
        unsupported_commitment = availability_unknown and asks_availability and bool(re.search(
            r"(?:^|[.!?]\s+)(?:sounds good|(?:that )?works for me|that works)\b|"
            r"\b(?:i['’]m|i am) (?:free|available)\b|\bcount me in\b|"
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) sounds good\b",
            cleaned, re.IGNORECASE,
        ))
        unsupported_verification = bool(re.search(
            r"\b(?:i(?:['’]ve| have) reviewed|i(?:['’]ve| have) checked|i(?: can| have) confirm(?:ed)? (?:that )?(?:the )?(?:file|attachment)|(?:the )?file opens (?:correctly|without issues))\b",
            cleaned, re.IGNORECASE,
        ))
        if len(cleaned) >= 35 and not cleaned.lower().startswith("subject:") and not unsupported_commitment and not unsupported_verification:
            return cleaned
    return ""
