"""
Gmail draft writer for AgentMail.

Creates Gmail drafts using the generated reply text from the Reply Agent
(agents/reply/agent.py -> run_reply_agent()). Does NOT send emails --
only calls drafts.create API.
"""
import base64
from email.mime.text import MIMEText
from typing import Dict, Any, Optional

from agents.reply.gate import should_generate_reply
from agents.reply.agent import run_reply_agent

from gmail.auth import get_gmail_service


def create_gmail_draft(
    email: Dict[str, Any],
    agent_output: Dict[str, Any],
    route: str,
    generated_reply: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Gmail draft reply for an email.

    Args:
        email: Dict from Gmail ingestion with keys:
            - sender_email: str (original sender's email)
            - subject: str (original subject)
            - message_id_rfc: str (RFC 2822 Message-ID header for threading)
            - thread_id: str (Gmail's threadId for threading)
            - body: str (original email body)
        agent_output: Dict from the Understanding Agent
            (agents/understanding/agent.py's understanding_agent.invoke(...)["result"]),
            with category, priority, summary, tasks. NOT from Gemini -- no
            Gemini anywhere in this pipeline.
        route: Security screening route string from screen_email()

    Returns:
        Dict with keys:
        - draft_id: str or None (Gmail draft ID)
        - thread_id: str or None (Gmail thread ID)
        - skipped: bool (True if gate blocked generation -- no reply attempted)
        - error: str or None (if generation or draft creation failed)
        - reply_result: the full dict from run_reply_agent() (draft,
          stopped_reason, trace, examples_used, retrieved_examples,
          validation), or None if skipped/never reached generation.
          Always populated when generation was attempted, so callers
          (including this module's own __main__ test) can log the trace
          without calling run_reply_agent() a second time.

    Always returns a dict -- never None.
    """
    subject = email.get("subject", "")
    body = email.get("body", "")
    sender_email = email.get("sender_email", "")
    email_text = f"{subject}\n\n{body}"
    list_unsubscribe = email.get("list_unsubscribe", "")
    precedence = email.get("precedence", "")

    # Gate check happens BEFORE calling the Reply Agent at all. Nothing in
    # run_reply_agent()'s real return dict has a "skipped" key -- that
    # concept only exists here, at the gate boundary.
    if not should_generate_reply(route, sender_email, list_unsubscribe=list_unsubscribe, precedence=precedence):
        return {
            "draft_id": None,
            "thread_id": None,
            "skipped": True,
            "error": None,
            "reply_result": None,
        }

    if generated_reply is not None:
        reply_result = {"draft": generated_reply, "stopped_reason": "provided_by_pipeline"}
    else:
        try:
            reply_result = run_reply_agent(
                email_text, agent_output,
                sender_email=sender_email,
                thread_id=email.get("thread_id"),
            )
        except Exception as e:
            return {
                "draft_id": None,
                "thread_id": None,
                "skipped": False,
                "error": f"Reply Agent raised an exception: {e}",
                "reply_result": None,
            }

    draft_text = reply_result.get("draft")
    if not draft_text:
        return {
            "draft_id": None,
            "thread_id": None,
            "skipped": False,
            "error": (
                "Generated draft text is empty "
                f"(stopped_reason={reply_result.get('stopped_reason')!r})"
            ),
            "reply_result": reply_result,
        }

    # Build MIME message
    original_subject = email.get("subject", "")
    message_id_rfc = email.get("message_id_rfc", "")
    thread_id = email.get("thread_id", "")

    # Handle "Re:" prefix - don't double up
    reply_subject = original_subject
    if not reply_subject.lower().startswith("re:"):
        reply_subject = f"Re: {reply_subject}"

    message = MIMEText(draft_text, "plain")
    message["To"] = sender_email
    message["Subject"] = reply_subject

    # Set threading headers if Message-ID is available
    if message_id_rfc:
        message["In-Reply-To"] = message_id_rfc
        message["References"] = message_id_rfc

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        service = get_gmail_service()

        draft_body = {"message": {"raw": raw_message}}
        if thread_id:
            draft_body["threadId"] = thread_id

        # Only drafts.create -- never send/delete/move. Load-bearing constraint.
        draft = service.users().drafts().create(
            userId="me",
            body=draft_body
        ).execute()

        draft_id = draft.get("id")
        draft_thread_id = draft.get("message", {}).get("threadId")

        return {
            "draft_id": draft_id,
            "thread_id": draft_thread_id,
            "skipped": False,
            "error": None,
            "reply_result": reply_result,
        }

    except Exception as e:
        return {
            "draft_id": None,
            "thread_id": None,
            "skipped": False,
            "error": f"Gmail draft creation failed: {str(e)}",
            "reply_result": reply_result,
        }


if __name__ == "__main__":
    """
    Test script that runs create_gmail_draft() against ONE real unread email
    from the authenticated Gmail INBOX and prints detailed output.

    Uses fetch_inbox_emails(), not fetch_emails() -- fetch_emails() is now
    reserved exclusively for sent-mail RAG-corpus ingestion. This script
    is the incoming-mail path: inbox -> screening -> Understanding Agent
    -> gate -> Reply Agent -> Gmail Draft.

    screening.py is a compatibility adapter around the current Security Agent.

    No Gemini anywhere -- Understanding Agent is called directly via its own
    real entrypoint, matching agents/reply/agent.py's own smoke test.
    """
    from screening import screen_email
    from langchain_core.messages import HumanMessage
    from agents.understanding.agent import understanding_agent
    from gmail.fetch import fetch_inbox_emails

    print("Fetching one real unread email from Gmail inbox...")
    emails = fetch_inbox_emails(max_emails=1)

    if not emails:
        print("No unread emails found. Exiting.")
        exit(1)

    email = emails[0]
    sender_email = email["sender_email"]
    print(f"\nUsing email: {email['subject'][:50]}...")
    print(f"From: {sender_email}")
    print(f"Thread ID: {email['thread_id']}")
    print(f"Message-ID (RFC): {email.get('message_id_rfc', 'N/A')}")

    # Get route from screening
    screening_result = screen_email(email["body"], sender_email)
    route = screening_result["route"]
    print(f"Route: {route}")

    # Run the real (non-Gemini) Understanding Agent
    subject = email.get("subject", "")
    body = email.get("body", "")
    email_text = f"{subject}\n\n{body}"
    understanding_result = understanding_agent.invoke({
        "messages": [HumanMessage(content=email_text)]
    })
    agent_output = understanding_result["result"]

    # Create Gmail draft -- single call, no duplicate run_reply_agent()
    print("\nCreating Gmail draft...")
    draft_result = create_gmail_draft(email, agent_output, route)

    if draft_result["skipped"]:
        print("Draft creation: SKIPPED (gate decision)")
    elif draft_result["error"]:
        print(f"Draft creation: ERROR - {draft_result['error']}")
    else:
        print("Draft creation: SUCCESS")
        print(f"Draft ID: {draft_result['draft_id']}")
        print(f"Thread ID: {draft_result['thread_id']}")

    # Log from the reply_result already returned by create_gmail_draft --
    # no second run_reply_agent() call.
    reply_result = draft_result.get("reply_result")
    if reply_result is not None:
        print("\n--- STOPPED REASON ---")
        print(reply_result["stopped_reason"])
        print("\n--- TRACE ---")
        for step in reply_result["trace"]:
            print(step)
        print("\n--- RETRIEVED EXAMPLES (final attempt) ---")
        for i, example in enumerate(reply_result["retrieved_examples"], 1):
            print(f"Example {i}:")
            print(f"  ID: {example.get('id')}")
            dist = example.get("distance")
            print(f"  Distance: {dist:.4f}" if isinstance(dist, (int, float)) else f"  Distance: {dist}")
            print(f"  Document: {example.get('document')}")
        print("\n--- GENERATED DRAFT TEXT ---")
        print(reply_result["draft"])

    print("\n--- CONFIRMATION ---")
    print("NO send/delete/move call was made.")
    print("Only drafts.create was called.")
    print(f"Draft ID {draft_result.get('draft_id')} should appear in Gmail drafts.")
