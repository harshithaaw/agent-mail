"""

python -m tests.test_agent_decisions
Test Reply Agent decisions end-to-end across email categories.

Runs gate -> Understanding Agent -> Reply Agent for a spread of email
types (matching the categories called out in the original plan: career,
academic, personal, info request, meeting, acknowledgement, urgent,
automated sender, no-context-needed, likely-poor-match) and prints the
full per-node trace for each -- not just the draft.

This script surfaces evidence; it doesn't grade itself. For each email,
manually judge:
  - Did it retrieve? Does the decide_retrieval trace entry's reasoning
    make sense for this email?
  - What came back, and at what distances (retrieve trace entry)?
  - Did validate accept or reject it, and was that the right call given
    what you can see in the retrieved examples?
  - Is the final draft actually grounded in the CURRENT email, not
    leaking specifics from the retrieved examples?
"""
from langchain_core.messages import HumanMessage

from agents.understanding.agent import understanding_agent
from agents.reply.gate import should_generate_reply
from agents.reply.agent import run_reply_agent

TEST_EMAILS = [
    {
        "message_id": "career_1",
        "sender_email": "recruiter@somecompany.com",
        "subject": "Following up on your application",
        "body": "Hi, I wanted to follow up on your application for the "
                "Senior Engineer role. Are you still interested, and do "
                "you have availability for a call next week?",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "academic_1",
        "sender_email": "professor.smith@university.edu",
        "subject": "Extension request question",
        "body": "Hi, I'm reaching out about the deadline for the final "
                "paper. Would it be possible to get a short extension "
                "given the group project workload this week?",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "personal_1",
        "sender_email": "friend@gmail.com",
        "subject": "Weekend plans?",
        "body": "Hey! Any interest in hiking this Saturday if the weather "
                "holds up? Let me know.",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "info_request_1",
        "sender_email": "client@business.com",
        "subject": "Question about pricing",
        "body": "Could you send over your current pricing sheet for the "
                "enterprise tier? Trying to compare a few options.",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "meeting_1",
        "sender_email": "colleague@company.com",
        "subject": "Sync tomorrow",
        "body": "Can we push our 1:1 tomorrow from 10am to 2pm? Something "
                "came up in the morning.",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "acknowledgement_1",
        "sender_email": "manager@company.com",
        "subject": "Re: Report",
        "body": "Got it, thanks for sending this over. Looks good.",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "urgent_1",
        "sender_email": "vendor@supplier.com",
        "subject": "URGENT: Contract needs signature today",
        "body": "We need the signed contract back by end of day today or "
                "we'll miss the shipping window. Please confirm receipt.",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "automated_1",
        "sender_email": "no-reply@service.com",
        "subject": "Your invoice is ready",
        "body": "Your monthly invoice is now available in your account.",
        # route is clean on purpose -- this case checks that gate.py's
        # sender heuristic blocks it even when the route alone wouldn't.
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "no_context_needed_1",
        "sender_email": "teammate@company.com",
        "subject": "Lunch?",
        "body": "Want to grab lunch at noon?",
        "route": "clean_full_pipeline",
    },
    {
        "message_id": "poor_match_likely_1",
        "sender_email": "stranger@unknown.org",
        "subject": "Inquiry about a very specific niche topic",
        "body": "I'm researching 15th-century Venetian glassmaking "
                "techniques and wondered if you had any sources on the "
                "cristallo process.",
        "route": "clean_full_pipeline",
    },
]


def run():
    for email in TEST_EMAILS:
        print("\n" + "=" * 80)
        print(f"{email['message_id']} -- {email['subject']!r}  (sender: {email['sender_email']})")
        print("-" * 80)

        if not should_generate_reply(email["route"], email["sender_email"]):
            print("GATE: blocked -- skipping (expected for the automated-sender case)")
            continue

        email_text = f"{email['subject']}\n\n{email['body']}"

        understanding_result = understanding_agent.invoke({
            "messages": [HumanMessage(content=email_text)]
        })
        agent_output = understanding_result["result"]
        print(f"UNDERSTANDING: category={agent_output.get('category')} "
              f"priority={agent_output.get('priority')}")

        result = run_reply_agent(email_text, agent_output)

        print(f"STOPPED REASON: {result['stopped_reason']}")
        print("TRACE:")
        for step in result["trace"]:
            print(f"  {step}")
        print(f"DRAFT:\n{result['draft']}")


if __name__ == "__main__":
    run()