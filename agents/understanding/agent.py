import json
from typing import TypedDict, Annotated, List

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.understanding.classifier import classify_email
from agents.understanding.extractor import extract_tasks as _extract_tasks
from agents.understanding.prioritizer import (
    has_urgency_words,
    detect_deadline as _detect_deadline,
    compute_priority,
)
from agents.understanding.summarizer import summarize_email


# ============================================================
# TOOLS
# ============================================================

@tool
def classify(email_text: str) -> str:
    """Classify the email into its appropriate category."""
    return classify_email(email_text)


@tool
def detect_deadline(email_text: str) -> str:
    """Detect whether the email contains a deadline and extract deadline information."""
    result = _detect_deadline(email_text)
    return json.dumps(result, default=str)


@tool
def extract_tasks(email_text: str) -> str:
    """Extract actionable tasks, urgency, and deadline mentions from the email."""
    result = _extract_tasks(email_text)
    return json.dumps(result, default=str)


@tool
def assess_priority(email_text: str) -> str:
    """Assess the priority of the email using urgency and deadline information."""
    urgency = has_urgency_words(email_text)
    deadline = _detect_deadline(email_text)
    priority = compute_priority(urgency, deadline)

    return json.dumps(
        {
            "priority": priority,
            "urgency": urgency,
            "deadline": deadline,
        },
        default=str,
    )


@tool
def summarize(email_text: str) -> str:
    """Generate a concise summary of the email."""
    result = summarize_email(email_text)

    if result is None:
        return email_text

    return result


TOOLS = [
    classify,
    detect_deadline,
    extract_tasks,
    assess_priority,
    summarize,
]


# ============================================================
# STATE
# ============================================================

class UnderstandingState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    result: dict


# ============================================================
# LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    num_predict=256,
    reasoning=False,
)

llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Understanding Agent in a secure email processing system.

Your job is to understand an email by deciding which analysis tools are
actually necessary.

IMPORTANT:
- Do NOT automatically call every tool.
- Decide which tools are useful based on the email content.
- Call only the tools whose results contribute to understanding the email.
- You may call multiple tools.
- You may call tools in any order.
- You may call another tool after inspecting a previous tool result if needed.
- Do not call a tool unnecessarily.
- Stop calling tools once you have enough information.

Available tools:

1. classify
   Determines the email category.

2. detect_deadline
   Detects explicit deadline information.

3. extract_tasks
   Extracts actionable tasks, urgency, and deadline mentions.

4. assess_priority
   Determines email priority using urgency and deadline information.

5. summarize
   Produces a concise summary.

Decision guidance:

- Classification is useful when the email category needs to be known.
- Deadline detection is useful when the email contains or may contain
  time-sensitive information.
- Task extraction is useful when the email contains actionable requests.
- Priority assessment is useful when urgency or importance matters.
- Summarization is useful when the email needs a concise representation.
- Do not call tools merely because they are available.

Once you have called every tool you need, stop calling tools and reply
with a short, plain-text confirmation (for example: "Done, gathered
classification, deadline, and priority."). Do NOT produce a final JSON
object yourself — a separate step assembles the result directly from the
tool outputs you already generated. Do not restate, reformat, or correct
any tool's output in your reply.
"""


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(state: UnderstandingState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(TOOLS)


# ============================================================
# ASSEMBLE NODE
# ============================================================

def assemble_result(state: UnderstandingState):
    """Build the final result dict directly from tool outputs already in
    state, rather than trusting the LLM to transcribe them. Any tool that
    was never called contributes None / [] instead of an invented value."""

    outputs = {
        m.name: m.content
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }

    def parse(name):
        raw = outputs.get(name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    priority_result = parse("assess_priority")

    return {
        "result": {
            "category": outputs.get("classify"),
            "deadline": parse("detect_deadline"),
            "priority": priority_result["priority"] if priority_result else None,
            "tasks": parse("extract_tasks") or [],
            "summary": outputs.get("summarize"),
        }
    }


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(UnderstandingState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("assemble", assemble_result)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: "assemble",
    },
)

graph.add_edge("tools", "agent")
graph.add_edge("assemble", END)

understanding_agent = graph.compile()


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":

    test_emails = [
        "Please submit your assignment by Friday. This is urgent.",

        "Reminder: your library book is due back on 10/15.",

        "URGENT: server is down, please respond immediately.",

        "Hey, just wanted to check if you are free for lunch tomorrow.",

        "Congratulations! You have been selected for the next round of our "
        "interview process. Please attend the interview on Monday at 10 AM.",

        "Your monthly bank statement is now available. You can view it "
        "through the mobile application.",

        "Please review the attached document and send your feedback before "
        "5 PM today.",

        "Thank you for your application. We will contact you if your profile "
        "is shortlisted.",

        "Can you please reset my account password? I am unable to log in.",

        "The project meeting has been moved to Thursday. No action is "
        "required from you.",
    ]

    for i, email in enumerate(test_emails, start=1):

        print("\n" + "=" * 80)
        print(f"TEST EMAIL {i}")
        print("=" * 80)

        print(f"\nEMAIL:\n{email}")

        result = understanding_agent.invoke(
            {
                "messages": [
                    HumanMessage(content=email)
                ]
            }
        )

        print("\n--- AGENT TRACE ---")

        for message in result["messages"]:

            message_type = message.__class__.__name__

            print(f"\n[{message_type}]")

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(
                        f"TOOL CALL: {tool_call['name']}"
                    )
                    print(
                        f"ARGUMENTS: {tool_call['args']}"
                    )

            elif message.content:
                print(message.content)

        print("\n--- ASSEMBLED RESULT (from tool outputs, not the LLM) ---")
        print(json.dumps(result["result"], indent=4, default=str))

        print("\n" + "-" * 80)
