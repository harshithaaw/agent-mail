"""Production entrypoint for the existing Understanding Agent graph."""
from langchain_core.messages import HumanMessage

from agents.understanding.agent import understanding_agent
from agents.understanding.classifier import classify_email
from agents.understanding.categories import CATEGORIES
from agents.understanding.agent import deadline_context


def understand_email(email_text: str, received_at=None, message_id=None) -> dict:
    """Run the real graph and guarantee a category from the shared classifier.

    Category classification is the same classify_email tool registered with
    the Understanding Agent. We call it explicitly so categories are present
    even when the graph decides other tools are unnecessary.
    """
    with deadline_context(received_at, message_id):
        result = understanding_agent.invoke({"messages": [HumanMessage(content=email_text)]})["result"]
    category = classify_email(email_text)
    if category not in CATEGORIES:
        raise ValueError(f"Understanding Agent classifier returned invalid category {category!r}")
    return {**result, "category": category}
