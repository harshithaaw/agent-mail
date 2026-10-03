"""Canonical labels shared by both email pipelines."""
from enum import StrEnum


class Category(StrEnum):
    ACADEMIC = "Academic"
    CAREER = "Career"
    PERSONAL = "Personal"
    PROMOTIONAL = "Promotional"


CATEGORIES = tuple(item.value for item in Category)


def categorize_email(text: str) -> str:
    """Deterministic baseline classifier; replace implementation, not taxonomy."""
    value = text.lower()
    rules = (
        (Category.PROMOTIONAL, ("unsubscribe", "discount", "offer", "sale", "newsletter", "claim your prize")),
        (Category.CAREER, ("interview", "job", "recruit", "application", "position", "resume", "hiring")),
        (Category.ACADEMIC, ("assignment", "course", "university", "professor", "research", "lecture", "thesis")),
    )
    for category, words in rules:
        if any(word in value for word in words):
            return category.value
    return Category.PERSONAL.value
