from typing import TypedDict


class SecurityState(TypedDict):
    email_text: str
    sender_email: str
    urls: list[str]
    cleaned_text: str
    spam_result: dict | None
    phishing_result: dict | None
    injection_result: dict | None
    decision: str | None
    route: str | None
