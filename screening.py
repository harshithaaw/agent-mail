"""Compatibility adapter for the local Security Agent used by app.py."""
from agents.security.agent import run_security_agent
from utils.preprocessing import clean_text


def screen_email(raw_body, sender_email):
    cleaned_text, urls = clean_text(raw_body)
    result = run_security_agent(cleaned_text, sender_email, urls)
    phishing = result.get("phishing") or {}
    spam = result.get("spam") or {}
    injection = result.get("injection") or {}
    trust_score = {"SAFE": "High", "REVIEW": "Medium", "BLOCK": "Low"}[result["decision"]]
    return {
        "trust_score": trust_score,
        "route": result["route"],
        "is_spam": spam.get("is_spam"),
        "spam_confidence": None,
        "phishing_risk": phishing.get("risk_level"),
        "phishing_details": phishing,
        "injection_flagged": injection.get("flagged"),
        "injection_details": injection.get("details"),
    }
