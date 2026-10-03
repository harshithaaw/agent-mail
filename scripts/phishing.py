import re
from urllib.parse import urlparse

# Words signaling urgency — common phishing pressure tactics
URGENCY_WORDS = [
    "urgent", "immediately", "verify your account", "suspended",
    "act now", "confirm your identity", "limited time", "click here",
    "your account will be", "password expires", "unusual activity"
]


def check_urgency_and_links(cleaned_text, urls):
    """
    Flags emails combining urgency language with at least one link —
    a common phishing pattern (create panic, provide an escape-hatch link).
    """
    urgency_hits = [word for word in URGENCY_WORDS if word in cleaned_text]
    has_link = len(urls) > 0
    score = 1.0 if (urgency_hits and has_link) else 0.0
    return score, urgency_hits


def check_sender_domain_mismatch(sender_email, urls):
    """
    Flags when the sender's domain doesn't match any linked domain —
    e.g. sender claims @paypal.com but links point to a different domain.
    Only meaningful when both a sender domain and at least one URL exist.
    """
    if not sender_email or "@" not in sender_email or not urls:
        return 0.0, None

    sender_domain = sender_email.split("@")[-1].lower()
    link_domains = set()
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower()
            link_domains.add(domain)
        except ValueError:
            continue

    mismatch = all(sender_domain not in domain for domain in link_domains)
    score = 1.0 if (link_domains and mismatch) else 0.0
    return score, (list(link_domains) if mismatch else None)


def check_keyword_flags(cleaned_text):
    """
    Direct keyword-based phishing indicators, distinct from generic
    spam vocabulary (e.g. brand-impersonation and credential-harvest phrasing).
    """
    keywords = [
        "verify your account", "update your billing", "confirm your password",
        "unusual sign-in activity", "your account has been locked",
        "click the link below to verify"
    ]
    hits = [kw for kw in keywords if kw in cleaned_text]
    score = 1.0 if hits else 0.0
    return score, hits


def phishing_score(cleaned_text, urls, sender_email):
    """
    Combines signals using co-occurrence logic, not independent max().
    A single weak signal (e.g. keyword match alone) is common in
    legitimate security emails and should not trigger a high score alone.
    """
    urgency_score, urgency_hits = check_urgency_and_links(cleaned_text, urls)
    domain_score, link_domains = check_sender_domain_mismatch(sender_email, urls)
    keyword_score, keyword_hits = check_keyword_flags(cleaned_text)

    signal_count = sum([urgency_score > 0, domain_score > 0, keyword_score > 0])

    if domain_score > 0 and (urgency_score > 0 or keyword_score > 0):
        # domain mismatch + any other signal = strongest, most explainable case
        risk_level = "High"
        combined = 0.9
    elif signal_count >= 2:
        # two weaker signals co-occurring = still meaningful
        risk_level = "Medium"
        combined = 0.6
    elif signal_count == 1:
        # single weak signal alone = common in legit security emails, don't over-flag
        risk_level = "Low"
        combined = 0.25
    else:
        risk_level = "Low"
        combined = 0.0

    details = {
        "risk_level": risk_level,
        "urgency_and_link_hit": urgency_hits,
        "domain_mismatch_links": link_domains,
        "keyword_hits": keyword_hits,
    }
    return combined, details