from urllib.parse import urlparse


# Deterministic scoring rules, maintained inside the Security Agent module.
URGENCY_WORDS = [
    "urgent", "immediately", "verify your account", "suspended",
    "act now", "confirm your identity", "limited time", "click here",
    "your account will be", "password expires", "unusual activity",
]


def check_urgency_and_links(cleaned_text: str, urls: list[str]):
    urgency_hits = [word for word in URGENCY_WORDS if word in cleaned_text]
    has_link = len(urls) > 0
    return (1.0 if urgency_hits and has_link else 0.0), urgency_hits


def check_sender_domain_mismatch(sender_email: str, urls: list[str]):
    if not sender_email or "@" not in sender_email or not urls:
        return 0.0, None

    sender_domain = sender_email.split("@")[-1].lower()
    link_domains = set()
    for url in urls:
        try:
            link_domains.add(urlparse(url).netloc.lower())
        except ValueError:
            continue

    mismatch = all(sender_domain not in domain for domain in link_domains)
    return (1.0 if link_domains and mismatch else 0.0), (
        list(link_domains) if mismatch else None
    )


def check_keyword_flags(cleaned_text: str):
    keywords = [
        "verify your account", "update your billing", "confirm your password",
        "unusual sign-in activity", "your account has been locked",
        "click the link below to verify",
    ]
    hits = [keyword for keyword in keywords if keyword in cleaned_text]
    return (1.0 if hits else 0.0), hits


def phishing_score(cleaned_text: str, urls: list[str], sender_email: str):
    urgency_score, urgency_hits = check_urgency_and_links(cleaned_text, urls)
    domain_score, link_domains = check_sender_domain_mismatch(sender_email, urls)
    keyword_score, keyword_hits = check_keyword_flags(cleaned_text)
    signal_count = sum(
        [urgency_score > 0, domain_score > 0, keyword_score > 0]
    )

    if domain_score > 0 and (urgency_score > 0 or keyword_score > 0):
        risk_level, combined = "High", 0.9
    elif signal_count >= 2:
        risk_level, combined = "Medium", 0.6
    elif signal_count == 1:
        risk_level, combined = "Low", 0.25
    else:
        risk_level, combined = "Low", 0.0

    details = {
        "risk_level": risk_level,
        "urgency_and_link_hit": urgency_hits,
        "domain_mismatch_links": link_domains,
        "keyword_hits": keyword_hits,
    }
    return combined, details


def run_phishing(cleaned_text: str, urls: list[str], sender_email: str) -> dict:
    try:
        score, details = phishing_score(cleaned_text, urls, sender_email)
        return {
            "score": float(score),
            "risk_level": details["risk_level"],
            "details": details,
        }
    except Exception as exc:
        return {"score": None, "risk_level": None, "details": {}, "error": str(exc)}
