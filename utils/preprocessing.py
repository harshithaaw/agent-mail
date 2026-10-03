import re

import html
def strip_quoted_replies(text):
    """
    Remove lines that are part of a quoted reply chain.
    Gmail/most clients prefix quoted lines with '>' (sometimes '> >' for nested replies).
    We remove any line starting with '>' after stripping leading whitespace.
    """
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not line.strip().startswith(">")]
    return "\n".join(cleaned_lines)

def extract_and_replace_urls(text):
    url_pattern = r"https?://[^\s\"'<>]+"
    urls_found = re.findall(url_pattern, text)
    urls_found = [html.unescape(u) for u in urls_found]
    text_with_placeholder = re.sub(url_pattern, "URLTOKEN", text)
    return text_with_placeholder, urls_found

def strip_signature_block(text):
    """
    Heuristically remove common email signature sign-offs and everything after them.
    We cut the text at the FIRST occurrence of a sign-off word on its own line,
    since signatures always come at the end and rarely repeat mid-body.
    """
    signoff_pattern = r"(?im)^\s*(regards|best regards|thanks|thank you|sincerely|cheers)\s*,?\s*$"
    match = re.search(signoff_pattern, text)
    if match:
        return text[:match.start()]
    return text


def clean_text(raw_body):
    """
    Full cleaning pipeline for one email body, used BOTH at training time
    and at live-screening time (training-serving consistency matters — see notes).
    """
    text = strip_quoted_replies(raw_body)
    text, urls = extract_and_replace_urls(text)
    text = strip_signature_block(text)

    text = text.lower()                          # normalize case: "FREE" and "free" should be the same token
    text = re.sub(r"\s+", " ", text).strip()      # collapse multiple newlines/spaces into one space

    return text, urls

if __name__ == "__main__":
    sample = """Hi Alex,

Check this out: https://suspicious-site.ru/login?id=839201

> This is the first message in an example thread.
> Let's test reply handling.

Thanks,
AgentMail Test
"""
    cleaned, urls = clean_text(sample)
    print("CLEANED TEXT:")
    print(cleaned)
    print("\nURLS FOUND:")
    print(urls)