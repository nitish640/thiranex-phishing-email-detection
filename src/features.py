"""Hand-crafted numeric features extracted from an email.

These complement TF-IDF text features: phishing mail is often betrayed by
its URLs and its language (urgency, threats, prizes) even when the words
themselves look plausible.
"""
import re
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "verify", "suspended", "urgent", "immediately", "click here", "claim",
    "prize", "winner", "locked", "confirm", "password", "bank", "refund",
    "limited time", "act now", "expires", "deactivated", "unusual activity",
]
URGENCY_WORDS = [
    "urgent", "immediately", "now", "within 24 hours", "expires", "act fast",
    "limited time", "hurry", "last chance", "today only",
]
FREE_TLDS = {".ru", ".xyz", ".top", ".ml", ".info", ".tk", ".gq"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"}

TRUSTED_DOMAINS = {"company.com", "shop.example.com", "blog.example.com",
                    "service.example.com", "airline.example.com", "store.example.com"}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def extract_features(subject: str, sender: str, body: str, url: str) -> dict:
    text = f"{subject} {body}".lower()
    domain = _domain(url or "")
    sender_domain = (sender or "").split("@")[-1].lower()

    url_count = len(re.findall(r"https?://\S+", f"{body} {url}"))
    suspicious_kw = sum(kw in text for kw in SUSPICIOUS_KEYWORDS)
    urgency = sum(w in text for w in URGENCY_WORDS)
    has_ip_url = 1 if re.search(r"https?://\d+\.\d+\.\d+\.\d+", url or "") else 0
    has_shortener = 1 if domain in SHORTENERS else 0
    has_free_tld = 1 if any(domain.endswith(t) for t in FREE_TLDS) else 0
    # Sender domain does not match the domain in the link (classic mismatch)
    sender_mismatch = 1 if (domain and sender_domain and domain != sender_domain
                            and sender_domain not in domain and domain not in sender_domain) else 0
    # Link text / URL mismatch is approximated by a URL in the body whose
    # domain differs from the standalone url field, or by 'click here' + a link
    link_mismatch = 1 if ("click here" in text or "click the link" in text) and url else 0
    has_http_not_https = 1 if (url or "").startswith("http://") else 0
    # Look-alike / digit substitution in sender domain, e.g. paypa1, amaz0n
    digit_in_sender = 1 if re.search(r"\d", sender_domain) else 0
    exclamations = text.count("!")
    dollar = 1 if "$" in text else 0
    body_len = len(body or "")
    url_len = len(url or "")

    return {
        "url_count": url_count,
        "suspicious_keyword_count": suspicious_kw,
        "urgency_word_count": urgency,
        "has_ip_url": has_ip_url,
        "has_shortener": has_shortener,
        "has_free_tld": has_free_tld,
        "sender_mismatch": sender_mismatch,
        "link_mismatch": link_mismatch,
        "has_http_not_https": has_http_not_https,
        "digit_in_sender_domain": digit_in_sender,
        "exclamation_count": exclamations,
        "has_dollar_amount": dollar,
        "body_length": body_len,
        "url_length": url_len,
    }


NUMERIC_FEATURES = [
    "url_count", "suspicious_keyword_count", "urgency_word_count",
    "has_ip_url", "has_shortener", "has_free_tld", "sender_mismatch",
    "link_mismatch", "has_http_not_https", "digit_in_sender_domain",
    "exclamation_count", "has_dollar_amount", "body_length", "url_length",
]
