"""Predict whether an email is Phishing or Safe.

Usage:
    python src/predict.py --subject "..." --sender "..." --body "..." --url "..."
    python src/predict.py --demo        # runs 4 built-in examples
"""
import argparse
import sys
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix, hstack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import NUMERIC_FEATURES, extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"


def load():
    bundle = joblib.load(MODEL_PATH)
    return bundle["tfidf"], bundle["scaler"], bundle["clf"]


def predict_one(subject, sender, body, url):
    tfidf, scaler, clf = load()
    feats = extract_features(subject, sender, body, url)
    text = f"{subject} {body}"
    X = hstack([tfidf.transform([text]),
                csr_matrix(scaler.transform([[feats[f] for f in NUMERIC_FEATURES]]))])
    proba = clf.predict_proba(X)[0]
    label = int(clf.predict(X)[0])
    reasons = []
    if feats["has_ip_url"]:
        reasons.append("URL uses a raw IP address instead of a domain name")
    if feats["has_shortener"]:
        reasons.append("URL uses a link shortener, hiding the real destination")
    if feats["has_free_tld"]:
        reasons.append("URL uses a suspicious free top-level domain (.ru/.xyz/.top)")
    if feats["sender_mismatch"]:
        reasons.append("Sender domain does not match the link domain")
    if feats["digit_in_sender_domain"]:
        reasons.append("Sender domain contains digits (look-alike brand name)")
    if feats["has_http_not_https"]:
        reasons.append("Link is plain http, not https")
    if feats["suspicious_keyword_count"] >= 3:
        reasons.append(f"{feats['suspicious_keyword_count']} suspicious keywords "
                       "(verify / urgent / suspended / prize …)")
    if feats["urgency_word_count"] >= 2:
        reasons.append(f"{feats['urgency_word_count']} urgency words "
                       "(urgent / immediately / within 24 hours …)")
    if not reasons:
        reasons.append("No classic phishing signals in URLs or wording; "
                       "language looks like routine mail")
    return {
        "label": "Phishing" if label == 1 else "Safe",
        "phishing_probability": round(float(proba[1]), 4),
        "safe_probability": round(float(proba[0]), 4),
        "reasons": reasons,
        "features": feats,
    }


DEMOS = [
    ("Urgent: Your account will be suspended in 24 hours",
     "support@paypa1-secure.com",
     "Dear customer, we detected unusual activity. Click the link below to verify "
     "your identity immediately, otherwise your account will be suspended within 24 hours.",
     "http://paypa1-secure.com/login/verify"),
    ("Team meeting on Friday at 10am",
     "team@company.com",
     "Hi team, just a reminder that our weekly meeting is on Friday at 10am in the "
     "main conference room. Please bring your project updates.",
     "https://www.company.com/meetings"),
    ("You have won $1,000 — claim your prize now",
     "prizes@winner-claim-today.top",
     "Congratulations! You have been selected to receive a $1,000 gift card. "
     "Claim your prize now by entering your personal details before the offer expires today.",
     "https://winner-claim-today.top/prize"),
    ("Your order has been shipped",
     "orders@shop.example.com",
     "Good news — your order has been shipped and should arrive in 3-5 business days. "
     "You can track your delivery using the tracking number in your account.",
     "https://shop.example.com/track-order"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="")
    ap.add_argument("--sender", default="")
    ap.add_argument("--body", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print("Model not found — run: python src/train.py")
        raise SystemExit(1)

    cases = DEMOS if args.demo else [(args.subject, args.sender, args.body, args.url)]
    for subject, sender, body, url in cases:
        r = predict_one(subject, sender, body, url)
        print(f"\nSubject: {subject}")
        print(f"Sender : {sender}")
        print(f"URL    : {url or '(none)'}")
        print(f"Result : {r['label']}  (phishing probability {r['phishing_probability']:.2%})")
        print("Reasons:")
        for reason in r["reasons"]:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
