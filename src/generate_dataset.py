"""Generate a synthetic, labelled phishing / legitimate email dataset.

The dataset is SYNTHETIC — written from realistic phishing / legitimate
email patterns (urgency, account threats, prize lures, mismatched senders,
suspicious URLs vs. routine business / order / newsletter mail). It is
training data for the model in this project, NOT real emails from real
people. No real inboxes were scraped and no personal data is included.

Output: data/emails.csv with columns:
    label (0=safe, 1=phishing), subject, sender, body, url
500 rows, balanced 250 phishing / 250 safe.
"""
import csv
import random
from pathlib import Path

random.seed(42)

OUT = Path(__file__).resolve().parents[1] / "data" / "emails.csv"

# ---------------------------------------------------------------- phishing
PHISH_SUBJECTS = [
    "Urgent: Your account will be suspended in 24 hours",
    "Verify your identity now to avoid account closure",
    "Your password expires today — reset immediately",
    "Unusual sign-in detected — confirm it was you",
    "Your bank account has been locked",
    "Payment failed — update your billing information",
    "You have won $1,000 — claim your prize now",
    "Your package delivery failed — reschedule now",
    "Invoice overdue — pay now to avoid legal action",
    "Security alert: confirm your account details",
    "Your Amazon account has been compromised",
    "Act now: your subscription will be cancelled",
    "Refund pending — claim your money back",
    "Verify your PayPal account immediately",
    "Your Netflix payment was declined",
    "Congratulations! You won a free iPhone",
    "Your mailbox is almost full — upgrade now",
    "Confirm your delivery address to receive your parcel",
    "Tax refund available — claim within 24 hours",
    "Your Apple ID was used in another country",
]

PHISH_BODIES = [
    "Dear customer, we detected unusual activity on your account. Click the link below to verify your identity immediately, otherwise your account will be suspended within 24 hours. Do not ignore this warning.",
    "Your account has been temporarily locked for security reasons. Please confirm your login credentials and update your password by following the link. Failure to do so will result in permanent closure.",
    "We were unable to process your payment. Your service will be interrupted unless you update your billing information now. Click here to update your card details urgently.",
    "Congratulations! You have been selected to receive a $1,000 gift card. Claim your prize now by entering your personal details on the verification page before the offer expires today.",
    "Your package could not be delivered because of an incorrect address. Click the link to confirm your address and pay the redelivery fee of $2.99 immediately.",
    "Dear user, your password will expire today. Reset your password now through the secure link below. If you do not act within 24 hours your account will be deactivated.",
    "Security alert: someone tried to sign in to your account from a new device. If this was not you, verify your account now by clicking the link and confirming your identity.",
    "Your invoice is overdue. Pay now to avoid legal action and additional charges. Download the invoice and complete the payment through the link provided.",
    "You have a tax refund of $450 waiting. Claim it now by confirming your bank details on our secure portal. This offer expires in 24 hours, act fast.",
    "Your Apple ID was used to sign in from another country. If this was not you, secure your account immediately by verifying your credentials at the link below.",
]

PHISH_SENDERS = [
    "support@paypa1-secure.com", "security@amaz0n-verify.net",
    "noreply@apple-id-support.xyz", "alerts@netf1ix-billing.com",
    "service@bank-secure-login.ru", "delivery@parcel-track-now.info",
    "prizes@winner-claim-today.top", "billing@microsoft-update-alert.com",
    "support@google-account-verify.ml", "admin@paypal-security-check.org",
]

PHISH_URLS = [
    "http://paypa1-secure.com/login/verify",
    "http://192.168.4.23/secure/login.php",
    "https://amaz0n-verify.net/account/update",
    "http://bit.ly/claim-prize-now",
    "https://apple-id-support.xyz/signin",
    "http://netf1ix-billing.com/payment/update",
    "https://bank-secure-login.ru/confirm",
    "http://tinyurl.com/delivery-reschedule",
    "https://winner-claim-today.top/prize",
    "http://microsoft-update-alert.com/verify",
]

# ------------------------------------------------------------------- safe
SAFE_SUBJECTS = [
    "Team meeting on Friday at 10am",
    "Your order has been shipped",
    "Monthly newsletter — November updates",
    "Project status update for this week",
    "Welcome to our service — getting started",
    "Your receipt for order #4821",
    "Reminder: dentist appointment tomorrow",
    "Quarterly report is now available",
    "Thank you for your feedback",
    "Invitation: product launch webinar",
    "Your flight itinerary for next week",
    "HR update: new leave policy",
    "Weekly team digest",
    "Your subscription renewal confirmation",
    "Notes from yesterday's workshop",
    "Invoice for your recent purchase",
    "Your password was changed successfully",
    "New comment on your document",
    "Upcoming maintenance window on Sunday",
    "Happy birthday from the team!",
]

SAFE_BODIES = [
    "Hi team, just a reminder that our weekly meeting is on Friday at 10am in the main conference room. Please bring your project updates. See you there.",
    "Good news — your order has been shipped and should arrive in 3-5 business days. You can track your delivery using the tracking number in your account.",
    "Welcome to our newsletter. In this month's edition we share product updates, upcoming events and helpful tips from our team. Thanks for reading.",
    "Hi, here is the project status update for this week. We completed the design phase and are now moving into testing. Let me know if you have questions.",
    "Thank you for signing up. This email confirms your account has been created. You can log in anytime from our official website to get started.",
    "Your receipt is attached. Thank you for your purchase. If you need help with your order, reply to this email or contact our support team.",
    "This is a friendly reminder about your appointment tomorrow at 11am. Please arrive 10 minutes early and bring your ID. Reply to reschedule if needed.",
    "The quarterly report is now available in the shared drive. Please review it before Friday's meeting and send any questions to the finance team.",
    "Thank you for taking the time to share your feedback. We read every response and use it to improve our service. We appreciate your support.",
    "You are invited to our product launch webinar next Tuesday at 4pm. We will demonstrate the new features and answer questions at the end.",
]

SAFE_SENDERS = [
    "team@company.com", "orders@shop.example.com", "newsletter@blog.example.com",
    "hr@company.com", "support@service.example.com", "noreply@airline.example.com",
    "finance@company.com", "events@company.com", "hello@store.example.com",
    "manager@company.com",
]

SAFE_URLS = [
    "https://www.company.com/meetings",
    "https://shop.example.com/track-order",
    "https://blog.example.com/newsletter/november",
    "https://www.company.com/dashboard",
    "https://service.example.com/help",
    "",
    "https://airline.example.com/my-trips",
    "https://www.company.com/reports/q3",
    "",
    "https://events.company.com/webinar",
]

NAMES = ["Alex", "Priya", "Rahul", "Sara", "John", "Meera", "David", "Anita",
         "Kiran", "Emma", "Arjun", "Neha", "Tom", "Lakshmi", "Sam", "Divya"]


def make_phishing(i: int) -> dict:
    return {
        "label": 1,
        "subject": random.choice(PHISH_SUBJECTS),
        "sender": random.choice(PHISH_SENDERS),
        "body": random.choice(PHISH_BODIES),
        "url": random.choice(PHISH_URLS),
    }


def make_safe(i: int) -> dict:
    return {
        "label": 0,
        "subject": random.choice(SAFE_SUBJECTS),
        "sender": random.choice(SAFE_SENDERS),
        "body": random.choice(SAFE_BODIES),
        "url": random.choice(SAFE_URLS),
    }


def main() -> None:
    rows = [make_phishing(i) for i in range(250)] + [make_safe(i) for i in range(250)]
    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "subject", "sender", "body", "url"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    print(f"  phishing: {sum(r['label'] == 1 for r in rows)}, safe: {sum(r['label'] == 0 for r in rows)}")


if __name__ == "__main__":
    main()
