"""
SCAM DETECTOR
-----------------------------------------------------
Hybrid approach:
  1. Rule-based analysis  -> finds scam patterns and gives an explanation
  2. ML model (TF-IDF + Logistic Regression) -> probability of scam
Final score = weighted mix of both -> "Potential Scam" / "Suspicious" / "Low Risk"

Run:  pip install -r requirements.txt
      streamlit run app.py

NOTE: This is a warning tool only. It cannot guarantee 100% detection.
"""

import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
st.set_page_config(page_title="My App", page_icon="🔥")
# ----------------------------------------------------------------------
# 1. Small sample dataset (replace with a bigger labelled dataset later,
#    e.g. the UCI SMS Spam Collection)
#    label 1 = scam, 0 = legitimate
# ----------------------------------------------------------------------
SAMPLES = [
    ("Congratulations! You have won a Rs 50,000 lottery. Click http://bit.ly/claim-now to claim your prize", 1),
    ("Your bank account will be blocked today. Update your KYC immediately at http://secure-bank-verify.xyz", 1),
    ("Work from home job! Earn Rs 5000 daily. Pay registration fee of Rs 500 to start", 1),
    ("URGENT: Your OTP is 482913. Share this OTP with our agent to receive your refund", 1),
    ("You have been selected for a free iPhone. Send your card number and CVV to confirm delivery", 1),
    ("Your package is held at customs. Pay a small fee here to release it: http://track-parcel.top", 1),
    ("Dear customer your SIM will be deactivated in 24 hours. Call this number now to verify Aadhaar", 1),
    ("Investment opportunity! Double your money in 7 days guaranteed. Send UPI payment now", 1),
    ("Final notice: income tax refund pending. Enter your bank details to receive the amount", 1),
    ("Part time job offer, no experience needed, earn from mobile. Contact on WhatsApp now, limited slots", 1),
    ("Hi, are we still meeting for lunch at 1 pm tomorrow?", 0),
    ("Your order #4821 has been shipped and will arrive on Friday. Thanks for shopping with us.", 0),
    ("Reminder: your dentist appointment is scheduled for Monday at 10:30 AM.", 0),
    ("Happy birthday! Wishing you a wonderful year ahead. Let's catch up soon.", 0),
    ("Your OTP for login is 123456. Do not share it with anyone. Valid for 5 minutes.", 0),
    ("Meeting moved to 3 PM. Please bring the project report with you.", 0),
    ("Thanks for your payment of Rs 499. Your electricity bill has been paid successfully.", 0),
    ("Can you send me the notes from yesterday's class? I missed the lecture.", 0),
    ("Your monthly account statement is now available in the official banking app.", 0),
    ("Mom called, please call her back when you are free.", 0),
]


@st.cache_resource
def train_model():
    texts, labels = zip(*SAMPLES)
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english"),
        LogisticRegression(max_iter=1000),
    )
    model.fit(texts, labels)
    return model


# ----------------------------------------------------------------------
# 2. Rule-based scam pattern checks: (name, regex, weight, explanation)
# ----------------------------------------------------------------------
RULES = [
    ("Urgency / pressure",
     r"\b(urgent|immediately|right now|within \d+ (hours?|minutes?)|last chance|final notice|expires? today|act now)\b",
     15, "Scammers create panic so you act without thinking."),
    ("Fake reward / lottery",
     r"\b(congratulations|you have won|you won|lottery|prize|free gift|free iphone|cashback|lucky winner|selected for)\b",
     20, "Unexpected prizes or rewards are a very common scam bait."),
    ("Asks for sensitive details",
     r"\b(otp|cvv|pin|password|card number|aadhaar|aadhar|pan number|bank details|account number|kyc)\b",
     25, "Genuine organisations never ask you to share OTP, PIN, CVV or passwords."),
    ("Payment / fee request",
     r"\b(registration fee|processing fee|pay (a )?(small )?fee|send (upi )?payment|transfer|deposit|upi)\b",
     20, "Asking you to pay money first (fee/deposit) is a classic scam trick."),
    ("Too-good-to-be-true money offer",
     r"\b(earn (rs\.?|₹|\$)?\s?\d+|double your money|guaranteed|no experience needed|work from home|daily income|part[- ]time job)\b",
     20, "Unrealistic earnings or guaranteed returns are warning signs of job/investment fraud."),
    ("Account threat",
     r"\b(account (will be )?(blocked|suspended|closed)|sim (will be )?deactivated|verify your account|unauthori[sz]ed)\b",
     20, "Threats about blocking your account are used to force you to click or call."),
    ("Contact on WhatsApp / unknown number",
     r"\b(whatsapp|telegram|call this number|contact (us )?on)\b",
     10, "Scammers move you to private chat apps where they are hard to trace."),
]

SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "cutt.ly", "is.gd", "rb.gy")
SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".work", ".loan", ".buzz", ".icu", ".tk", ".ml")
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)


def rule_analysis(text: str):
    """Return (score 0-100, list of reasons)."""
    score, reasons = 0, []
    lowered = text.lower()

    for name, pattern, weight, why in RULES:
        match = re.search(pattern, lowered)
        if match:
            score += weight
            reasons.append(f"**{name}** — found “{match.group(0)}”. {why}")

    urls = URL_RE.findall(text)
    for url in urls:
        u = url.lower()
        if any(s in u for s in SHORTENERS):
            score += 15
            reasons.append(f"**Shortened link** — `{url}` hides the real destination.")
        elif u.endswith(SUSPICIOUS_TLDS) or any(t + "/" in u for t in SUSPICIOUS_TLDS):
            score += 15
            reasons.append(f"**Suspicious link domain** — `{url}` uses a domain type often seen in scams.")
        elif u.startswith("http://"):
            score += 5
            reasons.append(f"**Insecure link** — `{url}` does not use HTTPS.")

    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.4 and len(letters) > 15:
        score += 5
        reasons.append("**Excessive CAPITAL letters** — used to look urgent or official.")

    if text.count("!") >= 3:
        score += 5
        reasons.append("**Too many exclamation marks** — typical of promotional scam text.")

    return min(score, 100), reasons


def analyse(text: str, model):
    rule_score, reasons = rule_analysis(text)
    ml_prob = float(model.predict_proba([text])[0][1]) * 100  # scam probability %
    final = round(0.6 * rule_score + 0.4 * ml_prob, 1)

    if final >= 55:
        label = "🚨 Potential Scam"
    elif final >= 30:
        label = "⚠️ Suspicious"
    else:
        label = "✅ Low Risk"
    return label, final, rule_score, ml_prob, reasons


# ----------------------------------------------------------------------
# 3. Streamlit user interface
# ----------------------------------------------------------------------
st.set_page_config(page_title="Scam Message Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ AI Scam Message Detector")
st.caption("Paste a suspicious SMS / WhatsApp / email / job offer to check its risk.")

model = train_model()

EXAMPLES = {
    "— choose an example —": "",
    "Lottery scam": "Congratulations! You have won Rs 50,000 lottery. Click http://bit.ly/claim-now to claim your prize URGENT!!!",
    "Fake job offer": "Work from home job! Earn Rs 5000 daily, no experience needed. Pay registration fee of Rs 500 and contact on WhatsApp now.",
    "Bank KYC scam": "Your bank account will be blocked today. Update KYC immediately at http://secure-bank-verify.xyz and share your OTP.",
    "Normal message": "Hi, are we still meeting for lunch at 1 pm tomorrow?",
}
choice = st.selectbox("Try a sample", list(EXAMPLES.keys()))
message = st.text_area("Message text", value=EXAMPLES[choice], height=180,
                       placeholder="Paste the message here...")

if st.button("Analyse Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        label, final, rule_score, ml_prob, reasons = analyse(message, model)

        st.subheader(label)
        st.progress(int(min(final, 100)), text=f"Risk score: {final}/100")

        col1, col2 = st.columns(2)
        col1.metric("Rule-based score", f"{rule_score}/100")
        col2.metric("ML scam probability", f"{ml_prob:.1f}%")

        st.markdown("### Why this result?")
        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("- No common scam patterns were found in this message.")

        st.markdown("### Safety tips")
        st.info(
            "- Never share OTP, PIN, CVV or passwords.\n"
            "- Don't click unknown links; open the official app/website yourself.\n"
            "- Don't pay any 'fee' to get a job, prize or refund.\n"
            "- Report scams: call **1930** or visit cybercrime.gov.in (India)."
        )

st.divider()
st.caption("⚠️ This tool is only a warning aid and cannot guarantee that every scam message is detected correctly.")
