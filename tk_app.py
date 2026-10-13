"""
AI-Based Scam Message Detection and Risk Analysis App
-----------------------------------------------------
Hybrid approach:
  1. Rule-based analysis  -> finds scam patterns and gives an explanation
  2. ML model (TF-IDF + Logistic Regression) -> probability of scam
Final score = weighted mix of both -> "Potential Scam" / "Suspicious" / "Low Risk"

Run:  pip install scikit-learn
      python tk_app.py

NOTE: This is a warning tool only. It cannot guarantee 100% detection.
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
# 3. Tkinter desktop user interface
# ----------------------------------------------------------------------
EXAMPLES = {
    "Lottery scam": "Congratulations! You have won Rs 50,000 lottery. Click http://bit.ly/claim-now to claim your prize URGENT!!!",
    "Fake job offer": "Work from home job! Earn Rs 5000 daily, no experience needed. Pay registration fee of Rs 500 and contact on WhatsApp now.",
    "Bank KYC scam": "Your bank account will be blocked today. Update KYC immediately at http://secure-bank-verify.xyz and share your OTP.",
    "Normal message": "Hi, are we still meeting for lunch at 1 pm tomorrow?",
}

model = train_model()

root = tk.Tk()
root.title("AI Scam Message Detector")
root.geometry("720x680")

tk.Label(root, text="AI Scam Message Detector", font=("Arial", 18, "bold")).pack(pady=(12, 0))
tk.Label(root, text="Paste a suspicious SMS / WhatsApp / email / job offer", fg="gray").pack()

top = tk.Frame(root)
top.pack(pady=6)
tk.Label(top, text="Sample: ").pack(side="left")
sample_var = tk.StringVar()
combo = ttk.Combobox(top, textvariable=sample_var, values=list(EXAMPLES.keys()), state="readonly", width=24)
combo.pack(side="left")

text_box = tk.Text(root, height=7, wrap="word", font=("Arial", 11))
text_box.pack(fill="x", padx=15, pady=6)


def load_sample(_event=None):
    text_box.delete("1.0", "end")
    text_box.insert("1.0", EXAMPLES[sample_var.get()])


combo.bind("<<ComboboxSelected>>", load_sample)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
score_label = tk.Label(root, text="", font=("Arial", 11))
bar = ttk.Progressbar(root, length=400, maximum=100)
reasons_box = tk.Text(root, height=12, wrap="word", font=("Arial", 10), state="disabled")


def on_analyse():
    message = text_box.get("1.0", "end").strip()
    if not message:
        messagebox.showwarning("Empty", "Please enter a message first.")
        return
    label, final, rule_score, ml_prob, reasons = analyse(message, model)

    color = "red" if "Scam" in label else ("orange" if "Suspicious" in label else "green")
    result_label.config(text=label, fg=color)
    score_label.config(text=f"Risk score: {final}/100   |   Rule score: {rule_score}/100   |   ML probability: {ml_prob:.1f}%")
    bar["value"] = min(final, 100)

    reasons_box.config(state="normal")
    reasons_box.delete("1.0", "end")
    reasons_box.insert("end", "Why this result?\n\n")
    if reasons:
        for r in reasons:
            reasons_box.insert("end", "- " + r.replace("**", "").replace("`", "") + "\n\n")
    else:
        reasons_box.insert("end", "- No common scam patterns were found.\n\n")
    reasons_box.insert("end", "Safety tips:\n- Never share OTP, PIN, CVV or passwords.\n"
                              "- Don't click unknown links.\n- Don't pay any fee for a job/prize/refund.\n"
                              "- Report scams: call 1930 or visit cybercrime.gov.in (India).")
    reasons_box.config(state="disabled")


tk.Button(root, text="Analyse Message", command=on_analyse, bg="#2563eb", fg="white",
          font=("Arial", 12, "bold"), padx=14, pady=4).pack(pady=8)
result_label.pack()
score_label.pack(pady=2)
bar.pack(pady=4)
reasons_box.pack(fill="both", expand=True, padx=15, pady=6)
tk.Label(root, text="Warning tool only. It cannot guarantee every scam is detected.",
         fg="gray", font=("Arial", 9)).pack(pady=(0, 8))

root.mainloop()
