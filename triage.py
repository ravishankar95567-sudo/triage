import argparse
import re
import pandas as pd
from rank_bm25 import BM25Okapi


# ----------------------------
# Support Corpus (replace with full provided corpus)
# ----------------------------
SUPPORT_CORPUS = [
    # HackerRank
    {
        "company": "HackerRank",
        "product_area": "Assessments",
        "title": "Assessment loading issues",
        "content": "If an assessment does not load, candidates should refresh the page, disable browser extensions, and try a supported browser. If the issue persists, contact support."
    },
    {
        "company": "HackerRank",
        "product_area": "Login & Account Access",
        "title": "Login help",
        "content": "If you cannot log in, reset your password using the forgot password option. If you no longer have access to your email, contact support for account recovery."
    },
    {
        "company": "HackerRank",
        "product_area": "Billing",
        "title": "Billing support",
        "content": "Billing questions related to subscriptions, invoices, or charges may require account verification and should be reviewed by support."
    },
    {
        "company": "HackerRank",
        "product_area": "Plagiarism / Integrity",
        "title": "Integrity review",
        "content": "Assessment integrity concerns, plagiarism flags, and cheating investigations require manual review by the trust and safety team."
    },

    # Claude
    {
        "company": "Claude",
        "product_area": "Account Access",
        "title": "Claude login issues",
        "content": "If you cannot access your Claude account, try resetting your password and confirming your email access. If access is still blocked, contact support."
    },
    {
        "company": "Claude",
        "product_area": "Billing & Subscription",
        "title": "Claude billing",
        "content": "Billing and subscription questions, including charges and refunds, may require account-specific review by the billing team."
    },
    {
        "company": "Claude",
        "product_area": "Usage Limits",
        "title": "Claude usage limits",
        "content": "Usage limits may vary by plan. If you encounter a usage cap, review your plan details in settings."
    },
    {
        "company": "Claude",
        "product_area": "Bugs / Performance",
        "title": "Claude performance issues",
        "content": "If Claude is slow or not responding, refresh the session and retry. If the issue continues, contact support."
    },

    # Visa
    {
        "company": "Visa",
        "product_area": "Fraud & Disputes",
        "title": "Unauthorized charges",
        "content": "If you notice an unauthorized transaction, contact your card issuer immediately to report fraud and secure your account."
    },
    {
        "company": "Visa",
        "product_area": "Card Transactions",
        "title": "Declined transaction",
        "content": "If your transaction was declined, contact your card issuer to verify account status, limits, or possible restrictions."
    },
    {
        "company": "Visa",
        "product_area": "Chargebacks",
        "title": "Dispute a transaction",
        "content": "To dispute a charge, contact your card issuer directly. They can review the transaction and guide you through the dispute process."
    },
    {
        "company": "Visa",
        "product_area": "Payments Security",
        "title": "Card security",
        "content": "For lost, stolen, or compromised cards, contact your card issuer immediately for assistance and account protection."
    },
]


# ----------------------------
# Helpers
# ----------------------------
def normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_company(company, text):
    if company in {"HackerRank", "Claude", "Visa"}:
        return company

    if any(k in text for k in ["assessment", "coding test", "hackerrank", "interview"]):
        return "HackerRank"
    if any(k in text for k in ["claude", "prompt", "chat", "conversation", "model"]):
        return "Claude"
    if any(k in text for k in ["visa", "card", "transaction", "payment", "charge", "fraud"]):
        return "Visa"

    return "None"


def classify_request_type(text):
    if not text or len(text.strip()) < 3:
        return "invalid"

    if any(k in text for k in ["feature request", "add support", "please add", "can you add", "request a feature"]):
        return "feature_request"

    if any(k in text for k in ["bug", "error", "broken", "crash", "not loading", "failed", "issue with", "slow", "not responding"]):
        return "bug"

    if any(k in text for k in ["asdf", "ignore previous instructions", "drop table", "hack the system"]):
        return "invalid"

    return "product_issue"


def risk_score(text):
    score = 0
    high_risk_terms = ["fraud", "unauthorized", "stolen", "compromised", "legal", "harassment", "security vulnerability", "cheating", "plagiarism"]
    medium_risk_terms = ["billing dispute", "refund", "locked out", "missing transaction", "account restricted"]

    for term in high_risk_terms:
        if term in text:
            score += 5
    for term in medium_risk_terms:
        if term in text:
            score += 3

    return score


# ----------------------------
# Retrieval
# ----------------------------
class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.tokenized = [self.tokenize(d["content"] + " " + d["title"]) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def tokenize(self, text):
        return normalize(text).split()

    def search(self, query, company=None, top_k=3):
        q = self.tokenize(query)
        scores = self.bm25.get_scores(q)

        ranked = []
        for i, score in enumerate(scores):
            doc = self.docs[i]
            if company != "None" and doc["company"] != company:
                continue
            ranked.append((score, doc))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:top_k]


# ----------------------------
# Triage Engine
# ----------------------------
def triage_issue(issue, subject, company, retriever):
    text = normalize(f"{subject or ''} {issue or ''}")
    company = infer_company(company, text)
    request_type = classify_request_type(text)
    risk = risk_score(text)

    if request_type == "invalid":
        return {
            "status": "replied",
            "product_area": "General",
            "response": "This request appears to be outside the scope of supported support issues.",
            "justification": "Input is irrelevant, malformed, or outside support scope.",
            "request_type": "invalid"
        }

    results = retriever.search(text, company=company, top_k=3)

    if not results:
        return {
            "status": "escalated",
            "product_area": "General",
            "response": "Thanks for reaching out. This request needs review by a support specialist for further assistance.",
            "justification": "No reliable support documentation match found in corpus.",
            "request_type": request_type
        }

    best_score, best_doc = results[0]
    product_area = best_doc["product_area"]

    if risk >= 5:
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": best_doc["content"],
            "justification": "Sensitive or high-risk issue detected; requires human review.",
            "request_type": request_type
        }

    if best_score < 0.5:
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": "Thanks for reaching out. This request needs review by a support specialist for further assistance.",
            "justification": "Low retrieval confidence; escalation is safer than unsupported guidance.",
            "request_type": request_type
        }

    return {
        "status": "replied",
        "product_area": product_area,
        "response": best_doc["content"],
        "justification": "Issue matched support corpus with sufficient confidence and low risk.",
        "request_type": request_type
    }


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to support_issues.csv")
    parser.add_argument("--output", required=True, help="Path to output predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    retriever = Retriever(SUPPORT_CORPUS)
    outputs = []

    for _, row in df.iterrows():
        result = triage_issue(
        issue=row.get("Issue", ""),
        subject=row.get("Subject", ""),
        company=row.get("Company", "None"),
        retriever=retriever
)
        outputs.append(result)

    out_df = pd.DataFrame(outputs, columns=[
        "status", "product_area", "response", "justification", "request_type"
    ])
    out_df.to_csv(args.output, index=False)

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()