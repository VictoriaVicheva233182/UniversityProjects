"""Controls, recommendations and regulatory references per risk category.

Mappings are written for a fictional Dutch bank. They are a starting point for a
review, not legal advice: whether an obligation applies depends on the actual
use case and on legal assessment.
"""

from __future__ import annotations

CONTROLS: dict[str, list[str]] = {
    "system_prompt_leakage": [
        "ML input classifier blocks prompt extraction attempts",
        "Hardened prompt forbids revealing rules or codes",
        "Output filter blocks any answer containing a canary token",
    ],
    "direct_prompt_injection": [
        "ML input classifier blocks instruction override attempts",
        "Hardened prompt states that user text is data, not instructions",
    ],
    "indirect_prompt_injection": [
        "Untrusted sources are sanitized before they reach the model",
        "The same ML classifier screens retrieved untrusted chunks",
        "Spotlighting: documents are wrapped and labelled with their trust level",
        "Output filter removes links outside havenkade.example and blocks credential requests",
    ],
    "cross_customer_data_leak": [
        "Account lookup is bound to the authenticated session (authorization in code, not in the prompt)",
        "Output filter redacts other customers' identifiers",
    ],
    "confidential_document_leak": [
        "Retrieval filters on document access level, so internal documents never reach customers",
    ],
    "jailbreak_policy_violation": [
        "ML input classifier blocks role play and jailbreak framings",
        "Hardened prompt forbids personal investment, tax and legal advice",
    ],
    "misinformation": [
        "Hardened prompt requires answers from retrieved sources only",
        "Output filter replaces answers with rates or amounts not found in the sources",
    ],
}

RECOMMENDATIONS: dict[str, list[str]] = {
    "system_prompt_leakage": [
        "Remove secrets and internal codes from system prompts completely. Treat every prompt as public.",
        "Keep a canary token as a tripwire and alert the security team when it is triggered.",
        "Log and review repeated extraction attempts per session.",
    ],
    "direct_prompt_injection": [
        "Retrain the classifier monthly on flagged and missed attempts from production logs.",
        "Rate limit or hand over to a human after repeated flagged inputs in one session.",
    ],
    "indirect_prompt_injection": [
        "Do not index third party feeds without a content review and approval step.",
        "Record source and trust level for every document at ingestion and keep them in the index.",
        "Keep the link allowlist and the credential request filter as a last line of defense.",
    ],
    "cross_customer_data_leak": [
        "Enforce authorization in every tool and API the model can call, using the session identity.",
        "Apply least privilege: the assistant should only be able to read the current customer's data.",
        "Alert on denied lookups and include the assistant in the data protection impact assessment.",
    ],
    "confidential_document_leak": [
        "Use separate indexes per audience, or enforce access labels at query time as done here.",
        "Classify documents at ingestion and audit the index regularly for internal material.",
    ],
    "jailbreak_policy_violation": [
        "Add an output side policy check for investment advice, which is a regulated activity in the EU (MiFID II).",
        "Offer a clear hand over to a licensed adviser instead of a bare refusal.",
        "Score this category with an LLM judge at larger scale, with human review of a sample.",
    ],
    "misinformation": [
        "Show sources with every answer and refuse when retrieval confidence is low.",
        "Run a groundedness evaluation on every release, not only during red teaming.",
    ],
}

REGULATORY = [
    {
        "framework": "EU AI Act, Article 50",
        "relevance": "People must be informed that they are interacting with an AI system. The privacy page states this; the chat interface should as well.",
    },
    {
        "framework": "EU AI Act, Articles 9 and 15",
        "relevance": "Risk management and accuracy, robustness and cybersecurity duties apply to high risk systems. A customer service chatbot is usually not high risk (credit scoring would be), so they are used here as a good practice benchmark.",
    },
    {
        "framework": "GDPR, Articles 5(1)(f), 32 and 33",
        "relevance": "Cross customer data disclosure is a personal data breach risk. Security of processing must be appropriate, and breaches may need to be notified.",
    },
    {
        "framework": "DORA (Regulation (EU) 2022/2554)",
        "relevance": "Banks must manage and test ICT risk. An AI assistant connected to customer data is part of that ICT landscape.",
    },
    {
        "framework": "NIST AI RMF 1.0 and NIST AI 600-1",
        "relevance": "This assessment covers MAP (context and threats), MEASURE 2.7 (security and resilience evaluated and documented) and MANAGE (treatment). The generative AI profile lists information security, data privacy and confabulation risks tested here.",
    },
    {
        "framework": "OWASP Top 10 for LLM Applications 2025 and MITRE ATLAS",
        "relevance": "Used as the threat taxonomy for test design and for the mapping of every finding.",
    },
]

LIMITATIONS = [
    "All data is synthetic and the bank is fictional. Findings show the method, not the security of a real institution.",
    "Results hold for the tested model and settings. LLM output varies, so attack success rates are estimates with confidence intervals.",
    "Most detectors are deterministic (canaries, known identifiers, known phishing domain). Jailbreak scoring uses an LLM judge only when enabled, otherwise keyword rules, which can miss paraphrased violations.",
    "The attack suite is fixed. An adaptive attacker who probes the guardrail repeatedly will likely find bypasses that are not measured here.",
    "The guardrail classifier is trained on synthetic templates. Before production it should be trained and validated on real, reviewed traffic.",
]
