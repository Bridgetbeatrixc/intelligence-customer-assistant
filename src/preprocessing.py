from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd


TEXT_SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"[^a-z0-9_,-]+")


@dataclass(frozen=True)
class PreparedEntry:
    source_id: str
    question: str
    answer: str
    content: str
    category: str
    tags: str
    info_type: str


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")
    return TEXT_SPACE_RE.sub(" ", text).strip()


def categorize(question: str, answer: str) -> tuple[str, list[str], str]:
    combined = f"{question} {answer}".lower()
    rules = [
        (
            "contact",
            ["contact", "email", "phone", "call", "support", "agent", "address"],
            "contact_information",
        ),
        (
            "policy",
            ["policy", "refund", "return", "cancel", "privacy", "terms", "shipping"],
            "policy",
        ),
        (
            "troubleshooting",
            ["error", "issue", "problem", "trouble", "failed", "fix", "not working"],
            "troubleshooting",
        ),
        (
            "faq",
            ["what", "how", "why", "when", "where", "can i", "do you", "is it"],
            "faq",
        ),
    ]
    for category, keywords, info_type in rules:
        matched = [word.replace(" ", "_") for word in keywords if word in combined]
        if matched:
            return category, matched[:5], info_type
    return "general", ["general"], "faq"


def make_source_id(question: str, answer: str) -> str:
    digest = hashlib.sha1(f"{question}\n{answer}".encode("utf-8")).hexdigest()
    return f"kb_{digest[:12]}"


def prepare_entries(input_path: str, limit: int = 300) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    if not {"prompt", "response"}.issubset(raw.columns):
        raise ValueError("Dataset must contain 'prompt' and 'response' columns.")

    entries_by_category: dict[str, list[PreparedEntry]] = {
        "faq": [],
        "policy": [],
        "troubleshooting": [],
        "contact": [],
        "general": [],
    }
    seen: set[str] = set()

    for row in raw.itertuples(index=False):
        question = clean_text(getattr(row, "prompt"))
        answer = clean_text(getattr(row, "response"))
        if len(question) < 8 or len(answer) < 20:
            continue
        if len(question) > 800 or len(answer) > 1600:
            continue

        dedupe_key = f"{question.lower()}::{answer.lower()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        category, tags, info_type = categorize(question, answer)
        safe_tags = [TAG_RE.sub("", tag.lower()).strip(",-") for tag in tags]
        safe_tags = [tag for tag in safe_tags if tag]
        content = f"Question: {question}\nAnswer: {answer}"

        entry = PreparedEntry(
            source_id=make_source_id(question, answer),
            question=question,
            answer=answer,
            content=content,
            category=category,
            tags=",".join(safe_tags),
            info_type=info_type,
        )
        entries_by_category[category].append(entry)

        if sum(len(items) for items in entries_by_category.values()) >= max(limit * 8, 1000):
            break

    entries: list[PreparedEntry] = []
    minimum_per_core_category = min(25, max(5, limit // 12))
    for category in ["faq", "policy", "troubleshooting", "contact"]:
        entries.extend(entries_by_category[category][:minimum_per_core_category])

    remaining: list[PreparedEntry] = []
    for category in ["faq", "policy", "troubleshooting", "contact", "general"]:
        remaining.extend(entries_by_category[category][minimum_per_core_category:])

    for entry in remaining:
        if len(entries) >= limit:
            break
        entries.append(entry)

    if len(entries) < 50:
        entries.extend(
            [
                PreparedEntry(
                    source_id="seed_refund_policy",
                    question="What is the refund policy?",
                    answer="Refund requests can be reviewed by support when the order meets the stated eligibility and timing requirements.",
                    content="Question: What is the refund policy?\nAnswer: Refund requests can be reviewed by support when the order meets the stated eligibility and timing requirements.",
                    category="policy",
                    tags="refund,policy",
                    info_type="policy",
                ),
                PreparedEntry(
                    source_id="seed_support_contact",
                    question="How can I contact customer support?",
                    answer="Customers can contact support through the official support channel with their account email and order details.",
                    content="Question: How can I contact customer support?\nAnswer: Customers can contact support through the official support channel with their account email and order details.",
                    category="contact",
                    tags="support,contact",
                    info_type="contact_information",
                ),
            ]
        )

    entries = entries[:limit]

    if len(entries) < 50:
        raise ValueError(f"Only found {len(entries)} usable entries; minimum is 50.")

    return pd.DataFrame([entry.__dict__ for entry in entries])
