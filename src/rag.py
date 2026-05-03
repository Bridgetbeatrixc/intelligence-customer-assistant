from __future__ import annotations

from typing import Iterable

from src.config import settings


GREETING_WORDS = {"halo", "hello", "hi", "hai", "hey", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"}


SYSTEM_PROMPT = """You are an intelligent customer assistant.
Answer in a warm, concise, and helpful tone.
Use only the provided knowledge base context when giving factual support answers.
If the context is not enough, say that you do not have enough information and suggest contacting support.
When useful, mention the most relevant source category."""


def format_context(results: Iterable) -> str:
    blocks = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"Category: {result.category}",
                    f"Score: {result.score:.3f}",
                    f"Question: {result.question}",
                    f"Answer: {result.answer}",
                ]
            )
        )
    return "\n\n".join(blocks)


def fallback_answer(question: str, results: list) -> str:
    normalized_question = question.lower().strip()
    if normalized_question in GREETING_WORDS:
        return "Halo! Ada yang bisa saya bantu hari ini?"

    if not results:
        return (
            "Saya belum menemukan konteks yang cukup relevan di knowledge base. "
            "Silakan hubungi tim support untuk bantuan lebih lanjut."
        )
    best = results[0]
    if best.score < 0.12:
        return (
            "Saya belum menemukan konteks yang cukup relevan di knowledge base. "
            "Silakan hubungi tim support untuk bantuan lebih lanjut."
        )

    return (
        f"Berdasarkan knowledge base kategori {best.category}, jawaban yang paling relevan adalah: "
        f"{best.answer}"
    )


def generate_answer(question: str, results: list) -> str:
    if not settings.groq_api_key:
        return fallback_answer(question, results)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq
    except ImportError:
        return fallback_answer(question, results)

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.2,
        max_tokens=600,
    )
    context = format_context(results)
    user_prompt = f"Customer question:\n{question}\n\nKnowledge base context:\n{context}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    return str(response.content)
