from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, text

from src.config import settings
from src.embeddings import embed_query, embed_texts


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    question: str
    answer: str
    category: str
    tags: list[str]
    score: float
    distance: float


def get_engine():
    engine = create_engine(settings.database_url)
    with engine.connect() as connection:
        raw_connection = connection.connection.driver_connection
        register_vector(raw_connection)
    return engine


def ensure_schema(engine) -> None:
    schema = open("schema.sql", encoding="utf-8").read()
    with engine.begin() as connection:
        for statement in schema.split(";"):
            if statement.strip():
                connection.execute(text(statement))


def ingest_dataframe(df: pd.DataFrame, batch_size: int = 32) -> int:
    engine = get_engine()
    ensure_schema(engine)
    records = df.to_dict("records")

    inserted = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embed_texts([item["content"] for item in batch], batch_size=batch_size)
        with engine.begin() as connection:
            for item, vector in zip(batch, vectors):
                tags = [tag.strip() for tag in str(item["tags"]).split(",") if tag.strip()]
                connection.execute(
                    text(
                        """
                        INSERT INTO knowledge_base
                        (source_id, question, answer, content, category, tags, info_type, metadata, embedding)
                        VALUES
                        (:source_id, :question, :answer, :content, :category, :tags, :info_type, :metadata, :embedding)
                        ON CONFLICT (source_id) DO UPDATE SET
                            question = EXCLUDED.question,
                            answer = EXCLUDED.answer,
                            content = EXCLUDED.content,
                            category = EXCLUDED.category,
                            tags = EXCLUDED.tags,
                            info_type = EXCLUDED.info_type,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "source_id": item["source_id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "content": item["content"],
                        "category": item["category"],
                        "tags": tags,
                        "info_type": item["info_type"],
                        "metadata": json.dumps({"source": "dataset_assignment.csv"}),
                        "embedding": vector,
                    },
                )
                inserted += 1
    return inserted


def semantic_search(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    threshold: float | None = None,
) -> tuple[list[SearchResult], float]:
    started = time.perf_counter()
    top_k = top_k or settings.top_k
    threshold = settings.similarity_threshold if threshold is None else threshold
    vector = embed_query(query)

    category_filter = "AND category = :category" if category and category != "all" else ""
    sql = text(
        f"""
        SELECT source_id, question, answer, category, tags,
               embedding <=> :embedding AS distance,
               1 - (embedding <=> :embedding) AS score
        FROM knowledge_base
        WHERE embedding IS NOT NULL
        {category_filter}
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
        """
    )
    params = {"embedding": vector, "top_k": top_k}
    if category and category != "all":
        params["category"] = category

    with get_engine().connect() as connection:
        rows = connection.execute(sql, params).mappings().all()

    results = [
        SearchResult(
            source_id=row["source_id"],
            question=row["question"],
            answer=row["answer"],
            category=row["category"],
            tags=list(row["tags"] or []),
            score=float(row["score"]),
            distance=float(row["distance"]),
        )
        for row in rows
        if float(row["score"]) >= threshold
    ]
    latency_ms = (time.perf_counter() - started) * 1000
    return results, latency_ms
