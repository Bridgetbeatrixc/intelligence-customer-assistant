from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class LocalSearchResult:
    source_id: str
    question: str
    answer: str
    category: str
    tags: list[str]
    score: float
    distance: float


class LocalSearchEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.df["content"].fillna(""))

    @classmethod
    def from_csv(cls, path: str) -> "LocalSearchEngine":
        return cls(pd.read_csv(path))

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        threshold: float = 0.05,
    ) -> tuple[list[LocalSearchResult], float]:
        started = time.perf_counter()
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        candidates = scores.argsort()[::-1]
        results: list[LocalSearchResult] = []
        exact_query = query.lower().strip()
        for idx in candidates:
            row = self.df.iloc[int(idx)]
            if category and category != "all" and row["category"] != category:
                continue
            score = float(scores[idx])
            if exact_query and exact_query in str(row["question"]).lower():
                score = min(1.0, score + 0.2)
            if score < threshold:
                continue
            tags = [tag.strip() for tag in str(row["tags"]).split(",") if tag.strip()]
            results.append(
                LocalSearchResult(
                    source_id=row["source_id"],
                    question=row["question"],
                    answer=row["answer"],
                    category=row["category"],
                    tags=tags,
                    score=score,
                    distance=1 - score,
                )
            )
            if len(results) >= top_k:
                break

        latency_ms = (time.perf_counter() - started) * 1000
        return results, latency_ms
