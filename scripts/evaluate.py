from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.local_search import LocalSearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local QnA retrieval quality.")
    parser.add_argument("--input", default="data/processed/knowledge_base.csv")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input).head(args.sample_size)
    engine = LocalSearchEngine(pd.read_csv(args.input))
    answer_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    answer_vectorizer.fit(df["answer"].fillna("").tolist())

    hits = []
    top_scores = []
    answer_sims = []
    latencies = []

    for row in df.itertuples(index=False):
        started = time.perf_counter()
        results, latency_ms = engine.search(row.question, top_k=args.top_k, threshold=0)
        latencies.append(latency_ms + ((time.perf_counter() - started) * 1000))
        hits.append(any(result.source_id == row.source_id for result in results))
        top_scores.append(results[0].score if results else 0)

        predicted = results[0].answer if results else ""
        pair = answer_vectorizer.transform([row.answer, predicted])
        answer_sims.append(float(cosine_similarity(pair[0], pair[1])[0][0]))

    print("Evaluation results")
    print(f"sample_size={len(df)}")
    print(f"hit_rate_at_{args.top_k}={statistics.mean(hits):.3f}")
    print(f"mean_top_score={statistics.mean(top_scores):.3f}")
    print(f"mean_answer_similarity={statistics.mean(answer_sims):.3f}")
    print(f"mean_latency_ms={statistics.mean(latencies):.2f}")


if __name__ == "__main__":
    main()
