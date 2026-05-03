from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pgvector_store import ingest_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into PostgreSQL + PgVector.")
    parser.add_argument("--input", default="data/processed/knowledge_base.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    count = ingest_dataframe(df, batch_size=args.batch_size)
    print(f"Upserted {count} rows into knowledge_base.")


if __name__ == "__main__":
    main()
