from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import prepare_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare customer assistant knowledge base CSV.")
    parser.add_argument("--input", required=True, help="Path to dataset_assignment.csv")
    parser.add_argument("--output", default="data/processed/knowledge_base.csv")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_entries(args.input, limit=args.limit)
    df.to_csv(output, index=False)
    print(f"Prepared {len(df)} knowledge base entries at {output}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
