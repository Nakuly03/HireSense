"""
One-time script: embed resumes with sentence-transformers and upsert to Pinecone.

Usage:
    python scripts/upsert_resumes.py --csv Resume/Resume_cleaned.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBEDDING_DIM, PINECONE_INDEX
from src.embed import embed_texts
from src.pinecone_client import ensure_index, get_index


BATCH_SIZE = 100


def upsert(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required = {"Resume_str", "Category"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


    df["Category"] = df["Category"].str.strip().str.upper()

    if "ID" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    else:
        df["id"] = df["ID"].astype(str)

    print(f"Loaded {len(df)} rows from '{csv_path}'")

    ensure_index(name=PINECONE_INDEX, dimension=EMBEDDING_DIM)
    index = get_index(PINECONE_INDEX)

    texts  = df["Resume_str"].tolist()
    ids    = df["id"].tolist()
    cats   = df["Category"].tolist()

    total_upserted = 0

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Upserting"):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_ids   = ids[start   : start + BATCH_SIZE]
        batch_cats  = cats[start  : start + BATCH_SIZE]

        vectors = embed_texts(batch_texts)

        upsert_data = [
            {
                "id":     bid,
                "values": vec,
                "metadata": {
                    "category": cat,
                    "text":     text[:2000],   # store first 2 000 chars
                    "row_id":   bid,
                },
            }
            for bid, vec, cat, text in zip(batch_ids, vectors, batch_cats, batch_texts)
        ]
        index.upsert(vectors=upsert_data)
        total_upserted += len(upsert_data)

    print(f"\n✅  Upserted {total_upserted} documents to index '{PINECONE_INDEX}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed resumes and upsert to Pinecone.")
    parser.add_argument("--csv", required=True, help="Path to Resume_cleaned.csv")
    args = parser.parse_args()
    upsert(args.csv)
