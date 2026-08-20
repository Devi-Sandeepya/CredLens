"""
One-off script to chunk, embed, and store policy documents in pgvector.

Run once from ml-service/ with the venv activated:
    python -m src.ingest_policy
"""

from __future__ import annotations

from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

POLICY_DIR = Path(__file__).resolve().parent.parent / "policy_docs"

DB_CONFIG = dict(
    host="127.0.0.1",
    port=5432,
    dbname="credlens",
    user="credlens",
    password="credlens_dev",
)

CHUNK_SIZE = 500


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < size:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def main():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for file_path in sorted(POLICY_DIR.glob("*.md")):
        title = file_path.stem.replace("_", " ").title()
        text = file_path.read_text(encoding="utf-8")

        cur.execute(
            "INSERT INTO policy_documents (title, source) VALUES (%s, %s) RETURNING id",
            (title, file_path.name),
        )
        doc_id = cur.fetchone()[0]

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO policy_embeddings (document_id, chunk_text, chunk_index, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, chunk, idx, emb.tolist()),
            )

        print(f"  Ingested '{title}' -> {len(chunks)} chunks")

    conn.commit()
    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()