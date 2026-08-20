"""
Semantic retrieval over pgvector-stored policy chunks.

Given a text query (built from decision context), embeds it and finds
the most similar policy chunks via cosine similarity search.
"""

from __future__ import annotations

import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = dict(
    host="127.0.0.1",
    port=5432,
    dbname="credlens",
    user="credlens",
    password="credlens_dev",
)

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def retrieve_policy_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Embed the query and return the top_k most semantically similar
    policy chunks from pgvector, ordered by relevance.
    """
    model = get_model()
    query_embedding = model.encode(query).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT pd.title, pe.chunk_text, 1 - (pe.embedding <=> %s::vector) AS similarity
        FROM policy_embeddings pe
        JOIN policy_documents pd ON pe.document_id = pd.id
        ORDER BY pe.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )

    results = [
        {"source": row[0], "text": row[1], "similarity": round(float(row[2]), 4)}
        for row in cur.fetchall()
    ]

    cur.close()
    conn.close()
    return results


def build_query_from_decision(decision: str, integrity_status: str, risk_score: float) -> str:
    """
    Turn a decision's context into a natural-language query for
    semantic retrieval — this is what gets embedded and matched
    against policy chunks.
    """
    parts = [f"Policy explanation for a {decision} decision"]
    if integrity_status == "UNUSUAL":
        parts.append("with an unusual integrity flag")
    parts.append(f"at a risk score of {round(risk_score * 100)} percent")
    return " ".join(parts)