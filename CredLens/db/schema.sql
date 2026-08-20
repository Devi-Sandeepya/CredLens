CREATE TABLE IF NOT EXISTS decisions (
    id              BIGSERIAL PRIMARY KEY,
    decision_id     VARCHAR(32) NOT NULL UNIQUE,
    applicant_id    BIGINT NOT NULL,
    risk_score      DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    integrity_status VARCHAR(16) NOT NULL,
    decision        VARCHAR(16) NOT NULL,
    mode            VARCHAR(32) NOT NULL,
    model_version   VARCHAR(64) NOT NULL,
    policy_version  VARCHAR(64) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_decisions_applicant_id ON decisions(applicant_id);
CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at);
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_documents (
    id              BIGSERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    source          VARCHAR(64) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS policy_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES policy_documents(id) ON DELETE CASCADE,
    chunk_text      TEXT NOT NULL,
    chunk_index     INT NOT NULL,
    embedding       VECTOR(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_embeddings_vector
    ON policy_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);