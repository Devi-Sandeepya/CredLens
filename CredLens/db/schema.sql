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