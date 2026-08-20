# CredLens Architecture

```text
                         +----------------------+
                         |      React UI        |
                         | Applicant 360 / Audit |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Spring Boot API    |
                         | auth / orchestration |
                         +----+------------+----+
                              |            |
                         REST |            | PostgreSQL
                              v            v
                    +----------------+  +------------------+
                    | FastAPI ML     |  | PostgreSQL       |
                    | Risk           |  | + pgvector       |
                    | Confidence     |  | policies/audit   |
                    | Integrity      |  +------------------+
                    | SHAP           |
                    +-------+--------+
                            |
             +--------------+---------------+
             |                              |
       LightGBM                         Isolation Forest
       Risk model                       Integrity model
             |
             +-------------------+
                                 v
                         +---------------+
                         | Policy Engine |
                         | deterministic |
                         +-------+-------+
                                 |
                          APPROVE/REFER/
                             DECLINE
                                 |
                                 v
                         Bedrock / LLM
                         explanation only
```

## Core invariant

The LLM never decides.

`ML -> probability`, `Confidence -> evidence quality`, `Integrity -> NORMAL/UNUSUAL`, `Policy Engine -> decision`, `LLM -> explanation`.
