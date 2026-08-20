# CredLens — Credit Intelligence for Thin-File / New-to-Credit Borrowers

CredLens is a hackathon prototype based on the frozen master plan. It combines:
- Home Credit historical data
- Polars-based feature engineering
- LightGBM risk prediction
- Evidence Confidence
- Isolation Forest integrity anomaly detection
- Deterministic Policy Engine
- SHAP-ready explanations
- Spring Boot API orchestration
- FastAPI ML service
- React frontend
- PostgreSQL/pgvector-ready architecture

## Important data note

The repository does **not** contain the large Home Credit CSV dataset. Put the extracted CSVs in:

`data/raw/`

Expected files:
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- installments_payments.csv
- POS_CASH_balance.csv
- credit_card_balance.csv

Your screenshot shows these files inside `home-credit-default-risk.zip`. Extract that ZIP and copy the CSVs into `data/raw/`.

## VS Code

Yes — use **VS Code** for the overall CredLens project. It is convenient because you can keep React, Spring Boot, FastAPI and the data pipeline in one workspace.

Recommended extensions:
- Extension Pack for Java
- Spring Boot Extension Pack
- Python
- Pylance
- ESLint
- Docker
- PostgreSQL (optional)

## Golden-path build order

1. Put the Home Credit CSVs into `data/raw/`.
2. Create Python environment in `ml-service/`.
3. Run the data pipeline to build an applicant-level feature table.
4. Train the LightGBM risk model and Isolation Forest integrity model.
5. Start FastAPI.
6. Start Spring Boot.
7. Start React.
8. Use the Applicant 360 screen to call the decision API.

## Windows setup

### Python / ML

```powershell
cd ml-service
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
python -m src.train
uvicorn src.app:app --reload --port 8001
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.pipeline
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m uvicorn src.app:app --reload --port 8001
```

For a fast hackathon test, set `CREDLENS_SAMPLE_ROWS=25000` before running the pipeline.

### Spring Boot

Open another VS Code terminal:

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

The backend runs on `http://localhost:8080`.

### React

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on the Vite URL shown in the terminal, normally `http://localhost:5173`.

## Architecture

```text
React
  |
  v
Spring Boot API
  |-----------------------> PostgreSQL / pgvector
  |
  +----> FastAPI ML Service
  |          |
  |          +--> LightGBM risk model
  |          +--> Evidence Confidence
  |          +--> Isolation Forest integrity model
  |          +--> SHAP explanation
  |
  +----> Policy Engine
  |
  +----> Bedrock explanation adapter (optional)
```

The LLM never makes the lending decision:

```text
ML -> Risk probability
Confidence -> Evidence quality
Integrity -> NORMAL / UNUSUAL
Policy Engine -> APPROVE / REFER / DECLINE
LLM -> Explanation only
```

## Data honesty

This prototype uses public Home Credit historical data. It demonstrates **alternative-data-style behavioral intelligence**, not actual Indian Account Aggregator data.

The production roadmap can replace the data-ingestion layer with consented Account Aggregator data without changing the decision architecture.

## Responsible AI

- Protected/demographic fields are not used as predictive inputs.
- Integrity output is `NORMAL` or `UNUSUAL`; it is not a claim of fraud.
- REFER cases remain candidates for human review.
- Evidence Confidence is a documented heuristic in the prototype.
- The LLM is explanation-only and cannot alter risk or decisions.

## Prototype limitations

- Public anonymized historical data
- No Indian Account Aggregator data
- No fraud ground truth in Home Credit
- Unsupervised integrity detection
- Confidence heuristic is not statistically calibrated initially
- Production deployment would require regulatory, security and fairness validation

## Main APIs

Spring Boot:
- `GET /api/v1/health`
- `POST /api/v1/decision`
- `GET /api/v1/applicants/{id}/features`

FastAPI:
- `GET /health`
- `POST /api/v1/predictions`
- `GET /api/v1/applicants/{id}/features`
- `GET /api/v1/applicants/{id}/timeline`
- `GET /api/v1/applicants/{id}/explanation-factors`
- `GET /api/v1/applicants/{id}/integrity-flag`
- `POST /api/v1/applicants/{id}/behavior/update`

## Demo personas

The final demo should use three fixed applicants after training:
- Thin-file / good -> APPROVE
- Thin-file / deteriorating -> REFER
- Suspicious behavior -> REFER / INVESTIGATE

Do not depend on random live applicant selection during the presentation.
