from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    RISK_MODEL_PATH,
    RISK_FEATURES_PATH,
    INTEGRITY_MODEL_PATH,
    INTEGRITY_FEATURES_PATH,
    APPLICANT_FEATURES_PATH,
)

from .explain import explain_row


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="CredLens ML Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class BehaviorEvent(BaseModel):
    timestamp: str
    paymentAmount: float = Field(ge=0)
    scheduledAmount: float = Field(ge=0)
    balance: float = Field(ge=0)
    daysPastDue: int = Field(ge=0)


# ============================================================
# ARTIFACT LOADING
# ============================================================

def load_artifacts():
    if not RISK_MODEL_PATH.exists():
        raise RuntimeError(
            "M5 risk model not trained. "
            "Run: python -m src.train"
        )

    if not RISK_FEATURES_PATH.exists():
        raise RuntimeError(
            "M5 risk feature list not found."
        )

    risk = joblib.load(RISK_MODEL_PATH)

    risk_features = json.loads(
        RISK_FEATURES_PATH.read_text()
    )

    integrity = None

    if INTEGRITY_MODEL_PATH.exists():
        integrity = joblib.load(
            INTEGRITY_MODEL_PATH
        )

    integrity_features = []

    if INTEGRITY_FEATURES_PATH.exists():
        integrity_features = json.loads(
            INTEGRITY_FEATURES_PATH.read_text()
        )

    return (
        risk,
        risk_features,
        integrity,
        integrity_features,
    )


# ============================================================
# DATA ACCESS
# ============================================================

def load_feature_table() -> pd.DataFrame:

    if not APPLICANT_FEATURES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "M5 feature table not available. "
                "Run: python -m src.pipeline"
            ),
        )

    return pd.read_parquet(
        APPLICANT_FEATURES_PATH
    )


def row_for(
    applicant_id: int,
) -> pd.Series:

    df = load_feature_table()

    hit = df[
        df["SK_ID_CURR"] == applicant_id
    ]

    if hit.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Applicant {applicant_id} "
                "not found"
            ),
        )

    return hit.iloc[0]


# ============================================================
# FEATURE PREPARATION
# ============================================================

def clean_feature_value(value) -> float:

    if value is None:
        return 0.0

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not np.isfinite(value):
        return 0.0

    return value


def build_feature_frame(
    row: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:

    values = {
        feature: clean_feature_value(
            row.get(feature, 0)
        )
        for feature in feature_names
    }

    return pd.DataFrame(
        [values],
        columns=feature_names,
    )


# ============================================================
# EVIDENCE CONFIDENCE
# ============================================================

def confidence_for(
    row: pd.Series,
) -> float:
    """
    Evidence Confidence is separate from the
    LightGBM risk probability.

    It measures the amount and quality of
    historical evidence available.
    """

    bureau_count = max(
        clean_feature_value(
            row.get(
                "bureau_account_count",
                0,
            )
        ),
        0.0,
    )

    installment_count = max(
        clean_feature_value(
            row.get(
                "installment_count",
                0,
            )
        ),
        0.0,
    )

    # --------------------------------------------------------
    # Evidence depth
    # --------------------------------------------------------

    bureau_history = min(
        bureau_count / 10.0,
        1.0,
    )

    repayment_history = min(
        installment_count / 50.0,
        1.0,
    )

    # --------------------------------------------------------
    # Historical coverage
    # --------------------------------------------------------

    evidence_columns = [
        "bureau_account_count",
        "bureau_active_count",
        "bureau_total_credit",
        "bureau_total_debt",
        "bureau_total_overdue",
        "bureau_history_depth",
        "bureau_delinquent_months",
        "bureau_month_count",
        "installment_count",
        "on_time_payment_ratio",
        "late_payment_ratio",
        "avg_payment_delay",
        "previous_application_count",
        "previous_approval_ratio",
        "previous_refusal_ratio",
        "recent_avg_delay",
        "recent_on_time_ratio",
        "historical_avg_delay",
        "historical_on_time_ratio",
    ]

    available = [
        column
        for column in evidence_columns
        if column in row.index
    ]

    if available:

        coverage = float(
            sum(
                pd.notna(row[column])
                and clean_feature_value(
                    row[column]
                ) != 0
                for column in available
            )
            / len(available)
        )

    else:

        coverage = 0.0

    # --------------------------------------------------------
    # Data completeness
    # --------------------------------------------------------

    completeness = float(
        1.0 - row.isna().mean()
    )

    # --------------------------------------------------------
    # Final evidence confidence
    # --------------------------------------------------------

    confidence = (
        0.30 * bureau_history
        + 0.25 * repayment_history
        + 0.25 * coverage
        + 0.20 * completeness
    )

    return round(
        100.0 * min(
            max(confidence, 0.0),
            1.0,
        ),
        1,
    )


# ============================================================
# DETERMINISTIC POLICY ENGINE
# ============================================================

# Frozen policy thresholds selected from the final-holdout policy
# analysis. These are decision-policy parameters, not model
# training parameters, and do not change the M5 probability.
APPROVE_THRESHOLD = 0.30
DECLINE_THRESHOLD = 0.60
MIN_CONFIDENCE = 60.0


def policy(
    risk: float,
    confidence: float,
    integrity: str,
) -> str:
    """
    Deterministic and auditable credit decision policy.

    Priority:
      1. UNUSUAL integrity -> REFER
      2. Low risk + sufficient evidence -> APPROVE
      3. High risk + sufficient evidence -> DECLINE
      4. Otherwise -> REFER

    The LLM does not participate in this decision.

    The thresholds were selected after final-holdout policy
    analysis. They do not alter the underlying M5 model.
    """

    if integrity == "UNUSUAL":
        return "REFER"

    if (
        risk < APPROVE_THRESHOLD
        and confidence >= MIN_CONFIDENCE
    ):
        return "APPROVE"

    if (
        risk >= DECLINE_THRESHOLD
        and confidence >= MIN_CONFIDENCE
    ):
        return "DECLINE"

    return "REFER"


# ============================================================
# INTEGRITY
# ============================================================

def calculate_integrity(
    x: pd.DataFrame,
    integrity_model,
    integrity_features: list[str],
) -> str:

    if (
        integrity_model is None
        or not integrity_features
    ):
        return "NORMAL"

    available = [
        feature
        for feature in integrity_features
        if feature in x.columns
    ]

    if not available:
        return "NORMAL"

    ix = (
        x[available]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    prediction = integrity_model.predict(
        ix
    )[0]

    return (
        "UNUSUAL"
        if int(prediction) == -1
        else "NORMAL"
    )


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(
    applicant_id: int,
) -> dict:

    row = row_for(
        applicant_id
    )

    (
        risk_model,
        risk_features,
        integrity_model,
        integrity_features,
    ) = load_artifacts()

    x = build_feature_frame(
        row,
        risk_features,
    )

    risk = float(
        risk_model.predict_proba(
            x
        )[:, 1][0]
    )

    integrity_status = calculate_integrity(
        x,
        integrity_model,
        integrity_features,
    )

    confidence = confidence_for(
        row
    )

    decision = policy(
        risk,
        confidence,
        integrity_status,
    )

    return {
        "applicantId": applicant_id,
        "riskScore": round(
            risk,
            4,
        ),
        "confidence": confidence,
        "integrityStatus": integrity_status,
        "decision": decision,
        "mode": "LIVE",
        "modelVersion": "m5-risk-model-v1",
        "policyVersion": "credit-policy-v2",
        "policyThresholds": {
            "approveBelow": APPROVE_THRESHOLD,
            "declineAtOrAbove": DECLINE_THRESHOLD,
            "minimumConfidence": MIN_CONFIDENCE,
            "unusualIntegrityAction": "REFER",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "CredLens ML Service",
    }


# ============================================================
# APPLICANT FEATURES
# ============================================================

@app.get(
    "/api/v1/applicants/{applicant_id}/features"
)
def applicant_features(
    applicant_id: int,
):

    row = row_for(
        applicant_id
    )

    features = row.drop(
        labels=[
            "TARGET",
        ],
        errors="ignore",
    ).to_dict()

    return {
        "applicantId": applicant_id,
        "features": features,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/api/v1/predictions")
def prediction(
    payload: dict,
):

    if "applicantId" not in payload:
        raise HTTPException(
            status_code=400,
            detail="applicantId is required",
        )

    try:
        applicant_id = int(
            payload["applicantId"]
        )

    except (
        TypeError,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail="applicantId must be an integer",
        )

    return make_prediction(
        applicant_id
    )


# ============================================================
# INTEGRITY FLAG
# ============================================================

@app.get(
    "/api/v1/applicants/{applicant_id}/integrity-flag"
)
def integrity_flag(
    applicant_id: int,
):

    result = make_prediction(
        applicant_id
    )

    return {
        "applicantId": applicant_id,
        "integrityStatus": result[
            "integrityStatus"
        ],
    }


# ============================================================
# MODEL + CONTEXTUAL EXPLANATION
# ============================================================

@app.get(
    "/api/v1/applicants/{applicant_id}/explanation-factors"
)
def explanation_factors(
    applicant_id: int,
):
    """
    Return model-grounded and contextual explanations.

    SHAP explains the actual M5 LightGBM prediction.

    Contextual factors provide deterministic domain
    interpretation around repayment behavior.

    Neither explanation layer changes the decision.
    """

    row = row_for(
        applicant_id
    )

    # --------------------------------------------------------
    # 1. Genuine M5 SHAP explanation
    # --------------------------------------------------------

    try:

        model_explanation = explain_row(
            row,
            top_n=5,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate SHAP explanation: "
                f"{str(exc)}"
            ),
        )

    # --------------------------------------------------------
    # 2. Deterministic contextual evidence
    # --------------------------------------------------------

    contextual_factors = []

    bureau_count = clean_feature_value(
        row.get(
            "bureau_account_count",
            0,
        )
    )

    avg_delay = clean_feature_value(
        row.get(
            "avg_payment_delay",
            0,
        )
    )

    on_time_ratio = clean_feature_value(
        row.get(
            "on_time_payment_ratio",
            0,
        )
    )

    recent_on_time_ratio = clean_feature_value(
        row.get(
            "recent_on_time_ratio",
            on_time_ratio,
        )
    )

    historical_on_time_ratio = clean_feature_value(
        row.get(
            "historical_on_time_ratio",
            on_time_ratio,
        )
    )

    delay_trend = clean_feature_value(
        row.get(
            "delay_trend",
            0,
        )
    )

    # --------------------------------------------------------
    # Credit history
    # --------------------------------------------------------

    if bureau_count < 2:

        contextual_factors.append(
            {
                "factor": "Limited credit history",
                "direction": "INCREASED_RISK",
            }
        )

    # --------------------------------------------------------
    # Payment timing
    # --------------------------------------------------------

    if avg_delay > 5:

        contextual_factors.append(
            {
                "factor": (
                    "Payments show meaningful "
                    "late timing"
                ),
                "direction": "INCREASED_RISK",
            }
        )

    elif avg_delay < -5:

        contextual_factors.append(
            {
                "factor": (
                    "Payments are generally "
                    "made early"
                ),
                "direction": "REDUCED_RISK",
            }
        )

    # --------------------------------------------------------
    # Repayment consistency
    # --------------------------------------------------------

    if on_time_ratio >= 0.80:

        contextual_factors.append(
            {
                "factor": (
                    "Strong repayment consistency"
                ),
                "direction": "REDUCED_RISK",
            }
        )

    # --------------------------------------------------------
    # Recent vs historical behavior
    # --------------------------------------------------------

    if (
        delay_trend > 0
        and recent_on_time_ratio >= 0.80
    ):

        contextual_factors.append(
            {
                "factor": (
                    "Payment timing is later than "
                    "the historical baseline, but "
                    "repayment remains mostly on-time "
                    "or early"
                ),
                "direction": "MIXED",
            }
        )

    elif delay_trend > 0:

        contextual_factors.append(
            {
                "factor": (
                    "Recent payment timing is later "
                    "than the historical baseline"
                ),
                "direction": "INCREASED_RISK",
            }
        )

    elif delay_trend < 0:

        contextual_factors.append(
            {
                "factor": (
                    "Recent payment timing is earlier "
                    "than the historical baseline"
                ),
                "direction": "REDUCED_RISK",
            }
        )

    # --------------------------------------------------------
    # Historical vs recent on-time behavior
    # --------------------------------------------------------

    if (
        recent_on_time_ratio
        > historical_on_time_ratio
    ):

        contextual_factors.append(
            {
                "factor": (
                    "Recent on-time repayment "
                    "performance is improving"
                ),
                "direction": "REDUCED_RISK",
            }
        )

    elif (
        recent_on_time_ratio
        < historical_on_time_ratio
    ):

        contextual_factors.append(
            {
                "factor": (
                    "Recent on-time repayment "
                    "performance is weakening"
                ),
                "direction": "INCREASED_RISK",
            }
        )

    if not contextual_factors:

        contextual_factors.append(
            {
                "factor": (
                    "No dominant contextual "
                    "repayment signal"
                ),
                "direction": "NEUTRAL",
            }
        )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "applicantId": applicant_id,

        "modelExplanation": {
            "method": model_explanation[
                "method"
            ],
            "model": model_explanation[
                "model"
            ],
            "riskScore": model_explanation[
                "riskScore"
            ],
            "topRiskFactors": model_explanation[
                "topRiskFactors"
            ],
            "topProtectiveFactors": model_explanation[
                "topProtectiveFactors"
            ],
        },

        "contextualFactors": contextual_factors,

        "explainabilityNote": (
            "SHAP factors describe the contribution "
            "of features to the M5 LightGBM prediction. "
            "Contextual factors are deterministic "
            "evidence signals and do not directly "
            "determine the decision."
        ),
    }


# ============================================================
# TEMPORAL JOURNEY
# ============================================================

@app.get(
    "/api/v1/applicants/{applicant_id}/timeline"
)
def timeline(
    applicant_id: int,
):

    row = row_for(
        applicant_id
    )

    current_result = make_prediction(
        applicant_id
    )

    current = current_result[
        "riskScore"
    ]

    # --------------------------------------------------------
    # Historical behavioral evidence
    # --------------------------------------------------------

    historical_on_time = clean_feature_value(
        row.get(
            "historical_on_time_ratio",
            0,
        )
    )

    recent_on_time = clean_feature_value(
        row.get(
            "recent_on_time_ratio",
            historical_on_time,
        )
    )

    historical_delay = clean_feature_value(
        row.get(
            "historical_avg_delay",
            0,
        )
    )

    recent_delay = clean_feature_value(
        row.get(
            "recent_avg_delay",
            historical_delay,
        )
    )

    delay_trend = clean_feature_value(
        row.get(
            "delay_trend",
            0,
        )
    )

    # --------------------------------------------------------
    # Behavioral journey
    #
    # IMPORTANT:
    # These historical points represent observed
    # behavioral evidence, NOT fabricated historical
    # LightGBM predictions.
    # --------------------------------------------------------

    if delay_trend > 0:

        behavior_trend = "DETERIORATING"

    elif delay_trend < 0:

        behavior_trend = "IMPROVING"

    else:

        behavior_trend = "STABLE"

    return {
        "applicantId": applicant_id,

        "source": (
            "historical-and-recent-repayment-features"
        ),

        "currentRiskScore": round(
            current,
            4,
        ),

        "behaviorTrend": behavior_trend,

        "points": [
            {
                "label": "Historical",
                "onTimeRatio": round(
                    historical_on_time,
                    4,
                ),
                "averagePaymentDelay": round(
                    historical_delay,
                    4,
                ),
                "period": "OLDER_THAN_12_MONTHS",
            },
            {
                "label": "Recent",
                "onTimeRatio": round(
                    recent_on_time,
                    4,
                ),
                "averagePaymentDelay": round(
                    recent_delay,
                    4,
                ),
                "period": "LAST_12_MONTHS",
            },
            {
                "label": "Current Risk",
                "riskScore": round(
                    current,
                    4,
                ),
                "decision": current_result[
                    "decision"
                ],
                "period": "CURRENT",
            },
        ],
    }


# ============================================================
# REAL-TIME BEHAVIOR UPDATE
# ============================================================

@app.post(
    "/api/v1/applicants/{applicant_id}/behavior/update"
)
def behavior_update(
    applicant_id: int,
    event: BehaviorEvent,
):
    """
    Apply a new behavioral event to the applicant's
    M5 feature representation and re-run the actual
    LightGBM model.

    The event does NOT directly modify the risk score.

    Instead:

        behavior event
              ↓
        updated behavioral features
              ↓
        M5 LightGBM
              ↓
        new risk probability
              ↓
        evidence confidence
              ↓
        integrity
              ↓
        policy decision
    """

    # --------------------------------------------------------
    # 1. Load current applicant state.
    # --------------------------------------------------------

    row = row_for(
        applicant_id
    ).copy()

    (
        risk_model,
        risk_features,
        integrity_model,
        integrity_features,
    ) = load_artifacts()

    # --------------------------------------------------------
    # 2. Convert incoming event into behavioral signals.
    # --------------------------------------------------------

    scheduled = max(
        float(event.scheduledAmount),
        1.0,
    )

    payment_ratio = min(
        max(
            float(event.paymentAmount)
            / scheduled,
            0.0,
        ),
        2.0,
    )

    delay_days = float(
        event.daysPastDue
    )

    event_on_time = (
        1.0
        if delay_days == 0
        else 0.0
    )

    event_late_ratio = (
        1.0 - event_on_time
    )

    # --------------------------------------------------------
    # 3. Read previous M5 behavioral features.
    # --------------------------------------------------------

    old_avg_delay = clean_feature_value(
        row.get(
            "avg_payment_delay",
            0,
        )
    )

    old_on_time_ratio = clean_feature_value(
        row.get(
            "on_time_payment_ratio",
            0,
        )
    )

    old_late_ratio = clean_feature_value(
        row.get(
            "late_payment_ratio",
            0,
        )
    )

    old_avg_payment_ratio = clean_feature_value(
        row.get(
            "avg_payment_ratio",
            0,
        )
    )

    old_recent_delay = clean_feature_value(
        row.get(
            "recent_avg_delay",
            0,
        )
    )

    old_recent_on_time = clean_feature_value(
        row.get(
            "recent_on_time_ratio",
            old_on_time_ratio,
        )
    )

    old_recent_volatility = clean_feature_value(
        row.get(
            "recent_delay_volatility",
            0,
        )
    )

    old_delay_trend = clean_feature_value(
        row.get(
            "delay_trend",
            0,
        )
    )

    old_on_time_trend = clean_feature_value(
        row.get(
            "on_time_trend",
            0,
        )
    )

    # --------------------------------------------------------
    # 4. Incrementally update behavioral features.
    #
    # The latest event gets a stronger weight because this
    # endpoint represents recent behavior.
    # --------------------------------------------------------

    event_weight = 0.25

    new_avg_delay = (
        (1.0 - event_weight)
        * old_avg_delay
        + event_weight
        * delay_days
    )

    new_on_time_ratio = (
        (1.0 - event_weight)
        * old_on_time_ratio
        + event_weight
        * event_on_time
    )

    new_late_ratio = (
        (1.0 - event_weight)
        * old_late_ratio
        + event_weight
        * event_late_ratio
    )

    new_payment_ratio = (
        (1.0 - event_weight)
        * old_avg_payment_ratio
        + event_weight
        * payment_ratio
    )

    new_recent_delay = (
        (1.0 - event_weight)
        * old_recent_delay
        + event_weight
        * delay_days
    )

    new_recent_on_time = (
        (1.0 - event_weight)
        * old_recent_on_time
        + event_weight
        * event_on_time
    )

    # --------------------------------------------------------
    # 5. Update temporal trends.
    # --------------------------------------------------------

    delay_change = (
        delay_days
        - old_recent_delay
    )

    new_delay_trend = (
        0.75 * old_delay_trend
        + 0.25 * delay_change
    )

    on_time_change = (
        event_on_time
        - old_recent_on_time
    )

    new_on_time_trend = (
        0.75 * old_on_time_trend
        + 0.25 * on_time_change
    )

    new_volatility = (
        0.75 * old_recent_volatility
        + 0.25 * abs(delay_change)
    )

    # --------------------------------------------------------
    # 6. Write updated features into a temporary applicant
    #    representation.
    #
    # IMPORTANT:
    # We do not modify the parquet source.
    # --------------------------------------------------------

    row["avg_payment_delay"] = (
        new_avg_delay
    )

    row["on_time_payment_ratio"] = (
        new_on_time_ratio
    )

    row["late_payment_ratio"] = (
        new_late_ratio
    )

    row["avg_payment_ratio"] = (
        new_payment_ratio
    )

    row["recent_avg_delay"] = (
        new_recent_delay
    )

    row["recent_on_time_ratio"] = (
        new_recent_on_time
    )

    row["recent_delay_volatility"] = (
        new_volatility
    )

    row["delay_trend"] = (
        new_delay_trend
    )

    row["on_time_trend"] = (
        new_on_time_trend
    )

    # --------------------------------------------------------
    # 7. Re-run the ACTUAL M5 LightGBM model.
    # --------------------------------------------------------

    x_updated = build_feature_frame(
        row,
        risk_features,
    )

    updated_risk = float(
        risk_model.predict_proba(
            x_updated
        )[:, 1][0]
    )

    # --------------------------------------------------------
    # 8. Re-run integrity independently.
    # --------------------------------------------------------

    integrity_status = calculate_integrity(
        x_updated,
        integrity_model,
        integrity_features,
    )

    # --------------------------------------------------------
    # 9. Recalculate evidence confidence.
    #
    # A new verified event is additional evidence.
    # Negative behavior affects risk, not evidence quality.
    # --------------------------------------------------------

    base_confidence = confidence_for(
        row
    )

    event_evidence_bonus = 2.0

    updated_confidence = min(
        base_confidence
        + event_evidence_bonus,
        100.0,
    )

    # --------------------------------------------------------
    # 10. Re-run deterministic policy.
    # --------------------------------------------------------

    decision = policy(
        updated_risk,
        updated_confidence,
        integrity_status,
    )

    # --------------------------------------------------------
    # 11. Determine behavioral trend.
    # --------------------------------------------------------

    trend = (
        "DETERIORATING"
        if (
            delay_days > 0
            or payment_ratio < 0.8
        )
        else "STABLE"
    )

    # --------------------------------------------------------
    # 12. Return complete live-update response.
    # --------------------------------------------------------

    return {
        "applicantId": applicant_id,

        "riskScore": round(
            updated_risk,
            4,
        ),

        "confidence": round(
            updated_confidence,
            1,
        ),

        "integrityStatus": integrity_status,

        "decision": decision,

        "mode": "LIVE_BEHAVIOR_UPDATE",

        "modelVersion": "m5-risk-model-v1",

        "policyVersion": "credit-policy-v2",

        "policyThresholds": {
            "approveBelow": APPROVE_THRESHOLD,
            "declineAtOrAbove": DECLINE_THRESHOLD,
            "minimumConfidence": MIN_CONFIDENCE,
            "unusualIntegrityAction": "REFER",
        },

        "trend": trend,

        "event": event.model_dump(),

        "updatedFeatures": {
            "avgPaymentDelay": round(
                new_avg_delay,
                4,
            ),

            "onTimePaymentRatio": round(
                new_on_time_ratio,
                4,
            ),

            "latePaymentRatio": round(
                new_late_ratio,
                4,
            ),

            "avgPaymentRatio": round(
                new_payment_ratio,
                4,
            ),

            "recentAvgDelay": round(
                new_recent_delay,
                4,
            ),

            "recentOnTimeRatio": round(
                new_recent_on_time,
                4,
            ),

            "recentDelayVolatility": round(
                new_volatility,
                4,
            ),

            "delayTrend": round(
                new_delay_trend,
                4,
            ),

            "onTimeTrend": round(
                new_on_time_trend,
                4,
            ),
        },
    }