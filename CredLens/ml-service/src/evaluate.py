from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from .config import (
    APPLICANT_FEATURES_PATH,
    RISK_MODEL_PATH,
    RISK_FEATURES_PATH,
    INTEGRITY_MODEL_PATH,
    INTEGRITY_FEATURES_PATH,
)


ID = "SK_ID_CURR"
TARGET = "TARGET"
PREDICTION = "risk_probability"


# ============================================================
# HOLDOUT ARTIFACT PATHS
# ============================================================

HOLDOUT_PREDICTIONS_PATH = (
    APPLICANT_FEATURES_PATH.parent.parent
    / "metrics"
    / "m5_holdout_predictions.parquet"
)

HOLDOUT_IDS_PATH = (
    APPLICANT_FEATURES_PATH.parent.parent
    / "metrics"
    / "m5_holdout_ids.json"
)


# ============================================================
# DATA
# ============================================================

def load_full_data() -> pd.DataFrame:
    """
    Load the feature table ONLY to recover actual feature values
    and TARGET labels for the saved holdout applicants.

    IMPORTANT:
    We never generate predictions on this full dataset.
    """

    if not APPLICANT_FEATURES_PATH.exists():
        raise RuntimeError(
            "M5 feature table not found. "
            "Run: python -m src.pipeline"
        )

    return pd.read_parquet(APPLICANT_FEATURES_PATH)


def load_holdout_predictions() -> pd.DataFrame:
    """
    Load predictions generated during training.

    These predictions were produced only on the 20% holdout
    that was never used to fit the corresponding LightGBM model.
    """

    if not HOLDOUT_PREDICTIONS_PATH.exists():
        raise RuntimeError(
            "M5 holdout predictions not found.\n"
            "Run: python -m src.train"
        )

    df = pd.read_parquet(HOLDOUT_PREDICTIONS_PATH)

    required = {
        ID,
        TARGET,
        PREDICTION,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Holdout prediction artifact is missing columns: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Strong artifact validation
    # --------------------------------------------------------

    if df[ID].duplicated().any():
        raise RuntimeError(
            "Duplicate applicant IDs found in holdout prediction artifact."
        )

    if len(df) == 0:
        raise RuntimeError(
            "Holdout prediction artifact is empty."
        )

    probabilities = pd.to_numeric(
        df[PREDICTION],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise RuntimeError(
            "Holdout prediction artifact contains "
            "non-numeric or NaN risk probabilities."
        )

    if not np.isfinite(probabilities.to_numpy()).all():
        raise RuntimeError(
            "Holdout prediction artifact contains "
            "infinite risk probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Holdout risk probabilities must be between 0 and 1."
        )

    return df


def load_holdout_ids() -> list[int]:
    """
    Load the exact applicant IDs used in the M5 holdout.
    """

    if not HOLDOUT_IDS_PATH.exists():
        raise RuntimeError(
            "M5 holdout ID artifact not found.\n"
            "Run: python -m src.train"
        )

    ids = json.loads(
        HOLDOUT_IDS_PATH.read_text()
    )

    ids = [int(x) for x in ids]

    if len(ids) == 0:
        raise RuntimeError(
            "Holdout ID artifact is empty."
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate applicant IDs found in holdout ID artifact."
        )

    return ids


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not RISK_MODEL_PATH.exists():
        raise RuntimeError(
            "M5 risk model not found. "
            "Run: python -m src.train"
        )

    if not RISK_FEATURES_PATH.exists():
        raise RuntimeError(
            "M5 feature list not found."
        )

    model = joblib.load(
        RISK_MODEL_PATH
    )

    features = json.loads(
        RISK_FEATURES_PATH.read_text()
    )

    return model, features


# ============================================================
# INTEGRITY
# ============================================================

def load_integrity():

    if not INTEGRITY_MODEL_PATH.exists():
        return None, []

    if not INTEGRITY_FEATURES_PATH.exists():
        return None, []

    model = joblib.load(
        INTEGRITY_MODEL_PATH
    )

    features = json.loads(
        INTEGRITY_FEATURES_PATH.read_text()
    )

    return model, features


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    X = (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return X


# ============================================================
# POLICY
# ============================================================

def policy(
    risk: float,
    confidence: float,
    integrity: str,
) -> str:

    if integrity == "UNUSUAL":
        return "REFER"

    if (
        risk < 0.25
        and confidence >= 60
    ):
        return "APPROVE"

    if (
        risk >= 0.55
        and confidence >= 60
    ):
        return "DECLINE"

    return "REFER"


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_for(
    row: pd.Series,
) -> float:

    bureau_count = max(
        float(
            row.get(
                "bureau_account_count",
                0,
            )
        ),
        0.0,
    )

    installment_count = max(
        float(
            row.get(
                "installment_count",
                0,
            )
        ),
        0.0,
    )

    bureau_history = min(
        bureau_count / 10.0,
        1.0,
    )

    repayment_history = min(
        installment_count / 50.0,
        1.0,
    )

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
        c
        for c in evidence_columns
        if c in row.index
    ]

    if available:

        coverage = float(
            sum(
                pd.notna(row[c])
                and float(row[c]) != 0
                for c in available
            )
            / len(available)
        )

    else:

        coverage = 0.0

    completeness = float(
        1.0 - row.isna().mean()
    )

    confidence = (
        0.30 * bureau_history
        + 0.25 * repayment_history
        + 0.25 * coverage
        + 0.20 * completeness
    )

    return round(
        100.0 * confidence,
        1,
    )


# ============================================================
# APPROVAL BAND ANALYSIS
# ============================================================

def approval_band(
    y_true: pd.Series,
    probabilities: np.ndarray,
    approval_rate: float,
) -> dict:

    n = len(probabilities)

    approved_count = int(
        n * approval_rate
    )

    order = np.argsort(
        probabilities
    )

    approved_idx = order[
        :approved_count
    ]

    approved_y = y_true.iloc[
        approved_idx
    ]

    default_rate = float(
        approved_y.mean()
    )

    return {
        "targetApprovalRate": approval_rate,
        "actualApprovedCount": int(
            approved_count
        ),
        "actualApprovalRate": round(
            approved_count / n,
            4,
        ),
        "approvedDefaultRate": round(
            default_rate,
            4,
        ),
        "approvedDefaults": int(
            approved_y.sum()
        ),
    }


# ============================================================
# RISK CAPTURE
# ============================================================

def risk_capture(
    y_true: pd.Series,
    probabilities: np.ndarray,
    percentile: float,
) -> dict:

    threshold = float(
        np.quantile(
            probabilities,
            percentile,
        )
    )

    predicted_high_risk = (
        probabilities >= threshold
    )

    actual_defaults = (
        y_true.to_numpy() == 1
    )

    captured = int(
        np.sum(
            predicted_high_risk
            & actual_defaults
        )
    )

    total_defaults = int(
        actual_defaults.sum()
    )

    capture_rate = (
        captured / total_defaults
        if total_defaults
        else 0.0
    )

    return {
        "threshold": round(
            threshold,
            6,
        ),
        "highRiskApplicants": int(
            predicted_high_risk.sum()
        ),
        "capturedDefaults": captured,
        "totalDefaults": total_defaults,
        "defaultCaptureRate": round(
            capture_rate,
            4,
        ),
    }


# ============================================================
# POLICY DISTRIBUTION
# ============================================================

def policy_distribution(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict:

    decisions = []

    integrity_model, integrity_features = (
        load_integrity()
    )

    if (
        integrity_model is not None
        and integrity_features
    ):

        available = [
            c
            for c in integrity_features
            if c in df.columns
        ]

        integrity_input = (
            df[available]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

        integrity_predictions = (
            integrity_model.predict(
                integrity_input
            )
        )

    else:

        integrity_predictions = np.ones(
            len(df),
            dtype=int,
        )

    for i, probability in enumerate(
        probabilities
    ):

        row = df.iloc[i]

        confidence = confidence_for(
            row
        )

        integrity = (
            "UNUSUAL"
            if int(
                integrity_predictions[i]
            ) == -1
            else "NORMAL"
        )

        decisions.append(
            policy(
                float(probability),
                confidence,
                integrity,
            )
        )

    counts = pd.Series(
        decisions
    ).value_counts()

    total = len(
        decisions
    )

    return {
        "APPROVE": {
            "count": int(
                counts.get(
                    "APPROVE",
                    0,
                )
            ),
            "rate": round(
                counts.get(
                    "APPROVE",
                    0,
                ) / total,
                4,
            ),
        },
        "REFER": {
            "count": int(
                counts.get(
                    "REFER",
                    0,
                )
            ),
            "rate": round(
                counts.get(
                    "REFER",
                    0,
                ) / total,
                4,
            ),
        },
        "DECLINE": {
            "count": int(
                counts.get(
                    "DECLINE",
                    0,
                )
            ),
            "rate": round(
                counts.get(
                    "DECLINE",
                    0,
                ) / total,
                4,
            ),
        },
    }


# ============================================================
# INTEGRITY DISTRIBUTION
# ============================================================

def integrity_distribution(
    df: pd.DataFrame,
) -> dict:

    model, features = load_integrity()

    if (
        model is None
        or not features
    ):

        return {
            "available": False,
            "NORMAL": len(df),
            "UNUSUAL": 0,
            "unusualRate": 0.0,
        }

    available = [
        c
        for c in features
        if c in df.columns
    ]

    if not available:
        return {
            "available": False,
            "NORMAL": len(df),
            "UNUSUAL": 0,
            "unusualRate": 0.0,
        }

    X = (
        df[available]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    predictions = model.predict(
        X
    )

    unusual = int(
        np.sum(
            predictions == -1
        )
    )

    total = len(
        predictions
    )

    return {
        "available": True,
        "NORMAL": int(
            total - unusual
        ),
        "UNUSUAL": unusual,
        "unusualRate": round(
            unusual / total,
            4,
        ),
    }


# ============================================================
# HOLDOUT ALIGNMENT
# ============================================================

def build_holdout_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the evaluation dataset by joining the saved M5
    holdout predictions with the corresponding applicant
    feature rows.

    NO prediction is generated here.

    This is critical: all model-quality metrics use the
    predictions saved by src.train on the untouched holdout.
    """

    full_df = load_full_data()
    holdout_predictions = load_holdout_predictions()
    holdout_ids = load_holdout_ids()

    # --------------------------------------------------------
    # Validate holdout artifact
    # --------------------------------------------------------

    if len(holdout_predictions) != len(holdout_ids):
        raise RuntimeError(
            "Holdout prediction count does not match "
            "holdout ID count."
        )

    prediction_ids = set(
        holdout_predictions[ID].astype(int)
    )

    id_file_ids = set(
        holdout_ids
    )

    if prediction_ids != id_file_ids:
        raise RuntimeError(
            "Holdout prediction IDs do not exactly match "
            "the saved holdout ID list."
        )

    # --------------------------------------------------------
    # Validate against full feature table
    # --------------------------------------------------------

    full_ids = set(
        full_df[ID].astype(int)
    )

    if not id_file_ids.issubset(full_ids):
        missing = sorted(
            id_file_ids - full_ids
        )[:10]

        raise RuntimeError(
            "Some holdout applicants are missing from "
            f"the M5 feature table. Examples: {missing}"
        )

    # --------------------------------------------------------
    # Select ONLY holdout rows
    # --------------------------------------------------------

    holdout_df = (
        full_df[
            full_df[ID].isin(
                id_file_ids
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Ensure exact one-to-one applicant alignment
    # --------------------------------------------------------

    if len(holdout_df) != len(holdout_ids):
        raise RuntimeError(
            "Holdout applicant count mismatch."
        )

    if holdout_df[ID].duplicated().any():
        raise RuntimeError(
            "Duplicate applicant IDs found in M5 "
            "feature table."
        )

    holdout_df = (
        holdout_df
        .set_index(ID)
        .loc[
            holdout_predictions[ID].astype(int)
        ]
        .reset_index()
    )

    # --------------------------------------------------------
    # Validate TARGET consistency
    # --------------------------------------------------------

    merged = holdout_df[
        [ID, TARGET]
    ].merge(
        holdout_predictions[
            [ID, TARGET, PREDICTION]
        ],
        on=ID,
        how="inner",
        suffixes=(
            "_features",
            "_prediction_artifact",
        ),
        validate="one_to_one",
    )

    if len(merged) != len(holdout_predictions):
        raise RuntimeError(
            "Holdout prediction alignment failed."
        )

    if not np.array_equal(
        merged[
            f"{TARGET}_features"
        ].astype(int).to_numpy(),
        merged[
            f"{TARGET}_prediction_artifact"
        ].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "TARGET mismatch between feature table "
            "and holdout prediction artifact."
        )

    probabilities = (
        merged[PREDICTION]
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Holdout predictions contain NaN or infinite values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError(
            "Holdout risk probabilities must be between 0 and 1."
        )

    return (
        holdout_df,
        probabilities,
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 72)
    print(
        "CREDLENS — HOLDOUT DECISION QUALITY EVALUATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load the SAVED holdout predictions.
    #
    # We intentionally do NOT call model.predict_proba().
    # --------------------------------------------------------

    df, probabilities = (
        build_holdout_dataset()
    )

    # Load model only to report its feature configuration.
    _, feature_names = load_model()

    y = df[
        TARGET
    ].astype(int)

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    brier = brier_score_loss(
        y,
        probabilities,
    )

    print()
    print(
        "CORE MODEL PERFORMANCE — TRUE HOLDOUT"
    )
    print("-" * 72)

    print(
        "Evaluation applicants:",
        len(df),
    )

    print(
        "Training applicants:",
        24000,
    )

    print(
        "Holdout applicants:",
        len(df),
    )

    print(
        "Holdout defaults:",
        int(y.sum()),
    )

    print(
        "Holdout default rate:",
        round(
            float(y.mean()),
            4,
        ),
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc,
            4,
        ),
    )

    print(
        "PR-AUC:",
        round(
            pr_auc,
            4,
        ),
    )

    print(
        "Brier:",
        round(
            brier,
            4,
        ),
    )

    # --------------------------------------------------------
    # Approval quality
    # --------------------------------------------------------

    print()
    print(
        "APPROVAL QUALITY — HOLDOUT"
    )
    print("-" * 72)

    approval_results = []

    for rate in (
        0.70,
        0.80,
        0.90,
    ):

        result = approval_band(
            y.reset_index(drop=True),
            probabilities,
            rate,
        )

        approval_results.append(
            result
        )

        print(
            f"{int(rate * 100)}% approval -> "
            f"default rate among approved: "
            f"{result['approvedDefaultRate']:.4f}"
        )

    # --------------------------------------------------------
    # Risk capture
    # --------------------------------------------------------

    print()
    print(
        "HIGH-RISK CAPTURE — HOLDOUT"
    )
    print("-" * 72)

    capture_results = []

    for percentile in (
        0.70,
        0.80,
        0.90,
    ):

        result = risk_capture(
            y.reset_index(drop=True),
            probabilities,
            percentile,
        )

        capture_results.append(
            result
        )

        print(
            f"Top {(1 - percentile) * 100:.0f}% "
            f"risk -> "
            f"default capture: "
            f"{result['defaultCaptureRate']:.4f}"
        )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    print()
    print(
        "CREDLENS POLICY DISTRIBUTION — HOLDOUT"
    )
    print("-" * 72)

    policy_results = policy_distribution(
        df.reset_index(drop=True),
        probabilities,
    )

    for decision in (
        "APPROVE",
        "REFER",
        "DECLINE",
    ):

        result = policy_results[
            decision
        ]

        print(
            f"{decision}: "
            f"{result['count']} "
            f"({result['rate']:.2%})"
        )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    print()
    print(
        "INTEGRITY DISTRIBUTION — HOLDOUT"
    )
    print("-" * 72)

    integrity_results = (
        integrity_distribution(
            df.reset_index(drop=True)
        )
    )

    print(
        "Normal:",
        integrity_results["NORMAL"],
    )

    print(
        "Unusual:",
        integrity_results["UNUSUAL"],
    )

    print(
        "Unusual rate:",
        f"{integrity_results['unusualRate']:.2%}",
    )

    # --------------------------------------------------------
    # Confusion matrix
    #
    # This is also computed ONLY on the holdout.
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    print()
    print(
        "THRESHOLD 0.50 CONFUSION MATRIX — HOLDOUT"
    )
    print("-" * 72)

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"TP: {tp}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "model": "m5-risk-model-v1",

        "evaluationProtocol": {
            "type": "TRUE_HOLDOUT",
            "trainingApplicants": 24000,
            "holdoutApplicants": int(
                len(df)
            ),
            "predictionsSource": str(
                HOLDOUT_PREDICTIONS_PATH
            ),
            "predictionColumn": PREDICTION,
            "predictionGeneration": (
                "Predictions generated during "
                "training on the untouched 20% holdout."
            ),
            "fullCohortPredictionUsed": False,
            "leakageCheck": {
                "predictionSource": "saved_training_holdout_artifact",
                "predictionsRegeneratedDuringEvaluation": False,
                "targetConsistencyValidated": True,
                "uniqueApplicantAlignmentValidated": True,
            },
        },

        "dataset": {
            "applicants": int(
                len(df)
            ),
            "defaults": int(
                y.sum()
            ),
            "defaultRate": round(
                float(y.mean()),
                6,
            ),
        },

        "metrics": {
            "rocAuc": round(
                roc_auc,
                6,
            ),
            "prAuc": round(
                pr_auc,
                6,
            ),
            "brier": round(
                brier,
                6,
            ),
        },

        "approvalQuality": approval_results,

        "riskCapture": capture_results,

        "policyDistribution": policy_results,

        "integrityDistribution": integrity_results,

        "confusionMatrix": {
            "threshold": threshold,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },

        "featureGroups": {
            "total": len(feature_names),

            "applicant": sum(
                any(
                    c.startswith(prefix)
                    for prefix in [
                        "AMT_",
                        "CNT_",
                        "REGION_",
                        "DAYS_",
                        "EXT_SOURCE_",
                    ]
                )
                for c in feature_names
            ),

            "bureau": sum(
                c.startswith("bureau_")
                for c in feature_names
            ),

            "repayment": sum(
                any(
                    c.startswith(prefix)
                    for prefix in [
                        "installment_",
                        "payment_",
                        "on_time_",
                        "late_",
                        "avg_payment_",
                        "max_payment_",
                    ]
                )
                for c in feature_names
            ),

            "previousApplications": sum(
                c.startswith("previous_")
                for c in feature_names
            ),

            "temporal": sum(
                c.startswith("recent_")
                or c.startswith("historical_")
                or c in [
                    "delay_trend",
                    "on_time_trend",
                ]
                for c in feature_names
            ),
        },
    }

    output = (
        APPLICANT_FEATURES_PATH
        .parent.parent
        / "metrics"
        / "decision_quality.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print(
        "CREDLENS HOLDOUT EVALUATION COMPLETE"
    )
    print("=" * 72)

    print(
        "Saved report ->",
        output,
    )


if __name__ == "__main__":
    main()