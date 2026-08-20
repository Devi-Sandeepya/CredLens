from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "artifacts" / "features"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
METRIC_DIR = BASE_DIR / "artifacts" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

ID = "SK_ID_CURR"
TARGET = "TARGET"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# Same configuration is deliberately used for M1 → M5.
#
# IMPORTANT:
# We are preserving the current baseline configuration.
# Do not tune this yet.
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}


# ============================================================
# INTEGRITY FEATURES
# ============================================================

INTEGRITY_FEATURES = [
    "credit_to_income",
    "annuity_to_income",
    "employment_years",
    "bureau_account_count",
    "bureau_total_credit",
    "bureau_total_debt",
    "bureau_total_overdue",
    "avg_payment_ratio",
    "max_payment_delay",
    "previous_application_count",
    "delay_trend",
    "on_time_trend",
]


# ============================================================
# FEATURE SET DEFINITIONS
# ============================================================

M1_EXCLUDE = {
    "SK_ID_CURR",
    "TARGET",
}


M1_PREFIXES = {
    "AMT_",
    "CNT_",
    "REGION_",
    "DAYS_",
    "OWN_CAR_AGE",
    "EXT_SOURCE_",
    "credit_to_income",
    "annuity_to_income",
    "age_years",
    "employment_years",
    "ext_source_mean",
}


def get_feature_columns(
    df: pd.DataFrame,
    model_name: str,
) -> list[str]:
    """
    Construct the feature set for M1 → M5.

    M1:
        Application + EXT_SOURCE

    M2:
        M1 + Bureau

    M3:
        M2 + Repayment

    M4:
        M3 + Previous Application / Behavioral

    M5:
        M4 + Temporal
    """

    excluded = {ID, TARGET}

    all_features = [
        c for c in df.columns
        if c not in excluded
    ]

    if model_name == "M5":
        return all_features

    bureau_features = {
        "bureau_account_count",
        "bureau_active_count",
        "bureau_total_credit",
        "bureau_total_debt",
        "bureau_total_overdue",
        "bureau_max_overdue_days",
        "bureau_history_depth",
        "bureau_delinquent_months",
        "bureau_month_count",
        "bureau_debt_to_credit",
        "bureau_delinquency_ratio",
    }

    repayment_features = {
        "installment_count",
        "on_time_payment_ratio",
        "late_payment_ratio",
        "avg_payment_delay",
        "max_payment_delay",
        "avg_payment_ratio",
        "payment_ratio_volatility",
    }

    behavioral_features = {
        "previous_application_count",
        "previous_approval_ratio",
        "previous_refusal_ratio",
        "previous_avg_requested",
        "previous_avg_credit",
        "previous_credit_to_requested_ratio",
        "previous_avg_term",
    }

    temporal_features = {
        "recent_avg_delay",
        "recent_on_time_ratio",
        "recent_delay_volatility",
        "historical_avg_delay",
        "historical_on_time_ratio",
        "delay_trend",
        "on_time_trend",
    }

    if model_name == "M1":
        remove = (
            bureau_features
            | repayment_features
            | behavioral_features
            | temporal_features
        )

    elif model_name == "M2":
        remove = (
            repayment_features
            | behavioral_features
            | temporal_features
        )

    elif model_name == "M3":
        remove = (
            behavioral_features
            | temporal_features
        )

    elif model_name == "M4":
        remove = temporal_features

    else:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    return [
        c for c in all_features
        if c not in remove
    ]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:

    X = df[feature_cols].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    return X


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Create the official development holdout.

    IMPORTANT:
    The returned test set is NEVER used to fit the model.

    The exact applicant IDs are returned so that downstream
    evaluation can reproduce the same holdout without ever
    evaluating on the complete development dataset.
    """

    X = clean_features(
        df,
        feature_cols,
    )

    y = df[TARGET].astype(int)

    ids = df[ID].astype(int)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test,
    ) = train_test_split(
        X,
        y,
        ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test,
    )


# ============================================================
# FIXED APPROVAL RATE METRIC
# ============================================================

def fixed_approval_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    approval_rate: float = 0.80,
) -> dict:

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    )

    n = len(probabilities)

    approval_count = int(
        np.floor(
            n * approval_rate
        )
    )

    # Lowest predicted-risk applicants are approved.
    order = np.argsort(
        probabilities
    )

    approved_idx = order[
        :approval_count
    ]

    rejected_idx = order[
        approval_count:
    ]

    approved_default_rate = (
        float(
            y_true_array[
                approved_idx
            ].mean()
        )
        if len(approved_idx)
        else 0.0
    )

    rejected_default_rate = (
        float(
            y_true_array[
                rejected_idx
            ].mean()
        )
        if len(rejected_idx)
        else 0.0
    )

    approved_default_capture = (
        float(
            y_true_array[
                approved_idx
            ].sum()
        )
        /
        float(
            y_true_array.sum()
        )
        if y_true_array.sum() > 0
        else 0.0
    )

    return {
        "approval_rate": approval_rate,
        "approved_default_rate": approved_default_rate,
        "rejected_default_rate": rejected_default_rate,
        "approved_default_capture": approved_default_capture,
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> dict:

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    calibration_fraction, calibration_mean = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    fixed_70 = fixed_approval_metrics(
        y_test,
        probabilities,
        approval_rate=0.70,
    )

    fixed_80 = fixed_approval_metrics(
        y_test,
        probabilities,
        approval_rate=0.80,
    )

    fixed_90 = fixed_approval_metrics(
        y_test,
        probabilities,
        approval_rate=0.90,
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),

        "calibration": {
            "fraction_of_positives": [
                float(x)
                for x in calibration_fraction
            ],
            "mean_predicted_value": [
                float(x)
                for x in calibration_mean
            ],
        },

        "fixed_approval_70": fixed_70,
        "fixed_approval_80": fixed_80,
        "fixed_approval_90": fixed_90,
    }


# ============================================================
# TRAIN ONE LIGHTGBM MODEL
# ============================================================

def train_one_model(
    model_name: str,
    df: pd.DataFrame,
) -> tuple[LGBMClassifier, dict]:

    print("\n" + "=" * 72)
    print(f"TRAINING {model_name}")
    print("=" * 72)

    feature_cols = get_feature_columns(
        df,
        model_name,
    )

    print(
        f"Features: {len(feature_cols)}"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test,
    ) = split_data(
        df,
        feature_cols,
    )

    print(
        f"Training applicants: {len(X_train):,}"
    )

    print(
        f"Holdout applicants:  {len(X_test):,}"
    )

    print(
        f"Training defaults:   {int(y_train.sum()):,}"
    )

    print(
        f"Holdout defaults:    {int(y_test.sum()):,}"
    )

    model = LGBMClassifier(
        **LIGHTGBM_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Predictions below are generated ONLY on the held-out
    # 20% that was not used during model fitting.
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_model(
        y_test,
        probabilities,
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC : {metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier  : {metrics['brier_score']:.4f}"
    )

    print(
        "70% approval → "
        f"approved default rate: "
        f"{metrics['fixed_approval_70']['approved_default_rate']:.4f}"
    )

    print(
        "80% approval → "
        f"approved default rate: "
        f"{metrics['fixed_approval_80']['approved_default_rate']:.4f}"
    )

    print(
        "90% approval → "
        f"approved default rate: "
        f"{metrics['fixed_approval_90']['approved_default_rate']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name.lower()}_risk_model.joblib"
    )

    feature_path = (
        MODEL_DIR
        / f"{model_name.lower()}_risk_features.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    feature_path.write_text(
        json.dumps(
            feature_cols,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # SAVE EXACT HOLDOUT PREDICTIONS
    #
    # This is the important fix.
    #
    # evaluate.py must use this artifact instead of loading
    # all 30,000 applicants and predicting on them again.
    # --------------------------------------------------------

    holdout_path = (
        METRIC_DIR
        / f"{model_name.lower()}_holdout_predictions.parquet"
    )

    holdout_predictions = pd.DataFrame(
        {
            ID: id_test.to_numpy(),
            TARGET: y_test.to_numpy(),
            "risk_probability": probabilities,
        }
    )

    holdout_predictions.to_parquet(
        holdout_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE HOLDOUT IDS
    #
    # Useful for reproducibility and auditability.
    # --------------------------------------------------------

    holdout_ids_path = (
        METRIC_DIR
        / f"{model_name.lower()}_holdout_ids.json"
    )

    holdout_ids_path.write_text(
        json.dumps(
            [int(x) for x in id_test.tolist()],
            indent=2,
        )
    )

    print(
        f"Saved model → {model_path}"
    )

    print(
        f"Saved holdout predictions → "
        f"{holdout_path}"
    )

    print(
        f"Saved holdout IDs → "
        f"{holdout_ids_path}"
    )

    return model, {
        "model": model_name,
        "feature_count": len(feature_cols),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "train_default_count": int(y_train.sum()),
        "holdout_default_count": int(y_test.sum()),
        **metrics,
    }


# ============================================================
# INTEGRITY MODEL
# ============================================================

def train_integrity_model(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 72)
    print("TRAINING INTEGRITY MODEL")
    print("=" * 72)

    available = [
        c
        for c in INTEGRITY_FEATURES
        if c in df.columns
    ]

    missing = [
        c
        for c in INTEGRITY_FEATURES
        if c not in df.columns
    ]

    print(
        f"Integrity features available: "
        f"{len(available)}"
    )

    if missing:
        print(
            "Integrity features not present: "
            f"{missing}"
        )

    if not available:
        raise ValueError(
            "No valid integrity features found."
        )

    X = clean_features(
        df,
        available,
    )

    integrity = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    integrity.fit(X)

    model_path = (
        MODEL_DIR
        / "integrity_model.joblib"
    )

    feature_path = (
        MODEL_DIR
        / "integrity_features.json"
    )

    joblib.dump(
        integrity,
        model_path,
    )

    feature_path.write_text(
        json.dumps(
            available,
            indent=2,
        )
    )

    print(
        f"Saved integrity model → {model_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CREDLENS — P0 RISK MODEL EXPERIMENT")
    print("=" * 72)

    results = []

    # --------------------------------------------------------
    # Train M1 → M5
    # --------------------------------------------------------

    for model_name in [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]:

        feature_path = (
            FEATURE_DIR
            / f"{model_name.lower()}_features.parquet"
        )

        print(
            f"\nLoading {model_name}: "
            f"{feature_path}"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing feature file: "
                f"{feature_path}"
            )

        df = pd.read_parquet(
            feature_path
        )

        if ID not in df.columns:
            raise ValueError(
                f"{model_name} missing {ID}"
            )

        if TARGET not in df.columns:
            raise ValueError(
                f"{model_name} missing {TARGET}"
            )

        _, metrics = train_one_model(
            model_name,
            df,
        )

        results.append(
            metrics
        )

    # --------------------------------------------------------
    # Save ablation results
    # --------------------------------------------------------

    results_path = (
        METRIC_DIR
        / "m1_m5_ablation.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Compact CSV
    # --------------------------------------------------------

    compact_rows = []

    for result in results:

        compact_rows.append(
            {
                "model": result["model"],
                "feature_count": result[
                    "feature_count"
                ],
                "train_rows": result[
                    "train_rows"
                ],
                "holdout_rows": result[
                    "holdout_rows"
                ],
                "train_default_count": result[
                    "train_default_count"
                ],
                "holdout_default_count": result[
                    "holdout_default_count"
                ],
                "roc_auc": result[
                    "roc_auc"
                ],
                "pr_auc": result[
                    "pr_auc"
                ],
                "brier_score": result[
                    "brier_score"
                ],
                "approval_70_default_rate": result[
                    "fixed_approval_70"
                ][
                    "approved_default_rate"
                ],
                "approval_80_default_rate": result[
                    "fixed_approval_80"
                ][
                    "approved_default_rate"
                ],
                "approval_90_default_rate": result[
                    "fixed_approval_90"
                ][
                    "approved_default_rate"
                ],
            }
        )

    pd.DataFrame(
        compact_rows
    ).to_csv(
        METRIC_DIR
        / "m1_m5_ablation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Train separate integrity model using M5 features.
    # --------------------------------------------------------

    m5_path = (
        FEATURE_DIR
        / "m5_features.parquet"
    )

    m5_df = pd.read_parquet(
        m5_path
    )

    train_integrity_model(
        m5_df
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("M1 → M5 ABLATION SUMMARY")
    print("=" * 72)

    for result in results:

        print(
            f"{result['model']}: "
            f"{result['feature_count']} features | "
            f"ROC-AUC={result['roc_auc']:.4f} | "
            f"PR-AUC={result['pr_auc']:.4f} | "
            f"Brier={result['brier_score']:.4f}"
        )

    print(
        f"\nSaved ablation results → "
        f"{results_path}"
    )

    print(
        "\nIMPORTANT:"
        "\n  Holdout predictions are saved separately."
        "\n  Downstream evaluation MUST use those holdout"
        "\n  predictions and MUST NOT predict on the full"
        "\n  30,000-applicant development cohort."
    )

    print(
        "\nP0 risk + integrity training complete."
    )


if __name__ == "__main__":
    main()