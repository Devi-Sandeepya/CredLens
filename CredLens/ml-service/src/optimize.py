from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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

FEATURE_PATH = (
    BASE_DIR
    / "artifacts"
    / "features"
    / "m5_features.parquet"
)

METRIC_DIR = (
    BASE_DIR
    / "artifacts"
    / "metrics"
)

MODEL_DIR = (
    BASE_DIR
    / "artifacts"
    / "models"
)

METRIC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

ID = "SK_ID_CURR"
TARGET = "TARGET"

RANDOM_STATE = 42

FINAL_HOLDOUT_SIZE = 0.20
VALIDATION_SIZE = 0.20


# ============================================================
# CANDIDATE CONFIGURATIONS
#
# Deliberately small search.
# We are testing meaningful model changes,
# not blindly searching hundreds of configurations.
# ============================================================

CANDIDATES = {

    "baseline": {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    },

    "regularized": {
        "n_estimators": 400,
        "learning_rate": 0.035,
        "num_leaves": 24,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 0.20,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    },

    "balanced_capacity": {
        "n_estimators": 400,
        "learning_rate": 0.035,
        "num_leaves": 40,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.10,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    },

    "conservative": {
        "n_estimators": 500,
        "learning_rate": 0.025,
        "num_leaves": 20,
        "min_child_samples": 40,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 0.30,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    },
}


# ============================================================
# DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not FEATURE_PATH.exists():
        raise RuntimeError(
            f"M5 feature table not found:\n{FEATURE_PATH}"
        )

    df = pd.read_parquet(FEATURE_PATH)

    if ID not in df.columns:
        raise RuntimeError(
            f"Missing required ID column: {ID}"
        )

    if TARGET not in df.columns:
        raise RuntimeError(
            f"Missing required target column: {TARGET}"
        )

    return df


def clean_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:

    X = df[feature_cols].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X.fillna(0)


# ============================================================
# EXACT FINAL HOLDOUT
#
# This MUST reproduce the same 6,000 applicants used by
# train.py Baseline v1.
# ============================================================

def create_final_holdout(
    df: pd.DataFrame,
    feature_cols: list[str],
):

    X = clean_features(
        df,
        feature_cols,
    )

    y = df[TARGET].astype(int)

    ids = df[ID].astype(int)

    (
        X_development,
        X_final_holdout,
        y_development,
        y_final_holdout,
        id_development,
        id_final_holdout,
    ) = train_test_split(
        X,
        y,
        ids,
        test_size=FINAL_HOLDOUT_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return (
        X_development,
        X_final_holdout,
        y_development,
        y_final_holdout,
        id_development,
        id_final_holdout,
    )


# ============================================================
# INTERNAL VALIDATION SPLIT
#
# This split is used ONLY for candidate selection.
# The final 6,000 remain untouched.
# ============================================================

def create_internal_validation(
    X_development: pd.DataFrame,
    y_development: pd.Series,
):

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================
# MODEL SELECTION SCORE
#
# PR-AUC is primary because defaults are rare.
# ROC-AUC and Brier remain important guardrails.
# ============================================================

def selection_score(
    metrics: dict,
) -> float:

    return (
        0.55 * metrics["pr_auc"]
        + 0.30 * metrics["roc_auc"]
        - 0.15 * metrics["brier_score"]
    )


# ============================================================
# MAIN OPTIMIZATION EXPERIMENT
# ============================================================

def main():

    print("=" * 72)
    print("CREDLENS — M5 MODEL OPTIMIZATION v1")
    print("=" * 72)

    df = load_data()

    feature_cols = [
        c
        for c in df.columns
        if c not in [ID, TARGET]
    ]

    print(
        f"\nApplicants: {len(df):,}"
    )

    print(
        f"M5 features: {len(feature_cols)}"
    )

    # --------------------------------------------------------
    # Recreate the exact final holdout.
    # --------------------------------------------------------

    (
        X_development,
        X_final_holdout,
        y_development,
        y_final_holdout,
        id_development,
        id_final_holdout,
    ) = create_final_holdout(
        df,
        feature_cols,
    )

    print(
        f"\nDevelopment applicants: "
        f"{len(X_development):,}"
    )

    print(
        f"FINAL HOLDOUT applicants: "
        f"{len(X_final_holdout):,}"
    )

    print(
        f"FINAL HOLDOUT defaults: "
        f"{int(y_final_holdout.sum()):,}"
    )

    # --------------------------------------------------------
    # Verify against the existing official holdout.
    # --------------------------------------------------------

    official_ids_path = (
        METRIC_DIR
        / "m5_holdout_ids.json"
    )

    if official_ids_path.exists():

        official_ids = set(
            json.loads(
                official_ids_path.read_text()
            )
        )

        recreated_ids = set(
            id_final_holdout.astype(int)
        )

        if recreated_ids != official_ids:

            raise RuntimeError(
                "FINAL HOLDOUT MISMATCH.\n"
                "Optimization aborted because the recreated "
                "holdout does not match Baseline v1."
            )

        print(
            "\nFinal holdout verification: PASSED"
        )

        print(
            "Exactly the same 6,000 applicants "
            "as Baseline v1."
        )

    else:

        raise RuntimeError(
            "Official M5 holdout ID artifact is missing:\n"
            f"{official_ids_path}"
        )

    # --------------------------------------------------------
    # Internal train / validation split.
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = create_internal_validation(
        X_development,
        y_development,
    )

    print(
        f"\nInternal training applicants: "
        f"{len(X_train):,}"
    )

    print(
        f"Internal validation applicants: "
        f"{len(X_validation):,}"
    )

    # --------------------------------------------------------
    # Candidate experiments.
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 72)
    print("INTERNAL MODEL SELECTION")
    print("=" * 72)

    for name, params in CANDIDATES.items():

        print("\n" + "-" * 72)
        print(f"CANDIDATE: {name}")
        print("-" * 72)

        model = LGBMClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = evaluate_predictions(
            y_validation,
            probabilities,
        )

        score = selection_score(
            metrics
        )

        result = {
            "model": name,
            "features": len(feature_cols),
            "train_rows": len(X_train),
            "validation_rows": len(X_validation),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "selection_score": score,
            "params": params,
        }

        results.append(result)

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
            f"Selection score: {score:.6f}"
        )

    # --------------------------------------------------------
    # Select candidate.
    # --------------------------------------------------------

    results_sorted = sorted(
        results,
        key=lambda x: x["selection_score"],
        reverse=True,
    )

    best = results_sorted[0]

    print("\n" + "=" * 72)
    print("MODEL SELECTION RESULT")
    print("=" * 72)

    print(
        f"Selected candidate: {best['model']}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{best['roc_auc']:.4f}"
    )

    print(
        f"Validation PR-AUC:  "
        f"{best['pr_auc']:.4f}"
    )

    print(
        f"Validation Brier:   "
        f"{best['brier_score']:.4f}"
    )

    # --------------------------------------------------------
    # Save experiment results.
    # --------------------------------------------------------

    results_path = (
        METRIC_DIR
        / "m5_optimization_v1.json"
    )

    results_path.write_text(
        json.dumps(
            {
                "experiment": (
                    "CredLens M5 Model Optimization v1"
                ),
                "selection_method": (
                    "internal validation only"
                ),
                "final_holdout_rows": len(
                    X_final_holdout
                ),
                "final_holdout_used_for_selection": False,
                "selected_model": best["model"],
                "results": results_sorted,
            },
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Retrain selected candidate on ALL 24,000 development
    # applicants.
    #
    # FINAL 6,000 STILL REMAIN UNTOUCHED.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("RETRAINING SELECTED CANDIDATE")
    print("=" * 72)

    selected_params = best["params"]

    final_candidate = LGBMClassifier(
        **selected_params
    )

    final_candidate.fit(
        X_development,
        y_development,
    )

    candidate_model_path = (
        MODEL_DIR
        / "m5_candidate_risk_model.joblib"
    )

    candidate_features_path = (
        MODEL_DIR
        / "m5_candidate_risk_features.json"
    )

    joblib.dump(
        final_candidate,
        candidate_model_path,
    )

    candidate_features_path.write_text(
        json.dumps(
            feature_cols,
            indent=2,
        )
    )

    print(
        f"Saved candidate model -> "
        f"{candidate_model_path}"
    )

    print(
        f"Saved candidate features -> "
        f"{candidate_features_path}"
    )

    print(
        f"Saved optimization report -> "
        f"{results_path}"
    )

    print("\n" + "=" * 72)
    print("OPTIMIZATION COMPLETE")
    print("=" * 72)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The 6,000-applicant final holdout "
        "was NOT used for model selection."
    )

    print(
        "Do NOT evaluate the candidate on the "
        "final holdout until the candidate is frozen."
    )


if __name__ == "__main__":
    main()