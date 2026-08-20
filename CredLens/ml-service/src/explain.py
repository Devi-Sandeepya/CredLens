from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from .config import (
    RISK_MODEL_PATH,
    RISK_FEATURES_PATH,
)


# ============================================================
# HUMAN-READABLE FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    # Applicant / financial profile
    "AMT_INCOME_TOTAL": "Total income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Loan annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family size",
    "REGION_POPULATION_RELATIVE": "Population density",

    # Raw temporal features
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Employment history",

    # External credit signals
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "ext_source_mean": "Combined external credit score",

    # Derived affordability
    "credit_to_income": "Credit-to-income ratio",
    "annuity_to_income": "Annuity-to-income ratio",
    "age_years": "Age",
    "employment_years": "Employment history",

    # Bureau history
    "bureau_account_count": "Credit history depth",
    "bureau_active_count": "Active credit accounts",
    "bureau_total_credit": "Total bureau credit",
    "bureau_total_debt": "Total existing debt",
    "bureau_total_overdue": "Total overdue amount",
    "bureau_max_overdue_days": "Maximum overdue days",
    "bureau_history_depth": "Bureau history depth",
    "bureau_delinquent_months": "Delinquent months",
    "bureau_month_count": "Bureau record count",
    "bureau_debt_to_credit": "Debt-to-credit ratio",
    "bureau_delinquency_ratio": "Delinquency ratio",

    # Repayment behavior
    "installment_count": "Repayment history length",
    "on_time_payment_ratio": "On-time payment ratio",
    "late_payment_ratio": "Late payment ratio",
    "avg_payment_delay": "Average payment timing",
    "max_payment_delay": "Maximum payment delay",
    "avg_payment_ratio": "Payment-to-installment ratio",
    "payment_ratio_volatility": "Payment consistency",

    # Previous applications
    "previous_application_count": "Previous applications",
    "previous_approval_ratio": "Previous approval rate",
    "previous_refusal_ratio": "Previous refusal rate",
    "previous_avg_requested": "Average requested amount",
    "previous_avg_credit": "Average previous credit",
    "previous_credit_to_requested_ratio": (
        "Previous credit-to-requested ratio"
    ),
    "previous_avg_term": "Average previous loan term",

    # Recent / historical behavior
    "recent_avg_delay": "Recent payment timing",
    "recent_on_time_ratio": "Recent on-time payment ratio",
    "recent_delay_volatility": "Recent payment volatility",
    "historical_avg_delay": "Historical payment timing",
    "historical_on_time_ratio": "Historical on-time payment ratio",
    "delay_trend": "Payment timing trend",
    "on_time_trend": "On-time payment trend",
}


# ============================================================
# MODEL + SHAP EXPLAINER
# ============================================================

def load_explainer():
    """
    Load the exact trained M5 LightGBM model and create
    a SHAP TreeExplainer for that model.

    No retraining occurs here.
    """

    if not RISK_MODEL_PATH.exists():
        raise RuntimeError(
            "M5 risk model not found. "
            "Run: python -m src.train"
        )

    if not RISK_FEATURES_PATH.exists():
        raise RuntimeError(
            "M5 risk feature list not found."
        )

    model = joblib.load(
        RISK_MODEL_PATH
    )

    feature_names = json.loads(
        RISK_FEATURES_PATH.read_text()
    )

    # SHAP versions may emit a LightGBM warning about
    # binary-classifier output format. We explicitly handle
    # both old and new output formats below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)

    return (
        model,
        explainer,
        feature_names,
    )


# ============================================================
# INPUT PREPARATION
# ============================================================

def clean_input(
    row: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Construct the exact feature matrix expected by M5.

    Missing/non-finite values are safely converted to zero.
    Feature ordering is preserved from the saved M5 feature list.
    """

    values = {}

    for feature in feature_names:

        value = row.get(
            feature,
            0,
        )

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        values[feature] = value

    return pd.DataFrame(
        [values],
        columns=feature_names,
    )


# ============================================================
# HUMAN-READABLE LABEL
# ============================================================

def human_label(
    feature: str,
) -> str:
    return FEATURE_LABELS.get(
        feature,
        feature.replace(
            "_",
            " ",
        ).title(),
    )


# ============================================================
# HUMAN-READABLE VALUE
# ============================================================

def display_value(
    feature: str,
    value: float,
) -> str:
    """
    Convert raw model values into values suitable for
    Applicant 360 / frontend display.

    rawValue is still retained separately for auditability.
    """

    # Home Credit stores DAYS_BIRTH as a negative day count.
    if feature == "DAYS_BIRTH":
        return f"{abs(value) / 365.25:.1f} years"

    # Home Credit employment duration can also use negative days.
    if feature == "DAYS_EMPLOYED":
        return f"{abs(value) / 365.25:.1f} years"

    if feature == "employment_years":
        return f"{value:.1f} years"

    if feature == "age_years":
        return f"{value:.1f} years"

    # Ratios are displayed as proportions.
    if (
        "ratio" in feature
        or feature.endswith("_trend")
    ):
        return f"{value:.2f}"

    # Credit/income monetary features.
    monetary_features = {
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "bureau_total_credit",
        "bureau_total_debt",
        "bureau_total_overdue",
        "previous_avg_requested",
        "previous_avg_credit",
    }

    if feature in monetary_features:
        return f"{value:,.0f}"

    # Payment timing uses signed semantics:
    # negative = early, zero = on time, positive = late.
    if (
        "delay" in feature
        or feature == "avg_payment_delay"
        or feature == "max_payment_delay"
    ):
        if value < 0:
            return f"{abs(value):.1f} days early"

        if value > 0:
            return f"{value:.1f} days late"

        return "On time"

    # Probability-like external scores.
    if feature.startswith("EXT_SOURCE"):
        return f"{value:.3f}"

    if feature == "ext_source_mean":
        return f"{value:.3f}"

    # Counts.
    count_features = {
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "bureau_account_count",
        "bureau_active_count",
        "bureau_delinquent_months",
        "bureau_month_count",
        "installment_count",
        "previous_application_count",
    }

    if feature in count_features:
        return f"{value:.0f}"

    return f"{value:.2f}"


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def _extract_shap_values(
    shap_values,
) -> np.ndarray:
    """
    Normalize SHAP output across SHAP versions.

    LightGBM binary classifiers may return either:
      - list[array]
      - 2D array
      - 3D array
    """

    if isinstance(
        shap_values,
        list,
    ):
        values = np.asarray(
            shap_values[-1]
        )

        if values.ndim == 2:
            return values[0]

        return values.reshape(-1)

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        return values[0, :, -1]

    if values.ndim == 2:
        return values[0]

    return values.reshape(-1)


# ============================================================
# COMPLETE MODEL EXPLANATION
# ============================================================

def explain_row(
    row: pd.Series,
    top_n: int = 5,
) -> dict:
    """
    Generate a genuine model-grounded explanation for
    one applicant using the trained M5 LightGBM model.

    Positive SHAP impact:
        pushes prediction toward higher default risk.

    Negative SHAP impact:
        pushes prediction toward lower default risk.
    """

    (
        model,
        explainer,
        feature_names,
    ) = load_explainer()

    X = clean_input(
        row,
        feature_names,
    )

    shap_values = explainer.shap_values(
        X
    )

    values = _extract_shap_values(
        shap_values
    )

    feature_values = X.iloc[0]

    contributions = []

    for (
        feature,
        value,
        impact,
    ) in zip(
        feature_names,
        feature_values,
        values,
    ):

        raw_value = float(
            value
        )

        shap_impact = float(
            impact
        )

        contributions.append(
            {
                "feature": feature,
                "label": human_label(
                    feature
                ),
                "rawValue": raw_value,
                "displayValue": display_value(
                    feature,
                    raw_value,
                ),
                "impact": round(
                    shap_impact,
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Risk-increasing factors
    # --------------------------------------------------------

    risk_increasing = sorted(
        [
            item
            for item in contributions
            if item["impact"] > 0
        ],
        key=lambda x: x["impact"],
        reverse=True,
    )[:top_n]

    # --------------------------------------------------------
    # Protective factors
    # --------------------------------------------------------

    risk_reducing = sorted(
        [
            item
            for item in contributions
            if item["impact"] < 0
        ],
        key=lambda x: x["impact"],
    )[:top_n]

    # --------------------------------------------------------
    # Explicit direction
    # --------------------------------------------------------

    for item in risk_increasing:
        item["direction"] = (
            "INCREASED_RISK"
        )

    for item in risk_reducing:
        item["direction"] = (
            "REDUCED_RISK"
        )

    # --------------------------------------------------------
    # Verify model probability
    # --------------------------------------------------------

    prediction = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    return {
        "riskScore": round(
            prediction,
            4,
        ),
        "topRiskFactors": risk_increasing,
        "topProtectiveFactors": risk_reducing,
        "method": "SHAP TreeExplainer",
        "model": "m5-risk-model-v1",
    }