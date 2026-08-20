from __future__ import annotations

import polars as pl


# ============================================================
# UTILITY
# ============================================================

def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """
    Safe division that avoids division-by-zero and invalid values.
    """
    return (
        pl.when(
            den.is_not_null() &
            (den.abs() > 1e-9)
        )
        .then(num / den)
        .otherwise(0.0)
    )


# ============================================================
# M1 — APPLICATION FEATURES
# ============================================================

def build_application_features(
    app: pl.DataFrame,
) -> pl.DataFrame:

    numeric = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    available = [
        c for c in numeric
        if c in app.columns
    ]

    out = app.select(
        ["SK_ID_CURR", "TARGET"] + available
    )

    # Normalize numerical columns.
    out = out.with_columns(
        [
            pl.col(c).cast(
                pl.Float64,
                strict=False,
            )
            for c in available
        ]
    )

    out = out.with_columns(
        safe_div(
            pl.col("AMT_CREDIT"),
            pl.col("AMT_INCOME_TOTAL"),
        ).alias("credit_to_income"),

        safe_div(
            pl.col("AMT_ANNUITY"),
            pl.col("AMT_INCOME_TOTAL"),
        ).alias("annuity_to_income"),

        (
            -pl.col("DAYS_BIRTH") / 365.25
        ).alias("age_years"),

        (
            pl.when(
                pl.col("DAYS_EMPLOYED") > 0
            )
            .then(0.0)
            .otherwise(
                -pl.col("DAYS_EMPLOYED") / 365.25
            )
        ).alias("employment_years"),
    )

    # Combined external-score signal.
    ext_cols = [
        c
        for c in [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]
        if c in out.columns
    ]

    if ext_cols:
        out = out.with_columns(
            pl.mean_horizontal(
                [pl.col(c) for c in ext_cols]
            ).alias("ext_source_mean")
        )

    return (
        out
        .fill_nan(0)
        .fill_null(0)
    )


# ============================================================
# M2 — CREDIT HISTORY
# BUREAU + BUREAU BALANCE
# ============================================================

def aggregate_bureau(
    bureau: pl.LazyFrame,
    bureau_balance: pl.LazyFrame,
    applicant_ids: pl.DataFrame,
) -> pl.DataFrame:

    ids = applicant_ids.lazy().select(
        "SK_ID_CURR"
    )

    # --------------------------------------------------------
    # Aggregate monthly bureau balance to bureau-account level
    # --------------------------------------------------------

    balance = (
        bureau_balance
        .join(
            bureau.select(
                [
                    "SK_ID_BUREAU",
                    "SK_ID_CURR",
                ]
            ),
            on="SK_ID_BUREAU",
            how="inner",
        )
        .join(
            ids,
            on="SK_ID_CURR",
            how="semi",
        )
        .group_by("SK_ID_BUREAU")
        .agg(
            pl.len().alias(
                "bureau_month_count"
            ),

            (
                pl.col("STATUS")
                .cast(pl.String)
                .is_in(
                    [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                    ]
                )
                .sum()
            ).alias(
                "bureau_delinquent_months"
            ),

            (
                pl.col("STATUS")
                .cast(pl.String)
                .is_in(
                    [
                        "C",
                        "X",
                    ]
                )
                .sum()
            ).alias(
                "bureau_closed_or_unknown_months"
            ),
        )
    )

    # --------------------------------------------------------
    # Join bureau account information
    # --------------------------------------------------------

    b = (
        bureau
        .join(
            ids,
            on="SK_ID_CURR",
            how="semi",
        )
        .join(
            balance,
            on="SK_ID_BUREAU",
            how="left",
        )
        .select(
            [
                "SK_ID_CURR",
                "SK_ID_BUREAU",
                "CREDIT_ACTIVE",
                "DAYS_CREDIT",
                "CREDIT_DAY_OVERDUE",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
                "bureau_month_count",
                "bureau_delinquent_months",
            ]
        )
    )

    # --------------------------------------------------------
    # Applicant-level aggregation
    # --------------------------------------------------------

    result = (
        b
        .group_by("SK_ID_CURR")
        .agg(
            pl.len().alias(
                "bureau_account_count"
            ),

            (
                pl.col("CREDIT_ACTIVE")
                == "Active"
            )
            .sum()
            .alias(
                "bureau_active_count"
            ),

            pl.col("AMT_CREDIT_SUM")
            .sum()
            .alias(
                "bureau_total_credit"
            ),

            pl.col("AMT_CREDIT_SUM_DEBT")
            .sum()
            .alias(
                "bureau_total_debt"
            ),

            pl.col("AMT_CREDIT_SUM_OVERDUE")
            .sum()
            .alias(
                "bureau_total_overdue"
            ),

            pl.col("CREDIT_DAY_OVERDUE")
            .max()
            .alias(
                "bureau_max_overdue_days"
            ),

            pl.col("DAYS_CREDIT")
            .min()
            .alias(
                "bureau_history_depth"
            ),

            pl.col("bureau_delinquent_months")
            .sum()
            .alias(
                "bureau_delinquent_months"
            ),

            pl.col("bureau_month_count")
            .sum()
            .alias(
                "bureau_month_count"
            ),
        )
        .with_columns(
            safe_div(
                pl.col("bureau_total_debt"),
                pl.col("bureau_total_credit"),
            ).alias(
                "bureau_debt_to_credit"
            ),

            safe_div(
                pl.col(
                    "bureau_delinquent_months"
                ),
                pl.col(
                    "bureau_month_count"
                ),
            ).alias(
                "bureau_delinquency_ratio"
            ),
        )
        .collect()
    )

    return (
        result
        .fill_nan(0)
        .fill_null(0)
    )


# ============================================================
# M3 — REPAYMENT FEATURES
# INSTALLMENTS
# ============================================================

def aggregate_installments(
    installments: pl.LazyFrame,
    applicant_ids: pl.DataFrame,
) -> pl.DataFrame:

    ids = applicant_ids.lazy().select(
        "SK_ID_CURR"
    )

    x = (
        installments
        .join(
            ids,
            on="SK_ID_CURR",
            how="semi",
        )
        .filter(
            pl.col(
                "DAYS_INSTALMENT"
            ).is_not_null()
        )
        .with_columns(
            (
                pl.col(
                    "DAYS_ENTRY_PAYMENT"
                )
                -
                pl.col(
                    "DAYS_INSTALMENT"
                )
            ).alias(
                "payment_delay"
            ),

            safe_div(
                pl.col("AMT_PAYMENT"),
                pl.col("AMT_INSTALMENT"),
            ).alias(
                "payment_ratio"
            ),
        )
    )

    result = (
        x
        .group_by("SK_ID_CURR")
        .agg(
            pl.len().alias(
                "installment_count"
            ),

            (
                pl.col("payment_delay") <= 0
            )
            .mean()
            .alias(
                "on_time_payment_ratio"
            ),

            (
                pl.col("payment_delay") > 0
            )
            .mean()
            .alias(
                "late_payment_ratio"
            ),

            pl.col("payment_delay")
            .mean()
            .alias(
                "avg_payment_delay"
            ),

            pl.col("payment_delay")
            .max()
            .alias(
                "max_payment_delay"
            ),

            pl.col("payment_ratio")
            .mean()
            .alias(
                "avg_payment_ratio"
            ),

            pl.col("payment_ratio")
            .std()
            .alias(
                "payment_ratio_volatility"
            ),
        )
        .collect()
    )

    return (
        result
        .fill_nan(0)
        .fill_null(0)
    )


# ============================================================
# M4 — BEHAVIORAL FEATURES
# PREVIOUS APPLICATIONS
# ============================================================

def aggregate_previous_applications(
    previous: pl.LazyFrame,
    applicant_ids: pl.DataFrame,
) -> pl.DataFrame:

    ids = applicant_ids.lazy().select(
        "SK_ID_CURR"
    )

    x = (
        previous
        .join(
            ids,
            on="SK_ID_CURR",
            how="semi",
        )
        .filter(
            pl.col(
                "DAYS_DECISION"
            ).is_not_null()
        )
        .with_columns(
            (
                pl.col(
                    "NAME_CONTRACT_STATUS"
                )
                == "Approved"
            )
            .cast(pl.Int8)
            .alias(
                "approved_flag"
            ),

            (
                pl.col(
                    "NAME_CONTRACT_STATUS"
                )
                == "Refused"
            )
            .cast(pl.Int8)
            .alias(
                "refused_flag"
            ),

            safe_div(
                pl.col("AMT_CREDIT"),
                pl.col("AMT_APPLICATION"),
            ).alias(
                "previous_credit_to_requested"
            ),
        )
    )

    result = (
        x
        .group_by("SK_ID_CURR")
        .agg(
            pl.len().alias(
                "previous_application_count"
            ),

            pl.col("approved_flag")
            .mean()
            .alias(
                "previous_approval_ratio"
            ),

            pl.col("refused_flag")
            .mean()
            .alias(
                "previous_refusal_ratio"
            ),

            pl.col("AMT_APPLICATION")
            .mean()
            .alias(
                "previous_avg_requested"
            ),

            pl.col("AMT_CREDIT")
            .mean()
            .alias(
                "previous_avg_credit"
            ),

            pl.col(
                "previous_credit_to_requested"
            )
            .mean()
            .alias(
                "previous_credit_to_requested_ratio"
            ),

            pl.col("CNT_PAYMENT")
            .mean()
            .alias(
                "previous_avg_term"
            ),
        )
        .collect()
    )

    return (
        result
        .fill_nan(0)
        .fill_null(0)
    )


# ============================================================
# M5 — TEMPORAL FEATURES
# ============================================================

def build_temporal_features(
    installments: pl.LazyFrame,
    applicant_ids: pl.DataFrame,
) -> pl.DataFrame:

    ids = applicant_ids.lazy().select(
        "SK_ID_CURR"
    )

    x = (
        installments
        .join(
            ids,
            on="SK_ID_CURR",
            how="semi",
        )
        .filter(
            pl.col(
                "DAYS_INSTALMENT"
            ).is_not_null()
        )
        .with_columns(
            (
                pl.col(
                    "DAYS_ENTRY_PAYMENT"
                )
                -
                pl.col(
                    "DAYS_INSTALMENT"
                )
            ).alias(
                "payment_delay"
            )
        )
    )

    # --------------------------------------------------------
    # Recent behavior: last 12 months
    # --------------------------------------------------------

    recent = (
        x
        .filter(
            pl.col(
                "DAYS_INSTALMENT"
            ) >= -365
        )
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("payment_delay")
            .mean()
            .alias(
                "recent_avg_delay"
            ),

            (
                pl.col("payment_delay") <= 0
            )
            .mean()
            .alias(
                "recent_on_time_ratio"
            ),

            pl.col("payment_delay")
            .std()
            .alias(
                "recent_delay_volatility"
            ),
        )
    )

    # --------------------------------------------------------
    # Historical behavior: older than 12 months
    # --------------------------------------------------------

    historical = (
        x
        .filter(
            pl.col(
                "DAYS_INSTALMENT"
            ) < -365
        )
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("payment_delay")
            .mean()
            .alias(
                "historical_avg_delay"
            ),

            (
                pl.col("payment_delay") <= 0
            )
            .mean()
            .alias(
                "historical_on_time_ratio"
            ),
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Full join can create SK_ID_CURR_right.
    # Coalesce the two keys and remove the duplicate.
    # --------------------------------------------------------

    result = (
        recent
        .join(
            historical,
            on="SK_ID_CURR",
            how="full",
        )
        .with_columns(
            pl.coalesce(
                [
                    pl.col("SK_ID_CURR"),
                    pl.col("SK_ID_CURR_right"),
                ]
            ).alias(
                "SK_ID_CURR"
            )
        )
        .drop(
            "SK_ID_CURR_right"
        )
        .with_columns(
            (
                pl.col(
                    "recent_avg_delay"
                )
                -
                pl.col(
                    "historical_avg_delay"
                )
            ).alias(
                "delay_trend"
            ),

            (
                pl.col(
                    "recent_on_time_ratio"
                )
                -
                pl.col(
                    "historical_on_time_ratio"
                )
            ).alias(
                "on_time_trend"
            ),
        )
        .collect()
        .fill_nan(0)
        .fill_null(0)
    )

    return result


# ============================================================
# M1 → M5 FEATURE SET BUILDER
# ============================================================

def build_feature_sets(
    app: pl.DataFrame,
    bureau: pl.LazyFrame,
    bureau_balance: pl.LazyFrame,
    installments: pl.LazyFrame,
    previous: pl.LazyFrame,
) -> dict[str, pl.DataFrame]:

    applicant_ids = app.select(
        "SK_ID_CURR"
    )

    # --------------------------------------------------------
    # M1 — Application
    # --------------------------------------------------------

    m1 = build_application_features(
        app
    )

    # --------------------------------------------------------
    # M2 — Application + Bureau
    # --------------------------------------------------------

    bureau_features = aggregate_bureau(
        bureau,
        bureau_balance,
        applicant_ids,
    )

    m2 = (
        m1
        .join(
            bureau_features,
            on="SK_ID_CURR",
            how="left",
        )
        .fill_nan(0)
        .fill_null(0)
    )

    # --------------------------------------------------------
    # M3 — M2 + Repayment
    # --------------------------------------------------------

    repayment_features = aggregate_installments(
        installments,
        applicant_ids,
    )

    m3 = (
        m2
        .join(
            repayment_features,
            on="SK_ID_CURR",
            how="left",
        )
        .fill_nan(0)
        .fill_null(0)
    )

    # --------------------------------------------------------
    # M4 — M3 + Behavioral
    # --------------------------------------------------------

    behavioral_features = (
        aggregate_previous_applications(
            previous,
            applicant_ids,
        )
    )

    m4 = (
        m3
        .join(
            behavioral_features,
            on="SK_ID_CURR",
            how="left",
        )
        .fill_nan(0)
        .fill_null(0)
    )

    # --------------------------------------------------------
    # M5 — M4 + Temporal
    # --------------------------------------------------------

    temporal_features = (
        build_temporal_features(
            installments,
            applicant_ids,
        )
    )

    m5 = (
        m4
        .join(
            temporal_features,
            on="SK_ID_CURR",
            how="left",
        )
        .fill_nan(0)
        .fill_null(0)
    )

    # --------------------------------------------------------
    # Final safety check:
    # Every feature table must contain exactly one applicant
    # row per SK_ID_CURR.
    # --------------------------------------------------------

    feature_sets = {
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m5,
    }

    for name, df in feature_sets.items():

        if df.select(
            "SK_ID_CURR"
        ).n_unique() != df.height:
            raise ValueError(
                f"{name} contains duplicate "
                "SK_ID_CURR values."
            )

        if "SK_ID_CURR_right" in df.columns:
            raise ValueError(
                f"{name} contains accidental "
                "SK_ID_CURR_right column."
            )

    return feature_sets