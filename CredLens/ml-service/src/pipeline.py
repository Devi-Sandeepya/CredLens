from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import DATA_RAW
from .features import build_feature_sets


# ============================================================
# CONFIGURATION
# ============================================================

# P0 development cohort.
# The master plan calls for a 20–30k applicant development cohort.
COHORT_SIZE = 30_000

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FEATURE_DIR = OUTPUT_DIR / "features"


# ============================================================
# DATA PATHS
# ============================================================

APPLICATION_TRAIN = DATA_RAW / "application_train.csv"
BUREAU = DATA_RAW / "bureau.csv"
BUREAU_BALANCE = DATA_RAW / "bureau_balance.csv"
INSTALLMENTS = DATA_RAW / "installments_payments.csv"
PREVIOUS_APPLICATION = DATA_RAW / "previous_application.csv"


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_APPLICATION_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
}

REQUIRED_BUREAU_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_BUREAU",
}

REQUIRED_BUREAU_BALANCE_COLUMNS = {
    "SK_ID_BUREAU",
    "MONTHS_BALANCE",
    "STATUS",
}

REQUIRED_INSTALLMENT_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_PREV",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
}

REQUIRED_PREVIOUS_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_PREV",
    "NAME_CONTRACT_STATUS",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "DAYS_DECISION",
}


def validate_columns(
    path: Path,
    required: set[str],
) -> None:
    """
    Validate a CSV schema without loading the complete dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    schema = pl.scan_csv(
        path,
        infer_schema_length=1000,
    ).collect_schema()

    actual = set(schema.names())
    missing = required - actual

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )


def validate_unique_applicants(app: pl.DataFrame) -> None:
    """
    application_train should contain one row per applicant.
    """
    duplicate_count = (
        app.group_by("SK_ID_CURR")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicated SK_ID_CURR values "
            "in the development application cohort."
        )


def validate_target(app: pl.DataFrame) -> None:
    """
    Verify the binary target is usable for supervised learning.
    """
    values = (
        app.select("TARGET")
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    if not set(values).issubset({0, 1}):
        raise ValueError(
            f"TARGET must be binary 0/1. Found: {values}"
        )

    if len(values) < 2:
        raise ValueError(
            "Development cohort contains only one target class."
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_development_application_cohort() -> pl.DataFrame:
    """
    Deterministically select the first 30,000 applicants from
    application_train.csv for the P0 development cohort.

    Historical tables are NOT truncated by row number.
    They are filtered by these applicant IDs later.
    """

    print("\n[1/7] Loading development applicant cohort...")

    app = (
        pl.scan_csv(
            APPLICATION_TRAIN,
            infer_schema_length=5000,
        )
        .select(
            pl.all()
        )
        .head(COHORT_SIZE)
        .collect()
    )

    print(f"      Applicants: {app.height:,}")
    print(f"      Columns:    {app.width}")

    validate_unique_applicants(app)
    validate_target(app)

    missing = REQUIRED_APPLICATION_COLUMNS - set(app.columns)

    if missing:
        raise ValueError(
            f"Application cohort missing: {sorted(missing)}"
        )

    return app


def lazy_historical_tables() -> tuple[
    pl.LazyFrame,
    pl.LazyFrame,
    pl.LazyFrame,
    pl.LazyFrame,
]:
    """
    Create lazy readers for the large historical datasets.

    Nothing substantial is loaded into memory here.
    """

    print("\n[2/7] Creating lazy historical readers...")

    bureau = pl.scan_csv(
        BUREAU,
        infer_schema_length=5000,
    )

    bureau_balance = pl.scan_csv(
        BUREAU_BALANCE,
        infer_schema_length=5000,
    )

    installments = pl.scan_csv(
        INSTALLMENTS,
        infer_schema_length=5000,
    )

    previous = pl.scan_csv(
        PREVIOUS_APPLICATION,
        infer_schema_length=5000,
    )

    return (
        bureau,
        bureau_balance,
        installments,
        previous,
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_table(
    name: str,
    df: pl.DataFrame,
) -> None:
    """
    Basic structural checks after feature construction.
    """

    if df.is_empty():
        raise ValueError(f"{name} is empty.")

    if "SK_ID_CURR" not in df.columns:
        raise ValueError(
            f"{name} does not contain SK_ID_CURR."
        )

    duplicate_ids = (
        df.group_by("SK_ID_CURR")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_ids:
        raise ValueError(
            f"{name} contains {duplicate_ids} duplicate applicants."
        )

    numeric_columns = [
        c
        for c, dtype in zip(df.columns, df.dtypes)
        if dtype.is_numeric()
    ]

    if numeric_columns:
        null_count = (
            df.select(
                [
                    pl.col(c).null_count().alias(c)
                    for c in numeric_columns
                ]
            )
            .sum_horizontal()
            .item()
        )

        if null_count:
            print(
                f"      WARNING: {name} contains "
                f"{null_count:,} numeric null values."
            )

    print(
        f"      {name}: "
        f"{df.height:,} applicants × {df.width:,} columns"
    )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_feature_table(
    name: str,
    df: pl.DataFrame,
) -> None:

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FEATURE_DIR / f"{name.lower()}_features.parquet"

    df.write_parquet(
        output_path,
        compression="zstd",
    )

    print(f"      Saved → {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print("=" * 72)
    print("CREDLENS — P0 FEATURE ENGINE")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Schema validation
    # --------------------------------------------------------

    print("\n[0/7] Validating source schemas...")

    validate_columns(
        APPLICATION_TRAIN,
        REQUIRED_APPLICATION_COLUMNS,
    )

    validate_columns(
        BUREAU,
        REQUIRED_BUREAU_COLUMNS,
    )

    validate_columns(
        BUREAU_BALANCE,
        REQUIRED_BUREAU_BALANCE_COLUMNS,
    )

    validate_columns(
        INSTALLMENTS,
        REQUIRED_INSTALLMENT_COLUMNS,
    )

    validate_columns(
        PREVIOUS_APPLICATION,
        REQUIRED_PREVIOUS_COLUMNS,
    )

    print("      All required source schemas validated.")

    # --------------------------------------------------------
    # 2. Application cohort
    # --------------------------------------------------------

    app = load_development_application_cohort()

    # --------------------------------------------------------
    # 3. Lazy historical datasets
    # --------------------------------------------------------

    (
        bureau,
        bureau_balance,
        installments,
        previous,
    ) = lazy_historical_tables()

    # --------------------------------------------------------
    # 4. Build M1 → M5
    # --------------------------------------------------------

    print("\n[3/7] Building M1 → M5 feature sets...")
    print("      Large historical tables remain lazy.")

    feature_sets = build_feature_sets(
        app=app,
        bureau=bureau,
        bureau_balance=bureau_balance,
        installments=installments,
        previous=previous,
    )

    # --------------------------------------------------------
    # 5. Validate each feature set
    # --------------------------------------------------------

    print("\n[4/7] Validating feature sets...")

    for name, df in feature_sets.items():
        validate_feature_table(name, df)

    # --------------------------------------------------------
    # 6. Verify cohort preservation
    # --------------------------------------------------------

    print("\n[5/7] Verifying applicant coverage...")

    expected_ids = set(
        app.select("SK_ID_CURR")
        .to_series()
        .to_list()
    )

    for name, df in feature_sets.items():

        actual_ids = set(
            df.select("SK_ID_CURR")
            .to_series()
            .to_list()
        )

        missing = expected_ids - actual_ids

        if missing:
            raise ValueError(
                f"{name} lost {len(missing):,} applicants."
            )

        extra = actual_ids - expected_ids

        if extra:
            raise ValueError(
                f"{name} contains {len(extra):,} unexpected applicants."
            )

        print(
            f"      {name}: "
            f"{len(actual_ids):,}/{len(expected_ids):,} applicants"
        )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    print("\n[6/7] Saving applicant-level feature tables...")

    for name, df in feature_sets.items():
        save_feature_table(name, df)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n[7/7] P0 feature generation complete.")

    print("\nFeature progression:")

    for name in ["M1", "M2", "M3", "M4", "M5"]:
        df = feature_sets[name]

        print(
            f"      {name}: "
            f"{df.height:,} applicants × "
            f"{df.width:,} columns"
        )

    print("\nNext stage:")
    print("      Train the SAME LightGBM configuration")
    print("      across M1 → M5.")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()