from pathlib import Path
import pandas as pd
import numpy as np
from .config import DATA_PROCESSED

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 12
    ids = np.arange(100001, 100001+n)
    df = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": rng.binomial(1, .15, n),
        "amt_income_total": rng.normal(180000, 40000, n).clip(60000),
        "amt_credit": rng.normal(350000, 80000, n).clip(50000),
        "amt_annuity": rng.normal(20000, 5000, n).clip(5000),
        "days_birth": -rng.integers(8000, 22000, n),
        "days_employed": -rng.integers(500, 8000, n),
        "ext_source_1": rng.random(n),
        "ext_source_2": rng.random(n),
        "ext_source_3": rng.random(n),
        "credit_to_income": rng.uniform(1, 3, n),
        "annuity_to_income": rng.uniform(.05, .2, n),
        "employment_years": rng.uniform(1, 20, n),
        "age_years": rng.uniform(22, 60, n),
        "bureau_count": rng.integers(0, 12, n),
        "bureau_total_credit": rng.uniform(0, 800000, n),
        "bureau_total_debt": rng.uniform(0, 500000, n),
        "bureau_total_overdue": rng.uniform(0, 20000, n),
        "bureau_avg_age_days": rng.uniform(-2000, 0, n),
        "installment_count": rng.integers(0, 100, n),
        "avg_payment_ratio": rng.uniform(.7, 1.1, n),
        "max_delay_days": rng.integers(0, 30, n),
        "avg_delay_days": rng.uniform(0, 8, n),
        "on_time_ratio": rng.uniform(.7, 1, n),
        "previous_application_count": rng.integers(0, 8, n),
        "previous_credit_sum": rng.uniform(0, 500000, n),
        "previous_application_sum": rng.uniform(0, 600000, n),
        "previous_down_payment_sum": rng.uniform(0, 100000, n),
        "repayment_consistency_trend": rng.uniform(-.2, .2, n),
        "financial_pressure_trend": rng.uniform(0, 3, n),
        "delinquency_trend": rng.uniform(0, .2, n)
    })
    df.to_parquet(DATA_PROCESSED / "applicant_features.parquet", index=False)
    print("Created demo applicant feature table.")
    print("Demo IDs:", ids.tolist())

if __name__ == "__main__":
    main()
