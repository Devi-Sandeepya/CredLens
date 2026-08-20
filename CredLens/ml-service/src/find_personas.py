"""
One-off script to find real applicant IDs for the two
missing demo personas (Section 30a):

  - Thin-file / good  -> low risk, high confidence, NORMAL
  - Suspicious         -> UNUSUAL integrity flag

Run once from ml-service/ with the venv activated:
    python -m src.find_personas
"""

from __future__ import annotations

from .app import (
    load_feature_table,
    load_artifacts,
    build_feature_frame,
    confidence_for,
    calculate_integrity,
    clean_feature_value,
)


def main():
    df = load_feature_table()
    risk_model, risk_features, integrity_model, integrity_features = load_artifacts()

    good_candidates = []
    suspicious_candidates = []

    for _, row in df.iterrows():
        applicant_id = int(row["SK_ID_CURR"])

        x = build_feature_frame(row, risk_features)
        risk = float(risk_model.predict_proba(x)[:, 1][0])
        confidence = confidence_for(row)
        integrity_status = calculate_integrity(
            x, integrity_model, integrity_features
        )

        if integrity_status == "UNUSUAL" and len(suspicious_candidates) < 5:
            suspicious_candidates.append((applicant_id, round(risk, 3)))

        if (
            risk < 0.15
            and confidence >= 75
            and integrity_status == "NORMAL"
            and len(good_candidates) < 5
        ):
            good_candidates.append((applicant_id, round(risk, 3), confidence))

        if len(good_candidates) >= 5 and len(suspicious_candidates) >= 5:
            break

    print("\n=== Thin-file / good candidates (low risk, high confidence) ===")
    for aid, risk, conf in good_candidates:
        print(f"  applicantId={aid}  risk={risk}  confidence={conf}")

    print("\n=== Suspicious candidates (UNUSUAL integrity) ===")
    for aid, risk in suspicious_candidates:
        print(f"  applicantId={aid}  risk={risk}")


if __name__ == "__main__":
    main()