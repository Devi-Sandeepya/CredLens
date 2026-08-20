# Evidence Confidence Methodology

Evidence Confidence is a documented heuristic measuring how much reliable
evidence supports a given risk assessment. It is distinct from the risk
score itself. A risk score answers "how risky is this applicant?" while
Evidence Confidence answers "how much do we actually know about this
applicant?"

## Components
Evidence Confidence combines four weighted signals: bureau history depth
(30%), repayment history depth (25%), historical data coverage (25%),
and overall data completeness (20%).

## Why It Matters for Thin-File Applicants
A thin-file applicant may receive a low risk score simply because there
is little negative history to penalize them, not because they are
genuinely low-risk. Evidence Confidence surfaces this distinction:
low risk with low confidence should not be treated the same as low
risk with high confidence. The Policy Engine uses both signals together,
never risk alone.

## Calibration Status
These weights are a documented heuristic based on domain reasoning, not
a statistically calibrated model. A production deployment would validate
and calibrate these weights against observed default and repayment
outcomes over time.