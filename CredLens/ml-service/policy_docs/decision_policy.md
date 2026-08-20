# CredLens Decision Policy v2

## Approval Criteria
An applicant is APPROVED when their model-predicted risk score is below 30%
AND their Evidence Confidence is at least 60%. Low risk alone is not
sufficient for approval if the evidence supporting that risk assessment
is weak — thin-file applicants with very little history may show a low
raw risk score but insufficient evidence to act on it confidently.

## Decline Criteria
An applicant is DECLINED when their model-predicted risk score is 60% or
higher AND Evidence Confidence is at least 60%. High risk with strong
evidence is treated as a confident, actionable decline decision.

## Refer Criteria
An applicant is REFERRED for manual underwriter review when: the risk
score falls between the approve and decline thresholds, when Evidence
Confidence is below 60% regardless of risk score, or when the Integrity
layer flags the application as UNUSUAL. Referral means human review is
required before any credit decision is finalized.

## Integrity Override
Regardless of the computed risk score or confidence level, if the
Integrity layer detects an UNUSUAL pattern, the decision is always
REFER, never APPROVE or DECLINE automatically. Unusual patterns require
human investigation before any credit action is taken.