# Fraud and Integrity Screening Policy

The Integrity layer runs in parallel with the risk model, not after it.
It uses unsupervised anomaly detection to flag applications with unusual
patterns in field combinations, behavioral ratios, or balance-to-payment
relationships that differ significantly from typical applicant behavior.

## Why Unsupervised Detection
The underlying dataset does not contain verified fraud labels — it is a
credit default dataset, not a fraud dataset. Training a supervised
"fraud classifier" on unlabeled proxy data would be scientifically
dishonest and could produce misleading confidence in fraud predictions
that were never actually validated against real fraud outcomes. An
anomaly-detection approach that flags unusual patterns for human review
is the responsible choice given the data actually available.

## Output Meaning
The Integrity layer never outputs a "fraud probability" or "fraud
detected" verdict. It outputs an integrity status of either NORMAL or
UNUSUAL. An UNUSUAL flag means the application pattern differs
meaningfully from typical applicants and warrants human investigation —
it is not a fraud accusation, and it does not by itself determine
guilt or wrongdoing.