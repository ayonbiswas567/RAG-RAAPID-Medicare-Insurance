# Eval report (full mode)

Generated 2026-09-29T16:59:37+00:00. Model: `gpt-4o-mini`. Cases: 10.

## Summary

- **Pass rate: 9/10 (90%)**
- Refusal precision 75%, refusal recall 100%
- Cost $0.0113 total (66245 prompt + 2326 completion tokens)
- Latency mean 3947 ms, p95 13298 ms

## Calibration by confidence band (answered / partial only)

| band | n | correct | accuracy |
|---|---|---|---|
| high | 6 | 6 | 100% |

## By category

| category | passed | n |
|---|---|---|
| gate_refusal | 1 | 1 |
| grounded_fact | 2 | 2 |
| guideline | 1 | 2 |
| multi_doc | 2 | 2 |
| refusal_canary | 1 | 1 |
| refusal_missing | 1 | 1 |
| tool | 1 | 1 |

## Failed checks

- `status`: 1
- `cited_docs_any`: 1
- `contains:not|cannot|should not`: 1
- `contains:polyarthralgia|symptom`: 1

## Cases

| id | category | status | conf | band | review | calls | cost | ms | result | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| recent_a1c | grounded_fact | answered | 0.85 | high | no | 1 | $0.0006 | 3045 | PASS | - |
| hcc_mapping | multi_doc | answered | 1.00 | high | no | 2 | $0.0013 | 5443 | PASS | - |
| los_and_eligibility | tool | answered | 0.99 | high | no | 2 | $0.0013 | 4086 | PASS | - |
| telehealth_submittable | multi_doc | answered | 0.87 | high | yes | 6 | $0.0035 | 13298 | PASS | - |
| uncertain_dx | guideline | insufficient_context | 0.00 | low | yes | 5 | $0.0027 | 5482 | FAIL | status (got 'insufficient_context', expected one of ['answered', 'partial']); cited_docs_any (cited []; expected any of ['guideline_001']); contains:not|cannot|should not (answer excerpt: ''); contains:polyarthralgia|symptom (answer excerpt: '') |
| history_of_cancer | guideline | answered | 1.00 | high | no | 1 | $0.0006 | 3241 | PASS | - |
| policy_id | grounded_fact | answered | 0.90 | high | no | 1 | $0.0005 | 2008 | PASS | - |
| canary_parametric | refusal_canary | insufficient_context | 0.00 | low | yes | 1 | $0.0004 | 1518 | PASS | - |
| missing_fact | refusal_missing | insufficient_context | 0.00 | low | yes | 1 | $0.0005 | 1332 | PASS | - |
| gate_refusal | gate_refusal | insufficient_context | 0.00 | low | yes | 0 | $0.0000 | 14 | PASS | - |

## Answers

- **recent_a1c**: Jane Doe-Test's most recent HbA1c was 8.2%, and it was collected on 2025-03-12.
- **hcc_mapping**: The diabetes codes documented for Jane Doe-Test, E11.22 (Type 2 diabetes mellitus with diabetic chronic kidney disease) and E11.42 (Type 2 diabetes mellitus with diabetic polyneuropathy), both map to 
- **los_and_eligibility**: John Q. Testpatient was hospitalized for 5 days, from March 2, 2025, to March 7, 2025. He was eligible for coverage on the discharge date, as his status was ACTIVE on that day.
- **telehealth_submittable**: The telehealth visit for Robert Nonreal cannot be submitted for risk adjustment because it was conducted via audio-only (telephone), which is not acceptable for risk adjustment submission according to
  - warning: tool-round limit reached; the final answer was forced without further tool calls
- **uncertain_dx**: (no answer text)
  - warning: tool-round limit reached; the final answer was forced without further tool calls
- **history_of_cancer**: No, Maria Fictitious's breast cancer should not be coded as an active condition for risk adjustment. It is documented as a personal history of malignant neoplasm of breast (Z85.3) with no evidence of 
- **policy_id**: Audio-only (telephone) encounters are not acceptable for risk adjustment submission, regardless of whether video was attempted. Diagnoses documented during audio-only visits must be recaptured at a qu
- **canary_parametric**: (no answer text)
- **missing_fact**: (no answer text)
- **gate_refusal**: (no answer text)
