# Eval report (full mode)

Generated 2026-09-29T17:03:28+00:00. Model: `gpt-4o-mini`. Cases: 10.

## Summary

- **Pass rate: 9/10 (90%)**
- Refusal precision 100%, refusal recall 100%
- Cost $0.0082 total (47128 prompt + 1872 completion tokens)
- Latency mean 2908 ms, p95 7630 ms

## Calibration by confidence band (answered / partial only)

| band | n | correct | accuracy |
|---|---|---|---|
| high | 7 | 6 | 86% |

## By category

| category | passed | n |
|---|---|---|
| gate_refusal | 1 | 1 |
| grounded_fact | 2 | 2 |
| guideline | 2 | 2 |
| multi_doc | 1 | 2 |
| refusal_canary | 1 | 1 |
| refusal_missing | 1 | 1 |
| tool | 1 | 1 |

## Failed checks

- `contains:Chronic Complications`: 1
- `tools_any`: 1

## Cases

| id | category | status | conf | band | review | calls | cost | ms | result | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| recent_a1c | grounded_fact | answered | 0.85 | high | no | 1 | $0.0006 | 3098 | PASS | - |
| hcc_mapping | multi_doc | answered | 1.00 | high | no | 1 | $0.0007 | 3761 | FAIL | contains:Chronic Complications (answer excerpt: 'The diabetes codes documented for Jane Doe-Test map to the diabetes-with-complic'); tools_any (used []; expected any of ['icd10_lookup']) |
| los_and_eligibility | tool | answered | 0.99 | high | no | 2 | $0.0013 | 4886 | PASS | - |
| telehealth_submittable | multi_doc | answered | 0.90 | high | no | 1 | $0.0005 | 2070 | PASS | - |
| uncertain_dx | guideline | answered | 0.99 | high | no | 5 | $0.0029 | 7630 | PASS | - |
| history_of_cancer | guideline | answered | 1.00 | high | no | 1 | $0.0006 | 2782 | PASS | - |
| policy_id | grounded_fact | answered | 0.90 | high | no | 1 | $0.0006 | 2175 | PASS | - |
| canary_parametric | refusal_canary | insufficient_context | 0.00 | low | yes | 1 | $0.0005 | 1377 | PASS | - |
| missing_fact | refusal_missing | insufficient_context | 0.00 | low | yes | 1 | $0.0005 | 1287 | PASS | - |
| gate_refusal | gate_refusal | insufficient_context | 0.00 | low | yes | 0 | $0.0000 | 9 | PASS | - |

## Answers

- **recent_a1c**: Jane Doe-Test's most recent HbA1c was 8.2%, and it was collected on 2025-03-12.
- **hcc_mapping**: The diabetes codes documented for Jane Doe-Test map to the diabetes-with-complications category. This is supported by the presence of E11.22 (type 2 diabetes with diabetic chronic kidney disease) and 
- **los_and_eligibility**: John Q. Testpatient was hospitalized for 5 days, from March 2, 2025, to March 7, 2025. He was eligible for coverage on the discharge date, as his status was ACTIVE on that date.
- **telehealth_submittable**: No, the telehealth visit for Robert Nonreal cannot be submitted for risk adjustment. The visit was conducted using audio-only (telephone) technology, which is not acceptable for risk adjustment submis
- **uncertain_dx**: No, 'possible rheumatoid arthritis' cannot be coded as rheumatoid arthritis. According to the coding guidelines, diagnoses documented as 'possible' should not be coded; instead, the signs and symptoms
  - warning: tool-round limit reached; the final answer was forced without further tool calls
- **history_of_cancer**: No, Maria Fictitious's breast cancer should not be coded as an active condition for risk adjustment. It is documented as a personal history of malignant neoplasm of breast (Z85.3) with no evidence of 
- **policy_id**: Audio-only (telephone) encounters are not acceptable for risk adjustment submission, regardless of whether video was attempted. Diagnoses documented during audio-only visits must be recaptured at a qu
- **canary_parametric**: (no answer text)
- **missing_fact**: (no answer text)
- **gate_refusal**: (no answer text)
