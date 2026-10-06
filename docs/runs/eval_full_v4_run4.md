# Eval report (full mode)

Generated 2026-09-29T17:15:38+00:00. Model: `gpt-4o-mini`. Cases: 10.

## Summary

- **Pass rate: 10/10 (100%)**
- Refusal precision 100%, refusal recall 100%
- Cost $0.0111 total (63716 prompt + 2646 completion tokens)
- Latency mean 3930 ms, p95 16905 ms

## Calibration by confidence band (answered / partial only)

| band | n | correct | accuracy |
|---|---|---|---|
| high | 6 | 6 | 100% |
| medium | 1 | 1 | 100% |

## By category

| category | passed | n |
|---|---|---|
| gate_refusal | 1 | 1 |
| grounded_fact | 2 | 2 |
| guideline | 2 | 2 |
| multi_doc | 2 | 2 |
| refusal_canary | 1 | 1 |
| refusal_missing | 1 | 1 |
| tool | 1 | 1 |

## Cases

| id | category | status | conf | band | review | calls | cost | ms | result | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| recent_a1c | grounded_fact | answered | 0.85 | high | no | 1 | $0.0006 | 2615 | PASS | - |
| hcc_mapping | multi_doc | answered | 1.00 | high | no | 2 | $0.0014 | 4550 | PASS | - |
| los_and_eligibility | tool | answered | 0.99 | high | no | 2 | $0.0014 | 4248 | PASS | - |
| telehealth_submittable | multi_doc | answered | 0.79 | medium | yes | 1 | $0.0006 | 1920 | PASS | - |
| uncertain_dx | guideline | answered | 0.96 | high | yes | 7 | $0.0049 | 16905 | PASS | - |
| history_of_cancer | guideline | answered | 1.00 | high | no | 1 | $0.0007 | 3372 | PASS | - |
| policy_id | grounded_fact | answered | 0.90 | high | no | 1 | $0.0006 | 2460 | PASS | - |
| canary_parametric | refusal_canary | insufficient_context | 0.00 | low | yes | 1 | $0.0005 | 1503 | PASS | - |
| missing_fact | refusal_missing | insufficient_context | 0.00 | low | yes | 1 | $0.0006 | 1716 | PASS | - |
| gate_refusal | gate_refusal | insufficient_context | 0.00 | low | yes | 0 | $0.0000 | 14 | PASS | - |

## Answers

- **recent_a1c**: Jane Doe-Test's most recent HbA1c was 8.2%, and it was collected on 2025-03-12.
- **hcc_mapping**: The diabetes codes documented for Jane Doe-Test, E11.22 (Type 2 diabetes mellitus with diabetic chronic kidney disease) and E11.42 (Type 2 diabetes mellitus with diabetic polyneuropathy), both map to 
  - tools: icd10_lookup, icd10_lookup
- **los_and_eligibility**: John Q. Testpatient was hospitalized for 5 days, from March 2, 2025, to March 7, 2025. He was eligible for coverage on the discharge date, as his status was ACTIVE on that day.
  - tools: days_between, check_eligibility
- **telehealth_submittable**: No, the telehealth visit for Robert Nonreal cannot be submitted for risk adjustment. The encounter was conducted via audio-only (telephone), which is not acceptable for risk adjustment submission acco
  - warning: question appears to require tool(s) check_eligibility but they were not called; answer capped below high confidence and flagged for review
- **uncertain_dx**: No, 'possible rheumatoid arthritis' from Robert Nonreal's telehealth note cannot be coded as rheumatoid arthritis. According to the coding guidelines, diagnoses documented as "possible" should not be 
  - tools: icd10_lookup, check_eligibility, check_eligibility (cached), check_eligibility (cached)
  - warning: tool-round limit reached; the final answer was forced without further tool calls
- **history_of_cancer**: No, Maria Fictitious's breast cancer should not be coded as an active condition for risk adjustment. It is documented as a personal history of malignant neoplasm of breast (Z85.3) with no evidence of 
- **policy_id**: Audio-only (telephone) encounters are not acceptable for risk adjustment submission, regardless of whether video was attempted. Diagnoses documented during audio-only visits must be recaptured at a qu
- **canary_parametric**: (no answer text)
  - missing information: Specific guidelines or recommendations regarding metformin dosing for patients with CKD stage 3 are not provided in the context.
- **missing_fact**: (no answer text)
  - missing information: There is no mention of any blood pressure medication being taken by Maria Fictitious in the provided context.
- **gate_refusal**: (no answer text)
  - missing information: No sufficiently relevant passage was found in the corpus for this question (gate: low_relevance(best_dense=0.080<0.37, best_bm25=3.70<5.0)).
