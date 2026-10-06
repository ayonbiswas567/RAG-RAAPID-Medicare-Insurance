# Eval report (full mode)

Generated 2026-09-29T17:14:44+00:00. Model: `gpt-4o-mini`. Cases: 10.

## Summary

- **Pass rate: 9/10 (90%)**
- Refusal precision 100%, refusal recall 100%
- Cost $0.0091 total (51076 prompt + 2462 completion tokens)
- Latency mean 3410 ms, p95 11208 ms

## Calibration by confidence band (answered / partial only)

| band | n | correct | accuracy |
|---|---|---|---|
| high | 5 | 5 | 100% |
| medium | 2 | 1 | 50% |

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

- `cited_docs_any`: 1

## Cases

| id | category | status | conf | band | review | calls | cost | ms | result | failed checks |
|---|---|---|---|---|---|---|---|---|---|---|
| recent_a1c | grounded_fact | answered | 0.85 | high | no | 1 | $0.0006 | 2967 | PASS | - |
| hcc_mapping | multi_doc | answered | 1.00 | high | no | 2 | $0.0014 | 4767 | PASS | - |
| los_and_eligibility | tool | answered | 0.99 | high | no | 2 | $0.0014 | 3734 | PASS | - |
| telehealth_submittable | multi_doc | answered | 0.79 | medium | yes | 1 | $0.0006 | 2071 | PASS | - |
| uncertain_dx | guideline | partial | 0.79 | medium | yes | 4 | $0.0029 | 11208 | FAIL | cited_docs_any (cited ['note_004']; expected any of ['guideline_001']) |
| history_of_cancer | guideline | answered | 1.00 | high | no | 1 | $0.0007 | 3545 | PASS | - |
| policy_id | grounded_fact | answered | 0.90 | high | no | 1 | $0.0006 | 2142 | PASS | - |
| canary_parametric | refusal_canary | insufficient_context | 0.00 | low | yes | 1 | $0.0005 | 2248 | PASS | - |
| missing_fact | refusal_missing | insufficient_context | 0.00 | low | yes | 1 | $0.0006 | 1405 | PASS | - |
| gate_refusal | gate_refusal | insufficient_context | 0.00 | low | yes | 0 | $0.0000 | 14 | PASS | - |

## Answers

- **recent_a1c**: Jane Doe-Test's most recent HbA1c was 8.2%, and it was collected on 2025-03-12.
- **hcc_mapping**: The diabetes codes documented for Jane Doe-Test, E11.22 (Type 2 diabetes mellitus with diabetic chronic kidney disease) and E11.42 (Type 2 diabetes mellitus with diabetic polyneuropathy), both map to 
  - tools: icd10_lookup, icd10_lookup
- **los_and_eligibility**: John Q. Testpatient was hospitalized for 5 days, from March 2, 2025, to March 7, 2025. He was eligible for coverage on the discharge date, as his status was ACTIVE on that day.
  - tools: days_between, check_eligibility
- **telehealth_submittable**: No, the telehealth visit for Robert Nonreal cannot be submitted for risk adjustment. The encounter was conducted via audio-only (telephone), which is not acceptable for risk adjustment submission acco
  - warning: question appears to require tool(s) check_eligibility but they were not called; answer capped below high confidence and flagged for review
- **uncertain_dx**: The diagnosis 'possible rheumatoid arthritis' cannot be coded as rheumatoid arthritis. According to the coding guideline, uncertain diagnoses such as 'possible' should not be coded. Instead, the signs
  - tools: icd10_lookup, check_eligibility
  - caveat: The member's coverage was INACTIVE on the date of service (2025-04-02), as it was terminated retroactively effective 2025-03-31. Therefore, any diagnoses documented during this telehealth visit cannot be submitted for risk adjustment.
  - warning: 1 unverifiable citation(s) removed after repair attempts were exhausted; answer downgraded to 'partial' and flagged for review
- **history_of_cancer**: No, Maria Fictitious's breast cancer should not be coded as an active condition for risk adjustment. It is documented as a personal history of malignant neoplasm of breast (Z85.3) with no evidence of 
- **policy_id**: Audio-only (telephone) encounters are not acceptable for risk adjustment submission, regardless of whether video was attempted. Diagnoses documented during audio-only visits must be recaptured at a qu
- **canary_parametric**: (no answer text)
  - missing information: Specific guidelines or recommendations regarding metformin dosing for patients with CKD stage 3 are not provided in the context.
- **missing_fact**: (no answer text)
  - missing information: There is no mention of any blood pressure medication being taken by Maria Fictitious in the provided context.
- **gate_refusal**: (no answer text)
  - missing information: No sufficiently relevant passage was found in the corpus for this question (gate: low_relevance(best_dense=0.080<0.37, best_bm25=3.70<5.0)).
