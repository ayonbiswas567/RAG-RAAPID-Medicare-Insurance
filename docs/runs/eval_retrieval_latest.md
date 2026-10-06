# Eval report (retrieval mode)

Generated 2026-10-04T14:22:34+00:00. Model: `none`. Prompt: `n/a`. Cases: 10.

## Summary

- **Pass rate: 10/10 (100%)**
- Refusal precision 100%, refusal recall 33%
- Cost $0.0000 total (0 prompt + 0 completion tokens)
- Latency mean 85 ms, p95 354 ms

## Calibration by confidence band (answered / partial only)

| band | n | correct | accuracy |
|---|---|---|---|

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
| recent_a1c | grounded_fact | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 129 | PASS | - |
| hcc_mapping | multi_doc | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 354 | PASS | - |
| los_and_eligibility | tool | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 61 | PASS | - |
| telehealth_submittable | multi_doc | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 50 | PASS | - |
| uncertain_dx | guideline | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 62 | PASS | - |
| history_of_cancer | guideline | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 47 | PASS | - |
| policy_id | grounded_fact | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 54 | PASS | - |
| canary_parametric | refusal_canary | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 24 | PASS | - |
| missing_fact | refusal_missing | retrieval_only | 0.00 | low | yes | 0 | $0.0000 | 43 | PASS | - |
| gate_refusal | gate_refusal | insufficient_context | 0.00 | low | yes | 0 | $0.0000 | 23 | PASS | - |

## Answers

- **recent_a1c**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **hcc_mapping**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **los_and_eligibility**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **telehealth_submittable**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **uncertain_dx**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **history_of_cancer**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **policy_id**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **canary_parametric**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **missing_fact**: (no answer text)
  - missing information: language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only
- **gate_refusal**: (no answer text)
  - missing information: No sufficiently relevant passage was found in the corpus for this question (gate: low_relevance(best_dense=0.080<0.37, best_bm25=3.70<5.0)).
