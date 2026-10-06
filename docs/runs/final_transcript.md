# Final CLI transcript

Captured 2026-09-29T18:35:34Z with `raapid-rag` (prompt v4, gpt-4o-mini, real embedder). Log lines (stderr) omitted.

## 1. Index

```
$ raapid-rag ingest
                              Index build report                               
+-----------------------------------------------------------------------------+
| field             | value                                                   |
|-------------------+---------------------------------------------------------|
| n_documents       | 8                                                       |
| n_chunks          | 62                                                      |
| fingerprint       | 9268ef4373e8d240                                        |
| loaded_from_cache | True                                                    |
| build_seconds     | 13.45                                                   |
| chunks_per_doc    | {"guideline_001": 7, "guideline_002": 7, "note_001": 9, |
|                   | "note_002": 10, "note_003": 8, "note_004": 7,           |
|                   | "policy_001": 8, "policy_002": 6}                       |
| max_chunk_tokens  | 183                                                     |
| embedder          | sentence-transformers/all-MiniLM-L6-v2                  |
+-----------------------------------------------------------------------------+
```

## 2. Retrieval only

```
$ raapid-rag search "Are audio-only telehealth visits acceptable for risk adjustment?" --top-k 3
     Retrieval for: 'Are audio-only telehealth visits acceptable for risk      
                                 adjustment?'                                  
+-----------------------------------------------------------------------------+
| # | chunk_id       | section                       | dense | bm25  | fused  |
|---+----------------+-------------------------------+-------+-------+--------|
| 1 | policy_002#c02 | 2. Telehealth Encounter       | 0.723 | 16.48 | 0.0328 |
|   |                | Requirements                  |       |       |        |
| 2 | policy_001#c02 | 2. Acceptable Encounter Types | 0.502 | 7.46  | 0.0318 |
| 3 | policy_002#c01 | 1. Scope                      | 0.455 | 8.26  | 0.0315 |
+-----------------------------------------------------------------------------+
gate: PASS (ok)  best_dense=0.723 best_bm25=16.48  context_tokens=254 
dropped=[] latency=12.4ms

policy_002#c02 2.1 A telehealth encounter is acceptable for risk adjustment 
only when it uses real-time interactive audio and video technology between the 
member and an acceptable provider. 2.2 Audio-only (telephone) encounters are 
not acceptable for risk...

policy_001#c02 2.1 A diagnosis may be submitted only when it is documented 
during a face-to-face encounter or an interactive audio-and-video telehealth 
encounter that meets the requirements of Policy RA-110. 2.2 Diagnoses 
documented only on the following ...

policy_002#c01 This policy is a companion to Policy RA-100. It defines when a 
telehealth encounter is acceptable for risk adjustment and how member 
eligibility must be verified before submission.
```

## 3. Ask: What was Jane Doe-Test's most recent HbA1c, and when was it collected?

```
$ raapid-rag ask "What was Jane Doe-Test's most recent HbA1c, and when was it collected?"
status: answered   confidence: 0.85 (high)   needs human review: no

answer: Jane Doe-Test's most recent HbA1c was 8.2%, and it was collected on 
2025-03-12.
                                   citations                                   
+-----------------------------------------------------------------------------+
| chunk_id     | section            | verified | evidence                     |
|--------------+--------------------+----------+------------------------------|
| note_001#c05 | Laboratory Results | yes      | HbA1c 8.2% (collected        |
|              |                    |          | 2025-03-12).                 |
+-----------------------------------------------------------------------------+

trace: prompt v4 | llm calls 1 | tokens 3530+132 | est $0.0006 | 3558.6 ms | 
anchored ['note_001'] | cache_hit False
```

## 4. Ask: How many days was John Q. Testpatient hospitalized, and was he eligible for coverage on the discharge date?

```
$ raapid-rag ask "How many days was John Q. Testpatient hospitalized, and was he eligible for coverage on the discharge date?"
status: answered   confidence: 0.99 (high)   needs human review: no

answer: John Q. Testpatient was hospitalized for 5 days, from March 2, 2025, to
March 7, 2025. He was eligible for coverage on the discharge date, as his 
status was ACTIVE on that day.
                                   citations                                   
+-----------------------------------------------------------------------------+
| chunk_id       | section                | verified | evidence               |
|----------------+------------------------+----------+------------------------|
| note_002#c01   | Patient Information    | yes      | Admitted 2025-03-02.   |
|                |                        |          | Discharged 2025-03-07. |
| policy_002#c05 | 5. Inpatient           | yes      | For inpatient stays,   |
|                | Encounters             |          | the date of service    |
|                |                        |          | for eligibility        |
|                |                        |          | purposes is the        |
|                |                        |          | discharge date.        |
| policy_002#c03 | 3. Member Eligibility  | yes      | The response must show |
|                | Verification           |          | coverage status ACTIVE |
|                |                        |          | for that date.         |
+-----------------------------------------------------------------------------+
tool calls:
  - days_between({"start_date": "2025-03-02", "end_date": "2025-03-07"})
  - check_eligibility({"member_id": "M-TEST-1002", "date_of_service": 
"2025-03-07"})

trace: prompt v4 | llm calls 2 | tokens 8039+283 | est $0.0014 | 4304.6 ms | 
anchored ['note_002'] | cache_hit False
```

## 5. Ask: Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?

```
$ raapid-rag ask "Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?"
status: answered   confidence: 0.89 (high)   needs human review: no

answer: The telehealth visit for Robert Nonreal cannot be submitted for risk 
adjustment because the member's coverage was inactive on the date of service, 
having been terminated retroactively effective March 31, 2025. Additionally, 
the telehealth visit was conducted via audio-only, which is not acceptable for 
risk adjustment submission according to payer policy.
                                   citations                                   
+-----------------------------------------------------------------------------+
| chunk_id       | section                | verified | evidence               |
|----------------+------------------------+----------+------------------------|
| policy_002#c02 | 2. Telehealth          | yes      | Audio-only (telephone) |
|                | Encounter Requirements |          | encounters are not     |
|                |                        |          | acceptable for risk    |
|                |                        |          | adjustment submission, |
|                |                        |          | regardless of whether  |
|                |                        |          | video was att          |
| policy_002#c02 | 2. Telehealth          | yes      | Diagnoses documented   |
|                | Encounter Requirements |          | during audio-only      |
|                |                        |          | visits must be         |
|                |                        |          | recaptured at a        |
|                |                        |          | qualifying encounter   |
|                |                        |          | before submission.     |
| policy_002#c04 | 4. Submission Timing   | yes      | Encounters must be     |
|                |                        |          | submitted within 90    |
|                |                        |          | calendar days of the   |
|                |                        |          | date of service, and   |
|                |                        |          | no later than the      |
|                |                        |          | final sweep deadline   |
| policy_001#c01 | 1. Purpose             | yes      | This policy defines    |
|                |                        |          | the minimum            |
|                |                        |          | documentation          |
|                |                        |          | standards for a        |
|                |                        |          | diagnosis to be        |
|                |                        |          | submitted for risk     |
|                |                        |          | adjustment under       |
|                |                        |          | Exampl                 |
+-----------------------------------------------------------------------------+
tool calls:
  - check_eligibility({"member_id": "M-TEST-1004", "date_of_service": 
"2025-04-02"})
  - icd10_lookup({"query": "M25.50"})

trace: prompt v4 | llm calls 2 | tokens 6650+358 | est $0.0012 | 5412.2 ms | 
anchored ['note_004'] | cache_hit False
```

## 6. Ask: What metformin dose is recommended for patients with CKD stage 3?

```
$ raapid-rag ask "What metformin dose is recommended for patients with CKD stage 3?"
status: insufficient_context   confidence: 0.00 (low)   needs human review: YES
missing information:
  - Specific guidelines or recommendations regarding metformin dosing for 
patients with CKD stage 3 are not provided in the context.

trace: prompt v4 | llm calls 1 | tokens 2933+87 | est $0.0005 | 1710.4 ms | 
anchored [] | cache_hit False
```

## 7. Ask: Who won the 2024 World Series?

```
$ raapid-rag ask "Who won the 2024 World Series?"
status: insufficient_context   confidence: 0.00 (low)   needs human review: YES
missing information:
  - No sufficiently relevant passage was found in the corpus for this question 
(gate: low_relevance(best_dense=0.080<0.37, best_bm25=3.70<5.0)).

trace: prompt v4 | llm calls 0 | tokens 0+0 | est $0.0000 | 11.6 ms | anchored 
[] | cache_hit False
```

## 8. Ask with fault injection (repair path)

```
$ FAULT_INJECTION=bad_citation raapid-rag ask "What was Jane Doe-Test's most recent HbA1c?"
status: answered   confidence: 0.77 (medium)   needs human review: YES

answer: Jane Doe-Test's most recent HbA1c was 8.2%, collected on 2025-03-12.
                                   citations                                   
+-----------------------------------------------------------------------------+
| chunk_id     | section            | verified | evidence                     |
|--------------+--------------------+----------+------------------------------|
| note_001#c05 | Laboratory Results | yes      | HbA1c 8.2% (collected        |
|              |                    |          | 2025-03-12).                 |
+-----------------------------------------------------------------------------+
warning: fault injection was enabled; the first model response was deliberately
corrupted for demonstration

trace: prompt v4 | llm calls 2 | tokens 7533+252 | est $0.0013 | 4207.6 ms | 
anchored ['note_001'] | cache_hit False
```

