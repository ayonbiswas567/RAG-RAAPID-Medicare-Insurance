# Grounded GenAI Micro-Service for Clinical and Policy Documents

**Design note for the RAAPID Applied AI Engineer take-home.** Code, tests, prompts, recorded runs
and a long-form engineering guide are in the repository; this is the two-page summary.

## What it does

A question comes in; the service retrieves relevant passages from a synthetic corpus of chart
notes, payer policies and coding guidelines, calls tools where facts must be looked up or computed,
and returns a schema-validated JSON envelope: answer, citations naming the exact chunk with a
verbatim quote, a status (`answered`, `partial`, `insufficient_context`, `retrieval_only`,
`error`), a confidence score and band, missing information, caveats, and a `needs_human_review`
flag. The guiding principle: grounding guarantees come from the system, not from trusting the
model. The default model is deliberately small (`gpt-4o-mini`); if the guardrails hold on it, they
hold on anything larger.

## Architecture

**Ingestion.** Documents are split on section headers (a diagnosis and its evidence share a
section) with a 220-word-piece window and 40 overlap as fallback, because the local embedding model
(`all-MiniLM-L6-v2`) truncates silently at 256. Chunk text is always a verbatim substring of the
source and chunk ids are deterministic. Chunks are embedded locally (the corpus never leaves the
machine) and indexed for dense cosine and BM25, since embeddings blur identifiers like `E11.22`
while BM25 misses paraphrase. The index is saved under a fingerprint of documents, parameters and
embedder, so query workers load an artefact instead of re-embedding. Long documents never reach
the model whole: context is capped at 1,800 tokens, an anchored document over 900 tokens
contributes only its best chunks, and map-reduce is the plan for whole-document summaries.

**Retrieval.** Twenty candidates fused by Reciprocal Rank Fusion. If the question names a patient,
member id or policy number, that document is pulled in as a block (a coder opens the chart before
reading the section); a per-document cap keeps one guideline from filling every slot. A refusal
gate on *absolute* scores (best cosine 0.37, best BM25 5.0, calibrated on probes) refuses in
milliseconds without a model call. RRF is never used for gating: a rank always puts something first.

**Reasoning.** One bounded loop over function calling with four tools: mock ICD-10 lookup with an
illustrative HCC mapping, mock eligibility check (one member retroactively terminated), a safe
calculator (AST-walked, no `eval`) and a date difference. Calls are validated, time-bounded,
memoised per request; failures return to the model as structured errors. The final answer is
requested in strict JSON-schema mode.

**Validation and repair.** Shape by pydantic; truth by our code: cited ids must be ones we
retrieved, quotes must be verbatim substrings, answers need citations, refusals need
`missing_information`. Failures go back to the model with the valid ids and, when a quote came
from a tool result, that fact. After two repairs, unverifiable citations are dropped and the
answer becomes a flagged `partial`; if nothing verifiable remains, a deterministic error envelope.
The caller always receives a valid object.

**Surfaces.** One stateless `QueryService` behind FastAPI (`/query`, `/search`, `/documents`,
`/health`; request ids; model outcomes always HTTP 200), a CLI, and a Streamlit UI that shows
status and review flag first, then citations with their source text.

## Prompt and reasoning design

Four recorded prompt versions. v1 ("use your best judgement to fill gaps") leaked training
knowledge on a canary question and quoted a context header as evidence. v2 named the only sources
of truth, required verbatim quotes and forbade citing tool results; cost halved, but two
cross-document questions were refused and tools went unused. v3 added "apply the rule in one chunk
to the facts in another" and mandatory tool use; its remaining failure was retrieval recall, fixed
by document anchoring, not prompting. v4 came from the eval: a required `caveats` field gives the
model somewhere to put a side concern instead of refusing, the eligibility mandate was narrowed,
and "a tool result is never a reason to refuse" was made explicit. Each change is tied to a run in
`PROMPT_CHANGELOG.md`.

## Grounding and guardrails

Six layers, most not the prompt: the retrieval gate; anchoring and the per-document cap so the
right facts are in context; prompt rules; citation-id integrity; the verbatim evidence check; and
post-processing that blanks prose on a refusal, caps and flags an answer that skipped a tool the
question clearly needed, and repairs a refusal whose reason is a tool result on a topic never
asked about. Confidence combines verifiable signals (retrieval strength 35%, verified-citation
fraction 35%, tool health, repair penalty, model self-report capped at 10%); partial answers never
reach the high band. A human stays in the loop before any code reaches a chart or a submission
(the corpus policy calls automated output a recommendation; RADV audits make unsupported codes a
liability), whenever `needs_human_review` is true, and on policy interpretation and eval regressions.

## Evaluation results

Ten cases with mechanical checks (status, cited documents, verified quotes, keywords, tools,
model-call budget), in `full` mode and a free `retrieval` mode for CI.

| configuration | runs | pass | refusal precision / recall | high-band accuracy | cost / run |
|---|---|---|---|---|---|
| prompt v3 | 1 | 9/10 | 0.75 / 1.0 | 6/6 | $0.011 |
| v4, emphatic tool mandates | 2 | 8/10, 8/10 | 0.6 / 1.0 | 5/5 | $0.009 |
| v4 + `caveats` + tool policy + refusal repair (shipped) | 2 | 9/10, 10/10 | 1.0 / 1.0 | 5/5, 6/6 | $0.010 |

Across four runs of the shipped family the high band was correct 22 of 22 times; every miss was a
refusal or a medium-band answer, so every miss carried the review flag. Fault injection recovered
in one repair on 6 of 6 questions. Retrieval mode: 10 of 10 in 13 ms per case.

## Cost, latency, scale

Per query at list price: refusal about $0.0004 (1.3 s), simple answer about $0.0005 (2 to 4 s),
tool-heavy $0.001 to $0.003 (4 to 9 s); worst observed 13 s. Ten thousand mixed queries a day is
roughly $10 to $15. Ingestion runs at minutes per thousand documents on one CPU core. Scaling:
stateless workers behind a load balancer, each loading the prebuilt index (14 s cold start, gated
by `/health`); a per-process semaphore and an answer cache protect the provider; a storage
protocol lets FAISS or pgvector replace the numpy store past a few million chunks.

## PHI and PII

Mock data only. With real charts: de-identify before embedding and prompting; local embeddings
keep the corpus on premise and only retrieved chunks reach the model, under a BAA with zero
retention; identifier-like patterns in questions are flagged or blocked; no question or chunk text
in logs; patient and tenant scoping as a mandatory retrieval filter derived from identity; an
immutable audit of every query, retrieved ids, tool call and answer; encryption everywhere.

## Known limitations

The verbatim check catches fabricated quotes, not fabricated attribution: in one run every quote
was real but a guideline's example codes were presented as the patient's (fixed by anchoring;
claim-level attribution is future work). Ten eval cases is a smoke test; the gate margin is thin
(0.025 cosine). The model varies run to run at temperature 0 and follows MUST inconsistently.
Tools are mocks; PHI scrubbing is format-based; no authentication or multi-tenancy; negation,
family history, copy-forward text and OCR noise are unhandled.

## Productionizing inside RAAPID

Plug-in points: coder assist ("show the sentence supporting E11.22 in this chart"), policy Q&A,
provider-query drafting, RADV audit response. Path: a de-identification and ingestion pipeline on
a queue; a tenant-partitioned vector store with patient filters; the same `QueryService` in
autoscaled containers behind a gateway adding identity, limits and scope; an enterprise LLM
endpoint under a BAA with a fallback model; the eval in CI plus nightly canaries; traces carrying
prompt version, index fingerprint and model id; coder override rate as the primary quality metric.
Next steps in order: claim-level attribution with an NLI check, a numeric-consistency check, a
cross-encoder re-ranker, and a knowledge graph so HCC hierarchy logic is computed deterministically
while the model only surfaces evidence.
