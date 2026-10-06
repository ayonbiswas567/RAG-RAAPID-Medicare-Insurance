# RAAPID Grounded GenAI Micro-Service: Engineering Notes

**What this file is.** A study guide and build log for the take-home project. It is written so that
the author can walk into a technical discussion with RAAPID's AI head and explain every design
decision, every piece of code, what breaks, and how it would be productionized. It is *not* the
1-2 page submission note (that will be `docs/DESIGN_NOTE.md`, written last). This file is allowed
to be long.

**How it grew.** The project was built in eight small, deliberate steps; each appended a "Step N
log" in Part 4 and, where a decision changed, updated Part 2. Nothing is written here that has
not been built or measured unless it is explicitly marked *planned*. All eight steps are complete;
the README is the run guide and `docs/DESIGN_NOTE.md` is the two-page submission note.

**Conventions.** Plain language first, jargon second (Part 5 is a glossary). Every decision records
*why*, *what else was considered*, *how it fails*, and *how it breaks under load*. A standing
requirement from the author applies throughout: every component must hold up at thousands of
documents per day and thousands of concurrent users, or leave a clean seam for the component that
will.

---

## Table of contents

- Part 0: The assignment, decoded
- Part 1: Architecture
- Part 2: Key decisions (D1 to D14)
- Part 3: Build plan and status
- Part 4: Step logs (Steps 1 to 4)
- Part 5: Glossary
- Part 6: Likely questions from an AI head, with answers
- Part 7: How to read this codebase (reading order, every function explained)
- Part 8: One request traced end to end, with real numbers
- Part 9: Where to change what
- Part 10: Interview preparation beyond the codebase (concepts, domain, production, demo script)

---

## Part 0: The assignment, decoded

The assignment asks for a "grounded GenAI micro-service". Stripped to its bones it is a test of
one idea: **can you make a language model answer only from documents you gave it, prove it,
and fail safely when it cannot?** Everything else (tools, schemas, evals, UI) exists to make that
idea verifiable and operable.

| Requirement (from the brief) | Where it is satisfied | Step |
|---|---|---|
| Ingest 5-10 short mock clinical/policy docs | `data/corpus/` (8 synthetic docs) | 1 *done* |
| Natural-language query, retrieve only relevant chunks, in-memory store | `ingestion/store.py`, `retrieval/retriever.py` (numpy cosine + BM25, RRF-fused, document anchoring) | 2 *done* |
| Answer strictly from retrieved context with inline citations to originating chunk | `schema.py` (citations carry chunk ids and verbatim evidence), `agent.py`, verbatim check in `validate_grounding` | 3, 4 *done* |
| Call at least one external tool as part of producing the answer | `tools/` (mock ICD-10 lookup, mock eligibility check, calculator, date difference) wired into OpenAI function calling | 3, 4 *done* |
| Schema-enforced JSON output, validated | `schema.py` (pydantic) + OpenAI strict JSON schema mode + our truth layer | 3, 4 *done* |
| Chunking strategy with reasoning, plan for oversized documents | `ingestion/chunker.py`, decision D5 | 2 *done* |
| Retry/repair path when output fails to parse or validate | `agent.py` repair loop; demonstrated live with `FAULT_INJECTION=bad_citation` | 4 *done* |
| Prompt design artifacts (v1 to v2 with explanation) | `llm/prompts/system_v1.md`, `v2`, `v3`, `v4`, `PROMPT_CHANGELOG.md` with recorded runs for every change | 4, 6 *done* |
| Explicit handling of at least two failure modes | D10 catalogue: gate refusal without a model call, tool timeout/exception, malformed JSON, invalid or unverifiable citations (repair, then salvage), provider outage (retries, then error envelope), context budget, PHI in the query, no API key (retrieval-only) | 2 to 5 *done* |
| Eval set (5-10 queries) with expected citations and pass/fail scoring | `eval/eval_set.json` (10 cases), `evaluation.py`, `raapid-rag eval`; reports in `docs/runs/eval_*.md` | 6 *done* |
| Refuse or flag when unsupported; confidence / missing-information signal | `answer_status`, `missing_information`, `confidence`, `confidence_band`, `needs_human_review`, `warnings`; composite in `guardrails.compute_confidence` | 4, 5 *done* |
| Where a human stays in the loop, and why | Decision D12; `needs_human_review` flag; `DESIGN_NOTE.md` section "Grounding and guardrails" | 5, 8 *done* |
| PHI/PII handling if pointed at real data | Decision D11; `LOG_CHUNK_TEXT=false`; `guardrails.scrub_query` with `PHI_POLICY` flag/block; `DESIGN_NOTE.md` section "PHI and PII" | 5, 8 *done* |
| Live end-to-end run including a refusal case | `eval/step4_live_run.py`, `eval/step5_live_run.py`, `eval/step7_smoke.py` (real servers over HTTP), eval runs; all recorded in `docs/runs/` (gate refusal, canary refusal, tools, PHI flag, cache hit, fault injection) | 4 to 7 *done* |
| Token cost, latency, scaling to thousands of documents/day | Measured in Steps 4 to 7 (Part 4); D2, D3, D4, D14; `DESIGN_NOTE.md` section "Cost, latency, scale" | 4 to 8 *done* |
| Streamlit frontend | `ui/app.py` (in-process over `QueryService`; headless tests with `AppTest`) | 7 *done* |
| HTTP micro-service | `api.py` (FastAPI: `/health`, `/documents`, `/search`, `/query`; request ids; every model outcome a 200 envelope) | 7 *done* |
| 1-2 page note | `docs/DESIGN_NOTE.md` (architecture, prompt design, grounding and guardrails, eval results, cost and scale, PHI, limitations, productionization) | 8 *done* |

---

## Part 1: Architecture

### 1.1 One-paragraph summary

A user asks a question. The service first searches a small in-memory index of document chunks
using two complementary methods (keyword matching and semantic embeddings) and keeps the top few
chunks. If the question names a patient, member or policy number, that whole document is pulled in
as a block, the way a coder opens a chart. If nothing relevant is found, the service refuses
*before* spending money on the language model. Otherwise it sends the question, the chunks, and
a set of tool definitions to the model. The model may call tools (an ICD-10 lookup, an
eligibility check, a calculator, a date difference); each call is executed by our code with a
timeout and the result is fed back. The model must then produce a JSON object matching a strict
schema: an answer, citations that name specific chunk ids, verbatim evidence quotes, a status,
a self-assessment, and a list of what information was missing. Our code validates that JSON
twice: once for shape, once for *truth* (do the cited chunk ids exist, and are the evidence
quotes really in those chunks?). If validation fails, the error is fed back to the model for
repair, at most twice. If it still fails, the service returns a deterministic, schema-valid error
envelope flagged for human review. The same core is exposed through a FastAPI endpoint, a CLI,
and a Streamlit UI.

### 1.2 Diagram

```
                +-----------------+     +------------------+
                |  Streamlit UI   |     |   CLI (typer)    |
                |  ui/app.py      |     |   cli.py         |
                +--------+--------+     +---------+--------+
                         |  (in-process)          |
                         v                        v
                +--------+------------------------+--------+      +----------------+
                |               QueryService  (service.py)        |<-----| FastAPI api.py |
                |                                                  |      | POST /query    |
                |  1  validate + scrub query          guardrails   |      +----------------+
                |  2  retrieve: rank + anchor + cap   retrieval    |
                |  3  refusal gate (absolute scores)  retrieval    |
                |  4  reasoning loop with tools       agent        |
                |  5  schema + truth validation, repair  agent     |
                |  6  confidence + human-review flag  guardrails   |
                +----+-----------------+------------------+--------+
                     |                 |                  |
        +------------v-----+   +-------v--------+   +-----v------------------+
        | Ingestion/Index  |   |  Tools         |   |  LLM client            |
        | loader.py        |   |  icd10.py      |   |  llm/client.py         |
        | chunker.py       |   |  eligibility.py|   |  retries, timeouts,    |
        | embedder.py      |   |  calculator.py |   |  rate-limit backoff,   |
        | store.py (numpy  |   |  registry.py   |   |  token accounting      |
        |  cosine + BM25)  |   |  (timeouts)    |   |  prompts/ v1, v2, v3   |
        | pipeline.py      |   +----------------+   +------------------------+
        | (fingerprint     |
        |  cache on disk)  |
        +------------------+
                 ^
                 |  data/corpus/*.md  (8 synthetic documents, front matter + ## sections)
```

### 1.3 Request lifecycle, step by step

1. **Input validation and PHI scrub** (step 5). Reject empty or absurdly long queries. Regex scan
   for SSN, phone and DOB patterns; flag (mock data) or block (real data).
2. **Retrieval.** Embed the query with the same local model used at ingestion. Score every chunk
   by cosine similarity (dense) and by BM25 (keyword). Merge with Reciprocal Rank Fusion into a
   candidate pool of `4 x top_k`. If the query contains an identifier that belongs to one
   document (patient name, member id, policy number), include that document as a block. Fill the
   remaining slots by rank with at most 3 chunks per document. Trim to the token budget.
3. **Refusal gate.** If the pool is empty, or the best dense cosine and the best BM25 score are
   both below their thresholds, return `insufficient_context` immediately. No model call, no cost,
   fully deterministic.
4. **Reasoning loop.** Messages: system prompt (v3), the retrieved chunks each labelled with its
   id, the question. Four tools registered as OpenAI function definitions. Loop: call the model;
   if it requests tools, run each with a timeout and append results; repeat up to
   `MAX_TOOL_ROUNDS`, then force a final answer. Strict JSON schema output is requested on every
   call.
5. **Two-layer validation and repair.** Layer one: pydantic parses the JSON. Layer two: cited ids
   exist; each evidence quote is a verbatim substring of its chunk; `answered` has citations;
   `insufficient_context` has `missing_information`. On failure the errors, the valid ids, and
   (when the quote came from a tool result) the real cause are sent back for repair, at most
   `MAX_REPAIR_ATTEMPTS` times. Then a deterministic error envelope.
6. **Confidence and review flag** (step 5). Composite of retrieval strength, citation
   verification, tool failures and the model's self-assessment; `needs_human_review` unless
   everything is clean.
7. **Respond** with the envelope plus a trace (retrieval scores, anchored documents, tool calls,
   repairs, token usage, estimated cost, latency).

### 1.4 Module map and the one-purpose rule

| Module | One-line purpose | Depends on | Status |
|---|---|---|---|
| `config.py` | Every tunable, validated at startup | pydantic-settings | done |
| `ingestion/models.py` | `Document` and `Chunk` frozen dataclasses | none | done |
| `ingestion/loader.py` | Read corpus files, parse front matter, enforce corpus invariants | filesystem | done |
| `ingestion/chunker.py` | Split a `Document` into `Chunk`s with stable ids, verbatim text | injected tokenizer | done |
| `ingestion/embedder.py` | `Embedder` protocol; MiniLM implementation; hashing test double | sentence-transformers | done |
| `ingestion/store.py` | `SearchIndex` protocol; in-memory hybrid index; save/load | numpy, rank-bm25 | done |
| `ingestion/pipeline.py` | corpus -> chunks -> index, fingerprint-keyed disk cache | all ingestion | done |
| `retrieval/retriever.py` | Pool, anchoring, per-doc cap, token budget, gate | store protocol, config | done |
| `tools/*.py` | Mock ICD-10, eligibility, calculator, days_between | none | done |
| `tools/registry.py` | Tool JSON schemas, dispatch, timeout, error containment | tools, schema | done |
| `schema.py` | `ModelOutput` (model-facing), `GroundedAnswer` (service-facing), truth validation | pydantic | done |
| `llm/client.py` | OpenAI wrapper: normalised results, retries, usage, cost | openai, tenacity | done |
| `llm/prompting.py` | System prompt loading, context block, repair messages | none | done |
| `llm/prompts/` | `system_v1.md` to `system_v4.md`, `PROMPT_CHANGELOG.md` | none | done |
| `agent.py` | Reasoning loop, structured output, repair, fault injection | llm, tools, schema | done |
| `cli.py` | `ingest`, `search`, `ask`, `eval` | pipeline, retriever, service, evaluation | done |
| `evaluation.py` | Eval set model, mechanical scoring, aggregation, reports | schema, service | done |
| `guardrails.py` | Query validation, PHI scrub, citation enrichment, refusal blanking, salvage, confidence | schema, config | done |
| `service.py` | `QueryService` facade: wiring, cache, concurrency bound, health | everything | done |
| `api.py` | FastAPI shell: app factory, request ids, 400 only for unusable input | service | done |
| `ui/app.py` | Streamlit page: question, scope, status, citations with source text, tools, trace | service | done |

### 1.5 Why no orchestration framework

LangChain 1.x is installed in the machine's global environment but is deliberately **not** used.
The brief wants evidence of understanding "how these systems fail and how to engineer around it".
Frameworks hide exactly the seams where failures occur: where JSON is parsed, where a tool times
out, where a retry happens, where a context window overflows. Writing the loop directly against
the OpenAI SDK (the agent is about 200 lines) means every seam is a line of code we can point at,
test with a fake, and explain. The live runs in Step 4 found three failure modes (header-line
quotes, tool text cited as evidence, guideline examples attributed to a patient) precisely because
every intermediate message is visible.

---

## Part 2: Key decisions

Format for each: **Decision**, **Why**, **Alternatives considered**, **How it fails / breaks under
load**, **interview soundbite**.

### D1. Hand-rolled agent loop on the raw OpenAI SDK; no LangChain

Covered in 1.5. Soundbite: *"I wanted every failure seam to be a line of code I can point at, not a
stack frame inside a framework."*

### D2. Language model: OpenAI `gpt-4o-mini` by default, configurable

- **Why.** Verified against the user's key. Cheap (list price about $0.15 per million input tokens
  and $0.60 per million output; verify before quoting). Supports function calling and strict JSON
  schema output simultaneously. Most importantly, using a *small* model is a design statement: the
  grounding guarantees come from the system (gate, verbatim check, schema validation, anchoring),
  not from a frontier model's judgement. If the guardrails hold on a mini model they hold on a
  larger one.
- **Measured cost (Step 4).** A simple grounded answer costs 2.1k to 3.3k prompt tokens (about
  $0.0005). A refusal costs about 2.5k prompt tokens and one call (about $0.0004). Tool-heavy
  questions cost 7k to 15k prompt tokens over 2 to 5 calls ($0.001 to $0.003). The worst observed
  run (two failed repairs) cost 24.5k tokens and 13 seconds. Six mixed questions under v3 cost
  $0.0085. Ten thousand mixed queries a day is on the order of $10 to $15 a day at list price.
- **Measured latency.** Refusals 1.2 to 1.5 s. Single-call answers 2 to 4 s. Two-to-five-call
  tool answers 3.5 to 9 s. Unrepaired worst case 13 s. Latency is dominated by the number of
  round trips, so the biggest levers are fewer tool rounds and fewer repairs.
- **Observed weakness.** At temperature 0 the model is not deterministic across runs (tool call
  counts varied between identical runs), it sometimes repeats an identical tool call (three
  `days_between` calls in one run), and it follows MUST instructions inconsistently. The service
  layer will memoise identical tool calls per request (planned, step 5).
- **How it breaks under load.** Rate limits (429) under burst; handled with exponential backoff
  and jitter bounded by `LLM_MAX_RETRIES`, and to be bounded further by a per-process concurrency
  semaphore in the service. Provider outage surfaces as `LLMUnavailable`, which becomes an error
  envelope, never a 500.
- **Soundbite.** *"The model is a replaceable component; the guardrails are the product."*

### D3. Embeddings: local `all-MiniLM-L6-v2` via sentence-transformers

- **Why.** (1) Data locality: the corpus never leaves the machine at ingestion; only retrieved
  chunks go to the external model. (2) Zero marginal cost and determinism, so evals are free and
  reproducible. (3) Speed: 384 dimensions, about 80 MB, cold load measured at 12 to 15 s once per
  process, then milliseconds per chunk.
- **Hard constraint that shapes chunking.** `max_seq_length` is 256 word-piece tokens; longer
  inputs are silently truncated. Hence `CHUNK_TARGET_TOKENS` is capped at 256 and defaults to 220.
- **Known weakness, and the fix.** Dense embeddings blur exact identifiers (`E11.22` vs `E11.21`)
  and, as the Step 4 ranking dump showed, they rank a patient's Assessment section below a
  guideline's preamble for a question that names the patient. Hybrid retrieval (D4) fixes the
  first; document anchoring (D4) fixes the second.
- **How it breaks under load.** CPU-bound and partly GIL-bound; ingestion in the same process
  as queries causes latency spikes. Fix: ingestion worker, one model instance per process, batch
  encoding. Thousands of documents a day is a few hundred thousand chunks a day, well under an
  hour on one CPU core.
- **Soundbite.** *"Local embeddings mean the corpus never leaves the building; only the retrieved
  chunks do."*

### D4. Retrieval: hybrid ranking, absolute-score gate, per-document cap, document anchoring

This decision changed materially during Steps 2 and 4. The record of what was wrong and why is
kept on purpose.

- **Hybrid ranking with RRF.** BM25 and dense cosine each cover the other's blind spot (identifiers
  versus paraphrase). Scores live on different scales, so they are merged by rank:
  `fused = sum 1/(60 + rank)`. Only chunks with some keyword overlap receive a BM25 vote; the dense
  ranker always votes.
- **What was wrong in the Step 1 design: gating on RRF.** RRF is a rank aggregator. The top chunk
  always gets a high fused score, even for "purple elephant quantum trombone". Rank answers "which
  is best", not "is the best any good". The gate therefore uses **absolute** signals: best dense
  cosine over the candidate pool and best raw BM25. It refuses only when both are weak, so an
  exact identifier match can pass on BM25 alone.
- **Calibration (measured with `eval/retrieval_probe.py`).** Relevant probes scored dense 0.379
  to 0.723 and BM25 6.3 to 20.4. Irrelevant probes scored dense at most 0.354 and BM25 at most
  4.22 (the sports question matched the token "2024"). Thresholds: dense 0.37, BM25 5.0. The
  dense margin is 0.025, which is thin; MiniLM compresses cosine into a narrow band. The LLM
  layer is the backstop for borderline queries that pass. A larger embedding model or a
  cross-encoder re-ranker would widen the margin and is the first thing to try if the eval
  shows gate errors.
- **Per-document cap.** Without it, the HCC question retrieved four guideline chunks and never
  saw a second document. At most 3 chunks per document in the ranked fill.
- **Document anchoring.** The Step 4 ranking dump showed the patient's Assessment chunk was not
  in the top 15 for "Which HCC category do the diabetes codes documented for Jane Doe-Test map
  to?": the name lives only in the Patient Information chunk, and "HCC category ... map" pulls the
  guideline. Prompting cannot fix a recall failure. So the retriever builds a map from
  *identifiers* (patient name from the `Name:` line, surname, member id, patient ref,
  identifier-like doc ids and policy numbers such as `RA-110`) to documents. If the query contains
  one, that document is included whole (in document order) when it fits a 900-token sub-budget,
  otherwise its best three chunks. Only identifiers anchor; topical words never do, which is why
  a first attempt that used BM25 over document text was rejected: "HCC" would have anchored the
  whole guideline on every HCC question. Anchoring adds about 700 tokens on named-patient
  questions and is what made the HCC answer correct.
- **Explicit scope.** `retrieve(query, doc_ids=[...])` restricts search to given documents. In
  production, questions are scoped to a chart this way (a metadata filter), and anchoring is the
  free-text fallback.
- **How it breaks under load.** Brute-force cosine is a single matrix-vector product: milliseconds
  at 100k chunks, a real cost past a few million, at which point FAISS or pgvector replaces the
  store behind `SearchIndex`. `rank-bm25` rebuilds on every add (seconds at a few hundred thousand
  chunks; belongs in the ingestion worker). Anchoring is a dictionary lookup per query. The
  in-memory store is process-local; `save()`/`load()` plus the fingerprint cache mean N workers
  load one artefact instead of embedding N times.
- **Soundbite.** *"BM25 catches the codes, embeddings catch the paraphrase, RRF merges them, an
  absolute gate decides whether to bother the model, and anchoring opens the chart when the
  question names the patient."*

### D5. Chunking: section-aware first, token window with overlap second

- **Decision.** Split on `## ` headers; a section within 220 tokens is one chunk; a longer section
  is windowed into 220-token pieces with 40 tokens of overlap. The text before the first header
  becomes a "Preamble" chunk. Chunk ids are `note_002#c05`. Every chunk carries document metadata.
- **Measured.** 8 documents, 52 sections plus 8 preambles; one section (note_002 Hospital Course,
  433 word pieces) exceeds the limit and becomes 3 windows; total **62 chunks**. Maximum chunk
  size 220 tokens in the embedder's tokenizer.
- **Two tokenizers, two jobs.** Chunk sizing uses the MiniLM word-piece tokenizer (the truncation
  limit is in word pieces). Context budgeting uses tiktoken `cl100k_base` (what the model bills).
- **Verbatim invariant.** The chunker works with character offsets into the original body and
  slices it; it never re-joins pieces. `Chunk.text` is therefore always a substring of the
  document, which is what makes the evidence check meaningful. Tested explicitly.
- **Windowing.** Units are lines, then sentences, then words if a single sentence exceeds the
  target. Windows are packed greedily; the next window starts by stepping back over trailing units
  worth up to 40 tokens, always advancing at least one unit so the loop terminates. Word-piece
  counts are not perfectly additive, so a packed window is re-counted and trimmed if it overshoots.
- **Plan for documents that exceed the context window.** (1) In place: the model only ever sees
  chunks trimmed to `RETRIEVAL_CONTEXT_TOKEN_BUDGET` (1,800). (2) In place: an anchored document
  longer than 900 tokens contributes only its best three chunks. (3) *Planned*: map-reduce for
  "summarise this document" questions, keeping citations through both passes. (4) Production:
  metadata filters (patient, date range, type) before vector search.
- **Soundbite.** *"Sections are the unit a coder thinks in, and 220 tokens is the largest chunk my
  embedding model can actually see."*

### D6. Structured output: two validation layers, a repair loop, and an always-valid envelope

- **Two schemas on purpose.** `ModelOutput` is what the model emits: small, no defaults, no
  optionals (strict mode forbids both), only what the model can honestly know. `GroundedAnswer` is
  what the service returns: the model's output plus facts only our code can establish (tool
  records, computed confidence, review flag, warnings, errors, trace).
- **Layer one, shape.** OpenAI strict JSON schema mode, schema generated from pydantic and
  post-processed so every object is closed and every property required (tested). Pydantic parses
  the result anyway; strict mode is a first defence, not a guarantee.
- **Layer two, truth** (`validate_grounding`). Cited ids must be among the retrieved chunks;
  evidence must be a verbatim substring of the cited chunk after whitespace, quote and case
  normalisation; `answered`/`partial` need at least one citation and a non-empty answer;
  `insufficient_context` needs `missing_information`; self-reported confidence in [0, 1]. Error
  messages are written to be fed straight back to the model: they name the field, the value, and
  what would be valid.
- **Repair loop.** Up to 2 repairs. When the rejected quote turns out to be text from a tool
  result, the message says so explicitly; this was added after a live run in which the model
  cited the eligibility tool's sentence as chunk evidence three times in a row.
- **Always-valid envelope.** If repairs are exhausted, `GroundedAnswer.error_envelope` is returned:
  status `error`, confidence 0, `needs_human_review` true, the validation errors listed. The caller
  never sees a stack trace for model misbehaviour.
- **Demonstrated live.** `FAULT_INJECTION=bad_citation` corrupted the first response on all six
  questions; all six recovered with exactly one repair (`docs/runs/step4_fault_injection.json`).
- **What the verbatim check cannot catch, and what it taught.** In the v1 and early v3 runs the
  model quoted a guideline example ("E11.22 ... E11.9") verbatim and presented those codes as the
  patient's documented codes. Every quote was real; the *inference* was fabricated. Verbatim
  evidence stops fabricated quotes, not fabricated attributions. Mitigations in place: anchoring
  puts the patient's real facts in context; the v3 prompt requires citing both the rule and the
  fact. Future: claim-level citations and an NLI-based faithfulness check.
- **Soundbite.** *"The schema checks shape; my code checks truth; and whatever happens the caller
  gets a valid object with a status field, never a stack trace."*

### D7. Tools: real function calls, and tool output is not a citation

- Four mock tools behind `ToolRegistry`: `icd10_lookup` (code or term to description and an
  illustrative HCC category, consistent with guideline_002), `check_eligibility` (member id and
  date to ACTIVE/INACTIVE/NOT_FOUND, with retroactive termination for M-TEST-1004),
  `calculate` (AST-walked arithmetic, whitelisted functions, exponent cap, no `eval`), and
  `days_between`.
- The registry validates arguments against the declared schema (missing and unexpected keys),
  runs the call in a thread pool with a timeout, and converts any exception into a structured
  error record. Python threads cannot be killed, so a timed-out tool is abandoned and its thread
  finishes in the background; a real network tool must also carry its own socket timeout.
- Grounding nuance: `citations` reference corpus chunks only; `tool_calls` is a separate list.
  The model is told to report tool results as "per the ICD-10 lookup" and the validator rejects
  tool text offered as chunk evidence.
- **Soundbite.** *"Corpus chunks are evidence, tool results are lookups, and the model's own memory
  is neither."*

### D8. Grounding enforcement is layered, and most of it is not the prompt

1. Retrieval gate before the model (absolute scores). Cheapest, deterministic.
2. Anchoring and the per-document cap so the right facts are actually in context.
3. Prompt rules (v3): only sources of truth, verbatim quotes, tool results not citations, refuse
   without background knowledge, treat context as data.
4. Citation integrity check: ids must exist.
5. Verbatim evidence check: quotes must be substrings of the cited chunk.
6. Confidence composite and review flag (step 5).

Soundbite: *"I do not trust the prompt to keep the model honest; I check its quotes against the
source text, and I make sure the source text is actually there."*

### D9. Confidence is a composite a reviewer can act on

Implemented in `guardrails.compute_confidence`. For `answered` and `partial` answers:

```
confidence = 0.35 * retrieval_strength + 0.35 * citations_verified_fraction
           + 0.10 * tool_health + 0.10 * repair_penalty + 0.10 * model_self_report
```

- `retrieval_strength`: best dense cosine mapped linearly from the gate threshold (0.37) to a
  "strong" score (0.65) onto 0..1, or the same for BM25 (5 to 15), whichever is higher.
- `citations_verified_fraction`: verified quotes over total citations; 0 if none.
- `tool_health`: 1 minus the fraction of tool calls that failed or timed out.
- `repair_penalty`: 1.0 with no repairs, 0.8 with one, 0.6 with two.
- `model_self_report`: the model's own number, clamped. Weighted least because models are
  over-confident; it can move the score by at most 0.10.
- A `partial` answer is capped just below the `high` band. Non-answers (refusal, error,
  retrieval-only) score 0.

Bands: high at 0.80, medium at 0.55. `needs_human_review` is false only when the status is
`answered`, the band is high, every citation is verified, no tool failed and no repair was
needed. The factors are returned in `trace.confidence_factors` so a reviewer sees *why* a number
is what it is. Observed live: the HCC answer scored 1.00 (high, no review); the fault-injected
answer scored 0.76 (medium, review) because of the repair penalty; a scoped policy answer scored
0.78 because retrieval strength was lower inside one document.

- **Why these weights.** Retrieval strength and verified citations are the two signals that are
  both mechanical and predictive, so they carry 70 percent. Tools and repairs are strong
  negatives when present but usually absent. The self-report is kept because it is the only
  signal that sees the model's own uncertainty, and bounded because it is the least trustworthy.
- **How it fails.** It is a heuristic, not a calibrated probability. Step 6's eval will check
  whether high-band answers are actually right more often than medium ones; if not, the weights
  change or the bands move.

### D10. Error handling catalogue

| Failure mode | Where detected | Behaviour | Output | Status |
|---|---|---|---|---|
| Empty retrieval | retriever | No LLM call | `insufficient_context` envelope, `trace.gate_reason` | done |
| Low-score retrieval | `evaluate_gate` | No LLM call (16 ms, $0 observed) | `insufficient_context` with reason | done |
| Tool exception / timeout / bad args | tool registry | Error returned to model as tool result; warning added | `tool_calls[i].error`, review flag, confidence penalty | done |
| Malformed JSON / schema violation | agent layer one | Repair, then salvage, then error envelope | `error` only if nothing salvageable | done |
| Invalid citation / unverifiable evidence | agent layer two | Repair (naming tool-result text), then salvage to `partial` | `partial` + warning, or `error` | done |
| Rate limit / timeout / 5xx | llm client | Exponential backoff with jitter, bounded | `LLMUnavailable` -> error envelope | done |
| Auth / bad request | llm client | No retry | `LLMRequestError` -> error envelope | done |
| Context budget overflow | retriever | Drop lowest-ranked; anchored chunks protected | `dropped_chunk_ids` in trace, warning | done |
| Tool rounds exhausted | agent | Force final answer with `tool_choice="none"` | `forced_final` in trace, warning | done |
| Model refuses but writes prose | guardrails | Answer text blanked | `answer == ""`, warning | done |
| Required tool not called (question matches tool policy) | guardrails `required_tools_missing` | Cap below high band, flag | warning, `needs_human_review = true` | done |
| Refusal caused by an unrelated tool result | guardrails `tool_induced_refusal` via agent | One targeted repair | answered with the finding in `caveats`, or refusal kept | done |
| No API key | service | Retrieval-only mode | `retrieval_only` with evidence citations | done |
| Query empty or too long | guardrails | Raise `InvalidQueryError` | HTTP 400 in the API | done |
| Query contains PHI patterns | guardrails | Flag (mock) or block (real) | `warnings[]` or error envelope | done |

### D11. PHI / PII handling if this pointed at real data

Mock data is obviously synthetic (names like "Jane Doe-Test", `TEST-` MRNs, `M-TEST-` member ids,
a synthetic banner on every file). With real charts: de-identify before embedding and prompting
(Presidio or Philter, reversible mapping in a vault); local embeddings keep the corpus on-premise
and only retrieved chunks reach the LLM, under a BAA with a zero-retention endpoint; never log
chunk text (`LOG_CHUNK_TEXT=false`); tenant and patient scoping as a retrieval filter (the
`doc_ids` parameter exists for this), not as a prompt instruction; immutable audit log of every
query, retrieved id set, tool call and answer (policy_002 section 3.4 mirrors this for
eligibility lookups); encryption at rest and in transit; deletion by document id is a filter, not
a search, thanks to deterministic chunk ids.

### D12. Where a human stays in the loop, and why

Before anything touches a chart or a payer submission (policy_001 section 7 in the corpus says
so: automated extraction is a recommendation). Whenever `needs_human_review` is true. On policy
interpretation. On every eval regression. The Step 4 runs give a concrete reason: a model can
produce an answer in which every quote is real and the conclusion is still misattributed.

### D13. Surfaces: one core, three thin shells

`QueryService` (step 5) is the only place logic lives. FastAPI exposes it over HTTP, the CLI runs
it for scripted demos and evals, Streamlit imports it in-process. The CLI already exists for
`ingest` and `search`, with `--json` on every command so the eval runner and the tests can drive it.

### D14. Scaling posture (standing requirement)

- **Stateless query path.** Retriever, agent and client hold only configuration and counters;
  N API workers behind a load balancer need no coordination.
- **Write path separated from read path.** `pipeline.build_index` is the ingestion job; it
  writes a fingerprint-named index directory. Query workers `load()` it. A new corpus version is
  a new directory, so switch-over is atomic and roll-back is a config change.
- **Incremental embedding.** `store.build()` embeds only chunks without vectors.
- **Storage behind a protocol.** `SearchIndex` is the seam for FAISS, pgvector or a hosted store.
- **Bounded fan-out to the provider.** Retries are bounded and jittered now; a per-process
  semaphore and a per-tenant rate limit belong in the service (step 5).
- **Cost control.** Retrieval gate refuses for free; anchoring is capped at 900 tokens per
  document; identical tool calls to be memoised per request; repairs bounded at 2.
- **Observability.** Every result carries usage, estimated cost, latency, retrieval scores and
  repair counts; the eval runner will aggregate them.

---

## Part 3: Build plan and status

| Step | Scope | Status |
|---|---|---|
| 1 | Design, scaffold, config, virtual environment, synthetic corpus, these notes | done |
| 2 | Loader, chunker, embedder, hybrid store with persistence, pipeline with cache, retriever with gate; CLI `ingest`/`search`; 54 tests | done |
| 3 | `ModelOutput` / `GroundedAnswer` schemas, truth validation, four tools, registry with timeouts; 46 tests | done |
| 4 | LLM client, prompts v1/v2/v3 with recorded changelog, agent loop, repair, fault injection, live runs; retrieval fixes found live (per-doc cap, anchoring); 131 fast tests in total | done |
| 5 | Guardrails (query validation, PHI scrub, refusal blanking, citation salvage, confidence composite), `QueryService` with answer cache and bounded concurrency, tool-call memoisation, CLI `ask`, live service run; 174 tests in total | done |
| 6 | Eval set (10 cases) and runner with full and retrieval modes; four eval-driven fixes (prompt v4, `caveats` field, tool policy, tool-induced-refusal repair); six recorded runs; 199 tests in total | done |
| 7 | FastAPI endpoint (`/health`, `/documents`, `/search`, `/query`), Streamlit UI, `raapid-rag serve` and `ui`, process-level smoke test over HTTP; 215 tests in total | done |
| 8 | `DESIGN_NOTE.md` (two pages), README as the run guide (Streamlit, FastAPI, CLI, library, Docker, scripts), recorded final CLI transcript, Dockerfile and compose (written, not built here), version 1.0.0 | done |

---

## Part 4: Step logs

### Step 1: Design, scaffold, configuration, synthetic corpus

**What exists after this step**: `.venv`, `.env.example`, `requirements.txt`, `pyproject.toml`,
`README.md`, `data/corpus/` (8 documents), `src/raapid_rag/config.py`, this file.

**Why a `Settings` class.** Types validated at startup; bounds enforced (`chunk_target_tokens`
cannot exceed 256, encoding the embedding-model constraint in the type system); every knob
documented in one place; cached singleton; tests construct `Settings(_env_file=None, ...)` with
overrides. `openai_api_key` is optional so ingestion, retrieval and tests run with no key.

**Corpus design.** 8 documents, 2,930 words, 4,693 tokens. Front matter is plain `key: value`
(no YAML dependency). Each document carries a deliberate trap for the eval set:

| Doc | Trap or hook |
|---|---|
| note_001 | Two A1c values (prior vs current); codes E11.22, N18.31, E11.42 for the ICD-10 tool; "LDL not collected" for a refusal |
| note_002 | Admission and discharge dates (length of stay 5 days); Hospital Course is 433 word pieces to exercise windowing; several codes for HCC hierarchy questions |
| note_003 | "History of" breast cancer must be Z85.3, not active cancer; no blood-pressure medication anywhere |
| note_004 | Audio-only telehealth (not acceptable per policy_002 2.2); "possible rheumatoid arthritis" (not codable per guideline_001 2.1); member retro-terminated in the eligibility tool |
| policy_001 | Section 7: automated output is a recommendation (anchors the human-in-the-loop argument) |
| policy_002 | Eligibility rule pairs with the tool; 90-day window pairs with days_between; identifier-heavy text tests BM25 and anchoring (`RA-110`) |
| guideline_001 | Worked examples matching note_003 and note_004 |
| guideline_002 | HCC labels mirrored by the mock ICD-10 tool; hierarchy logic for multi-hop questions |

**Mock-data hygiene.** Fake names, `TEST-` MRNs, `M-TEST-` member ids, repeated-digit DOBs, a
synthetic banner on every file.

### Step 2: Ingestion and retrieval

**What was built.** `ingestion/models.py`, `loader.py`, `chunker.py`, `embedder.py`, `store.py`,
`pipeline.py`; `retrieval/retriever.py`; `cli.py` (`ingest`, `search`); `eval/retrieval_probe.py`;
54 tests. Test-first throughout: every module's tests were written and seen failing (import
error) before the module existed.

**Loader, in plain language.** Reads each `*.md`, decodes strictly as UTF-8 (a replaced character
inside a dosage is a safety problem, so decoding errors are fatal), parses the front matter with a
20-line parser, requires `doc_id`, `doc_type`, `title`, rejects duplicate `doc_id`s across files
(colliding chunk ids would make citations ambiguous), skips empty bodies with a warning, ignores
non-Markdown files. Every error names the file.

**Chunker, in plain language.** Finds `## ` headers with a regex; the text before the first header
is a "Preamble" section; a body with no headers is one section named after the document title.
Each section is stripped and, if it fits the target, becomes one chunk. Otherwise it is windowed
(see D5). The token counter is injected: the real pipeline passes the embedder's word-piece
counter; tests pass a whitespace counter so they run in milliseconds. Ten tests cover: sequential
ids, windows within target with overlap, progress when overlap exceeds the window, word-splitting
of an over-long sentence, no-header bodies, preamble chunks, the verbatim-substring invariant,
`embed_text` prefixing, metadata propagation, and skipping empty sections.

**Embedder.** `Embedder` protocol with `name`, `dim`, `embed`, `count_tokens`. `HashingEmbedder`
is a real, weak bag-of-words embedder (CRC32 buckets, unit-normalised, zero rows stay zero so no
NaN) used in tests. `SentenceTransformerEmbedder` loads MiniLM lazily, encodes in batches with
normalisation, and counts word pieces with `verbose=False` so the tokenizer does not warn about
the very over-length it is being asked to measure. `name` lets a persisted index refuse to load
under a different model.

**Store.** `InMemoryHybridStore`: `add` (rejects duplicate chunk ids), `build` (embeds only
pending chunks; rebuilds BM25 fully), `dense_scores`, `bm25_scores`, `search(query, k, doc_ids)`
with RRF fusion and an optional document filter, `save`/`load` (vectors as `.npy`, chunks and
metadata as JSON, embedder-name check). BM25 tokenisation keeps `e11.22`, `m-test-1002`,
`ra-110`, `2025-03-12` and `note_001` as single tokens; a standard `\w+` tokenizer would split
`E11.22` into `E11` and `22` and lose the code. Tested: exact-code ranking, RRF maximum
`2/(k+1)` for a chunk first in both rankers, zero BM25 means no lexical rank, incremental
embedding counts, save/load round trip, embedder mismatch rejection, document scoping.

**Pipeline.** `build_index(settings, embedder, corpus_dir, cache_dir)` loads, fingerprints (SHA-256
over doc ids, bodies, metadata, chunk parameters, RRF constant, embedder name, schema version),
loads from `cache_dir/<fingerprint>` when present, otherwise chunks, embeds, saves. Returns the
store and an `IndexBuildReport`. Tested: cache hit embeds nothing; editing one document changes
the fingerprint and rebuilds; real corpus yields 8 documents and more than 60 chunks with the
Hospital Course windowed and no chunk over 220 tokens.

**Retriever.** See D4 for the final design. `RetrievalResult.to_dict()` is what the API and UI
will show: per-chunk ids, sections, dates, texts, all three scores and both ranks, whether the
chunk came from anchoring, the gate decision and reason, dropped ids, context tokens, latency.

**Measured.** Cold index build 12.8 s including the model load; 62 chunks. Gate probe (13 queries)
matches expectations on all 13 after calibration. Two probes that pass the gate deliberately
("in-corpus fact missing", "canary") must be refused by the model, and are (Step 4).

**Edge cases handled.** Malformed or unterminated front matter; missing keys; duplicate ids;
non-UTF-8; empty body; no headers; empty sections; a sentence longer than the window; overlap
larger than the window; a token budget smaller than the first chunk (`context_budget_too_small`);
empty store; `k` larger than the store; `doc_ids=[]` (empty result, not "no filter").

**How it breaks under load, beyond D4.** The fingerprint hashes every document body on each
`build_index` call: O(corpus) per startup, fine at megabytes, and at gigabytes the ingestion job
should track per-document hashes instead. `load()` reads all chunk text into memory: 100k chunks
of 220 tokens is about 100 MB of text, acceptable; beyond that, chunk text moves to a database and
the index keeps ids only.

### Step 3: Output contract and tools

**What was built.** `schema.py`, `tools/icd10.py`, `eligibility.py`, `calculator.py`,
`registry.py`; 46 tests.

**Schema, in plain language.** `ModelOutput` has six fields and no defaults. The strict-schema
test walks the generated JSON schema and asserts every object is closed and fully required and
no `default` key exists anywhere (the first version of the test searched for the substring
"default" and tripped on the docstring; the fix was to check keys). `validate_grounding` returns a
list of sentences, each one a repair instruction. `GroundedAnswer` adds `schema_version`, the
query, enriched citations (doc title, section, date, `verified` flag), tool records, computed
confidence and band, review flag, warnings, errors and a free-form trace. Two constructors,
`error_envelope` and `insufficient_context_envelope`, guarantee the service can always answer
with a valid object.

**Tools, in plain language.** The ICD-10 table holds 37 codes with accurate ICD-10-CM
descriptions and *illustrative* HCC categories, hierarchies and severity ranks that mirror
guideline_002; every result carries a disclaimer. Exact code lookup is case and whitespace
tolerant; term search matches descriptions and returns up to 8. The eligibility table has four
members; M-TEST-1004 is retroactively terminated on 2025-03-31 so the 2025-04-02 telehealth note
fails eligibility for a second, independent reason. Every response is stamped `checked_at` for
audit. The calculator walks the AST and evaluates only numbers, arithmetic operators, unary signs
and four whitelisted functions; attribute access, names, subscripts, strings and imports are
rejected before evaluation; exponents above 1000 and expressions above 200 characters are
refused. `days_between` is signed.

**Registry, in plain language.** `ToolSpec(name, description, parameters, fn)`. `openai_tools()`
emits strict function definitions. `dispatch` parses the JSON arguments, validates required and
unexpected keys, runs the function in a thread pool with `future.result(timeout=...)`, and
returns a `ToolCallRecord` in every case: success with result and latency, or error text for
unknown tool, bad JSON, bad arguments, timeout (`timed_out=True`) or exception (type and message,
never a traceback). Descriptions are written for the model: each says *when* to call the tool.

**Edge cases handled.** Empty ICD-10 query; unknown code; codes without a payment HCC (`None`, not
missing); non-ISO dates; unknown member; member id case; division by zero; syntax errors; unsafe
expressions (parametrised over seven attack shapes); duplicate tool registration; arguments that
are valid JSON but not an object.

### Step 4: LLM client, prompts, agent loop, live runs

**What was built.** `llm/client.py`, `llm/prompting.py`, `llm/prompts/system_v1.md`, `v2`, `v3`,
`PROMPT_CHANGELOG.md`; `agent.py`; `eval/step4_live_run.py`; recorded runs in `docs/runs/`;
retrieval fixes (per-document cap, document anchoring, explicit scope) with tests; 131 fast tests
in total.

**LLM client, in plain language.** Wraps `openai.OpenAI` with the SDK's own retries disabled so
every retry is ours and counted. Retries only transient errors (429, timeout, connection, 5xx)
with exponential backoff and jitter, bounded by `LLM_MAX_RETRIES`; raises `LLMUnavailable` when
exhausted and `LLMRequestError` immediately for 4xx. Normalises the SDK response into three small
dataclasses so the agent never touches SDK objects and can be tested with a scripted fake.
Accumulates usage and estimates cost from a small price table. Tested with a fake OpenAI client:
normalisation, tool-call normalisation, request parameters, retry-then-succeed, exhausted retries,
non-retryable errors, missing key, cost estimate.

**Agent, in plain language.** See D6 and D7 and the module docstring. The loop is about 60 lines.
Tested with a scripted fake LLM and the real registry: happy path (asserting on the exact
messages the model saw), tool round (asserting the assistant `tool_calls` message and the `tool`
message with matching id), tool error fed back not raised, bounded tool rounds with forced final
answer, repair on bad citation (asserting the repair message lists the bad id and the valid ids),
repair on malformed JSON, repair on non-verbatim evidence, tool-result text named in the repair,
exhausted repairs, fault injection in both modes, refusal to run with no chunks.

**Prompt iteration.** `PROMPT_CHANGELOG.md` records v1, v2 and v3 with the observed behaviour
that motivated each change and the recorded run files. Short version: v1 leaked training
knowledge on the canary and quoted a header line; v2 stopped both and halved cost but over-refused
two multi-document questions and stopped calling tools; v3 restored multi-document reasoning and
mandatory tool use. Prompting could not fix retrieval recall (anchoring did) or the model
repeating identical tool calls (memoisation will).

**Live-run measurements (gpt-4o-mini, six questions).**

| run | calls | prompt tokens | est. cost | notable |
|---|---|---|---|---|
| v1 | 21 | 38.8k | $0.0071 | canary leak; misattributed guideline example |
| v2 | 8 | 17.8k | $0.0032 | 2 over-refusals; no tools |
| v3, cap only | 11 | 27.7k | $0.0048 | HCC answered from guideline examples (misattribution) |
| v3 + anchoring | 14 | 48.1k | $0.0085 | HCC correct; telehealth unrepaired (tool text cited x3) |
| v3 + anchoring, telehealth retry | 5 | 15.3k | $0.0026 | answered, 4 verified citations, eligibility called |
| fault injection (v2) | 13 | 30.2k | $0.0053 | 6 of 6 recovered with exactly 1 repair |

**Three failures found only because every message is visible.**

1. The model quoted the context *header line* as evidence. Fix: header text is not part of the
   chunk, the verbatim check rejects it, and v3 says only the body is quotable.
2. The model cited a tool result sentence as evidence for a policy chunk and kept doing so
   through generic repairs. Fix: the repair message now identifies the tool the text came from.
3. The model attributed guideline example codes to the patient. Every quote was verbatim. Fix:
   anchoring puts the patient's Assessment in context; v3 requires citing rule and fact. Residual
   risk documented in D6.

**Edge cases handled.** Tool calls when forced final (ignored, content parsed); model returning
tool calls and content together (tools first); JSON that parses but fails the schema (errors
listed field by field); repair budget exhausted (raw outputs and errors preserved in the
result); fault injection on a non-JSON response (falls back to truncation).

**How it breaks under load.** Each tool round is another provider round trip; a chatty model
can turn one question into five calls (observed). `MAX_TOOL_ROUNDS` bounds it, memoisation
(Step 5) removes the repeats. The thread pool for tools is 8 workers per process; a slow real
tool under load queues behind it, so real tools need their own client-side timeouts and a
circuit breaker. The agent keeps the full message transcript in memory for the trace: a few
kilobytes per request, fine, but it must never be logged with real data (`LOG_CHUNK_TEXT`).

### Step 5: Guardrails, the service facade, memoisation, CLI `ask`

**What was built.** `guardrails.py`, `service.py`, tool-call memoisation and `last_output` in
`agent.py`, `ToolCallRecord.cached`, `Retriever.index_size`, the CLI `ask` command,
`eval/step5_live_run.py`, ten new config knobs; 42 new tests (174 in total). Every new behaviour
had a failing test first.

**Guardrails, in plain language.** Five pure functions, none of which touch the model:

- `validate_query` rejects empty or over-long questions with `InvalidQueryError` (the API will
  turn that into a 400; everything else in the system returns a valid envelope instead of
  raising).
- `scrub_query` looks for four identifier *formats* (SSN, phone, MM/DD/YYYY, email) and either
  warns or blocks, per `PHI_POLICY`. Formats, not names: regex has high recall on formats and no
  recall on names; a real deployment adds an entity recogniser. ISO dates, decimals, codes and
  member ids do not match, which is tested, because they are normal clinical content.
- `enrich_citations` adds document title, section and date to each citation and computes
  `verified` by the same normalised-substring rule the validator uses. An unknown chunk id gets
  `doc_id="unknown"` and `verified=False` rather than an exception.
- `blank_answer_on_refusal` empties the answer when the status is `insufficient_context` and
  adds a warning. This exists because prompt v1 wrote "however, it is generally recommended
  that..." after refusing; the guarantee that no uncited prose reaches the caller must not
  depend on the prompt.
- `salvage_citations` handles "repairs exhausted": drop unverifiable citations, downgrade
  `answered` to `partial`, warn; if nothing verifiable survives, return `None` and the service
  emits an error envelope. Observed live in this very step: the model cited the eligibility
  tool's sentence for a policy chunk three times despite a repair message naming the cause; the
  salvage path returned a `partial` answer with four verified citations and a review flag rather
  than throwing the answer away.
- `compute_confidence` is decision D9.

**Service, in plain language.** `QueryService.ask` is the request path in one screen: validate,
scrub, cache lookup, retrieve, gate, agent (inside a semaphore), salvage if needed, blank refusal
prose, enrich citations, confidence, envelope, cache store. `from_settings` builds the whole
stack (index from the fingerprint cache, retriever, tools, model client if a key exists, prompt
by version). `health()` reports chunk count, fingerprint, model, prompt version and cache size
for a load balancer or a status page.

- **Answer cache.** An LRU keyed by the normalised question (case-folded, whitespace-collapsed),
  the sorted scope and the index fingerprint. Thousands of users ask the same policy questions;
  after the first, each costs nothing. Error envelopes are never cached (they are transient).
  A hit returns a copy with `trace.cache_hit = true` and its own latency. A new corpus version
  changes the fingerprint and therefore misses the cache by construction.
- **Bounded concurrency.** A per-process semaphore (`LLM_MAX_CONCURRENCY`, default 8) around
  the agent call. Under a burst of users, requests queue in-process instead of stampeding the
  provider's rate limit; more capacity means more processes behind the load balancer.
- **Memoised tool calls.** Inside one request, an identical tool call (same name, same
  arguments after key sorting) is answered from the first result and marked `cached`. Failures
  are not memoised because a retry may succeed. Observed motivation: `days_between` asked three
  times in one run.
- **Retrieval-only mode.** With no key (or `--no-llm`) the service still returns the retrieved
  evidence as verified citations with status `retrieval_only`, so the system degrades to a
  search tool rather than failing.

**CLI `ask`.** Prints status, confidence and band, the review flag, the answer, a citations
table with the verified column, missing information, tool calls (with cached and error marks),
warnings, errors and a one-line trace; `--json` prints the full `GroundedAnswer`; `--doc-id`
scopes; `--no-llm` forces retrieval-only.

**Live service run (`docs/runs/step5_live_run.json`, gpt-4o-mini, prompt v3).**

| case | status | confidence | review | citations verified | tools | calls | cost | latency |
|---|---|---|---|---|---|---|---|---|
| gate refusal (gibberish) | insufficient_context | 0.00 low | yes | 0 | none | 0 | $0 | 16 ms |
| canary (metformin dose) | insufficient_context | 0.00 low | yes | 0 | none | 1 | $0.0004 | 3.2 s |
| HCC mapping (Jane) | answered | 1.00 high | no | 3 of 3 | icd10 x2 | 2 | $0.0013 | 5.3 s |
| telehealth submittable | partial (salvaged) | 0.79 medium | yes | 4 of 4 after salvage | eligibility, icd10 x3 | 7 | $0.0043 | 14.7 s |
| PHI in query | answered | 0.89 high | no | 1 of 1 | none | 1 | $0.0006 | 1.8 s, warning attached |
| scoped to policy_002 | answered | 0.78 medium | yes | 1 of 1 | none | 1 | $0.0004 | 2.1 s |
| cache hit (HCC again) | answered | 1.00 high | no | 3 of 3 | (cached) | 0 new | $0 | milliseconds |
| fault injection | answered | 0.76 medium | yes | 2 of 2 | none | 2 | $0.0012 | 4.1 s, 1 repair |

**Why the telehealth case matters.** The model kept citing the eligibility tool's sentence as
evidence even after the repair message named the tool. That is a limit of a small model's
instruction following. The system's answer was still correct, still fully verified after
salvage, and flagged for review. Compare Step 4, where the same behaviour produced an error
envelope: the salvage path is the difference between "no answer" and "a usable answer with a
warning". It also hit the tool-round limit, which is why the confidence is medium rather than
high.

**Edge cases handled.** Empty or whitespace query; query over the limit; PHI under both
policies; unknown chunk id in salvage; refusal with and without prose; cache disabled (size 0);
cache key insensitive to case and whitespace; identical tool calls with differently ordered
arguments; a tool failure is not memoised; retrieval-only with no model; provider outage after
retries; salvage with zero survivors.

**How Step 5 breaks under load.** The cache is per process and unbounded in bytes (bounded in
entries); at scale it becomes Redis with a TTL and the same key. The semaphore is per process;
a global rate limit needs a shared token bucket at the gateway. `from_settings` loads the
embedding model (12 to 15 s), so a process must warm up before it joins the load balancer
(`health()` is the readiness probe). The salvage path cannot recover an answer with zero
verifiable citations, by design.

### Step 6: Evaluation, and what it changed

**What was built.** `eval/eval_set.json` (10 cases), `evaluation.py` (importable harness),
`eval/run_eval.py`, the CLI `eval` command with `--mode full|retrieval` and `--min-pass-rate`
(exit code 1 below it, so it can gate CI), reports written to `docs/runs/eval_<mode>_latest.json`
and `.md`. Then four fixes the eval forced: prompt v4, a `caveats` field in the model's output
schema, a tool policy in the service, and a tool-induced-refusal repair in the agent. 25 new
tests (199 in total).

**Eval design, in plain language.** Each case has a question and an `expected` block: acceptable
statuses; documents that must appear among the citations (any-of and all-of); case-insensitive
keywords the answer must contain (alternatives separated by `|`) or must not; tools that must
have been called; a maximum number of model calls (0 for the gate case); and `no_citations` for
refusals. Every check is mechanical and a reviewer can verify it by eye; there is no LLM judge.
The ten cases cover: a single-document fact with two dated values, a two-document question with a
tool (HCC mapping), a two-tool question (length of stay and eligibility), a rule-applied-to-facts
question (telehealth), two guideline-application questions (uncertain diagnosis, historical
cancer), an identifier-driven retrieval (policy RA-110 section 2.2), a parametric canary, an
in-corpus missing fact, and an off-corpus question that must be refused by the gate with zero
model calls.

**Two modes.** `full` runs the model and scores everything. `retrieval` runs with no key and
scores only what retrieval can answer: did the gate behave as expected, and were the expected
documents retrieved. Retrieval mode passed 10 of 10 with the real embedder in 13 ms per case;
it is the free regression check.

**Metrics reported.** Pass rate; refusal precision (of the refusals, how many were expected)
and refusal recall (of the expected refusals, how many happened); calibration by confidence
band (for answered/partial results, how often each band was correct); pass rate by category;
which checks failed and how often; total cost and tokens; mean and p95 latency.

**What the eval found, run by run** (every report is kept in `docs/runs/`; the prompt changelog
has the same story from the prompt's side):

| configuration | runs | pass | refusal P / R | high band correct | cost / run | mean, p95 latency | what failed |
|---|---|---|---|---|---|---|---|
| v3 | 1 | 9/10 | 0.75 / 1.0 | 6 of 6 | $0.0113 | 3.9 s, 13.3 s | uncertain_dx refused after an eligibility-tool loop |
| v4 draft (soft tool wording) | 1 | 9/10 | 1.0 / 1.0 | 6 of 7 | $0.0082 | 2.9 s, 7.6 s | hcc_mapping answered at 1.00 without the ICD-10 tool |
| v4 with MUST for both tools | 2 | 8/10 twice | 0.6 / 1.0 | 5 of 5 | $0.0087 | 3.0 to 3.3 s, 5.9 to 7.2 s | telehealth and uncertain_dx refused over side issues |
| v4 + `caveats` field, tool policy | 2 | 10/10, 9/10 | 1.0 / 1.0; 0.75 / 1.0 | 6 of 6, 5 of 5 | $0.0075 | 3.0 to 3.2 s, 6.5 to 8.6 s | uncertain_dx refused once (inactive coverage) |
| + tool-induced-refusal repair (shipped) | 2 | 9/10, 10/10 | 1.0 / 1.0 | 5 of 5, 6 of 6 | $0.0091, $0.0111 | 3.4 to 3.9 s, 11.2 to 16.9 s | uncertain_dx once salvaged to partial, flagged |

**The four fixes, and why each is code rather than prose.**

1. *Prompt v4.* Narrowed the eligibility mandate, added "a concern is never a reason to refuse"
   and "applying a rule in the context to facts in the context is reporting, not a coding
   decision". Prompt changes alone moved failures around rather than removing them; see the
   MUST runs.
2. *`caveats` in the output schema.* The model kept refusing because it had a side concern and
   nowhere to put it. Giving it a required field for "concerns that do not change the answer"
   is structural: the schema itself offers the outlet. Over-refusals dropped immediately.
3. *Tool policy* (`required_tools_missing`). The eval's `tools_any` check caught an answer at
   confidence 1.00 that skipped the ICD-10 tool. The confidence composite cannot see that, so a
   deterministic rule now does: a question matching a tool's pattern whose answer did not use
   that tool is capped below the high band and flagged. In the shipped runs this is why the
   telehealth answer, correct but without the eligibility check, sits at medium with a review
   flag rather than at high.
4. *Tool-induced-refusal repair* (`tool_induced_refusal`). When the model refuses, the reason
   mentions a tool's topic, the tool ran, and the question never asked about that topic, one
   targeted repair message is sent. It fired in run 4 and turned the refusal into a correct
   answer flagged for review.

**Calibration.** Across the four runs of the shipped family the high band was correct 22 of 22
times; every failure was a refusal or a medium-band answer, so every failure carried
`needs_human_review = true`. That is the property a reviewer cares about: a high-confidence,
no-review answer has not yet been wrong.

**Nondeterminism.** Identical runs differ (10 of 10 then 9 of 10 with the same code). The eval
is therefore run more than once and every run is kept. A single run is evidence, not proof;
a CI gate at `--min-pass-rate 0.8` tolerates one flaky case.

**Edge cases handled.** Duplicate case ids rejected; keyword alternatives; refusals that must
not cite; the gate case's zero-call requirement; retrieval-mode scoring skips checks it cannot
judge (model-side refusals); cost and latency of a cache hit (zero and milliseconds) are recorded
honestly; warnings, missing information, caveats and tools appear in the report so a failure can
be diagnosed from the report alone.

**How Step 6 breaks.** Ten cases are a smoke test, not a benchmark; thresholds tuned on them
may be overfit. The keyword checks are brittle to wording (an answer saying "is not permitted"
would fail a check expecting "not acceptable|cannot"). Each full run costs about a cent and
40 seconds; at a hundred cases run nightly that is still trivial. The report shows the logging
pitfall found while running it: warnings go to stderr, so when piping `--json` output, do not
merge stderr into stdout.

### Step 7: HTTP API and Streamlit UI

**What was built.** `api.py` (FastAPI), `ui/app.py` (Streamlit), `Retriever.documents()`,
`QueryService.documents()` and `QueryService.search()`, CLI `serve` and `ui`, a shared test
helper module (`tests/helpers.py`) with the fake model, `eval/step7_smoke.py` (real servers over
HTTP); 16 new tests (215 in total), including three headless Streamlit tests.

**API design decisions, in plain language.**

- *Every model outcome is a 200.* An answer, a refusal, a salvaged partial and a provider outage
  all return the same envelope with `answer_status` set. A refusal is a correct result, not an
  HTTP error, and a client that switches on status codes would otherwise treat the system's most
  important behaviour as a failure. Only unusable input is a 4xx: 400 for an empty or over-long
  question (our `InvalidQueryError`), 422 for a malformed body (pydantic).
- *Sync handlers on purpose.* The endpoints are plain `def` functions, so FastAPI runs each in
  its threadpool and a five-second model call never blocks the event loop. The service's
  semaphore bounds how many of those threads talk to the provider at once. Scaling is more
  processes (`raapid-rag serve --workers N`), each loading its own index once.
- *Request ids.* Middleware stamps `X-Request-ID` on every response and the same id is written
  into `trace.request_id`, so a reviewer's screenshot can be matched to a log line. Clients may
  supply their own id for end-to-end tracing.
- *No question text in logs.* The access log records method, path, status, latency and request
  id. With real data the question itself may contain PHI.
- *App factory.* `create_app(service)` lets tests inject a service with a fake model;
  `create_app()` builds the real one at startup via the lifespan hook, so `uvicorn
  raapid_rag.api:app` just works and `/health` is the readiness probe.
- *`include_trace=false`.* Returns only summary trace fields (no chunk text) for callers that
  should not receive source passages.
- *`/search` and `/documents`.* Retrieval-only inspection and the document list for scope
  pickers; both reuse the service, so the UI and the API can never disagree.

**UI design decisions.** One page, reviewer-first ordering: status, confidence and the review
flag before anything else; then the answer; caveats and missing information; each citation with
its verification mark and an expander holding the exact source chunk; tool calls with results;
warnings; a trace panel (retrieval scores with the anchored flag, gate reason, calls, tokens,
cost, latency, cache hit, confidence factors, model reasoning summary) and a JSON download.
Sidebar: model toggle (off means retrieval-only, no key needed), fault-injection selector for
demonstrating the repair path, scope multiselect fed by `documents()`, index and prompt state.
Example-question chips seed the input. A session history keeps earlier answers in expanders. The
service is built once per process with `st.cache_resource` and imported in-process, so one
command runs the demo. Three environment switches exist for tests and key-less demos:
`RAAPID_UI_EMBEDDER=hashing`, `RAAPID_UI_USE_LLM=0`, `RAAPID_UI_CACHE_DIR`.

**How the UI is tested.** `streamlit.testing.v1.AppTest` runs the script headlessly, types into
the question box by widget key, clicks Ask, and inspects the rendered markdown. Three tests: the
page renders and shows index state; a retrieval-only answer shows `retrieval_only`, the review
flag and the cited policy; a gibberish question shows `insufficient_context`.

**Smoke test over real processes** (`docs/runs/step7_api_smoke.json`).

| step | result |
|---|---|
| uvicorn ready (`/health` 200) | 13.8 s after start (embedding model load) |
| Streamlit ready (`/_stcore/health` = ok) | immediately after |
| `POST /query` HCC question | 200, `answered`, confidence 1.00, no review, two ICD-10 lookups, 5.7 s |
| `POST /query` off-corpus question | 200, `insufficient_context`, 17 ms, no model call |
| `POST /query` scoped to policy_002, `include_trace=false` | 200, `answered`, 0.78 medium, review flag, trace without chunk text, 2.2 s |
| `POST /query` blank question | 400, `"query must not be empty"`, 2 ms |
| every response | `X-Request-ID` header present and echoed in the trace |

**Edge cases handled.** Blank versus missing query (400 versus 422); provider outage through
the API (200 error envelope, never a 500); scope with no matching chunks (gate refusal); trace
suppression keeps the request id; client-supplied request id honoured; UI with no key (toggle
disabled, retrieval-only); UI example chips setting the input before the widget renders.

**How Step 7 breaks under load.** Streamlit is a single-process demo tool: every widget change
re-runs the script, and `st.cache_resource` is per process; it is not the production front end.
The API has no authentication, no rate limiting and no CORS configuration; in production it sits
behind a gateway that adds all three, and tenant scoping becomes a mandatory `doc_ids` filter
derived from the caller's identity rather than an optional field. Each uvicorn worker loads its
own copy of the model and index (about 14 s and a few hundred megabytes), so worker count is a
memory decision and readiness must gate traffic. The smoke test found one hygiene issue: Streamlit
prints the machine's network and external URLs at startup, so server logs must not be written
into shared folders.

### Step 8: Design note, run guide, final transcript

**What was produced.**

- `docs/DESIGN_NOTE.md`: the two-page submission note. Sections: what it does, architecture
  (write path, read path, reasoning, validation and repair, surfaces), prompt and reasoning design
  (v1 to v4 in one paragraph), grounding and guardrails (six layers, confidence, where the human
  sits), evaluation results (the configuration table, calibration, fault injection), cost,
  latency and scale, PHI and PII, known limitations, productionizing inside RAAPID (plug-in
  points, path, next engineering steps). Every number in it comes from a recorded run.
- `README.md` rewritten as the run guide: setup for PowerShell and bash, Streamlit, FastAPI (with
  `curl`, PowerShell and Python client examples and the endpoint table), CLI, Python library,
  scripted runs, Docker, configuration, layout.
- `docs/runs/final_transcript.md`: a captured CLI session (index build, retrieval-only search,
  five `ask` calls covering a fact, two tools, a rule-plus-facts question, a canary refusal and a
  gate refusal, then a fault-injected ask showing the repair). Stderr log lines are omitted.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: the API and the UI from one image with the
  embedding model and the prebuilt index baked in so containers start offline. **Not built in
  the development environment** (the CPU torch wheel is large); reviewed by hand and stated as
  such in the files and the README.
- Version bumped to 1.0.0.

**How to run it, in one place.** The README is the canonical guide. The five ways:

| Method | Command | When to use it |
|---|---|---|
| Streamlit UI | `raapid-rag ui` | demos and reviewer walkthroughs; one process, no key needed with the model toggle off |
| HTTP API | `raapid-rag serve --port 8000`, then `POST /query` | integration by other services; docs at `/docs`, readiness at `/health` |
| CLI | `raapid-rag ask "..."`, `search`, `ingest`, `eval` | scripting, quick checks, CI (`eval --mode retrieval`) |
| Python library | `QueryService.from_settings(get_settings(), cache_dir=DEFAULT_CACHE_DIR).ask("...")` | embedding the service in another Python process |
| Docker | `docker compose up` | reproducible deployment of API (8000) and UI (8501) |

**Transcript observations.** In this captured run the telehealth question was answered at 0.89
high with no review flag because the model called the eligibility tool this time; in the eval
runs the same question sometimes skipped the tool and landed at medium with a review flag. Both
outcomes are correct answers; the difference is whether the deterministic tool policy had to step
in. That variability, and the system's honest reporting of it, is the point to make when showing
the transcript.

**What was left out, deliberately.** No authentication or multi-tenancy (gateway concerns in
production); no streaming responses; no real de-identification pipeline; the Docker image was
not built here. Each is named in the design note's limitations.

---

## Part 5: Glossary

- **RAG (Retrieval-Augmented Generation).** Find relevant text first, then have the model answer
  using only that text.
- **Chunk.** A piece of a document small enough to embed and cite. Here: one section, or a
  220-token window of a long section.
- **Embedding.** A vector of numbers representing meaning; similar texts have nearby vectors.
- **Cosine similarity.** How aligned two vectors are; 1.0 means identical direction.
- **BM25.** A classic keyword-ranking formula (the one behind Lucene and Elasticsearch defaults).
- **RRF (Reciprocal Rank Fusion).** Merge several ranked lists using only ranks: `1/(k + rank)`.
- **Document anchoring.** Pulling a whole document into context because the question named it.
- **Grounding.** Every claim in the answer traces to retrieved source text.
- **Hallucination.** The model states something not in the sources. Includes *misattribution*:
  real text, wrong subject.
- **Parametric knowledge.** What the model "remembers" from training. Banned as a source here.
- **Function / tool calling.** The model emits a structured request to run a named function; our
  code runs it and returns the result.
- **Strict JSON schema mode.** The provider constrains generation to a schema.
- **Pydantic.** Python library for typed data models with validation.
- **HCC (Hierarchical Condition Category).** Groups of diagnoses used in risk adjustment; only the
  most severe in a hierarchy counts.
- **RAF (Risk Adjustment Factor).** The score that HCCs and demographics produce; drives payment.
- **MEAT.** Monitor, Evaluate, Assess, Treat.
- **RADV.** Risk Adjustment Data Validation audit.
- **PHI / PII.** Protected health information / personally identifiable information.
- **BAA.** Business Associate Agreement.
- **ICD-10-CM.** The diagnosis code set.
- **Fingerprint.** A hash of everything that determines an index; a new fingerprint means a new
  index directory.

---

## Part 6: Likely questions from an AI head, with answers

**"Why not just put all eight documents in the prompt? They fit."**
They do, at 4.7k tokens. But the brief requires demonstrable restriction to relevant chunks; the
design must scale to thousands of documents; retrieval keeps a typical query at 2 to 3k tokens;
and models answer less precisely when the relevant sentence is buried. Retrieval is the scalable
design; the tiny corpus is the demo size.

**"How do you know the model is grounded and not answering from memory?"**
Mechanical checks, not trust: the gate refuses before the model is called when nothing relevant
exists; cited chunk ids must be ones we retrieved; evidence quotes must be verbatim substrings; and
the live runs include a canary the model knows from training that the corpus does not contain. v1
leaked it; v2 and v3 refuse it, one call each.

**"What did the verbatim check miss?"**
Misattribution. In one run every quote was real but the model presented a guideline's example
codes as the patient's documented codes. The fix was retrieval (anchoring the patient's note) plus
a prompt rule to cite both the rule and the fact. A claim-level attribution check is the next
layer. This is also why a coder confirms every suggestion.

**"Why a small model?"**
Because the safety properties should come from the system, not from the model's judgement. If the
guardrails hold on gpt-4o-mini they hold on anything larger, and the model is one environment
variable.

**"Why local embeddings in a healthcare setting?"**
The corpus never leaves the machine. Only the retrieved chunks reach the external model, under a
BAA in production.

**"What happens when the model returns garbage?"**
Strict mode makes malformed JSON rare. When it happens, or when the JSON cites a chunk that does
not exist, or quotes text that is not in the chunk, the validation error goes back to the model
for up to two repairs, with the valid ids and, when relevant, the note that the text came from a
tool result. After that the caller gets a deterministic, schema-valid error object flagged for
review. Fault injection showed six of six recoveries in one repair.

**"Why hybrid retrieval, and why not gate on the fused score?"**
Embeddings cannot tell E11.22 from E11.21; BM25 can. BM25 cannot tell "sugar control" from
"glycemic control"; embeddings can. RRF merges them without score normalisation, but RRF is a
rank, so it cannot say whether the best result is any good. The gate uses absolute cosine and BM25
scores calibrated on a probe set.

**"What is document anchoring and why did you need it?"**
A question that names a patient and asks about HCC mapping ranks the guideline above the patient's
Assessment section, because the name appears only in the Patient Information chunk. No prompt can
fix that. Anchoring recognises identifiers (names, member ids, policy numbers) and pulls that
document in as a block, capped at 900 tokens. Only identifiers anchor; topical words never do.

**"Why 220-token chunks?"**
The embedding model truncates at 256 word pieces. Sections are the natural unit of a clinical
note; 51 of 52 fit in one chunk, the one that does not is windowed with overlap.

**"How does this scale to thousands of documents a day and thousands of users?"**
The query path is stateless and horizontally scalable. Ingestion is a separate job that writes a
fingerprint-named index; workers load it rather than embedding. The store sits behind a protocol
so FAISS or pgvector replaces numpy past a few million chunks. Provider fan-out is bounded by
retries now and by a semaphore and per-tenant limits next. Every response carries tokens, cost and
latency so the system can be budgeted.

**"Where does a human sit?"**
Before any code reaches a chart or a submission (the corpus policy itself says so), on every
low-confidence or refused answer, on every policy interpretation, and on every eval regression.

**"Why no LangChain?"**
Because the seams where RAG fails are exactly what the assignment wants to see handled, and this
project found three real failures by reading the raw messages.

**"How is the confidence number computed, and can a reviewer trust it?"**
A weighted sum of verifiable signals: retrieval strength and the fraction of citations that
verify carry 70 percent; tool failures and repairs penalise; the model's self-report is capped at
10 percent. Partial answers never reach the high band; refusals score zero. The factors are
returned in the trace. It is a heuristic until the eval shows that high-band answers are right
more often than medium ones; that check is Step 6.

**"What happens when the model will not stop miscite a tool result?"**
It happened live. Two repairs named the cause; the model repeated it. The salvage path dropped
the one unverifiable citation, kept four verified ones, downgraded the status to partial and set
the review flag. The reviewer gets a correct, flagged answer instead of an error.

**"What did the eval change?"**
Four things, each with a recorded run. A prompt revision (v4). A structural change to the output
schema (a `caveats` field) after two identical runs showed the model refusing over side issues it
had nowhere to put. A deterministic tool policy after an answer at confidence 1.00 skipped a
required tool. And a targeted repair for refusals caused by an unrelated tool result. The
pass rate went from 9 of 10 to 9 or 10 of 10, and, more importantly, every remaining failure is
flagged for review.

**"Your eval is only ten questions."**
Yes. It is a smoke test with mechanical checks that a reviewer can verify by eye, run more than
once because the model is not deterministic, and gated in CI at 80 percent. The next step is
gold labels from coders, refusal cases drawn from real questions, and a faithfulness judge
calibrated against those labels.

**"Why does a refusal come back as HTTP 200?"**
Because a refusal is a correct result of the system, carrying a status, the missing information
and a review flag. HTTP status codes describe the transport and the request, not the model's
verdict. Clients switch on `answer_status`; only unusable input is a 400 and only a malformed
body a 422. A provider outage is also a 200 with status `error`, never a 500, because the
envelope is the contract.

**"Why are the API handlers synchronous?"**
FastAPI runs plain `def` handlers in a threadpool, so a five-second model call never blocks the
event loop, and the service's semaphore bounds how many of those threads reach the provider.
Making them `async` would require an async model client and an async tool registry for no
benefit at this scale.

**"How do thousands of users not stampede the provider?"**
A per-process semaphore bounds concurrent model calls, an LRU answer cache keyed by normalised
question, scope and index fingerprint serves repeats for free, the gate refuses irrelevant
questions in milliseconds, and identical tool calls inside a request run once.

**"What would you do next with more time?"**
Claim-level citations with an NLI faithfulness check; a cross-encoder re-ranker to widen the gate
margin; tool-call memoisation; metadata filters as the primary scoping mechanism; a larger eval
set with adversarial documents containing injected instructions.

---

## Part 7: How to read this codebase

### 7.1 Method

Read the code in the order data flows: document on disk, then chunk, then vector, then search
result, then prompt, then model call, then validated answer. Each file below has a matching test
file; read them side by side, because every test name is a one-line specification of a behaviour
("test_bm25_keeps_icd_codes_intact"). Three habits make the code stick:

1. **Run the tests for the file you are reading, verbosely**, so the behaviour names scroll past:
   `.venv\Scripts\python -m pytest tests/test_chunker.py -v`.
2. **Poke the objects in a Python session.** The snippet below builds everything with the fast
   hashing embedder so it runs in under a second:

   ```python
   from raapid_rag.config import Settings
   from raapid_rag.ingestion.embedder import HashingEmbedder
   from raapid_rag.ingestion.pipeline import build_index
   from raapid_rag.retrieval.retriever import Retriever
   s = Settings(_env_file=None)
   store, report = build_index(s, HashingEmbedder())
   print(report)                                  # 8 docs, 62 chunks, fingerprint
   c = store.chunks[6]; print(c.chunk_id, c.section); print(c.embed_text)
   r = Retriever(store, s).retrieve("Which HCC category do Jane Doe-Test's codes map to?")
   print(r.anchored_docs, r.chunk_ids, r.passes_gate, r.gate_reason)
   ```
3. **Read one recorded live run** (`docs/runs/step4_live_run_v3_anchored.json`). It holds, per
   question, the retrieved ids, every tool call with arguments and results, the raw model output
   strings, the validation errors, token counts and latency. It is the closest thing to watching
   the system think.

Total reading time for someone comfortable with Python: about two hours for the source (roughly
2,400 lines including docstrings), one more hour for the tests.

### 7.2 Reading order, file by file

Each entry: what the file is for, then every public and private function with what it takes,
what it returns, and the one thing to remember about it.

---

**1. `README.md`** (5 minutes). Setup and the commands that work. Run `python -m pytest` once so
you know the baseline is green (132 tests).

**2. `data/corpus/note_001_endocrinology_progress_note.md`** (5 minutes). The input format: a
front-matter block between `---` fences, a blockquote banner, then `## ` sections. Everything the
loader and chunker do is about this shape.

**3. `src/raapid_rag/config.py`**. One class, `Settings`, and one function.

- `Settings` (pydantic-settings `BaseSettings`). Every knob, grouped: LLM, chunking, retrieval,
  agent loop, paths, observability. Field names map to environment variables case-insensitively
  (`llm_model` is `LLM_MODEL`). Bounds are validation: `chunk_target_tokens` has `le=256` because
  the embedding model truncates there. `extra="ignore"` so unrelated variables in `.env` do not
  break startup. `prompt_version` is a `Literal["v1","v2","v3"]`: adding a v4 prompt means
  extending that literal.
- `Settings.llm_available` (property): true when a key is configured; the service will use it to
  fall back to retrieval-only mode.
- `get_settings()`: cached singleton (`lru_cache`). Tests never call it; they build
  `Settings(_env_file=None, **overrides)` directly so they do not read `.env`.

**4. `src/raapid_rag/ingestion/models.py`**. Two frozen dataclasses.

- `Document(doc_id, doc_type, title, metadata, body, source_path)`: one corpus file after parsing.
- `Chunk(chunk_id, doc_id, doc_type, doc_title, section, text, index, metadata)`: one citable
  unit. `chunk_id` is `<doc_id>#c<index:02d>`. `text` is always a verbatim substring of the body.
- `Chunk.embed_text` (property): `"<doc_title> | <section>\n<text>"`, the string that is embedded
  and BM25-indexed, so a chunk carries its provenance into the vector.
- `Chunk.date` (property): `date_of_service` or `effective_date` from metadata, else empty.

**5. `src/raapid_rag/ingestion/loader.py`**. Disk to `Document`. Fails loudly, always naming the
file.

- `parse_front_matter(raw) -> (meta, body)`: requires the first line to be `---`; reads
  `key: value` lines until the closing `---`; skips blank lines; a line without a colon raises
  `ValueError`; a missing closing fence raises `ValueError`. Returns the remaining lines joined as
  the body.
- `load_document(path) -> Document`: reads bytes and decodes as strict UTF-8 (`CorpusError` on
  failure, with the byte offset); calls `parse_front_matter` (wrapping `ValueError` into
  `CorpusError` with the file name); checks `REQUIRED_KEYS` (`doc_id`, `doc_type`, `title`);
  builds the `Document` with the body stripped of surrounding newlines.
- `load_corpus(corpus_dir) -> list[Document]`: checks the directory exists; iterates `*.md` in
  sorted order (determinism); raises `CorpusError` on a duplicate `doc_id` naming both files;
  skips documents whose body is empty with a `WARNING` log; logs the count.
- `CorpusError(ValueError)`: the only exception type this module raises.

**6. `src/raapid_rag/ingestion/chunker.py`**. `Document` to `list[Chunk]`. Works on character
offsets so chunk text is never re-joined.

- `chunk_document(doc, *, target_tokens, overlap_tokens, count_tokens) -> list[Chunk]`: the entry
  point. For each `(title, text)` from `_split_sections`, strips the text, skips it if empty, emits
  one chunk if `count_tokens(text) <= target_tokens`, otherwise calls `_window` and emits one chunk
  per span labelled `"<title> (part i/n)"`. Keeps a running `index` for ids.
- `_split_sections(doc) -> list[(title, text)]`: finds `## ` headers with `HEADER_RE`; with no
  headers returns one section named after the document title; text before the first header
  becomes `("Preamble", text)` if non-blank; each section's text runs from the end of its header
  line to the start of the next header.
- `_window(text, target, overlap, count) -> list[(start, end)]`: packs units greedily into
  windows of at most `target` tokens; if a packed window over-counts when measured as a whole
  (word pieces are not additive), sheds trailing units until it fits; the next window starts by
  stepping back over trailing units totalling at most `overlap` tokens, but always advances at
  least one unit so the loop terminates.
- `_units(text, target, count)`: lines, split into sentences, split into word groups when a
  sentence alone exceeds the target. Returns spans.
- `_line_spans(text)`: spans of non-blank lines.
- `_sentence_spans(text, start, end)`: splits a line on `SENTENCE_BOUNDARY_RE` (terminal
  punctuation, whitespace, then an upper-case letter, digit, quote or bracket).
- `_word_spans(text, start, end, target, count)`: greedy word grouping within `target`.
- `_make_chunk(doc, index, section, text) -> Chunk`: builds the id and copies metadata.

**7. `src/raapid_rag/ingestion/embedder.py`**. Text to unit vectors, two implementations.

- `Embedder` (Protocol): `name`, `dim`, `embed(texts) -> (n, dim) float32`, `count_tokens(text)`.
- `HashingEmbedder(dim=64)`: `embed` lower-cases, finds `[a-z0-9]+` tokens, adds 1 to bucket
  `crc32(token) % dim`, L2-normalises rows, leaves all-zero rows as zeros (no NaN).
  `count_tokens` is whitespace word count. `name` is `hashing-<dim>`. Used by 100+ tests.
- `SentenceTransformerEmbedder(model_name, device="cpu", batch_size=64)`: `_load()` imports and
  constructs the model once; `dim` and `max_tokens` read from the model; `embed` calls
  `model.encode(normalize_embeddings=True)` in batches; `count_tokens` runs the model's own
  tokenizer with `add_special_tokens=True, verbose=False` (the warning it silences is about the
  exact over-length the chunker is measuring).

**8. `src/raapid_rag/ingestion/store.py`**. The in-memory hybrid index.

- `tokenize_for_bm25(text)`: lower-case, `BM25_TOKEN_RE` keeps `e11.22`, `m-test-1002`,
  `ra-110`, `2025-03-12`, `note_001` whole.
- `ScoredChunk(chunk, dense_score, bm25_score, fused_score, dense_rank, bm25_rank)`: one search
  hit; `bm25_rank` is `None` when the chunk shares no keyword with the query.
- `SearchIndex` (Protocol): `search(query, k, doc_ids=None)`, `chunks`, `__len__`. The retriever
  depends on this, not on the class below.
- `InMemoryHybridStore(embedder, rrf_k=60)`:
  - `add(chunks)`: appends; `ValueError` on a duplicate `chunk_id`; marks BM25 stale.
  - `build()`: embeds only chunks beyond the current matrix length (incremental); rebuilds BM25
    over all chunks if stale.
  - `_ensure_built()`: calls `build` if anything is stale; every scoring method calls it, so
    `add` followed by `search` works without an explicit `build`.
  - `dense_scores(query) -> ndarray`: embeds the query, returns `matrix @ q` (cosine, both unit).
  - `bm25_scores(query) -> ndarray`: zeros if no tokens or no chunks, else `BM25Okapi.get_scores`.
  - `search(query, k, doc_ids=None) -> list[ScoredChunk]`: dense ranks by stable argsort; BM25
    ranks only over chunks with a positive score; `fused = 1/(k+dense_rank) [+ 1/(k+bm25_rank)]`;
    with `doc_ids` given, excluded chunks get `-inf`; returns the top `k` with finite scores.
    `doc_ids=[]` returns `[]` (an empty scope is a scope, not "no filter").
  - `save(path)`: `vectors.npy`, `meta.json` (embedder name, dim, rrf_k, count), `chunks.json`.
  - `load(path, embedder)` (classmethod): refuses if `meta.embedder != embedder.name`; rebuilds
    BM25 (not persisted; cheap); checks vector count equals chunk count.
  - `chunks` (property, copy), `__len__`.
- `_chunk_to_dict` / `_chunk_from_dict`: JSON (de)serialisation of `Chunk`.

**9. `src/raapid_rag/ingestion/pipeline.py`**. The ingestion job.

- `IndexBuildReport`: counts, fingerprint, cache flag, seconds, chunks per doc, max chunk tokens,
  embedder name; `to_dict()`.
- `corpus_fingerprint(docs, settings, embedder_name) -> str`: SHA-256 (first 16 hex) over the
  schema version, embedder name, chunk target and overlap, `rrf_k`, and for every document (sorted
  by id) its id, body and sorted metadata. Anything that changes the index changes this.
- `build_index(settings, embedder, corpus_dir=None, cache_dir=None) -> (store, report)`: loads
  the corpus; computes the fingerprint; if `cache_dir/<fingerprint>/meta.json` exists, tries
  `InMemoryHybridStore.load` and returns (a corrupt or mismatched cache logs a warning and falls
  through); otherwise chunks every document with `embedder.count_tokens`, builds the store, saves
  it when a cache dir is given.
- `_report(...)`: computes per-document counts and the maximum chunk token count.

**10. `src/raapid_rag/retrieval/retriever.py`**. Query to bounded, gated chunk set.

- `RetrievalResult`: `query`, `chunks` (list of `ScoredChunk`), `dropped_chunk_ids`,
  `best_dense_score`, `best_bm25_score`, `passes_gate`, `gate_reason`, `context_tokens`,
  `latency_ms`, `anchored_docs`, `anchor_scores`, `scope_doc_ids`; `chunk_ids` property;
  `to_dict()` flattens each chunk with its text, metadata, scores, ranks and an `anchored` flag.
- `evaluate_gate(*, best_dense, best_bm25, n_chunks, settings) -> (bool, reason)`: pure. Empty
  pool is `empty_retrieval`; pass if either signal meets its threshold; otherwise
  `low_relevance(...)` with the numbers.
- `DocumentAnchors(chunks)`: groups chunks by document and computes `_keys_for` each.
  - `_keys_for(doc_id, doc_chunks) -> set[str]`: the doc id if `_is_identifier` (so `note_001`
    yes, a plain word no); `patient_ref` and `member_id` metadata values; every identifier-like
    token found in any metadata value (`ra-110`, `cg-01`, `2025-03-14`); from the first two
    chunks, the value of any `Name: ... .` line as a full lower-case phrase plus the surname when
    it is at least 4 characters.
  - `match(query, max_docs) -> (doc_ids, scores)`: a key hits if it is a query token or, for
    multi-word keys, a substring of the tokenised query; documents ranked by hit count then
    corpus order; truncated to `max_docs`.
- `_is_identifier(token)`: has a digit and (a letter or `-`/`_`).
- `Retriever(index, settings, count_tokens=None)`: `count_tokens` defaults to tiktoken.
  - `retrieve(query, top_k=None, doc_ids=None) -> RetrievalResult`: pool of `4*k` from the index
    (scoped if `doc_ids`); best dense and BM25 over the pool; if anchoring is enabled and no
    explicit scope was given, match anchors and `_expand_document` each; ranked fill from the pool
    excluding anchored documents via `_apply_document_cap`; `anchored + ranked` through
    `_apply_budget`; gate on the pool size; `context_budget_too_small` if the gate passed but
    nothing fit.
  - `_document_anchors()`: builds `DocumentAnchors` lazily and rebuilds if the index size changed.
  - `_expand_document(query, doc_id)`: scoped search over all of the document's chunks; if their
    total tokens fit `retrieval_anchor_doc_token_budget`, returns them in document order,
    otherwise the top `retrieval_max_chunks_per_doc` by fused score.
  - `_apply_document_cap(pool, k)`: walks the pool in order, skipping chunks whose document has
    already contributed `retrieval_max_chunks_per_doc`, stops at `k`.
  - `_apply_budget(scored)`: walks in order; a chunk that would exceed the budget is dropped and
    the walk continues (a smaller later chunk may still fit); returns kept, dropped ids, tokens.
- `_tiktoken_counter()`: `cl100k_base` counter, the model-side tokenizer.

**11. `src/raapid_rag/cli.py`**. Three commands.

- `make_embedder(kind, settings)`: `"hashing"` or `"st"`.
- `ingest(...)`: `build_index` with the cache dir (default `data/.index`) unless `--no-cache`;
  prints the report as a table or `--json`.
- `search(query, ...)`: `build_index` then `Retriever.retrieve`; table of chunks with scores, a
  gate line, and the first three chunk snippets; or the full `to_dict()` as `--json`.
- `ask(query, --doc-id ..., --no-llm, --json)`: `QueryService.from_settings` then `service.ask`;
  `InvalidQueryError` becomes a usage error. Read this after `service.py` (entry 18c).
- `evaluate(...)` registered as `eval`: builds the service (model only in `--mode full`), loads
  the eval set, runs it, writes both reports, prints a table, exits 1 below `--min-pass-rate`.
- `_print_answer(answer)`: the human rendering: status line, answer, citations table with the
  `verified` column, missing information, tool calls (cached and error marks), warnings, errors,
  one-line trace.

**12. `src/raapid_rag/schema.py`**. The output contract.

- `Citation(chunk_id, evidence)`: `extra="forbid"` so the schema is closed.
- `ModelOutput(answer, answer_status, citations, missing_information, caveats,
  self_reported_confidence, reasoning_summary)`: what the model must emit; `answer_status` is a
  `Literal` of three values. No defaults, no `Optional`, on purpose. `caveats` (added in Step 6)
  is the outlet for concerns that do not change the answer; without it the model refused over
  side issues.
- `AnswerStatus` (enum, five values) and `ConfidenceBand` (enum) for the service side.
- `CitationOut(Citation + doc_id, doc_title, section, date, verified)`.
- `ToolCallRecord(name, arguments, result, error, latency_ms, timed_out)`.
- `GroundedAnswer(...)`: the service's return type, with `confidence` bounded `ge=0, le=1`.
  - `error_envelope(query, *, errors, trace, warnings, tool_calls)`: status `error`, confidence 0,
    review flag true.
  - `insufficient_context_envelope(query, *, reason, missing, trace, warnings)`: status
    `insufficient_context`, `trace["gate_reason"] = reason`.
- `strict_json_schema(model) -> dict`: `model_json_schema()` then `_make_strict` on the root and
  every `$defs` entry.
- `_make_strict(node)`: for every object node sets `additionalProperties: false` and
  `required = all properties`; removes any `default` key; recurses.
- `normalize_for_match(text)`: curly quotes and dashes to ASCII, whitespace collapsed, stripped,
  case-folded.
- `validate_grounding(output, chunks) -> list[str]`: per citation: empty evidence; unknown id
  (message lists valid ids); non-verbatim evidence (message says to copy the sentence exactly).
  Then status rules: `answered`/`partial` need a citation and a non-empty answer;
  `insufficient_context` needs `missing_information`; confidence in [0, 1].

**13. `src/raapid_rag/tools/icd10.py`, `eligibility.py`, `calculator.py`**. Pure functions,
errors returned as `{"error": ...}`.

- `icd10_lookup(query)`: blank query is an error; exact upper-cased code hit returns `found`,
  `match_type="code"`, the entry fields and a one-element `matches`; otherwise a case-folded
  substring search over descriptions returns up to 8 `matches` with `match_type="term"`; every
  result carries the `DISCLAIMER`. `ICD10_TABLE` maps code to (description, hcc_category,
  hierarchy, severity_rank); `_entry(code)` shapes one row.
- `check_eligibility(member_id, date_of_service)`: stamps `checked_at`; parses the ISO date (error
  if not); upper-cases and looks up the member (`NOT_FOUND` with null plan if absent); walks the
  coverage windows (`ACTIVE` on a hit); otherwise `INACTIVE` with the last window and, for
  `termination_type == "retroactive"`, a note with the effective and processed dates.
- `calculate(expression)`: length and emptiness checks; `ast.parse(mode="eval")`; `_eval` walks
  the tree allowing `Expression`, numeric `Constant`, `BinOp` with seven operators (exponent
  magnitude capped at 1000), `UnaryOp`, and `Call` to `abs/round/min/max` by bare name with no
  keywords; everything else raises `_Unsafe`. Maps `SyntaxError`, `_Unsafe`, `ZeroDivisionError`
  and numeric errors to error dicts.
- `days_between(start_date, end_date)`: ISO dates in, signed day count out.

**14. `src/raapid_rag/tools/registry.py`**.

- `ToolSpec(name, description, parameters, fn)`: `parameters` is a strict-style JSON schema.
- `ToolRegistry(max_workers=8)`:
  - `register(spec)`: `ValueError` on a duplicate name.
  - `names`, `openai_tools()`: the `{"type": "function", "function": {..., "strict": True}}` list.
  - `dispatch(name, arguments_json, *, timeout_seconds) -> ToolCallRecord`: unknown tool; JSON
    parse (must be an object); `_validate_arguments`; `executor.submit(fn, **arguments)` and
    `future.result(timeout=...)`; `TimeoutError` becomes `timed_out=True` with an "abandoned"
    message; any exception becomes `"tool 'x' raised Type: message"`; success records result and
    latency.
- `_validate_arguments(spec, arguments)`: missing required keys, then unexpected keys.
- `_ms(t0)`, `_params(**props)` (builds a closed, all-required schema), `default_registry()`
  (registers the four tools with model-facing descriptions that say *when* to call each).

**15. `src/raapid_rag/llm/client.py`**.

- `LLMError`, `LLMUnavailable(LLMError)`, `LLMRequestError(LLMError)`; `RETRYABLE` tuple.
- `ToolCall(id, name, arguments)`, `ChatMessage(content, tool_calls)`,
  `ChatResult(message, finish_reason, usage, latency_ms, model)`.
- `UsageTotals`: `add(usage)` increments tokens and `calls`; `retries` is bumped by the client;
  `total_tokens`, `to_dict()`.
- `PRICES_PER_MILLION`, `estimate_cost_usd(model, *, prompt_tokens, completion_tokens)`: longest
  matching prefix in the table, `None` if unknown.
- `LLMClient(settings, client=None, backoff_base_seconds=0.5)`: without an injected client it
  requires a key and builds `openai.OpenAI(timeout=..., max_retries=0)`.
  - `model` property.
  - `chat(messages, *, tools=None, response_format=None, tool_choice=None) -> ChatResult`: builds
    kwargs (`model`, `messages`, `temperature`, `max_completion_tokens`, `tools` and
    `tool_choice` only when tools are given, `response_format` when given); a tenacity
    `Retrying` over `RETRYABLE` with `wait_random_exponential` and `stop_after_attempt(retries+1)`;
    counts retries; maps exhausted retries to `LLMUnavailable`, other `APIStatusError` to
    `LLMRequestError`, any other SDK error to `LLMError`.
  - `_normalise(response, latency_ms)`: reads `choices[0].message`, builds `ToolCall`s, reads
    usage, updates totals, returns `ChatResult`.

**16. `src/raapid_rag/llm/prompting.py`**.

- `load_system_prompt(version, prompts_dir)`: reads `system_<version>.md`, `FileNotFoundError`
  otherwise.
- `format_chunk(chunk)`: `"[id] type | title | section: s | date: d\n<text>"`.
- `build_context_block(chunks)`: chunks joined inside `<context>...</context>`.
- `build_user_message(query, chunks)`: context block, `<question>`, one-line instruction.
- `build_repair_message(errors, valid_chunk_ids)`: bulleted errors, the valid ids, how to fix.
- `build_json_error_message(detail)`: for parse or schema failures, lists the required fields.

**17. `src/raapid_rag/llm/prompts/`**. Read `system_v3.md` (current), then `PROMPT_CHANGELOG.md`,
then `v1` and `v2` to see what changed and why.

**18. `src/raapid_rag/agent.py`**. The loop.

- `ChatClient` (Protocol): the one method the agent needs from a client.
- `AgentResult`: `model_output` (or `None`), `tool_records`, `repair_attempts`,
  `validation_errors`, `raw_outputs`, `fault_injected`, `llm_calls`, `forced_final`, `usage`,
  `latency_ms`, `messages` (the full transcript).
- `GroundedAgent(llm, tools, settings, system_prompt)`: precomputes the strict
  `response_format` from `ModelOutput`.
  - `run(query, chunks) -> AgentResult`: `ValueError` with no chunks. Builds system and user
    messages. Loop: `force_final` once tool rounds reach the maximum; `llm.chat(messages, tools,
    response_format, tool_choice="none" if forced)`; count the call and usage; if the reply has
    tool calls and we are not forcing, `_run_tools` and continue; otherwise take the content,
    apply fault injection once if configured, record the raw output, `_parse_and_validate`; on
    success store the output and stop; otherwise annotate tool-result evidence, stop if the repair
    budget is spent, else append the assistant content and a repair (or JSON error) message and
    loop. Records latency at the end.
  - `_run_tools(reply, messages, result)`: appends the assistant message with `tool_calls` in
    OpenAI format; dispatches each call through the registry with the configured timeout; appends
    a `tool` message per call containing the result or `{"error": ...}`.
  - `_parse_and_validate(content, chunks) -> (output|None, errors, parse_failure)`: `json.loads`
    (parse failure with position); `ModelOutput.model_validate` (field-by-field messages);
    `validate_grounding`.
- `_annotate_tool_result_evidence(errors, output, records)`: for every citation whose evidence
  failed the verbatim check, if the normalised evidence appears in any tool result JSON, add an
  error naming the tool and saying tool results must not be cited.
- `_add_usage(total, usage)`, `_inject_fault(content, mode, valid_ids)` (`malformed_json`
  truncates to half; `bad_citation` replaces the first citation with `note_999#c99`).

**18b. `src/raapid_rag/guardrails.py`**. Pure functions around the model.

- `InvalidQueryError(ValueError)`: the only exception the service lets escape (HTTP 400).
- `validate_query(query, *, max_chars) -> str`: strips; rejects empty and over-long.
- `PHI_PATTERNS`: four `(label, regex)` pairs for SSN, phone, MM/DD/YYYY and email. Written so
  ISO dates, decimals, codes and member ids do not match.
- `ScrubResult(warnings, blocked, matched_labels)`; `scrub_query(query, *, policy)`: one warning
  per matched label naming the policy; `blocked` only under `policy == "block"`.
- `enrich_citations(citations, chunks) -> list[CitationOut]`: adds doc metadata and computes
  `verified` with `normalize_for_match`; unknown ids become `doc_id="unknown"`, unverified.
- `blank_answer_on_refusal(answer, status) -> (answer, warnings)`: empties the answer on
  `insufficient_context` when it had text; otherwise unchanged.
- `salvage_citations(output, chunks) -> (ModelOutput | None, warnings)`: keeps verified
  citations, downgrades `answered` to `partial`, warns; `None` when nothing survives.
- `required_tools_missing(query, tool_records, policy) -> list[str]`: tools whose policy regex
  matches the question but which were not called successfully (Step 6).
- `tool_induced_refusal(query, output, tool_records, policy, topics) -> str | None`: a repair
  instruction when a refusal's stated reason is about a tool's topic, the tool ran, and the
  question did not ask about that topic (Step 6). Called by the agent.
- `ConfidenceReport(confidence, band, needs_human_review, factors)`; `WEIGHTS`;
  `compute_confidence(*, status, retrieval, citations, tool_records, repair_attempts,
  self_reported, settings, missing_required_tools=None)`: decision D9, including the `partial`
  and missing-tool caps and the "clean" rule for the review flag.

**18c. `src/raapid_rag/service.py`**. The wired request path.

- `QueryService(*, settings, retriever, tools, llm, system_prompt, index_fingerprint)`: builds
  the agent when a model client is given; creates the semaphore and the LRU cache.
- `from_settings(settings, *, embedder=None, cache_dir=None, use_llm=True)` (classmethod):
  index via `build_index` (cache when possible), `Retriever`, `default_registry`, `LLMClient`
  only if `use_llm` and a key exists, prompt by `prompt_version`.
- `health() -> dict`: chunk count, fingerprint, model availability and name, prompt version,
  embedding model, cache entries. Use as the readiness probe.
- `ask(query, doc_ids=None) -> GroundedAnswer`: the sequence described in the Step 5 log. Exits:
  PHI block (error envelope), cache hit (copy with `cache_hit`), gate refusal
  (`insufficient_context_envelope`), retrieval-only (no model), provider failure (error
  envelope), unsalvageable output (error envelope with validation errors and tool records),
  and the normal envelope.
- `_retrieval_only(...)`: status `retrieval_only`, first 200 characters of each retrieved chunk
  as verified evidence, confidence 0, review flag.
- `_finish(key, answer, t0)`: stamps `total_latency_ms`; caches unless the status is `error`.
- `_cache_key(q, doc_ids)`: normalised question, sorted scope, fingerprint. `_cache_get`,
  `_cache_put`: LRU with `move_to_end` and eviction from the front, under a lock.

**18d. `src/raapid_rag/evaluation.py`** and **`eval/eval_set.json`**. The harness.

- `Expected` (pydantic): `statuses`, `cited_docs_any`, `cited_docs_all`, `answer_must_contain`,
  `answer_must_not_contain`, `tools_any`, `max_llm_calls`, `no_citations`.
- `EvalCase`: `id`, `category`, `query`, `doc_ids`, `expected`, `notes`; `expects_refusal`
  property (statuses are exactly `["insufficient_context"]`).
- `load_eval_set(path)`: parses the JSON list, rejects duplicate ids.
- `CheckResult(name, passed, detail)`; `CaseResult` (per-case verdict plus status, confidence,
  band, review flag, calls, tokens, cost, latency, retrieved and cited docs, `retrieval_hit`,
  answer excerpt, warnings, missing information, caveats, tools called); `failed_checks`,
  `to_dict()`.
- `score_case(case, answer)`: full-mode checks in this order: status, no_citations,
  cited_docs_any, cited_docs_all, evidence_verified, each `contains:` and `excludes:` keyword,
  tools_any, max_llm_calls. `passed` is "every check passed".
- `score_retrieval_case(case, answer)`: `gate_refused` when the case expects zero calls,
  `gate_passed` when it expects an answer, `retrieval_hit` when documents are expected.
- `_retrieved_docs`, `_retrieval_hit`, `_contains_any_alternative`, `_ordered_unique`,
  `_metrics`: helpers named for what they do.
- `EvalSummary` and `aggregate(results)`: pass rate, by category, by band (answered/partial
  only), refusal precision and recall, cost, tokens, mean and p95 latency, failed-check counts.
- `run_eval(service, cases, *, mode)`: asks the service each question and scores it.
- `render_markdown(...)`, `write_reports(...)`: the `.md` and `.json` reports, stamped with mode,
  model and prompt version.
- The eval set itself is plain JSON; each case carries a `notes` field saying what it is meant
  to catch. Read it after this module.

**18e. `src/raapid_rag/api.py`**. The HTTP shell.

- `QueryRequest(query, doc_ids, include_trace)`, `SearchRequest(query, doc_ids)`: request
  bodies; pydantic enforces length bounds (a malformed body is a 422).
- `create_app(service=None, *, settings=None) -> FastAPI`: the factory. A `lifespan` hook builds
  the real service when none was injected. Inside: `svc()` accessor; the request-id and
  access-log middleware; the `InvalidQueryError` handler (400); routes `/` (pointers), `/health`
  (readiness), `/documents`, `/search` (retrieval only), `/query` (full path; copies the trace,
  trims it when `include_trace` is false, adds `request_id`).
- `TRACE_KEYS_WITHOUT_RETRIEVAL`: the summary fields kept when the trace is suppressed.
- `app = create_app()`: module-level instance for `uvicorn raapid_rag.api:app`.

**18f. `src/raapid_rag/ui/app.py`**. The Streamlit page (a script, run top to bottom).

- `EXAMPLES`, `STATUS_STYLE`: example questions and the colour per status.
- `load_service(embedder_kind, use_llm, fault)` (cached with `st.cache_resource`): builds the
  service once per process; a fault-injection choice or the model toggle creates a separate
  cached instance.
- `render_answer(answer, show_trace)`: the reviewer-first rendering described in the Step 7 log.
- `main()`: sidebar (toggle, fault selector, trace checkbox, index state, scope multiselect),
  title and caption, example chips (set `st.session_state["question"]` before the input renders),
  the question input (`key="question"`) and Ask button (`key="ask"`), the call to `service.ask`
  inside a spinner, the session history.

**11b. `cli.py` additions.** `serve(--host, --port, --workers)` runs uvicorn on
`raapid_rag.api:app`; `ui(--port)` runs `streamlit run` on the app script in a subprocess.

**20b. `tests/helpers.py`.** `EchoLLM` (the fake model with selectable misbehaviours), `final`,
`build_test_store()` (real corpus, hashing embedder), `make_test_service(store, mode, llm,
**overrides)`. Shared by `test_service.py` and `test_api.py`; `test_ui.py` drives the real app
script headlessly instead.

**18g. `Dockerfile`, `docker-compose.yml`, `.dockerignore`.** One image: CPU torch, requirements,
editable install, the embedding model pre-downloaded and `raapid-rag ingest` run at build time so
the container starts with `HF_HUB_OFFLINE=1`; `HEALTHCHECK` on `/health`; the key injected at run
time. Compose runs the API and the UI from the same image. Not built in this environment.

**19. `eval/retrieval_probe.py`**, **`eval/step4_live_run.py`**, **`eval/step5_live_run.py`**,
**`eval/run_eval.py`** (a one-line wrapper around `raapid-rag eval`) and **`eval/step7_smoke.py`**
(starts uvicorn and Streamlit as subprocesses, waits on their health endpoints, exercises the
API over HTTP, terminates both, writes `docs/runs/step7_api_smoke.json`).
The first builds the real index and prints gate signals for 13 labelled probes. The second runs
the six changelog questions through retriever and agent for each prompt version and writes raw
model outputs to `docs/runs/` (flags `--versions`, `--fault`, `--only`, `--out`). The third runs
seven cases plus a fault-injected one through the full `QueryService` and writes the complete
`GroundedAnswer` objects.

**20. `tests/`**. Read in the same order as the source. Fixtures worth knowing: `FakeOpenAI` in
`test_llm_client.py` (scripted SDK responses and exceptions), `FakeLLM` in `test_agent.py`
(scripted `ChatResult`s that record every request), `CountingEmbedder` in `test_store.py` and
`test_pipeline.py` (counts embed calls to prove incrementality and cache hits), `anchored_store()`
in `test_retriever.py` (three documents built to exercise anchoring). `@pytest.mark.slow` marks
the only test that loads the real model.

---

## Part 8: One request traced end to end, with real numbers

Question: **"Which HCC category do the diabetes codes documented for Jane Doe-Test map to?"**
Numbers are from `docs/runs/step4_live_run_v3_anchored.json` and the retrieval dump made while
building anchoring.

1. `Retriever.retrieve` asks the store for a pool of `4 x 5 = 20` candidates.
   `InMemoryHybridStore.search` embeds the query with MiniLM, computes cosine against all 62 chunk
   vectors, computes BM25 over the same 62, ranks both, fuses with RRF. Best dense in the pool is
   0.655 (`guideline_002#c00`), best BM25 is 16.96 (`guideline_002#c05`). Neither of the top 15 is
   the patient's Assessment chunk.
2. `DocumentAnchors.match` tokenises the query and finds the phrase `jane doe-test` among
   `note_001`'s keys (built from its `Name:` line). `anchored_docs = ["note_001"]`,
   `anchor_scores = {"note_001": 1}`.
3. `_expand_document` runs a scoped search over `note_001`'s 9 chunks, sums their tokens (717 with
   tiktoken), sees that fits the 900-token anchor budget, and returns all 9 in document order.
4. `_apply_document_cap` fills 5 more slots from the pool, skipping `note_001`: three
   `guideline_002` chunks (the cap of 3 stops a fourth) and two `guideline_001` chunks.
5. `_apply_budget` keeps all 14 chunks: 1,066 tokens, under the 1,800 budget. Nothing dropped.
6. `evaluate_gate(0.655, 16.96, 20)` returns `(True, "ok")`.
7. `GroundedAgent.run` builds two messages: the v3 system prompt and a user message whose
   `<context>` block lists 14 labelled chunks followed by the question. Four tool definitions and
   the strict `ModelOutput` schema go with the request.
8. `LLMClient.chat` call 1. The model returns `tool_calls`: `icd10_lookup({"query": "E11.22"})`
   and `icd10_lookup({"query": "E11.42"})`. `_run_tools` appends the assistant message, dispatches
   both through `ToolRegistry.dispatch` (each returns `found: True`, `hcc_category: "Diabetes with
   Chronic Complications"` in about a millisecond), and appends two `tool` messages.
9. `LLMClient.chat` call 2. The model returns JSON: `answer_status: "answered"`, an answer naming
   E11.22 and E11.42 and their shared category "per the ICD-10 lookup", three citations,
   `missing_information: []`, `self_reported_confidence` 0.9.
10. `_parse_and_validate`: `json.loads` succeeds; `ModelOutput.model_validate` succeeds;
    `validate_grounding` finds all three chunk ids among the 14 and all three evidence strings as
    verbatim substrings of their chunks. Errors: none.
11. `AgentResult`: `llm_calls = 2`, `tool_records` = 2, `repair_attempts = 0`,
    `usage = {prompt: 6991, completion: 374}`, `latency_ms` about 4,600, estimated cost $0.0013.

Steps 0 and 12 to 14 are the service layer around the above (`step5_live_run.json`):

0. Before step 1, `QueryService.ask` ran `validate_query` (fine), `scrub_query` (no PHI
   patterns), and looked up the cache (miss; `cache_hit = false`).
12. `blank_answer_on_refusal` left the answer untouched (status `answered`).
    `enrich_citations` produced three `CitationOut`s, all `verified = true`:
    `guideline_002#c03` (the hierarchy rule) and `note_001#c06` twice (E11.22 and E11.42 in
    the Assessment).
13. `compute_confidence`: retrieval strength 1.0 (dense 0.655 is above the 0.65 "strong" mark),
    citations 3/3, tool health 1.0, no repairs, self-report about 1.0. Composite 1.00, band
    high, `needs_human_review = false`. Factors written to `trace.confidence_factors`.
14. The `GroundedAnswer` was stored in the cache. The next identical question (case
    `cache_hit`) returned the same object with `cache_hit = true` and no model call.

Compare the same question **without anchoring** (`step4_live_run_v2_v3.json`): retrieval returned
three `guideline_002` chunks, `note_001#c01` (Patient Information) and one `guideline_001` chunk;
the model called `icd10_lookup` four times, quoted the guideline's example codes, and claimed the
patient had E11.9. Every quote was verbatim, so validation passed. The difference between a wrong
and a right answer here was entirely in step 3.

---

## Part 9: Where to change what

| I want to... | Change this | Then |
|---|---|---|
| Add or edit a corpus document | Drop or edit a `.md` in `data/corpus/` with front matter (`doc_id`, `doc_type`, `title` required) and `## ` sections | `raapid-rag ingest` (the fingerprint changes, a new index directory is built) |
| Change chunk size or overlap | `CHUNK_TARGET_TOKENS` (max 256), `CHUNK_OVERLAP_TOKENS` in `.env`; algorithm in `chunker._window` | re-ingest; the fingerprint includes both values |
| Change how sections are detected | `chunker.HEADER_RE`, `_split_sections` | `pytest tests/test_chunker.py` |
| Change the embedding model | `EMBEDDING_MODEL`; if its token limit differs, the `le=256` bound on `chunk_target_tokens` in `config.py` | the cache refuses the old index by embedder name and rebuilds |
| Swap the vector store for FAISS or pgvector | New class implementing `SearchIndex` (`search(query, k, doc_ids)`, `chunks`, `__len__`) in `ingestion/`; construct it in `pipeline.build_index` | retriever and agent unchanged |
| Change BM25 tokenisation (what counts as one token) | `store.BM25_TOKEN_RE` | also affects anchor matching (`tokenize_for_bm25`) |
| Change RRF behaviour | `store.search` fused-score lines; `RRF_K` in `.env` | |
| Make the gate stricter or looser | `RETRIEVAL_MIN_DENSE_SCORE`, `RETRIEVAL_MIN_BM25_SCORE`; logic in `retriever.evaluate_gate` | `python eval/retrieval_probe.py` to see the effect on 13 probes |
| Change how many chunks reach the model | `RETRIEVAL_TOP_K`, `RETRIEVAL_MAX_CHUNKS_PER_DOC`, `RETRIEVAL_CONTEXT_TOKEN_BUDGET` | |
| Change what anchors a document | `retriever.DocumentAnchors._keys_for` (which metadata keys, the `Name:` regex `NAME_LINE_RE`, surname rule), `_is_identifier` | `pytest tests/test_retriever.py -k anchor` |
| Turn anchoring off, or change its budget | `RETRIEVAL_ENABLE_DOCUMENT_ANCHORING`, `RETRIEVAL_ANCHOR_DOC_TOKEN_BUDGET`, `RETRIEVAL_MAX_ANCHORED_DOCS` | |
| Scope a question to specific documents | `doc_ids` in `POST /query` and `POST /search`, `--doc-id` on the CLI, the scope picker in the UI; all pass through to `Retriever.retrieve` | |
| Change how a chunk looks inside the prompt | `prompting.format_chunk`, `build_context_block` | remember only the body text is quotable; the header is not in `Chunk.text` |
| Change the system prompt | Add `llm/prompts/system_v5.md`; extend `prompt_version` `Literal` in `config.py`; set `PROMPT_VERSION=v5`; add a changelog entry | `raapid-rag eval --mode full` (twice, the model is not deterministic) and keep the reports |
| Change the repair instructions | `prompting.build_repair_message`, `build_json_error_message`; tool-result annotation in `agent._annotate_tool_result_evidence` | |
| Allow more or fewer repairs / tool rounds | `MAX_REPAIR_ATTEMPTS`, `MAX_TOOL_ROUNDS`; loop in `agent.run` | |
| Add a tool | Function in `tools/<name>.py` returning a dict (errors as `{"error": ...}`); `ToolSpec` in `registry.default_registry` with a closed, all-required schema and a description that says *when* to call it; mention it in the prompt's Tools section | tests in `tests/test_tools.py` and `test_registry.py` |
| Change a tool's timeout or thread pool | `TOOL_TIMEOUT_SECONDS`; `ToolRegistry(max_workers=...)` | |
| Edit the mock ICD-10 table or HCC labels | `tools/icd10.ICD10_TABLE` | keep it consistent with `guideline_002` in the corpus |
| Edit mock members or coverage | `tools/eligibility.MEMBERS` | |
| Change what the model must output | `schema.ModelOutput` (no defaults, no `Optional`, or strict mode rejects it) | `pytest tests/test_schema.py` checks strict compatibility |
| Add a grounding rule | `schema.validate_grounding`; write the message as an instruction the model can act on | `tests/test_schema.py`, and the agent will feed it back automatically |
| Change what the service returns | `schema.GroundedAnswer` and its envelope constructors | |
| Change model, temperature, timeouts, retries | `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_MAX_OUTPUT_TOKENS` | |
| Add a price for a new model | `llm/client.PRICES_PER_MILLION` | |
| Change which provider errors are retried | `llm/client.RETRYABLE` | |
| Swap the LLM provider | Reimplement `LLMClient.chat` and `_normalise` to return `ChatResult`; the agent only sees `ChatResult` | |
| Demonstrate the repair path live | `FAULT_INJECTION=bad_citation` or `malformed_json`; code in `agent._inject_fault` | `python eval/step4_live_run.py --fault bad_citation` |
| Change CLI output or add a command | `cli.py` (keep `--json` on every command; tests parse it) | |
| Log chunk text for debugging | `LOG_CHUNK_TEXT` (default false; nothing logs chunk or question text today, the API access log records only method, path, status, latency and request id; the flag is the switch any future debug logging must check) | |
| Change what counts as PHI in a query | `guardrails.PHI_PATTERNS` (add a `(label, regex)` pair) | `pytest tests/test_guardrails.py -k scrub` |
| Block instead of flag PHI | `PHI_POLICY=block` | |
| Change the confidence weights or bands | `guardrails.WEIGHTS`; `CONFIDENCE_HIGH`, `CONFIDENCE_MEDIUM`; "strong" scores `RETRIEVAL_STRONG_DENSE_SCORE`, `RETRIEVAL_STRONG_BM25_SCORE` | `pytest tests/test_guardrails.py -k confidence` |
| Change when a human must review | the `clean` rule at the end of `guardrails.compute_confidence` | |
| Change the salvage rule (what survives exhausted repairs) | `guardrails.salvage_citations` | |
| Change the maximum query length | `MAX_QUERY_CHARS` | |
| Change the answer cache size or key | `ANSWER_CACHE_SIZE`; `service._cache_key` | |
| Change the concurrency bound | `LLM_MAX_CONCURRENCY` | |
| Add a step to the request path | `service.QueryService.ask` (keep every exit a valid `GroundedAnswer`) | `pytest tests/test_service.py` |
| Change what the retrieval-only envelope contains | `service._retrieval_only` | |
| Change memoisation of tool calls | `agent._run_tools`, `_canonical_arguments` | |
| Add or edit an eval case | `eval/eval_set.json` (one object; `expected` keys are documented in `evaluation.Expected`) | `pytest tests/test_eval.py`, then `raapid-rag eval --mode retrieval` |
| Change what counts as a pass | `evaluation.score_case` (add a `CheckResult`) | |
| Change the CI gate | `--min-pass-rate` on `raapid-rag eval` | |
| Change which questions require which tool | `TOOL_POLICY` (JSON in `.env`) or the default in `config.py` | `pytest tests/test_guardrails.py -k tools` |
| Change how a tool-induced refusal is detected | `TOOL_REFUSAL_TOPICS`; `guardrails.tool_induced_refusal` | `pytest tests/test_agent.py -k refusal` |
| Give the model a new structured outlet (like `caveats`) | `schema.ModelOutput` (required field, no default), `GroundedAnswer`, `service.ask` mapping, CLI printing, test fixtures | `pytest tests/test_schema.py` |
| Add an API endpoint | a route inside `api.create_app` (keep handlers sync; return envelopes, not error codes, for model outcomes) | `pytest tests/test_api.py` |
| Change which trace fields a trace-suppressed response keeps | `api.TRACE_KEYS_WITHOUT_RETRIEVAL` | |
| Add authentication, rate limiting or CORS | middleware in `api.create_app` (or the gateway in front of it); derive `doc_ids` from the caller's tenant there | |
| Change the UI layout or what a reviewer sees first | `ui/app.py` `render_answer` (order is deliberate: status and review flag first) | `pytest tests/test_ui.py` |
| Add an example question to the UI | `ui/app.py` `EXAMPLES` | |
| Run the demo | `raapid-rag ui` (one process) or `raapid-rag serve` plus a client; README sections 2 to 7 list every method | `python eval/step7_smoke.py` for a scripted check |
| Rebuild the final transcript | the loop in the Step 8 log's source (see `docs/runs/final_transcript.md` header); each `ask` costs about a tenth of a cent | |
| Ship a container | `Dockerfile`, `docker-compose.yml` (build not verified here) | `docker build -t raapid-rag .` |
| Change ports or worker count | `raapid-rag serve --host --port --workers`, `raapid-rag ui --port` | |

---

## Part 10: Interview preparation beyond the codebase

Everything above is about *this* project. An interviewer for an Applied AI Engineer role will
also probe the concepts underneath it, the healthcare domain it serves, and how it would run for
real. This part collects that material. Where a formula helps, it is written out; where a
definition can be said in one breath, it is kept to one breath.

### 10.1 Concepts you must be able to explain in one breath each

- **Token.** The unit a model reads and bills. Roughly three quarters of an English word; clinical
  text with codes and units tokenises worse (about 1.3 tokens per word here). Two tokenizers matter
  in this project: MiniLM word pieces (chunk sizing) and tiktoken `cl100k_base` (billing).
- **Embedding.** A fixed-length vector representing the meaning of a text, trained so that texts
  used in similar contexts land near each other. MiniLM: 384 dimensions, 256-token input limit.
- **Cosine similarity.** `cos(a, b) = (a . b) / (|a| |b|)`. With unit-normalised vectors it is the
  dot product, which is why the store computes `matrix @ q`. Range -1 to 1; for MiniLM on real text
  almost everything lands between 0 and 0.8, which is why the gate margin is thin.
- **BM25.** Keyword ranking. Per query term: `IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`
  with `IDF(t) = log((N - n + 0.5) / (n + 0.5))`, `k1 = 1.5`, `b = 0.75`. Intuition: rare terms
  count more, repeated terms saturate, long documents are penalised. Corner case seen in the tests:
  with 2 documents, a term present in 1 of them has IDF 0.
- **Reciprocal Rank Fusion.** `score(d) = sum over rankers of 1 / (k + rank(d))`, `k = 60`. Robust
  because it ignores score scales; useless for gating because it ignores absolute relevance.
- **Dense versus sparse retrieval.** Dense (embeddings): paraphrase-friendly, blind to exact
  identifiers and to negation. Sparse (BM25): exact-match-friendly, blind to synonyms. Hybrid takes
  the union of strengths.
- **Chunking.** Small chunks give precise citations and dilute meaning; large chunks give context
  and blur retrieval. Overlap prevents boundary splits at the cost of duplicate tokens.
- **Context window.** Maximum tokens per call (128k for gpt-4o-mini). Exceeding it is a 400 error,
  which is why the retriever enforces a budget before the call. "Lost in the middle": models
  attend less to facts buried mid-context, another reason to keep context small.
- **Temperature.** Sampling randomness. 0 is greedy but *not* deterministic across calls
  (batching, floating point, mixture-of-experts routing); observed here as varying tool-call
  counts between identical runs.
- **Structured output / constrained decoding.** The provider restricts the next-token distribution
  to tokens that keep the output valid against a grammar derived from the JSON schema. Strict mode
  needs a closed schema: every property required, `additionalProperties: false`, no defaults, a
  subset of JSON Schema keywords. It guarantees shape, never truth.
- **Function calling.** The model emits `tool_calls` (name plus JSON arguments); the application
  executes them and appends a `role: tool` message carrying the matching `tool_call_id`; the model
  continues. The model never runs anything; it asks. Strict function schemas constrain the
  arguments the same way strict output does.
- **Agent loop.** Call, act on tool requests, call again, until the model answers or a bound is
  hit. The bounds (`MAX_TOOL_ROUNDS`, `MAX_REPAIR_ATTEMPTS`) are the engineering; without them an
  agent can loop at your expense. This project's loop is ReAct-style with a forced final answer.
- **Prompt caching.** Providers cache a stable prompt prefix (OpenAI: automatic above 1,024
  tokens, roughly half price on cached input). Put static content (system prompt, tool schemas)
  first and variable content (context, question) last. This project already orders it that way.
- **Hallucination taxonomy.** *Intrinsic*: contradicts the sources. *Extrinsic*: adds what the
  sources do not say (the canary). *Misattribution*: real text, wrong subject (the guideline
  example presented as the patient's code). Verbatim checks catch the first two when the claim is
  quoted; only attribution-aware checks catch the third.
- **Grounded versus faithful versus correct.** Grounded: every claim points at a source. Faithful:
  the claim follows from the source. Correct: the claim is true. A system can be grounded and
  unfaithful (misattribution), and faithful but incorrect (the source is wrong).
- **Guardrails.** Deterministic checks around a probabilistic component: pre-call (gate, PHI
  scrub, injection filter), in-call (schema, tools), post-call (verification, confidence, review).
- **RAG versus fine-tuning.** RAG for knowledge that changes, must be cited, and must be auditable.
  Fine-tuning for format, tone, latency, a narrow skill. Fine-tuning does not reliably add facts
  and cannot produce citations. In healthcare, auditability decides it.
- **Re-ranking.** A cross-encoder scores query-chunk pairs jointly and reorders the top
  candidates: better precision, 100 to 300 ms, cannot rescue a chunk the first stage never found.
- **Vector database.** Approximate nearest-neighbour search (HNSW, IVF) plus metadata filtering.
  Needed past a few million vectors; here numpy is exact and fast enough.
- **LLM-as-judge.** A model grading outputs. Scales faithfulness checks; biased toward verbose,
  confident answers; must itself be validated against human labels.

### 10.2 Risk-adjustment domain primer (what RAAPID lives on)

- **Risk adjustment.** Medicare Advantage plans are paid a capitated amount per member, adjusted
  by a **Risk Adjustment Factor (RAF)** reflecting how sick the member is. RAF = demographic
  coefficients + a coefficient per **Hierarchical Condition Category (HCC)** documented in the
  payment year (+ interaction terms), then normalised and reduced by a coding-intensity
  adjustment. Higher RAF means higher payment, which is why an unsupported code is a compliance
  and fraud exposure, not just an error.
- **HCCs and hierarchies.** Diagnosis codes map to categories; within a hierarchy only the most
  severe category pays. CMS moved from the V24 model to **V28** (phased in over payment years 2024
  to 2026; 115 categories; about 2,000 codes dropped; several diabetes categories constrained to
  one coefficient, which is why `guideline_002` says complicated and uncomplicated diabetes carry
  the same weight).
- **Documentation standards.** A condition counts only if documented in a **face-to-face** (or
  qualifying audio-video) encounter by an **acceptable provider type**, supported by **MEAT**
  (Monitor, Evaluate, Assess, Treat; some teams use TAMPER) in that encounter, and **recaptured
  every calendar year**. Problem lists, lab-only reports and "history of" do not count. Outpatient
  uncertain diagnoses are not coded (ICD-10-CM guidelines section IV); inpatient discharge
  diagnoses may be (section II.H). The corpus encodes all of this on purpose.
- **RADV.** Risk Adjustment Data Validation: CMS or the OIG samples members and demands the chart
  supporting each HCC. Unsupported HCCs are clawed back, and since 2023 CMS may **extrapolate**
  findings across the contract. This is the strongest answer to "why does a human confirm every
  suggestion".
- **Retrospective versus prospective.** Retrospective: coders review last year's charts to find
  documented-but-unsubmitted conditions (recapture). Prospective, or **suspecting**: predict
  conditions a member likely has (labs, medications, prior years) so a provider can assess and
  document at the next visit. Suspecting is where AI adds most value and carries most risk: a
  suspect is a *question for a clinician*, never a code.
- **Submission mechanics.** Encounters reach CMS through the Encounter Data System with **sweep
  deadlines** (initial, mid-year, final). The corpus's 90-day rule is a simplified stand-in.
- **Where a grounded QA service plugs in.** Coder assist ("show me the evidence for E11.22 in
  this chart, with the sentence"), policy Q&A for compliance staff, provider query drafting ("the
  note lists CKD and diabetes but does not link them; please clarify"), audit response (retrieve
  the supporting passage for a RADV sample). All are evidence-surfacing tasks with a human at the
  end.
- **Standards to recognise.** ICD-10-CM (diagnoses), CPT/HCPCS (procedures), SNOMED CT (clinical
  terms), LOINC (labs), RxNorm (medications), HL7 v2 and **FHIR** (interchange), C-CDA (clinical
  documents). Real charts arrive as PDFs, scanned faxes and HL7 feeds, not Markdown.
- **HIPAA in two sentences.** PHI is any of 18 identifiers (names, dates other than year, MRNs,
  member ids, phone numbers, and so on) linked to health information. A vendor touching PHI needs
  a Business Associate Agreement, the minimum-necessary principle applies, every access is logged,
  and de-identification is either Safe Harbor (strip the 18) or Expert Determination.

### 10.3 Clinical-text failure modes this project does not yet handle

| Failure | Why it matters | Mitigation |
|---|---|---|
| **Negation.** "Denies chest pain" embeds near "chest pain". | Retrieval surfaces the negated sentence; a model may read it as positive. | Evidence at sentence level including the cue (verbatim already); NegEx-style pre-tagging; NLI faithfulness check; clinical encoders (MedCPT, BioClinicalBERT-based). |
| **Family and social history.** "Mother with rheumatoid arthritis" (note_004). | The condition belongs to a relative. | Section labels already exist on chunks; add a rule the prompt and validator can use ("never code from Family History"). |
| **Copy-forward and templates.** Identical text pasted visit after visit. | Inflates both scores; a "documented" condition may be stale. | Near-duplicate detection at ingestion (MinHash); date-aware retrieval preferring the latest distinct mention. |
| **Abbreviations and misspellings.** HTN, DM2, CKD3a, "metfromin". | Embeddings partly cope; BM25 does not. | Abbreviation expansion at ingestion; query rewriting; fuzzy identifier matching. |
| **Temporal reasoning.** Prior A1c 7.6 in January versus 8.2 on 2025-03-12. | "Most recent" is a date comparison the model can get wrong. | Extract dated values into structured fields at ingestion; `days_between` tool (done); prefer structured lookups. |
| **Numbers and units.** 8.2 versus 8.20 percent; mg versus mcg. | The verbatim check passes; a paraphrased number in the answer may not. | Numeric consistency check: every number in the answer must appear in a cited chunk or tool result. Cheap, high value, a good next feature. |
| **Scanned documents and OCR noise.** | Garbage in, confident garbage out. | OCR confidence thresholds; layout-aware parsing; route low-confidence pages to humans. |
| **Very long documents.** A 300-page stay. | Thousands of chunks; anchoring falls back to the top three. | Hierarchical retrieval (section summaries, then chunks); map-reduce summaries; encounter-date filters. |
| **Prompt injection through documents.** | The model may obey text it should quote. | v3 says context is data (done); ingestion scan for instruction-like text; never give the agent write tools. |
| **Over-refusal.** v2 refused two answerable questions. | A coder refused ten times stops using the tool. | Track refusal precision and recall in the eval; tune prompt and gate against both. |
| **Multi-hop questions.** "Does Jane's CKD stage change her HCC given her diabetes?" | Needs three chunks and a hierarchy rule. | Anchoring (done); explicit multi-step tool use; a rules engine or knowledge graph for hierarchies (10.6). |

### 10.4 Production architecture at RAAPID scale

```
 charts, policies        ingestion workers                index store             query service
 (PDF, HL7, FHIR) -> queue -> parse -> de-identify ->   -> vector DB with     <- stateless pods
                              chunk -> embed -> write      tenant + patient      (autoscaled)
                                                           partitions               |
                                                                                    v
                                                                          LLM gateway (BAA endpoint,
                                                                          rate limits, fallback model,
                                                                          zero retention, prompt cache)
                                                                                    |
             eval service (nightly regression, canaries) <--- traces, metrics ------+
             audit log (immutable: query, ids, tools, answer, reviewer decision)
```

- **Ingestion** is asynchronous and idempotent (the fingerprint idea generalises to per-document
  content hashes). De-identification precedes embedding. Thousands of documents a day is a few
  hundred thousand chunks; one CPU worker embeds MiniLM at several hundred chunks a second, so this
  is minutes. A GPU is a cost decision, not a requirement.
- **Index store.** Past a few million chunks: pgvector (transactions, row-level security, one
  database) or a managed vector database (Qdrant, Weaviate, Pinecone). Partition by tenant; filter
  by patient *before* similarity search so a query can never retrieve another patient's chart.
- **Query service.** `QueryService` behind FastAPI, N replicas, no shared state, health checks,
  request ids. Semantic cache keyed by normalised question, corpus fingerprint and scope for the
  many repeated policy questions. Bounded concurrency to the gateway.
- **LLM gateway.** Enterprise or Azure OpenAI under a BAA with zero retention, private networking,
  per-tenant rate limits, a fallback model, model version pinned and recorded in every response.
  Degrade to retrieval-only when the gateway is down (`retrieval_only` status exists for this).
- **Observability.** OpenTelemetry traces carrying prompt version, index fingerprint, model id.
  Metrics: p50/p95 latency, cost per query, refusal rate, verified-citation rate, repair rate,
  tool error rate, and **human override rate** (how often a coder disagrees), the truest quality
  signal.
- **Evaluation in CI.** The eval set runs on every prompt or threshold change; regressions block
  the merge. Nightly canaries against the live model catch silent provider changes.
- **Governance and security.** Prompt registry with versions and changelogs (this project's
  changelog is the seed), model cards, change control, RBAC per tenant and role, encryption at rest
  and in transit, secrets from a vault, no PHI in logs, SOC 2 controls, audit retention per policy.

### 10.5 Evaluation vocabulary (Step 6 implements a subset)

- **Retrieval:** recall@k (a gold chunk is in the top k), precision@k, MRR (1 / rank of the first
  gold chunk), nDCG. Report at the k you actually send to the model.
- **Grounding:** citation accuracy (cited ids are gold ids), verified-evidence rate (quotes are
  verbatim; measured mechanically here), attribution accuracy (the claim is about the right
  entity; needs human or NLI labels).
- **Answer quality:** exact match for single-value facts; keyword or rubric match otherwise;
  LLM-as-judge with human calibration for free text.
- **Refusal behaviour:** refusal precision (of the refusals, how many were right) and refusal
  recall (of the questions that should be refused, how many were). Both directions are defects.
- **Calibration:** does `confidence` track correctness? Bin by confidence, compare accuracy per
  bin (expected calibration error). A confidence a reviewer can act on must mean something.
- **Operational:** cost per query, latency percentiles, tool error rate, repair rate.
- **Frameworks to name:** RAGAS (faithfulness, answer relevance, context precision/recall),
  TruLens, DeepEval, promptfoo. Why not used: a ten-question eval with mechanical checks is more
  transparent than a framework score; the framework can come later.

### 10.6 Trade-offs you should be ready to argue either way

- **Small model plus guardrails versus large model.** Chosen: small. For large: better instruction
  following (fewer MUST violations, fewer repeated tool calls), better multi-hop reasoning. Counter:
  guardrails must hold regardless; the upgrade is one variable; cost is 10 to 20x.
- **Local versus API embeddings.** Chosen: local. For API: quality on long or technical text, no
  hosting. Counter: PHI egress of the whole corpus; any embedding upgrade means re-embedding
  everything either way, so version the index.
- **Anchoring versus larger top-k.** Chosen: anchoring. For top-k: simpler. Counter: the
  Assessment chunk was not in the top 15; more k adds noise to every question.
- **Strict refusal versus helpfulness.** v2 versus v3. The right operating point depends on the
  user: compliance wants strict; coders want helpful with flags.
- **Deterministic logic in code versus in the model.** Hierarchy resolution, eligibility rules and
  date arithmetic are deterministic: tools or a rules engine, never the prompt. The model extracts
  evidence; code computes consequences.
- **Knowledge graph complement.** Member -> Encounter -> Diagnosis -> Code -> HCC -> Hierarchy as a
  graph makes trumping, recapture gaps and suspect generation deterministic queries. RAG surfaces
  the evidence sentence; the graph computes the RAF impact. GraphRAG also helps policies that
  cross-reference each other (RA-100 cites RA-110).
- **Agent autonomy versus bounded loops.** Planning and self-reflection can help hard questions;
  every extra call is latency, cost and a failure point. In a compliance setting, bounded and
  inspectable wins.
- **Framework versus hand-rolled.** Hand-rolled for the take-home; in a team, a thin internal
  library with the same seams beats both a heavyweight framework and everyone rolling their own.

### 10.7 Whiteboard-style questions with short model answers

- *Cosine similarity of a query against a matrix.* Normalise rows and query; `M @ q`. Top-k with
  `np.argpartition(-scores, k)[:k]` then sort those k: O(N) instead of O(N log N).
- *Reciprocal Rank Fusion.* For each ranker, for each (rank, doc) starting at 1, add
  `1 / (60 + rank)` to `fused[doc]`; sort descending.
- *Exponential backoff with jitter.* Sleep `min(cap, base * 2 ** attempt) * uniform(0.5, 1.5)`;
  retry only transient errors; cap attempts; raise a typed error.
- *Chunk text with overlap.* Unit splitting (lines, sentences, words), greedy packing, step back
  over trailing units for overlap, always advance; state the verbatim-substring invariant.
- *Validate JSON from a model.* Parse, then schema (pydantic), then semantic rules the schema
  cannot express, then feed errors back; always have a fallback object.
- *Detect PHI in a string.* Regex for SSN, phone, dates, MRN patterns; an entity recogniser for
  names; regex has high recall on formats and poor recall on names.
- *How do you know retrieval works?* Gold chunk ids per question, recall@k; a probe set of
  irrelevant questions for the gate.
- *Big-O of your search.* O(N d) per query brute force; ANN roughly O(log N) at the cost of
  approximation and index build time.
- *Context exceeds the window?* 400 from the provider; the token budget prevents it; fallback is
  dropping the lowest-ranked chunks.
- *Idempotent pipeline?* Content hashes per document, deterministic chunk ids, index directories
  named by fingerprint.
- *Test the agent without paying?* A scripted fake returning `ChatResult`s; assert on the exact
  messages sent (this project's `FakeLLM`).
- *Prevent one tenant seeing another's data?* Filter before search (partition or metadata
  filter), enforced in code, never in the prompt; test it.

### 10.8 A five-minute narrative and a demo script

**Narrative, in this order.**
1. The problem: answer questions over clinical and policy text without inventing anything, in a
   domain where an unsupported code is an audit finding.
2. The principle: guarantees come from the system, not from trusting the model; a small model on
   purpose.
3. The architecture in one breath: hybrid retrieval with an absolute gate, document anchoring,
   strict-schema output validated twice, tools for anything deterministic, bounded repair, an
   always-valid envelope.
4. What the live runs taught: header quotes, tool text as evidence, misattributed guideline
   examples; each fixed by a specific line of code, each recorded.
5. Numbers: cost per query, latency, tests, thresholds and their margins.
6. Limitations, honestly (10.10), and what comes next.
7. How it would run at RAAPID (10.4) and where the human sits.

**Demo script, about eight minutes.**

```
python -m pytest                                              # 214 green in about 3 s
raapid-rag search "purple elephant quantum trombone"          # gate REFUSES, no model call
raapid-rag ask "Which HCC category do Jane Doe-Test's diabetes codes map to?"
                                                              # anchored note, 2 ICD-10 lookups, 3 verified citations, high, no review
raapid-rag ask "What metformin dose is recommended for CKD stage 3?"
                                                              # model-side refusal, empty answer, missing_information
FAULT_INJECTION=bad_citation raapid-rag ask "What was Jane Doe-Test's most recent HbA1c?"
                                                              # repair fires, answer recovers at medium with review flag
raapid-rag eval --mode full                                   # 10 cases, calibration table, about a cent
raapid-rag ui                                                 # then ask the telehealth question and open a citation's source chunk
```

Then open `PROMPT_CHANGELOG.md`, `docs/runs/eval_full_latest.md` and `docs/runs/final_transcript.md`.

### 10.9 Numbers to memorise

| Fact | Value |
|---|---|
| Corpus | 8 documents, 2,930 words, 4,693 tokens, 52 sections, 62 chunks |
| Chunking | 220 word-piece target, 40 overlap, embedder limit 256 |
| Embedding model | all-MiniLM-L6-v2, 384 dims, about 80 MB, 12 to 15 s cold load |
| Gate thresholds | dense 0.37, BM25 5.0; relevant probes >= 0.379 / 6.3; irrelevant <= 0.354 / 4.2 |
| Retrieval | top_k 5, pool 20, cap 3 per doc, budget 1,800 tokens, anchor budget 900 |
| Agent bounds | 4 tool rounds, 2 repairs, tool timeout 5 s, LLM timeout 30 s, 3 retries |
| Model list price | gpt-4o-mini $0.15 / $0.60 per million input / output tokens |
| Cost per query | refusal about $0.0004; simple answer about $0.0005; tool-heavy $0.001 to $0.003 |
| Latency | refusal 1.2 to 1.5 s; simple 2 to 4 s; tool-heavy 3.5 to 9 s; worst seen 13 s |
| Prompt versions (6 questions) | v1 21 calls / $0.0071; v2 8 / $0.0032; v3 + anchoring 14 / $0.0085 |
| Fault injection | 6 of 6 recovered in exactly 1 repair (Step 4); through the service: 1 repair, confidence 0.76 medium |
| Confidence weights | 0.35 retrieval, 0.35 citations, 0.10 tools, 0.10 repairs, 0.10 self-report; high 0.80, medium 0.55 |
| Service live run | gate refusal 0 calls / 16 ms; HCC answer 1.00 high, no review; salvaged telehealth 0.79 medium, review |
| Cache and concurrency | LRU 256 entries per process; 8 concurrent model calls per process |
| Eval set | 10 cases: 7 answers (2 with tools), 3 refusals (1 at the gate) |
| Eval results, shipped configuration | 9/10 and 10/10; refusal precision and recall 1.0; high band 22 of 22 across 4 runs; about $0.01 and 40 s per run |
| Eval, retrieval mode | 10/10, 13 ms per case, $0 |
| API over HTTP | ready in 13.8 s (model load); HCC answer 5.7 s; gate refusal 17 ms; blank query 400 in 2 ms |
| Tests | 215 (214 fast, 1 slow); fast suite about 3 s (the headless UI tests dominate) |
| Source size | about 4,000 lines including docstrings |

### 10.10 Limitations to volunteer before they are asked

- Eight documents and thirteen probes: thresholds are calibrated, not validated; they may be
  overfit to this corpus.
- MiniLM compresses cosine into a narrow band; the gate margin is 0.025.
- Anchoring recognises names only from a `Name:` line and identifiers from metadata; real charts
  need an entity recogniser and, more importantly, explicit patient scoping.
- The verbatim check cannot catch misattribution; claim-level attribution is future work.
- The confidence composite is a heuristic with hand-chosen weights. On the eval it is well
  behaved (high band 22 of 22), but ten cases and four runs is a smoke test, not a calibration
  study.
- Identical runs differ at temperature 0; the pass rate is 9 or 10 of 10 depending on the run.
- PHI scrubbing is regex on formats; it will not catch a name. Real data needs an entity
  recogniser and de-identification before ingestion.
- The API has no authentication, rate limiting or CORS; the UI is a single-process demo tool.
  Both belong behind a gateway that adds identity, tenant scoping and limits.
- Tools are mocks; real ones need client timeouts, circuit breakers and audit logging.
- Temperature 0 is not deterministic; a small model repeats tool calls (now memoised) and, in
  one live case, kept miscite a tool sentence through two repairs (now salvaged, flagged).
- No multi-tenancy, no streaming; cache and concurrency bound are per process; each API worker
  loads its own model and index.

### 10.11 Questions worth asking them

- Which documents dominate the workload: chart notes, policies, or both? Scanned or structured?
- Where does the human sit in the current coding workflow, and what does the review UI show?
- What refusal rate is acceptable for a coder-facing tool, and how is disagreement tracked?
- Which LLM providers and deployment models are approved under your BAAs?
- How are prompts and models versioned and rolled back today?
- What does the eval set look like, and who owns the gold labels?
- How is the team handling the V24 to V28 transition in its tooling?
