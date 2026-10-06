# RAAPID Grounded GenAI Micro-Service (take-home)

A retrieval-augmented question-answering service over synthetic clinical notes, payer policies
and coding guidelines. It answers only from retrieved context, cites the exact chunk with a
verbatim quote that the service verifies, calls real tools (mock ICD-10 lookup, eligibility
check, calculator, date difference) through function calling, and returns a schema-validated JSON
envelope with a status, a confidence band, missing information, caveats and a human-review flag.

**Status: complete (steps 1 to 8).** 215 tests. Ten-case eval at 9 to 10 of 10 with every miss
flagged for review. Three surfaces over one stateless service: Streamlit UI, FastAPI, CLI.

| Read this for... | File |
|---|---|
| The two-page submission note (architecture, prompts, grounding, eval, limitations, productionization) | [docs/DESIGN_NOTE.md](docs/DESIGN_NOTE.md) |
| The long-form study guide: every decision, every function, where to change what, interview prep | [docs/ENGINEERING_NOTES.md](docs/ENGINEERING_NOTES.md) |
| Prompt v1 to v4 with the recorded run behind every change | [src/raapid_rag/llm/prompts/PROMPT_CHANGELOG.md](src/raapid_rag/llm/prompts/PROMPT_CHANGELOG.md) |
| Recorded live runs, eval reports, API smoke test, final CLI transcript | [docs/runs/](docs/runs/) |
| The eval set (10 cases with expectations) | [eval/eval_set.json](eval/eval_set.json) |
| The synthetic corpus (8 documents) | [data/corpus/](data/corpus/) |

---

## 1. Setup

Prerequisites: Python 3.11 or newer (developed on 3.13), about 2 GB of disk for the CPU torch
wheel, and an OpenAI API key for the language-model features (everything else runs without one).

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env        # then set OPENAI_API_KEY=sk-... in .env
```

```bash
# macOS / Linux / Git Bash
python -m venv .venv
source .venv/bin/activate      # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env           # then set OPENAI_API_KEY=sk-... in .env
```

The first run downloads the embedding model `all-MiniLM-L6-v2` (about 80 MB) into the Hugging
Face cache and builds the index (about 15 seconds). Later runs load the cached index from
`data/.index/` in under a second. Set `HF_HUB_OFFLINE=1` once the model is cached if the machine
is air-gapped.

Verify the install:

```bash
python -m pytest                 # 214 fast tests in about 3 s
python -m pytest -m slow         # plus the one test that loads the real embedding model
raapid-rag --help
```

Every command below accepts `--embedder hashing` to run without loading the embedding model
(fast, weaker retrieval; the tests use it) and most accept `--json` for machine-readable output.

## 2. Run it with Streamlit (recommended for a demo)

```bash
raapid-rag ui                                   # http://localhost:8501
# equivalent:
streamlit run src/raapid_rag/ui/app.py
```

The page shows status, confidence and the review flag first, then the answer, caveats, missing
information, every citation with its verification mark and the source chunk, tool calls,
warnings, and a trace panel with retrieval scores, cost, latency and confidence factors, plus a
JSON download. The sidebar has a model toggle (off means retrieval-only, no key needed), a
fault-injection selector to demonstrate the repair path, a document-scope picker, and the index
state. Example-question chips seed the input; earlier answers stay in the session.

Environment switches for demos without a key: `RAAPID_UI_USE_LLM=0` (retrieval only),
`RAAPID_UI_EMBEDDER=hashing` (fast embedder), `RAAPID_UI_CACHE_DIR=<dir>`.

## 3. Run it as an HTTP micro-service (FastAPI)

```bash
raapid-rag serve --host 0.0.0.0 --port 8000     # or: uvicorn raapid_rag.api:app --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | readiness probe: chunk count, index fingerprint, model, prompt version |
| GET | `/documents` | indexed documents for scope pickers |
| POST | `/search` | retrieval only: chunks, scores, gate decision (`{"query": "...", "doc_ids": [...]}`) |
| POST | `/query` | full grounded answer (`{"query": "...", "doc_ids": [...], "include_trace": true}`) |

Every model outcome is an HTTP 200 with `answer_status` set (`answered`, `partial`,
`insufficient_context`, `retrieval_only`, `error`); only an empty or over-long question is a 400
and a malformed body a 422. Every response carries an `X-Request-ID` header, also written into
`trace.request_id`.

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/query -H "content-type: application/json" \
  -d '{"query": "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?"}'

curl -s -X POST localhost:8000/query -H "content-type: application/json" \
  -d '{"query": "What does section 2.2 say about audio-only encounters?", "doc_ids": ["policy_002"], "include_trace": false}'
```

```powershell
# PowerShell equivalent
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query -ContentType "application/json" `
  -Body '{"query": "Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?"}'
```

```python
# Python client
import httpx
r = httpx.post("http://localhost:8000/query", json={"query": "What was Jane Doe-Test's most recent HbA1c?"}, timeout=60)
answer = r.json()
print(answer["answer_status"], answer["confidence_band"], answer["needs_human_review"])
for c in answer["citations"]:
    print(c["chunk_id"], c["verified"], c["evidence"])
```

Scaling: run more workers (`raapid-rag serve --workers 4`); each loads its own copy of the index
and model (about 14 s), so gate traffic on `/health`. Concurrency to the model provider is bounded
per process (`LLM_MAX_CONCURRENCY`) and repeated questions are served from an answer cache.

## 4. Run it from the command line

```bash
raapid-rag ingest                                # build or load the index; prints the report
raapid-rag search "Are audio-only telehealth visits acceptable?"     # retrieval only, with scores and gate
raapid-rag ask "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?"
raapid-rag ask "What metformin dose is recommended for CKD stage 3?"          # model-side refusal
raapid-rag ask "Who won the 2024 World Series?"                               # gate refusal, no model call
raapid-rag ask "What does section 2.2 say?" --doc-id policy_002 --json        # scoped, machine-readable
raapid-rag ask "audio-only telehealth" --no-llm                               # retrieval-only, no key needed
raapid-rag eval --mode retrieval                 # free CI check: gate behaviour and document recall
raapid-rag eval --mode full                      # about one US cent; reports in docs/runs/eval_full_latest.*
raapid-rag eval --mode full --min-pass-rate 0.9  # exit code 1 below the threshold
```

Demonstrate the repair path live by setting `FAULT_INJECTION=bad_citation` in `.env` (or the
environment) before `raapid-rag ask ...`; the first model response is deliberately corrupted and
the trace shows the repair. Leave it empty otherwise.

## 5. Use it as a Python library

```python
from raapid_rag.config import get_settings, DEFAULT_CACHE_DIR
from raapid_rag.service import QueryService

service = QueryService.from_settings(get_settings(), cache_dir=DEFAULT_CACHE_DIR)   # loads index + model once
answer = service.ask("How many days was John Q. Testpatient hospitalized?")
print(answer.answer_status, answer.confidence, answer.needs_human_review)
print(answer.model_dump_json(indent=2))

retrieval = service.search("audio-only telehealth", doc_ids=["policy_002"])          # no model call
print(retrieval.chunk_ids, retrieval.passes_gate)
```

## 6. Scripted runs and recorded evidence

```bash
python eval/retrieval_probe.py       # gate calibration on 13 probe queries (no model)
python eval/step4_live_run.py --versions v3,v4     # prompt versions side by side, raw model outputs recorded
python eval/step5_live_run.py        # every exit of the service path, including fault injection
python eval/step7_smoke.py           # starts uvicorn and Streamlit as real processes, exercises the API
python eval/run_eval.py --mode full  # same as raapid-rag eval
```

Recorded outputs live in `docs/runs/` (eval reports for every prompt configuration, the API smoke
test, and `final_transcript.md`, a captured CLI session).

## 7. Docker (written, not built in the development environment)

```bash
docker build -t raapid-rag .                    # bakes the embedding model and the prebuilt index into the image
docker run --rm -p 8000:8000 --env-file .env raapid-rag
docker compose up                               # API on 8000 and Streamlit on 8501
```

The image installs the CPU torch wheel, pre-downloads the embedding model and runs
`raapid-rag ingest` at build time, so containers start with `HF_HUB_OFFLINE=1` and no cold
embedding step. The API key is injected at run time, never baked in.

## 8. Configuration

Every knob is an environment variable with a documented default; see `.env.example` and
`src/raapid_rag/config.py`. The ones people change first: `LLM_MODEL`, `PROMPT_VERSION` (v1 to
v4), `RETRIEVAL_MIN_DENSE_SCORE` / `RETRIEVAL_MIN_BM25_SCORE` (the refusal gate), `PHI_POLICY`
(`flag` or `block`), `FAULT_INJECTION`, `LLM_MAX_CONCURRENCY`, `ANSWER_CACHE_SIZE`.

## 9. Layout

```
data/corpus/            8 synthetic documents (front matter + ## sections)
src/raapid_rag/
  config.py             every tunable, validated at startup
  ingestion/            loader, chunker, embedder, hybrid store (save/load), pipeline (fingerprint cache)
  retrieval/            retriever: hybrid pool, document anchoring, per-doc cap, token budget, refusal gate
  tools/                mock ICD-10, eligibility, calculator, days_between + registry with timeouts
  llm/                  OpenAI client wrapper (retries, usage, cost), prompting helpers, prompts v1-v4 + changelog
  schema.py             ModelOutput (model-facing, strict) and GroundedAnswer (service-facing), truth validation
  agent.py              reasoning loop: tools (memoised) -> strict JSON -> validate -> repair -> result
  guardrails.py         query validation, PHI scrub, citation enrichment, refusal blanking, salvage, tool policy, confidence
  service.py            QueryService: the wired request path, answer cache, bounded concurrency, health
  evaluation.py         eval harness: cases, mechanical scoring, aggregation, reports
  api.py                FastAPI: /health, /documents, /search, /query; request ids; envelopes, never 5xx for model outcomes
  ui/app.py             Streamlit page
  cli.py                ingest, search, ask, eval, serve, ui
eval/                   eval_set.json, run_eval.py, retrieval_probe.py, step4/5 live runs, step7_smoke.py
tests/                  214 fast tests (incl. API and headless Streamlit) + 1 slow model test
docs/                   DESIGN_NOTE.md, ENGINEERING_NOTES.md, runs/
Dockerfile, docker-compose.yml
```

Synthetic data only. No real PHI anywhere in this repository.
