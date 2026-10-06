"""raapid_rag - grounded question answering over clinical and policy documents.

Package layout (built incrementally; see docs/ENGINEERING_NOTES.md):

    config.py       every tunable knob, read from .env, validated at startup
    ingestion/      load -> chunk -> embed -> in-memory hybrid index
    retrieval/      query -> ranked chunks with scores (dense + BM25, fused)
    tools/          mock ICD-10 lookup, eligibility check, calculator + registry
    llm/            OpenAI client wrapper (retries, timeouts, token accounting) + prompts
    schema.py       the enforced JSON output contract (pydantic)
    agent.py        reasoning loop: retrieve -> tool calls -> structured answer -> validate/repair
    guardrails.py   refusal gate, evidence verification, confidence, PHI scrub
    service.py      QueryService facade shared by API, CLI and UI
    api.py / cli.py / ui/app.py   thin surfaces over the service
"""

__version__ = "1.0.0"
