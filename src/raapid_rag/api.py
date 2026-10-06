"""HTTP surface: a thin FastAPI shell over :class:`QueryService`.

Design points:

* **Every model outcome is a 200.** Answers, refusals, salvaged partials and provider outages
  all come back as a schema-valid :class:`GroundedAnswer` with ``answer_status`` set. A client
  never has to parse error bodies to learn that the model refused. Only unusable *input* is a
  4xx (400 for an empty or over-long question, 422 for a malformed body).
* **Sync handlers on purpose.** The handlers are plain ``def`` functions, so FastAPI runs them
  in its threadpool and the blocking model call never stalls the event loop. Concurrency to the
  provider is bounded inside the service (semaphore); more capacity means more processes.
* **Request ids.** Middleware stamps every response with ``X-Request-ID`` and the id is written
  into the answer's trace, so a reviewer's screenshot can be matched to a server log line.
* **No question text in logs.** The access log records method, path, status, latency and the
  request id, never the query, because with real data the query itself may contain PHI.
* **App factory.** :func:`create_app` accepts a prebuilt service (tests inject a fake model);
  without one it builds the real service at startup (lifespan), which loads the embedding model
  once per process. ``GET /health`` doubles as the readiness probe.

Run:  uvicorn raapid_rag.api:app --host 0.0.0.0 --port 8000     (or: raapid-rag serve)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import DEFAULT_CACHE_DIR, Settings, get_settings
from .guardrails import InvalidQueryError
from .schema import GroundedAnswer
from .service import QueryService

logger = logging.getLogger(__name__)

TRACE_KEYS_WITHOUT_RETRIEVAL = ("prompt_version", "model", "index_fingerprint", "cache_hit", "llm_calls", "repair_attempts",
                                "usage", "est_cost_usd", "total_latency_ms", "confidence_factors", "gate_reason", "request_id")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="Natural-language question.")
    doc_ids: list[str] | None = Field(default=None, description="Restrict retrieval to these document ids.")
    include_trace: bool = Field(default=True, description="False returns only summary trace fields (no chunk text).")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = None


def create_app(service: QueryService | None = None, *, settings: Settings | None = None) -> FastAPI:
    state: dict[str, QueryService | None] = {"service": service}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["service"] is None:
            s = settings or get_settings()
            logger.info("building QueryService at startup (model %s, prompt %s)", s.llm_model, s.prompt_version)
            state["service"] = QueryService.from_settings(s, cache_dir=DEFAULT_CACHE_DIR)
        yield

    app = FastAPI(
        title="RAAPID grounded GenAI micro-service",
        version=__version__,
        description="Question answering strictly grounded in an ingested corpus, with verified citations, "
                    "tool calls, and a schema-enforced JSON envelope that includes a confidence and review signal.",
        lifespan=lifespan,
    )

    def svc() -> QueryService:
        if state["service"] is None:  # pragma: no cover - only if called before lifespan
            raise RuntimeError("service not initialised")
        return state["service"]

    @app.middleware("http")
    async def request_id_and_access_log(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s -> %d in %.0f ms [%s]", request.method, request.url.path, response.status_code,
                    (time.perf_counter() - t0) * 1000, request_id)
        return response

    @app.exception_handler(InvalidQueryError)
    async def invalid_query(_: Request, exc: InvalidQueryError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "raapid-rag", "version": __version__, "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return svc().health()

    @app.get("/documents")
    def documents() -> list[dict[str, Any]]:
        return svc().documents()

    @app.post("/search")
    def search(body: SearchRequest) -> dict[str, Any]:
        return svc().search(body.query, doc_ids=body.doc_ids).to_dict()

    @app.post("/query", response_model=GroundedAnswer)
    def query(body: QueryRequest, request: Request) -> GroundedAnswer:
        answer = svc().ask(body.query, doc_ids=body.doc_ids)
        trace = dict(answer.trace)
        if not body.include_trace:
            trace = {k: v for k, v in trace.items() if k in TRACE_KEYS_WITHOUT_RETRIEVAL}
        trace["request_id"] = request.state.request_id
        return answer.model_copy(update={"trace": trace})

    return app


# Module-level app for `uvicorn raapid_rag.api:app`.
app = create_app()
