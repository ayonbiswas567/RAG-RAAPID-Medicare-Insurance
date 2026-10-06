"""QueryService: the one place the request path is wired together.

The API, the CLI and the UI each hold one instance and call :meth:`ask`.  The service is
stateless apart from configuration, a bounded answer cache and a semaphore, so several
processes can run side by side behind a load balancer without coordination.

Request path (see docs/ENGINEERING_NOTES.md, Part 1.3):

    validate -> scrub -> cache lookup -> retrieve -> gate -> agent -> validate/salvage
    -> blank refusal prose -> enrich citations -> confidence -> envelope (+ trace)

Every exit produces a schema-valid :class:`GroundedAnswer`.  The only exception that escapes
is :class:`InvalidQueryError`, which means the caller sent something unusable (HTTP 400).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .agent import ChatClient, GroundedAgent
from .config import Settings
from .guardrails import (
    InvalidQueryError,
    blank_answer_on_refusal,
    compute_confidence,
    enrich_citations,
    required_tools_missing,
    salvage_citations,
    scrub_query,
    validate_query,
)
from .ingestion.embedder import Embedder, SentenceTransformerEmbedder
from .ingestion.models import Chunk
from .ingestion.pipeline import build_index
from .llm.client import LLMClient, LLMError, estimate_cost_usd
from .llm.prompting import load_system_prompt
from .retrieval.retriever import RetrievalResult, Retriever
from .schema import AnswerStatus, CitationOut, ConfidenceBand, GroundedAnswer
from .tools.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)

__all__ = ["QueryService", "InvalidQueryError"]


class QueryService:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: Retriever,
        tools: ToolRegistry,
        llm: ChatClient | None,
        system_prompt: str,
        index_fingerprint: str = "",
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._agent = GroundedAgent(llm, tools, settings, system_prompt) if llm is not None else None
        self._fingerprint = index_fingerprint
        self._semaphore = threading.Semaphore(settings.llm_max_concurrency)
        self._cache: OrderedDict[tuple, GroundedAnswer] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------ construction

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        cache_dir: Path | None = None,
        use_llm: bool = True,
    ) -> QueryService:
        """Build the full stack: index (from cache when possible), retriever, tools, model, prompt."""
        embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
        store, report = build_index(settings, embedder, cache_dir=cache_dir)
        llm = LLMClient(settings) if (use_llm and settings.llm_available) else None
        if llm is None:
            logger.warning("no language model configured; running in retrieval-only mode")
        return cls(
            settings=settings,
            retriever=Retriever(store, settings),
            tools=default_registry(),
            llm=llm,
            system_prompt=load_system_prompt(settings.prompt_version, settings.prompts_dir),
            index_fingerprint=report.fingerprint,
        )

    # ------------------------------------------------------------------ public

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "chunks": self._retriever.index_size,
            "index_fingerprint": self._fingerprint,
            "llm_available": self._llm is not None,
            "model": self._settings.llm_model,
            "prompt_version": self._settings.prompt_version,
            "embedding_model": self._settings.embedding_model,
            "cache_entries": len(self._cache),
        }

    def documents(self) -> list[dict[str, Any]]:
        """Indexed documents: id, title, type, date, chunk count. Feeds scope pickers."""
        return self._retriever.documents()

    def search(self, query: str, doc_ids: list[str] | None = None) -> RetrievalResult:
        """Retrieval only, no model: the same validation and retrieval the full path uses."""
        q = validate_query(query, max_chars=self._settings.max_query_chars)
        return self._retriever.retrieve(q, doc_ids=doc_ids)

    def ask(self, query: str, doc_ids: list[str] | None = None) -> GroundedAnswer:
        t0 = time.perf_counter()
        s = self._settings
        q = validate_query(query, max_chars=s.max_query_chars)

        scrub = scrub_query(q, policy=s.phi_policy)
        warnings = list(scrub.warnings)
        trace: dict[str, Any] = {
            "prompt_version": s.prompt_version,
            "model": s.llm_model,
            "index_fingerprint": self._fingerprint,
            "cache_hit": False,
        }
        if scrub.blocked:
            return self._finish(None, GroundedAnswer.error_envelope(
                q, errors=["request blocked by PHI policy: the query contains identifier-like patterns"],
                warnings=warnings, trace=trace), t0)

        key = self._cache_key(q, doc_ids)
        cached = self._cache_get(key)
        if cached is not None:
            hit_trace = {**cached.trace, "cache_hit": True, "total_latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return cached.model_copy(update={"trace": hit_trace})

        retrieval = self._retriever.retrieve(q, doc_ids=doc_ids)
        trace["retrieval"] = retrieval.to_dict()
        if retrieval.dropped_chunk_ids:
            warnings.append(f"{len(retrieval.dropped_chunk_ids)} retrieved chunk(s) dropped to fit the context budget")

        if not retrieval.passes_gate:
            missing = [f"No sufficiently relevant passage was found in the corpus for this question (gate: {retrieval.gate_reason})."]
            return self._finish(key, GroundedAnswer.insufficient_context_envelope(
                q, reason=retrieval.gate_reason, missing=missing, trace=trace, warnings=warnings), t0)

        chunks = [c.chunk for c in retrieval.chunks]
        if self._agent is None:
            return self._finish(key, self._retrieval_only(q, retrieval, chunks, trace, warnings), t0)

        try:
            with self._semaphore:
                result = self._agent.run(q, chunks)
        except LLMError as exc:
            logger.error("language model call failed: %s", exc)
            return self._finish(None, GroundedAnswer.error_envelope(
                q, errors=[f"language model call failed: {exc}"], warnings=warnings, trace=trace), t0)

        usage = result.usage
        trace.update({
            "llm_calls": result.llm_calls,
            "repair_attempts": result.repair_attempts,
            "forced_final": result.forced_final,
            "fault_injected": result.fault_injected,
            "usage": usage,
            "est_cost_usd": estimate_cost_usd(s.llm_model, prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"]),
            "agent_latency_ms": result.latency_ms,
            "validation_errors": result.validation_errors,
        })
        if result.fault_injected:
            warnings.append("fault injection was enabled; the first model response was deliberately corrupted for demonstration")
        if result.forced_final:
            warnings.append("tool-round limit reached; the final answer was forced without further tool calls")
        for rec in result.tool_records:
            if rec.error:
                warnings.append(f"tool '{rec.name}' failed: {rec.error}")

        output = result.model_output
        if output is None:
            salvaged, salvage_warnings = (salvage_citations(result.last_output, chunks) if result.last_output else (None, []))
            warnings.extend(salvage_warnings)
            if salvaged is None:
                return self._finish(None, GroundedAnswer.error_envelope(
                    q, errors=["model output failed validation after all repair attempts", *result.validation_errors],
                    warnings=warnings, trace=trace, tool_calls=result.tool_records), t0)
            output = salvaged

        status = AnswerStatus(output.answer_status)
        answer_text, blank_warnings = blank_answer_on_refusal(output.answer, status)
        warnings.extend(blank_warnings)
        citations = enrich_citations(output.citations, chunks) if status is not AnswerStatus.INSUFFICIENT_CONTEXT else []

        missing_tools = required_tools_missing(q, result.tool_records, s.tool_policy) if status is not AnswerStatus.INSUFFICIENT_CONTEXT else []
        if missing_tools:
            warnings.append(
                f"question appears to require tool(s) {', '.join(missing_tools)} but they were not called; "
                "answer capped below high confidence and flagged for review"
            )
        report = compute_confidence(
            status=status, retrieval=retrieval, citations=citations, tool_records=result.tool_records,
            repair_attempts=result.repair_attempts, self_reported=output.self_reported_confidence, settings=s,
            missing_required_tools=missing_tools,
        )
        trace["confidence_factors"] = report.factors
        trace["reasoning_summary"] = output.reasoning_summary

        answer = GroundedAnswer(
            query=q, answer=answer_text, answer_status=status, citations=citations, tool_calls=result.tool_records,
            confidence=report.confidence, confidence_band=report.band, missing_information=list(output.missing_information),
            caveats=list(output.caveats), needs_human_review=report.needs_human_review, warnings=warnings, errors=[], trace=trace,
        )
        return self._finish(key, answer, t0)

    # ----------------------------------------------------------------- helpers

    def _retrieval_only(self, q: str, retrieval: RetrievalResult, chunks: list[Chunk], trace: dict, warnings: list[str]) -> GroundedAnswer:
        citations = [
            CitationOut(chunk_id=c.chunk_id, evidence=c.text[:200], doc_id=c.doc_id, doc_title=c.doc_title,
                        section=c.section, date=c.date, verified=True)
            for c in chunks
        ]
        return GroundedAnswer(
            query=q, answer="", answer_status=AnswerStatus.RETRIEVAL_ONLY, citations=citations, tool_calls=[],
            confidence=0.0, confidence_band=ConfidenceBand.LOW,
            missing_information=["language model not configured (no OPENAI_API_KEY, or --no-llm); returning retrieved evidence only"],
            needs_human_review=True, warnings=warnings, errors=[], trace=trace,
        )

    def _finish(self, key: tuple | None, answer: GroundedAnswer, t0: float) -> GroundedAnswer:
        answer.trace["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if key is not None and answer.answer_status is not AnswerStatus.ERROR:
            self._cache_put(key, answer)
        return answer

    def _cache_key(self, q: str, doc_ids: list[str] | None) -> tuple:
        normalised = " ".join(q.casefold().split())
        return (normalised, tuple(sorted(doc_ids)) if doc_ids is not None else None, self._fingerprint)

    def _cache_get(self, key: tuple) -> GroundedAnswer | None:
        if self._settings.answer_cache_size <= 0:
            return None
        with self._cache_lock:
            answer = self._cache.get(key)
            if answer is not None:
                self._cache.move_to_end(key)
            return answer

    def _cache_put(self, key: tuple, answer: GroundedAnswer) -> None:
        if self._settings.answer_cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = answer
            self._cache.move_to_end(key)
            while len(self._cache) > self._settings.answer_cache_size:
                self._cache.popitem(last=False)
