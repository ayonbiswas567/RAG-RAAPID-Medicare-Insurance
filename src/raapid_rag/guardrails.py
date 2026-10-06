"""Deterministic checks around the probabilistic core.

Everything in this module is a pure function: no model calls, no I/O, no state.  That is
the point.  These are the guarantees the service can make regardless of what the model does:

* :func:`validate_query` and :func:`scrub_query` run *before* retrieval: reject junk, and flag
  (or block) identifier-like patterns so PHI is not sent in free text.
* :func:`enrich_citations` turns the model's bare citations into reviewer-facing citations with
  document metadata and a ``verified`` flag computed from the source text.
* :func:`blank_answer_on_refusal` guarantees that no uncited prose reaches the caller when the
  model says the context was insufficient (observed with prompt v1: a refusal followed by
  "however, it is generally recommended that...").
* :func:`salvage_citations` turns "repairs exhausted" into a flagged partial answer when at least
  one citation is verifiable, instead of throwing the whole answer away (observed live: five
  verified citations and one tool sentence miscited as evidence).
* :func:`compute_confidence` combines verifiable signals into one number and one decision a
  reviewer can act on.  The weights are deliberately simple and are reported back in the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .ingestion.models import Chunk
from .retrieval.retriever import RetrievalResult
from .schema import AnswerStatus, Citation, CitationOut, ConfidenceBand, ModelOutput, ToolCallRecord, normalize_for_match


class InvalidQueryError(ValueError):
    """The request itself is unusable (empty, too long). Maps to HTTP 400 in the API."""


# ---------------------------------------------------------------- query checks


def validate_query(query: str, *, max_chars: int) -> str:
    q = (query or "").strip()
    if not q:
        raise InvalidQueryError("query must not be empty")
    if len(q) > max_chars:
        raise InvalidQueryError(f"query is {len(q)} characters; the limit is {max_chars}")
    return q


# Patterns are deliberately format-based (high recall on formats, no recall on names).  ISO
# dates (2025-03-12), decimals (8.2), codes (E11.22) and member ids (M-TEST-1002) must not match.
PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("US Social Security number", re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")),
    ("phone number", re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?!\d)")),
    ("date of birth (MM/DD/YYYY)", re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b")),
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]


@dataclass
class ScrubResult:
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    matched_labels: list[str] = field(default_factory=list)


def scrub_query(query: str, *, policy: str) -> ScrubResult:
    """Flag or block identifier-like patterns. ``policy`` is 'flag' (mock data) or 'block'."""
    result = ScrubResult()
    for label, pattern in PHI_PATTERNS:
        if pattern.search(query):
            result.matched_labels.append(label)
            result.warnings.append(
                f"query contains a pattern that looks like a {label}; identifiers must not be sent in free text "
                f"(PHI policy: {policy})"
            )
    result.blocked = bool(result.matched_labels) and policy == "block"
    return result


# ------------------------------------------------------------ citation helpers


def enrich_citations(citations: list[Citation], chunks: list[Chunk]) -> list[CitationOut]:
    by_id = {c.chunk_id: c for c in chunks}
    out: list[CitationOut] = []
    for cit in citations:
        chunk = by_id.get(cit.chunk_id)
        if chunk is None:
            out.append(CitationOut(chunk_id=cit.chunk_id, evidence=cit.evidence, doc_id="unknown", doc_title="unknown",
                                   section="unknown", date="", verified=False))
            continue
        verified = bool(cit.evidence.strip()) and normalize_for_match(cit.evidence) in normalize_for_match(chunk.text)
        out.append(CitationOut(chunk_id=cit.chunk_id, evidence=cit.evidence, doc_id=chunk.doc_id, doc_title=chunk.doc_title,
                               section=chunk.section, date=chunk.date, verified=verified))
    return out


def blank_answer_on_refusal(answer: str, status: AnswerStatus | str) -> tuple[str, list[str]]:
    """No uncited prose reaches the caller on a refusal, whatever the model wrote."""
    status_value = status.value if isinstance(status, AnswerStatus) else str(status)
    if status_value == AnswerStatus.INSUFFICIENT_CONTEXT.value and answer.strip():
        return "", ["model answer text discarded because answer_status is insufficient_context; "
                    "uncited prose never reaches the caller"]
    return answer, []


def salvage_citations(output: ModelOutput, chunks: list[Chunk]) -> tuple[ModelOutput | None, list[str]]:
    """After repairs are exhausted: drop unverifiable citations, downgrade to partial, warn.

    Returns ``(None, warnings)`` when nothing verifiable remains; the caller then emits an error
    envelope.
    """
    enriched = enrich_citations(output.citations, chunks)
    kept = [cit for cit, meta in zip(output.citations, enriched) if meta.verified]
    dropped = len(output.citations) - len(kept)
    if not kept:
        return None, [f"no verifiable citation remained after dropping {dropped} unverifiable citation(s)"]
    status = "partial" if output.answer_status == "answered" else output.answer_status
    salvaged = output.model_copy(update={"citations": kept, "answer_status": status})
    return salvaged, [f"{dropped} unverifiable citation(s) removed after repair attempts were exhausted; "
                      f"answer downgraded to '{status}' and flagged for review"]


# ----------------------------------------------------------------- tool policy


def required_tools_missing(query: str, tool_records: list[ToolCallRecord], policy: dict[str, str]) -> list[str]:
    """Tools the question appears to need (by pattern) that were not called successfully.

    Deterministic complement to the prompt's MUST rules: the prompt asks the model to call a
    tool; this notices when it did not, so the omission becomes a review flag instead of a
    silent high-confidence answer.
    """
    called = {r.name for r in tool_records if r.error is None and not r.timed_out}
    q = query.casefold()
    return [tool for tool, pattern in policy.items() if re.search(pattern, q) and tool not in called]


def tool_induced_refusal(
    query: str,
    output: ModelOutput,
    tool_records: list[ToolCallRecord],
    policy: dict[str, str],
    topics: dict[str, list[str]],
) -> str | None:
    """Detect a refusal whose stated reason is a tool result on a topic the question never asked about.

    Observed live: asked whether an uncertain diagnosis can be coded, the model called the
    eligibility tool anyway, then refused "due to the member's inactive coverage" while its own
    reasoning summary stated the correct coding answer. Returns a repair instruction, or None.
    """
    if output.answer_status != "insufficient_context":
        return None
    q = query.casefold()
    stated = " ".join(output.missing_information + output.caveats + [output.reasoning_summary]).casefold()
    for record in tool_records:
        words = topics.get(record.name)
        if not words or record.error:
            continue
        pattern = policy.get(record.name)
        if pattern and re.search(pattern, q):
            continue  # the question was about this tool's topic; the refusal may be legitimate
        if any(w in stated for w in words):
            return (
                f"Your refusal is based on the '{record.name}' tool result, but the question did not ask about that topic. "
                "Answer the question from the context (set answer_status to \"answered\" or \"partial\" with citations) "
                "and put the tool finding in caveats instead of missing_information."
            )
    return None


# ------------------------------------------------------------------ confidence


@dataclass
class ConfidenceReport:
    confidence: float
    band: ConfidenceBand
    needs_human_review: bool
    factors: dict[str, Any]


WEIGHTS = {"retrieval": 0.35, "citations": 0.35, "tools": 0.10, "repairs": 0.10, "self_report": 0.10}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_confidence(
    *,
    status: AnswerStatus,
    retrieval: RetrievalResult,
    citations: list[CitationOut],
    tool_records: list[ToolCallRecord],
    repair_attempts: int,
    self_reported: float | None,
    settings: Settings,
    missing_required_tools: list[str] | None = None,
) -> ConfidenceReport:
    """Composite of verifiable signals; the model's own opinion is the smallest term."""
    if status not in (AnswerStatus.ANSWERED, AnswerStatus.PARTIAL):
        return ConfidenceReport(0.0, ConfidenceBand.LOW, True, {"reason": f"no answer to score (status {status.value})"})
    missing_required_tools = list(missing_required_tools or [])

    dense_span = max(settings.retrieval_strong_dense_score - settings.retrieval_min_dense_score, 1e-6)
    bm25_span = max(settings.retrieval_strong_bm25_score - settings.retrieval_min_bm25_score, 1e-6)
    r = max(
        _clamp((retrieval.best_dense_score - settings.retrieval_min_dense_score) / dense_span),
        _clamp((retrieval.best_bm25_score - settings.retrieval_min_bm25_score) / bm25_span),
    )
    n_verified = sum(1 for c in citations if c.verified)
    v = n_verified / len(citations) if citations else 0.0
    failed_tools = sum(1 for t in tool_records if t.error or t.timed_out)
    t = 1.0 - failed_tools / len(tool_records) if tool_records else 1.0
    p = {0: 1.0, 1: 0.8}.get(repair_attempts, 0.6)
    s = _clamp(self_reported if self_reported is not None else 0.5)

    confidence = WEIGHTS["retrieval"] * r + WEIGHTS["citations"] * v + WEIGHTS["tools"] * t + WEIGHTS["repairs"] * p + WEIGHTS["self_report"] * s
    if status is AnswerStatus.PARTIAL or missing_required_tools:
        confidence = min(confidence, settings.confidence_high - 0.01)  # never "high" when partial or a required tool was skipped
    confidence = round(_clamp(confidence), 3)

    if confidence >= settings.confidence_high:
        band = ConfidenceBand.HIGH
    elif confidence >= settings.confidence_medium:
        band = ConfidenceBand.MEDIUM
    else:
        band = ConfidenceBand.LOW

    clean = (
        status is AnswerStatus.ANSWERED
        and band is ConfidenceBand.HIGH
        and bool(citations)
        and n_verified == len(citations)
        and failed_tools == 0
        and repair_attempts == 0
        and not missing_required_tools
    )
    factors = {
        "retrieval_strength": round(r, 3),
        "citations_verified": f"{n_verified}/{len(citations)}",
        "tool_health": round(t, 3),
        "repair_penalty": p,
        "self_reported": round(s, 3),
        "missing_required_tools": missing_required_tools,
        "weights": WEIGHTS,
    }
    return ConfidenceReport(confidence, band, not clean, factors)
