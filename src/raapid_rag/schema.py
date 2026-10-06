"""The output contract.

Two schemas, on purpose:

* :class:`ModelOutput` is what the **language model** must produce.  It is small, has no
  defaults or optionals (OpenAI strict mode forbids both), and contains only what the model
  can honestly know: an answer, citations with verbatim evidence, what was missing, and a
  self-assessment.
* :class:`GroundedAnswer` is what the **service** returns.  It wraps the model's output with
  facts only our code can establish: which tools ran and what they returned, a confidence
  computed from verifiable signals, whether a human must review, warnings, errors, and a
  trace.  The caller always receives a valid ``GroundedAnswer``, even when the model
  misbehaved (see the envelope constructors).

:func:`validate_grounding` is the "truth" layer described in decision D6: checks a JSON
schema cannot express, such as "this chunk id was actually retrieved" and "this quote really
appears in that chunk".
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .ingestion.models import Chunk

SCHEMA_VERSION = "1.0"


# ------------------------------------------------------------- model-facing


class Citation(BaseModel):
    """A pointer from a claim to the exact retrieved text that supports it."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="One of the chunk ids shown in the context, e.g. 'note_001#c05'.")
    evidence: str = Field(description="A short verbatim quote copied exactly from that chunk.")


ModelAnswerStatus = Literal["answered", "partial", "insufficient_context"]


class ModelOutput(BaseModel):
    """Exactly what the LLM is asked to emit. Strict-mode compatible: no defaults, no optionals."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Plain-language answer grounded only in the context and tool results. Empty when insufficient_context.")
    answer_status: ModelAnswerStatus = Field(description="'answered' = fully supported; 'partial' = some of the question is supported; 'insufficient_context' = the context does not support an answer.")
    citations: list[Citation] = Field(description="Every factual claim must be supported by at least one citation. Empty when insufficient_context.")
    missing_information: list[str] = Field(description="What the context did not contain that would be needed. Empty when fully answered.")
    caveats: list[str] = Field(description="Considerations that do not change the answer: a tool result about a separate matter, a related rule, a follow-up that has not happened. Put such concerns here and still answer; a concern is never a reason to refuse. Empty if none.")
    self_reported_confidence: float = Field(description="Your own confidence between 0 and 1 that the answer is fully supported by the cited evidence.")
    reasoning_summary: str = Field(description="One or two sentences on how the answer was derived from the context and tools.")


# ----------------------------------------------------------- service-facing


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    RETRIEVAL_ONLY = "retrieval_only"  # no LLM configured; chunks returned without an answer
    ERROR = "error"  # model output could not be repaired, or the provider was unavailable


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CitationOut(Citation):
    """Citation enriched with document metadata so a reviewer can find the source quickly."""

    doc_id: str
    doc_title: str
    section: str
    date: str = ""
    verified: bool = Field(description="True when the evidence was confirmed to be a verbatim quote from the chunk.")


class ToolCallRecord(BaseModel):
    """What our code observed when executing one tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    timed_out: bool = False
    cached: bool = Field(default=False, description="True when an identical call earlier in the same request was reused.")


class GroundedAnswer(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query: str
    answer: str
    answer_status: AnswerStatus
    citations: list[CitationOut]
    tool_calls: list[ToolCallRecord]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    missing_information: list[str]
    caveats: list[str] = Field(default_factory=list, description="Side issues the model raised that do not change the answer; shown to the reviewer.")
    needs_human_review: bool
    warnings: list[str]
    errors: list[str]
    trace: dict[str, Any]

    # ---- deterministic envelopes: the service must always be able to answer *something* ----

    @classmethod
    def error_envelope(
        cls, query: str, *, errors: list[str], trace: dict[str, Any] | None = None, warnings: list[str] | None = None,
        tool_calls: list[ToolCallRecord] | None = None,
    ) -> GroundedAnswer:
        return cls(
            query=query, answer="", answer_status=AnswerStatus.ERROR, citations=[], tool_calls=tool_calls or [],
            confidence=0.0, confidence_band=ConfidenceBand.LOW, missing_information=[],
            needs_human_review=True, warnings=warnings or [], errors=errors, trace=trace or {},
        )

    @classmethod
    def insufficient_context_envelope(
        cls, query: str, *, reason: str, missing: list[str], trace: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> GroundedAnswer:
        trace = dict(trace or {})
        trace["gate_reason"] = reason
        return cls(
            query=query, answer="", answer_status=AnswerStatus.INSUFFICIENT_CONTEXT, citations=[], tool_calls=[],
            confidence=0.0, confidence_band=ConfidenceBand.LOW, missing_information=missing,
            needs_human_review=True, warnings=warnings or [], errors=[], trace=trace,
        )


# --------------------------------------------------------------- utilities


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for OpenAI strict mode: every object closed, every property required."""
    schema = model.model_json_schema()
    _make_strict(schema)
    for definition in schema.get("$defs", {}).values():
        _make_strict(definition)
    return schema


def _make_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}).keys())
        node.pop("default", None)
        for value in node.values():
            _make_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_strict(item)


_WS_RE = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_for_match(text: str) -> str:
    """Whitespace-collapsed, quote-normalised, case-folded text for verbatim comparison."""
    return _WS_RE.sub(" ", text.translate(_QUOTE_MAP)).strip().casefold()


def validate_grounding(output: ModelOutput, chunks: list[Chunk]) -> list[str]:
    """Return human-readable errors for everything the JSON schema cannot check.

    The messages are written to be fed straight back to the model as repair instructions,
    so they name the offending field, the offending value, and what would be valid.
    """
    by_id = {c.chunk_id: c for c in chunks}
    valid_ids = sorted(by_id)
    errors: list[str] = []

    for i, cit in enumerate(output.citations):
        if not cit.evidence.strip():
            errors.append(f"citations[{i}] has empty evidence; quote the supporting text verbatim")
            continue
        chunk = by_id.get(cit.chunk_id)
        if chunk is None:
            errors.append(
                f"citations[{i}] references unknown chunk_id '{cit.chunk_id}'; valid chunk ids are: {', '.join(valid_ids)}"
            )
            continue
        if normalize_for_match(cit.evidence) not in normalize_for_match(chunk.text):
            errors.append(
                f"citations[{i}] evidence is not a verbatim quote from {cit.chunk_id}: {cit.evidence[:120]!r}. "
                "Copy the supporting sentence exactly as it appears in that chunk."
            )

    if output.answer_status in ("answered", "partial"):
        if not output.citations:
            errors.append(f"answer_status '{output.answer_status}' requires at least one citation")
        if not output.answer.strip():
            errors.append("answer must not be empty when answer_status is 'answered' or 'partial'")
    elif output.answer_status == "insufficient_context" and not output.missing_information:
        errors.append("answer_status 'insufficient_context' requires at least one missing_information entry explaining what the context lacks")

    if not 0.0 <= output.self_reported_confidence <= 1.0:
        errors.append(f"self_reported_confidence must be between 0 and 1 (got {output.self_reported_confidence})")

    return errors
