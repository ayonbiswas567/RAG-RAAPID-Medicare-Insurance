"""Output contract tests: strict-schema compatibility, grounding validation, envelopes."""

import json

import pytest
from pydantic import ValidationError

from raapid_rag.ingestion.models import Chunk
from raapid_rag.schema import (
    AnswerStatus,
    Citation,
    ConfidenceBand,
    GroundedAnswer,
    ModelOutput,
    strict_json_schema,
    validate_grounding,
)

CHUNK = Chunk(
    chunk_id="note_001#c05",
    doc_id="note_001",
    doc_type="clinical_note",
    doc_title="Endocrinology Progress Note",
    section="Laboratory Results",
    text="HbA1c 8.2% (collected 2025-03-12). eGFR 52 mL/min/1.73m2 (2025-03-12), consistent with\nprior values of 50 to 55 over the past six months.",
    index=5,
)


def output(**overrides) -> ModelOutput:
    base = dict(
        answer="The most recent HbA1c was 8.2%, collected on 2025-03-12.",
        answer_status="answered",
        citations=[Citation(chunk_id="note_001#c05", evidence="HbA1c 8.2% (collected 2025-03-12).")],
        missing_information=[],
        caveats=[],
        self_reported_confidence=0.9,
        reasoning_summary="The lab section states the value directly.",
    )
    base.update(overrides)
    return ModelOutput(**base)


def test_model_output_requires_caveats_field():
    # The outlet for side issues is part of the contract, so the model must always emit it.
    schema = strict_json_schema(ModelOutput)
    assert "caveats" in schema["required"]
    assert schema["properties"]["caveats"]["type"] == "array"


# ------------------------------------------------------------- strict schema


def _walk_objects(node, path="root"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for key, value in node.items():
            yield from _walk_objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk_objects(item, f"{path}[{i}]")


def test_model_output_schema_is_openai_strict_compatible():
    schema = strict_json_schema(ModelOutput)
    objects = list(_walk_objects(schema))
    assert objects, "schema has no object nodes"
    for path, obj in objects:
        assert obj.get("additionalProperties") is False, path
        assert sorted(obj.get("required", [])) == sorted(obj.get("properties", {}).keys()), path
    assert not _has_key(schema, "default"), "strict mode forbids defaults"
    assert schema["properties"]["answer_status"]["enum"] == ["answered", "partial", "insufficient_context"]


def _has_key(node, key) -> bool:
    if isinstance(node, dict):
        return key in node or any(_has_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_has_key(v, key) for v in node)
    return False


# --------------------------------------------------------- grounding checks


def test_valid_output_has_no_grounding_errors():
    assert validate_grounding(output(), [CHUNK]) == []


def test_unknown_chunk_id_is_reported_with_valid_ids():
    out = output(citations=[Citation(chunk_id="note_009#c02", evidence="HbA1c 8.2%")])
    errors = validate_grounding(out, [CHUNK])
    assert len(errors) == 1
    assert "note_009#c02" in errors[0] and "note_001#c05" in errors[0]


def test_non_verbatim_evidence_is_rejected():
    out = output(citations=[Citation(chunk_id="note_001#c05", evidence="The HbA1c was 8.2 percent.")])
    errors = validate_grounding(out, [CHUNK])
    assert len(errors) == 1
    assert "verbatim" in errors[0] and "note_001#c05" in errors[0]


def test_whitespace_quote_and_case_differences_are_tolerated():
    out = output(citations=[Citation(chunk_id="note_001#c05", evidence="consistent with  prior values of 50 to 55")])
    assert validate_grounding(out, [CHUNK]) == []
    chunk = Chunk(**{**CHUNK.__dict__, "text": 'described as "history of" breast cancer'})
    out = output(citations=[Citation(chunk_id="note_001#c05", evidence="Described as “history of” breast cancer")])
    assert validate_grounding(out, [chunk]) == []


def test_answered_without_citations_is_rejected():
    errors = validate_grounding(output(citations=[]), [CHUNK])
    assert any("at least one citation" in e for e in errors)


def test_answered_with_empty_answer_is_rejected():
    errors = validate_grounding(output(answer="   "), [CHUNK])
    assert any("answer must not be empty" in e for e in errors)


def test_insufficient_context_requires_missing_information():
    out = output(answer="", answer_status="insufficient_context", citations=[], missing_information=[])
    errors = validate_grounding(out, [CHUNK])
    assert any("missing_information" in e for e in errors)
    ok = output(answer="", answer_status="insufficient_context", citations=[], missing_information=["No LDL value is recorded."])
    assert validate_grounding(ok, [CHUNK]) == []


def test_empty_evidence_is_rejected():
    out = output(citations=[Citation(chunk_id="note_001#c05", evidence="  ")])
    assert any("empty evidence" in e for e in validate_grounding(out, [CHUNK]))


def test_out_of_range_self_reported_confidence_is_rejected():
    assert any("self_reported_confidence" in e for e in validate_grounding(output(self_reported_confidence=1.7), [CHUNK]))


# ------------------------------------------------------------------ envelopes


def test_error_envelope_is_valid_and_flags_review():
    env = GroundedAnswer.error_envelope("q", errors=["validation failed twice"], trace={"repair_attempts": 2})
    assert env.answer_status is AnswerStatus.ERROR
    assert env.confidence == 0.0
    assert env.confidence_band is ConfidenceBand.LOW
    assert env.needs_human_review is True
    assert env.errors == ["validation failed twice"]
    assert env.citations == [] and env.answer == ""
    round_trip = GroundedAnswer.model_validate_json(env.model_dump_json())
    assert round_trip == env


def test_insufficient_context_envelope_explains_why():
    env = GroundedAnswer.insufficient_context_envelope("q", reason="low_relevance", missing=["Nothing relevant retrieved."])
    assert env.answer_status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert env.missing_information == ["Nothing relevant retrieved."]
    assert env.needs_human_review is True
    assert env.trace["gate_reason"] == "low_relevance"


def test_grounded_answer_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        GroundedAnswer(
            query="q", answer="a", answer_status=AnswerStatus.ANSWERED, citations=[], tool_calls=[],
            confidence=1.5, confidence_band=ConfidenceBand.HIGH, missing_information=[],
            needs_human_review=False, warnings=[], errors=[], trace={},
        )
