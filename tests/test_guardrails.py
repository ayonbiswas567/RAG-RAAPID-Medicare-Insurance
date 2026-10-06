"""Guardrail tests: query validation and PHI scrub, citation enrichment, refusal blanking,
citation salvage, and the confidence composite."""

import pytest

from raapid_rag.config import Settings
from raapid_rag.guardrails import (
    InvalidQueryError,
    blank_answer_on_refusal,
    compute_confidence,
    enrich_citations,
    required_tools_missing,
    salvage_citations,
    scrub_query,
    validate_query,
)
from raapid_rag.ingestion.models import Chunk
from raapid_rag.ingestion.store import ScoredChunk
from raapid_rag.retrieval.retriever import RetrievalResult
from raapid_rag.schema import AnswerStatus, Citation, CitationOut, ConfidenceBand, ModelOutput, ToolCallRecord

CHUNK = Chunk(
    chunk_id="note_001#c05", doc_id="note_001", doc_type="clinical_note", doc_title="Endocrinology Progress Note",
    section="Laboratory Results", text="HbA1c 8.2% (collected 2025-03-12). eGFR 52 mL/min/1.73m2.", index=5,
    metadata={"date_of_service": "2025-03-14"},
)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def retrieval(best_dense=0.6, best_bm25=12.0) -> RetrievalResult:
    scored = ScoredChunk(chunk=CHUNK, dense_score=best_dense, bm25_score=best_bm25, fused_score=0.03, dense_rank=1, bm25_rank=1)
    return RetrievalResult(query="q", chunks=[scored], best_dense_score=best_dense, best_bm25_score=best_bm25, passes_gate=True, gate_reason="ok")


# ------------------------------------------------------------- query checks


def test_validate_query_strips_and_returns_text():
    assert validate_query("  What is the A1c?  ", max_chars=100) == "What is the A1c?"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_validate_query_rejects_empty(bad):
    with pytest.raises(InvalidQueryError, match="empty"):
        validate_query(bad, max_chars=100)


def test_validate_query_rejects_over_long():
    with pytest.raises(InvalidQueryError, match="100"):
        validate_query("x" * 101, max_chars=100)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Patient SSN 123-45-6789 asks about coverage", "Social Security"),
        ("Call me at (555) 123-4567 about the claim", "phone"),
        ("Born 03/03/1958, what conditions are coded?", "date of birth"),
        ("Send results to jane.doe@example.com", "email"),
    ],
)
def test_scrub_flags_phi_like_patterns(query, expected):
    result = scrub_query(query, policy="flag")
    assert result.blocked is False
    assert any(expected in w for w in result.warnings)


def test_scrub_is_silent_on_clean_query():
    result = scrub_query("What was the most recent HbA1c?", policy="flag")
    assert result.warnings == [] and result.blocked is False


def test_scrub_blocks_under_block_policy():
    result = scrub_query("SSN 123-45-6789", policy="block")
    assert result.blocked is True
    assert result.warnings


def test_scrub_does_not_flag_iso_dates_or_lab_values():
    # ISO service dates and decimals are normal clinical content, not PHI patterns.
    assert scrub_query("HbA1c 8.2% collected 2025-03-12", policy="flag").warnings == []


# --------------------------------------------------------- citation helpers


def test_enrich_citations_adds_metadata_and_verified_flag():
    cits = [Citation(chunk_id="note_001#c05", evidence="HbA1c 8.2% (collected 2025-03-12)."),
            Citation(chunk_id="note_001#c05", evidence="not in the chunk at all")]
    out = enrich_citations(cits, [CHUNK])
    assert out[0].verified is True and out[0].doc_title == "Endocrinology Progress Note"
    assert out[0].section == "Laboratory Results" and out[0].date == "2025-03-14"
    assert out[1].verified is False


def test_enrich_citations_marks_unknown_chunk_unverified():
    out = enrich_citations([Citation(chunk_id="note_999#c99", evidence="x")], [CHUNK])
    assert out[0].verified is False and out[0].doc_id == "unknown"


def test_blank_answer_on_refusal_discards_text_and_warns():
    answer, warnings = blank_answer_on_refusal("The context lacks this, but generally...", AnswerStatus.INSUFFICIENT_CONTEXT)
    assert answer == ""
    assert warnings and "discarded" in warnings[0]


def test_blank_answer_on_refusal_leaves_answers_alone():
    assert blank_answer_on_refusal("The A1c was 8.2%.", AnswerStatus.ANSWERED) == ("The A1c was 8.2%.", [])
    assert blank_answer_on_refusal("", AnswerStatus.INSUFFICIENT_CONTEXT) == ("", [])


def test_salvage_drops_invalid_citations_and_downgrades_to_partial():
    out = ModelOutput(
        answer="A1c was 8.2%.", answer_status="answered",
        citations=[Citation(chunk_id="note_001#c05", evidence="HbA1c 8.2% (collected 2025-03-12)."),
                   Citation(chunk_id="note_001#c05", evidence="Coverage not active on the date of service.")],
        missing_information=[], caveats=[], self_reported_confidence=0.9, reasoning_summary="r",
    )
    salvaged, warnings = salvage_citations(out, [CHUNK])
    assert salvaged is not None
    assert [c.chunk_id for c in salvaged.citations] == ["note_001#c05"] and len(salvaged.citations) == 1
    assert salvaged.answer_status == "partial"
    assert any("1 unverifiable citation" in w for w in warnings)


def test_salvage_returns_none_when_no_citation_survives():
    out = ModelOutput(
        answer="x", answer_status="answered",
        citations=[Citation(chunk_id="note_999#c99", evidence="x")],
        missing_information=[], caveats=[], self_reported_confidence=0.9, reasoning_summary="r",
    )
    salvaged, warnings = salvage_citations(out, [CHUNK])
    assert salvaged is None and warnings


# --------------------------------------------------------------- confidence


def verified(n: int, verified_n: int) -> list[CitationOut]:
    return [CitationOut(chunk_id="note_001#c05", evidence="e", doc_id="note_001", doc_title="t", section="s", verified=i < verified_n) for i in range(n)]


def test_clean_answer_is_high_confidence_and_needs_no_review():
    report = compute_confidence(
        status=AnswerStatus.ANSWERED, retrieval=retrieval(0.65, 16.0), citations=verified(2, 2),
        tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings(),
    )
    assert report.band is ConfidenceBand.HIGH
    assert report.confidence >= 0.8
    assert report.needs_human_review is False
    assert report.factors["citations_verified"] == "2/2"


def test_unverified_citation_lowers_confidence_and_forces_review():
    clean = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(2, 2), tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings())
    dirty = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(2, 1), tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings())
    assert dirty.confidence < clean.confidence
    assert dirty.needs_human_review is True


def test_repairs_and_tool_errors_penalise_and_force_review():
    base = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(1, 1), tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings())
    repaired = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(1, 1), tool_records=[], repair_attempts=2, self_reported=0.9, settings=settings())
    tool_err = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(1, 1),
                                  tool_records=[ToolCallRecord(name="t", arguments={}, error="timed out", timed_out=True)], repair_attempts=0, self_reported=0.9, settings=settings())
    assert repaired.confidence < base.confidence and repaired.needs_human_review is True
    assert tool_err.confidence < base.confidence and tool_err.needs_human_review is True


def test_weak_retrieval_lowers_confidence():
    strong = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(0.7, 18.0), citations=verified(1, 1), tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings())
    weak = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(0.38, 5.1), citations=verified(1, 1), tool_records=[], repair_attempts=0, self_reported=0.9, settings=settings())
    assert weak.confidence < strong.confidence


def test_partial_answers_are_capped_at_medium():
    report = compute_confidence(status=AnswerStatus.PARTIAL, retrieval=retrieval(0.7, 18.0), citations=verified(2, 2), tool_records=[], repair_attempts=0, self_reported=1.0, settings=settings())
    assert report.band in (ConfidenceBand.MEDIUM, ConfidenceBand.LOW)
    assert report.needs_human_review is True


def test_non_answers_have_zero_confidence_and_need_review():
    for status in (AnswerStatus.INSUFFICIENT_CONTEXT, AnswerStatus.ERROR, AnswerStatus.RETRIEVAL_ONLY):
        report = compute_confidence(status=status, retrieval=retrieval(), citations=[], tool_records=[], repair_attempts=0, self_reported=None, settings=settings())
        assert report.confidence == 0.0 and report.band is ConfidenceBand.LOW and report.needs_human_review is True


def test_required_tools_missing_flags_uncalled_tools_by_question_pattern():
    policy = settings().tool_policy
    assert required_tools_missing("Which HCC category do the codes map to?", [], policy) == ["icd10_lookup"]
    assert required_tools_missing("Was member M-TEST-1002 eligible on the discharge date?", [], policy) == ["check_eligibility"]
    assert required_tools_missing("Can the visit be submitted for risk adjustment?", [], policy) == ["check_eligibility"]
    assert required_tools_missing("What was the most recent HbA1c?", [], policy) == []
    called = [ToolCallRecord(name="icd10_lookup", arguments={"query": "E11.22"}, result={"found": True})]
    assert required_tools_missing("Which HCC category do the codes map to?", called, policy) == []
    failed = [ToolCallRecord(name="icd10_lookup", arguments={}, error="timed out", timed_out=True)]
    assert required_tools_missing("Which HCC category do the codes map to?", failed, policy) == ["icd10_lookup"]


def test_missing_required_tool_caps_confidence_below_high_and_forces_review():
    clean = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(0.7, 18.0), citations=verified(2, 2), tool_records=[], repair_attempts=0, self_reported=1.0, settings=settings())
    assert clean.band is ConfidenceBand.HIGH
    flagged = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(0.7, 18.0), citations=verified(2, 2), tool_records=[], repair_attempts=0, self_reported=1.0, settings=settings(),
                                 missing_required_tools=["icd10_lookup"])
    assert flagged.band is not ConfidenceBand.HIGH
    assert flagged.needs_human_review is True
    assert flagged.factors["missing_required_tools"] == ["icd10_lookup"]


def test_self_report_is_clamped_and_weighted_least():
    a = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(1, 1), tool_records=[], repair_attempts=0, self_reported=5.0, settings=settings())
    b = compute_confidence(status=AnswerStatus.ANSWERED, retrieval=retrieval(), citations=verified(1, 1), tool_records=[], repair_attempts=0, self_reported=0.0, settings=settings())
    assert 0.0 <= a.confidence <= 1.0
    assert a.confidence - b.confidence <= 0.15  # the model's opinion moves the score only a little
