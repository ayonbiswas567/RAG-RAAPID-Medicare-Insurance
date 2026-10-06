"""QueryService tests: the full request path with a fake model and the real corpus."""

import json

import pytest
from helpers import build_test_store, make_test_service

from raapid_rag.guardrails import InvalidQueryError
from raapid_rag.retrieval.retriever import RetrievalResult
from raapid_rag.schema import AnswerStatus, ConfidenceBand, GroundedAnswer


@pytest.fixture(scope="module")
def store():
    return build_test_store()


make_service = make_test_service


def test_documents_lists_the_corpus_with_titles_and_chunk_counts(store):
    service, _ = make_service(store)
    docs = service.documents()
    assert len(docs) == 8
    by_id = {d["doc_id"]: d for d in docs}
    assert by_id["note_001"]["title"].startswith("Endocrinology")
    assert by_id["note_001"]["doc_type"] == "clinical_note" and by_id["note_001"]["n_chunks"] >= 8
    assert [d["doc_id"] for d in docs] == sorted(d["doc_id"] for d in docs)


def test_search_returns_retrieval_result_and_honours_scope(store):
    service, fake = make_service(store)
    result = service.search("telehealth", doc_ids=["policy_002"])
    assert isinstance(result, RetrievalResult)
    assert result.chunks and all(c.chunk.doc_id == "policy_002" for c in result.chunks)
    assert fake.calls == 0
    with pytest.raises(InvalidQueryError):
        service.search("   ")


def test_happy_path_returns_verified_answer_with_trace(store):
    service, fake = make_service(store)
    answer = service.ask("What was the most recent HbA1c for Jane Doe-Test?")
    assert isinstance(answer, GroundedAnswer)
    assert answer.answer_status is AnswerStatus.ANSWERED
    assert answer.citations and all(c.verified for c in answer.citations)
    assert answer.answer.startswith("Echo:")
    assert answer.confidence_band in (ConfidenceBand.HIGH, ConfidenceBand.MEDIUM)
    assert fake.calls == 1
    trace = answer.trace
    assert trace["prompt_version"] and trace["index_fingerprint"] == "fp-test"
    assert trace["usage"]["prompt_tokens"] == 500 and "est_cost_usd" in trace
    assert trace["retrieval"]["anchored_docs"] == ["note_001"]
    assert trace["llm_calls"] == 1 and trace["repair_attempts"] == 0
    GroundedAnswer.model_validate_json(answer.model_dump_json())  # round-trips


def test_gate_refusal_never_calls_the_model(store):
    service, fake = make_service(store)
    answer = service.ask("zxqv plorth wibble")
    assert answer.answer_status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert answer.needs_human_review is True and answer.confidence == 0.0
    assert answer.missing_information and "relevant" in answer.missing_information[0].lower()
    assert answer.trace["gate_reason"].startswith(("low_relevance", "empty_retrieval"))
    assert fake.calls == 0


def test_retrieval_only_mode_when_no_llm(store):
    service, _ = make_service(store, llm=False)
    answer = service.ask("audio-only telehealth encounters")
    assert answer.answer_status is AnswerStatus.RETRIEVAL_ONLY
    assert answer.answer == ""
    assert answer.citations and all(c.verified for c in answer.citations)
    assert any("not configured" in m for m in answer.missing_information)
    assert answer.needs_human_review is True


def test_provider_outage_becomes_error_envelope(store):
    service, _ = make_service(store, mode="raise")
    answer = service.ask("What was the most recent HbA1c?")
    assert answer.answer_status is AnswerStatus.ERROR
    assert any("unavailable" in e.lower() for e in answer.errors)
    assert answer.needs_human_review is True


def test_unrepairable_output_is_salvaged_to_partial_when_a_citation_survives(store):
    service, fake = make_service(store, mode="one_bad", max_repair_attempts=1)
    answer = service.ask("What was the most recent HbA1c?")
    assert fake.calls == 2  # first attempt + one repair
    assert answer.answer_status is AnswerStatus.PARTIAL
    assert len(answer.citations) == 1 and answer.citations[0].verified
    assert any("unverifiable citation" in w for w in answer.warnings)
    assert answer.needs_human_review is True


def test_unrepairable_output_without_valid_citation_is_an_error(store):
    service, _ = make_service(store, mode="all_bad", max_repair_attempts=1)
    answer = service.ask("What was the most recent HbA1c?")
    assert answer.answer_status is AnswerStatus.ERROR
    assert any("note_999#c99" in e for e in answer.errors)


def test_refusal_prose_is_blanked(store):
    service, _ = make_service(store, mode="refuse_with_text")
    answer = service.ask("What metformin dose is recommended for CKD stage 3?")
    assert answer.answer_status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert answer.answer == ""
    assert any("discarded" in w for w in answer.warnings)
    assert answer.missing_information == ["No dosing guidance in the context."]


def test_phi_pattern_is_flagged_but_answered_under_flag_policy(store):
    service, fake = make_service(store)
    answer = service.ask("Patient SSN 123-45-6789: what was the most recent HbA1c?")
    assert any("Social Security" in w for w in answer.warnings)
    assert answer.answer_status is AnswerStatus.ANSWERED and fake.calls == 1


def test_phi_pattern_is_blocked_under_block_policy(store):
    service, fake = make_service(store, phi_policy="block")
    answer = service.ask("Patient SSN 123-45-6789: what was the most recent HbA1c?")
    assert answer.answer_status is AnswerStatus.ERROR
    assert any("blocked" in e.lower() for e in answer.errors)
    assert fake.calls == 0


def test_invalid_query_raises(store):
    service, _ = make_service(store)
    with pytest.raises(InvalidQueryError):
        service.ask("   ")


def test_identical_question_is_served_from_cache(store):
    service, fake = make_service(store)
    first = service.ask("What was the most recent HbA1c for Jane Doe-Test?")
    second = service.ask("  what was the most recent hba1c for jane doe-test?  ")
    assert fake.calls == 1
    assert second.trace["cache_hit"] is True and first.trace.get("cache_hit") is False
    assert second.answer == first.answer


def test_cache_can_be_disabled(store):
    service, fake = make_service(store, answer_cache_size=0)
    service.ask("What was the most recent HbA1c?")
    service.ask("What was the most recent HbA1c?")
    assert fake.calls == 2


def test_doc_ids_scope_is_honoured_and_recorded(store):
    service, _ = make_service(store)
    answer = service.ask("telehealth", doc_ids=["policy_002"])
    assert answer.trace["retrieval"]["scope_doc_ids"] == ["policy_002"]
    assert all(c.doc_id == "policy_002" for c in answer.citations)


def test_caveats_from_the_model_reach_the_reviewer(store):
    service, _ = make_service(store, mode="with_caveat")
    answer = service.ask("What was the most recent HbA1c?")
    assert answer.answer_status is AnswerStatus.ANSWERED
    assert answer.caveats == ["Per the eligibility check, coverage was inactive on the date of service."]
    assert "caveats" in json.loads(answer.model_dump_json())


def test_answer_without_a_required_tool_is_flagged_and_capped(store):
    # EchoLLM never calls tools; an HCC-mapping question therefore violates the tool policy.
    service, _ = make_service(store)
    answer = service.ask("Which HCC category do Jane Doe-Test's diabetes codes map to?")
    assert answer.answer_status is AnswerStatus.ANSWERED
    assert answer.needs_human_review is True
    assert answer.confidence_band is not ConfidenceBand.HIGH
    assert any("icd10_lookup" in w and "not called" in w for w in answer.warnings)


def test_health_reports_index_and_model_state(store):
    service, _ = make_service(store)
    health = service.health()
    assert health["chunks"] == len(store) and health["index_fingerprint"] == "fp-test"
    assert health["llm_available"] is True and health["prompt_version"]
