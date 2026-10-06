"""Evaluation harness tests: eval-set loading, per-case scoring, aggregation, reporting,
and the retrieval-only run over the real corpus."""

import json

import pytest

from raapid_rag.config import PROJECT_ROOT, Settings
from raapid_rag.evaluation import (
    EvalCase,
    aggregate,
    load_eval_set,
    render_markdown,
    run_eval,
    score_case,
    score_retrieval_case,
)
from raapid_rag.ingestion.embedder import HashingEmbedder
from raapid_rag.ingestion.pipeline import build_index
from raapid_rag.retrieval.retriever import Retriever
from raapid_rag.schema import AnswerStatus, CitationOut, ConfidenceBand, GroundedAnswer, ToolCallRecord
from raapid_rag.service import QueryService
from raapid_rag.tools.registry import default_registry

EVAL_SET = PROJECT_ROOT / "eval" / "eval_set.json"


def make_answer(status="answered", answer="The most recent HbA1c was 8.2%, collected on 2025-03-12.",
                cited=(("note_001#c05", "note_001", True),), tools=(), llm_calls=1, retrieved_docs=("note_001", "guideline_002"),
                confidence=0.9, band="high", review=False) -> GroundedAnswer:
    return GroundedAnswer(
        query="q", answer=answer, answer_status=AnswerStatus(status),
        citations=[CitationOut(chunk_id=cid, evidence="e", doc_id=doc, doc_title="t", section="s", verified=ok) for cid, doc, ok in cited],
        tool_calls=[ToolCallRecord(name=n, arguments={}) for n in tools],
        confidence=confidence, confidence_band=ConfidenceBand(band), missing_information=[], needs_human_review=review,
        warnings=[], errors=[],
        trace={"llm_calls": llm_calls, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}, "est_cost_usd": 0.0002,
               "total_latency_ms": 1500.0, "retrieval": {"chunks": [{"doc_id": d, "chunk_id": f"{d}#c00"} for d in retrieved_docs]}},
    )


def case(**expected) -> EvalCase:
    return EvalCase(id="c", category="grounded_fact", query="q", expected=expected)


# ---------------------------------------------------------------- eval set


def test_eval_set_loads_with_unique_ids_and_required_fields():
    cases = load_eval_set(EVAL_SET)
    assert 5 <= len(cases) <= 12
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert all(c.query and c.expected.statuses for c in cases)
    categories = {c.category for c in cases}
    assert {"grounded_fact", "tool", "multi_doc", "refusal_canary", "refusal_missing", "gate_refusal"} <= categories


def test_eval_set_has_at_least_two_refusal_cases_and_one_gate_case():
    cases = load_eval_set(EVAL_SET)
    refusals = [c for c in cases if c.expected.statuses == ["insufficient_context"]]
    assert len(refusals) >= 2
    assert any(c.expected.max_llm_calls == 0 for c in refusals)


def test_load_eval_set_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([
        {"id": "a", "category": "x", "query": "q", "expected": {"statuses": ["answered"]}},
        {"id": "a", "category": "x", "query": "q", "expected": {"statuses": ["answered"]}},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(p)


# ------------------------------------------------------------------ scoring


def test_fully_correct_answer_passes_every_check():
    c = case(statuses=["answered"], cited_docs_any=["note_001"], answer_must_contain=["8.2", "2025-03-12"])
    result = score_case(c, make_answer())
    assert result.passed is True
    assert all(ch.passed for ch in result.checks)
    assert result.retrieval_hit is True
    assert result.cost_usd == 0.0002 and result.llm_calls == 1


def test_wrong_status_fails():
    result = score_case(case(statuses=["answered"]), make_answer(status="insufficient_context", answer=""))
    assert result.passed is False
    assert any(ch.name == "status" and not ch.passed for ch in result.checks)


def test_case_result_records_missing_information_and_tools_for_diagnosis():
    answer = make_answer(status="insufficient_context", answer="", cited=(), tools=("check_eligibility",))
    answer.missing_information = ["Coverage was inactive on the date of service."]
    result = score_case(case(statuses=["answered"]), answer)
    assert result.missing_information == ["Coverage was inactive on the date of service."]
    assert result.tools_called == ["check_eligibility"]
    md = render_markdown(aggregate([result]), [result], mode="full", model="fake")
    assert "Coverage was inactive" in md


def test_missing_expected_document_in_citations_fails():
    result = score_case(case(statuses=["answered"], cited_docs_any=["note_002"]), make_answer())
    assert result.passed is False
    failed = [ch for ch in result.checks if not ch.passed]
    assert failed and failed[0].name == "cited_docs_any" and "note_002" in failed[0].detail


def test_cited_docs_all_requires_every_document():
    ok = score_case(case(statuses=["answered"], cited_docs_all=["note_001", "guideline_002"]),
                    make_answer(cited=(("note_001#c05", "note_001", True), ("guideline_002#c03", "guideline_002", True))))
    assert ok.passed
    bad = score_case(case(statuses=["answered"], cited_docs_all=["note_001", "guideline_002"]), make_answer())
    assert not bad.passed


def test_unverified_citation_fails_evidence_check():
    result = score_case(case(statuses=["answered"]), make_answer(cited=(("note_001#c05", "note_001", False),)))
    assert any(ch.name == "evidence_verified" and not ch.passed for ch in result.checks)


def test_keyword_alternatives_and_forbidden_terms():
    c = case(statuses=["answered"], answer_must_contain=["worsened|increased", "8.2"], answer_must_not_contain=["improved"])
    assert score_case(c, make_answer(answer="Control worsened: 7.6% to 8.2%.")).passed
    assert not score_case(c, make_answer(answer="Control improved: 8.2%.")).passed
    assert not score_case(c, make_answer(answer="8.2% recorded.")).passed  # neither alternative present


def test_expected_tool_must_have_been_called():
    c = case(statuses=["answered"], tools_any=["icd10_lookup"])
    assert score_case(c, make_answer(tools=("icd10_lookup",))).passed
    assert not score_case(c, make_answer(tools=())).passed


def test_refusal_expectations_check_no_citations_and_call_budget():
    c = case(statuses=["insufficient_context"], no_citations=True, max_llm_calls=0)
    good = make_answer(status="insufficient_context", answer="", cited=(), llm_calls=0)
    assert score_case(c, good).passed
    leaked = make_answer(status="insufficient_context", answer="", cited=(("note_001#c05", "note_001", True),), llm_calls=0)
    assert not score_case(c, leaked).passed
    called = make_answer(status="insufficient_context", answer="", cited=(), llm_calls=1)
    assert not score_case(c, called).passed


def test_retrieval_hit_is_none_when_no_documents_expected():
    result = score_case(case(statuses=["insufficient_context"], no_citations=True), make_answer(status="insufficient_context", answer="", cited=()))
    assert result.retrieval_hit is None


def test_retrieval_mode_scoring_checks_gate_and_recall():
    expect_answer = case(statuses=["answered"], cited_docs_any=["note_001"])
    assert score_retrieval_case(expect_answer, make_answer(status="retrieval_only", answer="", cited=())).passed
    assert not score_retrieval_case(expect_answer, make_answer(status="retrieval_only", answer="", cited=(), retrieved_docs=("policy_001",))).passed
    expect_gate = case(statuses=["insufficient_context"], no_citations=True, max_llm_calls=0)
    assert score_retrieval_case(expect_gate, make_answer(status="insufficient_context", answer="", cited=(), llm_calls=0)).passed
    assert not score_retrieval_case(expect_gate, make_answer(status="retrieval_only", answer="", cited=())).passed


# -------------------------------------------------------------- aggregation


def test_aggregate_computes_pass_rate_calibration_and_refusal_metrics():
    cases = [
        EvalCase(id="a", category="grounded_fact", query="q", expected={"statuses": ["answered"]}),
        EvalCase(id="b", category="grounded_fact", query="q", expected={"statuses": ["answered"]}),
        EvalCase(id="c", category="refusal_canary", query="q", expected={"statuses": ["insufficient_context"], "no_citations": True}),
        EvalCase(id="d", category="refusal_missing", query="q", expected={"statuses": ["insufficient_context"], "no_citations": True}),
    ]
    answers = [
        make_answer(band="high"),
        make_answer(status="insufficient_context", answer="", cited=(), band="low", confidence=0.0),  # over-refusal: wrong
        make_answer(status="insufficient_context", answer="", cited=(), band="low", confidence=0.0),  # correct refusal
        make_answer(band="medium", confidence=0.6),  # under-refusal: answered when it should have refused
    ]
    results = [score_case(c, a) for c, a in zip(cases, answers)]
    summary = aggregate(results)
    assert summary.n == 4 and summary.passed == 2 and summary.pass_rate == 0.5
    assert summary.by_band["high"] == {"n": 1, "correct": 1}
    assert summary.by_band["medium"] == {"n": 1, "correct": 0}
    # refusals: 2 actual (b, c), 1 correct -> precision 0.5; 2 expected (c, d), 1 refused -> recall 0.5
    assert summary.refusal_precision == 0.5 and summary.refusal_recall == 0.5
    assert summary.total_cost_usd == pytest.approx(0.0008)
    assert summary.p95_latency_ms == 1500.0
    assert summary.by_category["grounded_fact"] == {"n": 2, "passed": 1}
    assert summary.failed_checks["status"] == 2


def test_markdown_report_contains_pass_rate_and_every_case():
    cases = [EvalCase(id="alpha", category="grounded_fact", query="q1", expected={"statuses": ["answered"]}),
             EvalCase(id="beta", category="refusal_canary", query="q2", expected={"statuses": ["insufficient_context"], "no_citations": True})]
    results = [score_case(cases[0], make_answer()), score_case(cases[1], make_answer(status="insufficient_context", answer="", cited=()))]
    md = render_markdown(aggregate(results), results, mode="full", model="fake", prompt_version="v9")
    assert "100%" in md or "1.00" in md
    assert "alpha" in md and "beta" in md
    assert "Refusal" in md and "band" in md.lower()
    assert "v9" in md


# -------------------------------------------------------- retrieval-only run


def test_retrieval_mode_run_over_real_corpus_reports_every_case():
    # Dense is switched off (hashing cosine is not on MiniLM's scale); BM25 keeps its real
    # threshold so the gate behaves as in production for lexical signals.
    s = Settings(_env_file=None, retrieval_min_dense_score=0.99)
    store, report = build_index(s, HashingEmbedder(), corpus_dir=PROJECT_ROOT / "data" / "corpus")
    service = QueryService(settings=s, retriever=Retriever(store, s, count_tokens=lambda t: len(t.split())),
                           tools=default_registry(), llm=None, system_prompt="x", index_fingerprint=report.fingerprint)
    cases = load_eval_set(EVAL_SET)
    results, summary = run_eval(service, cases, mode="retrieval")
    assert len(results) == len(cases) and summary.n == len(cases)
    assert summary.total_cost_usd == 0.0
    by_id = {r.case_id: r for r in results}
    assert by_id["gate_refusal"].passed is True  # gibberish/off-corpus query refused at the gate, no model call
    # Every case that expects citations should at least retrieve one of its documents with the real corpus.
    expected_docs = [c for c in cases if c.expected.cited_docs_any or c.expected.cited_docs_all]
    assert all(by_id[c.id].retrieval_hit for c in expected_docs)
