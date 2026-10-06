"""Retriever tests: top-k, token budget, and the refusal gate."""

from raapid_rag.config import Settings
from raapid_rag.ingestion.embedder import HashingEmbedder
from raapid_rag.ingestion.models import Chunk
from raapid_rag.ingestion.store import InMemoryHybridStore
from raapid_rag.retrieval.retriever import Retriever, evaluate_gate


def count(text: str) -> int:
    return len(text.split())


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_store(*texts: tuple[str, str]) -> InMemoryHybridStore:
    store = InMemoryHybridStore(HashingEmbedder(dim=64))
    store.add([
        Chunk(chunk_id=cid, doc_id="doc", doc_type="clinical_note", doc_title="Title", section="S", text=t, index=i)
        for i, (cid, t) in enumerate(texts)
    ])
    store.build()
    return store


FIVE = (
    ("A", "Continue metformin 1000 mg twice daily for diabetes control."),
    ("B", "Metformin was well tolerated with no hypoglycemia."),
    ("C", "Ejection fraction was 30 percent on echocardiogram."),
    ("D", "Audio-only telehealth encounters are not acceptable for submission."),
    ("E", "Diagnosis E11.22 documented with supporting labs."),
)


def test_retrieve_respects_top_k():
    r = Retriever(make_store(*FIVE), settings(retrieval_top_k=2), count_tokens=count)
    result = r.retrieve("metformin")
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk.chunk_id in {"A", "B"}


def test_token_budget_drops_lowest_scored_chunks_and_records_them():
    r = Retriever(
        make_store(*FIVE),
        settings(retrieval_top_k=5, retrieval_context_token_budget=20, retrieval_max_chunks_per_doc=5),
        count_tokens=count,
    )
    result = r.retrieve("metformin")
    kept = [c.chunk.chunk_id for c in result.chunks]
    assert 0 < len(kept) < 5
    assert result.context_tokens <= 20
    assert set(kept) | set(result.dropped_chunk_ids) == {"A", "B", "C", "D", "E"}
    # Kept chunks are the best-scored ones (fused order is preserved).
    fused = [c.fused_score for c in result.chunks]
    assert fused == sorted(fused, reverse=True)


def test_gate_passes_when_dense_score_meets_threshold():
    r = Retriever(make_store(*FIVE), settings(retrieval_min_dense_score=0.1, retrieval_min_bm25_score=99.0), count_tokens=count)
    result = r.retrieve("metformin diabetes")
    assert result.passes_gate is True
    assert result.gate_reason == "ok"
    assert result.best_dense_score >= 0.1


def test_gate_refuses_when_both_signals_are_weak():
    r = Retriever(make_store(*FIVE), settings(retrieval_min_dense_score=0.99, retrieval_min_bm25_score=1.0), count_tokens=count)
    result = r.retrieve("zebra giraffe")
    assert result.passes_gate is False
    assert "low_relevance" in result.gate_reason
    assert result.best_bm25_score == 0


def test_gate_passes_on_strong_keyword_match_even_if_dense_is_weak():
    r = Retriever(make_store(*FIVE), settings(retrieval_min_dense_score=0.99, retrieval_min_bm25_score=0.1), count_tokens=count)
    result = r.retrieve("E11.22")
    assert result.passes_gate is True
    assert result.best_bm25_score > 0


def test_empty_store_gives_empty_result_and_empty_retrieval_reason():
    r = Retriever(make_store(), settings(), count_tokens=count)
    result = r.retrieve("anything")
    assert result.chunks == []
    assert result.passes_gate is False
    assert result.gate_reason == "empty_retrieval"


def test_evaluate_gate_is_a_pure_function():
    s = settings(retrieval_min_dense_score=0.3, retrieval_min_bm25_score=3.0)
    assert evaluate_gate(best_dense=0.5, best_bm25=0.0, n_chunks=1, settings=s) == (True, "ok")
    assert evaluate_gate(best_dense=0.1, best_bm25=5.0, n_chunks=1, settings=s) == (True, "ok")
    ok, reason = evaluate_gate(best_dense=0.1, best_bm25=1.0, n_chunks=1, settings=s)
    assert ok is False and reason.startswith("low_relevance")
    assert evaluate_gate(best_dense=0.0, best_bm25=0.0, n_chunks=0, settings=s) == (False, "empty_retrieval")


def test_per_document_cap_keeps_room_for_other_documents():
    # Five strongly matching chunks from doc A would normally fill top_k=3; the cap forces
    # room for the weaker but distinct doc B so multi-document questions see both sources.
    store = InMemoryHybridStore(HashingEmbedder(dim=64))
    store.add([
        Chunk(chunk_id=f"A{i}", doc_id="A", doc_type="coding_guideline", doc_title="Guide", section="S", text=f"diabetes HCC category hierarchy diabetes {i}", index=i)
        for i in range(5)
    ] + [Chunk(chunk_id="B0", doc_id="B", doc_type="clinical_note", doc_title="Note", section="S", text="diabetes assessment for the patient", index=0)])
    store.build()
    r = Retriever(store, settings(retrieval_top_k=3, retrieval_max_chunks_per_doc=2), count_tokens=count)
    ids = r.retrieve("diabetes HCC category hierarchy").chunk_ids
    assert len(ids) == 3
    assert sum(i.startswith("A") for i in ids) == 2
    assert "B0" in ids


def test_per_document_cap_does_not_pad_when_only_one_document_matches():
    store = make_store(("A0", "metformin one"), ("A1", "metformin two"), ("A2", "metformin three"))
    r = Retriever(store, settings(retrieval_top_k=3, retrieval_max_chunks_per_doc=2), count_tokens=count)
    # All chunks share doc_id "doc" in make_store, so the cap limits the result to 2.
    assert len(r.retrieve("metformin").chunks) == 2


# ------------------------------------------------------------ document anchoring


def anchored_store() -> InMemoryHybridStore:
    """Three documents: a guideline that dominates HCC queries, a patient note whose name
    appears only in its first chunk, and an unrelated policy."""
    store = InMemoryHybridStore(HashingEmbedder(dim=64))
    guide_meta = {"doc_id": "guide", "doc_type": "coding_guideline", "title": "HCC Hierarchy Reference"}
    note_meta = {"doc_id": "note_001", "doc_type": "clinical_note", "title": "Endocrinology Progress Note",
                 "patient_ref": "TEST-0001", "member_id": "M-TEST-1001", "date_of_service": "2025-03-14"}
    policy_meta = {"doc_id": "policy", "doc_type": "payer_policy", "title": "Telehealth Policy RA-110"}
    chunks = [
        Chunk(chunk_id=f"guide#c0{i}", doc_id="guide", doc_type="coding_guideline", doc_title="HCC Hierarchy Reference",
              section=f"S{i}", text=f"HCC category hierarchy for diabetes codes, rule {i}.", index=i, metadata=guide_meta)
        for i in range(5)
    ] + [
        Chunk(chunk_id="note_001#c00", doc_id="note_001", doc_type="clinical_note", doc_title="Endocrinology Progress Note",
              section="Patient Information", text="Name: Jane Doe-Test. MRN: TEST-0001. Office visit.", index=0, metadata=note_meta),
        Chunk(chunk_id="note_001#c01", doc_id="note_001", doc_type="clinical_note", doc_title="Endocrinology Progress Note",
              section="Assessment", text="Type 2 diabetes mellitus with diabetic chronic kidney disease (E11.22).", index=1, metadata=note_meta),
        Chunk(chunk_id="policy#c00", doc_id="policy", doc_type="payer_policy", doc_title="Telehealth Policy RA-110",
              section="1", text="Audio-only telehealth encounters are not acceptable.", index=0, metadata=policy_meta),
        Chunk(chunk_id="policy#c01", doc_id="policy", doc_type="payer_policy", doc_title="Telehealth Policy RA-110",
              section="2", text="Eligibility must be verified on the date of service.", index=1, metadata=policy_meta),
    ]
    store.add(chunks)
    store.build()
    return store


def test_patient_name_anchors_the_document_and_includes_it_as_a_block():
    r = Retriever(anchored_store(), settings(retrieval_top_k=3), count_tokens=count)
    result = r.retrieve("Which HCC category do the diabetes codes documented for Jane Doe-Test map to?")
    assert result.anchored_docs == ["note_001"]
    ids = result.chunk_ids
    # The Assessment chunk shares almost no words with the query, yet it is included because
    # the query names the patient whose document it belongs to.
    assert "note_001#c01" in ids and "note_001#c00" in ids
    assert any(i.startswith("guide#") for i in ids)  # ranked results still fill the remaining slots
    assert ids[:2] == ["note_001#c00", "note_001#c01"]  # anchored block first, in document order


def test_topical_words_alone_never_anchor_a_document():
    r = Retriever(anchored_store(), settings(retrieval_top_k=3), count_tokens=count)
    assert r.retrieve("HCC category hierarchy for diabetes codes").anchored_docs == []
    assert r.retrieve("telehealth policy rules for audio-only visits").anchored_docs == []


def test_member_id_surname_and_policy_number_anchor_documents():
    r = Retriever(anchored_store(), settings(retrieval_top_k=3), count_tokens=count)
    assert r.retrieve("Is member M-TEST-1001 covered?").anchored_docs == ["note_001"]
    assert r.retrieve("What does RA-110 say about audio-only visits?").anchored_docs == ["policy"]
    assert r.retrieve("Summarise Ms. Doe-Test's assessment").anchored_docs == ["note_001"]
    both = r.retrieve("Is member M-TEST-1001 eligible under RA-110?")
    assert both.anchored_docs == ["note_001", "policy"]
    assert both.anchor_scores == {"note_001": 1, "policy": 1}


def test_anchoring_can_be_disabled():
    r = Retriever(anchored_store(), settings(retrieval_top_k=3, retrieval_enable_document_anchoring=False), count_tokens=count)
    result = r.retrieve("Which HCC category do the diabetes codes documented for Jane Doe-Test map to?")
    assert result.anchored_docs == []
    assert "note_001#c01" not in result.chunk_ids


def test_anchored_document_over_its_token_budget_falls_back_to_its_best_chunks():
    r = Retriever(
        anchored_store(),
        settings(retrieval_top_k=3, retrieval_anchor_doc_token_budget=8, retrieval_max_chunks_per_doc=1),
        count_tokens=count,
    )
    result = r.retrieve("Jane Doe-Test diabetes codes")
    assert result.anchored_docs == ["note_001"]
    assert sum(i.startswith("note_001#") for i in result.chunk_ids) == 1


def test_explicit_doc_ids_scope_restricts_results():
    r = Retriever(anchored_store(), settings(retrieval_top_k=5), count_tokens=count)
    result = r.retrieve("diabetes", doc_ids=["note_001"])
    assert result.chunk_ids and all(i.startswith("note_001#") for i in result.chunk_ids)
    assert result.scope_doc_ids == ["note_001"]


def test_retrieval_result_is_json_serialisable():
    r = Retriever(make_store(*FIVE), settings(retrieval_top_k=2), count_tokens=count)
    payload = r.retrieve("metformin").to_dict()
    assert payload["query"] == "metformin"
    assert len(payload["chunks"]) == 2
    assert {"chunk_id", "section", "dense_score", "bm25_score", "fused_score"} <= set(payload["chunks"][0])
