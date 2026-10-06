"""In-memory hybrid store tests (dense cosine + BM25, fused with RRF)."""

import numpy as np
import pytest

from raapid_rag.ingestion.embedder import HashingEmbedder
from raapid_rag.ingestion.models import Chunk
from raapid_rag.ingestion.store import InMemoryHybridStore, ScoredChunk, tokenize_for_bm25


def make_chunk(cid: str, text: str, doc: str = "doc") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, doc_type="clinical_note", doc_title="Title", section="S", text=text, index=0)


def build(*texts: tuple[str, str], rrf_k: int = 60) -> InMemoryHybridStore:
    store = InMemoryHybridStore(HashingEmbedder(dim=64), rrf_k=rrf_k)
    store.add([make_chunk(cid, text) for cid, text in texts])
    store.build()
    return store


THREE = (
    ("A", "Continue metformin 1000 mg twice daily for diabetes."),
    ("B", "Ejection fraction was 30 percent on echocardiogram."),
    ("C", "Audio-only telehealth encounters are not acceptable."),
)


def test_tokenize_for_bm25_preserves_codes_and_identifiers():
    assert tokenize_for_bm25("Member M-TEST-1002 has E11.22 per RA-110.") == [
        "member", "m-test-1002", "has", "e11.22", "per", "ra-110",
    ]


def test_dense_search_ranks_overlapping_chunk_first():
    store = build(*THREE)
    results = store.search("metformin dose", k=3)
    assert results[0].chunk.chunk_id == "A"


def test_bm25_keeps_icd_codes_intact():
    store = build(
        ("A", "Diagnosis E11.22 documented today."),
        ("B", "Diagnosis E11.9 noted and item 22 reviewed."),
        ("C", "Unrelated follow-up instructions."),
    )
    scores = store.bm25_scores("E11.22")
    assert scores[0] > 0
    assert scores[1] == 0
    assert store.search("E11.22", k=3)[0].chunk.chunk_id == "A"


def test_fused_result_carries_both_ranks_and_first_in_both_scores_max():
    store = build(*THREE, rrf_k=60)
    top = store.search("metformin diabetes", k=3)[0]
    assert isinstance(top, ScoredChunk)
    assert top.chunk.chunk_id == "A"
    assert top.dense_rank == 1 and top.bm25_rank == 1
    assert abs(top.fused_score - 2 / 61) < 1e-9
    assert 0.0 <= top.dense_score <= 1.0
    assert top.bm25_score > 0


def test_no_keyword_overlap_gives_zero_bm25_and_no_bm25_rank():
    store = build(*THREE)
    results = store.search("zebra giraffe", k=3)
    assert all(r.bm25_score == 0 for r in results)
    assert all(r.bm25_rank is None for r in results)
    # Only the dense ranker votes, so fused == 1/(k + dense_rank).
    for r in results:
        assert abs(r.fused_score - 1 / (60 + r.dense_rank)) < 1e-9


def test_search_on_empty_store_returns_empty_list():
    store = InMemoryHybridStore(HashingEmbedder(dim=16))
    assert store.search("anything", k=5) == []
    assert len(store) == 0


def test_k_larger_than_store_returns_all_chunks():
    store = build(*THREE)
    assert len(store.search("anything", k=50)) == 3
    assert len(store) == 3


def test_add_after_build_reindexes_on_next_search():
    store = build(*THREE)
    store.add([make_chunk("D", "Spironolactone 12.5 mg daily was added.")])
    assert store.search("spironolactone", k=1)[0].chunk.chunk_id == "D"


def test_dense_scores_are_cosine_similarities_in_range():
    store = build(*THREE)
    scores = store.dense_scores("metformin")
    assert scores.shape == (3,)
    assert np.all(scores <= 1.0 + 1e-6) and np.all(scores >= -1.0 - 1e-6)


def test_search_can_be_scoped_to_document_ids():
    store = InMemoryHybridStore(HashingEmbedder(dim=64))
    store.add([make_chunk("A0", "metformin one", doc="A"), make_chunk("B0", "metformin two", doc="B"), make_chunk("B1", "metformin three", doc="B")])
    store.build()
    assert {r.chunk.chunk_id for r in store.search("metformin", k=5, doc_ids=["B"])} == {"B0", "B1"}
    assert store.search("metformin", k=5, doc_ids=[]) == []
    assert len(store.search("metformin", k=5, doc_ids=None)) == 3


def test_duplicate_chunk_id_is_rejected():
    store = build(*THREE)
    with pytest.raises(ValueError, match="A"):
        store.add([make_chunk("A", "duplicate id")])


def test_incremental_add_embeds_only_new_chunks():
    embedder = CountingEmbedder(dim=32)
    store = InMemoryHybridStore(embedder)
    store.add([make_chunk(cid, t) for cid, t in THREE])
    store.build()
    assert embedder.texts_embedded == 3
    store.add([make_chunk("D", "Spironolactone 12.5 mg daily was added.")])
    store.build()
    assert embedder.texts_embedded == 4  # not 7


def test_save_and_load_round_trip_preserves_search_results(tmp_path):
    store = build(*THREE)
    before = [(r.chunk.chunk_id, round(r.fused_score, 9)) for r in store.search("metformin", k=3)]
    store.save(tmp_path / "index")
    loaded = InMemoryHybridStore.load(tmp_path / "index", HashingEmbedder(dim=64))
    after = [(r.chunk.chunk_id, round(r.fused_score, 9)) for r in loaded.search("metformin", k=3)]
    assert before == after
    assert len(loaded) == 3
    assert loaded.search("metformin", k=1)[0].chunk.section == "S"


def test_load_rejects_index_built_with_a_different_embedder(tmp_path):
    build(*THREE).save(tmp_path / "index")
    with pytest.raises(ValueError, match="embedder"):
        InMemoryHybridStore.load(tmp_path / "index", HashingEmbedder(dim=32))


class CountingEmbedder(HashingEmbedder):
    """Real hashing embedder that also counts how many texts it was asked to embed."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.texts_embedded = 0

    def embed(self, texts):
        self.texts_embedded += len(texts)
        return super().embed(texts)
