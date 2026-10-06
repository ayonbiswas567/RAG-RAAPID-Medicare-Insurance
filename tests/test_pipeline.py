"""Ingestion pipeline tests: corpus -> chunks -> index, with a fingerprint-keyed disk cache."""

from raapid_rag.config import PROJECT_ROOT, Settings
from raapid_rag.ingestion.embedder import HashingEmbedder
from raapid_rag.ingestion.pipeline import build_index, corpus_fingerprint
from raapid_rag.ingestion.loader import load_corpus


DOC_A = "---\ndoc_id: a\ndoc_type: clinical_note\ntitle: A\n---\n## S1\nMetformin continued.\n\n## S2\nEgfr stable.\n"
DOC_B = "---\ndoc_id: b\ndoc_type: payer_policy\ntitle: B\n---\n## 1\nAudio-only is not acceptable.\n"


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.texts_embedded = 0

    def embed(self, texts):
        self.texts_embedded += len(texts)
        return super().embed(texts)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def write_corpus(path):
    path.mkdir(exist_ok=True)
    (path / "a.md").write_text(DOC_A, encoding="utf-8")
    (path / "b.md").write_text(DOC_B, encoding="utf-8")
    return path


def test_build_index_chunks_every_document(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    store, report = build_index(settings(), CountingEmbedder(), corpus_dir=corpus)
    assert len(store) == 3
    assert report.n_documents == 2
    assert report.n_chunks == 3
    assert report.loaded_from_cache is False
    assert report.chunks_per_doc == {"a": 2, "b": 1}
    assert store.search("audio-only", k=1)[0].chunk.doc_id == "b"


def test_second_build_with_cache_dir_loads_instead_of_embedding(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    cache = tmp_path / "cache"
    first = CountingEmbedder()
    _, report1 = build_index(settings(), first, corpus_dir=corpus, cache_dir=cache)
    assert report1.loaded_from_cache is False and first.texts_embedded == 3

    second = CountingEmbedder()
    store2, report2 = build_index(settings(), second, corpus_dir=corpus, cache_dir=cache)
    assert report2.loaded_from_cache is True
    assert second.texts_embedded == 0
    assert report2.fingerprint == report1.fingerprint
    assert len(store2) == 3


def test_changing_a_document_invalidates_the_cache(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    cache = tmp_path / "cache"
    _, report1 = build_index(settings(), CountingEmbedder(), corpus_dir=corpus, cache_dir=cache)
    (corpus / "b.md").write_text(DOC_B.replace("not acceptable", "acceptable"), encoding="utf-8")
    emb = CountingEmbedder()
    _, report2 = build_index(settings(), emb, corpus_dir=corpus, cache_dir=cache)
    assert report2.loaded_from_cache is False
    assert report2.fingerprint != report1.fingerprint
    assert emb.texts_embedded == 3


def test_fingerprint_depends_on_chunking_parameters_and_embedder():
    docs = load_corpus(PROJECT_ROOT / "data" / "corpus")
    base = corpus_fingerprint(docs, settings(), "hashing-32")
    assert base == corpus_fingerprint(docs, settings(), "hashing-32")  # stable
    assert base != corpus_fingerprint(docs, settings(chunk_target_tokens=100), "hashing-32")
    assert base != corpus_fingerprint(docs, settings(), "sentence-transformers/all-MiniLM-L6-v2")


def test_real_corpus_builds_and_windows_the_long_section():
    store, report = build_index(settings(), HashingEmbedder(), corpus_dir=PROJECT_ROOT / "data" / "corpus")
    assert report.n_documents == 8
    # 52 sections plus 8 preamble banners, plus extra windows for the Hospital Course section.
    assert report.n_chunks > 60
    hospital_course = [c for c in store.chunks if c.doc_id == "note_002" and c.section.startswith("Hospital Course")]
    assert len(hospital_course) >= 2
    assert all("(part " in c.section for c in hospital_course)
    assert report.max_chunk_tokens <= 220
