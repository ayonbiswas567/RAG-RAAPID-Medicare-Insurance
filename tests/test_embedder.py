"""Embedder tests.

HashingEmbedder is a real (if weak) embedder used to keep the store/retriever tests fast.
The sentence-transformers embedder is exercised by one slow integration test.
"""

import numpy as np
import pytest

from raapid_rag.ingestion.embedder import HashingEmbedder, SentenceTransformerEmbedder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_hashing_embedder_is_deterministic_and_unit_norm():
    e = HashingEmbedder(dim=32)
    v1 = e.embed(["metformin dose", "something else entirely"])
    v2 = e.embed(["metformin dose", "something else entirely"])
    assert v1.shape == (2, 32)
    assert e.dim == 32
    assert np.allclose(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)


def test_hashing_embedder_scores_overlapping_text_higher():
    e = HashingEmbedder(dim=64)
    q, a, b = e.embed(["metformin dose", "Continue metformin 1000 mg twice daily.", "Ejection fraction was 30 percent."])
    assert cosine(q, a) > cosine(q, b)


def test_hashing_embedder_handles_empty_text_without_nan():
    e = HashingEmbedder(dim=16)
    v = e.embed([""])
    assert not np.isnan(v).any()


def test_hashing_embedder_counts_whitespace_tokens():
    assert HashingEmbedder().count_tokens("one two  three") == 3


@pytest.mark.slow
def test_sentence_transformer_embedder_loads_and_counts_word_pieces():
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert e.dim == 384
    assert e.max_tokens == 256
    # Word pieces: "HbA1c" alone splits into several pieces, so the count exceeds the word count.
    assert e.count_tokens("HbA1c 8.2% collected 2025-03-12") > 5
    v = e.embed(["HbA1c 8.2% collected 2025-03-12"])
    assert v.shape == (1, 384)
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-4
