"""Text -> vector.

Two implementations behind one protocol:

* :class:`SentenceTransformerEmbedder` is the real one (local ``all-MiniLM-L6-v2`` by default).
  The model is loaded lazily and once per instance; one instance is shared per process.
* :class:`HashingEmbedder` is a real but deliberately weak bag-of-words embedder used in unit
  tests so they run in milliseconds without loading torch.  It is not a mock: it produces
  genuine unit vectors and genuine similarity orderings for lexical overlap.

Every embedder exposes ``name`` so a persisted index can refuse to load under a different
embedder (vectors from two models are not comparable), and ``count_tokens`` so the chunker
can size chunks in the *embedding model's* tokenizer, which is what enforces the truncation
limit.
"""

from __future__ import annotations

import logging
import re
import zlib
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...

    def count_tokens(self, text: str) -> int: ...


# --------------------------------------------------------------------------- hashing


class HashingEmbedder:
    """Bag-of-words hashed into ``dim`` buckets, L2-normalised. Deterministic across processes."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.name = f"hashing-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._TOKEN_RE.findall(text.lower()):
                out[row, zlib.crc32(tok.encode("utf-8")) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)  # leave all-zero rows as zeros, never NaN
        return out

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ------------------------------------------------------------ sentence-transformers


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers. CPU by default."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64) -> None:
        self.name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # loaded on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, keep local

            logger.info("loading embedding model %s on %s", self.name, self._device)
            self._model = SentenceTransformer(self.name, device=self._device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    @property
    def max_tokens(self) -> int:
        """Inputs longer than this (in word pieces) are silently truncated by the model."""
        return int(self._load().max_seq_length)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        tokenizer = self._load().tokenizer
        # verbose=False silences the "longer than the specified maximum" warning: exceeding the
        # limit is exactly what the chunker is checking for.
        return len(tokenizer(text, add_special_tokens=True, verbose=False)["input_ids"])
