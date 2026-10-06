"""In-memory hybrid index: dense cosine + BM25, fused with Reciprocal Rank Fusion.

Why hybrid (decision D4): dense embeddings capture paraphrase but blur exact identifiers
(``E11.22`` vs ``E11.21``); BM25 nails identifiers but misses paraphrase.  RRF merges the
two rankings using only ranks, so the incomparable score scales never have to be normalised.

Scaling seams (kept deliberately explicit):

* :class:`SearchIndex` is the protocol the retriever depends on.  This class is one
  implementation; FAISS, pgvector or a hosted vector database can replace it without touching
  the retriever or the agent.
* ``build()`` embeds only chunks that do not yet have vectors, so ingestion is incremental.
  BM25 (``rank-bm25``) has no incremental API and is rebuilt in full; at a few hundred
  thousand chunks that is seconds and belongs in an ingestion worker, not a request.
* ``save()``/``load()`` persist the vectors and chunk metadata.  Many API workers load one
  prebuilt index instead of each spending CPU re-embedding the corpus at startup.
* Brute-force cosine is O(N) per query: a single matrix-vector product, milliseconds for
  100k chunks, a real cost past a few million.  That is the point at which an ANN index
  replaces this class behind the same protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .models import Chunk

# Keeps "e11.22", "m-test-1002", "ra-110", "2025-03-12", "note_001" as single tokens.
BM25_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")


def tokenize_for_bm25(text: str) -> list[str]:
    return BM25_TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    dense_score: float  # cosine similarity, [-1, 1]
    bm25_score: float  # raw BM25, >= 0, unbounded
    fused_score: float  # RRF, (0, 2/(k+1)]
    dense_rank: int  # 1-based
    bm25_rank: int | None  # None when the chunk shares no keyword with the query


class SearchIndex(Protocol):
    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[ScoredChunk]: ...

    @property
    def chunks(self) -> list[Chunk]: ...

    def __len__(self) -> int: ...


class InMemoryHybridStore:
    def __init__(self, embedder: Embedder, rrf_k: int = 60) -> None:
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: BM25Okapi | None = None
        self._bm25_stale = True

    # ------------------------------------------------------------------ mutation

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self._ids:
                raise ValueError(f"duplicate chunk_id '{chunk.chunk_id}'")
            self._ids.add(chunk.chunk_id)
            self._chunks.append(chunk)
        self._bm25_stale = True

    def build(self) -> None:
        """Embed any not-yet-embedded chunks and (re)build the BM25 index."""
        pending = self._chunks[len(self._matrix) :]
        if pending:
            new = self._embedder.embed([c.embed_text for c in pending])
            self._matrix = np.vstack([self._matrix, new]) if len(self._matrix) else new
        if self._bm25_stale:
            corpus = [tokenize_for_bm25(c.embed_text) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_stale = False

    def _ensure_built(self) -> None:
        if self._bm25_stale or len(self._matrix) != len(self._chunks):
            self.build()

    # ------------------------------------------------------------------- scoring

    def dense_scores(self, query: str) -> np.ndarray:
        self._ensure_built()
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        q = self._embedder.embed([query])[0]
        return self._matrix @ q  # both sides unit-norm, so this is cosine similarity

    def bm25_scores(self, query: str) -> np.ndarray:
        self._ensure_built()
        tokens = tokenize_for_bm25(query)
        if not self._chunks or not tokens or self._bm25 is None:
            return np.zeros(len(self._chunks), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[ScoredChunk]:
        """Top-k by fused score. ``doc_ids`` restricts results to those documents (metadata filter)."""
        if not self._chunks:
            return []
        if doc_ids is not None:
            allowed = set(doc_ids)
            if not allowed:
                return []
        dense = self.dense_scores(query)
        bm25 = self.bm25_scores(query)
        n = len(self._chunks)

        dense_rank = np.empty(n, dtype=np.int64)
        dense_rank[np.argsort(-dense, kind="stable")] = np.arange(1, n + 1)

        # Only chunks with some keyword overlap receive a lexical vote.
        bm25_rank = np.zeros(n, dtype=np.int64)  # 0 == no rank
        positive = np.flatnonzero(bm25 > 0)
        if positive.size:
            order = positive[np.argsort(-bm25[positive], kind="stable")]
            bm25_rank[order] = np.arange(1, order.size + 1)

        fused = 1.0 / (self._rrf_k + dense_rank)
        has_lex = bm25_rank > 0
        fused[has_lex] += 1.0 / (self._rrf_k + bm25_rank[has_lex])

        if doc_ids is not None:
            excluded = np.array([c.doc_id not in allowed for c in self._chunks])
            fused[excluded] = -np.inf

        top = [i for i in np.argsort(-fused, kind="stable")[: max(0, k)] if fused[i] > -np.inf]
        return [
            ScoredChunk(
                chunk=self._chunks[i],
                dense_score=float(dense[i]),
                bm25_score=float(bm25[i]),
                fused_score=float(fused[i]),
                dense_rank=int(dense_rank[i]),
                bm25_rank=int(bm25_rank[i]) if bm25_rank[i] else None,
            )
            for i in top
        ]

    # --------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        self._ensure_built()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._matrix)
        meta = {"embedder": self._embedder.name, "dim": self._embedder.dim, "rrf_k": self._rrf_k, "n": len(self._chunks)}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (path / "chunks.json").write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> InMemoryHybridStore:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"index at {path} was built with embedder '{meta['embedder']}', "
                f"cannot load with '{embedder.name}'"
            )
        store = cls(embedder, rrf_k=int(meta["rrf_k"]))
        store.add([_chunk_from_dict(d) for d in json.loads((path / "chunks.json").read_text(encoding="utf-8"))])
        store._matrix = np.load(path / "vectors.npy").astype(np.float32)
        if len(store._matrix) != len(store._chunks):
            raise ValueError(f"index at {path} is corrupt: {len(store._matrix)} vectors for {len(store._chunks)} chunks")
        store.build()  # embeds nothing (all vectors present); rebuilds BM25
        return store

    # --------------------------------------------------------------------- misc

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_type": c.doc_type, "doc_title": c.doc_title,
        "section": c.section, "text": c.text, "index": c.index, "metadata": c.metadata,
    }


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(**d)
