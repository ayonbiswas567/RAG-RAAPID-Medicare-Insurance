"""Turn a query into a bounded, scored set of chunks plus a go/no-go signal.

The retriever depends only on the :class:`SearchIndex` protocol, so the storage backend can
change underneath it.  It is stateless apart from configuration and a small derived index of
document anchors, which is what lets many API workers run it concurrently.

Pipeline for one query:

1. **Ranked pool.** Hybrid search (dense + BM25, RRF-fused) over-fetches ``4 x top_k``
   candidates, optionally restricted to ``doc_ids`` (the production way to scope a question
   to one chart).
2. **Document anchoring** (:class:`DocumentAnchors`).  If the query contains an identifier
   that belongs to one document (patient name, member id, patient ref, doc id, policy or
   guideline number), that document is pulled in as a block: whole if it fits the anchor
   token budget, otherwise its best-ranked chunks.  Only identifiers anchor; topical words
   never do.  This is the "open the chart, then read the section" behaviour a coder has, and
   it fixes the multi-hop recall failure seen in the Step 4 live run, where the question
   named a patient but the patient's Assessment chunk did not rank in the top 15.
3. **Per-document cap** on the ranked fill so one long on-topic document cannot occupy every
   slot.
4. **Token budget.** Anchored chunks come first and are protected; ranked chunks are dropped
   lowest-first when the context budget is exceeded.
5. **Refusal gate** (:func:`evaluate_gate`) on *absolute* signals: best dense cosine and best
   raw BM25 over the pool.  RRF fused scores are never used for gating: RRF is a rank
   aggregator, so the top chunk always scores well even when nothing is relevant.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import Settings
from ..ingestion.models import Chunk
from ..ingestion.store import ScoredChunk, SearchIndex, tokenize_for_bm25

TokenCounter = Callable[[str], int]

IDENTIFIER_METADATA_KEYS = ("patient_ref", "member_id")  # doc_id is handled separately with an identifier check
NAME_LINE_RE = re.compile(r"\bName:\s*([^.\n]+?)\.", re.IGNORECASE)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk]
    dropped_chunk_ids: list[str] = field(default_factory=list)
    best_dense_score: float = 0.0
    best_bm25_score: float = 0.0
    passes_gate: bool = False
    gate_reason: str = "empty_retrieval"
    context_tokens: int = 0
    latency_ms: float = 0.0
    anchored_docs: list[str] = field(default_factory=list)
    anchor_scores: dict[str, int] = field(default_factory=dict)
    scope_doc_ids: list[str] | None = None

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.chunks]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "chunks": [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "doc_id": c.chunk.doc_id,
                    "doc_type": c.chunk.doc_type,
                    "doc_title": c.chunk.doc_title,
                    "section": c.chunk.section,
                    "date": c.chunk.date,
                    "text": c.chunk.text,
                    "dense_score": round(c.dense_score, 4),
                    "bm25_score": round(c.bm25_score, 3),
                    "fused_score": round(c.fused_score, 5),
                    "dense_rank": c.dense_rank,
                    "bm25_rank": c.bm25_rank,
                    "anchored": c.chunk.doc_id in self.anchored_docs,
                }
                for c in self.chunks
            ],
            **{k: v for k, v in asdict(self).items() if k not in ("query", "chunks")},
        }


def evaluate_gate(*, best_dense: float, best_bm25: float, n_chunks: int, settings: Settings) -> tuple[bool, str]:
    """Pure function: should the service proceed to the LLM with these retrieval signals?"""
    if n_chunks == 0:
        return False, "empty_retrieval"
    if best_dense >= settings.retrieval_min_dense_score or best_bm25 >= settings.retrieval_min_bm25_score:
        return True, "ok"
    return (
        False,
        f"low_relevance(best_dense={best_dense:.3f}<{settings.retrieval_min_dense_score}, "
        f"best_bm25={best_bm25:.2f}<{settings.retrieval_min_bm25_score})",
    )


class DocumentAnchors:
    """Identifier -> document map built from chunk metadata and the opening chunks.

    Anchor keys per document: identifier-like metadata values (doc id, patient ref, member id),
    identifier tokens found in any metadata value (``ra-110``, ``cg-01``), and the patient
    name from a ``Name: ...`` line in the first two chunks (full name plus surname).
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        self._keys: dict[str, set[str]] = {}
        for doc_id, doc_chunks in by_doc.items():
            doc_chunks.sort(key=lambda c: c.index)
            self._keys[doc_id] = self._keys_for(doc_id, doc_chunks)

    @staticmethod
    def _keys_for(doc_id: str, doc_chunks: list[Chunk]) -> set[str]:
        # A doc id anchors only when it looks like an identifier (note_001, policy_002), never
        # when it happens to be an ordinary word.
        keys: set[str] = {doc_id.casefold()} if _is_identifier(doc_id.casefold()) else set()
        meta = doc_chunks[0].metadata if doc_chunks else {}
        for key in IDENTIFIER_METADATA_KEYS:
            value = meta.get(key)
            if value:
                keys.add(value.casefold())
        for value in meta.values():
            keys.update(t for t in tokenize_for_bm25(value) if _is_identifier(t))
        for chunk in doc_chunks[:2]:
            for match in NAME_LINE_RE.finditer(chunk.text):
                full = " ".join(tokenize_for_bm25(match.group(1)))
                if full:
                    keys.add(full)
                    surname = full.split()[-1]
                    if len(surname) >= 4:
                        keys.add(surname)
        return keys

    def match(self, query: str, max_docs: int) -> tuple[list[str], dict[str, int]]:
        q_tokens = set(tokenize_for_bm25(query))
        q_text = " ".join(tokenize_for_bm25(query))
        scores: dict[str, int] = {}
        for doc_id, keys in self._keys.items():
            hits = sum(1 for k in keys if (k in q_tokens) or (" " in k and k in q_text))
            if hits:
                scores[doc_id] = hits
        ranked = sorted(scores, key=lambda d: (-scores[d], list(self._keys).index(d)))[:max_docs]
        return ranked, {d: scores[d] for d in ranked}


def _is_identifier(token: str) -> bool:
    """A token with a digit and a letter/hyphen/underscore: m-test-1002, ra-110, cg-01, note_001."""
    return any(ch.isdigit() for ch in token) and any(ch.isalpha() or ch in "-_" for ch in token)


class Retriever:
    def __init__(self, index: SearchIndex, settings: Settings, count_tokens: TokenCounter | None = None) -> None:
        self._index = index
        self._settings = settings
        self._count = count_tokens or _tiktoken_counter()
        self._anchors: DocumentAnchors | None = None
        self._anchors_size = -1

    @property
    def index_size(self) -> int:
        return len(self._index)

    def documents(self) -> list[dict[str, Any]]:
        """One row per indexed document (for scope pickers and the /documents endpoint)."""
        rows: dict[str, dict[str, Any]] = {}
        for c in self._index.chunks:
            row = rows.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title, "doc_type": c.doc_type, "date": c.date, "n_chunks": 0})
            row["n_chunks"] += 1
        return [rows[k] for k in sorted(rows)]

    def retrieve(self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        k = top_k or self._settings.retrieval_top_k
        pool = self._index.search(query, k=k * 4, doc_ids=doc_ids)

        best_dense = max((c.dense_score for c in pool), default=0.0)
        best_bm25 = max((c.bm25_score for c in pool), default=0.0)

        anchored_docs: list[str] = []
        anchor_scores: dict[str, int] = {}
        anchored: list[ScoredChunk] = []
        if self._settings.retrieval_enable_document_anchoring and doc_ids is None and len(self._index):
            anchored_docs, anchor_scores = self._document_anchors().match(query, self._settings.retrieval_max_anchored_docs)
            for doc_id in anchored_docs:
                anchored.extend(self._expand_document(query, doc_id))

        # Anchored documents are fully represented already; the ranked fill only adds others.
        ranked = self._apply_document_cap([c for c in pool if c.chunk.doc_id not in anchored_docs], k)
        scored = anchored + ranked

        kept, dropped, used = self._apply_budget(scored)
        passes, reason = evaluate_gate(best_dense=best_dense, best_bm25=best_bm25, n_chunks=len(scored), settings=self._settings)
        if passes and not kept:
            passes, reason = False, "context_budget_too_small"

        return RetrievalResult(
            query=query,
            chunks=kept,
            dropped_chunk_ids=dropped,
            best_dense_score=best_dense,
            best_bm25_score=best_bm25,
            passes_gate=passes,
            gate_reason=reason,
            context_tokens=used,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            anchored_docs=anchored_docs,
            anchor_scores=anchor_scores,
            scope_doc_ids=list(doc_ids) if doc_ids is not None else None,
        )

    # ------------------------------------------------------------------ helpers

    def _document_anchors(self) -> DocumentAnchors:
        if self._anchors is None or self._anchors_size != len(self._index):
            self._anchors = DocumentAnchors(self._index.chunks)
            self._anchors_size = len(self._index)
        return self._anchors

    def _expand_document(self, query: str, doc_id: str) -> list[ScoredChunk]:
        """All chunks of an anchored document (document order) if short enough, else its best few."""
        scored_all = self._index.search(query, k=len(self._index), doc_ids=[doc_id])
        total = sum(self._count(c.chunk.text) for c in scored_all)
        if total <= self._settings.retrieval_anchor_doc_token_budget:
            return sorted(scored_all, key=lambda c: c.chunk.index)
        return scored_all[: self._settings.retrieval_max_chunks_per_doc]

    def _apply_document_cap(self, pool: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Take the top-k in fused order but allow at most N chunks from any one document."""
        cap = self._settings.retrieval_max_chunks_per_doc
        per_doc: dict[str, int] = {}
        selected: list[ScoredChunk] = []
        for c in pool:
            if per_doc.get(c.chunk.doc_id, 0) >= cap:
                continue
            per_doc[c.chunk.doc_id] = per_doc.get(c.chunk.doc_id, 0) + 1
            selected.append(c)
            if len(selected) == k:
                break
        return selected

    def _apply_budget(self, scored: list[ScoredChunk]) -> tuple[list[ScoredChunk], list[str], int]:
        """Keep chunks in order until the token budget is exhausted; drop the rest."""
        budget = self._settings.retrieval_context_token_budget
        kept: list[ScoredChunk] = []
        dropped: list[str] = []
        used = 0
        for c in scored:
            n = self._count(c.chunk.text)
            if used + n > budget:
                dropped.append(c.chunk.chunk_id)
                continue
            kept.append(c)
            used += n
        return kept, dropped, used


def _tiktoken_counter() -> TokenCounter:
    """LLM-side token counter (what the model bills), distinct from the embedder's tokenizer."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))
