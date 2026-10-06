"""Corpus -> chunks -> index, with a fingerprint-keyed on-disk cache.

This is the *write path*.  In production it runs in an ingestion worker on a schedule or on
document arrival, never inside a user request.  Query workers only ever call
:meth:`InMemoryHybridStore.load` on the artefact this module produces.

The cache key is a fingerprint over everything that changes the index: document ids,
bodies and metadata, the chunking parameters, the RRF constant and the embedder name.  Any
change produces a new fingerprint and therefore a new directory, so a stale index is never
served by accident and the previous one is untouched (atomic switch-over by name).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Settings
from .chunker import chunk_document
from .embedder import Embedder
from .loader import load_corpus
from .models import Chunk, Document
from .store import InMemoryHybridStore

logger = logging.getLogger(__name__)

FINGERPRINT_SCHEMA_VERSION = "1"  # bump when the chunker or index format changes incompatibly


@dataclass
class IndexBuildReport:
    n_documents: int
    n_chunks: int
    fingerprint: str
    loaded_from_cache: bool
    build_seconds: float
    chunks_per_doc: dict[str, int]
    max_chunk_tokens: int
    embedder: str

    def to_dict(self) -> dict:
        return asdict(self)


def corpus_fingerprint(docs: list[Document], settings: Settings, embedder_name: str) -> str:
    h = hashlib.sha256()
    h.update(
        f"{FINGERPRINT_SCHEMA_VERSION}|{embedder_name}|{settings.chunk_target_tokens}|"
        f"{settings.chunk_overlap_tokens}|{settings.rrf_k}".encode("utf-8")
    )
    for d in sorted(docs, key=lambda d: d.doc_id):
        h.update(d.doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(d.body.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(d.metadata, sort_keys=True).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_index(
    settings: Settings,
    embedder: Embedder,
    corpus_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[InMemoryHybridStore, IndexBuildReport]:
    t0 = time.perf_counter()
    docs = load_corpus(Path(corpus_dir or settings.corpus_dir))
    fingerprint = corpus_fingerprint(docs, settings, embedder.name)

    if cache_dir is not None:
        cached = Path(cache_dir) / fingerprint
        if (cached / "meta.json").exists():
            try:
                store = InMemoryHybridStore.load(cached, embedder)
                logger.info("loaded index %s from cache (%d chunks)", fingerprint, len(store))
                return store, _report(docs, store.chunks, fingerprint, True, t0, embedder)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                logger.warning("cached index %s unusable (%s); rebuilding", fingerprint, exc)

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                count_tokens=embedder.count_tokens,
            )
        )

    store = InMemoryHybridStore(embedder, rrf_k=settings.rrf_k)
    store.add(chunks)
    store.build()

    if cache_dir is not None:
        store.save(Path(cache_dir) / fingerprint)
        logger.info("saved index %s to %s", fingerprint, cache_dir)

    return store, _report(docs, chunks, fingerprint, False, t0, embedder)


def _report(
    docs: list[Document], chunks: list[Chunk], fingerprint: str, cached: bool, t0: float, embedder: Embedder
) -> IndexBuildReport:
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    return IndexBuildReport(
        n_documents=len(docs),
        n_chunks=len(chunks),
        fingerprint=fingerprint,
        loaded_from_cache=cached,
        build_seconds=round(time.perf_counter() - t0, 2),
        chunks_per_doc=per_doc,
        max_chunk_tokens=max((embedder.count_tokens(c.text) for c in chunks), default=0),
        embedder=embedder.name,
    )
