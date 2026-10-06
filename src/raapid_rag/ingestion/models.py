"""Core data types shared across ingestion, retrieval, and the agent.

Both types are frozen dataclasses: once a document is loaded or a chunk is produced it is
never mutated, which makes them safe to cache, hash, and pass between threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """One corpus file after front-matter parsing."""

    doc_id: str
    doc_type: str
    title: str
    metadata: dict[str, str]
    body: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    """A citable unit of text.

    ``chunk_id`` is deterministic (``<doc_id>#c<index>``) so that a citation produced
    today still points at the same text after the corpus is re-ingested.  ``text`` is
    always a verbatim substring of the parent document body; the evidence-verification
    guardrail depends on that property.
    """

    chunk_id: str
    doc_id: str
    doc_type: str
    doc_title: str
    section: str
    text: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text that is embedded and BM25-indexed.

        The document title and section name are prepended so that a chunk "knows" where
        it came from.  Without this, a chunk that reads only "Continue metformin 1000 mg"
        has no signal linking it to diabetes follow-up or to a particular note.
        """
        return f"{self.doc_title} | {self.section}\n{self.text}"

    @property
    def date(self) -> str:
        return self.metadata.get("date_of_service") or self.metadata.get("effective_date") or ""
