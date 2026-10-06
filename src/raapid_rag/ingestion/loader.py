"""Load corpus files from disk into :class:`Document` objects.

Corpus files are Markdown with a leading front-matter block::

    ---
    doc_id: note_001
    doc_type: clinical_note
    title: ...
    ---
    body

The front matter is plain ``key: value`` lines.  A tiny parser is used instead of a YAML
library so there is one less dependency and the accepted format is exactly what a
reviewer can see in the files.

Every failure raises :class:`CorpusError` with the offending file name.  Ingestion is a
place to fail loudly: a silently skipped or mis-parsed document produces wrong citations
later, which is much harder to diagnose.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import Document

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("doc_id", "doc_type", "title")
FENCE = "---"


class CorpusError(ValueError):
    """Raised when a corpus file cannot be loaded or violates a corpus-level invariant."""


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split ``raw`` into ``(metadata, body)``.

    Raises ``ValueError`` (without a file name; the caller adds it) when the block is
    missing or unterminated.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != FENCE:
        raise ValueError("file does not start with a '---' front-matter fence")

    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == FENCE:
            body = "\n".join(lines[i + 1 :])
            return meta, body
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"front-matter line {i + 1} is not 'key: value': {line!r}")
        meta[key.strip()] = value.strip()

    raise ValueError("front-matter block is not terminated by a closing '---'")


def load_document(path: Path) -> Document:
    """Load a single Markdown corpus file."""
    try:
        raw = path.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{path.name}: file is not valid UTF-8 ({exc.reason} at byte {exc.start})") from exc

    try:
        meta, body = parse_front_matter(raw)
    except ValueError as exc:
        raise CorpusError(f"{path.name}: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if not meta.get(k)]
    if missing:
        raise CorpusError(f"{path.name}: front matter is missing required key(s): {', '.join(missing)}")

    return Document(
        doc_id=meta["doc_id"],
        doc_type=meta["doc_type"],
        title=meta["title"],
        metadata=meta,
        body=body.strip("\n"),
        source_path=str(path),
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every ``*.md`` file in ``corpus_dir`` (sorted by name).

    Enforces the corpus-level invariant that ``doc_id`` is unique, because chunk ids are
    derived from it and a collision would make citations ambiguous.  Documents with an
    empty body are skipped with a warning rather than indexed as nothing.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise CorpusError(f"corpus directory does not exist: {corpus_dir}")

    docs: list[Document] = []
    seen: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        doc = load_document(path)
        if doc.doc_id in seen:
            raise CorpusError(
                f"{path.name}: duplicate doc_id '{doc.doc_id}' (already used by {seen[doc.doc_id]})"
            )
        seen[doc.doc_id] = path.name
        if not doc.body.strip():
            logger.warning("skipping %s: front matter present but body is empty", path.name)
            continue
        docs.append(doc)

    logger.info("loaded %d document(s) from %s", len(docs), corpus_dir)
    return docs
