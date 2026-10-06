"""Section-aware chunking with a token-window fallback.

Strategy (see docs/ENGINEERING_NOTES.md, decision D5):

1. Split the document body on ``## `` headers.  Clinical notes and policies are written in
   sections, and a diagnosis usually sits in the same section as the evidence for it, so a
   section is the most meaningful unit to cite.
2. A section whose token count is within ``target_tokens`` becomes exactly one chunk.
3. A longer section is split into windows.  Windows are built from *units* (lines, then
   sentences) packed greedily up to ``target_tokens``; consecutive windows share the last
   ``overlap_tokens`` worth of units so that a sentence cut at a boundary still appears
   whole in one window.  A single unit longer than the target is split on words.

Two invariants matter downstream and are tested:

* ``Chunk.text`` is always a verbatim substring of ``Document.body``.  The chunker works
  with character offsets and slices the original text; it never re-joins pieces.  The
  evidence-verification guardrail relies on this.
* Chunk ids are deterministic for a given document.

Token counting is injected (``count_tokens``) so that chunk sizing can use the embedding
model's own tokenizer, which is what actually enforces the 256-token limit.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Chunk, Document

HEADER_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
# Sentence boundary: terminal punctuation followed by whitespace and an upper-case letter,
# digit, quote or bracket.  Deliberately simple; it only has to be good enough for packing.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
WORD_RE = re.compile(r"\S+")

Span = tuple[int, int]  # [start, end) character offsets into the section text
TokenCounter = Callable[[str], int]


def chunk_document(
    doc: Document,
    *,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    """Split ``doc`` into chunks. See module docstring for the strategy."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    chunks: list[Chunk] = []
    index = 0
    for section_title, section_text in _split_sections(doc):
        stripped = section_text.strip()
        if not stripped:
            continue

        if count_tokens(stripped) <= target_tokens:
            chunks.append(_make_chunk(doc, index, section_title, stripped))
            index += 1
            continue

        windows = _window(stripped, target_tokens, overlap_tokens, count_tokens)
        total = len(windows)
        for part, (start, end) in enumerate(windows, start=1):
            text = stripped[start:end].strip()
            chunks.append(_make_chunk(doc, index, f"{section_title} (part {part}/{total})", text))
            index += 1
    return chunks


# ----------------------------------------------------------------------------- sections


def _split_sections(doc: Document) -> list[tuple[str, str]]:
    """Return ``(title, text)`` pairs. Text excludes the header line itself."""
    body = doc.body
    headers = list(HEADER_RE.finditer(body))
    if not headers:
        return [(doc.title, body)]

    sections: list[tuple[str, str]] = []
    preamble = body[: headers[0].start()]
    if preamble.strip():
        sections.append(("Preamble", preamble))

    for i, match in enumerate(headers):
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        sections.append((match.group(1).strip(), body[start:end]))
    return sections


# ------------------------------------------------------------------------------ windows


def _window(text: str, target: int, overlap: int, count: TokenCounter) -> list[Span]:
    """Pack units into overlapping windows of at most ``target`` tokens."""
    units = _units(text, target, count)
    sizes = [count(text[s:e]) for s, e in units]

    windows: list[Span] = []
    start_unit = 0
    n = len(units)
    while start_unit < n:
        end_unit = start_unit
        used = 0
        while end_unit < n and used + sizes[end_unit] <= target:
            used += sizes[end_unit]
            end_unit += 1
        if end_unit == start_unit:  # single unit larger than target (cannot happen after _units, but be safe)
            end_unit = start_unit + 1

        # Token counts are not perfectly additive for word-piece tokenizers; if the joined
        # window overshoots, shed units from the end until it fits (keeping at least one).
        while end_unit - start_unit > 1 and count(text[units[start_unit][0] : units[end_unit - 1][1]]) > target:
            end_unit -= 1

        windows.append((units[start_unit][0], units[end_unit - 1][1]))
        if end_unit >= n:
            break

        # Overlap: step back over trailing units worth up to ``overlap`` tokens, but always
        # advance by at least one unit so the loop terminates.
        next_start = end_unit
        carried = 0
        while next_start - 1 > start_unit and carried + sizes[next_start - 1] <= overlap:
            carried += sizes[next_start - 1]
            next_start -= 1
        start_unit = max(next_start, start_unit + 1)
    return windows


def _units(text: str, target: int, count: TokenCounter) -> list[Span]:
    """Split text into line -> sentence -> word units, none larger than ``target`` tokens."""
    units: list[Span] = []
    for line_start, line_end in _line_spans(text):
        for s, e in _sentence_spans(text, line_start, line_end):
            if count(text[s:e]) <= target:
                units.append((s, e))
            else:
                units.extend(_word_spans(text, s, e, target, count))
    return units


def _line_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        if line.strip():
            spans.append((pos, end))
        pos = end + 1
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[Span]:
    segment = text[start:end]
    spans: list[Span] = []
    cursor = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(segment):
        spans.append((start + cursor, start + match.start()))
        cursor = match.end()
    spans.append((start + cursor, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _word_spans(text: str, start: int, end: int, target: int, count: TokenCounter) -> list[Span]:
    """Greedily group words so each group stays within ``target`` tokens."""
    words = [(start + m.start(), start + m.end()) for m in WORD_RE.finditer(text[start:end])]
    groups: list[Span] = []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and count(text[words[i][0] : words[j][1]]) <= target:
            j += 1
        groups.append((words[i][0], words[j - 1][1]))
        i = j
    return groups


# -------------------------------------------------------------------------------- misc


def _make_chunk(doc: Document, index: int, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#c{index:02d}",
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        doc_title=doc.title,
        section=section,
        text=text,
        index=index,
        metadata=dict(doc.metadata),
    )
