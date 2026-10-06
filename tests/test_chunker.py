"""Chunker tests.

A whitespace word counter stands in for the real tokenizer so these tests are fast and
deterministic. The chunker only ever calls the counter, so the behaviour under test is the
same.
"""

from raapid_rag.ingestion.chunker import chunk_document
from raapid_rag.ingestion.models import Document


def count(text: str) -> int:
    return len(text.split())


def make_doc(body: str, title: str = "Doc Title", doc_id: str = "doc_a") -> Document:
    return Document(
        doc_id=doc_id,
        doc_type="clinical_note",
        title=title,
        metadata={"doc_id": doc_id, "doc_type": "clinical_note", "title": title, "date_of_service": "2025-01-01"},
        body=body,
        source_path="memory",
    )


def test_each_short_section_becomes_one_chunk_with_sequential_ids():
    body = "## Alpha\nOne two three.\n\n## Beta\nFour five six.\n"
    chunks = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)
    assert [c.section for c in chunks] == ["Alpha", "Beta"]
    assert [c.chunk_id for c in chunks] == ["doc_a#c00", "doc_a#c01"]
    assert chunks[0].text == "One two three."
    assert chunks[1].text == "Four five six."


def test_long_section_is_windowed_within_target_and_windows_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(1, 21)]
    body = "## Course\n" + " ".join(sentences)
    chunks = chunk_document(make_doc(body), target_tokens=30, overlap_tokens=8, count_tokens=count)

    assert len(chunks) > 1
    assert all(count(c.text) <= 30 for c in chunks)
    # Every sentence survives somewhere.
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)
    # The first sentence of each window already appeared in the previous window (overlap).
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence = nxt.text.split(".")[0].strip() + "."
        assert first_sentence in prev.text
    # Windowed sections are labelled with their part number.
    assert chunks[0].section == f"Course (part 1/{len(chunks)})"
    assert chunks[-1].section == f"Course (part {len(chunks)}/{len(chunks)})"


def test_windowing_always_makes_progress_when_overlap_is_large():
    # overlap_tokens larger than any window must not produce an infinite loop or duplicates.
    sentences = [f"Short sentence {i}." for i in range(1, 30)]
    body = "## X\n" + " ".join(sentences)
    chunks = chunk_document(make_doc(body), target_tokens=6, overlap_tokens=100, count_tokens=count)
    assert len(chunks) < 60
    assert len({c.text for c in chunks}) == len(chunks)


def test_single_unit_longer_than_target_is_split_by_words():
    body = "## X\n" + " ".join(f"w{i}" for i in range(100))  # one "sentence", no punctuation
    chunks = chunk_document(make_doc(body), target_tokens=25, overlap_tokens=5, count_tokens=count)
    assert len(chunks) >= 4
    assert all(count(c.text) <= 25 for c in chunks)


def test_body_without_headers_is_one_section_named_after_document_title():
    body = "Just some text without any headers at all."
    chunks = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)
    assert len(chunks) == 1
    assert chunks[0].section == "Doc Title"
    assert chunks[0].text == body


def test_preamble_before_first_header_becomes_its_own_chunk():
    body = "> SYNTHETIC banner text.\n\n## Alpha\nText here.\n"
    chunks = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)
    assert [c.section for c in chunks] == ["Preamble", "Alpha"]
    assert chunks[0].text == "> SYNTHETIC banner text."


def test_chunk_text_is_always_a_verbatim_substring_of_the_body():
    sentences = [f"Sentence {i} ends here." for i in range(1, 40)]
    body = "> banner\n\n## A\n" + "\n".join(sentences[:20]) + "\n\n## B\n" + " ".join(sentences[20:])
    doc = make_doc(body)
    chunks = chunk_document(doc, target_tokens=20, overlap_tokens=4, count_tokens=count)
    assert len(chunks) > 3
    assert all(c.text in doc.body for c in chunks)


def test_embed_text_prefixes_document_title_and_section():
    body = "## Alpha\nText here.\n"
    chunks = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)
    assert chunks[0].embed_text.startswith("Doc Title | Alpha\n")
    assert chunks[0].embed_text.endswith("Text here.")


def test_document_metadata_propagates_to_chunks():
    body = "## Alpha\nText here.\n"
    chunk = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)[0]
    assert chunk.doc_id == "doc_a"
    assert chunk.doc_type == "clinical_note"
    assert chunk.doc_title == "Doc Title"
    assert chunk.metadata["date_of_service"] == "2025-01-01"
    assert chunk.index == 0


def test_empty_sections_are_skipped():
    body = "## Empty\n\n## Full\nSome text.\n"
    chunks = chunk_document(make_doc(body), target_tokens=50, overlap_tokens=5, count_tokens=count)
    assert [c.section for c in chunks] == ["Full"]
