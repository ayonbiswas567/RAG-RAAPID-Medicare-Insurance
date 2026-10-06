"""Loader tests: front matter parsing, validation, and corpus-level invariants."""

import logging

import pytest

from raapid_rag.ingestion.loader import (
    CorpusError,
    load_corpus,
    load_document,
    parse_front_matter,
)

VALID = """---
doc_id: doc_a
doc_type: clinical_note
title: Test Note
date_of_service: 2025-01-01
---
> banner line

## Section One
Body text one.
"""


def test_parse_front_matter_returns_metadata_and_body():
    meta, body = parse_front_matter(VALID)
    assert meta["doc_id"] == "doc_a"
    assert meta["title"] == "Test Note"
    assert meta["date_of_service"] == "2025-01-01"
    assert body.lstrip().startswith("> banner line")
    assert "## Section One" in body


def test_missing_front_matter_raises_with_filename(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("## No front matter\ntext\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="bad.md"):
        load_document(p)


def test_unterminated_front_matter_raises(tmp_path):
    p = tmp_path / "unterminated.md"
    p.write_text("---\ndoc_id: x\ndoc_type: t\ntitle: T\n## Body without closing fence\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="unterminated.md"):
        load_document(p)


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "nokey.md"
    p.write_text("---\ndoc_id: x\ntitle: T\n---\n## S\ntext\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="doc_type"):
        load_document(p)


def test_non_utf8_file_raises(tmp_path):
    p = tmp_path / "latin1.md"
    p.write_bytes(b"---\ndoc_id: x\ndoc_type: t\ntitle: T\n---\n## S\nbad \xff\xfe bytes\n")
    with pytest.raises(CorpusError, match="UTF-8"):
        load_document(p)


def test_duplicate_doc_id_across_files_raises(tmp_path):
    (tmp_path / "one.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "two.md").write_text(VALID, encoding="utf-8")
    with pytest.raises(CorpusError, match="doc_a"):
        load_corpus(tmp_path)


def test_empty_body_is_skipped_with_warning(tmp_path, caplog):
    (tmp_path / "empty.md").write_text("---\ndoc_id: e\ndoc_type: t\ntitle: T\n---\n\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        docs = load_corpus(tmp_path)
    assert docs == []
    assert "empty.md" in caplog.text


def test_load_document_populates_fields(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(VALID, encoding="utf-8")
    doc = load_document(p)
    assert doc.doc_id == "doc_a"
    assert doc.doc_type == "clinical_note"
    assert doc.title == "Test Note"
    assert doc.metadata["date_of_service"] == "2025-01-01"
    assert "Body text one." in doc.body
    assert doc.source_path.endswith("doc.md")


def test_load_corpus_ignores_non_markdown_files(tmp_path):
    (tmp_path / "doc.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a document", encoding="utf-8")
    assert [d.doc_id for d in load_corpus(tmp_path)] == ["doc_a"]


def test_real_corpus_loads_eight_documents():
    from raapid_rag.config import PROJECT_ROOT

    docs = load_corpus(PROJECT_ROOT / "data" / "corpus")
    assert {d.doc_id for d in docs} == {
        "note_001", "note_002", "note_003", "note_004",
        "policy_001", "policy_002", "guideline_001", "guideline_002",
    }
    assert {d.doc_type for d in docs} == {"clinical_note", "payer_policy", "coding_guideline"}
