"""CLI tests (retrieval-only commands). The hashing embedder keeps these fast."""

import json

from typer.testing import CliRunner

from raapid_rag.cli import app

runner = CliRunner()


def test_ingest_reports_document_and_chunk_counts_as_json(tmp_path):
    result = runner.invoke(app, ["ingest", "--embedder", "hashing", "--cache-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n_documents"] == 8
    assert report["n_chunks"] > 60
    assert report["loaded_from_cache"] is False


def test_search_returns_ranked_chunks_and_gate_decision_as_json(tmp_path):
    result = runner.invoke(
        app,
        ["search", "most recent HbA1c result", "--embedder", "hashing", "--cache-dir", str(tmp_path), "--top-k", "3", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["chunks"]) == 3
    assert payload["chunks"][0]["chunk_id"].startswith("note_001#")
    assert payload["passes_gate"] is True


def test_ask_without_llm_returns_retrieval_only_envelope(tmp_path):
    result = runner.invoke(
        app, ["ask", "Are audio-only telehealth visits acceptable?", "--embedder", "hashing", "--cache-dir", str(tmp_path), "--no-llm", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer_status"] == "retrieval_only"
    assert payload["schema_version"] == "1.0"
    assert payload["citations"] and payload["needs_human_review"] is True


def test_ask_human_output_shows_status_and_review_flag(tmp_path):
    result = runner.invoke(app, ["ask", "audio-only telehealth", "--embedder", "hashing", "--cache-dir", str(tmp_path), "--no-llm"])
    assert result.exit_code == 0, result.output
    assert "retrieval_only" in result.output and "review" in result.output.lower()


def test_search_human_output_mentions_gate(tmp_path):
    result = runner.invoke(app, ["search", "HbA1c", "--embedder", "hashing", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "gate" in result.output.lower()
