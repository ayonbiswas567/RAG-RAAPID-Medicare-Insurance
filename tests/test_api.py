"""HTTP API tests over the real corpus with the fake model.

Design points under test: every model outcome (answer, refusal, provider outage) is a 200 with
a schema-valid envelope; only unusable input is a 4xx; every response carries a request id that
is also written into the trace.
"""

import pytest
from fastapi.testclient import TestClient
from helpers import build_test_store, make_test_service

from raapid_rag.api import create_app


@pytest.fixture(scope="module")
def store():
    return build_test_store()


def client_for(store, **kwargs):
    service, fake = make_test_service(store, **kwargs)
    return TestClient(create_app(service)), fake


def test_root_points_to_docs(store):
    client, _ = client_for(store)
    body = client.get("/").json()
    assert body["service"] == "raapid-rag" and body["docs"] == "/docs"


def test_health_reports_index_and_model(store):
    client, _ = client_for(store)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["chunks"] == len(store) and body["llm_available"] is True


def test_query_returns_grounded_answer_with_request_id(store):
    client, fake = client_for(store)
    r = client.post("/query", json={"query": "What was the most recent HbA1c for Jane Doe-Test?"})
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.0" and body["answer_status"] == "answered"
    assert body["citations"] and all(c["verified"] for c in body["citations"])
    assert r.headers["X-Request-ID"] and body["trace"]["request_id"] == r.headers["X-Request-ID"]
    assert fake.calls == 1


def test_query_scope_is_honoured(store):
    client, _ = client_for(store)
    body = client.post("/query", json={"query": "telehealth", "doc_ids": ["policy_002"]}).json()
    assert body["citations"] and all(c["doc_id"] == "policy_002" for c in body["citations"])
    assert body["trace"]["retrieval"]["scope_doc_ids"] == ["policy_002"]


def test_query_can_omit_the_trace(store):
    client, _ = client_for(store)
    body = client.post("/query", json={"query": "What was the most recent HbA1c?", "include_trace": False}).json()
    assert "retrieval" not in body["trace"]
    assert body["trace"]["request_id"] and "prompt_version" in body["trace"]


def test_blank_query_is_400(store):
    client, _ = client_for(store)
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_missing_query_field_is_422(store):
    client, _ = client_for(store)
    assert client.post("/query", json={}).status_code == 422


def test_gate_refusal_is_a_200_envelope(store):
    client, fake = client_for(store)
    body = client.post("/query", json={"query": "zxqv plorth wibble"}).json()
    assert body["answer_status"] == "insufficient_context" and body["needs_human_review"] is True
    assert fake.calls == 0


def test_provider_outage_is_a_200_error_envelope_never_a_500(store):
    client, _ = client_for(store, mode="raise")
    r = client.post("/query", json={"query": "What was the most recent HbA1c?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer_status"] == "error" and any("unavailable" in e.lower() for e in body["errors"])


def test_documents_endpoint_lists_the_corpus(store):
    client, _ = client_for(store)
    docs = client.get("/documents").json()
    assert len(docs) == 8
    assert {"doc_id", "title", "doc_type", "n_chunks"} <= set(docs[0])


def test_search_endpoint_returns_retrieval_only(store):
    client, fake = client_for(store)
    body = client.post("/search", json={"query": "audio-only telehealth encounters"}).json()
    assert body["passes_gate"] is True
    assert body["chunks"] and {"chunk_id", "dense_score", "bm25_score", "fused_score", "text"} <= set(body["chunks"][0])
    assert fake.calls == 0
    assert client.post("/search", json={"query": " "}).status_code == 400
