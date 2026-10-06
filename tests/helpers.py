"""Shared test helpers: a real corpus index built with the fast hashing embedder, and a fake
model that answers by quoting the first retrieved chunk (with selectable misbehaviours)."""

from __future__ import annotations

import json
import re

from raapid_rag.config import PROJECT_ROOT, Settings
from raapid_rag.ingestion.embedder import HashingEmbedder
from raapid_rag.ingestion.pipeline import build_index
from raapid_rag.ingestion.store import InMemoryHybridStore
from raapid_rag.llm.client import ChatMessage, ChatResult, LLMUnavailable
from raapid_rag.retrieval.retriever import Retriever
from raapid_rag.service import QueryService
from raapid_rag.tools.registry import default_registry

HEADER_RE = re.compile(r"^\[([^\]]+)\] .*\n(.+)$", re.MULTILINE)


def final(payload) -> ChatResult:
    return ChatResult(message=ChatMessage(content=json.dumps(payload), tool_calls=[]), finish_reason="stop",
                      usage={"prompt_tokens": 500, "completion_tokens": 60}, latency_ms=2.0, model="fake")


class EchoLLM:
    """A fake model that answers by quoting the first retrieved chunk verbatim.

    ``mode`` selects a misbehaviour so the service's handling of each can be tested:
    echo (valid), one_bad (adds an unverifiable citation every time), all_bad (only bogus
    citations), refuse_with_text (insufficient_context but with prose in the answer),
    with_caveat (valid answer plus a caveat), raise (provider unavailable).
    """

    def __init__(self, mode: str = "echo"):
        self.mode = mode
        self.calls = 0

    def chat(self, messages, *, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.mode == "raise":
            raise LLMUnavailable("LLM unavailable after 4 attempt(s): rate limited")
        user = messages[1]["content"]
        chunk_id, first_line = HEADER_RE.search(user).groups()
        good = {"chunk_id": chunk_id, "evidence": first_line.strip()[:60]}
        bogus = {"chunk_id": "note_999#c99", "evidence": "fabricated"}
        if self.mode == "refuse_with_text":
            payload = {"answer": "The context does not say, but generally metformin is reduced when eGFR falls.",
                       "answer_status": "insufficient_context", "citations": [],
                       "missing_information": ["No dosing guidance in the context."], "caveats": [], "self_reported_confidence": 0.3,
                       "reasoning_summary": "Nothing relevant."}
        else:
            citations = {"echo": [good], "one_bad": [good, bogus], "all_bad": [bogus], "with_caveat": [good]}[self.mode]
            caveats = ["Per the eligibility check, coverage was inactive on the date of service."] if self.mode == "with_caveat" else []
            payload = {"answer": f"Echo: {good['evidence']}", "answer_status": "answered", "citations": citations,
                       "missing_information": [], "caveats": caveats, "self_reported_confidence": 0.9, "reasoning_summary": "Quoted the first chunk."}
        return final(payload)


def build_test_store() -> InMemoryHybridStore:
    store, _ = build_index(Settings(_env_file=None), HashingEmbedder(), corpus_dir=PROJECT_ROOT / "data" / "corpus")
    return store


def make_test_service(store: InMemoryHybridStore, mode: str = "echo", llm: bool = True, **overrides) -> tuple[QueryService, EchoLLM | None]:
    """Service over the real corpus with the fake model. Dense gating is switched off (hashing
    cosine is not on MiniLM's scale); relevance is carried by BM25 at its real threshold unless
    overridden."""
    kwargs = {"openai_api_key": "sk-test", "retrieval_min_dense_score": 0.99, "retrieval_min_bm25_score": 1.0}
    kwargs.update(overrides)
    s = Settings(_env_file=None, **kwargs)
    fake = EchoLLM(mode) if llm else None
    service = QueryService(settings=s, retriever=Retriever(store, s, count_tokens=lambda t: len(t.split())),
                           tools=default_registry(), llm=fake, system_prompt="TEST PROMPT", index_fingerprint="fp-test")
    return service, fake
