"""Streamlit UI over QueryService.

A reviewer-facing view of one answer: status, confidence and review flag first; then the answer,
caveats and missing information; then every citation with its verification mark and the source
chunk text; tool calls with their results; warnings; and a trace panel with retrieval scores,
cost, latency, confidence factors and a JSON download.

The service is built once per process (``st.cache_resource``) and imported in-process, so one
command starts the demo. Environment switches for tests and key-less demos:
``RAAPID_UI_EMBEDDER=hashing`` (fast embedder), ``RAAPID_UI_USE_LLM=0`` (retrieval only),
``RAAPID_UI_CACHE_DIR`` (index cache location).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/raapid_rag/ui/app.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from raapid_rag.config import DEFAULT_CACHE_DIR, get_settings  # noqa: E402
from raapid_rag.guardrails import InvalidQueryError  # noqa: E402
from raapid_rag.ingestion.embedder import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from raapid_rag.schema import GroundedAnswer  # noqa: E402
from raapid_rag.service import QueryService  # noqa: E402

EXAMPLES = [
    "What was Jane Doe-Test's most recent HbA1c, and when was it collected?",
    "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?",
    "How many days was John Q. Testpatient hospitalized, and was he eligible on the discharge date?",
    "Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?",
    "What metformin dose is recommended for patients with CKD stage 3?",
    "Who won the 2024 World Series?",
]

STATUS_STYLE = {
    "answered": ("green", "answered"),
    "partial": ("orange", "partial"),
    "insufficient_context": ("red", "insufficient_context (refused)"),
    "retrieval_only": ("blue", "retrieval_only (no model)"),
    "error": ("red", "error"),
}


@st.cache_resource(show_spinner="Loading index and models (first run takes about 15 seconds)...")
def load_service(embedder_kind: str, use_llm: bool, fault: str) -> QueryService:
    base = get_settings()
    settings = base.model_copy(update={"fault_injection": fault}) if fault else base
    embedder = HashingEmbedder() if embedder_kind == "hashing" else SentenceTransformerEmbedder(settings.embedding_model)
    cache_dir = Path(os.environ.get("RAAPID_UI_CACHE_DIR", DEFAULT_CACHE_DIR))
    return QueryService.from_settings(settings, embedder=embedder, cache_dir=cache_dir, use_llm=use_llm)


def render_answer(answer: GroundedAnswer, show_trace: bool) -> None:
    colour, label = STATUS_STYLE.get(answer.answer_status.value, ("gray", answer.answer_status.value))
    review = "**yes**" if answer.needs_human_review else "no"
    st.markdown(
        f"**Status:** :{colour}[{label}] &nbsp;|&nbsp; **Confidence:** {answer.confidence:.2f} ({answer.confidence_band.value}) "
        f"&nbsp;|&nbsp; **Needs human review:** {review}"
    )
    if answer.answer:
        st.markdown(f"#### Answer\n{answer.answer}")
    for c in answer.caveats:
        st.info(f"Caveat: {c}")
    for m in answer.missing_information:
        st.warning(f"Missing information: {m}")
    for e in answer.errors:
        st.error(e)

    chunk_text = {c["chunk_id"]: c for c in (answer.trace.get("retrieval") or {}).get("chunks", [])}
    if answer.citations:
        st.markdown("#### Citations")
        for i, cit in enumerate(answer.citations, start=1):
            mark = ":green[verified]" if cit.verified else ":red[NOT verified]"
            st.markdown(f"{i}. `{cit.chunk_id}` · {cit.doc_title} · {cit.section} · {mark}\n\n> {cit.evidence}")
            source = chunk_text.get(cit.chunk_id)
            if source:
                with st.expander(f"Source chunk {cit.chunk_id}"):
                    st.text(source["text"])

    if answer.tool_calls:
        st.markdown("#### Tool calls")
        for t in answer.tool_calls:
            suffix = " (cached)" if t.cached else (" (ERROR)" if t.error else "")
            with st.expander(f"{t.name}({json.dumps(t.arguments)}){suffix}"):
                st.json({"result": t.result, "error": t.error, "latency_ms": t.latency_ms})

    for w in answer.warnings:
        st.warning(w)

    if show_trace:
        trace = answer.trace
        with st.expander("Trace: retrieval, cost, latency, confidence factors"):
            retrieval = trace.get("retrieval") or {}
            rows = [
                {"chunk_id": c["chunk_id"], "section": c["section"], "dense": c["dense_score"], "bm25": c["bm25_score"],
                 "fused": c["fused_score"], "anchored": c.get("anchored", False)}
                for c in retrieval.get("chunks", [])
            ]
            if rows:
                st.dataframe(rows, use_container_width=True)
            st.markdown(
                f"gate: `{retrieval.get('gate_reason', trace.get('gate_reason', 'n/a'))}` · anchored: `{retrieval.get('anchored_docs', [])}` · "
                f"best dense {retrieval.get('best_dense_score', 0):.3f} · best BM25 {retrieval.get('best_bm25_score', 0):.2f}"
            )
            usage = trace.get("usage") or {}
            st.markdown(
                f"model calls: {trace.get('llm_calls', 0)} · repairs: {trace.get('repair_attempts', 0)} · "
                f"tokens: {usage.get('prompt_tokens', 0)} + {usage.get('completion_tokens', 0)} · "
                f"est. cost: ${(trace.get('est_cost_usd') or 0):.4f} · latency: {trace.get('total_latency_ms', 0):.0f} ms · "
                f"cache hit: {trace.get('cache_hit', False)} · prompt: {trace.get('prompt_version')}"
            )
            if trace.get("confidence_factors"):
                st.json(trace["confidence_factors"])
            if trace.get("reasoning_summary"):
                st.caption(f"Model reasoning summary: {trace['reasoning_summary']}")
        st.download_button("Download JSON", answer.model_dump_json(indent=2), file_name="grounded_answer.json", mime="application/json")


def main() -> None:
    st.set_page_config(page_title="RAAPID grounded QA", layout="wide")

    embedder_kind = os.environ.get("RAAPID_UI_EMBEDDER", "st")
    key_available = bool(get_settings().openai_api_key) and os.environ.get("RAAPID_UI_USE_LLM", "1") != "0"

    st.sidebar.title("RAAPID grounded QA")
    use_llm = st.sidebar.toggle("Use language model", value=key_available, disabled=not key_available,
                                help="Off: retrieval only, no key needed. On: full grounded answer with tools.")
    fault = st.sidebar.selectbox("Fault injection (demo of the repair path)", ["", "bad_citation", "malformed_json"])
    show_trace = st.sidebar.checkbox("Show trace", value=True)

    service = load_service(embedder_kind, use_llm, fault)
    health = service.health()
    model_line = health["model"] if health["llm_available"] else "none (retrieval only)"
    st.sidebar.markdown(
        f"**Index:** {health['chunks']} chunks · fingerprint `{health['index_fingerprint']}`  \n"
        f"**Model:** {model_line}  \n**Prompt:** {health['prompt_version']}  \n**Embedder:** {health['embedding_model']}"
    )
    docs = service.documents()
    titles = {d["doc_id"]: d["title"] for d in docs}
    scope = st.sidebar.multiselect("Scope to documents (optional)", list(titles), format_func=lambda i: f"{i}: {titles[i][:40]}")

    st.title("Grounded question answering over clinical and policy documents")
    st.caption("Answers come only from the ingested corpus and tool results. Every citation is checked against the source text. "
               "Refusals and low confidence are signals, not failures. Synthetic data only.")

    st.markdown("**Examples**")
    cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 3].button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["question"] = example

    question = st.text_input("Question", key="question", placeholder="Ask about the documents...")
    ask = st.button("Ask", key="ask", type="primary")

    if ask and question.strip():
        try:
            with st.spinner("Retrieving, reasoning, validating..."):
                answer = service.ask(question, doc_ids=scope or None)
        except InvalidQueryError as exc:
            st.error(str(exc))
        else:
            st.session_state.setdefault("history", []).insert(0, answer)

    history: list[GroundedAnswer] = st.session_state.get("history", [])
    if history:
        st.markdown(f"### {history[0].query}")
        render_answer(history[0], show_trace)
        if len(history) > 1:
            st.markdown("---")
            st.markdown("#### Earlier in this session")
            for earlier in history[1:]:
                with st.expander(f"{earlier.answer_status.value} · {earlier.query[:80]}"):
                    render_answer(earlier, show_trace=False)


main()
