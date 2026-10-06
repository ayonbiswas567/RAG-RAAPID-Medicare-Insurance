"""Print retrieval signals for a fixed list of probe queries.

Purpose: see, with the real embedding model, what "relevant" and "irrelevant" look like in
absolute numbers so that the refusal-gate thresholds are calibrated rather than guessed.
Run:  .venv/Scripts/python eval/retrieval_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raapid_rag.cli import DEFAULT_CACHE_DIR  # noqa: E402
from raapid_rag.config import get_settings  # noqa: E402
from raapid_rag.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from raapid_rag.ingestion.pipeline import build_index  # noqa: E402
from raapid_rag.retrieval.retriever import Retriever  # noqa: E402

PROBES: list[tuple[str, str, str]] = [
    # (expected, label, query)
    ("PASS", "recent A1c", "What was Jane Doe-Test's most recent HbA1c?"),
    ("PASS", "length of stay", "How long was John Testpatient's hospital stay?"),
    ("PASS", "audio-only", "Are audio-only telehealth visits acceptable for risk adjustment?"),
    ("PASS", "uncertain dx", "Can 'possible rheumatoid arthritis' be coded at an outpatient visit?"),
    ("PASS", "HCC for E11.22", "Which HCC category does E11.22 map to?"),
    ("PASS", "history of cancer", "Does history of breast cancer count as an active condition?"),
    ("PASS", "policy id", "What does policy RA-110 section 2.2 say?"),
    ("PASS", "paraphrase", "How well controlled is the diabetic patient's sugar?"),
    ("PASS*", "in-corpus, fact missing", "What blood pressure medication is Maria Fictitious taking?"),
    ("PASS*", "canary (parametric)", "What metformin dose is recommended for CKD stage 3?"),
    ("REFUSE", "out of corpus, medical", "What is the recommended amoxicillin dose for strep throat?"),
    ("REFUSE", "out of corpus, general", "Who won the 2024 World Series?"),
    ("REFUSE", "gibberish", "purple elephant quantum trombone"),
]


def main() -> None:
    settings = get_settings()
    store, report = build_index(settings, SentenceTransformerEmbedder(settings.embedding_model), cache_dir=DEFAULT_CACHE_DIR)
    retriever = Retriever(store, settings)
    print(f"index: {report.n_chunks} chunks from {report.n_documents} docs (cache={report.loaded_from_cache}, {report.build_seconds}s)")
    print(f"gate thresholds: dense>={settings.retrieval_min_dense_score} OR bm25>={settings.retrieval_min_bm25_score}\n")
    print(f"{'expect':7} {'gate':7} {'dense':>6} {'bm25':>6}  {'label':24} top-3 chunk ids")
    for expected, label, query in PROBES:
        r = retriever.retrieve(query)
        verdict = "PASS" if r.passes_gate else "REFUSE"
        flag = "" if verdict == expected.rstrip("*") else "  <-- MISMATCH"
        print(f"{expected:7} {verdict:7} {r.best_dense_score:6.3f} {r.best_bm25_score:6.2f}  {label:24} {', '.join(r.chunk_ids[:3])}{flag}")


if __name__ == "__main__":
    main()
