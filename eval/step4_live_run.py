"""Live end-to-end run of retrieval + agent against the real model, prompt v1 vs v2.

This is the prompt-iteration experiment: the same questions are asked under both system
prompts and the outcomes (status, citations, verification, tool use, repairs) are recorded
to docs/runs/step4_live_run.json so the changelog can cite real behaviour, not intent.

Run:  .venv/Scripts/python eval/step4_live_run.py [--fault bad_citation]
Cost: roughly a dozen gpt-4o-mini calls; well under one US cent in total.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raapid_rag.agent import GroundedAgent  # noqa: E402
from raapid_rag.cli import DEFAULT_CACHE_DIR  # noqa: E402
from raapid_rag.config import Settings, get_settings  # noqa: E402
from raapid_rag.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from raapid_rag.ingestion.pipeline import build_index  # noqa: E402
from raapid_rag.llm.client import LLMClient, estimate_cost_usd  # noqa: E402
from raapid_rag.llm.prompting import load_system_prompt  # noqa: E402
from raapid_rag.retrieval.retriever import Retriever  # noqa: E402
from raapid_rag.schema import CitationOut, normalize_for_match  # noqa: E402
from raapid_rag.tools.registry import default_registry  # noqa: E402

QUERIES = [
    ("recent_a1c", "What was Jane Doe-Test's most recent HbA1c, and what was the prior value?"),
    ("hcc_mapping", "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?"),
    ("los_and_eligibility", "How many days was John Q. Testpatient hospitalized, and was he eligible for coverage on the discharge date?"),
    ("telehealth_submittable", "Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?"),
    ("canary_parametric", "What metformin dose is recommended for patients with CKD stage 3?"),
    ("missing_fact", "What blood pressure medication is Maria Fictitious taking?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", default="", choices=["", "bad_citation", "malformed_json"])
    parser.add_argument("--versions", default="v1,v2")
    parser.add_argument("--out", default=str(ROOT / "docs" / "runs" / "step4_live_run.json"))
    parser.add_argument("--only", default="", help="Comma-separated query labels to run (default: all).")
    args = parser.parse_args()
    queries = [(l, q) for l, q in QUERIES if not args.only or l in args.only.split(",")]

    base = get_settings()
    store, report = build_index(base, SentenceTransformerEmbedder(base.embedding_model), cache_dir=DEFAULT_CACHE_DIR)
    retriever = Retriever(store, base)
    tools = default_registry()
    print(f"index: {report.n_chunks} chunks (cache={report.loaded_from_cache}) | model: {base.llm_model} | fault={args.fault or 'none'}\n")

    runs: list[dict] = []
    for version in args.versions.split(","):
        settings = Settings(_env_file=None, openai_api_key=base.openai_api_key, llm_model=base.llm_model,
                            prompt_version=version, fault_injection=args.fault)
        llm = LLMClient(settings)
        agent = GroundedAgent(llm, tools, settings, load_system_prompt(version, settings.prompts_dir))
        print(f"===== prompt {version} =====")
        for label, query in queries:
            retrieval = retriever.retrieve(query)
            entry: dict = {"prompt_version": version, "label": label, "query": query,
                           "retrieval": {"passes_gate": retrieval.passes_gate, "gate_reason": retrieval.gate_reason,
                                         "best_dense": round(retrieval.best_dense_score, 3), "best_bm25": round(retrieval.best_bm25_score, 2),
                                         "chunk_ids": retrieval.chunk_ids}}
            if not retrieval.passes_gate:
                entry["outcome"] = "refused_by_gate"
                print(f"[{label}] REFUSED BY GATE ({retrieval.gate_reason})")
                runs.append(entry)
                continue

            chunks = [c.chunk for c in retrieval.chunks]
            result = agent.run(query, chunks)
            by_id = {c.chunk_id: c for c in chunks}
            out = result.model_output
            cits = []
            if out is not None:
                for c in out.citations:
                    chunk = by_id.get(c.chunk_id)
                    verified = bool(chunk) and normalize_for_match(c.evidence) in normalize_for_match(chunk.text)
                    cits.append(CitationOut(chunk_id=c.chunk_id, evidence=c.evidence, doc_id=chunk.doc_id if chunk else "?",
                                            doc_title=chunk.doc_title if chunk else "?", section=chunk.section if chunk else "?",
                                            date=chunk.date if chunk else "", verified=verified).model_dump())
            entry.update({
                "outcome": "ok" if out is not None else "unrepaired",
                "answer_status": out.answer_status if out else None,
                "answer": out.answer if out else None,
                "self_reported_confidence": out.self_reported_confidence if out else None,
                "missing_information": out.missing_information if out else None,
                "reasoning_summary": out.reasoning_summary if out else None,
                "citations": cits,
                "tool_calls": [r.model_dump(mode="json") for r in result.tool_records],
                "repair_attempts": result.repair_attempts,
                "validation_errors": result.validation_errors,
                "fault_injected": result.fault_injected,
                "llm_calls": result.llm_calls,
                "usage": result.usage,
                "est_cost_usd": estimate_cost_usd(settings.llm_model, prompt_tokens=result.usage["prompt_tokens"], completion_tokens=result.usage["completion_tokens"]),
                "latency_ms": result.latency_ms,
                "raw_outputs": result.raw_outputs,
            })
            runs.append(entry)
            status = entry["answer_status"] or "UNREPAIRED"
            tools_used = ", ".join(f"{r.name}{'!' if r.error else ''}" for r in result.tool_records) or "-"
            print(f"[{label}] {status:21} cits={len(cits)} verified={sum(c['verified'] for c in cits)} "
                  f"tools={tools_used} repairs={result.repair_attempts} calls={result.llm_calls} "
                  f"tok={result.usage['prompt_tokens']}+{result.usage['completion_tokens']} {result.latency_ms:.0f}ms")
            print(f"    answer: {(entry['answer'] or '')[:220]}")
            if entry["missing_information"]:
                print(f"    missing: {entry['missing_information']}")
            if result.validation_errors:
                print(f"    errors: {result.validation_errors}")
        print(f"usage {version}: {llm.usage.to_dict()} est ${estimate_cost_usd(settings.llm_model, prompt_tokens=llm.usage.prompt_tokens, completion_tokens=llm.usage.completion_tokens):.4f}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "model": base.llm_model, "fault": args.fault, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
