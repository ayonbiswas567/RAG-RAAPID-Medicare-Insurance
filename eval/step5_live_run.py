"""Live run of the full QueryService path against the real model.

Exercises every exit of the service: gate refusal (no model call), model-side refusal, tool
use with verified citations, PHI flagging, explicit document scoping, the answer cache, and
fault injection through the salvage/repair path.  Results (full GroundedAnswer objects) are
written to docs/runs/step5_live_run.json.

Run:  .venv/Scripts/python eval/step5_live_run.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raapid_rag.cli import DEFAULT_CACHE_DIR  # noqa: E402
from raapid_rag.config import Settings, get_settings  # noqa: E402
from raapid_rag.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from raapid_rag.service import QueryService  # noqa: E402

CASES = [
    ("gate_refusal", "purple elephant quantum trombone", None),
    ("canary_parametric", "What metformin dose is recommended for patients with CKD stage 3?", None),
    ("hcc_mapping", "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?", None),
    ("telehealth_submittable", "Can the telehealth visit for Robert Nonreal be submitted for risk adjustment?", None),
    ("phi_flag", "Patient SSN 123-45-6789: what was Jane Doe-Test's most recent HbA1c?", None),
    ("scoped_to_policy", "What does section 2.2 say about audio-only encounters?", ["policy_002"]),
    ("cache_hit", "Which HCC category do the diabetes codes documented for Jane Doe-Test map to?", None),
]


def summarise(label: str, answer) -> str:
    t = answer.trace
    usage = t.get("usage") or {}
    tools = ", ".join(f"{x.name}{'*' if x.cached else ''}{'!' if x.error else ''}" for x in answer.tool_calls) or "-"
    verified = sum(c.verified for c in answer.citations)
    return (
        f"[{label}] {answer.answer_status.value:21} conf={answer.confidence:.2f} ({answer.confidence_band.value:6}) "
        f"review={'Y' if answer.needs_human_review else 'n'} cits={len(answer.citations)}/{verified}ok "
        f"tools={tools} calls={t.get('llm_calls', 0)} repairs={t.get('repair_attempts', 0)} "
        f"tok={usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} "
        f"${(t.get('est_cost_usd') or 0):.4f} {t.get('total_latency_ms', 0):.0f}ms cache={t.get('cache_hit')}"
    )


def main() -> None:
    base = get_settings()
    embedder = SentenceTransformerEmbedder(base.embedding_model)
    service = QueryService.from_settings(base, embedder=embedder, cache_dir=DEFAULT_CACHE_DIR)
    print("health:", json.dumps(service.health()))
    records = []

    for label, query, doc_ids in CASES:
        answer = service.ask(query, doc_ids=doc_ids)
        print(summarise(label, answer))
        if answer.answer:
            print(f"    answer: {answer.answer[:200]}")
        for w in answer.warnings:
            print(f"    warning: {w[:140]}")
        for e in answer.errors:
            print(f"    error: {e[:140]}")
        if answer.missing_information:
            print(f"    missing: {answer.missing_information[0][:140]}")
        records.append({"label": label, "doc_ids": doc_ids, "answer": answer.model_dump(mode="json")})

    # Fault injection through the service: the repair path should recover, and if it did not,
    # the salvage path should still produce a flagged partial answer rather than an error.
    faulty = Settings(_env_file=None, openai_api_key=base.openai_api_key, llm_model=base.llm_model,
                      prompt_version=base.prompt_version, fault_injection="bad_citation")
    faulty_service = QueryService.from_settings(faulty, embedder=embedder, cache_dir=DEFAULT_CACHE_DIR)
    answer = faulty_service.ask("What was Jane Doe-Test's most recent HbA1c, and what was the prior value?")
    print(summarise("fault_injection", answer))
    for w in answer.warnings:
        print(f"    warning: {w[:140]}")
    records.append({"label": "fault_injection", "doc_ids": None, "answer": answer.model_dump(mode="json")})

    out = ROOT / "docs" / "runs" / "step5_live_run.json"
    out.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": base.llm_model,
                               "prompt_version": base.prompt_version, "runs": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
