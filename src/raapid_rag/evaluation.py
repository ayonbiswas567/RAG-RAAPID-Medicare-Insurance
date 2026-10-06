"""Evaluation harness: an eval set of questions with expectations, mechanical scoring, and a report.

Why mechanical checks rather than an LLM judge: every check here is something a reviewer can
verify by eye (status, which documents were cited, whether quotes verified, whether a keyword
is present, whether a tool ran, how many model calls were made).  That makes a failure
actionable and the pass rate defensible.  An LLM judge can be layered on later for free-text
quality; it is not needed to catch the failure modes this project is about.

Two modes:

* ``full``: the real service path, model included.  Scores every expectation.
* ``retrieval``: no model.  Scores only what retrieval can answer: did the gate behave as
  expected, and were the expected documents retrieved.  Runs with no API key, so it can gate a
  CI pipeline for free.

The harness is importable (unit-tested) and driven by ``raapid-rag eval`` / ``eval/run_eval.py``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .schema import GroundedAnswer
from .service import QueryService

ANSWER_STATUSES = {"answered", "partial"}
REFUSAL_STATUS = "insufficient_context"
Mode = Literal["full", "retrieval"]


# ------------------------------------------------------------------ eval set


class Expected(BaseModel):
    statuses: list[str] = Field(min_length=1, description="Acceptable answer_status values.")
    cited_docs_any: list[str] = Field(default_factory=list, description="At least one citation must come from one of these documents.")
    cited_docs_all: list[str] = Field(default_factory=list, description="Each of these documents must be cited.")
    answer_must_contain: list[str] = Field(default_factory=list, description="Case-insensitive substrings; alternatives separated by '|'.")
    answer_must_not_contain: list[str] = Field(default_factory=list)
    tools_any: list[str] = Field(default_factory=list, description="At least one of these tools must have been called.")
    max_llm_calls: int | None = Field(default=None, description="Upper bound on model calls (0 for gate refusals).")
    no_citations: bool = Field(default=False, description="Refusals must not cite anything.")


class EvalCase(BaseModel):
    id: str
    category: str
    query: str
    doc_ids: list[str] | None = None
    expected: Expected
    notes: str = ""

    @property
    def expects_refusal(self) -> bool:
        return set(self.expected.statuses) == {REFUSAL_STATUS}


def load_eval_set(path: Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(item) for item in raw]
    counts = Counter(c.id for c in cases)
    duplicates = sorted(cid for cid, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"eval set has duplicate case id(s): {', '.join(duplicates)}")
    return cases


# ------------------------------------------------------------------- scoring


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    passed: bool
    checks: list[CheckResult]
    status: str
    confidence: float
    band: str
    needs_human_review: bool
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    retrieved_docs: list[str]
    cited_docs: list[str]
    retrieval_hit: bool | None
    expected_refusal: bool
    answer_excerpt: str
    warnings: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _retrieved_docs(answer: GroundedAnswer) -> list[str]:
    chunks = (answer.trace.get("retrieval") or {}).get("chunks") or []
    return _ordered_unique([c.get("doc_id", "") for c in chunks if c.get("doc_id")])


def _retrieval_hit(case: EvalCase, retrieved: list[str]) -> bool | None:
    must_all = set(case.expected.cited_docs_all)
    must_any = set(case.expected.cited_docs_any)
    if not must_all and not must_any:
        return None
    have = set(retrieved)
    return must_all <= have and (not must_any or bool(must_any & have))


def _contains_any_alternative(text: str, spec: str) -> bool:
    haystack = text.casefold()
    return any(alt.strip().casefold() in haystack for alt in spec.split("|") if alt.strip())


def _metrics(case: EvalCase, answer: GroundedAnswer, checks: list[CheckResult], retrieved: list[str], cited: list[str]) -> CaseResult:
    trace = answer.trace
    usage = trace.get("usage") or {}
    return CaseResult(
        case_id=case.id,
        category=case.category,
        query=case.query,
        passed=all(c.passed for c in checks),
        checks=checks,
        status=answer.answer_status.value,
        confidence=answer.confidence,
        band=answer.confidence_band.value,
        needs_human_review=answer.needs_human_review,
        llm_calls=int(trace.get("llm_calls", 0) or 0),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        cost_usd=float(trace.get("est_cost_usd") or 0.0),
        latency_ms=float(trace.get("total_latency_ms") or 0.0),
        retrieved_docs=retrieved,
        cited_docs=cited,
        retrieval_hit=_retrieval_hit(case, retrieved),
        expected_refusal=case.expects_refusal,
        answer_excerpt=answer.answer[:200],
        warnings=list(answer.warnings),
        missing_information=list(answer.missing_information),
        caveats=list(answer.caveats),
        tools_called=[t.name + (" (cached)" if t.cached else "") for t in answer.tool_calls],
    )


def score_case(case: EvalCase, answer: GroundedAnswer) -> CaseResult:
    """Full-mode scoring: every expectation in the case."""
    e = case.expected
    status = answer.answer_status.value
    cited = _ordered_unique([c.doc_id for c in answer.citations])
    checks: list[CheckResult] = [CheckResult("status", status in e.statuses, f"got '{status}', expected one of {e.statuses}")]

    if e.no_citations:
        checks.append(CheckResult("no_citations", not answer.citations, f"{len(answer.citations)} citation(s) present"))
    if e.cited_docs_any:
        ok = bool(set(e.cited_docs_any) & set(cited))
        checks.append(CheckResult("cited_docs_any", ok, f"cited {cited}; expected any of {e.cited_docs_any}"))
    if e.cited_docs_all:
        missing = sorted(set(e.cited_docs_all) - set(cited))
        checks.append(CheckResult("cited_docs_all", not missing, f"missing {missing}" if missing else "all expected documents cited"))
    if answer.citations:
        n_ok = sum(1 for c in answer.citations if c.verified)
        checks.append(CheckResult("evidence_verified", n_ok == len(answer.citations), f"{n_ok}/{len(answer.citations)} citations verified"))
    for spec in e.answer_must_contain:
        checks.append(CheckResult(f"contains:{spec}", _contains_any_alternative(answer.answer, spec), f"answer excerpt: {answer.answer[:80]!r}"))
    for spec in e.answer_must_not_contain:
        checks.append(CheckResult(f"excludes:{spec}", not _contains_any_alternative(answer.answer, spec), f"answer excerpt: {answer.answer[:80]!r}"))
    if e.tools_any:
        used = sorted({t.name for t in answer.tool_calls})
        checks.append(CheckResult("tools_any", bool(set(e.tools_any) & set(used)), f"used {used}; expected any of {e.tools_any}"))
    if e.max_llm_calls is not None:
        calls = int(answer.trace.get("llm_calls", 0) or 0)
        checks.append(CheckResult("max_llm_calls", calls <= e.max_llm_calls, f"{calls} model call(s), maximum {e.max_llm_calls}"))

    return _metrics(case, answer, checks, _retrieved_docs(answer), cited)


def score_retrieval_case(case: EvalCase, answer: GroundedAnswer) -> CaseResult:
    """Retrieval-mode scoring: gate behaviour and document recall only."""
    e = case.expected
    status = answer.answer_status.value
    checks: list[CheckResult] = []
    if e.max_llm_calls == 0:
        checks.append(CheckResult("gate_refused", status == REFUSAL_STATUS, f"got '{status}'"))
    elif set(e.statuses) & ANSWER_STATUSES:
        checks.append(CheckResult("gate_passed", status == "retrieval_only", f"got '{status}'"))
    retrieved = _retrieved_docs(answer)
    hit = _retrieval_hit(case, retrieved)
    if hit is not None:
        checks.append(CheckResult("retrieval_hit", hit, f"retrieved {retrieved}"))
    return _metrics(case, answer, checks, retrieved, [])


# --------------------------------------------------------------- aggregation


@dataclass
class EvalSummary:
    n: int
    passed: int
    pass_rate: float
    by_category: dict[str, dict[str, int]]
    by_band: dict[str, dict[str, int]]
    refusal_precision: float | None
    refusal_recall: float | None
    total_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    mean_latency_ms: float
    p95_latency_ms: float
    failed_checks: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct * len(ordered)) - 1)
    return float(ordered[index])


def aggregate(results: list[CaseResult]) -> EvalSummary:
    n = len(results)
    passed = sum(1 for r in results if r.passed)

    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"n": 0, "passed": 0})
        bucket["n"] += 1
        bucket["passed"] += int(r.passed)

    by_band: dict[str, dict[str, int]] = {}
    for r in results:
        if r.status in ANSWER_STATUSES:
            bucket = by_band.setdefault(r.band, {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(r.passed)

    actual_refusals = [r for r in results if r.status == REFUSAL_STATUS]
    expected_refusals = [r for r in results if r.expected_refusal]
    true_refusals = sum(1 for r in actual_refusals if r.expected_refusal)
    precision = true_refusals / len(actual_refusals) if actual_refusals else None
    recall = true_refusals / len(expected_refusals) if expected_refusals else None

    latencies = [r.latency_ms for r in results]
    failed = Counter(check.name for r in results for check in r.failed_checks)

    return EvalSummary(
        n=n,
        passed=passed,
        pass_rate=round(passed / n, 4) if n else 0.0,
        by_category=by_category,
        by_band=by_band,
        refusal_precision=precision,
        refusal_recall=recall,
        total_cost_usd=round(sum(r.cost_usd for r in results), 6),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        mean_latency_ms=round(sum(latencies) / n, 1) if n else 0.0,
        p95_latency_ms=_percentile(latencies, 0.95),
        failed_checks=dict(failed),
    )


# ------------------------------------------------------------------- running


def run_eval(service: QueryService, cases: list[EvalCase], *, mode: Mode = "full") -> tuple[list[CaseResult], EvalSummary]:
    scorer = score_case if mode == "full" else score_retrieval_case
    results: list[CaseResult] = []
    for case in cases:
        answer = service.ask(case.query, doc_ids=case.doc_ids)
        results.append(scorer(case, answer))
    return results, aggregate(results)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_markdown(summary: EvalSummary, results: list[CaseResult], *, mode: str, model: str, prompt_version: str = "") -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# Eval report ({mode} mode)",
        "",
        f"Generated {stamp}. Model: `{model}`. Prompt: `{prompt_version or 'n/a'}`. Cases: {summary.n}.",
        "",
        "## Summary",
        "",
        f"- **Pass rate: {summary.passed}/{summary.n} ({_pct(summary.pass_rate)})**",
        f"- Refusal precision {_pct(summary.refusal_precision)}, refusal recall {_pct(summary.refusal_recall)}",
        f"- Cost ${summary.total_cost_usd:.4f} total ({summary.total_prompt_tokens} prompt + {summary.total_completion_tokens} completion tokens)",
        f"- Latency mean {summary.mean_latency_ms:.0f} ms, p95 {summary.p95_latency_ms:.0f} ms",
        "",
        "## Calibration by confidence band (answered / partial only)",
        "",
        "| band | n | correct | accuracy |",
        "|---|---|---|---|",
    ]
    for band in ("high", "medium", "low"):
        bucket = summary.by_band.get(band)
        if bucket:
            acc = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
            lines.append(f"| {band} | {bucket['n']} | {bucket['correct']} | {_pct(acc)} |")
    lines += ["", "## By category", "", "| category | passed | n |", "|---|---|---|"]
    for cat, bucket in sorted(summary.by_category.items()):
        lines.append(f"| {cat} | {bucket['passed']} | {bucket['n']} |")
    if summary.failed_checks:
        lines += ["", "## Failed checks", ""]
        for name, count in sorted(summary.failed_checks.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{name}`: {count}")
    lines += [
        "",
        "## Cases",
        "",
        "| id | category | status | conf | band | review | calls | cost | ms | result | failed checks |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        failed = "; ".join(f"{c.name} ({c.detail})" for c in r.failed_checks) or "-"
        lines.append(
            f"| {r.case_id} | {r.category} | {r.status} | {r.confidence:.2f} | {r.band} | {'yes' if r.needs_human_review else 'no'} | "
            f"{r.llm_calls} | ${r.cost_usd:.4f} | {r.latency_ms:.0f} | {'PASS' if r.passed else 'FAIL'} | {failed} |"
        )
    lines += ["", "## Answers", ""]
    for r in results:
        lines.append(f"- **{r.case_id}**: {r.answer_excerpt or '(no answer text)'}")
        if r.tools_called:
            lines.append(f"  - tools: {', '.join(r.tools_called)}")
        for m in r.missing_information:
            lines.append(f"  - missing information: {m}")
        for c in r.caveats:
            lines.append(f"  - caveat: {c}")
        for w in r.warnings:
            lines.append(f"  - warning: {w}")
    return "\n".join(lines) + "\n"


def write_reports(
    results: list[CaseResult], summary: EvalSummary, out_dir: Path, *, mode: str, model: str, prompt_version: str = ""
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{mode}_latest.json"
    md_path = out_dir / f"eval_{mode}_latest.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "prompt_version": prompt_version,
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(summary, results, mode=mode, model=model, prompt_version=prompt_version), encoding="utf-8")
    return json_path, md_path
