"""Command-line surface.

Deliberately thin: parse arguments, build the same objects the API and UI use, print.
No logic lives here.  ``--json`` on every command makes the CLI scriptable (the eval runner
and the tests use it); the default output is a human-readable table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import DEFAULT_CACHE_DIR, PROJECT_ROOT, Settings, get_settings
from .evaluation import load_eval_set, run_eval, write_reports
from .guardrails import InvalidQueryError
from .ingestion.embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .ingestion.pipeline import build_index
from .retrieval.retriever import Retriever
from .schema import GroundedAnswer
from .service import QueryService

app = typer.Typer(help="RAAPID grounded GenAI micro-service", no_args_is_help=True, add_completion=False)
console = Console()

EmbedderOpt = typer.Option("st", "--embedder", help="'st' = local sentence-transformers model (default); 'hashing' = fast test embedder.")
CacheDirOpt = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", help="Where prebuilt indexes are stored, keyed by corpus fingerprint.")
NoCacheOpt = typer.Option(False, "--no-cache", help="Always rebuild; never read or write the index cache.")
JsonOpt = typer.Option(False, "--json", help="Machine-readable output.")


def make_embedder(kind: str, settings: Settings) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise typer.BadParameter(f"unknown embedder '{kind}' (use 'st' or 'hashing')")


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    embedder: str = EmbedderOpt,
    cache_dir: Path = CacheDirOpt,
    no_cache: bool = NoCacheOpt,
    json_out: bool = JsonOpt,
) -> None:
    """Load the corpus, chunk it, embed it, and build (or load) the hybrid index."""
    settings = get_settings()
    _configure_logging(settings)
    _, report = build_index(settings, make_embedder(embedder, settings), cache_dir=None if no_cache else cache_dir)

    if json_out:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        return

    table = Table(title="Index build report")
    table.add_column("field")
    table.add_column("value")
    for key, value in report.to_dict().items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language question."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override RETRIEVAL_TOP_K."),
    embedder: str = EmbedderOpt,
    cache_dir: Path = CacheDirOpt,
    no_cache: bool = NoCacheOpt,
    json_out: bool = JsonOpt,
) -> None:
    """Retrieval only: show which chunks would be sent to the model and whether the gate passes."""
    settings = get_settings()
    _configure_logging(settings)
    store, _ = build_index(settings, make_embedder(embedder, settings), cache_dir=None if no_cache else cache_dir)
    result = Retriever(store, settings).retrieve(query, top_k=top_k)

    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return

    table = Table(title=f"Retrieval for: {query!r}")
    for col in ("#", "chunk_id", "section", "dense", "bm25", "fused"):
        table.add_column(col)
    for i, c in enumerate(result.chunks, start=1):
        table.add_row(str(i), c.chunk.chunk_id, c.chunk.section, f"{c.dense_score:.3f}", f"{c.bm25_score:.2f}", f"{c.fused_score:.4f}")
    console.print(table)
    verdict = "PASS" if result.passes_gate else "REFUSE"
    console.print(
        f"gate: [bold]{verdict}[/bold] ({result.gate_reason})  "
        f"best_dense={result.best_dense_score:.3f} best_bm25={result.best_bm25_score:.2f}  "
        f"context_tokens={result.context_tokens} dropped={result.dropped_chunk_ids} latency={result.latency_ms}ms"
    )
    for c in result.chunks[:3]:
        snippet = c.chunk.text.replace("\n", " ")
        console.print(f"\n[dim]{c.chunk.chunk_id}[/dim] {snippet[:240]}{'...' if len(snippet) > 240 else ''}")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Natural-language question."),
    doc_id: Optional[list[str]] = typer.Option(None, "--doc-id", help="Restrict retrieval to these document ids (repeatable)."),
    embedder: str = EmbedderOpt,
    cache_dir: Path = CacheDirOpt,
    no_cache: bool = NoCacheOpt,
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the language model; return retrieved evidence only."),
    json_out: bool = JsonOpt,
) -> None:
    """Full pipeline: retrieve, gate, reason with tools, validate, and return the grounded JSON answer."""
    settings = get_settings()
    _configure_logging(settings)
    service = QueryService.from_settings(
        settings, embedder=make_embedder(embedder, settings), cache_dir=None if no_cache else cache_dir, use_llm=not no_llm,
    )
    try:
        answer = service.ask(query, doc_ids=doc_id or None)
    except InvalidQueryError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_out:
        typer.echo(answer.model_dump_json(indent=2))
        return
    _print_answer(answer)


@app.command("eval")
def evaluate(
    mode: str = typer.Option("full", "--mode", help="'full' runs the model; 'retrieval' scores gate and recall only (no key needed)."),
    eval_set: Path = typer.Option(PROJECT_ROOT / "eval" / "eval_set.json", "--eval-set"),
    out_dir: Path = typer.Option(PROJECT_ROOT / "docs" / "runs", "--out-dir", help="Where eval_<mode>_latest.json/.md are written."),
    min_pass_rate: float = typer.Option(0.8, "--min-pass-rate", help="Exit code 1 when the pass rate is below this."),
    embedder: str = EmbedderOpt,
    cache_dir: Path = CacheDirOpt,
    no_cache: bool = NoCacheOpt,
    json_out: bool = JsonOpt,
) -> None:
    """Run the eval set through the full service and report pass/fail, calibration, cost and latency."""
    if mode not in ("full", "retrieval"):
        raise typer.BadParameter("mode must be 'full' or 'retrieval'")
    settings = get_settings()
    _configure_logging(settings)
    service = QueryService.from_settings(
        settings, embedder=make_embedder(embedder, settings), cache_dir=None if no_cache else cache_dir, use_llm=(mode == "full"),
    )
    cases = load_eval_set(eval_set)
    results, summary = run_eval(service, cases, mode=mode)  # type: ignore[arg-type]
    json_path, md_path = write_reports(
        results, summary, out_dir, mode=mode, model=settings.llm_model if mode == "full" else "none",
        prompt_version=settings.prompt_version if mode == "full" else "",
    )

    if json_out:
        typer.echo(json.dumps({"summary": summary.to_dict(), "results": [r.to_dict() for r in results]}, indent=2))
    else:
        table = Table(title=f"Eval ({mode} mode): {summary.passed}/{summary.n} passed")
        for col in ("id", "status", "conf", "band", "review", "calls", "cost", "ms", "result", "failed checks"):
            table.add_column(col)
        for r in results:
            table.add_row(
                r.case_id, r.status, f"{r.confidence:.2f}", r.band, "yes" if r.needs_human_review else "no", str(r.llm_calls),
                f"${r.cost_usd:.4f}", f"{r.latency_ms:.0f}", "PASS" if r.passed else "FAIL",
                "; ".join(c.name for c in r.failed_checks) or "-",
            )
        console.print(table)
        console.print(
            f"pass rate {summary.pass_rate:.0%} | refusal precision {summary.refusal_precision} recall {summary.refusal_recall} | "
            f"cost ${summary.total_cost_usd:.4f} | latency mean {summary.mean_latency_ms:.0f} ms p95 {summary.p95_latency_ms:.0f} ms | "
            f"bands {summary.by_band}"
        )
        console.print(f"reports: {json_path}  {md_path}")

    raise typer.Exit(code=0 if summary.pass_rate >= min_pass_rate else 1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    workers: int = typer.Option(1, "--workers", help="Processes. Each loads its own copy of the index and embedding model."),
) -> None:
    """Start the HTTP API (FastAPI via uvicorn). Docs at /docs, readiness at /health."""
    import uvicorn

    uvicorn.run("raapid_rag.api:app", host=host, port=port, workers=workers, log_level=get_settings().log_level.lower())


@app.command()
def ui(port: int = typer.Option(8501, "--port")) -> None:
    """Start the Streamlit UI (imports the service in-process; no separate API needed)."""
    import subprocess
    import sys

    script = PROJECT_ROOT / "src" / "raapid_rag" / "ui" / "app.py"
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)]))


def _print_answer(answer: GroundedAnswer) -> None:
    review = "YES" if answer.needs_human_review else "no"
    console.print(
        f"[bold]status:[/bold] {answer.answer_status.value}   "
        f"[bold]confidence:[/bold] {answer.confidence:.2f} ({answer.confidence_band.value})   "
        f"[bold]needs human review:[/bold] {review}"
    )
    if answer.answer:
        console.print(f"\n[bold]answer:[/bold] {answer.answer}")
    if answer.citations:
        table = Table(title="citations")
        for col in ("chunk_id", "section", "verified", "evidence"):
            table.add_column(col)
        for c in answer.citations:
            table.add_row(c.chunk_id, c.section, "yes" if c.verified else "NO", c.evidence[:120])
        console.print(table)
    if answer.missing_information:
        console.print("[bold]missing information:[/bold]")
        for m in answer.missing_information:
            console.print(f"  - {m}")
    if answer.caveats:
        console.print("[bold]caveats:[/bold]")
        for c in answer.caveats:
            console.print(f"  - {c}")
    if answer.tool_calls:
        console.print("[bold]tool calls:[/bold]")
        for t in answer.tool_calls:
            flag = " (cached)" if t.cached else (" (ERROR: " + t.error + ")" if t.error else "")
            console.print(f"  - {t.name}({json.dumps(t.arguments)}){flag}")
    for w in answer.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    for e in answer.errors:
        console.print(f"[red]error:[/red] {e}")
    trace = answer.trace
    usage = trace.get("usage") or {}
    console.print(
        f"\n[dim]trace: prompt {trace.get('prompt_version')} | llm calls {trace.get('llm_calls', 0)} | "
        f"tokens {usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} | "
        f"est ${(trace.get('est_cost_usd') or 0):.4f} | {trace.get('total_latency_ms', 0)} ms | "
        f"anchored {trace.get('retrieval', {}).get('anchored_docs', [])} | cache_hit {trace.get('cache_hit')}[/dim]"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
