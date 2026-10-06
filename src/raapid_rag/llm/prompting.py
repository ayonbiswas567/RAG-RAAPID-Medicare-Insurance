"""Prompt assembly: system prompt loading, context blocks, user and repair messages.

Kept separate from the agent so the exact text the model sees can be unit-tested and so
prompt versions live as files (``prompts/system_v1.md``, ``system_v2.md``) that a reviewer
can diff.
"""

from __future__ import annotations

from pathlib import Path

from ..ingestion.models import Chunk


def load_system_prompt(version: str, prompts_dir: Path) -> str:
    path = Path(prompts_dir) / f"system_{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no system prompt for version '{version}' at {path}")
    return path.read_text(encoding="utf-8").strip()


def format_chunk(chunk: Chunk) -> str:
    header = f"[{chunk.chunk_id}] {chunk.doc_type} | {chunk.doc_title} | section: {chunk.section}"
    if chunk.date:
        header += f" | date: {chunk.date}"
    return f"{header}\n{chunk.text}"


def build_context_block(chunks: list[Chunk]) -> str:
    body = "\n\n".join(format_chunk(c) for c in chunks)
    return f"<context>\n{body}\n</context>"


def build_user_message(query: str, chunks: list[Chunk]) -> str:
    return (
        f"{build_context_block(chunks)}\n\n"
        f"<question>\n{query}\n</question>\n\n"
        "Answer using only the context above and any tool results. "
        "Return a single JSON object matching the schema."
    )


def build_repair_message(errors: list[str], valid_chunk_ids: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response failed validation:\n"
        f"{bullet}\n\n"
        f"Valid chunk ids for this question are: {', '.join(valid_chunk_ids)}.\n"
        "Return a corrected JSON object. Cite only these ids, copy evidence word-for-word from the "
        "cited chunk, and if the context does not support the answer set answer_status to "
        "\"insufficient_context\" and explain what is missing."
    )


def build_json_error_message(detail: str) -> str:
    return (
        f"Your previous response was not valid JSON matching the schema: {detail}\n"
        "Return only a single JSON object with the fields answer, answer_status, citations, "
        "missing_information, self_reported_confidence and reasoning_summary."
    )
