"""The reasoning loop: retrieved chunks in, validated structured answer out.

Sequence (decisions D6 and D7 in the engineering notes):

1. Build ``[system prompt, user message]`` where the user message carries the labelled
   context block and the question.  Register the tools and request strict JSON output.
2. Call the model.  If it asks for tools, execute each through the registry (validated,
   time-bounded, errors contained), append the results as ``tool`` messages, and loop.  At
   most ``max_tool_rounds`` rounds; after that the final answer is *forced* with
   ``tool_choice="none"``.
3. When the model answers, parse and validate in two layers: pydantic for shape, then
   :func:`validate_grounding` for truth (ids exist, evidence verbatim, status consistent).
4. On failure, append the model's output and a repair instruction naming every problem and
   the valid ids, and ask again.  At most ``max_repair_attempts`` times.
5. Return an :class:`AgentResult`.  ``model_output`` is ``None`` only when repairs were
   exhausted; the service layer then emits a deterministic error envelope.

The agent never raises for model misbehaviour.  It does propagate :class:`LLMError` from
the client, because "the provider is down" is a different failure from "the model produced
bad output" and is handled differently upstream.

``fault_injection`` corrupts the first final answer so the repair path can be demonstrated
live; it does nothing unless explicitly configured.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import ValidationError

from .config import Settings
from .guardrails import tool_induced_refusal
from .ingestion.models import Chunk
from .llm.client import ChatResult
from .llm.prompting import build_json_error_message, build_repair_message, build_user_message
from .schema import ModelOutput, ToolCallRecord, normalize_for_match, strict_json_schema, validate_grounding
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], *, tools=None, response_format=None, tool_choice=None) -> ChatResult: ...


@dataclass
class AgentResult:
    model_output: ModelOutput | None
    last_output: ModelOutput | None = None  # parsed but invalid; the service may salvage it
    tool_records: list[ToolCallRecord] = field(default_factory=list)
    repair_attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    fault_injected: bool = False
    llm_calls: int = 0
    forced_final: bool = False
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    latency_ms: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)


class GroundedAgent:
    def __init__(self, llm: ChatClient, tools: ToolRegistry, settings: Settings, system_prompt: str) -> None:
        self._llm = llm
        self._tools = tools
        self._settings = settings
        self._system_prompt = system_prompt
        self._response_format = {
            "type": "json_schema",
            "json_schema": {"name": "grounded_answer", "strict": True, "schema": strict_json_schema(ModelOutput)},
        }

    def run(self, query: str, chunks: list[Chunk]) -> AgentResult:
        if not chunks:
            raise ValueError("agent requires retrieved chunks but got none; the retrieval gate should have refused first")

        t0 = time.perf_counter()
        valid_ids = [c.chunk_id for c in chunks]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_message(query, chunks)},
        ]
        result = AgentResult(model_output=None, messages=messages)
        tool_rounds = 0
        fault_pending = bool(self._settings.fault_injection)
        tool_memo: dict[tuple[str, str], ToolCallRecord] = {}  # identical calls within one request run once

        while True:
            force_final = tool_rounds >= self._settings.max_tool_rounds
            reply = self._llm.chat(
                messages,
                tools=self._tools.openai_tools(),
                response_format=self._response_format,
                tool_choice="none" if force_final else None,
            )
            result.llm_calls += 1
            result.forced_final = result.forced_final or force_final
            _add_usage(result.usage, reply.usage)

            if reply.message.tool_calls and not force_final:
                tool_rounds += 1
                self._run_tools(reply, messages, result, tool_memo)
                continue

            content = reply.message.content or ""
            if fault_pending:
                content = _inject_fault(content, self._settings.fault_injection, valid_ids)
                result.fault_injected = True
                fault_pending = False
            result.raw_outputs.append(content)

            output, errors, parse_failure = self._parse_and_validate(content, chunks)
            if output is not None:
                result.last_output = output
            if output is not None and not errors:
                induced = tool_induced_refusal(query, output, result.tool_records, self._settings.tool_policy, self._settings.tool_refusal_topics)
                if induced:
                    errors = [induced]  # treat as a validation failure so the normal repair path runs
            if output is not None and not errors:
                result.model_output = output
                result.validation_errors = []
                break
            if output is not None and result.tool_records:
                errors = _annotate_tool_result_evidence(errors, output, result.tool_records)

            result.validation_errors = errors
            if result.repair_attempts >= self._settings.max_repair_attempts:
                logger.warning("model output still invalid after %d repair attempt(s): %s", result.repair_attempts, errors)
                break

            result.repair_attempts += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": build_json_error_message(errors[0]) if parse_failure else build_repair_message(errors, valid_ids),
            })

        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # ------------------------------------------------------------------ helpers

    def _run_tools(
        self, reply: ChatResult, messages: list[dict[str, Any]], result: AgentResult, memo: dict[tuple[str, str], ToolCallRecord]
    ) -> None:
        messages.append({
            "role": "assistant",
            "content": reply.message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in reply.message.tool_calls
            ],
        })
        for tc in reply.message.tool_calls:
            key = (tc.name, _canonical_arguments(tc.arguments))
            if key in memo:
                # Observed live: the model asked days_between the same question three times in one
                # run. Reuse the earlier result; the transcript still gets a tool message per call id.
                record = memo[key].model_copy(update={"cached": True})
            else:
                record = self._tools.dispatch(tc.name, tc.arguments, timeout_seconds=self._settings.tool_timeout_seconds)
                if record.error is None:
                    memo[key] = record  # failures are not memoised; a retry may succeed
            result.tool_records.append(record)
            payload: Any = record.result if record.error is None else {"error": record.error}
            if record.cached:
                note = "identical call already made in this conversation; result repeated, do not call again"
                payload = {**payload, "note_from_system": note} if isinstance(payload, dict) else {"result": payload, "note_from_system": note}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(payload, default=str)})

    @staticmethod
    def _parse_and_validate(content: str, chunks: list[Chunk]) -> tuple[ModelOutput | None, list[str], bool]:
        """Returns (output_or_None, errors, parse_failure)."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return None, [f"response was not valid JSON ({exc.msg} at position {exc.pos})"], True
        try:
            output = ModelOutput.model_validate(data)
        except ValidationError as exc:
            errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            return None, [f"response did not match the JSON schema: {'; '.join(errors)}"], True
        return output, validate_grounding(output, chunks), False


def _annotate_tool_result_evidence(errors: list[str], output: ModelOutput, records: list[ToolCallRecord]) -> list[str]:
    """Name the real cause when a rejected quote was copied from a tool result.

    Observed live: the model cited a policy chunk with the eligibility tool's own text
    ("Coverage not active on the date of service.") and repeated the mistake through two
    generic repair rounds. Telling it *where* the text came from fixes the loop.
    """
    tool_texts = [(r.name, normalize_for_match(json.dumps(r.result, default=str))) for r in records if r.result is not None]
    annotated = list(errors)
    for i, cit in enumerate(output.citations):
        if not any(f"citations[{i}] evidence is not a verbatim quote" in e for e in errors):
            continue
        needle = normalize_for_match(cit.evidence)
        for name, text in tool_texts:
            if needle and needle in text:
                annotated.append(
                    f"citations[{i}] evidence {cit.evidence[:80]!r} is text from the '{name}' tool result. "
                    "Tool results must not be cited: remove this citation and report the tool result in the answer instead."
                )
                break
    return annotated


def _canonical_arguments(arguments: str) -> str:
    """Key for memoisation: JSON with sorted keys so argument order does not defeat the cache."""
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return arguments or ""


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0))


def _inject_fault(content: str, mode: str, valid_ids: list[str]) -> str:
    """Deliberately corrupt a (probably valid) model response. Demo-only."""
    if mode == "malformed_json":
        return content[: max(1, len(content) // 2)]
    if mode == "bad_citation":
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content[: max(1, len(content) // 2)]
        fake = {"chunk_id": "note_999#c99", "evidence": "fabricated evidence"}
        citations = data.get("citations") or []
        data["citations"] = [fake] + citations[1:] if citations else [fake]
        return json.dumps(data)
    return content
