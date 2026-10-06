"""Tool registry: OpenAI function schemas out, validated and time-bounded execution in.

Why a registry rather than calling functions directly from the agent:

* The agent stays generic.  It asks "what tools exist?" and "run this one"; it never imports
  a tool module.  Adding a tool is one ``register`` call.
* Every call gets the same treatment: argument validation against the declared schema, a
  hard timeout, and exception containment.  A tool failure becomes a structured
  :class:`ToolCallRecord` with ``error`` set, never an exception in the request path.

Timeout caveat: Python threads cannot be killed.  A tool that exceeds its timeout is
*abandoned* (the record says so) but its thread runs to completion in the background.  For
mock tools that is harmless; a real network tool should also carry its own socket timeout.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from ..schema import ToolCallRecord
from .calculator import calculate, days_between
from .eligibility import check_eligibility
from .icd10 import icd10_lookup


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema, strict-mode style (all required, closed)
    fn: Callable[..., Any]


class ToolRegistry:
    def __init__(self, max_workers: int = 8) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool")

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "strict": True,
                },
            }
            for spec in self._tools.values()
        ]

    def dispatch(self, name: str, arguments_json: str, *, timeout_seconds: float) -> ToolCallRecord:
        t0 = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolCallRecord(name=name, arguments={}, error=f"unknown tool '{name}'; available: {', '.join(self.names)}")

        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolCallRecord(name=name, arguments={}, error=f"could not parse tool arguments as JSON: {exc}")

        problem = _validate_arguments(spec, arguments)
        if problem:
            return ToolCallRecord(name=name, arguments=arguments, error=problem)

        future = self._executor.submit(spec.fn, **arguments)
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeout:
            return ToolCallRecord(
                name=name, arguments=arguments, timed_out=True,
                error=f"tool '{name}' timed out after {timeout_seconds}s and was abandoned",
                latency_ms=_ms(t0),
            )
        except Exception as exc:  # noqa: BLE001 - containment is the point
            return ToolCallRecord(
                name=name, arguments=arguments,
                error=f"tool '{name}' raised {type(exc).__name__}: {exc}",
                latency_ms=_ms(t0),
            )
        return ToolCallRecord(name=name, arguments=arguments, result=result, latency_ms=_ms(t0))


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    declared = set(spec.parameters.get("properties", {}))
    required = set(spec.parameters.get("required", []))
    missing = sorted(required - set(arguments))
    unexpected = sorted(set(arguments) - declared)
    if missing:
        return f"missing required argument(s) for '{spec.name}': {', '.join(missing)}"
    if unexpected:
        return f"unexpected argument(s) for '{spec.name}': {', '.join(unexpected)}"
    return None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


# ------------------------------------------------------------------ defaults


def _params(**props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="icd10_lookup",
        description="Look up an ICD-10-CM diagnosis code (for example 'E11.22') or search diagnosis descriptions by term (for example 'heart failure'). Returns the description and the illustrative HCC category and hierarchy. Use it whenever the question involves a code, an HCC, or mapping a documented condition to a code.",
        parameters=_params(query={"type": "string", "description": "An ICD-10-CM code such as 'E11.22', or a diagnosis term such as 'heart failure'."}),
        fn=icd10_lookup,
    ))
    reg.register(ToolSpec(
        name="check_eligibility",
        description="Check whether a member had active plan coverage on a given date of service. Returns ACTIVE, INACTIVE (with termination details) or NOT_FOUND. Use it whenever the question asks about eligibility, coverage, or whether an encounter can be submitted.",
        parameters=_params(
            member_id={"type": "string", "description": "Member id exactly as documented, for example 'M-TEST-1002'."},
            date_of_service={"type": "string", "description": "Date of service in ISO format YYYY-MM-DD. For inpatient stays use the discharge date."},
        ),
        fn=check_eligibility,
    ))
    reg.register(ToolSpec(
        name="calculate",
        description="Evaluate a numeric expression exactly (e.g. BMI, dose totals, differences). Supports + - * / // % **, parentheses, abs, round, min, max. Use it instead of doing arithmetic in your head.",
        parameters=_params(expression={"type": "string", "description": "A numeric expression such as '84 / (1.65 ** 2)' or 'round(7.6 - 8.2, 1)'."}),
        fn=calculate,
    ))
    reg.register(ToolSpec(
        name="days_between",
        description="Number of days from start_date to end_date (ISO dates). Use for length of stay, submission windows, and any date difference.",
        parameters=_params(
            start_date={"type": "string", "description": "Start date, YYYY-MM-DD."},
            end_date={"type": "string", "description": "End date, YYYY-MM-DD."},
        ),
        fn=days_between,
    ))
    return reg
