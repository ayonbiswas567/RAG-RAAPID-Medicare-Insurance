"""Tool registry tests: OpenAI schema export, dispatch, timeouts, error containment."""

import time

from raapid_rag.schema import ToolCallRecord
from raapid_rag.tools.registry import ToolRegistry, ToolSpec, default_registry

EMPTY_PARAMS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def test_default_registry_exposes_four_strict_openai_tools():
    tools = default_registry().openai_tools()
    assert {t["function"]["name"] for t in tools} == {"icd10_lookup", "check_eligibility", "calculate", "days_between"}
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["strict"] is True
        assert fn["description"]
        params = fn["parameters"]
        assert params["additionalProperties"] is False
        assert sorted(params["required"]) == sorted(params["properties"].keys())


def test_dispatch_runs_tool_and_records_arguments_and_latency():
    rec = default_registry().dispatch("icd10_lookup", '{"query": "E11.22"}', timeout_seconds=5)
    assert isinstance(rec, ToolCallRecord)
    assert rec.name == "icd10_lookup"
    assert rec.arguments == {"query": "E11.22"}
    assert rec.error is None and rec.timed_out is False
    assert rec.result["found"] is True
    assert rec.latency_ms >= 0


def test_dispatch_unknown_tool_returns_error_record():
    rec = default_registry().dispatch("not_a_tool", "{}", timeout_seconds=1)
    assert rec.result is None
    assert "unknown tool" in rec.error


def test_dispatch_malformed_json_arguments_returns_error():
    rec = default_registry().dispatch("icd10_lookup", "{not json", timeout_seconds=1)
    assert "arguments" in rec.error and rec.result is None


def test_dispatch_missing_required_argument_returns_error():
    rec = default_registry().dispatch("icd10_lookup", "{}", timeout_seconds=1)
    assert "query" in rec.error


def test_dispatch_unexpected_argument_returns_error():
    rec = default_registry().dispatch("icd10_lookup", '{"query": "E11.22", "extra": 1}', timeout_seconds=1)
    assert "extra" in rec.error


def test_dispatch_times_out_and_reports_it():
    reg = ToolRegistry()
    reg.register(ToolSpec("sleepy", "sleeps", EMPTY_PARAMS, lambda: time.sleep(0.5) or {"ok": True}))
    t0 = time.perf_counter()
    rec = reg.dispatch("sleepy", "{}", timeout_seconds=0.05)
    assert rec.timed_out is True
    assert "timed out" in rec.error
    assert time.perf_counter() - t0 < 0.4  # returned at the timeout, not after the sleep


def test_dispatch_contains_tool_exceptions_without_traceback():
    def boom():
        raise RuntimeError("database unreachable")

    reg = ToolRegistry()
    reg.register(ToolSpec("boom", "raises", EMPTY_PARAMS, boom))
    rec = reg.dispatch("boom", "{}", timeout_seconds=1)
    assert rec.result is None
    assert "RuntimeError" in rec.error and "database unreachable" in rec.error
    assert "Traceback" not in rec.error


def test_register_rejects_duplicate_names():
    reg = ToolRegistry()
    reg.register(ToolSpec("a", "d", EMPTY_PARAMS, lambda: 1))
    try:
        reg.register(ToolSpec("a", "d", EMPTY_PARAMS, lambda: 2))
    except ValueError as exc:
        assert "a" in str(exc)
    else:
        raise AssertionError("duplicate registration should raise")


def test_record_serialises_to_json():
    rec = default_registry().dispatch("days_between", '{"start_date": "2025-03-02", "end_date": "2025-03-07"}', timeout_seconds=1)
    payload = rec.model_dump(mode="json")
    assert payload["result"]["days"] == 5
    assert set(payload) >= {"name", "arguments", "result", "error", "latency_ms", "timed_out"}
