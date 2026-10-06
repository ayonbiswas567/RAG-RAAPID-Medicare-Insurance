"""Agent loop tests with a scripted fake LLM and the real tool registry.

The fake returns pre-written ChatResults in order and records every request, so the tests
can assert on exactly what the model was shown (context blocks, tool results, repair
instructions) as well as on the outcome.
"""

import copy
import json

import pytest

from raapid_rag.agent import AgentResult, GroundedAgent
from raapid_rag.config import PROJECT_ROOT, Settings
from raapid_rag.ingestion.models import Chunk
from raapid_rag.llm.client import ChatMessage, ChatResult, ToolCall
from raapid_rag.llm.prompting import build_context_block, load_system_prompt
from raapid_rag.tools.registry import default_registry

CHUNKS = [
    Chunk(
        chunk_id="note_001#c05", doc_id="note_001", doc_type="clinical_note",
        doc_title="Endocrinology Progress Note", section="Laboratory Results",
        text="HbA1c 8.2% (collected 2025-03-12). eGFR 52 mL/min/1.73m2 (2025-03-12).", index=5,
        metadata={"date_of_service": "2025-03-14"},
    ),
    Chunk(
        chunk_id="note_001#c06", doc_id="note_001", doc_type="clinical_note",
        doc_title="Endocrinology Progress Note", section="Assessment",
        text="1. Type 2 diabetes mellitus with diabetic chronic kidney disease (E11.22).", index=6,
        metadata={"date_of_service": "2025-03-14"},
    ),
]

VALID = {
    "answer": "The most recent HbA1c was 8.2%, collected on 2025-03-12.",
    "answer_status": "answered",
    "citations": [{"chunk_id": "note_001#c05", "evidence": "HbA1c 8.2% (collected 2025-03-12)."}],
    "missing_information": [],
    "caveats": [],
    "self_reported_confidence": 0.9,
    "reasoning_summary": "Stated directly in the lab results.",
}


def final(payload) -> ChatResult:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return ChatResult(message=ChatMessage(content=content, tool_calls=[]), finish_reason="stop",
                      usage={"prompt_tokens": 100, "completion_tokens": 50}, latency_ms=1.0, model="fake")


def tool_call(name, args, call_id="call_1") -> ChatResult:
    return ChatResult(message=ChatMessage(content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))]),
                      finish_reason="tool_calls", usage={"prompt_tokens": 100, "completion_tokens": 20}, latency_ms=1.0, model="fake")


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, response_format=None, tool_choice=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "response_format": response_format, "tool_choice": tool_choice})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, openai_api_key="sk-test", **overrides)


def make_agent(llm, **overrides) -> GroundedAgent:
    s = settings(**overrides)
    return GroundedAgent(llm=llm, tools=default_registry(), settings=s, system_prompt="SYSTEM PROMPT v-test")


# ------------------------------------------------------------------ prompting


def test_context_block_labels_each_chunk_with_id_and_metadata():
    block = build_context_block(CHUNKS)
    assert "[note_001#c05]" in block and "[note_001#c06]" in block
    assert "Laboratory Results" in block and "Endocrinology Progress Note" in block and "2025-03-14" in block
    assert "HbA1c 8.2%" in block
    assert block.strip().startswith("<context>") and block.strip().endswith("</context>")


def test_system_prompts_load_by_version():
    prompts_dir = PROJECT_ROOT / "src" / "raapid_rag" / "llm" / "prompts"
    v1 = load_system_prompt("v1", prompts_dir)
    v2 = load_system_prompt("v2", prompts_dir)
    assert v1 != v2
    assert "verbatim" in v2.lower() and "insufficient_context" in v2
    with pytest.raises(FileNotFoundError):
        load_system_prompt("v9", prompts_dir)


# ---------------------------------------------------------------- happy path


def test_happy_path_returns_validated_output_and_shows_context_to_model():
    llm = FakeLLM([final(VALID)])
    result = make_agent(llm).run("What was the most recent HbA1c?", CHUNKS)
    assert isinstance(result, AgentResult)
    assert result.model_output is not None
    assert result.model_output.answer_status == "answered"
    assert result.repair_attempts == 0 and result.validation_errors == []
    assert result.llm_calls == 1
    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT v-test"}
    assert "[note_001#c05]" in messages[1]["content"] and "What was the most recent HbA1c?" in messages[1]["content"]
    assert llm.calls[0]["response_format"]["type"] == "json_schema"
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} >= {"icd10_lookup", "check_eligibility"}
    assert result.usage == {"prompt_tokens": 100, "completion_tokens": 50}


# ----------------------------------------------------------------- tool loop


def test_tool_call_is_executed_and_result_fed_back():
    llm = FakeLLM([tool_call("icd10_lookup", {"query": "E11.22"}, "call_abc"), final(VALID)])
    result = make_agent(llm).run("Which HCC does E11.22 map to?", CHUNKS)
    assert result.llm_calls == 2
    assert len(result.tool_records) == 1
    assert result.tool_records[0].name == "icd10_lookup" and result.tool_records[0].result["found"] is True

    second = llm.calls[1]["messages"]
    assistant = second[-2]
    tool_msg = second[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0] == {"id": "call_abc", "type": "function", "function": {"name": "icd10_lookup", "arguments": '{"query": "E11.22"}'}}
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_abc"
    assert json.loads(tool_msg["content"])["hcc_category"] == "Diabetes with Chronic Complications"


def test_tool_error_is_returned_to_model_not_raised():
    llm = FakeLLM([tool_call("not_a_tool", {}), final(VALID)])
    result = make_agent(llm).run("q", CHUNKS)
    assert result.tool_records[0].error and "unknown tool" in result.tool_records[0].error
    tool_msg = llm.calls[1]["messages"][-1]
    assert "error" in json.loads(tool_msg["content"])
    assert result.model_output is not None


def test_tool_rounds_are_bounded_and_final_answer_is_forced():
    calls = [tool_call("icd10_lookup", {"query": "E11.22"}, f"call_{i}") for i in range(2)]
    llm = FakeLLM(calls + [final(VALID)])
    result = make_agent(llm, max_tool_rounds=2).run("q", CHUNKS)
    assert result.llm_calls == 3
    assert result.forced_final is True
    assert llm.calls[-1]["tool_choice"] == "none"
    assert result.model_output is not None


# ---------------------------------------------------------------- repair path


def test_bad_citation_triggers_repair_with_valid_ids_listed():
    bad = {**VALID, "citations": [{"chunk_id": "note_009#c02", "evidence": "HbA1c 8.2%"}]}
    llm = FakeLLM([final(bad), final(VALID)])
    result = make_agent(llm).run("q", CHUNKS)
    assert result.repair_attempts == 1
    assert result.model_output is not None and result.validation_errors == []
    repair_msgs = llm.calls[1]["messages"]
    assert repair_msgs[-2]["role"] == "assistant" and "note_009#c02" in repair_msgs[-2]["content"]
    assert repair_msgs[-1]["role"] == "user"
    assert "note_009#c02" in repair_msgs[-1]["content"] and "note_001#c05" in repair_msgs[-1]["content"]


def test_malformed_json_triggers_repair():
    llm = FakeLLM([final('{"answer": "oops", "answer_status": '), final(VALID)])
    result = make_agent(llm).run("q", CHUNKS)
    assert result.repair_attempts == 1
    assert "JSON" in llm.calls[1]["messages"][-1]["content"]
    assert result.model_output is not None


def test_non_verbatim_evidence_triggers_repair():
    bad = {**VALID, "citations": [{"chunk_id": "note_001#c05", "evidence": "The A1c was 8.2 percent."}]}
    llm = FakeLLM([final(bad), final(VALID)])
    result = make_agent(llm).run("q", CHUNKS)
    assert result.repair_attempts == 1
    assert "verbatim" in llm.calls[1]["messages"][-1]["content"]


def test_evidence_copied_from_a_tool_result_is_named_as_such_in_the_repair():
    policy = Chunk(
        chunk_id="policy_002#c03", doc_id="policy_002", doc_type="payer_policy", doc_title="RA-110",
        section="3. Member Eligibility Verification", text="3.1 Before any diagnosis is submitted, the member must be verified as actively enrolled.", index=3,
    )
    bad = {**VALID, "citations": [
        {"chunk_id": "note_001#c05", "evidence": "HbA1c 8.2% (collected 2025-03-12)."},
        {"chunk_id": "policy_002#c03", "evidence": "Coverage not active on the date of service."},  # from the tool, not the chunk
    ]}
    llm = FakeLLM([tool_call("check_eligibility", {"member_id": "M-TEST-1004", "date_of_service": "2025-04-02"}), final(bad), final(VALID)])
    result = make_agent(llm).run("q", CHUNKS + [policy])
    assert result.repair_attempts == 1
    repair = llm.calls[2]["messages"][-1]["content"]
    assert "tool result" in repair and "must not be cited" in repair
    assert result.model_output is not None


def test_exhausted_repairs_return_no_output_with_errors():
    bad = {**VALID, "citations": [{"chunk_id": "note_009#c02", "evidence": "x"}]}
    llm = FakeLLM([final(bad)] * 5)
    result = make_agent(llm, max_repair_attempts=2).run("q", CHUNKS)
    assert result.model_output is None
    assert result.repair_attempts == 2
    assert result.llm_calls == 3
    assert any("note_009#c02" in e for e in result.validation_errors)
    assert len(result.raw_outputs) == 3
    # The last parsed-but-invalid output is kept so the service can try to salvage it.
    assert result.last_output is not None and result.last_output.citations[0].chunk_id == "note_009#c02"


def test_identical_tool_calls_within_a_run_are_memoised():
    calls = [tool_call("icd10_lookup", {"query": "E11.22"}, "call_1"), tool_call("icd10_lookup", {"query": "E11.22"}, "call_2"), final(VALID)]
    llm = FakeLLM(calls)
    result = make_agent(llm).run("q", CHUNKS)
    assert len(result.tool_records) == 2
    assert result.tool_records[0].cached is False and result.tool_records[1].cached is True
    assert result.tool_records[1].result == result.tool_records[0].result
    # The second tool message still carries the second call id so the transcript stays valid,
    # and tells the model the result is a repeat so it stops re-asking.
    second_tool_msg = llm.calls[2]["messages"][-1]
    assert second_tool_msg["tool_call_id"] == "call_2"
    assert "already made" in second_tool_msg["content"]
    assert "already made" not in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("mode", ["bad_citation", "malformed_json"])
def test_fault_injection_corrupts_first_response_exactly_once(mode):
    llm = FakeLLM([final(VALID), final(VALID)])
    result = make_agent(llm, fault_injection=mode).run("q", CHUNKS)
    assert result.fault_injected is True
    assert result.repair_attempts == 1
    assert result.model_output is not None and result.model_output.answer_status == "answered"


TOOL_INDUCED_REFUSAL = {
    "answer": "", "answer_status": "insufficient_context", "citations": [],
    "missing_information": ["The context does not say whether the diagnosis can be submitted due to the member's inactive coverage."],
    "caveats": [], "self_reported_confidence": 0.3,
    "reasoning_summary": "Guidelines say it cannot be coded, but coverage was inactive on the date of service.",
}
ELIGIBILITY_CALL = tool_call("check_eligibility", {"member_id": "M-TEST-1004", "date_of_service": "2025-04-02"}, "call_e")


def test_refusal_caused_by_an_unrelated_tool_result_is_repaired():
    # A coding question: the model calls the eligibility tool anyway, then refuses because of coverage.
    llm = FakeLLM([ELIGIBILITY_CALL, final(TOOL_INDUCED_REFUSAL), final(VALID)])
    result = make_agent(llm).run("Can 'possible rheumatoid arthritis' be coded as rheumatoid arthritis?", CHUNKS)
    assert result.repair_attempts == 1
    assert result.model_output is not None and result.model_output.answer_status == "answered"
    repair = llm.calls[2]["messages"][-1]["content"]
    assert "check_eligibility" in repair and "caveats" in repair


def test_refusal_is_kept_when_the_question_did_ask_about_eligibility():
    llm = FakeLLM([ELIGIBILITY_CALL, final(TOOL_INDUCED_REFUSAL)])
    result = make_agent(llm).run("Was member M-TEST-1004 eligible for coverage on 2025-04-02?", CHUNKS)
    assert result.repair_attempts == 0
    assert result.model_output is not None and result.model_output.answer_status == "insufficient_context"


def test_refusal_is_kept_when_no_tool_was_involved():
    llm = FakeLLM([final(TOOL_INDUCED_REFUSAL)])
    result = make_agent(llm).run("Can 'possible rheumatoid arthritis' be coded as rheumatoid arthritis?", CHUNKS)
    assert result.repair_attempts == 0 and result.model_output.answer_status == "insufficient_context"


def test_agent_refuses_to_run_without_chunks():
    with pytest.raises(ValueError, match="chunks"):
        make_agent(FakeLLM([])).run("q", [])
