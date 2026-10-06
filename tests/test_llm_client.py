"""LLM client tests: response normalisation, retries, error classification, accounting.

A fake OpenAI client (duck-typed ``chat.completions.create``) is injected so no network is
used. The fake mimics the SDK's response object shape, which is the only thing the wrapper
depends on.
"""

from types import SimpleNamespace

import httpx
import openai
import pytest

from raapid_rag.config import Settings
from raapid_rag.llm.client import (
    ChatResult,
    LLMClient,
    LLMError,
    LLMRequestError,
    LLMUnavailable,
    ToolCall,
    estimate_cost_usd,
)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, openai_api_key="sk-test", **overrides)


def sdk_response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5, finish_reason="stop"):
    message = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(id=tc["id"], function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]))
            for tc in (tool_calls or [])
        ] or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="gpt-4o-mini-2024-07-18",
    )


class FakeOpenAI:
    """Scripted stand-in for openai.OpenAI. Each script entry is a response or an exception."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rate_limit() -> openai.RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


def auth_error() -> openai.AuthenticationError:
    resp = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.AuthenticationError("bad key", response=resp, body=None)


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_chat_returns_normalised_result_and_accounts_usage():
    fake = FakeOpenAI([sdk_response(content='{"answer": "hi"}')])
    client = LLMClient(settings(), client=fake, backoff_base_seconds=0)
    result = client.chat(MESSAGES)
    assert isinstance(result, ChatResult)
    assert result.message.content == '{"answer": "hi"}'
    assert result.message.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert result.latency_ms >= 0
    assert client.usage.prompt_tokens == 10 and client.usage.completion_tokens == 5 and client.usage.calls == 1


def test_chat_normalises_tool_calls():
    fake = FakeOpenAI([sdk_response(tool_calls=[{"id": "call_1", "name": "icd10_lookup", "arguments": '{"query": "E11.22"}'}], finish_reason="tool_calls")])
    result = LLMClient(settings(), client=fake, backoff_base_seconds=0).chat(MESSAGES)
    assert result.message.content is None
    assert result.message.tool_calls == [ToolCall(id="call_1", name="icd10_lookup", arguments='{"query": "E11.22"}')]


def test_request_carries_model_temperature_tools_and_response_format():
    fake = FakeOpenAI([sdk_response(content="{}")])
    client = LLMClient(settings(llm_model="gpt-4o-mini", llm_temperature=0.0, llm_max_output_tokens=777), client=fake, backoff_base_seconds=0)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    client.chat(MESSAGES, tools=tools, response_format=fmt, tool_choice="none")
    kwargs = fake.calls[0]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_completion_tokens"] == 777
    assert kwargs["tools"] == tools and kwargs["tool_choice"] == "none"
    assert kwargs["response_format"] == fmt
    assert kwargs["messages"] == MESSAGES


def test_omits_tools_and_tool_choice_when_no_tools_given():
    fake = FakeOpenAI([sdk_response(content="{}")])
    LLMClient(settings(), client=fake, backoff_base_seconds=0).chat(MESSAGES)
    assert "tools" not in fake.calls[0] and "tool_choice" not in fake.calls[0]


def test_rate_limit_is_retried_then_succeeds():
    fake = FakeOpenAI([rate_limit(), rate_limit(), sdk_response(content="ok")])
    client = LLMClient(settings(llm_max_retries=3), client=fake, backoff_base_seconds=0)
    assert client.chat(MESSAGES).message.content == "ok"
    assert len(fake.calls) == 3
    assert client.usage.retries == 2
    assert client.usage.calls == 1  # only successful completions count as calls


def test_exhausted_retries_raise_llm_unavailable():
    fake = FakeOpenAI([rate_limit()] * 10)
    client = LLMClient(settings(llm_max_retries=2), client=fake, backoff_base_seconds=0)
    with pytest.raises(LLMUnavailable, match="rate"):
        client.chat(MESSAGES)
    assert len(fake.calls) == 3  # 1 attempt + 2 retries


def test_non_retryable_error_is_raised_immediately():
    fake = FakeOpenAI([auth_error(), sdk_response(content="never")])
    client = LLMClient(settings(llm_max_retries=3), client=fake, backoff_base_seconds=0)
    with pytest.raises(LLMRequestError, match="bad key"):
        client.chat(MESSAGES)
    assert len(fake.calls) == 1


def test_missing_api_key_without_injected_client_is_an_error():
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        LLMClient(Settings(_env_file=None, openai_api_key=None))


def test_estimate_cost_uses_price_table():
    assert estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000) == pytest.approx(0.75)
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", prompt_tokens=2_000, completion_tokens=500) == pytest.approx(0.0006)
    assert estimate_cost_usd("some-unknown-model", prompt_tokens=1, completion_tokens=1) is None
