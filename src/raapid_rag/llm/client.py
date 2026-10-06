"""Thin wrapper over the OpenAI Chat Completions API.

What it adds on top of the SDK, and why each piece exists:

* **Normalised result types** (:class:`ChatResult`, :class:`ChatMessage`, :class:`ToolCall`).
  The agent depends on these small dataclasses, not on SDK objects, so it can be tested with
  a scripted fake and the provider can be swapped behind this file.
* **Explicit retries** with exponential backoff and jitter (tenacity) for transient failures
  only: rate limits (429), timeouts, connection errors, 5xx.  The SDK's own retries are
  disabled so that every retry is visible and counted.  Non-retryable errors (401, 400) are
  raised immediately as :class:`LLMRequestError`; exhausting retries raises
  :class:`LLMUnavailable`.  The service layer turns both into a schema-valid error envelope.
* **Usage accounting** (:class:`UsageTotals`) and a small price table so each request can
  report tokens and an estimated cost in its trace.

Scaling note: this class is stateless apart from counters, so one instance per process is
shared across requests.  Concurrency to the provider is bounded in the service layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from ..config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for all failures raised by this module."""


class LLMUnavailable(LLMError):
    """Transient provider failure that persisted through all retries."""


class LLMRequestError(LLMError):
    """The provider rejected the request; retrying would not help (auth, bad schema, ...)."""


RETRYABLE: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it


@dataclass(frozen=True)
class ChatMessage:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ChatResult:
    message: ChatMessage
    finish_reason: str | None
    usage: dict[str, int]
    latency_ms: float
    model: str | None


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: dict[str, int]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.calls += 1

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens, "calls": self.calls, "retries": self.retries}


# USD per million tokens (input, output). List prices at the time of writing; verify before quoting.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost_usd(model: str, *, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Best-effort cost estimate; matches the longest price-table key that prefixes the model name."""
    matches = [k for k in PRICES_PER_MILLION if model.startswith(k)]
    if not matches:
        return None
    inp, out = PRICES_PER_MILLION[max(matches, key=len)]
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class LLMClient:
    def __init__(self, settings: Settings, client: Any | None = None, backoff_base_seconds: float = 0.5) -> None:
        if client is None:
            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY is not configured; set it in .env or run in retrieval-only mode")
            client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=0)
        self._client = client
        self._settings = settings
        self._backoff = backoff_base_seconds
        self.usage = UsageTotals()

    @property
    def model(self) -> str:
        return self._settings.llm_model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_completion_tokens": self._settings.llm_max_output_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        attempts = self._settings.llm_max_retries + 1
        retryer = Retrying(
            retry=retry_if_exception_type(RETRYABLE),
            wait=wait_random_exponential(multiplier=self._backoff, max=self._backoff * 20),
            stop=stop_after_attempt(attempts),
            reraise=True,
        )

        t0 = time.perf_counter()
        try:
            for attempt in retryer:
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        self.usage.retries += 1
                        logger.warning("LLM call retry %d/%d", attempt.retry_state.attempt_number - 1, attempts - 1)
                    response = self._client.chat.completions.create(**kwargs)
        except RETRYABLE as exc:
            raise LLMUnavailable(f"LLM unavailable after {attempts} attempt(s): {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMRequestError(f"LLM request rejected ({exc.status_code}): {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return self._normalise(response, latency_ms)

    def _normalise(self, response: Any, latency_ms: float) -> ChatResult:
        choice = response.choices[0]
        raw_msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (getattr(raw_msg, "tool_calls", None) or [])
        ]
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        }
        self.usage.add(usage)
        return ChatResult(
            message=ChatMessage(content=getattr(raw_msg, "content", None), tool_calls=tool_calls),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            latency_ms=latency_ms,
            model=getattr(response, "model", None),
        )
