"""Configurable OpenAI-compatible chat client with budget and audit hooks.

No request is made at construction time. Formal use remains blocked until the
provider endpoint, token limits, official pricing and accounting are verified.
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import Request
from uuid import uuid4

from runtime.budget import BudgetManager
from runtime.logger import ExperimentLogger


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_present: bool = False
    finish_reason: str | None = None


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse: ...


class MockLLMClient:
    """Development stub; makes no model call and produces no usage claim."""

    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse:
        return LLMResponse("MOCK_ONLY", "mock", None, None)


class LLMCallError(RuntimeError):
    """No provider response body, credentials or prompt are exposed in errors."""


class LLMCallLimitError(LLMCallError):
    """A known local call cap stopped a request before it was sent."""


class RealLLMClient:
    """OpenAI-compatible chat adapter; one sequential client per task.

    input_token_upper_bound must be an independently verified upper bound on
    provider-billed input tokens, including all provider-added overhead. This
    cannot be established from prompt bytes alone. Uncertain calls block further
    calls until externally reconciled; never retry a potentially billed request.
    """

    def __init__(
        self, model: str, base_url: str, budget: BudgetManager | None,
        logger: ExperimentLogger, input_token_upper_bound: int | None,
        api_key_env: str = "SUPERAGENT_API_KEY", timeout_seconds: int = 30,
        transport=None, *, development_mode: bool = False, max_calls: int | None = None,
        allowed_response_models: tuple[str, ...] = (),
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTPS API base URL without query or fragment")
        if not model or timeout_seconds <= 0:
            raise ValueError("model and positive timeout are required")
        if development_mode:
            if type(max_calls) is not int or max_calls <= 0:
                raise ValueError("Development mode requires a positive call limit")
            if budget is not None:
                raise ValueError("Development mode cannot claim priced accounting")
            if model.casefold() not in {"qwen", "qwen3.8-27b", "deepseek-v4-pro", "dsp", "deepseek-v4.1-flash", "dsf"}:
                raise ValueError("Development mode requires an identified permitted school model")
        elif budget is None or input_token_upper_bound is None or input_token_upper_bound <= 0:
            raise ValueError("Priced mode requires a budget and positive input bound")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.budget = budget
        self.logger = logger
        self.input_token_upper_bound = input_token_upper_bound
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.transport = transport or self._direct_transport
        self.development_mode = development_mode
        self.max_calls = max_calls
        self.calls_started = 0
        self.usage_records: list[dict] = []
        self.allowed_response_models = {model.casefold(), *(item.casefold() for item in allowed_response_models)}
        # Explicit aliases from the school model list; never permit unrelated families.
        aliases = {
            "qwen3.8-27b": ("qwen", "qwen3.8-27b-w8a8"),
            "qwen": ("qwen3.8-27b", "qwen3.8-27b-w8a8"),
            "deepseek-v4-pro": ("dsp",), "dsp": ("deepseek-v4-pro",),
            "deepseek-v4.1-flash": ("dsf", "deepseek-v4.1-flash-1017"),
            "dsf": ("deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"),
        }
        self.allowed_response_models.update(aliases.get(model.casefold(), ()))
        self._blocked = False

    @staticmethod
    def _direct_transport(request, timeout):
        # Campus endpoint must bypass inherited proxy settings on Windows.
        import requests
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(request.full_url, data=request.data, headers=dict(request.header_items()),
                                    timeout=timeout, allow_redirects=False)
            response.raise_for_status()
            if response.status_code >= 300:
                raise requests.HTTPError("Unexpected redirect")
            return BytesIO(response.content)

    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse:
        if self._blocked:
            raise LLMCallError("Client blocked after an unresolved or invalid call")
        if not isinstance(prompt, str) or not prompt or max_output_tokens <= 0:
            raise ValueError("prompt and positive max_output_tokens are required")
        if self.development_mode and self.calls_started >= self.max_calls:
            raise LLMCallLimitError("Development call limit reached")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMCallError(f"API key environment variable {self.api_key_env} is missing")
        if not self.development_mode:
            self.budget.ensure_capacity(self.model, self.input_token_upper_bound, max_output_tokens)
        call_id = uuid4().hex
        body = json.dumps({
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens, "stream": False,
        }).encode("utf-8")
        request = Request(self.url, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
        }, method="POST")
        # Persist a started event before any paid request. If logging fails, no call occurs.
        self.logger.log("llm_call_started", call_id=call_id, model=self.model,
                        input_token_upper_bound=self.input_token_upper_bound,
                        max_output_tokens=max_output_tokens,
                        accounting_mode="development_unpriced" if self.development_mode else "priced",
                        max_calls=self.max_calls)
        self.calls_started += 1
        self._blocked = True
        try:
            with self.transport(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                            reason="transport_or_response_error", error_type=type(exc).__name__,
                            usage_status="unknown")
            raise LLMCallError("Provider request failed; usage is unknown and client is blocked") from None
        usage = payload.get("usage") if isinstance(payload, dict) else None
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if (type(input_tokens) is not int or type(output_tokens) is not int
                or input_tokens < 0 or output_tokens < 0):
            self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                            reason="missing_or_invalid_usage", usage_status="unknown")
            raise LLMCallError("Provider usage missing or invalid; client is blocked")
        # Preserve observed usage even if it exceeds a preflight limit or budget.
        if self.development_mode:
            record = {"model": self.model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                      "cost_rmb": None, "cost_status": "unconfigured"}
            self.usage_records.append(record)
            self.logger.log("llm_usage_observed", call_id=call_id, **record)
            self.logger.log("llm_call_usage", call_id=call_id, usage=record)
        else:
            input_price, output_price = self.budget._validated_prices(self.model)
            observed_cost = (input_price * input_tokens + output_price * output_tokens) / Decimal(1_000_000)
            self.logger.log("llm_usage_observed", call_id=call_id, model=self.model,
                            input_tokens=input_tokens, output_tokens=output_tokens,
                            input_price_per_million_rmb=input_price,
                            output_price_per_million_rmb=output_price, cost_rmb=observed_cost)
            try:
                record = self.budget.record(self.model, input_tokens, output_tokens)
            except Exception:
                self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                                reason="usage_accounting_failed", usage_status="observed")
                raise LLMCallError("Observed usage could not be accounted; client is blocked") from None
            self.usage_records.append({"model": self.model, "input_tokens": input_tokens,
                                       "output_tokens": output_tokens, "cost_rmb": record.cost_rmb})
            self.logger.log("llm_call_usage", call_id=call_id, usage=record)
        if (not self.development_mode and input_tokens > self.input_token_upper_bound) or output_tokens > max_output_tokens:
            self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                            reason="provider_exceeded_token_bounds", usage_status="recorded")
            raise LLMCallError("Provider exceeded configured token bounds; client is blocked")
        returned_model = payload.get("model")
        if returned_model is not None and (not isinstance(returned_model, str) or returned_model.casefold() not in self.allowed_response_models):
            self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                            reason="provider_model_mismatch", usage_status="recorded", returned_model=returned_model)
            raise LLMCallError("Provider returned a different model; client is blocked")
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            text = message.get("content")
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            finish_reason = choice.get("finish_reason")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("No text response")
        except (KeyError, IndexError, TypeError, ValueError):
            self.logger.log("llm_call_unresolved", call_id=call_id, model=self.model,
                            reason="empty_or_invalid_content", usage_status="recorded")
            raise LLMCallError("Provider response has no content (possibly reasoning-only or truncated); client is blocked") from None
        self.logger.log("llm_call_completed", call_id=call_id, model=self.model,
                        returned_model=returned_model, reasoning_present=bool(reasoning), finish_reason=finish_reason)
        self._blocked = False
        return LLMResponse(text, self.model, input_tokens, output_tokens, bool(reasoning), finish_reason)
