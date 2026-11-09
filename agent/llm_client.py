"""LLM adapters. No paid provider is invoked by the MVP."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse: ...


class MockLLMClient:
    """Development stub; makes no model call and produces no usage claim."""

    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse:
        return LLMResponse("MOCK_ONLY", "mock", None, None)


class RealLLMClient:
    """TODO: implement provider adapter, preflight reservation, usage capture and audit log."""

    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMResponse:
        raise NotImplementedError("Real provider adapter and autonomous LLM planning are not implemented")
