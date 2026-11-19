"""Provider-agnostic request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    json_mode: bool = True
    temperature: float = 0.0
    max_tokens: int = 1200
    # Metadata for tracing (and for the offline mock); real providers ignore it.
    role: str = ""
    task_id: str = ""
    attempt: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ProviderError(Exception):
    """Raised by adapters. `retryable` errors (429, 5xx, timeouts) are retried with backoff."""

    def __init__(self, provider: str, message: str, *, status: Optional[int] = None, retryable: bool = True):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.retryable = retryable


class Provider(Protocol):
    name: str
    model: str

    async def complete(self, request: LLMRequest) -> Completion: ...


@dataclass
class CallRecord:
    provider: str
    model: str
    role: str
    task_id: str
    attempt: int
    ok: bool
    latency_ms: float
    status: Optional[int] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)
