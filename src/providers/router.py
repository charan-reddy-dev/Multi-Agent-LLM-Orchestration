"""Multi-provider LLM router with retry backoff and automatic fallback.

Providers are tried in order. Retryable failures (HTTP 429/5xx, timeouts, transport errors) are retried
with exponential backoff and jitter; non-retryable failures (bad key, bad request, blocked content)
move straight to the next provider. Every attempt is recorded in `calls`.

Configuration (environment):
    PEVR_PROVIDERS        comma-separated order, e.g. "groq,gemini,openai" (default: every provider
                          that has an API key, in that order; "mock" if none has one)
    GROQ_API_KEY          GROQ_MODEL     (default llama-3.3-70b-versatile)
    GEMINI_API_KEY        GEMINI_MODEL   (default gemini-2.5-flash)
    OPENAI_API_KEY        OPENAI_MODEL   (default gpt-4o-mini)
    OPENAI_BASE_URL       any OpenAI-compatible server (default https://api.openai.com/v1)
    PEVR_MAX_ATTEMPTS     attempts per provider (default 3)
    PEVR_BACKOFF_BASE_S   first backoff delay in seconds (default 0.5)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Awaitable, Callable, Optional, Sequence

from src.providers.adapters import GeminiProvider, OpenAICompatibleProvider
from src.providers.base import CallRecord, Completion, LLMRequest, Provider, ProviderError

Sleep = Callable[[float], Awaitable[None]]


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[ProviderError]):
        super().__init__("all providers failed: " + "; ".join(str(e) for e in errors))
        self.errors = errors


class LLMRouter:
    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 8.0,
        sleep: Sleep = asyncio.sleep,
        rng: Optional[random.Random] = None,
    ):
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self.providers = list(providers)
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.calls: list[CallRecord] = []

    def backoff(self, attempt: int) -> float:
        """Full-jitter exponential backoff: uniform(0, min(max, base * 2**attempt))."""
        return self._rng.uniform(0, min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt)))

    async def complete(self, request: LLMRequest) -> Completion:
        errors: list[ProviderError] = []
        for provider in self.providers:
            for attempt in range(self.max_attempts):
                start = time.perf_counter()
                try:
                    completion = await provider.complete(request)
                except ProviderError as exc:
                    self.calls.append(CallRecord(provider.name, provider.model, request.role, request.task_id, attempt,
                                                 False, (time.perf_counter() - start) * 1000, exc.status, str(exc)[:200]))
                    errors.append(exc)
                    if not exc.retryable or attempt == self.max_attempts - 1:
                        break
                    await self._sleep(self.backoff(attempt))
                    continue
                self.calls.append(CallRecord(provider.name, provider.model, request.role, request.task_id, attempt,
                                             True, completion.latency_ms))
                return completion
        raise AllProvidersFailed(errors)

    @property
    def names(self) -> list[str]:
        return [f"{p.name}:{p.model}" for p in self.providers]


def build_provider(name: str, env: Optional[dict] = None) -> Provider:
    env = os.environ if env is None else env
    name = name.strip().lower()
    if name == "groq":
        return OpenAICompatibleProvider(name="groq", base_url="https://api.groq.com/openai/v1",
                                        api_key=env["GROQ_API_KEY"], model=env.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    if name == "gemini":
        return GeminiProvider(api_key=env["GEMINI_API_KEY"], model=env.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if name == "openai":
        return OpenAICompatibleProvider(name="openai", base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                        api_key=env["OPENAI_API_KEY"], model=env.get("OPENAI_MODEL", "gpt-4o-mini"))
    if name == "mock":
        from src.providers.mock import MockProvider

        return MockProvider()
    raise ValueError(f"unknown provider {name!r}; expected groq, gemini, openai or mock")


KEY_FOR = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


def router_from_env(env: Optional[dict] = None) -> LLMRouter:
    env = os.environ if env is None else env
    if env.get("PEVR_PROVIDERS"):
        names = [n for n in env["PEVR_PROVIDERS"].split(",") if n.strip()]
    else:
        names = [n for n, key in KEY_FOR.items() if env.get(key)] or ["mock"]
    return LLMRouter(
        [build_provider(n, env) for n in names],
        max_attempts=int(env.get("PEVR_MAX_ATTEMPTS", 3)),
        backoff_base_s=float(env.get("PEVR_BACKOFF_BASE_S", 0.5)),
    )
