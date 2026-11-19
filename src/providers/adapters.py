"""HTTP adapters for Gemini and OpenAI-compatible chat APIs (OpenAI, Groq, vLLM, Ollama, LM Studio...)."""

from __future__ import annotations

import time
from typing import Optional

import httpx

from src.providers.base import Completion, LLMRequest, ProviderError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _raise_for(provider: str, response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body = response.text[:300]
    raise ProviderError(provider, f"HTTP {response.status_code}: {body}", status=response.status_code,
                        retryable=response.status_code in RETRYABLE_STATUS)


class OpenAICompatibleProvider:
    """Any endpoint that implements POST {base_url}/chat/completions."""

    def __init__(self, *, name: str, base_url: str, api_key: str, model: str,
                 timeout_s: float = 60.0, client: Optional[httpx.AsyncClient] = None):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def complete(self, request: LLMRequest) -> Completion:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": request.user}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", retryable=True) from exc
        _raise_for(self.name, response)
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {str(data)[:200]}", retryable=False) from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text, provider=self.name, model=self.model, latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
        )


class GeminiProvider:
    """Google Gemini via the Generative Language REST API (generateContent)."""

    def __init__(self, *, api_key: str, model: str, base_url: str = "https://generativelanguage.googleapis.com",
                 timeout_s: float = 60.0, client: Optional[httpx.AsyncClient] = None):
        self.name = "gemini"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def complete(self, request: LLMRequest) -> Completion:
        config: dict = {"temperature": request.temperature, "maxOutputTokens": request.max_tokens}
        if request.json_mode:
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": config,
        }
        start = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self.base_url}/v1beta/models/{self.model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self._api_key},
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"transport error: {exc}", retryable=True) from exc
        _raise_for(self.name, response)
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            # Safety blocks and empty candidates come back without content parts.
            reason = (data.get("promptFeedback") or {}).get("blockReason") or str(data)[:200]
            raise ProviderError(self.name, f"no content: {reason}", retryable=False) from exc
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text, provider=self.name, model=self.model, latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=usage.get("promptTokenCount"), output_tokens=usage.get("candidatesTokenCount"),
        )
