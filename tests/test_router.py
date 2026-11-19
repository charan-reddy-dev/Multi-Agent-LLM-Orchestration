import json
import random

import httpx
import pytest

from src.providers.adapters import GeminiProvider, OpenAICompatibleProvider
from src.providers.base import Completion, LLMRequest, ProviderError
from src.providers.mock import MockProvider
from src.providers.router import AllProvidersFailed, LLMRouter, router_from_env

REQUEST = LLMRequest(system="sys", user="hello", role="executor", task_id="t1")


class FakeProvider:
    def __init__(self, name, outcomes):
        self.name, self.model = name, "fake"
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Completion(text=outcome, provider=self.name, model=self.model, latency_ms=1.0)


class SleepRecorder:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


def retryable(name):
    return ProviderError(name, "HTTP 429", status=429, retryable=True)


async def test_retries_with_backoff_then_succeeds():
    sleep = SleepRecorder()
    groq = FakeProvider("groq", [retryable("groq"), retryable("groq"), "ok"])
    router = LLMRouter([groq], max_attempts=3, backoff_base_s=0.5, sleep=sleep, rng=random.Random(1))
    assert (await router.complete(REQUEST)).text == "ok"
    assert groq.calls == 3 and len(sleep.delays) == 2
    assert 0 <= sleep.delays[0] <= 0.5 and 0 <= sleep.delays[1] <= 1.0
    assert [c.ok for c in router.calls] == [False, False, True]


async def test_falls_back_to_next_provider_after_exhausting_retries():
    sleep = SleepRecorder()
    groq = FakeProvider("groq", [retryable("groq")] * 2)
    gemini = FakeProvider("gemini", ["from gemini"])
    router = LLMRouter([groq, gemini], max_attempts=2, sleep=sleep, rng=random.Random(1))
    completion = await router.complete(REQUEST)
    assert completion.provider == "gemini" and groq.calls == 2 and len(sleep.delays) == 1


async def test_non_retryable_error_skips_straight_to_fallback():
    sleep = SleepRecorder()
    groq = FakeProvider("groq", [ProviderError("groq", "HTTP 401", status=401, retryable=False)])
    gemini = FakeProvider("gemini", ["ok"])
    router = LLMRouter([groq, gemini], max_attempts=3, sleep=sleep)
    assert (await router.complete(REQUEST)).provider == "gemini"
    assert groq.calls == 1 and sleep.delays == []


async def test_raises_when_every_provider_fails():
    router = LLMRouter([FakeProvider("a", [retryable("a")]), FakeProvider("b", [retryable("b")])],
                       max_attempts=1, sleep=SleepRecorder())
    with pytest.raises(AllProvidersFailed) as info:
        await router.complete(REQUEST)
    assert len(info.value.errors) == 2


def test_backoff_is_capped():
    router = LLMRouter([MockProvider()], backoff_base_s=1.0, backoff_max_s=4.0, rng=random.Random(3))
    assert all(router.backoff(10) <= 4.0 for _ in range(100))


def test_router_from_env_order_and_defaults():
    assert router_from_env({}).names == ["mock:deterministic-v1"]
    env = {"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m", "PEVR_MAX_ATTEMPTS": "5"}
    router = router_from_env(env)
    assert router.names == ["groq:llama-3.3-70b-versatile", "gemini:gemini-2.5-flash"]
    assert router.max_attempts == 5
    custom = router_from_env({**env, "PEVR_PROVIDERS": "gemini,groq", "GEMINI_MODEL": "gemini-2.0-flash"})
    assert custom.names == ["gemini:gemini-2.0-flash", "groq:llama-3.3-70b-versatile"]


async def test_openai_compatible_adapter_request_and_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"a\": 1}"}}],
                                         "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(name="groq", base_url="https://api.groq.com/openai/v1/", api_key="k",
                                        model="llama", client=client)
    completion = await provider.complete(REQUEST)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("{\"a\": 1}", 7, 3)


async def test_gemini_adapter_request_and_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{\"b\": "}, {"text": "2}"}]}}],
                                         "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    completion = await GeminiProvider(api_key="secret", model="gemini-2.5-flash", client=client).complete(REQUEST)
    assert seen["url"].endswith("/v1beta/models/gemini-2.5-flash:generateContent")
    assert seen["key"] == "secret"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert (completion.text, completion.input_tokens) == ("{\"b\": 2}", 9)


@pytest.mark.parametrize("status,retryable_expected", [(429, True), (503, True), (400, False), (401, False)])
async def test_adapter_error_classification(status, retryable_expected):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, text="nope")))
    provider = OpenAICompatibleProvider(name="openai", base_url="http://x", api_key="k", model="m", client=client)
    with pytest.raises(ProviderError) as info:
        await provider.complete(REQUEST)
    assert info.value.status == status and info.value.retryable is retryable_expected


async def test_gemini_safety_block_is_not_retryable():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})))
    with pytest.raises(ProviderError) as info:
        await GeminiProvider(api_key="k", model="m", client=client).complete(REQUEST)
    assert not info.value.retryable and "SAFETY" in str(info.value)


async def test_transport_errors_are_retryable():
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    with pytest.raises(ProviderError) as info:
        await OpenAICompatibleProvider(name="o", base_url="http://x", api_key="k", model="m", client=client).complete(REQUEST)
    assert info.value.retryable
