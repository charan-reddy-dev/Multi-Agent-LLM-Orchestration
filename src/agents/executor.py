"""Executor: runs one DAG node and validates the reply against the NodeOutput schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from src.prompts import EXECUTOR_SYSTEM, extract_json, node_payload, render
from src.providers.base import LLMRequest
from src.providers.router import AllProvidersFailed, LLMRouter
from src.schemas import NodeOutput, Plan, Task


@dataclass
class Attempt:
    raw: str
    output: Optional[NodeOutput]
    schema_error: Optional[str]
    provider: str
    latency_ms: float


def parse_output(text: str) -> tuple[Optional[NodeOutput], Optional[str]]:
    try:
        data = extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"reply is not valid JSON ({exc.__class__.__name__})"
    try:
        return NodeOutput.model_validate(data), None
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors()[:5])
        return None, f"reply does not match the schema: {problems}"


async def call_node(router: LLMRouter, request: LLMRequest) -> Attempt:
    try:
        completion = await router.complete(request)
    except AllProvidersFailed as exc:
        return Attempt("", None, f"no provider produced a reply: {exc}", "none", 0.0)
    output, error = parse_output(completion.text)
    return Attempt(completion.text, output, error, f"{completion.provider}:{completion.model}", completion.latency_ms)


class Executor:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def execute(self, task: Task, plan: Plan, upstream: dict[str, NodeOutput]) -> Attempt:
        payload = node_payload(task, plan, upstream)
        return await call_node(self.router, LLMRequest(
            system=EXECUTOR_SYSTEM,
            user=render("Complete this sub-task.", payload),
            role="executor",
            task_id=task.id,
            attempt=0,
        ))
