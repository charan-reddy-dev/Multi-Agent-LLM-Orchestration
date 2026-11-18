"""Refiner: closed-loop repair of a draft the Verifier rejected."""

from __future__ import annotations

from typing import Optional

from src.agents.executor import Attempt, call_node
from src.prompts import REFINER_SYSTEM, node_payload, render
from src.providers.base import LLMRequest
from src.providers.router import LLMRouter
from src.schemas import NodeOutput, Plan, Task, Verdict


class Refiner:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def refine(
        self,
        task: Task,
        plan: Plan,
        upstream: dict[str, NodeOutput],
        previous_raw: str,
        previous_output: Optional[NodeOutput],
        verdict: Verdict,
        attempt: int,
    ) -> Attempt:
        payload = node_payload(task, plan, upstream)
        payload["draft"] = previous_output.model_dump() if previous_output else previous_raw[:4000]
        payload["violations"] = [v.as_dict() for v in verdict.violations]
        return await call_node(self.router, LLMRequest(
            system=REFINER_SYSTEM,
            user=render("Repair this draft.", payload),
            role="refiner",
            task_id=task.id,
            attempt=attempt,
        ))
