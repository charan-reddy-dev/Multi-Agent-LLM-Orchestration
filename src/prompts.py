"""Prompt templates and JSON helpers shared by the agents.

Every agent sends a short instruction plus a JSON payload after the INPUT marker, and asks for JSON
back, so the same prompts work for any provider (and the offline mock can read the payload).
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.schemas import Anchor, AnchorKind, NodeOutput, Plan, Task

INPUT_MARKER = "INPUT (JSON):"

PLANNER_SYSTEM = (
    "You are the Planner in a Planner-Executor-Verifier-Refiner pipeline. Decompose the user's request into a "
    "small directed acyclic graph of sub-tasks (2-6 analysis tasks plus exactly one final synthesis task). "
    "Each analysis task lists the anchor ids it must use; the synthesis task depends on every analysis task. "
    "Anchors are immutable facts extracted from the request: never invent, change or drop them. "
    'Reply with JSON only: {"tasks": [{"id": "t1", "kind": "analysis"|"synthesis", "instruction": "...", '
    '"depends_on": ["..."], "anchor_ids": ["N1", "E2"]}]}'
)

EXECUTOR_SYSTEM = (
    "You are the Executor in a Planner-Executor-Verifier-Refiner pipeline. Complete one sub-task. "
    "Copy every required anchor into your content exactly as written (same digits, units and spelling). "
    "Do not introduce figures that are not in the anchors. Obey every constraint. "
    'Reply with JSON only, matching: {"title": str, "content": str, "anchors_used": [anchor ids]}'
)

REFINER_SYSTEM = (
    "You are the Refiner in a Planner-Executor-Verifier-Refiner pipeline. The Verifier rejected a draft. "
    "Fix every listed violation and change nothing else: restore missing anchors verbatim, remove unsupported "
    "figures, satisfy the constraints and return valid JSON. "
    'Reply with JSON only, matching: {"title": str, "content": str, "anchors_used": [anchor ids]}'
)


def render(instruction: str, payload: dict[str, Any]) -> str:
    return f"{instruction}\n\n{INPUT_MARKER}\n{json.dumps(payload, ensure_ascii=False, indent=1)}"


def read_payload(user_message: str) -> dict[str, Any]:
    return json.loads(user_message.split(INPUT_MARKER, 1)[1])


def anchor_payload(anchors: list[Anchor]) -> list[dict]:
    return [
        {"id": a.id, "kind": a.kind.value, "text": a.text, **({"rule": list(a.rule)} if a.rule else {})}
        for a in anchors
    ]


def node_payload(task: Task, plan: Plan, upstream: dict[str, NodeOutput]) -> dict[str, Any]:
    anchors = [plan.anchor(a) for a in task.anchor_ids]
    return {
        "query": plan.query,
        "task": {"id": task.id, "kind": task.kind, "instruction": task.instruction},
        "required_anchors": anchor_payload([a for a in anchors if a.kind is not AnchorKind.CONSTRAINT]),
        "constraints": anchor_payload([a for a in anchors if a.kind is AnchorKind.CONSTRAINT]),
        "upstream": {tid: {"title": o.title, "content": o.content} for tid, o in upstream.items()},
        "schema": NodeOutput.model_json_schema(),
    }


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> Any:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    cleaned = FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
