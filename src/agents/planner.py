"""Planner: immutable anchors (rules) + task DAG (model, validated, with a deterministic fallback)."""

from __future__ import annotations

import re

from src.anchors import contains_anchor, extract_anchors
from src.dag import validate_dag
from src.prompts import PLANNER_SYSTEM, anchor_payload, extract_json, render
from src.providers.base import LLMRequest
from src.providers.router import AllProvidersFailed, LLMRouter
from src.schemas import Anchor, AnchorKind, Plan, Task

CLAUSE_SPLIT = re.compile(r"(?<=[.;?!])\s+|,?\s+(?=(?:and\s+)?then\b)", re.I)
CHAIN_START = re.compile(r"^(?:and\s+)?then\b|^based on\b|^using (?:that|this|the result)", re.I)


def heuristic_tasks(query: str, anchors: tuple[Anchor, ...]) -> list[Task]:
    """Deterministic decomposition: one analysis task per clause; 'then ...' clauses depend on the previous one."""
    clauses = [c.strip(" ,;") for c in CLAUSE_SPLIT.split(query) if c and len(c.split()) >= 3]
    tasks: list[Task] = []
    for clause in clauses[:5]:
        facts = tuple(a.id for a in anchors if a.kind is not AnchorKind.CONSTRAINT and contains_anchor(clause, a))
        if not facts and any(a.kind is AnchorKind.CONSTRAINT and a.text.lower() in clause.lower() for a in anchors):
            continue  # pure formatting instruction ("Keep it under 120 words") belongs to the synthesis step
        deps = (tasks[-1].id,) if tasks and CHAIN_START.match(clause) else ()
        tasks.append(Task(id=f"t{len(tasks) + 1}", instruction=clause, depends_on=deps, anchor_ids=facts))
    if not tasks:
        facts = tuple(a.id for a in anchors if a.kind is not AnchorKind.CONSTRAINT)
        tasks.append(Task(id="t1", instruction=query, anchor_ids=facts))
    return tasks


def with_synthesis(tasks: list[Task], anchors: tuple[Anchor, ...], query: str) -> list[Task]:
    """Ensure exactly one synthesis task that depends on every analysis task and owns every anchor."""
    analysis = [t for t in tasks if t.kind == "analysis"]
    synth_id = next((t.id for t in tasks if t.kind == "synthesis"), f"t{len(analysis) + 1}")
    synth = Task(
        id=synth_id,
        kind="synthesis",
        instruction=f"Write the final answer to the request: {query}",
        depends_on=tuple(t.id for t in analysis),
        anchor_ids=tuple(a.id for a in anchors),
    )
    # Analysis tasks never own constraints: those govern the final answer only.
    constraint_ids = {a.id for a in anchors if a.kind is AnchorKind.CONSTRAINT}
    cleaned = [
        Task(id=t.id, instruction=t.instruction, depends_on=t.depends_on,
             anchor_ids=tuple(a for a in t.anchor_ids if a not in constraint_ids), kind="analysis")
        for t in analysis
    ]
    return cleaned + [synth]


def parse_tasks(text: str) -> list[Task]:
    data = extract_json(text)
    raw = data["tasks"] if isinstance(data, dict) else data
    tasks = []
    for item in raw:
        tasks.append(Task(
            id=str(item["id"]),
            instruction=str(item.get("instruction", "")).strip(),
            depends_on=tuple(str(d) for d in item.get("depends_on", []) or []),
            anchor_ids=tuple(str(a) for a in item.get("anchor_ids", []) or []),
            kind="synthesis" if item.get("kind") == "synthesis" else "analysis",
        ))
    return tasks


class Planner:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def plan(self, query: str) -> Plan:
        anchors = extract_anchors(query)
        source = "llm"
        try:
            completion = await self.router.complete(LLMRequest(
                system=PLANNER_SYSTEM,
                user=render("Plan this request.", {"query": query, "anchors": anchor_payload(list(anchors))}),
                role="planner",
                task_id="plan",
            ))
            tasks = parse_tasks(completion.text)
            if not any(t.kind == "synthesis" for t in tasks):
                tasks = with_synthesis(tasks, anchors, query)
            errors = validate_dag(tasks, anchors)
        except (AllProvidersFailed, ValueError, KeyError, TypeError) as exc:
            tasks, errors = [], [f"planner call failed: {exc}"]
        if errors:
            tasks, source = heuristic_tasks(query, anchors), "fallback"
        tasks = with_synthesis(tasks, anchors, query)
        assert not validate_dag(tasks, anchors), validate_dag(tasks, anchors)
        return Plan(query=query, anchors=anchors, tasks=tuple(tasks), planner_source=source)
