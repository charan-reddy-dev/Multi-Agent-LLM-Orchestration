"""Core data types for the PEVR pipeline.

Anchors and plans are frozen: once the Planner has produced them, no downstream agent can change them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnchorKind(str, Enum):
    NUMBER = "number"
    ENTITY = "entity"
    CONSTRAINT = "constraint"


@dataclass(frozen=True)
class Anchor:
    """A fact from the query that must survive verbatim into the output.

    `text` is the exact surface form from the query. `rule` is set for machine-checkable constraints,
    e.g. ("max_words", 120), ("bullets", 3), ("forbid", "Kafka").
    """

    id: str
    kind: AnchorKind
    text: str
    rule: Optional[tuple[str, Any]] = None

    @property
    def must_appear(self) -> bool:
        """Numbers and entities must appear in the output; constraints are checked by their rule."""
        return self.kind is not AnchorKind.CONSTRAINT


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    depends_on: tuple[str, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    kind: Literal["analysis", "synthesis"] = "analysis"


@dataclass(frozen=True)
class Plan:
    query: str
    anchors: tuple[Anchor, ...]
    tasks: tuple[Task, ...]
    planner_source: str = "llm"  # "llm", or "fallback" when the model's DAG was invalid

    def anchor(self, anchor_id: str) -> Anchor:
        return next(a for a in self.anchors if a.id == anchor_id)

    def task(self, task_id: str) -> Task:
        return next(t for t in self.tasks if t.id == task_id)

    @property
    def synthesis(self) -> Task:
        return next(t for t in self.tasks if t.kind == "synthesis")


class NodeOutput(BaseModel):
    """JSON schema every Executor/Refiner response must satisfy."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    anchors_used: list[str] = Field(default_factory=list)


class ViolationCode(str, Enum):
    SCHEMA = "schema_violation"
    MISSING_ANCHOR = "missing_anchor"
    UNSUPPORTED_NUMBER = "unsupported_number"
    CONSTRAINT = "constraint_violation"


@dataclass(frozen=True)
class Violation:
    code: ViolationCode
    detail: str
    anchor_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {"code": self.code.value, "detail": self.detail, "anchor_id": self.anchor_id}


@dataclass(frozen=True)
class Verdict:
    score: float
    violations: tuple[Violation, ...]
    anchors_required: int
    anchors_present: int
    schema_ok: bool
    constraints_checked: int
    constraints_passed: int

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "anchors_required": self.anchors_required,
            "anchors_present": self.anchors_present,
            "schema_ok": self.schema_ok,
            "constraints_checked": self.constraints_checked,
            "constraints_passed": self.constraints_passed,
            "violations": [v.as_dict() for v in self.violations],
        }


@dataclass
class NodeRun:
    """Mutable record of one DAG node's execution and repair history."""

    task_id: str
    raw: str = ""
    output: Optional[NodeOutput] = None
    verdicts: list[Verdict] = field(default_factory=list)
    refinements: int = 0
    latency_ms: float = 0.0
    providers: list[str] = field(default_factory=list)

    @property
    def first_verdict(self) -> Verdict:
        return self.verdicts[0]

    @property
    def final_verdict(self) -> Verdict:
        return self.verdicts[-1]


@dataclass
class Report:
    query: str
    plan: Plan
    nodes: dict[str, NodeRun]
    final_text: str
    verdict: Verdict
    latency_ms: float
    llm_calls: int

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "final_text": self.final_text,
            "verdict": self.verdict.as_dict(),
            "latency_ms": round(self.latency_ms, 3),
            "llm_calls": self.llm_calls,
            "plan": {
                "planner_source": self.plan.planner_source,
                "anchors": [
                    {"id": a.id, "kind": a.kind.value, "text": a.text, "rule": list(a.rule) if a.rule else None}
                    for a in self.plan.anchors
                ],
                "tasks": [
                    {
                        "id": t.id,
                        "kind": t.kind,
                        "instruction": t.instruction,
                        "depends_on": list(t.depends_on),
                        "anchor_ids": list(t.anchor_ids),
                    }
                    for t in self.plan.tasks
                ],
            },
            "nodes": {
                tid: {
                    "refinements": n.refinements,
                    "providers": n.providers,
                    "latency_ms": round(n.latency_ms, 3),
                    "first_verdict": n.first_verdict.as_dict(),
                    "final_verdict": n.final_verdict.as_dict(),
                }
                for tid, n in self.nodes.items()
            },
        }
