"""Deterministic offline provider for tests and for benchmarking the pipeline's control loop.

The mock reads the JSON payload each agent sends and answers like a cooperative but imperfect model.
With a FaultProfile it injects the failure modes the Verifier is meant to catch:

* planner      returns an invalid DAG (a dependency cycle) with probability `invalid_plan`
* executor     drops a required anchor, alters a number (e.g. 36% -> 38%), breaks the JSON schema, or
               (on the synthesis node) violates a constraint, each with its own probability
* refiner      fixes each flagged violation with probability `fix`, and with probability `regression`
               drops a different anchor that was previously correct

Every random choice is seeded from (seed, role, task id, attempt, query), so a run is exactly
reproducible. Results obtained with the mock measure the pipeline's detection and repair loop, not the
quality of any real language model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

from src.prompts import read_payload
from src.providers.base import Completion, LLMRequest


@dataclass(frozen=True)
class FaultProfile:
    invalid_plan: float = 0.0
    drop_anchor: float = 0.0
    alter_number: float = 0.0
    schema_break: float = 0.0
    constraint_break: float = 0.0
    fix: float = 1.0
    regression: float = 0.0

    @classmethod
    def none(cls) -> "FaultProfile":
        return cls()

    @classmethod
    def benchmark(cls) -> "FaultProfile":
        """Fault rates used for benchmarks/results.json (fixed before the first run)."""
        return cls(invalid_plan=0.10, drop_anchor=0.25, alter_number=0.15, schema_break=0.10,
                   constraint_break=0.20, fix=0.80, regression=0.05)


def _alter(text: str) -> str:
    """Change the last digit of the first number: '36%' -> '38%', '2.5 s' -> '2.7 s'."""
    m = re.search(r"\d(?!.*\d)", re.search(r"[\d.,]+", text).group(0)) if re.search(r"\d", text) else None
    if not m:
        return text
    number = re.search(r"[\d.,]+", text)
    digits = number.group(0)
    last = max(i for i, ch in enumerate(digits) if ch.isdigit())
    changed = digits[:last] + str((int(digits[last]) + 2) % 10) + digits[last + 1:]
    return text[: number.start()] + changed + text[number.end():]


class MockProvider:
    name = "mock"

    def __init__(self, faults: Optional[FaultProfile] = None, seed: int = 7):
        self.faults = faults or FaultProfile.none()
        self.seed = seed
        self.model = "deterministic-v1" if self.faults == FaultProfile.none() else "deterministic-faulty-v1"

    def _rng(self, request: LLMRequest, payload: dict, salt: str = "") -> random.Random:
        key = f"{self.seed}|{request.role}|{request.task_id}|{request.attempt}|{payload.get('query', '')}|{salt}"
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

    async def complete(self, request: LLMRequest) -> Completion:
        start = time.perf_counter()
        payload = read_payload(request.user)
        if request.role == "planner":
            text = self._plan(request, payload)
        elif request.role == "refiner":
            text = self._refine(request, payload)
        else:
            text = self._execute(request, payload)
        return Completion(text=text, provider=self.name, model=self.model, latency_ms=(time.perf_counter() - start) * 1000)

    # ------------------------------------------------------------------ planner

    def _plan(self, request: LLMRequest, payload: dict) -> str:
        from src.agents.planner import heuristic_tasks
        from src.schemas import Anchor, AnchorKind

        anchors = tuple(Anchor(id=a["id"], kind=AnchorKind(a["kind"]), text=a["text"]) for a in payload["anchors"])
        tasks = heuristic_tasks(payload["query"], anchors)
        items = [{"id": t.id, "kind": "analysis", "instruction": t.instruction, "depends_on": list(t.depends_on),
                  "anchor_ids": list(t.anchor_ids)} for t in tasks]
        items.append({"id": f"t{len(items) + 1}", "kind": "synthesis", "instruction": "Write the final answer.",
                      "depends_on": [t["id"] for t in items], "anchor_ids": [a["id"] for a in payload["anchors"]]})
        if self._rng(request, payload).random() < self.faults.invalid_plan:
            items[0]["depends_on"] = [items[-1]["id"]]  # cycle: t1 -> synthesis -> t1
        return json.dumps({"tasks": items})

    # ------------------------------------------------------------------ executor

    def _clean(self, payload: dict, drop: set[str], altered: set[str]) -> dict:
        task = payload["task"]
        facts = [a for a in payload["required_anchors"] if a["id"] not in drop]
        texts = [_alter(a["text"]) if a["id"] in altered else a["text"] for a in facts]
        constraints = {c["rule"][0]: c["rule"][1] for c in payload.get("constraints", []) if "rule" in c}
        used = [a["id"] for a in facts if a["id"] not in altered]

        if task["kind"] == "synthesis":
            bullets = constraints.get("bullets")
            if isinstance(bullets, int) and bullets > 0:
                groups = [texts[i::bullets] for i in range(bullets)]
                lines = [f"- Point {i + 1}: " + (", ".join(g) if g else "see above") + "." for i, g in enumerate(groups)]
                content = "\n".join(lines)
            else:
                content = "Answer: " + (", ".join(texts) if texts else "no fixed figures apply") + "."
                if "max_words" not in constraints and payload.get("upstream"):
                    content += " This combines " + ", ".join(sorted(payload["upstream"])) + "."
            title = "Final answer"
        else:
            # Never echo the instruction: it quotes the query, so a dropped anchor would still appear verbatim.
            content = f"Findings for step {task['id']}. Facts used: " + ("; ".join(texts) if texts else "none") + "."
            title = f"Step {task['id']}"
        return {"title": title, "content": content, "anchors_used": used}

    def _execute(self, request: LLMRequest, payload: dict) -> str:
        rng = self._rng(request, payload)
        f = self.faults
        anchors = payload["required_anchors"]
        numbers = [a for a in anchors if a["kind"] == "number"]
        drop: set[str] = set()
        altered: set[str] = set()
        if anchors and rng.random() < f.drop_anchor:
            drop.add(rng.choice(anchors)["id"])
        if numbers and rng.random() < f.alter_number:
            altered.add(rng.choice(numbers)["id"])
        out = self._clean(payload, drop, altered)
        if payload["task"]["kind"] == "synthesis" and payload.get("constraints") and rng.random() < f.constraint_break:
            out["content"] = self._break_constraint(out["content"], rng.choice(payload["constraints"]))
        if rng.random() < f.schema_break:
            return self._break_schema(out, rng)
        return json.dumps(out)

    @staticmethod
    def _break_constraint(content: str, constraint: dict) -> str:
        rule, value = constraint["rule"]
        if rule == "forbid":
            return content + f" It also compares well with {value}."
        if rule == "bullets":
            return content + "\n- Extra point: further detail."
        if rule == "max_words":
            return content + " " + " ".join(["detail"] * (int(value) + 5))
        return content

    @staticmethod
    def _break_schema(out: dict, rng: random.Random) -> str:
        mode = rng.choice(["prose", "missing_field", "extra_field"])
        if mode == "prose":
            return f"Sure! Here is the answer: {out['content']}"
        if mode == "missing_field":
            return json.dumps({"title": out["title"], "anchors_used": out["anchors_used"]})
        return json.dumps({**out, "confidence": "high"})

    # ------------------------------------------------------------------ refiner

    def _refine(self, request: LLMRequest, payload: dict) -> str:
        rng = self._rng(request, payload)
        f = self.faults
        drop: set[str] = set()
        keep_numbers: list[str] = []
        keep_constraints: list[dict] = []
        schema_unfixed = False
        constraints = {c["id"]: c for c in payload.get("constraints", [])}
        for v in payload["violations"]:
            if rng.random() < f.fix:
                continue
            if v["code"] == "missing_anchor" and v["anchor_id"]:
                drop.add(v["anchor_id"])
            elif v["code"] == "unsupported_number":
                keep_numbers.append(re.search(r"'([^']+)'", v["detail"]).group(1))
            elif v["code"] == "constraint_violation" and v["anchor_id"] in constraints:
                keep_constraints.append(constraints[v["anchor_id"]])
            elif v["code"] == "schema_violation":
                schema_unfixed = True
        others = [a for a in payload["required_anchors"] if a["id"] not in drop]
        if others and rng.random() < f.regression:
            drop.add(rng.choice(others)["id"])
        out = self._clean(payload, drop, set())
        for n in keep_numbers:
            out["content"] += f" (earlier draft: {n})"
        for c in keep_constraints:
            out["content"] = self._break_constraint(out["content"], c)
        if schema_unfixed:
            return self._break_schema(out, rng)
        return json.dumps(out)
