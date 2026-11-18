"""Verifier: deterministic audit of a node output against the Planner's anchors and the schema.

No model is involved, so the same output always gets the same score:

    score = 0.6 * anchor_recall + 0.2 * schema_ok + 0.2 * constraint_pass_rate
            - 0.1 * unsupported_numbers (penalty capped at 0.3), clamped to [0, 1]

* anchor_recall        share of required number/entity anchors present verbatim
* schema_ok            1 if the reply parsed and matched NodeOutput, else 0 (and recall is 0)
* constraint_pass_rate share of checkable constraints satisfied (synthesis node only; 1 if none)
* unsupported_numbers  figures in the output that appear nowhere in the request (small counts 0-10 allowed)
"""

from __future__ import annotations

from typing import Optional

from src.anchors import contains_anchor, count_bullets, count_words, numeric_values
from src.schemas import Anchor, AnchorKind, NodeOutput, Plan, Task, Verdict, Violation, ViolationCode

SMALL_COUNTS = {str(i) for i in range(11)}


def check_constraint(anchor: Anchor, content: str) -> Optional[str]:
    """Return a failure message, or None if satisfied (or not machine-checkable)."""
    rule, value = anchor.rule or (None, None)
    if rule == "max_words":
        words = count_words(content)
        return None if words <= value else f"{words} words; limit is {value}"
    if rule == "bullets":
        bullets = count_bullets(content)
        return None if bullets == value else f"{bullets} bullet points; expected {value}"
    if rule == "forbid":
        term = Anchor(id="", kind=AnchorKind.ENTITY, text=str(value))
        return f"mentions forbidden term '{value}'" if contains_anchor(content, term) else None
    return None


class Verifier:
    def verify(self, task: Task, plan: Plan, output: Optional[NodeOutput], schema_error: Optional[str] = None) -> Verdict:
        anchors = [plan.anchor(a) for a in task.anchor_ids]
        required = [a for a in anchors if a.must_appear]
        constraints = [a for a in anchors if a.kind is AnchorKind.CONSTRAINT and a.rule] if task.kind == "synthesis" else []
        violations: list[Violation] = []

        if output is None:
            violations.append(Violation(ViolationCode.SCHEMA, schema_error or "no output"))
            violations += [Violation(ViolationCode.MISSING_ANCHOR, f"'{a.text}' not present", a.id) for a in required]
            return Verdict(
                score=0.0, violations=tuple(violations), anchors_required=len(required), anchors_present=0,
                schema_ok=False, constraints_checked=len(constraints), constraints_passed=0,
            )

        text = f"{output.title}\n{output.content}"
        present = 0
        for a in required:
            if contains_anchor(text, a):
                present += 1
            else:
                violations.append(Violation(ViolationCode.MISSING_ANCHOR, f"'{a.text}' not present verbatim", a.id))

        allowed = numeric_values(plan.query) | SMALL_COUNTS
        unsupported = sorted(numeric_values(text) - allowed)
        violations += [Violation(ViolationCode.UNSUPPORTED_NUMBER, f"'{n}' does not appear in the request") for n in unsupported]

        passed_constraints = 0
        for c in constraints:
            failure = check_constraint(c, output.content)
            if failure:
                violations.append(Violation(ViolationCode.CONSTRAINT, f"{c.text}: {failure}", c.id))
            else:
                passed_constraints += 1

        recall = present / len(required) if required else 1.0
        constraint_rate = passed_constraints / len(constraints) if constraints else 1.0
        penalty = min(0.3, 0.1 * len(unsupported))
        score = max(0.0, min(1.0, 0.6 * recall + 0.2 + 0.2 * constraint_rate - penalty))
        return Verdict(
            score=round(score, 4), violations=tuple(violations), anchors_required=len(required),
            anchors_present=present, schema_ok=True, constraints_checked=len(constraints),
            constraints_passed=passed_constraints,
        )
