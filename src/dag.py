"""DAG validation and scheduling."""

from __future__ import annotations

from collections import deque

from src.schemas import Anchor, Task

MAX_TASKS = 8


def validate_dag(tasks: list[Task], anchors: tuple[Anchor, ...]) -> list[str]:
    """Return a list of problems; an empty list means the plan is executable."""
    errors: list[str] = []
    if not tasks:
        return ["plan has no tasks"]
    if len(tasks) > MAX_TASKS:
        errors.append(f"plan has {len(tasks)} tasks; the limit is {MAX_TASKS}")
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        errors.append("duplicate task ids")
    known = set(ids)
    anchor_ids = {a.id for a in anchors}
    for t in tasks:
        if not t.instruction.strip():
            errors.append(f"{t.id}: empty instruction")
        for dep in t.depends_on:
            if dep == t.id:
                errors.append(f"{t.id}: depends on itself")
            elif dep not in known:
                errors.append(f"{t.id}: unknown dependency {dep}")
        for aid in t.anchor_ids:
            if aid not in anchor_ids:
                errors.append(f"{t.id}: unknown anchor {aid}")
    synth = [t for t in tasks if t.kind == "synthesis"]
    if len(synth) != 1:
        errors.append(f"expected exactly one synthesis task, found {len(synth)}")
    if not errors and has_cycle(tasks):
        errors.append("dependency cycle")
    if not errors:
        reachable = ancestors(tasks, synth[0].id)
        missing = [t.id for t in tasks if t.kind == "analysis" and t.id not in reachable]
        if missing:
            errors.append(f"synthesis does not depend on {missing}")
    return errors


def has_cycle(tasks: list[Task]) -> bool:
    try:
        topological_waves(tasks)
    except ValueError:
        return True
    return False


def ancestors(tasks: list[Task], task_id: str) -> set[str]:
    by_id = {t.id: t for t in tasks}
    seen: set[str] = set()
    stack = list(by_id[task_id].depends_on)
    while stack:
        current = stack.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        stack.extend(by_id[current].depends_on)
    return seen


def topological_waves(tasks: list[Task]) -> list[list[Task]]:
    """Kahn's algorithm grouped into waves: every task in a wave depends only on earlier waves."""
    by_id = {t.id: t for t in tasks}
    indegree = {t.id: len(t.depends_on) for t in tasks}
    children: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            children[dep].append(t.id)
    ready = deque(sorted(tid for tid, d in indegree.items() if d == 0))
    waves: list[list[Task]] = []
    done = 0
    while ready:
        wave = [by_id[tid] for tid in ready]
        ready.clear()
        waves.append(wave)
        done += len(wave)
        nxt = []
        for t in wave:
            for child in children[t.id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    nxt.append(child)
        ready.extend(sorted(nxt))
    if done != len(tasks):
        raise ValueError("dependency cycle")
    return waves


def depth(tasks: list[Task]) -> int:
    return len(topological_waves(tasks))
