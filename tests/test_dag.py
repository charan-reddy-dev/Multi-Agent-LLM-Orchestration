from src.agents.planner import heuristic_tasks, with_synthesis
from src.anchors import extract_anchors
from src.dag import depth, topological_waves, validate_dag
from src.schemas import Anchor, AnchorKind, Task

ANCHORS = (Anchor(id="a1", kind=AnchorKind.NUMBER, text="36%"),)


def T(tid, deps=(), kind="analysis", anchors=()):
    return Task(id=tid, instruction=f"do {tid}", depends_on=tuple(deps), anchor_ids=tuple(anchors), kind=kind)


def test_valid_diamond_dag_runs_in_waves():
    tasks = [T("t1"), T("t2", ["t1"]), T("t3", ["t1"]), T("t4", ["t2", "t3"], kind="synthesis", anchors=["a1"])]
    assert validate_dag(tasks, ANCHORS) == []
    assert [[t.id for t in w] for w in topological_waves(tasks)] == [["t1"], ["t2", "t3"], ["t4"]]
    assert depth(tasks) == 3


def test_rejects_cycles_unknown_references_and_missing_synthesis():
    cyclic = [T("t1", ["t2"]), T("t2", ["t1"]), T("t3", ["t1", "t2"], kind="synthesis")]
    assert "dependency cycle" in validate_dag(cyclic, ANCHORS)
    unknown = [T("t1", ["t9"]), T("t2", ["t1"], kind="synthesis", anchors=["a7"])]
    errors = validate_dag(unknown, ANCHORS)
    assert any("unknown dependency t9" in e for e in errors) and any("unknown anchor a7" in e for e in errors)
    assert any("exactly one synthesis" in e for e in validate_dag([T("t1")], ANCHORS))
    assert validate_dag([], ANCHORS) == ["plan has no tasks"]


def test_rejects_analysis_task_the_synthesis_never_reads():
    tasks = [T("t1"), T("t2"), T("t3", ["t1"], kind="synthesis")]
    assert any("does not depend on ['t2']" in e for e in validate_dag(tasks, ANCHORS))


def test_heuristic_plan_chains_then_clauses_and_owns_every_anchor():
    query = "Estimate the cost of 3 replicas at $40 each, then compare it with a $100 managed tier. Keep it under 60 words."
    anchors = extract_anchors(query)
    tasks = with_synthesis(heuristic_tasks(query, anchors), anchors, query)
    assert validate_dag(tasks, anchors) == []
    synth = tasks[-1]
    assert synth.kind == "synthesis" and set(synth.anchor_ids) == {a.id for a in anchors}
    assert any(t.depends_on for t in tasks if t.kind == "analysis")
    constraint_ids = {a.id for a in anchors if a.kind is AnchorKind.CONSTRAINT}
    assert all(not constraint_ids & set(t.anchor_ids) for t in tasks if t.kind == "analysis")
