import pytest

from src.agents.verifier import Verifier
from src.anchors import extract_anchors
from src.schemas import NodeOutput, Plan, Task, ViolationCode

QUERY = "GM cut latency 36% from 2.5 s to 1.6 s with FAISS. Answer in under 30 words and do not mention Pinecone."


@pytest.fixture
def plan():
    anchors = extract_anchors(QUERY)
    synth = Task(id="t1", instruction="answer", anchor_ids=tuple(a.id for a in anchors), kind="synthesis")
    return Plan(query=QUERY, anchors=anchors, tasks=(synth,))


def verify(plan, content):
    return Verifier().verify(plan.synthesis, plan, NodeOutput(title="Result", content=content))


def codes(verdict):
    return [v.code for v in verdict.violations]


def test_perfect_output_scores_one(plan):
    verdict = verify(plan, "GM used FAISS to cut latency 36% from 2.5 s to 1.6 s.")
    assert verdict.passed and verdict.score == 1.0
    assert verdict.anchors_present == verdict.anchors_required == 5
    assert verdict.constraints_checked == verdict.constraints_passed == 2


def test_missing_and_altered_numbers_are_flagged(plan):
    verdict = verify(plan, "GM used FAISS to cut latency 38% from 2.5 s.")
    assert codes(verdict).count(ViolationCode.MISSING_ANCHOR) == 2  # 36% and 1.6 s
    assert ViolationCode.UNSUPPORTED_NUMBER in codes(verdict)  # 38 is invented
    # 0.6 * 3/5 + 0.2 + 0.2 - 0.1
    assert verdict.score == pytest.approx(0.66)


def test_constraints_are_checked_on_synthesis(plan):
    long_text = "GM used FAISS and Pinecone to cut latency 36% from 2.5 s to 1.6 s " + "really " * 30
    verdict = verify(plan, long_text)
    assert codes(verdict).count(ViolationCode.CONSTRAINT) == 2
    assert verdict.score == pytest.approx(0.8)


def test_unparseable_output_scores_zero_and_misses_every_anchor(plan):
    verdict = Verifier().verify(plan.synthesis, plan, None, "invalid JSON")
    assert verdict.score == 0.0 and not verdict.schema_ok
    assert codes(verdict).count(ViolationCode.MISSING_ANCHOR) == verdict.anchors_required


def test_constraints_are_not_checked_on_analysis_tasks(plan):
    analysis = Task(id="a", instruction="x", anchor_ids=plan.synthesis.anchor_ids)
    verdict = Verifier().verify(analysis, plan, NodeOutput(title="t", content="GM FAISS 36% 2.5 s 1.6 s Pinecone"))
    assert verdict.passed and verdict.constraints_checked == 0
