import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.anchors import extract_anchors
from src.engine import PEVREngine
from src.providers.mock import FaultProfile, MockProvider
from src.providers.router import LLMRouter

ROOT = Path(__file__).resolve().parents[1]
QUERY = ("Our checkout service serves 12,000 RPS on PostgreSQL with p99 latency of 840 ms. "
         "Identify the likely bottleneck, then propose a Redis caching plan based on that. "
         "Keep the answer under 120 words and do not mention Memcached.")


def engine(faults=None, max_retries=2):
    return PEVREngine(LLMRouter([MockProvider(faults, seed=7)]), max_retries=max_retries)


async def test_clean_run_passes_first_time_with_no_refinement():
    report = await engine().run(QUERY)
    assert report.plan.planner_source == "llm"
    assert len(report.plan.tasks) >= 3  # two analysis steps + synthesis
    assert report.verdict.passed and report.verdict.score == 1.0
    assert all(n.refinements == 0 for n in report.nodes.values())
    for anchor in ("12,000 RPS", "840 ms", "PostgreSQL", "Redis"):
        assert anchor in report.final_text
    assert "Memcached" not in report.final_text
    assert report.llm_calls == 1 + len(report.plan.tasks)


async def test_anchors_are_immutable_across_the_run():
    report = await engine(FaultProfile(alter_number=1.0, fix=1.0)).run(QUERY)
    # Anchors come from rules, not from the (faulty) model, and the frozen dataclasses reject mutation.
    assert report.plan.anchors == extract_anchors(QUERY)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.plan.anchors[0].text = "changed"


async def test_faults_are_detected_and_repaired():
    report = await engine(FaultProfile(drop_anchor=1.0, alter_number=1.0, fix=1.0)).run(QUERY)
    for node in report.nodes.values():
        if report.plan.task(node.task_id).anchor_ids:
            assert not node.first_verdict.passed, f"{node.task_id}: injected fault went undetected"
            assert node.final_verdict.passed and node.refinements >= 1


async def test_refinement_stops_at_max_retries_when_repairs_fail():
    report = await engine(FaultProfile(drop_anchor=1.0, fix=0.0), max_retries=2).run(QUERY)
    for node in report.nodes.values():
        if not node.first_verdict.passed:
            assert node.refinements == 2 and len(node.verdicts) == 3
            assert not node.final_verdict.passed


async def test_invalid_llm_plan_falls_back_to_valid_heuristic_dag():
    report = await engine(FaultProfile(invalid_plan=1.0)).run(QUERY)
    assert report.plan.planner_source == "fallback"
    assert report.verdict.passed


async def test_schema_breaks_never_replace_a_valid_draft():
    report = await engine(FaultProfile(drop_anchor=1.0, schema_break=0.0, fix=0.0), max_retries=1).run(QUERY)
    assert all(n.output is not None for n in report.nodes.values())


def test_cli_runs_offline(tmp_path):
    out = subprocess.run([sys.executable, "-m", "src.cli", QUERY, "--json"], cwd=ROOT, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["verdict"]["passed"] is True


def test_eval_smoke(tmp_path):
    cases = json.loads((ROOT / "benchmarks" / "cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    small = tmp_path / "cases.json"
    small.write_text(json.dumps(cases[:2]), encoding="utf-8")
    result_file = tmp_path / "results.json"
    out = subprocess.run([sys.executable, "benchmarks/eval_pevr.py", "--cases", str(small), "--output", str(result_file)],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert set(result["runs"]) == {"none", "benchmark"}
    clean = result["runs"]["none"]
    assert clean["cases"] == 2 and clean["final_report"]["verified_pass_rate_final"] == 1.0
