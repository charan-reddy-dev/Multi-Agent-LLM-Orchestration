"""PEVR evaluation benchmark.

Runs every case in benchmarks/cases.json through the full Planner-Executor-Verifier-Refiner pipeline and
records anchor extraction recall, DAG validity, anchor retention, schema compliance, constraint
satisfaction and fidelity (first draft vs. after refinement), retry convergence and latency.

    python benchmarks/eval_pevr.py                                    # offline mock, both fault profiles
    python benchmarks/eval_pevr.py --provider groq,gemini --output benchmarks/results-live.json

With the mock provider, the "none" profile is a sanity baseline and "benchmark" injects the fault
rates documented in src/providers/mock.py. Those results measure how well the Verifier detects and the
Refiner repairs faults; they say nothing about the quality of a real model. Use --provider for that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anchors import normalize  # noqa: E402
from src.dag import depth, validate_dag  # noqa: E402
from src.engine import PEVREngine  # noqa: E402
from src.providers.mock import FaultProfile, MockProvider  # noqa: E402
from src.providers.router import LLMRouter, router_from_env  # noqa: E402
from src.schemas import Report  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))]


def ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def anchor_found(expected: str, extracted: list[str]) -> bool:
    e = normalize(expected)
    return any(e == normalize(x) or e in normalize(x) or normalize(x) in e for x in extracted)


async def run_profile(cases: list[dict], router_factory, max_retries: int) -> dict:
    per_case = []
    reports: list[tuple[dict, Report]] = []
    detection = {"nodes_with_injected_fault": 0, "detected": 0, "flagged_without_injected_fault": 0}
    injected_by_type: dict[str, int] = {}
    for case in cases:
        router = router_factory()
        engine = PEVREngine(router, max_retries=max_retries)
        start = time.perf_counter()
        report = await engine.run(case["query"])
        elapsed = (time.perf_counter() - start) * 1000
        reports.append((case, report))

        # Ground truth from the mock (absent for live providers): which first drafts really had a fault.
        injected = [x for p in router.providers for x in getattr(p, "injected", [])]
        for _, _, _, fault in injected:
            injected_by_type[fault] = injected_by_type.get(fault, 0) + 1
        faulty = {tid for role, tid, attempt, _ in injected if role == "executor" and attempt == 0}
        flagged_ids = {tid for tid, n in report.nodes.items() if not n.first_verdict.passed}
        detection["nodes_with_injected_fault"] += len(faulty)
        detection["detected"] += len(faulty & flagged_ids)
        detection["flagged_without_injected_fault"] += len(flagged_ids - faulty)
        synth = report.nodes[report.plan.synthesis.id]
        extracted = [a.text for a in report.plan.anchors]
        per_case.append({
            "id": case["id"],
            "category": case["category"],
            "tasks": len(report.plan.tasks),
            "planner_source": report.plan.planner_source,
            "anchors_extracted": len(extracted),
            "expected_anchors_found": sum(anchor_found(e, extracted) for e in case["expected_anchors"]),
            "expected_anchors": len(case["expected_anchors"]),
            "unexpected_anchors": [x for x in extracted if not any(anchor_found(e, [x]) for e in case["expected_anchors"])],
            "first_draft_score": synth.first_verdict.score,
            "final_score": synth.final_verdict.score,
            "final_passed": synth.final_verdict.passed,
            "refinements": sum(n.refinements for n in report.nodes.values()),
            "llm_calls": report.llm_calls,
            "latency_ms": round(elapsed, 3),
            "final_violations": [v.as_dict() for v in synth.final_verdict.violations],
        })

    nodes = [n for _, r in reports for n in r.nodes.values()]
    synths = [r.nodes[r.plan.synthesis.id] for _, r in reports]
    flagged = [n for n in nodes if not n.first_verdict.passed]
    converged = [n for n in flagged if n.final_verdict.passed]
    multi = [(c, r) for c, r in reports if c["expect_dependency"]]
    latencies = [c["latency_ms"] for c in per_case]

    def retention(verdicts):
        return ratio(sum(v.anchors_present for v in verdicts), sum(v.anchors_required for v in verdicts))

    def constraints(verdicts):
        return ratio(sum(v.constraints_passed for v in verdicts), sum(v.constraints_checked for v in verdicts))

    return {
        "cases": len(cases),
        "anchor_extraction": {
            "expected_anchor_recall": ratio(sum(c["expected_anchors_found"] for c in per_case),
                                            sum(c["expected_anchors"] for c in per_case)),
            "anchors_extracted_total": sum(c["anchors_extracted"] for c in per_case),
            "expected_anchors_total": sum(c["expected_anchors"] for c in per_case),
            "unexpected_anchors_total": sum(len(c["unexpected_anchors"]) for c in per_case),
            "note": "expected anchors are hand-labelled by the extractor's author; not an independent test set",
        },
        "planning": {
            "valid_dag_rate": ratio(sum(not validate_dag(list(r.plan.tasks), r.plan.anchors) for _, r in reports), len(reports)),
            "planner_fallback_rate": ratio(sum(r.plan.planner_source == "fallback" for _, r in reports), len(reports)),
            "mean_tasks": round(statistics.fmean(len(r.plan.tasks) for _, r in reports), 3),
            "mean_depth": round(statistics.fmean(depth(list(r.plan.tasks)) for _, r in reports), 3),
            "multi_step_cases_with_dependent_subtasks": ratio(
                sum(any(t.kind == "analysis" and t.depends_on for t in r.plan.tasks) for _, r in multi), len(multi)),
        },
        "final_report": {
            "anchor_retention_first_draft": retention([n.first_verdict for n in synths]),
            "anchor_retention_final": retention([n.final_verdict for n in synths]),
            "constraint_satisfaction_first_draft": constraints([n.first_verdict for n in synths]),
            "constraint_satisfaction_final": constraints([n.final_verdict for n in synths]),
            "mean_fidelity_first_draft": round(statistics.fmean(n.first_verdict.score for n in synths), 4),
            "mean_fidelity_final": round(statistics.fmean(n.final_verdict.score for n in synths), 4),
            "verified_pass_rate_first_draft": ratio(sum(n.first_verdict.passed for n in synths), len(synths)),
            "verified_pass_rate_final": ratio(sum(n.final_verdict.passed for n in synths), len(synths)),
        },
        "all_nodes": {
            "nodes": len(nodes),
            "anchor_retention_first_draft": retention([n.first_verdict for n in nodes]),
            "anchor_retention_final": retention([n.final_verdict for n in nodes]),
            "schema_compliance_first_draft": ratio(sum(n.first_verdict.schema_ok for n in nodes), len(nodes)),
            "schema_compliance_final": ratio(sum(n.final_verdict.schema_ok for n in nodes), len(nodes)),
        },
        "refinement": {
            "max_retries": max_retries,
            "nodes_flagged_on_first_draft": len(flagged),
            "nodes_converged": len(converged),
            "retry_convergence_rate": ratio(len(converged), len(flagged)),
            "mean_rounds_to_converge": round(statistics.fmean(n.refinements for n in converged), 3) if converged else None,
            "refinement_rounds_total": sum(n.refinements for n in nodes),
            "violations_first_draft_by_code": _count_codes(n.first_verdict for n in nodes),
            "violations_final_by_code": _count_codes(n.final_verdict for n in nodes),
        },
        "fault_detection": {
            "faults_injected_by_type": dict(sorted(injected_by_type.items())),
            **detection,
            "detection_rate": ratio(detection["detected"], detection["nodes_with_injected_fault"]),
            "note": "first-draft executor faults injected by the mock vs. nodes the Verifier flagged; "
                    "empty for live providers, which have no ground truth",
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "max": round(max(latencies), 3),
        },
        "llm_calls_per_case_mean": round(statistics.fmean(c["llm_calls"] for c in per_case), 3),
        "per_case": per_case,
    }


def _count_codes(verdicts) -> dict:
    counts: dict[str, int] = {}
    for v in verdicts:
        for violation in v.violations:
            counts[violation.code.value] = counts.get(violation.code.value, 0) + 1
    return dict(sorted(counts.items()))


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default=str(ROOT / "benchmarks" / "cases.json"))
    p.add_argument("--provider", default="mock", help="mock, or a comma-separated live provider order")
    p.add_argument("--profiles", default="none,benchmark", help="mock fault profiles to run (ignored for live)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default=str(ROOT / "benchmarks" / "results.json"))
    args = p.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    runs = {}
    if args.provider == "mock":
        for profile_name in [x.strip() for x in args.profiles.split(",") if x.strip()]:
            profile = FaultProfile.benchmark() if profile_name == "benchmark" else FaultProfile.none()
            print(f"profile {profile_name}: {len(cases)} cases ...", flush=True)
            result = asyncio.run(run_profile(cases, lambda: LLMRouter([MockProvider(profile, seed=args.seed)]), args.max_retries))
            result["provider"] = "mock"
            result["fault_profile"] = {"name": profile_name, **asdict(profile)}
            runs[profile_name] = result
    else:
        env = {**os.environ, "PEVR_PROVIDERS": args.provider}
        print(f"live providers {args.provider}: {len(cases)} cases ...", flush=True)
        result = asyncio.run(run_profile(cases, lambda: router_from_env(env), args.max_retries))
        result["provider"] = router_from_env(env).names
        runs["live"] = result

    output = {
        "benchmark": "pevr-eval",
        "version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": "python benchmarks/eval_pevr.py " + " ".join(sys.argv[1:]),
        "environment": {"python": sys.version.split()[0], "os": platform.platform(), "cpu": platform.processor()},
        "cases_file": os.path.relpath(args.cases, ROOT).replace("\\", "/"),
        "seed": args.seed,
        "runs": runs,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for name, r in runs.items():
        fr, rf = r["final_report"], r["refinement"]
        print(f"[{name}] anchor recall (extraction) {r['anchor_extraction']['expected_anchor_recall']}  "
              f"retention first/final {fr['anchor_retention_first_draft']}/{fr['anchor_retention_final']}  "
              f"pass first/final {fr['verified_pass_rate_first_draft']}/{fr['verified_pass_rate_final']}  "
              f"convergence {rf['retry_convergence_rate']} ({rf['nodes_converged']}/{rf['nodes_flagged_on_first_draft']})  "
              f"detection {r['fault_detection']['detection_rate']}  "
              f"latency mean {r['latency_ms']['mean']} ms")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
