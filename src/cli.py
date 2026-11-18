"""Command-line entry point.

    python -m src.cli "Compare Redis Streams and Apache Kafka for 10,000 RPS in under 120 words."
    python -m src.cli --provider groq,gemini "..."     # live providers, fallback in that order
    python -m src.cli --json "..."                     # full trace as JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from src.engine import PEVREngine
from src.providers.mock import FaultProfile, MockProvider
from src.providers.router import LLMRouter, router_from_env


def main() -> None:
    p = argparse.ArgumentParser(description="Run the PEVR pipeline on one request.")
    p.add_argument("query")
    p.add_argument("--provider", help="comma-separated provider order (groq, gemini, openai, mock); default from env")
    p.add_argument("--mock-faults", choices=["none", "benchmark"], default="none")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = p.parse_args()

    if args.provider == "mock" or (not args.provider and router_from_env().names[0].startswith("mock")):
        faults = FaultProfile.benchmark() if args.mock_faults == "benchmark" else FaultProfile.none()
        router = LLMRouter([MockProvider(faults)])
    else:
        env = dict(os.environ)
        if args.provider:
            env["PEVR_PROVIDERS"] = args.provider
        router = router_from_env(env)

    report = asyncio.run(PEVREngine(router, max_retries=args.max_retries).run(args.query))
    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2, ensure_ascii=False)
        print()
        return
    print(report.final_text)
    v = report.verdict
    print(f"\n[fidelity {v.score:.2f} | anchors {v.anchors_present}/{v.anchors_required} | "
          f"constraints {v.constraints_passed}/{v.constraints_checked} | {'PASS' if v.passed else 'FAIL'} | "
          f"{report.llm_calls} LLM calls via {', '.join(router.names)}]")
    for violation in v.violations:
        print(f"  - {violation.code.value}: {violation.detail}")


if __name__ == "__main__":
    main()
