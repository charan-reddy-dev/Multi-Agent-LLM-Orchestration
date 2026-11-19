"""PEVR engine: Planner -> Executor (DAG waves, async) -> Verifier -> Refiner loop."""

from __future__ import annotations

import asyncio
import time

from src.agents.executor import Attempt, Executor
from src.agents.planner import Planner
from src.agents.refiner import Refiner
from src.agents.verifier import Verifier
from src.dag import topological_waves
from src.providers.router import LLMRouter
from src.schemas import NodeOutput, NodeRun, Plan, Report, Task


class PEVREngine:
    def __init__(self, router: LLMRouter, *, max_retries: int = 2, max_concurrency: int = 4):
        self.router = router
        self.max_retries = max_retries
        self.planner = Planner(router)
        self.executor = Executor(router)
        self.verifier = Verifier()
        self.refiner = Refiner(router)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, query: str) -> Report:
        start = time.perf_counter()
        calls_before = len(self.router.calls)
        plan = await self.planner.plan(query)
        nodes: dict[str, NodeRun] = {}
        outputs: dict[str, NodeOutput] = {}

        for wave in topological_waves(list(plan.tasks)):
            # Every node in a wave depends only on earlier waves, so the wave runs concurrently.
            runs = await asyncio.gather(*(self._run_node(task, plan, outputs) for task in wave))
            for run in runs:
                nodes[run.task_id] = run
                if run.output is not None:
                    outputs[run.task_id] = run.output

        final = nodes[plan.synthesis.id]
        return Report(
            query=query,
            plan=plan,
            nodes=nodes,
            final_text=final.output.content if final.output else "",
            verdict=final.final_verdict,
            latency_ms=(time.perf_counter() - start) * 1000,
            llm_calls=len(self.router.calls) - calls_before,
        )

    async def _run_node(self, task: Task, plan: Plan, outputs: dict[str, NodeOutput]) -> NodeRun:
        upstream = {dep: outputs[dep] for dep in task.depends_on if dep in outputs}
        run = NodeRun(task_id=task.id)
        async with self._semaphore:
            start = time.perf_counter()
            attempt: Attempt = await self.executor.execute(task, plan, upstream)
            self._record(run, attempt)
            verdict = self.verifier.verify(task, plan, attempt.output, attempt.schema_error)
            run.verdicts.append(verdict)

            # Closed loop: repair only when the Verifier flags something, at most max_retries times.
            while not verdict.passed and run.refinements < self.max_retries:
                run.refinements += 1
                attempt = await self.refiner.refine(
                    task, plan, upstream, run.raw, run.output, verdict, attempt=run.refinements
                )
                if attempt.output is not None or run.output is None:
                    self._record(run, attempt)
                verdict = self.verifier.verify(task, plan, attempt.output, attempt.schema_error)
                run.verdicts.append(verdict)
                if attempt.output is None and run.output is not None:
                    # Keep the last schema-valid draft rather than replacing it with an unparseable reply.
                    run.verdicts[-1] = self.verifier.verify(task, plan, run.output)
                    verdict = run.verdicts[-1]
            run.latency_ms = (time.perf_counter() - start) * 1000
        return run

    @staticmethod
    def _record(run: NodeRun, attempt: Attempt) -> None:
        run.raw = attempt.raw
        run.output = attempt.output
        run.providers.append(attempt.provider)
