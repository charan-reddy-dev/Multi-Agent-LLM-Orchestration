"""
Master Orchestrator Agent.
Dynamically decides at runtime which sub-agents to invoke, in what order,
and with what context window budget.
Does NOT follow a hardcoded chain.
All routing decisions are logged with justification.
All inter-agent communication goes through SharedContext — agents never call each other directly.
"""
from __future__ import annotations
import time
import uuid
import logging
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
from anthropic import Anthropic

from agents.context_schema import (
    AgentID, SharedContext, TaskStatus, ToolCall
)
from context_manager.budget_manager import ContextBudgetManager, count_tokens
from tools.tools import dispatch_tool, ToolResult

import agents.decomposition_agent as decomp
import agents.retrieval_agent as retrieval
import agents.critique_agent as critique
import agents.synthesis_agent as synthesis
import agents.compression_agent as compression

logger = logging.getLogger(__name__)
client = Anthropic()

ROUTER_SYSTEM = """You are the Master Orchestrator of a multi-agent LLM pipeline.
Given a query and current pipeline state, decide which agent to invoke next.

Agents available:
- decomposition: Break ambiguous queries into typed sub-tasks with dependency graphs
- retrieval: Multi-hop RAG across knowledge chunks. Use when factual info is needed.
- critique: Review and score per-claim confidence. Use after retrieval or synthesis.
- synthesis: Merge all outputs into a final answer. Use as the last step.
- compression: Compress context to free budget. Use when token budgets are nearly full.
- DONE: Pipeline is complete.

Output ONLY valid JSON:
{
  "next_agent": "decomposition|retrieval|critique|synthesis|compression|DONE",
  "justification": "Why you chose this agent next",
  "context_budget_allocated": 3000,
  "skip_if_empty": false
}
"""


def _route(ctx: SharedContext, budget_mgr: ContextBudgetManager) -> Dict[str, Any]:
    """Ask the LLM to decide the next agent dynamically."""
    state_summary = {
        "query": ctx.original_query[:200],
        "subtasks_total": len(ctx.subtasks),
        "subtasks_done": sum(1 for t in ctx.subtasks if t.status == TaskStatus.DONE),
        "agents_run": list({m.sender.value for m in ctx.messages}),
        "flagged_claims": sum(1 for c in ctx.claims if c.flagged),
        "has_final_answer": ctx.final_answer is not None,
        "budget_report": budget_mgr.usage_report(),
    }
    prompt = f"Current pipeline state:\n{json.dumps(state_summary, indent=2)}\n\nWhat should run next?"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=ROUTER_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback routing if JSON parse fails
        if not ctx.subtasks:
            decision = {"next_agent": "decomposition", "justification": "No subtasks yet", "context_budget_allocated": 3000}
        elif not ctx.get_agent_outputs(AgentID.RETRIEVAL):
            decision = {"next_agent": "retrieval", "justification": "No retrieval done yet", "context_budget_allocated": 6000}
        elif not ctx.get_agent_outputs(AgentID.CRITIQUE):
            decision = {"next_agent": "critique", "justification": "Need critique pass", "context_budget_allocated": 4000}
        elif not ctx.final_answer:
            decision = {"next_agent": "synthesis", "justification": "Need synthesis", "context_budget_allocated": 5000}
        else:
            decision = {"next_agent": "DONE", "justification": "Pipeline complete", "context_budget_allocated": 0}

    ctx.log_routing(
        AgentID.ORCHESTRATOR,
        AgentID(decision.get("next_agent", "DONE")) if decision.get("next_agent") != "DONE" else AgentID.ORCHESTRATOR,
        decision.get("justification", ""),
        budget_mgr.remaining(AgentID.ORCHESTRATOR),
    )
    return decision


def _handle_tool_with_retry(tool_name: str, max_retries: int = 2, **kwargs) -> ToolResult:
    """
    Call a tool with up to max_retries on failure.
    Each retry is logged separately with modified input.
    """
    parent_call_id = None
    for attempt in range(max_retries + 1):
        result = dispatch_tool(tool_name, **kwargs)
        call_id = str(uuid.uuid4())

        # Log the tool call
        tool_call = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            input=kwargs,
            output=result.data,
            latency_ms=result.latency_ms,
            accepted=result.success,
            retry_of=parent_call_id,
            error=result.error_msg,
        )

        if result.success:
            logger.info(f"[Orchestrator] Tool {tool_name} succeeded on attempt {attempt+1}")
            return result

        # Handle each failure mode differently
        if result.error_type == "timeout":
            logger.warning(f"[Orchestrator] Tool {tool_name} timed out (attempt {attempt+1}). Retrying with shorter query...")
            # Modify input for retry: truncate query
            if "query" in kwargs:
                kwargs["query"] = kwargs["query"][:50] + "..."
        elif result.error_type == "empty":
            logger.warning(f"[Orchestrator] Tool {tool_name} returned empty (attempt {attempt+1}). Broadening query...")
            if "query" in kwargs and attempt == 0:
                kwargs["query"] = kwargs["query"].split()[0]  # use first word only
        elif result.error_type == "malformed":
            logger.error(f"[Orchestrator] Tool {tool_name} malformed input. Skipping retries.")
            return result  # don't retry malformed input

        parent_call_id = call_id

    return result


async def run_streaming(ctx: SharedContext) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main pipeline entry point. Streams events as agents execute.
    Yields dicts that are serialized to SSE in the API layer.
    """
    budget_mgr = ContextBudgetManager(ctx)
    max_steps = 8  # safety ceiling to prevent infinite loops

    yield {"event": "pipeline_start", "job_id": ctx.job_id, "query": ctx.original_query}

    for step in range(max_steps):
        # Check if any agent needs compression
        for agent in AgentID:
            if agent != AgentID.COMPRESSION and budget_mgr.needs_compression(agent, threshold=0.85):
                yield {"event": "compression_triggered", "agent": agent.value,
                       "budget_report": budget_mgr.usage_report()}
                compression.run(ctx, budget_mgr, target_agent=agent)

        # Dynamic routing decision
        decision = _route(ctx, budget_mgr)
        next_agent_name = decision.get("next_agent", "DONE")
        justification = decision.get("justification", "")

        yield {
            "event": "routing_decision",
            "step": step,
            "next_agent": next_agent_name,
            "justification": justification,
            "budget_report": budget_mgr.usage_report(),
        }

        if next_agent_name == "DONE":
            break

        # Execute the chosen agent
        yield {"event": "agent_start", "agent": next_agent_name}

        try:
            if next_agent_name == "decomposition":
                msg = decomp.run(ctx, budget_mgr)
            elif next_agent_name == "retrieval":
                # Retrieval runs for each pending subtask
                pending = [t for t in ctx.subtasks
                           if t.status == TaskStatus.PENDING
                           and all(
                               any(d.task_id == dep and d.status == TaskStatus.DONE
                                   for d in ctx.subtasks)
                               for dep in t.dependencies
                           )]
                if pending:
                    for subtask in pending:
                        subtask.status = TaskStatus.RUNNING
                        msg = retrieval.run(ctx, budget_mgr, subtask=subtask)
                        subtask.status = TaskStatus.DONE
                        subtask.result = msg.content
                        subtask.resolved_at = time.time()
                        yield {"event": "subtask_done", "task_id": subtask.task_id}
                else:
                    msg = retrieval.run(ctx, budget_mgr)
            elif next_agent_name == "critique":
                msg = critique.run(ctx, budget_mgr)
            elif next_agent_name == "synthesis":
                msg = synthesis.run(ctx, budget_mgr)
            elif next_agent_name == "compression":
                msg = compression.run(ctx, budget_mgr)
            else:
                logger.warning(f"[Orchestrator] Unknown agent: {next_agent_name}")
                break

            yield {
                "event": "agent_done",
                "agent": next_agent_name,
                "tokens_used": msg.tokens_used,
                "content_preview": msg.content[:200] if msg.content else "",
                "budget_report": budget_mgr.usage_report(),
            }

        except OverflowError as e:
            yield {"event": "budget_overflow", "agent": next_agent_name, "error": str(e)}
            logger.error(f"[Orchestrator] Budget overflow in {next_agent_name}: {e}")
            # Trigger compression and continue
            compression.run(ctx, budget_mgr)
            continue

        except Exception as e:
            yield {"event": "agent_error", "agent": next_agent_name, "error": str(e)}
            logger.exception(f"[Orchestrator] Error in {next_agent_name}: {e}")
            break

        # Stop once we have a final answer
        if ctx.final_answer:
            break

    ctx.completed_at = time.time()
    yield {
        "event": "pipeline_done",
        "job_id": ctx.job_id,
        "final_answer": ctx.final_answer or "Pipeline completed without final answer.",
        "provenance_map": [
            {"sentence": p.sentence, "source_agent": p.source_agent.value, "source_chunk": p.source_chunk}
            for p in ctx.provenance_map
        ],
        "policy_violations": ctx.policy_violations,
        "routing_log": ctx.routing_log,
        "budget_report": budget_mgr.usage_report(),
        "total_claims": len(ctx.claims),
        "flagged_claims": sum(1 for c in ctx.claims if c.flagged),
    }


def run_sync(query: str) -> SharedContext:
    """Synchronous wrapper for non-streaming use (e.g. eval harness)."""
    ctx = SharedContext(original_query=query)

    async def _collect():
        async for event in run_streaming(ctx):
            logger.debug(f"[Pipeline Event] {event.get('event')}")

    asyncio.run(_collect())
    return ctx