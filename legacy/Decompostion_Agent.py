"""
Decomposition Agent.
Breaks ambiguous queries into typed sub-tasks with explicit dependency graphs.
Dependent sub-tasks do NOT execute until their dependencies resolve.
"""
from __future__ import annotations
import uuid
import time
import os
import json
import logging
from typing import List, Optional
from anthropic import Anthropic

from agents.context_schema import (
    AgentID, AgentMessage, SharedContext, SubTask, TaskStatus
)
from context_manager.budget_manager import ContextBudgetManager, count_tokens

logger = logging.getLogger(__name__)
client = Anthropic()

SYSTEM_PROMPT = """You are a Decomposition Agent. Your job is to break down a potentially ambiguous query into structured sub-tasks.

Output ONLY valid JSON in this exact format:
{
  "subtasks": [
    {
      "task_id": "t1",
      "task_type": "factual|analytical|comparative|procedural|evaluative",
      "query": "the specific sub-question to answer",
      "dependencies": []
    },
    {
      "task_id": "t2",
      "task_type": "analytical",
      "query": "another sub-question that depends on t1",
      "dependencies": ["t1"]
    }
  ],
  "reasoning": "Why you broke it down this way"
}

Rules:
- Sub-tasks with dependencies MUST NOT run until all dependencies are resolved.
- Each sub-task must have a clear, answerable query.
- Be explicit about dependency ordering.
- Do not output anything outside the JSON block.
"""


def run(ctx: SharedContext, budget_mgr: ContextBudgetManager) -> AgentMessage:
    budget_mgr.declare_budget(AgentID.DECOMPOSITION, 3000)

    prompt = f"Query to decompose:\n{ctx.original_query}"

    if not budget_mgr.check_budget(AgentID.DECOMPOSITION, prompt):
        logger.warning("[Decomposition] Budget exceeded before running")
        return AgentMessage(
            sender=AgentID.DECOMPOSITION,
            content="BUDGET_EXCEEDED",
            metadata={"error": "context budget exceeded"},
        )

    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - t0) * 1000

    raw = response.content[0].text.strip()
    tokens_used = count_tokens(prompt + raw)
    budget_mgr.consume(AgentID.DECOMPOSITION, prompt + raw)

    # Parse JSON
    try:
        parsed = json.loads(raw)
        subtasks_data = parsed.get("subtasks", [])
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        logger.error(f"[Decomposition] Failed to parse JSON: {raw[:200]}")
        subtasks_data = []
        reasoning = raw

    # Build SubTask objects and add to shared context
    subtasks: List[SubTask] = []
    for st in subtasks_data:
        task = SubTask(
            task_id=st.get("task_id", str(uuid.uuid4())),
            task_type=st.get("task_type", "factual"),
            query=st.get("query", ""),
            dependencies=st.get("dependencies", []),
            status=TaskStatus.PENDING,
            assigned_agent=None,
        )
        subtasks.append(task)
        ctx.subtasks.append(task)

    msg = AgentMessage(
        sender=AgentID.DECOMPOSITION,
        content=reasoning,
        tokens_used=tokens_used,
        metadata={
            "subtask_count": len(subtasks),
            "subtasks": [{"id": t.task_id, "type": t.task_type, "deps": t.dependencies} for t in subtasks],
            "latency_ms": latency,
        }
    )
    ctx.add_message(msg)
    logger.info(f"[Decomposition] Produced {len(subtasks)} subtasks. Reasoning: {reasoning[:100]}")
    return msg