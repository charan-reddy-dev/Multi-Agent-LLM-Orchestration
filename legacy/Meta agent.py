"""
Meta Agent — Self-Improving Prompt Loop.
After each eval run, reads failure cases, identifies worst-performing prompt,
and proposes a rewritten version with structured diff and justification.
Proposed rewrites are stored but NOT automatically applied.
A human must approve via the /approve-rewrite endpoint.
"""
from __future__ import annotations
import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from db.database import get_conn

logger = logging.getLogger(__name__)
client = Anthropic()

# Current active prompts for each agent (modifiable by approved rewrites)
ACTIVE_PROMPTS: Dict[str, str] = {
    "decomposition": """You are a Decomposition Agent. Your job is to break down a potentially ambiguous query into structured sub-tasks.

Output ONLY valid JSON in this exact format:
{
  "subtasks": [...],
  "reasoning": "Why you broke it down this way"
}""",
    "retrieval": """You are a Retrieval-Augmented Agent. You receive a query and multiple retrieved text chunks.
You MUST perform multi-hop reasoning — use at least 2 chunks to build your answer.
Cite each chunk explicitly in your answer using [chunk_id].""",
    "critique": """You are a Critique Agent. You review outputs from other agents with rigorous skepticism.
You do NOT reject an output as a whole. You identify specific claims and score each one.""",
    "synthesis": """You are a Synthesis Agent. You receive outputs from multiple agents and must:
1. Merge all relevant content into a coherent final answer
2. Resolve any contradictions flagged by the Critique Agent
3. Produce a provenance map linking each output sentence to its source""",
    "orchestrator": """You are the Master Orchestrator of a multi-agent LLM pipeline.
Given a query and current pipeline state, decide which agent to invoke next.""",
}

META_SYSTEM = """You are a Meta-Agent responsible for improving the pipeline's prompts.
You receive a list of failed test cases and their scores, and the current prompt for each agent.

Your job:
1. Identify the single worst-performing prompt (by which dimension had the lowest average score)
2. Propose a rewritten version of that prompt
3. Provide a structured diff explaining exactly what changed and why

Output ONLY valid JSON:
{
  "worst_agent": "agent_id",
  "worst_dimension": "dimension_name",
  "avg_score_on_dimension": 0.35,
  "analysis": "Why this prompt is underperforming",
  "original_prompt": "The current prompt text",
  "proposed_prompt": "The rewritten prompt text",
  "diff_summary": [
    {"change_type": "added|removed|modified", "original": "old text", "proposed": "new text", "reason": "why"}
  ],
  "justification": "Overall reasoning for the rewrite",
  "expected_improvement": "What specifically should improve and why"
}
"""


def propose_rewrite(eval_run_id: str) -> Optional[Dict[str, Any]]:
    """
    Reads failure cases from the eval run, proposes a prompt rewrite.
    Stores the proposal in DB. Does NOT apply it automatically.
    Returns the rewrite proposal dict.
    """
    conn = get_conn()

    # Get failed cases
    failed_rows = conn.execute("""
        SELECT * FROM eval_results
        WHERE run_id = ? AND total_score < 0.6
        ORDER BY total_score ASC
        LIMIT 10
    """, (eval_run_id,)).fetchall()

    if not failed_rows:
        logger.info(f"[Meta] No failed cases in run {eval_run_id}. No rewrite needed.")
        conn.close()
        return None

    failed_cases = [dict(r) for r in failed_rows]

    # Compute per-dimension averages from failed cases
    dim_names = ["answer_correctness", "citation_accuracy", "contradiction_resolution",
                 "tool_efficiency", "budget_compliance", "critique_agreement"]
    dim_avgs = {}
    for dim in dim_names:
        scores = [r[dim] for r in failed_cases if r.get(dim) is not None]
        dim_avgs[dim] = sum(scores) / len(scores) if scores else 1.0

    worst_dim = min(dim_avgs, key=dim_avgs.get)

    # Map dimension to responsible agent
    DIM_TO_AGENT = {
        "answer_correctness": "synthesis",
        "citation_accuracy": "retrieval",
        "contradiction_resolution": "critique",
        "tool_efficiency": "orchestrator",
        "budget_compliance": "orchestrator",
        "critique_agreement": "critique",
    }
    worst_agent = DIM_TO_AGENT.get(worst_dim, "synthesis")
    current_prompt = ACTIVE_PROMPTS.get(worst_agent, "")

    # Format failed cases for meta agent
    cases_summary = "\n".join(
        f"- [{r['case_id']}] ({r['case_category']}) score={r['total_score']:.2f} | "
        f"query='{r['query'][:80]}' | {worst_dim}={r.get(worst_dim, 'N/A')}"
        for r in failed_cases[:5]
    )

    prompt = f"""Failed test cases from eval run {eval_run_id}:

{cases_summary}

Dimension averages (failed cases only):
{json.dumps(dim_avgs, indent=2)}

Worst dimension: {worst_dim} (avg={dim_avgs[worst_dim]:.3f})
Responsible agent: {worst_agent}

Current prompt for {worst_agent}:
---
{current_prompt}
---

Propose a rewrite that will improve {worst_dim} scores."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=META_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()

    try:
        proposal = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"[Meta] Failed to parse proposal JSON: {raw[:200]}")
        conn.close()
        return None

    rewrite_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO prompt_rewrites
        (rewrite_id, eval_run_id, agent_id, dimension, original_prompt,
         proposed_prompt, diff_summary, justification, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
    """, (
        rewrite_id,
        eval_run_id,
        proposal.get("worst_agent", worst_agent),
        proposal.get("worst_dimension", worst_dim),
        proposal.get("original_prompt", current_prompt),
        proposal.get("proposed_prompt", ""),
        json.dumps(proposal.get("diff_summary", [])),
        proposal.get("justification", ""),
        time.time(),
    ))
    conn.commit()
    conn.close()

    logger.info(f"[Meta] Proposed rewrite {rewrite_id} for agent={worst_agent}, dim={worst_dim}")
    return {"rewrite_id": rewrite_id, **proposal}


def approve_rewrite(rewrite_id: str, approved: bool, approved_by: str = "human") -> Dict[str, Any]:
    """
    Human approves or rejects a prompt rewrite.
    If approved, updates ACTIVE_PROMPTS but does NOT trigger re-eval (separate endpoint).
    """
    conn = get_conn()
    status = "approved" if approved else "rejected"

    row = conn.execute("SELECT * FROM prompt_rewrites WHERE rewrite_id = ?",
                       (rewrite_id,)).fetchone()
    if not row:
        conn.close()
        return {"error": "rewrite_id not found"}

    rewrite = dict(row)
    conn.execute("""
        UPDATE prompt_rewrites
        SET status = ?, approved_by = ?, approved_at = ?
        WHERE rewrite_id = ?
    """, (status, approved_by, time.time(), rewrite_id))
    conn.commit()
    conn.close()

    if approved:
        agent_id = rewrite["agent_id"]
        new_prompt = rewrite["proposed_prompt"]
        ACTIVE_PROMPTS[agent_id] = new_prompt
        logger.info(f"[Meta] Rewrite {rewrite_id} APPROVED. Updated prompt for {agent_id}.")

    return {"rewrite_id": rewrite_id, "status": status, "agent_id": rewrite["agent_id"]}


def rerun_failed_cases(eval_run_id: str, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Re-runs only the previously failed cases using latest approved prompts.
    Returns delta scores.
    """
    from eval.eval_harness import TEST_CASES, run_eval, get_failed_cases

    failed = get_failed_cases(eval_run_id, threshold=threshold)
    if not failed:
        return {"message": "No failed cases to rerun", "delta": {}}

    # Find matching test cases
    failed_ids = {r["case_id"] for r in failed}
    cases_to_rerun = [c for c in TEST_CASES if c["case_id"] in failed_ids]

    new_run_id = str(uuid.uuid4())
    new_summary = run_eval(cases=cases_to_rerun, run_id=new_run_id)

    # Compute delta
    old_scores = {r["case_id"]: r["total_score"] for r in failed}
    conn = get_conn()
    new_rows = conn.execute(
        "SELECT case_id, total_score FROM eval_results WHERE run_id = ?", (new_run_id,)
    ).fetchall()
    conn.close()
    new_scores = {r["case_id"]: r["total_score"] for r in new_rows}

    delta = {
        cid: round(new_scores.get(cid, 0) - old_scores.get(cid, 0), 3)
        for cid in failed_ids
    }
    avg_delta = round(sum(delta.values()) / len(delta), 3) if delta else 0.0

    logger.info(f"[Meta] Rerun complete. Avg delta: {avg_delta:+.3f}")
    return {
        "new_run_id": new_run_id,
        "cases_retested": len(cases_to_rerun),
        "avg_delta": avg_delta,
        "per_case_delta": delta,
        "new_summary": new_summary,
    }