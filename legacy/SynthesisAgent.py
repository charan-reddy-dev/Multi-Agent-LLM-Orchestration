"""
Synthesis Agent.
Merges outputs from all sub-agents.
Resolves contradictions flagged by the critique agent.
Produces a final answer with a provenance map linking each sentence to its source agent and chunk.
"""
from __future__ import annotations
import time
import uuid
import logging
import json
from typing import List, Dict, Any
from anthropic import Anthropic

from agents.context_schema import (
    AgentID, AgentMessage, ProvenanceEntry, SharedContext
)
from context_manager.budget_manager import ContextBudgetManager, count_tokens

logger = logging.getLogger(__name__)
client = Anthropic()

SYNTHESIS_SYSTEM = """You are a Synthesis Agent. You receive outputs from multiple agents and must:
1. Merge all relevant content into a coherent final answer
2. Resolve any contradictions flagged by the Critique Agent (do NOT surface them to the user)
3. Produce a provenance map linking each output sentence to its source

Output ONLY valid JSON:
{
  "final_answer": "Complete, coherent answer that resolves all contradictions",
  "provenance": [
    {
      "sentence": "First sentence of the answer",
      "source_agent": "retrieval",
      "source_chunk": "c001",
      "claim_ids": ["claim_id_1"]
    },
    {
      "sentence": "Second sentence",
      "source_agent": "decomposition",
      "source_chunk": null,
      "claim_ids": []
    }
  ],
  "contradictions_resolved": [
    {
      "contradiction": "Agent A said X, Agent B said Y",
      "resolution": "Used X because of evidence Z",
      "resolution_method": "evidence_weight|recency|confidence_score"
    }
  ],
  "synthesis_confidence": 0.88
}
"""


def run(ctx: SharedContext, budget_mgr: ContextBudgetManager) -> AgentMessage:
    budget_mgr.declare_budget(AgentID.SYNTHESIS, 5000)

    # Gather all agent outputs
    decomp_msgs = ctx.get_agent_outputs(AgentID.DECOMPOSITION)
    retrieval_msgs = ctx.get_agent_outputs(AgentID.RETRIEVAL)
    critique_msgs = ctx.get_agent_outputs(AgentID.CRITIQUE)

    # Gather flagged claims
    flagged_claims = [c for c in ctx.claims if c.flagged]
    high_conf_claims = [c for c in ctx.claims if not c.flagged and c.confidence >= 0.7]

    sections = [f"Original Query: {ctx.original_query}\n"]

    if decomp_msgs:
        sections.append(f"Decomposition Output:\n{decomp_msgs[-1].content[:800]}")

    if retrieval_msgs:
        sections.append(f"Retrieval Output:\n{retrieval_msgs[-1].content[:1200]}")

    if critique_msgs:
        sections.append(f"Critique Output:\n{critique_msgs[-1].content[:600]}")

    if flagged_claims:
        sections.append("Flagged Claims to Resolve:\n" + "\n".join(
            f"- [{c.claim_id[:8]}] {c.text}: {c.flag_reason}" for c in flagged_claims[:5]
        ))

    if high_conf_claims:
        sections.append("High-Confidence Claims to Incorporate:\n" + "\n".join(
            f"- [{c.claim_id[:8]}] (conf={c.confidence:.2f}) {c.text}" for c in high_conf_claims[:5]
        ))

    prompt = "\n\n".join(sections)
    prompt += "\n\nSynthesize all of the above into a final, contradiction-free answer with full provenance."

    if not budget_mgr.check_budget(AgentID.SYNTHESIS, prompt):
        logger.warning("[Synthesis] Budget exceeded")
        return AgentMessage(sender=AgentID.SYNTHESIS, content="BUDGET_EXCEEDED",
                            metadata={"error": "budget exceeded"})

    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - t0) * 1000
    raw = response.content[0].text.strip()
    tokens_used = count_tokens(prompt + raw)
    budget_mgr.consume(AgentID.SYNTHESIS, prompt + raw)

    try:
        parsed = json.loads(raw)
        final_answer = parsed.get("final_answer", raw)
        provenance_data = parsed.get("provenance", [])
        contradictions_resolved = parsed.get("contradictions_resolved", [])
        synthesis_confidence = parsed.get("synthesis_confidence", 0.8)
    except json.JSONDecodeError:
        final_answer = raw
        provenance_data = []
        contradictions_resolved = []
        synthesis_confidence = 0.7

    # Build provenance map
    for entry in provenance_data:
        prov = ProvenanceEntry(
            sentence=entry.get("sentence", ""),
            source_agent=AgentID(entry.get("source_agent", "synthesis")),
            source_chunk=entry.get("source_chunk"),
            claim_ids=entry.get("claim_ids", []),
        )
        ctx.provenance_map.append(prov)

    ctx.final_answer = final_answer

    msg = AgentMessage(
        sender=AgentID.SYNTHESIS,
        content=final_answer,
        tokens_used=tokens_used,
        metadata={
            "provenance_entries": len(provenance_data),
            "contradictions_resolved": len(contradictions_resolved),
            "contradiction_details": contradictions_resolved,
            "synthesis_confidence": synthesis_confidence,
            "latency_ms": latency,
        }
    )
    ctx.add_message(msg)
    logger.info(f"[Synthesis] Final answer produced. "
                f"Provenance entries={len(provenance_data)}, "
                f"Contradictions resolved={len(contradictions_resolved)}, "
                f"Confidence={synthesis_confidence}")
    return msg