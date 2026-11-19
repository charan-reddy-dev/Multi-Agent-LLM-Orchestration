"""
Critique Agent.
Reviews output of every other agent.
Assigns structured confidence score PER CLAIM (not output as a whole).
Flags specific spans of text it disagrees with.
"""
from __future__ import annotations
import time
import uuid
import logging
import json
from typing import List, Dict, Any
from anthropic import Anthropic

from agents.context_schema import (
    AgentID, AgentMessage, Claim, SharedContext
)
from context_manager.budget_manager import ContextBudgetManager, count_tokens

logger = logging.getLogger(__name__)
client = Anthropic()

CRITIQUE_SYSTEM = """You are a Critique Agent. You review outputs from other agents with rigorous skepticism.
You do NOT reject an output as a whole. You identify specific claims and score each one.

For each claim or sentence you review, assign:
- confidence_score: 0.0 (completely wrong) to 1.0 (definitely correct)
- flagged: true if you disagree with this claim
- flag_reason: specific reason why you disagree (quote the exact span)

Output ONLY valid JSON:
{
  "reviewed_agent": "agent_name",
  "claim_reviews": [
    {
      "claim_id": "existing_id_or_new",
      "text_span": "exact quoted text from the output",
      "confidence_score": 0.85,
      "flagged": false,
      "flag_reason": null
    },
    {
      "claim_id": "new_or_existing",
      "text_span": "another claim from output",
      "confidence_score": 0.3,
      "flagged": true,
      "flag_reason": "This contradicts X because Y. The specific span 'Z' is unsupported."
    }
  ],
  "overall_quality": 0.75,
  "critique_summary": "Overall assessment with specific issues noted"
}
"""


def run(ctx: SharedContext, budget_mgr: ContextBudgetManager,
        target_agent: AgentID = None) -> AgentMessage:

    budget_mgr.declare_budget(AgentID.CRITIQUE, 4000)

    # Gather all agent outputs to review (or specific agent if targeted)
    agents_to_review = [target_agent] if target_agent else [
        AgentID.DECOMPOSITION, AgentID.RETRIEVAL, AgentID.SYNTHESIS
    ]

    all_critique_results = []

    for agent in agents_to_review:
        agent_msgs = ctx.get_agent_outputs(agent)
        if not agent_msgs:
            continue

        # Review the latest output from this agent
        latest = agent_msgs[-1]
        content_to_review = latest.content
        existing_claims = latest.claims

        # Format existing claims for context
        claims_context = ""
        if existing_claims:
            claims_context = "\n\nExisting claims in this output:\n" + "\n".join(
                f"- [{c.claim_id[:8]}] {c.text}" for c in existing_claims
            )

        prompt = f"""Review this output from the {agent.value} agent:

---
{content_to_review[:2000]}
---
{claims_context}

Provide per-claim critique with confidence scores. Flag specific spans you disagree with."""

        if not budget_mgr.check_budget(AgentID.CRITIQUE, prompt):
            logger.warning(f"[Critique] Budget exceeded reviewing {agent.value}")
            break

        t0 = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=CRITIQUE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - t0) * 1000
        raw = response.content[0].text.strip()
        budget_mgr.consume(AgentID.CRITIQUE, prompt + raw)

        try:
            parsed = json.loads(raw)
            claim_reviews = parsed.get("claim_reviews", [])
            overall_quality = parsed.get("overall_quality", 0.7)
            summary = parsed.get("critique_summary", "")
        except json.JSONDecodeError:
            claim_reviews = []
            overall_quality = 0.7
            summary = raw

        # Update existing claims with critique scores; create new ones if needed
        for review in claim_reviews:
            claim_id = review.get("claim_id", str(uuid.uuid4()))
            # Find existing claim or create new
            existing = next((c for c in ctx.claims if c.claim_id.startswith(claim_id[:8])), None)
            if existing:
                existing.confidence = review.get("confidence_score", existing.confidence)
                existing.flagged = review.get("flagged", False)
                existing.flag_reason = review.get("flag_reason")
            else:
                new_claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    text=review.get("text_span", ""),
                    source_agent=agent,
                    confidence=review.get("confidence_score", 0.7),
                    flagged=review.get("flagged", False),
                    flag_reason=review.get("flag_reason"),
                )
                ctx.claims.append(new_claim)

        all_critique_results.append({
            "agent": agent.value,
            "overall_quality": overall_quality,
            "flagged_count": sum(1 for r in claim_reviews if r.get("flagged")),
            "claim_count": len(claim_reviews),
            "summary": summary,
            "latency_ms": latency,
        })

    combined_summary = "\n\n".join(
        f"[{r['agent']}] Quality={r['overall_quality']:.2f}, "
        f"Flagged={r['flagged_count']}/{r['claim_count']} claims\n{r['summary']}"
        for r in all_critique_results
    )

    tokens_used = count_tokens(combined_summary)
    msg = AgentMessage(
        sender=AgentID.CRITIQUE,
        content=combined_summary,
        tokens_used=tokens_used,
        metadata={
            "reviews": all_critique_results,
            "total_flagged": sum(r["flagged_count"] for r in all_critique_results),
        }
    )
    ctx.add_message(msg)
    logger.info(f"[Critique] Reviewed {len(all_critique_results)} agents, "
                f"flagged {msg.metadata['total_flagged']} claims")
    return msg