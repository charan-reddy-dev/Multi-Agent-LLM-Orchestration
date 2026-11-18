"""
Context Window Budget Manager.
Tracks token consumption per agent per turn.
Agents must declare max budget before execution.
Overflow is caught and logged as a policy violation — never silently truncated.
"""
from __future__ import annotations
import logging
import tiktoken
from typing import Dict, Optional
from agents.context_schema import AgentID, SharedContext

logger = logging.getLogger(__name__)

# Default budgets per agent (in tokens)
DEFAULT_BUDGETS: Dict[AgentID, int] = {
    AgentID.ORCHESTRATOR:  4096,
    AgentID.DECOMPOSITION: 3000,
    AgentID.RETRIEVAL:     6000,
    AgentID.CRITIQUE:      4000,
    AgentID.SYNTHESIS:     5000,
    AgentID.COMPRESSION:   2000,
    AgentID.META:          3000,
}

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


class ContextBudgetManager:
    """
    Central token budget tracker. Every agent calls check_budget() before
    adding context. Violations are logged and surfaced — never silently dropped.
    """

    def __init__(self, ctx: SharedContext):
        self.ctx = ctx
        # Initialize budgets from defaults if not already set
        for agent, budget in DEFAULT_BUDGETS.items():
            if agent not in ctx.token_budget:
                ctx.token_budget[agent] = budget
            if agent not in ctx.token_used:
                ctx.token_used[agent] = 0

    def declare_budget(self, agent: AgentID, max_tokens: int):
        """Agent declares its context budget before execution."""
        self.ctx.token_budget[agent] = max_tokens
        logger.info(f"[BudgetManager] {agent.value} declared budget: {max_tokens} tokens")

    def remaining(self, agent: AgentID) -> int:
        budget = self.ctx.token_budget.get(agent, DEFAULT_BUDGETS.get(agent, 2048))
        used = self.ctx.token_used.get(agent, 0)
        return max(0, budget - used)

    def check_budget(self, agent: AgentID, text: str) -> bool:
        """
        Returns True if adding `text` is within budget.
        Returns False and logs a policy violation if it would overflow.
        """
        tokens = count_tokens(text)
        rem = self.remaining(agent)
        if tokens > rem:
            detail = (
                f"Agent {agent.value} tried to add {tokens} tokens "
                f"but only {rem} remain (budget={self.ctx.token_budget.get(agent)})."
            )
            self.ctx.log_policy_violation(agent, detail)
            logger.warning(f"[BudgetManager] POLICY VIOLATION — {detail}")
            return False
        return True

    def consume(self, agent: AgentID, text: str) -> int:
        """
        Consume tokens for `text`. Raises if over budget (hard enforcement).
        Returns tokens consumed.
        """
        tokens = count_tokens(text)
        if not self.check_budget(agent, text):
            raise OverflowError(
                f"Agent {agent.value} exceeded context budget. "
                f"Trigger compression before proceeding."
            )
        self.ctx.token_used[agent] = self.ctx.token_used.get(agent, 0) + tokens
        return tokens

    def needs_compression(self, agent: AgentID, threshold: float = 0.85) -> bool:
        """Returns True when agent has used >= threshold of its budget."""
        budget = self.ctx.token_budget.get(agent, DEFAULT_BUDGETS.get(agent, 2048))
        used = self.ctx.token_used.get(agent, 0)
        return (used / budget) >= threshold if budget > 0 else False

    def usage_report(self) -> Dict[str, Dict]:
        report = {}
        for agent in AgentID:
            budget = self.ctx.token_budget.get(agent, DEFAULT_BUDGETS.get(agent, 0))
            used = self.ctx.token_used.get(agent, 0)
            report[agent.value] = {
                "budget": budget,
                "used": used,
                "remaining": max(0, budget - used),
                "pct_used": round(used / budget * 100, 1) if budget > 0 else 0,
            }
        return report